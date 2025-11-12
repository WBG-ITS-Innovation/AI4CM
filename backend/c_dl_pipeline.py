# -*- coding: utf-8 -*-
"""
C · Deep Learning (Sequential) — Daily + Weekly + Monthly
--------------------------------------------------------
Models: LSTM, GRU, Dilated CNN (causal), Transformer encoder, MLP (flattened).
Direct forecasting per horizon; simple conformal 90% intervals from a calibration slice.
Outputs: predictions_long.csv, metrics_long.csv, leaderboard.csv, plots/*, artifacts/*.

Author: Georgia project (DL family)
"""

from __future__ import annotations
import os, json, math, warnings, time, sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(suppress=True)

# =============================================================================
# Config
# =============================================================================

@dataclass
class ConfigDL:
    # I/O
    data_path: str
    date_col: str = "date"
    holidays_csv: Optional[str] = None
    out_root_uni: str = "./outputs/dl_univariate"
    out_root_multi: str = "./outputs/dl_multivariate"

    # Targets (UI usually passes one; CLI can pass more)
    targets: Optional[List[str]] = None

    # Cadences + horizons
    cadences: Optional[List[str]] = None                  # ["daily"|"weekly"|"monthly"]
    horizons_daily: Optional[List[int]] = None            # [1,5,20]
    horizons_weekly: Optional[List[int]] = None           # [1,4,12]
    horizons_monthly: Optional[List[int]] = None          # [1,3,6]
    min_train_years: int = 4

    # Sequence lengths (history window)
    seq_len_daily: int = 64
    seq_len_weekly: int = 52
    seq_len_monthly: int = 36

    # Feature sets for multivariate
    mv_feature_cols: Optional[List[str]] = None   # whitelist; None => auto-select by corr
    mv_max_auto_features: int = 20                # hard cap

    # Models to try
    models_univariate: Optional[List[str]] = None  # ["lstm","gru","dcnn","transformer","mlp"]
    models_multivariate: Optional[List[str]] = None

    # Training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    valid_frac: float = 0.1             # from fit-slice (not the calibration tail)
    conformal_calib_frac: float = 0.2   # last part of train used for calibration
    nominal_pi: float = 0.90
    sample_weight_scheme: str = "none"  # "none" | "month_end_boost"
    eom_boost_weight: float = 3.0
    target_transform: str = "none"      # "none" | "log1p" (flows only)

    # Runtime
    random_seed: int = 42
    device: str = "auto"                # "auto" | "cpu" | "cuda"
    quick_mode: bool = False            # if True: small epochs for smoke runs

    def __post_init__(self):
        if self.targets is None:
            self.targets = ["State budget balance"]
        if self.cadences is None:
            self.cadences = ["daily"]
        if self.horizons_daily is None:
            self.horizons_daily = [1,5,20]
        if self.horizons_weekly is None:
            self.horizons_weekly = [1,4,12]
        if self.horizons_monthly is None:
            self.horizons_monthly = [1,3,6]
        if self.models_univariate is None:
            self.models_univariate = ["lstm","gru","dcnn","transformer","mlp"]
        if self.models_multivariate is None:
            self.models_multivariate = ["lstm","gru","dcnn","transformer","mlp"]
        if self.quick_mode:
            self.epochs = min(self.epochs, 5)
            self.batch_size = max(64, self.batch_size)

# =============================================================================
# Shared utilities (calendar, baseline, metrics, plotting)
# =============================================================================

def ensure_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    for sub in ["plots","artifacts"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)

def is_stock(name: str) -> bool:
    return name.strip().lower() in {"state budget balance","balance","t0"}

def load_holidays(holidays_csv: Optional[str], idx: pd.DatetimeIndex) -> pd.Series:
    if not holidays_csv or not os.path.exists(holidays_csv):
        return pd.Series(0, index=idx, dtype=int)
    h = pd.read_csv(holidays_csv)
    dcols = [c for c in h.columns if c.lower()=="date"]
    if not dcols:
        return pd.Series(0, index=idx, dtype=int)
    d = pd.to_datetime(h[dcols[0]], errors="coerce").dt.normalize()
    d = d.dropna().drop_duplicates()
    s = pd.Series(1, index=pd.DatetimeIndex(d))
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(idx, fill_value=0).astype(int)

def calendar_exog(idx: pd.DatetimeIndex, cadence: str, holidays: pd.Series) -> pd.DataFrame:
    """
    Calendar features aligned to idx; robust for daily/weekly/monthly.
    """
    idx = pd.DatetimeIndex(idx)
    cad = cadence.lower()

    if cad == "daily":
        df = pd.DataFrame(index=idx)
        df["dow"] = idx.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["month"] = idx.month
        df["doy"] = idx.dayofyear
        # holidays aligned on same index
        h = pd.Series(0, index=idx)
        if isinstance(holidays, pd.Series):
            try: h = holidays.reindex(idx).fillna(0)
            except Exception: pass
        df["is_holiday"] = h.astype(int).values
        # eom/eoq on business-day calendar
        eom = (idx + pd.offsets.BDay(1)).to_period("M") != idx.to_period("M")
        eoq = (idx + pd.offsets.BDay(1)).to_period("Q") != idx.to_period("Q")
        df["is_eom"] = eom.astype(int); df["is_eoq"] = eoq.astype(int)
        dow_d = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
        return pd.concat([df.drop(columns=["dow"]), dow_d], axis=1)

    # weekly / monthly
    df = pd.DataFrame(index=idx)
    if cad == "weekly":
        df["week_of_year"] = idx.isocalendar().week.to_numpy(dtype="int64")
    month_codes = pd.Categorical(idx.to_period("M").astype(str)).codes
    quarter_codes = pd.Categorical(idx.to_period("Q").astype(str)).codes
    df["month_code"] = month_codes; df["quarter_code"] = quarter_codes
    return df

def resample_cadence(df: pd.DataFrame, target: str, cadence: str) -> pd.Series:
    y = df[target].astype(float)
    if cadence=="daily":
        bidx = pd.date_range(y.index.min().normalize(), y.index.max().normalize(), freq="B")
        yb = y.reindex(bidx)
        yb = yb.ffill() if is_stock(target) else yb.fillna(0.0)
        yb.index.freq = "B"
        return yb
    elif cadence=="weekly":
        s = y.resample("W-FRI").last() if is_stock(target) else y.resample("W-FRI").sum()
        s.index.freq = s.index.freq or pd.infer_freq(s.index)
        return s
    elif cadence=="monthly":
        s = y.resample("ME").last() if is_stock(target) else y.resample("ME").sum()
        s.index.freq = s.index.freq or pd.infer_freq(s.index)
        return s
    else:
        raise ValueError("cadence must be {'daily','weekly','monthly'}")

def horizons_for(cadence: str, cfg: ConfigDL) -> List[int]:
    return {"daily": cfg.horizons_daily, "weekly": cfg.horizons_weekly, "monthly": cfg.horizons_monthly}[cadence]

def seq_len_for(cadence: str, cfg: ConfigDL) -> int:
    return {"daily": cfg.seq_len_daily, "weekly": cfg.seq_len_weekly, "monthly": cfg.seq_len_monthly}[cadence]

# Treasury baseline (flows)
def ops_monthly_baseline_treasury(series: pd.Series, years_window: int = 3) -> pd.Series:
    m = series.resample("ME").sum().astype(float)
    if m.empty: return m
    df = m.to_frame("val"); df["year"] = df.index.year; df["month"] = df.index.month
    counts = df.groupby("year")["val"].size()
    full_years = counts.index[counts.eq(12)].tolist()
    if len(full_years) < years_window:
        return df.groupby("month")["val"].transform(lambda s: s.shift(12).rolling(36, min_periods=1).mean()).reindex(m.index)
    out = []
    for ts, mo, Y in zip(df.index, df["month"], df["year"]):
        prev = [Y - k for k in range(1, years_window + 1)]
        if not all(py in full_years for py in prev):
            out.append(np.nan); continue
        annual_prev = (df[df["year"].isin(prev)].groupby("year")["val"].sum()).reindex(prev)
        ann_mean = annual_prev.mean()
        shares = []
        for py in prev:
            mon_val = df.loc[(df["year"] == py) & (df["month"] == mo), "val"].iloc[0]
            shares.append(mon_val / annual_prev.loc[py] if annual_prev.loc[py] > 0 else np.nan)
        share_mean = np.nanmean(shares)
        out.append(ann_mean * share_mean if np.isfinite(share_mean) else np.nan)
    return pd.Series(out, index=m.index).ffill()

def ops_daily_from_monthly(series: pd.Series, monthly_baseline: pd.Series, method: str = "profile") -> pd.Series:
    daily = series.copy()
    mb = monthly_baseline.dropna()
    if mb.empty: return pd.Series(index=daily.index, dtype=float)
    result = pd.Series(index=pd.date_range(min(daily.index.min(), mb.index.min().replace(day=1)),
                                           max(daily.index.max(), mb.index.max()), freq="B"), dtype=float)
    for m, mval in mb.items():
        days = pd.date_range(m.replace(day=1), m, freq="B")
        if method=="profile":
            profiles = []
            for k in (1,2,3):
                my = m - pd.DateOffset(years=k)
                hist_days = pd.date_range(my.replace(day=1), my, freq="B")
                s = daily[(daily.index>=hist_days.min()) & (daily.index<=hist_days.max())].reindex(hist_days, fill_value=0.0)
                if s.sum()>0:
                    p = (s/s.sum()).reindex(days, fill_value=0.0).values
                    profiles.append(p)
            if profiles:
                p = np.mean(np.vstack(profiles), axis=0)
                p = p / (p.sum() if p.sum()>0 else 1.0)
            else:
                p = np.ones(len(days))/len(days)
            result.loc[days] = float(mval)*p
        else:
            result.loc[days] = float(mval)/len(days)
    return result.reindex(daily.index)

# Metrics
def safe_mape(y_true, y_pred):
    eps=1e-9; t = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(np.abs(t-p)/np.maximum(np.abs(t), eps)))

def smape(y_true, y_pred):
    eps=1e-9; t = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(2*np.abs(p-t)/(np.abs(t)+np.abs(p)+eps)))

def coverage_width(y_true, lo, hi):
    if lo is None or hi is None: return (np.nan, np.nan)
    m = np.isfinite(lo) & np.isfinite(hi)
    if not m.any(): return (np.nan, np.nan)
    cover = float(np.mean((y_true[m]>=lo[m]) & (y_true[m]<=hi[m])))
    width = float(np.mean(hi[m]-lo[m]))
    return cover, width

def evaluate_block(df_pred: pd.DataFrame, target: str, h: int, cadence: str,
                   ops_pred: Optional[pd.Series]) -> pd.DataFrame:
    rows = []
    for model, g in df_pred.groupby("model", sort=False):
        g = g.sort_values("date")
        yt = g["y_true"].to_numpy(); yp = g["y_pred"].to_numpy()
        mae  = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp)**2)))
        r2   = np.nan
        ss_tot = np.sum((yt - np.mean(yt))**2)
        if ss_tot>0: r2 = 1 - np.sum((yt - yp)**2)/ss_tot

        if is_stock(target):
            mape = np.nan; smp = np.nan
            m_true = pd.Series(yt, index=pd.to_datetime(g["date"])).resample("ME").last()
            m_pred = pd.Series(yp, index=pd.to_datetime(g["date"])).resample("ME").last()
        else:
            mape = safe_mape(yt, yp); smp = smape(yt, yp)
            m_true = pd.Series(yt, index=pd.to_datetime(g["date"])).resample("ME").sum()
            m_pred = pd.Series(yp, index=pd.to_datetime(g["date"])).resample("ME").sum()

        tol10 = float(np.mean((np.abs(m_true - m_pred)/np.maximum(np.abs(m_true),1e-9)) <= 0.10)) if len(m_true) else np.nan
        cover, width = coverage_width(yt, g["y_lo"].to_numpy(), g["y_hi"].to_numpy())

        mae_skill = np.nan
        if (ops_pred is not None) and (not is_stock(target)):
            aligned = g.set_index("date").join(ops_pred.rename("ops_pred"), how="left")
            if aligned["ops_pred"].notna().any():
                mae_ops = float(np.mean(np.abs(aligned["y_true"] - aligned["ops_pred"])))
                mae_skill = (1.0 - mae/mae_ops) if mae_ops>0 else np.nan

        rows.append({"target":target,"horizon":h,"cadence":cadence,"model":model,
                     "MAE":mae,"RMSE":rmse,"sMAPE":smp,"MAPE":mape,"R2":r2,
                     "PI_coverage@90":cover,"PI_width@90":width,
                     "Monthly_TOL10_Accuracy":tol10,"MAE_skill_vs_Ops":mae_skill})
    return pd.DataFrame(rows)

# Plots
def plot_overlay_all(df_slice, target, h, cadence, ops_series, out_png):
    fig, ax = plt.subplots(figsize=(12,4))
    for model, g in df_slice.groupby("model"):
        ax.plot(g["date"], g["y_pred"], label=model, alpha=0.8)
    ax.plot(df_slice["date"], df_slice["y_true"], label="Actual", linewidth=2, color="black")
    if ops_series is not None:
        ops_aligned = ops_series.reindex(pd.to_datetime(df_slice["date"]))
        ax.plot(df_slice["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | {cadence} | h=+{h} | ALL vs Ops")
    ax.legend(loc="best"); ax.grid(True); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_overlay_top(df_slice, target, h, cadence, best_model, ops_series, out_png):
    fig, ax = plt.subplots(figsize=(12,4))
    g = df_slice[df_slice["model"]==best_model].sort_values("date")
    ax.plot(g["date"], g["y_pred"], label=best_model, linewidth=2)
    if g["y_lo"].notna().any() and g["y_hi"].notna().any():
        ax.fill_between(g["date"], g["y_lo"], g["y_hi"], alpha=0.2, label="PI 90%")
    ax.plot(g["date"], g["y_true"], label="Actual", linewidth=2)
    if ops_series is not None:
        ops_aligned = ops_series.reindex(pd.to_datetime(g["date"]))
        ax.plot(g["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | {cadence} | h=+{h} | TOP vs Ops")
    ax.legend(loc="best"); ax.grid(True); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_leaderboard_bar(mdf, target, h, cadence, out_png):
    g = mdf.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(g["model"], g["MAE"]); ax.invert_yaxis()
    ax.set_title(f"{target} | {cadence} | h=+{h} | Leaderboard by MAE"); ax.set_xlabel("MAE (lower is better)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_monthly_bars(df_slice, target, h, cadence, ops_series, out_png):
    idx = pd.to_datetime(df_slice["date"])
    s_true = pd.Series(df_slice["y_true"].to_numpy(), index=idx)
    s_pred = pd.Series(df_slice["y_pred"].to_numpy(), index=idx)
    if is_stock(target):
        m_true = s_true.resample("ME").last(); m_pred = s_pred.resample("ME").last()
    else:
        m_true = s_true.resample("ME").sum();  m_pred = s_pred.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12,4))
    x = m_true.index
    ax.bar(x - pd.Timedelta(days=3), m_true.values, width=6, label="Actual")
    ax.bar(x, m_pred.values, width=6, label="TOP model", alpha=0.7)
    if ops_series is not None and (not is_stock(target)):
        ops_m = ops_series.reindex(idx).resample("ME").sum()
        ax.bar(x + pd.Timedelta(days=3), ops_m.values, width=6, label="Ops baseline", alpha=0.7)
    ax.set_title(f"{target} | {cadence} | h=+{h} | Monthly aggregates"); ax.legend(loc="best"); ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def build_yearly_folds(idx: pd.DatetimeIndex, min_train_years: int):
    years = sorted(set(idx.year))
    if not years: return []
    first_year, last_year = min(years), max(years)
    folds = []
    for Y in range(first_year+min_train_years, last_year+1):
        train_end = pd.Timestamp(f"{Y-1}-12-31")
        if not (idx<=train_end).any(): continue
        train_end = idx[idx<=train_end][-1]
        test_start = pd.Timestamp(f"{Y}-01-01")
        if not (idx>=test_start).any(): continue
        Y_end = pd.Timestamp(f"{Y}-12-31")
        c = idx[(idx>=test_start)&(idx<=Y_end)]
        if c.empty:
            c = idx[idx>=test_start]
            if c.empty: continue
        test_end = c[-1]
        folds.append((train_end, test_start, test_end))
    return folds

def _last_window_fallback_masks(label_idx: np.ndarray, horizon: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """If no yearly folds are available, build a single last-window fold on label times."""
    t = pd.DatetimeIndex(label_idx)
    n = t.size
    if n == 0: return []
    tl = min(max(1, horizon), n)
    test_mask = np.zeros(n, dtype=bool); test_mask[-tl:] = True
    train_mask = np.zeros(n, dtype=bool); train_mask[:max(0, n - tl)] = True
    if not train_mask.any(): return []
    return [(train_mask, test_mask)]

# =============================================================================
# Feature assembly & sequences
# =============================================================================

def pick_exogenous_columns(df_num: pd.DataFrame, target: str, reserved: List[str], max_cols: int) -> List[str]:
    num = df_num.select_dtypes(include=[np.number]).copy()
    for col in reserved:
        if col in num.columns:
            num.drop(columns=[col], inplace=True)
    stds = num.std(axis=0, ddof=0)
    num = num.loc[:, stds > 0]
    if target not in df_num.columns:
        return []
    common = num.join(df_num[[target]], how="inner")
    corrs = common.corr(method="spearman")[target].drop(labels=[target], errors="ignore").abs().sort_values(ascending=False)
    return list(corrs.index[:max_cols])

def make_feature_frame(df_all: pd.DataFrame, target: str, cadence: str,
                       multivariate: bool, mv_cols: Optional[List[str]], mv_max: int,
                       holidays_master: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    y = resample_cadence(df_all, target, cadence)
    idx = y.index
    holidays = holidays_master.reindex(idx).fillna(0).astype(int)
    cal = calendar_exog(idx, cadence, holidays)
    F = pd.DataFrame(index=idx)
    F[target] = y.values
    F = F.join(cal, how="left").astype(float)

    if not multivariate:
        return F.fillna(0.0), y

    pool = {}
    for col in df_all.columns:
        s = df_all[col].astype(float)
        if cadence == "daily":
            bidx = pd.date_range(idx.min(), idx.max(), freq="B")
            s = s.reindex(bidx)
            s = s.ffill() if is_stock(col) else s.fillna(0.0)
        elif cadence == "weekly":
            s = s.resample("W-FRI").last() if is_stock(col) else s.resample("W-FRI").sum()
        else:
            s = s.resample("ME").last() if is_stock(col) else s.resample("ME").sum()
        pool[col] = s
    pool_df = pd.DataFrame(pool).reindex(idx).ffill().fillna(0.0)

    reserved = list(F.columns)
    candidates = pool_df.columns.tolist() if mv_cols is None else [c for c in mv_cols if c in pool_df.columns]
    chosen = [c for c in candidates if c not in reserved and c != target]
    if mv_max is not None and mv_max > 0:
        chosen = chosen[:mv_max]
    if chosen:
        F = F.join(pool_df[chosen], how="left")

    return F.fillna(0.0), y

def build_sequences(F: pd.DataFrame, y: pd.Series, seq_len: int, horizon: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    idx = F.index
    Xs, ys, label_dates = [], [], []
    n = len(idx)
    for end_i in range(seq_len-1, n - horizon):
        sl = F.iloc[end_i-seq_len+1 : end_i+1].values  # (L,C)
        y_target = float(y.iloc[end_i + horizon])
        Xs.append(sl.astype(np.float32))
        ys.append(np.float32(y_target))
        label_dates.append(pd.Timestamp(idx[end_i + horizon]))
    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, seq_len, F.shape[1]), dtype=np.float32)
    yv = np.array(ys, dtype=np.float32)
    ld = np.array(label_dates, dtype="datetime64[ns]")
    return X, yv, ld

def fit_feature_scaler(X: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0); sd = flat.std(axis=0)
    sd[sd==0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_feature_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)

def transform_target(y: np.ndarray, mode: str):
    if mode=="log1p":
        fwd = lambda a: np.log1p(np.maximum(a, 0.0))
        inv = lambda a: np.expm1(a)
        return fwd(y), fwd, inv
    return y.copy(), (lambda a: a), (lambda a: a)

# =============================================================================
# Models
# =============================================================================

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hid=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
    def forward(self, x):  # x: (B,L,C)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(-1)

class GRUReg(nn.Module):
    def __init__(self, in_dim, hid=64, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(-1)

class DilatedCNNReg(nn.Module):
    def __init__(self, in_dim, hid=64, layers=4, k=3, dropout=0.1):
        super().__init__()
        chans = [in_dim] + [hid]*layers
        mods = []
        for i in range(layers):
            d = 2**i
            pad = (k-1)*d
            mods += [
                nn.ConstantPad1d((pad,0), 0.0),            # left-causal pad
                nn.Conv1d(chans[i], hid, k, dilation=d),   # (B,C,L)->(B,hid,L)
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*mods)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
    def forward(self, x):
        z = x.transpose(1,2)
        z = self.net(z)            # (B,hid,L)
        h = z[:, :, -1]            # last time step
        return self.head(h).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe)   # (max_len, d_model)
    def forward(self, x):  # x: (L,B,D)
        L = x.size(0)
        return x + self.pe[:L].unsqueeze(1)

class TransformerReg(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, layers=2, dropout=0.1, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):  # x: (B,L,C)
        z = self.proj(x)             # (B,L,D)
        z = z.transpose(0,1)         # (L,B,D)
        z = self.pos(z)
        z = self.encoder(z)          # (L,B,D)
        h = z[-1]                    # (B,D)
        return self.head(h).squeeze(-1)

class MLPReg(nn.Module):
    def __init__(self, in_dim, seq_len, hid=128, layers=2, dropout=0.1):
        super().__init__()
        layers_list = [nn.Linear(in_dim*seq_len, hid), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(layers-1):
            layers_list += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout)]
        layers_list += [nn.Linear(hid, 1)]
        self.net = nn.Sequential(*layers_list)
        self.seq_len = seq_len
        self.in_dim = in_dim
    def forward(self, x):  # x: (B,L,C)
        b, L, C = x.shape
        return self.net(x.reshape(b, L*C)).squeeze(-1)

def make_model(name: str, in_dim: int, seq_len: int, cfg: ConfigDL) -> nn.Module:
    n = name.lower()
    if n=="lstm":        return LSTMReg(in_dim, hid=96, layers=2, dropout=cfg.dropout)
    if n=="gru":         return GRUReg(in_dim,  hid=96, layers=2, dropout=cfg.dropout)
    if n=="dcnn":        return DilatedCNNReg(in_dim, hid=96, layers=4, k=3, dropout=cfg.dropout)
    if n=="transformer": return TransformerReg(in_dim, d_model=96, nhead=4, layers=2, dropout=cfg.dropout, dim_feedforward=192)
    if n=="mlp":         return MLPReg(in_dim, seq_len, hid=192, layers=2, dropout=cfg.dropout)
    raise ValueError(f"Unknown DL model: {name}")

# =============================================================================
# Training / prediction
# =============================================================================

def _device(cfg: ConfigDL) -> str:
    if cfg.device == "cpu": return "cpu"
    if cfg.device == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None):
    err = (pred - target)**2
    if weight is None: return err.mean()
    w = weight.view(-1).to(dtype=pred.dtype, device=pred.device)
    w = w * (w.numel() / (w.sum() + 1e-8))
    return (w*err).mean()

@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for Xb in loader:
        if isinstance(Xb, (tuple, list)):
            Xb = Xb[0]
        Xb = Xb.to(device)
        yhat = model(Xb)
        preds.append(yhat.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)

def train_model(model: nn.Module,
                X_fit: np.ndarray, y_fit: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                sample_weight: Optional[np.ndarray],
                cfg: ConfigDL) -> nn.Module:
    device = _device(cfg)
    model = model.to(device)
    bs = cfg.batch_size
    ne = cfg.epochs

    Xf = torch.from_numpy(X_fit); yf = torch.from_numpy(y_fit)
    Xv = torch.from_numpy(X_val); yv = torch.from_numpy(y_val)
    wf = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2 if cfg.quick_mode else 3)

    best = (1e18, None, 0)  # (val_loss, state_dict, epoch_of_best)
    patience = 3 if cfg.quick_mode else 8
    for epoch in range(ne):
        model.train()
        idx = np.arange(len(X_fit))
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            sel = idx[i:i+bs]
            xb = Xf[sel].to(device)
            yb = yf[sel].to(device)
            wb = None if wf is None else wf[sel].to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = weighted_mse(yhat, yb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            yhat_v = model(Xv.to(device))
            val_loss = ((yhat_v - yv.to(device))**2).mean().item()
            sched.step(val_loss)
        if val_loss < best[0] - 1e-6:
            best = (val_loss, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= patience:
            break

    if best[1] is not None:
        model.load_state_dict(best[1])
    return model

def conformal_band_from_residuals(y_cal_true: np.ndarray, y_cal_pred: np.ndarray, nominal_pi: float):
    resid = np.abs(y_cal_true - y_cal_pred)
    q = np.quantile(resid, nominal_pi)
    return q

# =============================================================================
# Core runner
# =============================================================================

def _run_family(config: ConfigDL, out_root: str, family: str):
    torch.manual_seed(config.random_seed); np.random.seed(config.random_seed)
    device = _device(config)
    print(f"[DL] device: {device}")

    df = pd.read_csv(config.data_path)
    dcol = [c for c in df.columns if c.lower()==config.date_col.lower()]
    assert dcol, f"Date column '{config.date_col}' not found."
    df[dcol[0]] = pd.to_datetime(df[dcol[0]], errors="coerce")
    df = df.dropna(subset=[dcol[0]]).sort_values(dcol[0]).drop_duplicates(subset=[dcol[0]]).set_index(dcol[0])

    holidays_master = load_holidays(config.holidays_csv, idx=df.index)

    for cadence in config.cadences:
        out_root_cad = os.path.join(out_root, cadence.lower())
        ensure_dirs(out_root_cad)
        predictions_all = []
        metrics_all = []

        for target in config.targets:
            if target not in df.columns:
                print(f"[WARN] Missing target '{target}', skipping."); continue

            stock = is_stock(target)
            seq_len = seq_len_for(cadence, config)

            multivariate = (family=="multivariate")
            F, y = make_feature_frame(df, target, cadence.lower(), multivariate,
                                      config.mv_feature_cols, config.mv_max_auto_features,
                                      holidays_master)

            # Treasury baselines (flows only)
            if not stock:
                y_daily = resample_cadence(df, target, "daily")
                m_base  = ops_monthly_baseline_treasury(y_daily)
                d_base  = ops_daily_from_monthly(y_daily, m_base, method="profile")
                if cadence.lower()=="daily":
                    ops_series = d_base
                elif cadence.lower()=="weekly":
                    ops_series = d_base.resample("W-FRI").sum()
                else:
                    ops_series = d_base.resample("ME").sum()
                m_base.rename("forecast").to_csv(os.path.join(out_root_cad, f"{target}_ops_baseline_monthly.csv"))
                pd.DataFrame({"date": d_base.index, "forecast": d_base.values}).to_csv(
                    os.path.join(out_root_cad, f"{target}_ops_baseline_daily.csv"), index=False
                )
            else:
                ops_series = None

            # Sequences are built against label dates; folds should be generated on labels
            horizons = horizons_for(cadence.lower(), config)
            for h in horizons:
                X_all, y_all, ld_all = build_sequences(F, y, seq_len, h)
                if X_all.shape[0] == 0:
                    print(f"[WARN] No sequences for {target} {cadence} h={h}. Need >= lookback({seq_len})+horizon({h}).")
                    continue

                # Primary: yearly folds on label times
                fld = build_yearly_folds(pd.DatetimeIndex(ld_all), config.min_train_years)
                masks: List[Tuple[np.ndarray,np.ndarray]] = []
                for (tr_end, ts_start, ts_end) in fld:
                    ld = pd.to_datetime(ld_all)
                    tr_mask = (ld <= tr_end)
                    te_mask = (ld >= ts_start) & (ld <= ts_end)
                    if tr_mask.any() and te_mask.any():
                        masks.append((tr_mask, te_mask))

                # Fallback: last-window fold if no yearly folds
                if not masks:
                    print(f"[WARN] No yearly folds for {target} at {cadence} (h={h}). Using last-window fallback.")
                    masks = _last_window_fallback_masks(ld_all, h)
                    if not masks:
                        print(f"[WARN] Still no usable window for {target} {cadence} h={h}.")
                        continue

                # Train/predict per fold and model
                model_list = (config.models_univariate if family=="univariate" else config.models_multivariate)
                for tr_mask, te_mask in masks:
                    X_tr_all = X_all[tr_mask]; y_tr_all = y_all[tr_mask]
                    X_te = X_all[te_mask];     y_te = y_all[te_mask]
                    ld_tr = pd.to_datetime(ld_all[tr_mask]); ld_te = pd.to_datetime(ld_all[te_mask])

                    if X_tr_all.shape[0] < 1 or X_te.shape[0] < 1:
                        continue

                    # Split into fit/val; reserve a calibration tail
                    n_tr = len(X_tr_all)
                    split_cal = max(int(n_tr*(1.0-config.conformal_calib_frac)), 1)
                    X_fit_all, y_fit_all = X_tr_all[:split_cal], y_tr_all[:split_cal]
                    X_cal, y_cal_t       = X_tr_all[split_cal:], y_tr_all[split_cal:]

                    n_fit = len(X_fit_all)
                    split_val = max(int(n_fit*(1.0-config.valid_frac)), 1)
                    X_fit, y_fit = X_fit_all[:split_val], y_fit_all[:split_val]
                    X_val, y_val = X_fit_all[split_val:], y_fit_all[split_val:]
                    if len(X_val) == 0:
                        split_val = max(n_fit-1, 1)
                        X_fit, y_fit = X_fit_all[:split_val], y_fit_all[:split_val]
                        X_val, y_val = X_fit_all[split_val:], y_fit_all[split_val:]

                    # Feature scaling on fit slice only
                    mu, sd = fit_feature_scaler(X_fit)
                    X_fit = apply_feature_scaler(X_fit, mu, sd)
                    X_val = apply_feature_scaler(X_val, mu, sd)
                    X_cal = apply_feature_scaler(X_cal, mu, sd)
                    X_te_s = apply_feature_scaler(X_te, mu, sd)

                    # Target transform (flows only if requested)
                    if (config.target_transform == "log1p") and (not stock):
                        y_fit_t = np.log1p(np.maximum(y_fit, 0.0))
                        inv = np.expm1
                        y_cal_raw = y_cal_t.copy()
                        y_te_raw  = y_te.copy()
                        y_cal_t = np.log1p(np.maximum(y_cal_t, 0.0))
                    else:
                        y_fit_t = y_fit.copy()
                        inv = (lambda a: a)
                        y_cal_raw = y_cal_t.copy()
                        y_te_raw  = y_te.copy()

                    for mname in model_list:
                        try:
                            model = make_model(mname, X_fit.shape[-1], seq_len, config)
                        except Exception as ex:
                            print(f"[WARN] Skip model={mname}: {ex}")
                            continue

                        model = train_model(model, X_fit, y_fit_t, X_val, y_val, None, config)

                        # Conformal from calibration tail (if present)
                        if len(X_cal) > 0:
                            cal_loader  = DataLoader(TensorDataset(torch.from_numpy(X_cal)), batch_size=512, shuffle=False)
                            yhat_cal_t  = predict_model(model, cal_loader, device)
                            q_t = conformal_band_from_residuals(y_cal_t, yhat_cal_t, nominal_pi=config.nominal_pi)
                        else:
                            q_t = 0.0

                        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te_s)), batch_size=512, shuffle=False)
                        yhat_t = predict_model(model, test_loader, device)
                        yhat = inv(yhat_t)
                        lo = inv(yhat_t - q_t) if q_t else np.full_like(yhat, np.nan)
                        hi = inv(yhat_t + q_t) if q_t else np.full_like(yhat, np.nan)

                        df_m = pd.DataFrame({
                            "date": ld_te.values,
                            "target": target,
                            "horizon": h,
                            "model": mname.upper(),
                            "y_true": y_te_raw.astype(float),
                            "y_pred": yhat.astype(float),
                            "y_lo":  lo.astype(float),
                            "y_hi":  hi.astype(float),
                            "split_id": f"{ld_tr[-1].date()}→{ld_te[0].date()}..{ld_te[-1].date()}",
                            "cadence": cadence,
                            "defn_variant": f"DL_{family}_{cadence}" + ("_log1p" if ((config.target_transform=='log1p') and (not stock)) else "")
                        })
                        predictions_all.append(df_m)

                # end folds loop

                if predictions_all:
                    df_pred_h = pd.concat([p for p in predictions_all
                                           if (p["target"].iloc[0]==target and p["horizon"].iloc[0]==h and p["cadence"].iloc[0]==cadence)],
                                          ignore_index=True)
                    mdf = evaluate_block(df_pred_h, target, h, cadence, ops_pred=ops_series)
                    metrics_all.append(mdf)
                    lb = mdf.sort_values("MAE").reset_index(drop=True)
                    best_model = lb.iloc[0]["model"] if len(lb) else None
                    # Save leaderboards per target/h
                    lb.assign(target=target, horizon=h, cadence=cadence).to_csv(
                        os.path.join(out_root_cad, f"leaderboard_{target.replace(' ','_')}_h{h}.csv"), index=False
                    )
                    # Plots (optional robustness)
                    try:
                        plot_overlay_all(df_pred_h, target, h, cadence, ops_series,
                                         os.path.join(out_root_cad, f"{target.replace(' ','_')}_h{h}_overlay_all.png"))
                        if best_model:
                            plot_overlay_top(df_pred_h[df_pred_h["model"]==best_model], target, h, cadence, best_model, ops_series,
                                             os.path.join(out_root_cad, f"{target.replace(' ','_')}_h{h}_overlay_top.png"))
                            plot_monthly_bars(df_pred_h[df_pred_h["model"]==best_model], target, h, cadence, ops_series,
                                              os.path.join(out_root_cad, f"{target.replace(' ','_')}_h{h}_monthly_bars_top_vs_ops.png"))
                            plot_leaderboard_bar(mdf, target, h, cadence,
                                                 os.path.join(out_root_cad, f"{target.replace(' ','_')}_h{h}_leaderboard_mae.png"))
                    except Exception as e:
                        print(f"[WARN] Plotting failed for {target} {cadence} h={h}: {e}")

        # end for target

        # Master CSVs for this cadence
        if predictions_all:
            pd.concat(predictions_all, ignore_index=True).sort_values(["target","horizon","date","model"]).to_csv(
                os.path.join(out_root_cad, "predictions_long.csv"), index=False
            )
        else:
            # write empty stub so UI doesn't show "missing" — makes it obvious it's an empty run
            pd.DataFrame(columns=["date","target","horizon","model","y_true","y_pred","y_lo","y_hi","split_id","cadence","defn_variant"])\
              .to_csv(os.path.join(out_root_cad, "predictions_long.csv"), index=False)
        if metrics_all:
            metr = pd.concat(metrics_all, ignore_index=True)
            metr.to_csv(os.path.join(out_root_cad, "metrics_long.csv"), index=False)
            rows = []
            for (t,h), g in metr.groupby(["target","horizon"]):
                g2 = g.groupby("model", as_index=False)["MAE"].mean().sort_values("MAE")
                for rank, row in enumerate(g2.itertuples(index=False), start=1):
                    rows.append({"target":t,"horizon":h,"model":row.model,"MAE":row.MAE,"rank":rank})
            pd.DataFrame(rows).to_csv(os.path.join(out_root_cad, "leaderboard.csv"), index=False)

        os.makedirs(os.path.join(out_root_cad, "artifacts"), exist_ok=True)
        with open(os.path.join(out_root_cad, "artifacts", "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        print(f"[OK] DL {family} -> {cadence} outputs: {out_root_cad}")

# -----------------------------------------------------------------------------

def run_pipeline(config: ConfigDL = None, run_univariate: bool = True, run_multivariate: bool = True):
    assert config is not None, "ConfigDL is required."
    if run_univariate:
        _run_family(config, out_root=config.out_root_uni, family="univariate")
    if run_multivariate:
        _run_family(config, out_root=config.out_root_multi, family="multivariate")
