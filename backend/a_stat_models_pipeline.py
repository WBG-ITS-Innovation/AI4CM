# -*- coding: utf-8 -*-
"""
A · Statistical (baselines & calendar-aware) pipeline
-----------------------------------------------------
(unchanged preamble)

Usage (example):
    from a_stat_models_pipeline import run_pipeline, CONFIG
    CONFIG.data_path = r"...\\data\\processed\\master_daily_raw.csv"
    CONFIG.out_root  = r"...\\outputs\\statistical"
    run_pipeline(CONFIG)
"""

from __future__ import annotations

import os, json, math, warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Optional (available in statsmodels >= 0.13)
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    HAS_THETA = True
except Exception:
    HAS_THETA = False

# Quiet known, harmless messages once we’ve regularized the index
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency")
warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")
warnings.filterwarnings("ignore", message="Optimization failed to converge", module="statsmodels")

# =========================
# Configuration
# =========================

@dataclass
class ConfigAStat:
    # --- INPUTS ---
    data_path: str = "./data/processed/master_daily_raw.csv"
    date_col: str = "date"
    targets: List[str] = None
    holidays_csv: Optional[str] = "./data/holidays_georgia_2015_2025.csv"

    # --- CADENCE (NEW) ---
    cadence: str = "Daily"  # Daily | Weekly | Monthly

    # --- HORIZONS ---
    horizons: List[int] = None
    min_train_years: int = 4

    # --- MODELS ---
    models: List[str] = None
    seasonal_period: int = 5
    sarimax_refit_every: int = 5

    # --- OUTPUTS ---
    out_root: str = str((Path.cwd() / "outputs" / "statistical").resolve())
    nominal_pi: float = 0.90

    # --- Treasury baseline ---
    build_ops_baseline: bool = True
    ops_daily_method: str = "profile"

    # --- NEW knobs from UI ---
    model_params: Dict[str, dict] = field(default_factory=dict)  # per-model overrides
    max_folds: Optional[int] = None
    demo_clip_months: Optional[int] = None

    # --- MISC ---
    random_seed: int = 42

    def __post_init__(self):
        if self.targets is None:
            self.targets = ["Revenues", "Expenditure", "State budget balance"]
        if self.horizons is None:
            self.horizons = [1, 5, 20]
        if self.models is None:
            base = ["naive_last", "weekday_mean", "movavg7", "ets", "sarimax", "stl_arima"]
            if HAS_THETA:
                base.append("theta")
            self.models = base

CONFIG = ConfigAStat()

# =========================
# Helpers
# =========================

def ensure_dirs(root: str):
    Path(root).mkdir(parents=True, exist_ok=True)
    for sub in ["plots", "artifacts"]:
        Path(root, sub).mkdir(parents=True, exist_ok=True)

def is_stock(target: str) -> bool:
    return target.strip().lower() in {"state budget balance", "balance", "t0"}

def _cad_freq(cadence: str) -> str:
    return {"Daily": "B", "Weekly": "W-FRI", "Monthly": "ME"}.get(cadence, "B")

def regularize_series(y: pd.Series, target: str, cadence: str) -> pd.Series:
    """Regularize to cadence with appropriate aggregation."""
    if y.empty:
        return y
    if cadence == "Daily":
        bidx = pd.date_range(y.index.min().normalize(), y.index.max().normalize(), freq="B")
        yb = y.reindex(bidx)
        yb = yb.ffill() if is_stock(target) else yb.fillna(0.0)
        yb.index.freq = "B"
        return yb
    if cadence == "Weekly":
        # anchor on Friday week end
        return (y.resample("W-FRI").last() if is_stock(target) else y.resample("W-FRI").sum()).astype(float)
    if cadence == "Monthly":
        return (y.resample("ME").last() if is_stock(target) else y.resample("ME").sum()).astype(float)
    return y

def load_holidays(holidays_csv: Optional[str], idx: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(idx)
    if not holidays_csv or not Path(holidays_csv).exists():
        return pd.Series(0, index=idx, dtype=int)
    h = pd.read_csv(holidays_csv)
    date_cols = [c for c in h.columns if c.lower() == "date"]
    if not date_cols:
        return pd.Series(0, index=idx, dtype=int)
    d = pd.to_datetime(h[date_cols[0]], errors="coerce").dt.normalize().dropna().drop_duplicates()
    hs = pd.Series(1, index=pd.DatetimeIndex(d))
    hs = hs[~hs.index.duplicated(keep="first")]
    return hs.reindex(idx, fill_value=0).astype(int)

def calendar_exog(idx: pd.DatetimeIndex, holidays: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = idx.month
    df["doy"] = idx.dayofyear
    df["is_holiday"] = holidays.reindex(idx).fillna(0).astype(int).values
    eom = (idx + pd.offsets.BDay(1)).to_period("M") != idx.to_period("M")
    eoq = (idx + pd.offsets.BDay(1)).to_period("Q") != idx.to_period("Q")
    df["is_eom"] = eom.astype(int)
    df["is_eoq"] = eoq.astype(int)
    dow_d = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
    return pd.concat([df.drop(columns=["dow"]), dow_d], axis=1)

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def coverage_width(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    if lo is None or hi is None:
        return np.nan, np.nan
    mask = np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return np.nan, np.nan
    cover = np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask]))
    width = float(np.mean(hi[mask] - lo[mask]))
    return float(cover), width

def ops_monthly_baseline(series: pd.Series, years_window: int = 3) -> pd.Series:
    m = series.resample("ME").sum().astype(float)
    if len(m) < 24:
        return m.groupby(m.index.month).transform(
            lambda x: x.shift(12).rolling(36, min_periods=1).mean()
        )
    y = m.to_period("Y")
    annual = y.groupby(y.index).sum().astype(float)
    annual_full = annual[annual.index.isin(
        m.index.to_period("Y")[m.index.to_period("Y").value_counts().eq(12)].unique()
    )]
    out = []
    for month_end in m.index:
        Y = month_end.year
        prev_years = [Y-k for k in (1,2,3)]
        if not all((str(py) in annual_full.index.astype(str)) for py in prev_years):
            out.append(np.nan); continue
        A = np.mean([annual_full.loc[annual_full.index.astype(str)==str(py)].values[0] for py in prev_years])
        shares, mo = [], month_end.month
        for py in prev_years:
            mask = (m.index.year==py) & (m.index.month==mo)
            if not mask.any(): continue
            msum = m[(m.index.year==py)].sum()
            val  = float(m[mask].values[0])
            shares.append(val / msum if msum>0 else np.nan)
        share = np.nanmean(shares) if shares else np.nan
        out.append(A * share if np.isfinite(share) else np.nan)
    return pd.Series(out, index=m.index).ffill()

def ops_daily_from_monthly(series: pd.Series, monthly_baseline: pd.Series,
                           method: str = "profile") -> pd.Series:
    daily = series.copy()
    mb = monthly_baseline.dropna()
    if mb.empty:
        return pd.Series(index=daily.index, dtype=float)
    result_index = pd.date_range(min(daily.index.min().normalize(), mb.index.min().replace(day=1)),
                                 max(daily.index.max().normalize(), mb.index.max().normalize()), freq="B")
    result = pd.Series(index=result_index, dtype=float)
    for m, mval in mb.items():
        m_start = m.replace(day=1)
        days = pd.date_range(m_start, m, freq="B")
        if method == "profile":
            profiles = []
            for k in (1, 2, 3):
                my = m - pd.DateOffset(years=k)
                hist_days = pd.date_range(my.replace(day=1), my, freq="B")
                s = daily[(daily.index >= hist_days.min()) & (daily.index <= hist_days.max())]
                s = s.reindex(hist_days, fill_value=0.0)
                if s.sum() > 0:
                    p = (s / s.sum()).reindex(days, fill_value=0.0).values
                    profiles.append(p)
            if profiles:
                p = np.mean(np.vstack(profiles), axis=0); s = p.sum(); p = p / s if s > 0 else np.ones(len(days)) / len(days)
            else:
                p = np.ones(len(days)) / len(days
                                            )
            result.loc[days] = float(mval) * p
        else:
            vals = []
            for d in days:
                samples = [daily.get(d - pd.DateOffset(years=k), np.nan) for k in (1, 2, 3)]
                samples = [x for x in samples if pd.notna(x)]
                vals.append(np.mean(samples) if samples else np.nan)
            v = np.array(vals, dtype=float); s = np.nansum(v)
            if not np.isfinite(v).any() or s == 0:
                v = np.ones(len(days)) * (float(mval) / len(days))
            else:
                v = (v / s) * float(mval)
            result.loc[days] = v
    return result.reindex(daily.index)

def build_yearly_folds(y_index: pd.DatetimeIndex, min_train_years: int = 4) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(set(y_index.year))
    first_year, last_year = min(years), max(years)
    folds = []
    for Y in range(first_year + min_train_years, last_year + 1):
        train_end = (pd.Period(f"{Y-1}-12-31", freq="D").to_timestamp()).normalize()
        train_end = y_index[y_index <= train_end][-1] if (y_index <= train_end).any() else y_index[0]
        test_start = (pd.Period(f"{Y}-01-01", freq="D").to_timestamp()).normalize()
        if not (y_index >= test_start).any():
            continue
        Y_end = (pd.Period(f"{Y}-12-31", freq="D").to_timestamp()).normalize()
        candidates = y_index[(y_index >= test_start) & (y_index <= Y_end)]
        if candidates.empty:
            candidates = y_index[y_index >= test_start]
            if candidates.empty:
                continue
        test_end = candidates[-1]
        folds.append((train_end, test_start, test_end))
    return folds

# =========================
# Models (allow overrides)
# =========================

def forecast_naive_last(train_y: pd.Series) -> Tuple[float, float, float]:
    v = float(train_y.iloc[-1]); return v, np.nan, np.nan

def forecast_weekday_mean(train_y: pd.Series, target_date: pd.Timestamp) -> Tuple[float, float, float]:
    g = train_y.groupby(train_y.index.dayofweek).mean()
    v = float(g.get(target_date.dayofweek, train_y.mean())); return v, np.nan, np.nan

def forecast_movavg(train_y: pd.Series, window: int = 7) -> Tuple[float, float, float]:
    v = float(train_y.iloc[-window:].mean()); return v, np.nan, np.nan

def forecast_ets(train_y: pd.Series, h: int, seasonal_period: int, nominal_pi: float,
                 params: Optional[dict] = None) -> Tuple[float, float, float]:
    if params:
        try:
            m = ExponentialSmoothing(train_y,
                                     trend=params.get("trend"),
                                     seasonal=params.get("seasonal"),
                                     seasonal_periods=params.get("seasonal_periods", seasonal_period),
                                     damped_trend=params.get("damped_trend", False),
                                     initialization_method="estimated")
            r = m.fit(optimized=True, use_brute=False)
            f = r.get_forecast(steps=h)
            mean = float(f.predicted_mean.iloc[-1])
            ci = f.conf_int(alpha=1 - nominal_pi).iloc[-1]
            return mean, float(ci.iloc[0]), float(ci.iloc[1])
        except Exception:
            pass
    # small grid fallback
    specs = [
        dict(trend=None, seasonal=None),
        dict(trend="add", seasonal=None),
        dict(trend="add", seasonal="add"),
    ]
    best, best_aic = None, np.inf
    for spec in specs:
        try:
            m = ExponentialSmoothing(
                train_y,
                trend=spec["trend"],
                seasonal=spec["seasonal"],
                seasonal_periods=(seasonal_period if spec["seasonal"] else None),
                damped_trend=(spec["trend"] is not None),
                initialization_method="estimated",
            )
            r = m.fit(optimized=True, use_brute=False)
            if r.aic < best_aic:
                best_aic, best = r.aic, r
        except Exception:
            continue
    if best is None:
        v = float(train_y.iloc[-seasonal_period:].mean()); return v, np.nan, np.nan
    try:
        f = best.get_forecast(steps=h)
        mean = float(f.predicted_mean.iloc[-1])
        ci = f.conf_int(alpha=1 - nominal_pi).iloc[-1]
        return mean, float(ci.iloc[0]), float(ci.iloc[1])
    except Exception:
        v = float(train_y.iloc[-seasonal_period:].mean()); return v, np.nan, np.nan

def forecast_stl_arima(train_y: pd.Series, h: int, seasonal_period: int, nominal_pi: float,
                       cad_freq: str) -> Tuple[float, float, float]:
    try:
        stl = STL(train_y, period=seasonal_period, robust=True).fit()
        remainder = stl.resid + stl.trend
        freq = train_y.index.freqstr or cad_freq
        mod = SARIMAX(remainder, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False,
                      dates=remainder.index, freq=freq)
        res = mod.fit(disp=False, maxiter=200)
        f = res.get_forecast(steps=h)
        mean = float(f.predicted_mean.iloc[-1]) + float(
            stl.seasonal.iloc[-(seasonal_period - ((h - 1) % seasonal_period))]
        )
        ci = f.conf_int(alpha=1 - nominal_pi).iloc[-1]
        return mean, float(ci.iloc[0]), float(ci.iloc[1])
    except Exception:
        v = float(train_y.iloc[-1]); return v, np.nan, np.nan

def forecast_sarimax(train_y: pd.Series, exog_train: pd.DataFrame,
                     exog_future_h: pd.DataFrame, h: int, seasonal_period: int, nominal_pi: float,
                     cad_freq: str, params: Optional[dict] = None) -> Tuple[float, float, float]:
    freq = train_y.index.freqstr or cad_freq
    if params and (("order" in params) or ("seasonal_order" in params)):
        try:
            order = tuple(params.get("order", (1, 0, 1)))
            seas  = tuple(params.get("seasonal_order", (1, 0, 1, seasonal_period)))
            m = SARIMAX(train_y, exog=exog_train, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False,
                        dates=train_y.index, freq=freq)
            r = m.fit(disp=False, maxiter=200)
            f = r.get_forecast(steps=h, exog=exog_future_h)
            mean = float(f.predicted_mean.iloc[-1]); ci = f.conf_int(alpha=1-nominal_pi).iloc[-1]
            return mean, float(ci.iloc[0]), float(ci.iloc[1])
        except Exception:
            pass
    candidates = [((1, 0, 1), (0, 0, 0, 0)), ((1, 1, 1), (0, 0, 0, 0)), ((1, 0, 1), (1, 0, 1, seasonal_period))]
    best, best_aic = None, np.inf
    for order, seas in candidates:
        try:
            m = SARIMAX(train_y, exog=exog_train, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False,
                        dates=train_y.index, freq=freq)
            r = m.fit(disp=False, maxiter=200)
            if r.aic < best_aic:
                best_aic, best = r.aic, r
        except Exception:
            continue
    if best is None:
        return float(train_y.iloc[-1]), np.nan, np.nan
    try:
        f = best.get_forecast(steps=h, exog=exog_future_h)
        mean = float(f.predicted_mean.iloc[-1]); ci = f.conf_int(alpha=1-nominal_pi).iloc[-1]
        return mean, float(ci.iloc[0]), float(ci.iloc[1])
    except Exception:
        return float(train_y.iloc[-1]), np.nan, np.nan

def forecast_theta(train_y: pd.Series, h: int, seasonal_period: int, nominal_pi: float) -> Tuple[float, float, float]:
    if not HAS_THETA:
        return float(train_y.iloc[-1]), np.nan, np.nan
    try:
        tm = ThetaModel(train_y, period=seasonal_period); res = tm.fit()
        f = res.get_prediction(h); mean = float(f.predicted_mean.iloc[-1])
        return mean, np.nan, np.nan
    except Exception:
        return float(train_y.iloc[-1]), np.nan, np.nan

# =========================
# Evaluation & plotting (unchanged)
# =========================

def evaluate_block(df_pred: pd.DataFrame, target: str, horizon: int, ops_pred: Optional[pd.Series]) -> pd.DataFrame:
    rows = []
    for model, g in df_pred.groupby("model", sort=False):
        g = g.sort_values("date")
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = np.nan
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / ss_tot
        if is_stock(target):
            mape = np.nan; smp = np.nan
            m_true = pd.Series(y_true, index=pd.to_datetime(g["date"])).resample("ME").last()
            m_pred = pd.Series(y_pred, index=pd.to_datetime(g["date"])).resample("ME").last()
        else:
            mape = safe_mape(y_true, y_pred); smp  = smape(y_true, y_pred)
            m_true = pd.Series(y_true, index=pd.to_datetime(g["date"])).resample("ME").sum()
            m_pred = pd.Series(y_pred, index=pd.to_datetime(g["date"])).resample("ME").sum()
        tol10 = float(np.mean((np.abs(m_true - m_pred) / np.maximum(np.abs(m_true), 1e-9)) <= 0.10)) if len(m_true) else np.nan
        cover, width = coverage_width(
            y_true,
            g["y_lo"].to_numpy() if "y_lo" in g else None,
            g["y_hi"].to_numpy() if "y_hi" in g else None,
        )
        mae_skill = np.nan
        if (ops_pred is not None) and (not is_stock(target)):
            aligned = g.set_index("date").join(ops_pred.rename("ops_pred"), how="left")
            if aligned["ops_pred"].notna().any():
                mae_ops = float(np.mean(np.abs(aligned["y_true"] - aligned["ops_pred"])))
                mae_skill = (1.0 - mae / mae_ops) if mae_ops > 0 else np.nan
        rows.append({
            "target": target, "horizon": horizon, "model": model,
            "MAE": mae, "RMSE": rmse, "sMAPE": smp, "MAPE": mape, "R2": r2,
            "PI_coverage@90": cover, "PI_width@90": width,
            "Monthly_TOL10_Accuracy": tol10, "MAE_skill_vs_Ops": mae_skill,
        })
    return pd.DataFrame(rows)

def plot_overlay_all(df_slice: pd.DataFrame, target: str, h: int, ops_series: Optional[pd.Series], out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for model, g in df_slice.groupby("model"):
        ax.plot(g["date"], g["y_pred"], label=model, alpha=0.7)
    ax.plot(df_slice["date"], df_slice["y_true"], label="Actual", linewidth=2)
    if ops_series is not None:
        ops_aligned = ops_series.reindex(pd.to_datetime(df_slice["date"]))
        ax.plot(df_slice["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | h=D+{h} | ALL vs Ops")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_overlay_top(df_slice: pd.DataFrame, target: str, h: int, best_model: str,
                     ops_series: Optional[pd.Series], out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    g = df_slice[df_slice["model"] == best_model].sort_values("date")
    ax.plot(g["date"], g["y_pred"], label=best_model, linewidth=2)
    if g["y_lo"].notna().any() and g["y_hi"].notna().any():
        ax.fill_between(g["date"], g["y_lo"], g["y_hi"], alpha=0.2, label="PI 90%")
    ax.plot(g["date"], g["y_true"], label="Actual", linewidth=2)
    if ops_series is not None:
        ops_aligned = ops_series.reindex(pd.to_datetime(g["date"]))
        ax.plot(g["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | h=D+{h} | TOP={best_model} vs Ops")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_leaderboard_bar(mdf: pd.DataFrame, target: str, h: int, out_png: Path):
    g = mdf.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(g["model"], g["MAE"]); ax.invert_yaxis()
    ax.set_title(f"{target} | h=D+{h} | Leaderboard by MAE")
    ax.set_xlabel("MAE (lower is better)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_monthly_bars(df_slice: pd.DataFrame, target: str, h: int, ops_series: Optional[pd.Series], out_png: Path):
    idx = pd.to_datetime(df_slice["date"])
    s_true = pd.Series(df_slice["y_true"].to_numpy(), index=idx)
    s_pred = pd.Series(df_slice["y_pred"].to_numpy(), index=idx)
    if is_stock(target):
        m_true = s_true.resample("ME").last()
        m_pred = s_pred.resample("ME").last()
    else:
        m_true = s_true.resample("ME").sum()
        m_pred = s_pred.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    x = m_true.index
    ax.bar(x - pd.Timedelta(days=3), m_true.values, width=6, label="Actual")
    ax.bar(x, m_pred.values, width=6, label="TOP model", alpha=0.7)
    if ops_series is not None and (not is_stock(target)):
        ops_m = ops_series.reindex(idx).resample("ME").sum()
        ax.bar(x + pd.Timedelta(days=3), ops_m.values, width=6, label="Ops baseline", alpha=0.7)
    ax.set_title(f"{target} | h=D+{h} | Monthly aggregates")
    ax.legend(loc="best"); ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# =========================
# Main runner
# =========================

def _future_index_from_origin(origin_date: pd.Timestamp, h: int, cadence: str) -> pd.DatetimeIndex:
    if cadence == "Daily":
        return pd.date_range(origin_date + pd.offsets.BDay(1), periods=h, freq="B")
    if cadence == "Weekly":
        return pd.date_range(origin_date + pd.offsets.Week(weekday=4), periods=h, freq="W-FRI")
    return pd.date_range(origin_date + pd.offsets.MonthEnd(1), periods=h, freq="ME")

def run_pipeline(config: ConfigAStat = CONFIG) -> str:
    np.random.seed(config.random_seed)

    out_root = Path(config.out_root)
    ensure_dirs(str(out_root))

    # Load data
    df = pd.read_csv(config.data_path)
    dcol = [c for c in df.columns if c.lower() == config.date_col.lower()]
    if not dcol:
        raise ValueError(f"Date column '{config.date_col}' not found in {config.data_path}.")
    df[dcol[0]] = pd.to_datetime(df[dcol[0]])
    df = df.sort_values(dcol[0]).drop_duplicates(subset=[dcol[0]]).set_index(dcol[0])

    # Optional quick clip for demo-mode speed
    if config.demo_clip_months and config.demo_clip_months > 0:
        cutoff = df.index.max() - pd.DateOffset(months=config.demo_clip_months)
        df = df[df.index >= cutoff]

    # Prepare holidays series (aligned per target later)
    holidays_master = load_holidays(config.holidays_csv, idx=df.index)

    predictions_all, metrics_all = [], []

    cad_freq = _cad_freq(config.cadence)

    for target in config.targets:
        if target not in df.columns:
            print(f"[WARN] Target '{target}' not found; skipping.")
            continue

        y_raw = df[target].astype(float)
        y = regularize_series(y_raw, target, config.cadence)
        holidays = holidays_master.reindex(y.index).fillna(0).astype(int)
        exog_all = calendar_exog(y.index, holidays=holidays)

        # Ops baselines (flows)
        ops_daily = None
        if config.build_ops_baseline and (not is_stock(target)):
            m_base = ops_monthly_baseline(y if config.cadence!="Monthly" else y.asfreq("ME"))
            d_base = ops_daily_from_monthly(y if config.cadence=="Daily" else y_raw.asfreq("B", method="ffill").fillna(0.0),
                                            m_base, method=config.ops_daily_method)
            m_base.rename("forecast").to_csv(out_root / f"{target}_ops_baseline_monthly.csv", header=True)
            pd.DataFrame({"date": d_base.index, "forecast": d_base.values}).to_csv(
                out_root / f"{target}_ops_baseline_daily.csv", index=False
            )
            ops_daily = d_base

        folds = build_yearly_folds(y.index, min_train_years=config.min_train_years)
        if config.max_folds:
            folds = folds[-int(config.max_folds):]
        if not folds:
            raise RuntimeError("No folds were created. Check data span or min_train_years.")

        for h in config.horizons:
            for (train_end, test_start, test_end) in folds:
                test_idx_full = y[(y.index >= test_start) & (y.index <= test_end)].index
                if len(test_idx_full) == 0:
                    continue
                target_positions = np.where(np.isin(y.index, test_idx_full))[0]
                target_positions = [pos for pos in target_positions if pos - h >= 0]
                if len(target_positions) == 0:
                    continue

                for model in config.models:
                    preds, los, his, ytrues, dates = [], [], [], [], []
                    last_refit_origin_pos = None

                    for pos_t in target_positions:
                        t = y.index[pos_t]
                        pos_origin = pos_t - h
                        origin_date = y.index[pos_origin]

                        # Train up to origin (clip to fold)
                        train_y = y.iloc[:pos_origin + 1]
                        if origin_date > train_end:
                            train_y = train_y[train_y.index <= train_end]
                            if train_y.empty:
                                continue

                        # Future exog
                        future_idx = _future_index_from_origin(origin_date, h, config.cadence)
                        exog_train  = exog_all.loc[train_y.index]
                        exog_future = exog_all.reindex(future_idx)

                        # Per-model overrides (optional)
                        p = config.model_params.get(model, {})

                        if model == "naive_last":
                            mean, lo, hi = forecast_naive_last(train_y)

                        elif model == "weekday_mean":
                            mean, lo, hi = forecast_weekday_mean(train_y, target_date=t)

                        elif model == "movavg7":
                            mean, lo, hi = forecast_movavg(train_y, window=int(p.get("window", 7)))

                        elif model == "ets":
                            mean, lo, hi = forecast_ets(train_y, h=h, seasonal_period=config.seasonal_period,
                                                        nominal_pi=config.nominal_pi, params=p or None)

                        elif model == "stl_arima":
                            mean, lo, hi = forecast_stl_arima(train_y, h=h, seasonal_period=config.seasonal_period,
                                                              nominal_pi=config.nominal_pi, cad_freq=cad_freq)

                        elif model == "sarimax":
                            if (last_refit_origin_pos is None) or ((pos_origin - last_refit_origin_pos) >= config.sarimax_refit_every):
                                last_refit_origin_pos = pos_origin
                            mean, lo, hi = forecast_sarimax(train_y, exog_train, exog_future, h=h,
                                                            seasonal_period=config.seasonal_period,
                                                            nominal_pi=config.nominal_pi,
                                                            cad_freq=cad_freq, params=p or None)

                        elif model == "theta" and HAS_THETA:
                            mean, lo, hi = forecast_theta(train_y, h=h, seasonal_period=config.seasonal_period,
                                                          nominal_pi=config.nominal_pi)
                        else:
                            continue

                        preds.append(mean); los.append(lo); his.append(hi)
                        ytrues.append(float(y.loc[t])); dates.append(t)

                    if dates:
                        df_m = pd.DataFrame({
                            "date": dates, "target": target, "horizon": h, "model": model,
                            "y_true": ytrues, "y_pred": preds, "y_lo": los, "y_hi": his,
                            "split_id": f"{train_end.date()}→{test_start.date()}..{test_end.date()}",
                            "defn_variant": "Bridge",
                        })
                        predictions_all.append(df_m)

                        # quick fold metrics in log (progress visibility)
                        try:
                            g = df_m.sort_values("date")
                            y_true = g["y_true"].to_numpy(); y_pred = g["y_pred"].to_numpy()
                            mae = float(np.mean(np.abs(y_true - y_pred)))
                            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                            smp = smape(y_true, y_pred)
                            print(f"[runner]  Fold {len(predictions_all)}: train={len(train_y):d}, test={len(g):d}, MAE={mae:,.2f}, RMSE={rmse:,.2f}, sMAPE={smp*100:.2f}%")
                        except Exception:
                            pass

            # after all folds for this target×horizon → metrics + plots
            if predictions_all:
                df_pred_h = pd.concat([p for p in predictions_all if (p["target"].iloc[0] == target and p["horizon"].iloc[0] == h)], ignore_index=True)
                ops_series = None if (is_stock(target) or ops_daily is None) else ops_daily.loc[df_pred_h["date"].min(): df_pred_h["date"].max()]
                mdf = evaluate_block(df_pred_h, target, h, ops_pred=ops_series)
                metrics_all.append(mdf)

                lb = mdf.sort_values("MAE").reset_index(drop=True)
                best_model = lb.iloc[0]["model"] if len(lb) else None
                lb.assign(target=target, horizon=h).to_csv(
                    out_root / f"leaderboard_{target.replace(' ','_')}_h{h}.csv", index=False
                )
                try:
                    plot_overlay_all(df_pred_h, target, h, ops_series, out_root / f"{target.replace(' ','_')}_h{h}_overlay_all.png")
                    if best_model:
                        plot_overlay_top(df_pred_h, target, h, best_model, ops_series,
                                         out_root / f"{target.replace(' ','_')}_h{h}_overlay_top.png")
                        plot_monthly_bars(
                            df_pred_h[df_pred_h["model"] == best_model], target, h, ops_series,
                            out_root / f"{target.replace(' ','_')}_h{h}_monthly_bars_top_vs_ops.png"
                        )
                        plot_leaderboard_bar(mdf, target, h, out_root / f"{target.replace(' ','_')}_h{h}_leaderboard_mae.png")
                except Exception as e:
                    print(f"[WARN] Plotting failed for {target} h={h}: {e}")

    # Save master tables (stable names)
    if predictions_all:
        pd.concat(predictions_all, ignore_index=True).sort_values(["target","horizon","date","model"]).to_csv(out_root / "predictions_long.csv", index=False)
    if metrics_all:
        metr = pd.concat(metrics_all, ignore_index=True)
        metr.to_csv(out_root / "metrics_long.csv", index=False)
        # global leaderboard
        glb_rows = []
        for (t, h), g in metr.groupby(["target", "horizon"]):
            g2 = g.groupby("model", as_index=False)["MAE"].mean().sort_values("MAE")
            for rank, row in enumerate(g2.itertuples(index=False), start=1):
                glb_rows.append({"target": t, "horizon": h, "model": row.model, "MAE": row.MAE, "rank": rank})
        pd.DataFrame(glb_rows).to_csv(out_root / "leaderboard.csv", index=False)

    # Run manifest + config snapshot
    with open(out_root / "artifacts" / "RUN.md", "w", encoding="utf-8") as f:
        f.write(f"# A · Statistical run (master outputs)\n\n")
        f.write(f"- timestamp: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- data_path: {config.data_path}\n")
        f.write(f"- cadence: {config.cadence}\n")
        f.write(f"- targets: {config.targets}\n")
        f.write(f"- horizons: {config.horizons}\n")
        f.write(f"- models: {config.models}\n")
        f.write(f"- min_train_years: {config.min_train_years}\n")
        f.write(f"- seasonal_period: {config.seasonal_period}\n")
        f.write(f"- max_folds: {config.max_folds}\n")
        f.write(f"- demo_clip_months: {config.demo_clip_months}\n")
        f.write(f"\nMaster files written to: {config.out_root}\n")
        f.write("- predictions_long.csv\n- metrics_long.csv\n- leaderboard.csv\n")
        f.write("- <target>_ops_baseline_daily.csv (flows)\n- <target>_ops_baseline_monthly.csv (flows)\n")
        f.write("- plots: *_overlay_all.png, *_overlay_top.png, *_monthly_bars_top_vs_ops.png, *_leaderboard_mae.png\n")

    with open(out_root / "artifacts" / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"[OK] Master outputs in: {config.out_root}")
    return str(config.out_root)

if __name__ == "__main__":
    run_pipeline(CONFIG)
