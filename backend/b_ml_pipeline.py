# -*- coding: utf-8 -*-
"""
B · Machine Learning pipeline (univariate & multivariate)
--------------------------------------------------------

Contract (same as A·Stat):
- Read env vars: TG_FAMILY, TG_MODEL_FILTER (optional), TG_TARGET, TG_CADENCE,
  TG_HORIZON, TG_DATA_PATH (absolute), TG_DATE_COL, TG_PARAM_OVERRIDES (JSON),
  TG_OUT_ROOT (absolute).
- Write under TG_OUT_ROOT:
  - predictions_long.csv, metrics_long.csv, leaderboard.csv
  - plots: *_overlay_all.png, *_overlay_top.png, *_monthly_bars_top_vs_ops.png, *_leaderboard_mae.png
  - artifacts/RUN.md, artifacts/config.json
  - For flows: <target>_ops_baseline_daily.csv / _monthly.csv

Models exposed:
  Univariate  : Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, HistGBDT
  Multivariate: same set (with optional exogenous/top-K)
  Optional    : XGBoost, LightGBM (if installed)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Optional libraries
try:
    from xgboost import XGBRegressor  # type: ignore
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False


# =========================
# Config
# =========================

@dataclass
class ConfigBML:
    data_path: str
    date_col: str
    target: str
    cadence: str                 # "Daily" | "Weekly" | "Monthly"
    horizon: int
    variant: str                 # "uni" | "multi"
    model_filter: Optional[str]  # if None => run all
    out_root: str

    # overrides
    folds: Optional[int] = None
    min_train_years: int = 4
    demo_clip_months: Optional[int] = None

    # feature recipe
    lags_daily: List[int] = None
    windows_daily: List[int] = None
    lags_weekly: List[int] = None
    windows_weekly: List[int] = None
    lags_monthly: List[int] = None
    windows_monthly: List[int] = None
    exog_top_k: int = 12

    random_seed: int = 42
    nominal_pi: float = 0.90
    allow_zero_values: bool = True  # If True, skip the zero-replacement heuristic (zeros are valid data)

    # ✅ Delta modeling option for stock/level targets
    use_delta_modeling: bool = False  # If True, model delta = y(t+h) - y(t) instead of y(t+h)

    def __post_init__(self):
        # Auto-enable delta modeling for stock/level targets.
        # For stocks, predicting delta = y(t+h) - y(t) avoids the persistence
        # trap where the model learns y_hat ≈ y_origin because the level
        # dominates all features.  Delta modeling forces genuine learning.
        if not self.use_delta_modeling and is_stock(self.target):
            self.use_delta_modeling = True
            print(f"[config] Auto-enabled delta modeling for stock target '{self.target}'")

        if self.lags_daily is None:
            self.lags_daily = [1, 3, 7, 14]
        if self.windows_daily is None:
            self.windows_daily = [3, 7, 14]
        if self.lags_weekly is None:
            self.lags_weekly = [1, 4, 12]
        if self.windows_weekly is None:
            self.windows_weekly = [4, 8, 12]
        if self.lags_monthly is None:
            self.lags_monthly = [1, 12]
        if self.windows_monthly is None:
            self.windows_monthly = [3, 6, 12]


# =========================
# Utilities & calendar
# =========================

def ensure_dirs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)


def is_stock(target: str) -> bool:
    return target.strip().lower() in {"state budget balance", "balance", "t0"}


def to_business_index(df: pd.DataFrame, date_col: str, target: str) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col])
    s = df.set_index(date_col)[target].astype(float)
    bidx = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="B")
    s = s.reindex(bidx)
    s.index.freq = "B"
    if is_stock(target):
        return s.ffill()
    return s.fillna(0.0)


def calendar_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_eom"] = ((idx + pd.offsets.BDay(1)).to_period("M") != idx.to_period("M")).astype(int)
    df["is_eoq"] = ((idx + pd.offsets.BDay(1)).to_period("Q") != idx.to_period("Q")).astype(int)
    dow = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
    return pd.concat([df.drop(columns="dow"), dow], axis=1)


def build_yearly_folds(idx: pd.DatetimeIndex, min_train_years: int, folds_override: Optional[int]) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Annual rolling-origin folds on business-day index. If folds_override is set,
    only the last N folds are kept to control runtime.
    """
    years = sorted(set(idx.year))
    first_year, last_year = min(years), max(years)
    folds = []
    for Y in range(first_year + min_train_years, last_year + 1):
        train_end = (pd.Period(f"{Y-1}-12-31", freq="D").to_timestamp()).normalize()
        train_end = idx[idx <= train_end][-1] if (idx <= train_end).any() else idx[0]
        test_start = (pd.Period(f"{Y}-01-01", freq="D").to_timestamp()).normalize()
        year_end = (pd.Period(f"{Y}-12-31", freq="D").to_timestamp()).normalize()
        test_idx = idx[(idx >= test_start) & (idx <= year_end)]
        if test_idx.empty:
            continue
        test_end = test_idx[-1]
        folds.append((train_end, test_start, test_end))
    if folds_override and len(folds) > folds_override:
        folds = folds[-folds_override:]
    return folds


def ops_monthly_baseline(series: pd.Series) -> pd.Series:
    m = series.resample("ME").sum().astype(float)
    # fallback: 3y same-month rolling mean
    return m.groupby(m.index.month).transform(lambda x: x.shift(12).rolling(36, min_periods=1).mean())


def ops_daily_from_monthly(series: pd.Series, monthly_baseline: pd.Series) -> pd.Series:
    daily = series.copy()
    mb = monthly_baseline.dropna()
    out = pd.Series(index=daily.index, dtype=float)
    for m, mval in mb.items():
        days = pd.date_range(m.replace(day=1), m, freq="B")
        if len(days) == 0:
            continue
        out.loc[days] = float(mval) / len(days)
    return out.reindex(daily.index)


# =========================
# Features
# =========================

def lag_window_features(s: pd.Series, lags: List[int], windows: List[int]) -> pd.DataFrame:
    """
    Build lag and rolling window features.

    For lag=0 (if present) this is the current value at origin_date — equivalent
    to a persistence anchor.  It is **no longer included by default** because it
    dominates the model for stock targets.  See ``choose_recipe`` for details.
    """
    feats = pd.DataFrame(index=s.index)
    for L in lags:
        L = int(L)
        if L == 0:
            feats["lag_0"] = s
        else:
            feats[f"lag_{L}"] = s.shift(L)
    for W in windows:
        feats[f"rmean_{W}"] = s.rolling(W, min_periods=1).mean().shift(1)
    return feats


def choose_recipe(cfg: ConfigBML) -> Tuple[List[int], List[int]]:
    """
    Choose lag and window recipe based on cadence.

    NOTE: lag_0 is intentionally NOT auto-added.  Including lag_0 (the value
    at origin date) makes the model a trivial persistence predictor for
    stock/level targets and masks genuine learning for flow targets.
    If users explicitly include 0 in their lag list via overrides it will be
    honoured, but the pipeline will log a warning.
    """
    cad = cfg.cadence.lower()
    if cad == "daily":
        lags, wins = list(cfg.lags_daily), list(cfg.windows_daily)
    elif cad == "weekly":
        lags, wins = list(cfg.lags_weekly), list(cfg.windows_weekly)
    elif cad == "monthly":
        lags, wins = list(cfg.lags_monthly), list(cfg.windows_monthly)
    else:
        lags, wins = list(cfg.lags_daily), list(cfg.windows_daily)

    # When delta modeling is active, include lag_0 (origin value) as a feature.
    # This is safe because delta modeling removes the persistence component:
    # we model delta = y(t+h) - y(t), so lag_0 provides change-context, not level.
    if cfg.use_delta_modeling and 0 not in lags:
        lags = [0] + lags
        print(f"[config] Added lag_0 to features (delta modeling active, "
              f"lag_0 provides change-context not level info)")

    # Warn if user explicitly included lag_0 WITHOUT delta modeling
    if 0 in lags and not cfg.use_delta_modeling:
        print(f"[WARN] lag_0 (origin value) is included in lags={lags} "
              f"WITHOUT delta modeling. This makes the model a near-persistence "
              f"predictor for stock targets. Consider enabling delta modeling or "
              f"removing lag_0.")

    return lags, wins


def multivariate_exog(train_df: pd.DataFrame, target: str, top_k: int) -> List[str]:
    # pick top_k columns (excluding target) by absolute Pearson corr with target
    corr = train_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").abs().sort_values(ascending=False)
    return list(corr.head(max(0, top_k)).index)


# =========================
# Models
# =========================

def available_models() -> Dict[str, object]:
    models = {
        "Ridge": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler(with_mean=True, with_std=True)),
                           ("est", Ridge(random_state=0))]),
        "Lasso": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler(with_mean=True, with_std=True)),
                           ("est", Lasso(random_state=0, max_iter=20000, tol=1e-4))]),
        "ElasticNet": Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler(with_mean=True, with_std=True)),
                                ("est", ElasticNet(random_state=0))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=-1),
        "HistGBDT": HistGradientBoostingRegressor(random_state=0),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            random_state=0, tree_method="hist", n_jobs=-1
        )
    if HAVE_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            random_state=0, n_jobs=-1
        )
    return models


# =========================
# Train / evaluate
# =========================

def evaluate_block(df_pred: pd.DataFrame, target: str, horizon: int, ops_series: Optional[pd.Series]) -> pd.DataFrame:
    rows = []
    for model, g in df_pred.groupby("model", sort=False):
        g = g.sort_values("date")
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = np.nan
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / ss_tot

        if is_stock(target):
            m_true = pd.Series(y_true, index=pd.to_datetime(g["date"])).resample("ME").last()
            m_pred = pd.Series(y_pred, index=pd.to_datetime(g["date"])).resample("ME").last()
            mape = np.nan
            smp = np.nan
        else:
            m_true = pd.Series(y_true, index=pd.to_datetime(g["date"])).resample("ME").sum()
            m_pred = pd.Series(y_pred, index=pd.to_datetime(g["date"])).resample("ME").sum()
            eps = 1e-9
            mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)))
            smp = float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))

        tol10 = float(np.mean((np.abs(m_true - m_pred) / np.maximum(np.abs(m_true), 1e-9)) <= 0.10)) if len(m_true) else np.nan

        # Compute PI coverage if intervals exist
        pi_cov = np.nan
        pi_width = np.nan
        if "y_lo" in g.columns and "y_hi" in g.columns:
            lo = g["y_lo"].to_numpy()
            hi = g["y_hi"].to_numpy()
            valid_pi = np.isfinite(lo) & np.isfinite(hi)
            if valid_pi.any():
                pi_cov = float(np.mean((y_true[valid_pi] >= lo[valid_pi]) & (y_true[valid_pi] <= hi[valid_pi])))
                pi_width = float(np.mean(hi[valid_pi] - lo[valid_pi]))

        rows.append({
            "target": target,
            "horizon": horizon,
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smp,
            "MAPE": mape,
            "R2": r2,
            "PI_coverage@90": pi_cov,
            "PI_width@90": pi_width,
            "Monthly_TOL10_Accuracy": tol10,
            "MAE_skill_vs_Ops": np.nan,
        })
    return pd.DataFrame(rows)


def plot_overlay_all(df_slice: pd.DataFrame, target: str, h: int, ops_series: Optional[pd.Series], out_png: Path):
    # ✅ FIX 5: Only plot out-of-sample predictions - drop NaNs in y_true/y_pred
    df_plot = df_slice.dropna(subset=["y_true", "y_pred"]).copy()
    if df_plot.empty:
        print(f"[WARN] No valid predictions to plot for {target} h={h}")
        return
    
    # ✅ FIX 5: Use target_date (not origin_date) for x-axis
    date_col = "target_date" if "target_date" in df_plot.columns else "date"
    fig, ax = plt.subplots(figsize=(12, 4))
    for model, g in df_plot.groupby("model"):
        ax.plot(g[date_col], g["y_pred"], label=model, alpha=0.7)
    ax.plot(df_plot[date_col], df_plot["y_true"], label="Actual", linewidth=2, color="black")
    if ops_series is not None and (not is_stock(target)):
        ops_aligned = ops_series.reindex(pd.to_datetime(df_plot[date_col]))
        ax.plot(df_plot[date_col], ops_aligned.values, label="Ops baseline", linestyle="--")
    
    # ✅ FIX 5: Add debug markers showing origin_date → target_date arrows for a few samples
    if "origin_date" in df_plot.columns:
        sample_rows = df_plot.sample(min(5, len(df_plot))) if len(df_plot) > 5 else df_plot
        for _, row in sample_rows.iterrows():
            origin = pd.to_datetime(row["origin_date"])
            target = pd.to_datetime(row[date_col])
            if pd.notna(origin) and pd.notna(target):
                # Draw arrow from origin to target
                ax.annotate('', xy=(target, row["y_true"]), xytext=(origin, row["y_true"]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))
    
    ax.set_title(f"{target} | h=D+{h} | ALL vs Ops (ML) [Out-of-sample only]")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_overlay_top(df_slice: pd.DataFrame, target: str, h: int, best_model: str,
                     ops_series: Optional[pd.Series], out_png: Path):
    # ✅ Only plot out-of-sample predictions - drop NaNs in y_true/y_pred
    g = df_slice[df_slice["model"] == best_model].dropna(subset=["y_true", "y_pred"]).copy()
    if g.empty:
        print(f"[WARN] No valid predictions to plot for {target} h={h} model={best_model}")
        return
    
    # ✅ Use target_date (not origin_date) for x-axis
    date_col = "target_date" if "target_date" in g.columns else "date"
    g = g.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(g[date_col], g["y_pred"], label=best_model, linewidth=2)
    ax.plot(g[date_col], g["y_true"], label="Actual", linewidth=2)
    if ops_series is not None and (not is_stock(target)):
        ops_aligned = ops_series.reindex(pd.to_datetime(g[date_col]))
        ax.plot(g[date_col], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | h=D+{h} | TOP={best_model} vs Ops (ML) [Out-of-sample only]")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_leaderboard_bar(mdf: pd.DataFrame, target: str, h: int, out_png: Path):
    g = mdf.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(g) + 1)))
    # Colour persistence baseline differently so it stands out as reference line
    colours = ["#d62728" if "Persistence" in str(m) or "baseline" in str(m).lower()
                else "#1f77b4" for m in g["model"]]
    ax.barh(g["model"], g["MAE"], color=colours)
    ax.invert_yaxis()
    ax.set_title(f"{target} | h=D+{h} | Leaderboard by MAE (ML)")
    ax.set_xlabel("MAE (lower is better)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_monthly_bars(df_slice: pd.DataFrame, target: str, h: int, ops_series: Optional[pd.Series], out_png: Path):
    # ✅ Only plot out-of-sample predictions - drop NaNs
    df_plot = df_slice.dropna(subset=["y_true", "y_pred"]).copy()
    if df_plot.empty:
        print(f"[WARN] No valid predictions to plot for {target} h={h}")
        return
    
    # ✅ Use target_date (not origin_date) for x-axis
    date_col = "target_date" if "target_date" in df_plot.columns else "date"
    idx = pd.to_datetime(df_plot[date_col])
    s_true = pd.Series(df_plot["y_true"].to_numpy(), index=idx)
    s_pred = pd.Series(df_plot["y_pred"].to_numpy(), index=idx)
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
    ax.set_title(f"{target} | h=D+{h} | Monthly aggregates (ML)")
    ax.legend(loc="best"); ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# =========================
# Main ML pipeline
# =========================

def run_pipeline_ml(cfg: ConfigBML) -> str:
    np.random.seed(cfg.random_seed)
    out_root = Path(cfg.out_root).resolve()
    ensure_dirs(out_root)

    # Load data and standardize index/cadence
    raw = pd.read_csv(cfg.data_path)
    s = to_business_index(raw, cfg.date_col, cfg.target)
    
    # ✅ FIX 2: Standardize horizon definition - h = steps forward in index (not calendar days)
    # Log inferred frequency for debugging
    try:
        inferred_freq = pd.infer_freq(s.index)
        print(f"[pipeline] Inferred frequency: {inferred_freq} (horizon h={cfg.horizon} means {cfg.horizon} steps forward in index)")
    except:
        print(f"[pipeline] Could not infer frequency (horizon h={cfg.horizon} means {cfg.horizon} steps forward in index)")
    
    # Data sanity: optionally detect and replace suspicious zeros
    if len(s) > 0:
        if not cfg.allow_zero_values:
            last_10 = s.tail(10)
            zero_count = (last_10 == 0.0).sum()
            if zero_count > 0:
                if not is_stock(cfg.target):
                    median_val = s.median()
                    if median_val > 0:
                        suspicious_zeros = last_10[(last_10 == 0.0) & (abs(last_10 - median_val) / median_val > 0.5)]
                        if len(suspicious_zeros) > 0:
                            print(f"[WARN] Found {len(suspicious_zeros)} suspicious zero values in last 10 observations. "
                                  f"Treating as NaN (allow_zero_values=False).")
                            s.loc[suspicious_zeros.index] = np.nan
                else:
                    print(f"[WARN] Found {zero_count} zero values in last 10 observations for stock target. "
                          f"Treating as NaN (allow_zero_values=False).")
                    s.loc[last_10[last_10 == 0.0].index] = np.nan
            s = s.dropna()
            if len(s) == 0:
                raise RuntimeError("All data points were dropped after zero handling. Check data quality.")

        print(f"[pipeline] Data loaded: {len(s)} observations from {s.index.min().date()} to {s.index.max().date()}")
    
    cal = calendar_exog(s.index)

    # Optional demo clip (kept for compatibility)
    if cfg.demo_clip_months and cfg.demo_clip_months > 0:
        last = s.index.max()
        first = last - pd.offsets.BMonthBegin(cfg.demo_clip_months)
        s = s[s.index >= first]
        cal = cal.loc[s.index]

    # Treasury baseline (flows only) written once
    ops_daily = None
    if not is_stock(cfg.target):
        mbase = ops_monthly_baseline(s)
        dbase = ops_daily_from_monthly(s, mbase)
        mbase.rename("forecast").to_csv(out_root / f"{cfg.target}_ops_baseline_monthly.csv", index_label="date")
        pd.DataFrame({"date": dbase.index, "forecast": dbase.values}).to_csv(
            out_root / f"{cfg.target}_ops_baseline_daily.csv", index=False
        )
        ops_daily = dbase

    # Folds & features
    # ✅ Ensure folds is set: if None, default based on data availability
    if cfg.folds is None:
        # Calculate max possible folds and use min(5, max_possible) for thorough mode
        years = sorted(set(s.index.year))
        max_folds = max(0, len(years) - cfg.min_train_years)
        cfg.folds = min(5, max_folds) if max_folds > 0 else 1
        print(f"[pipeline] folds was None, set to {cfg.folds} based on data availability")
    
    folds = build_yearly_folds(s.index, cfg.min_train_years, cfg.folds)
    lags, wins = choose_recipe(cfg)
    print(f"[pipeline] Recipe: lags={lags}, windows={wins}, "
          f"delta_modeling={cfg.use_delta_modeling}, is_stock={is_stock(cfg.target)}")

    # ✅ Log fold details
    print(f"[pipeline] Using {len(folds)} folds (requested={cfg.folds})")
    for i, (train_end, test_start, test_end) in enumerate(folds, 1):
        train_start = s.index[0]  # First available data point
        print(f"[pipeline] Fold {i}/{len(folds)}: train=[{train_start.date()}..{train_end.date()}], "
              f"test=[{test_start.date()}..{test_end.date()}]")

    # Model set
    all_models = available_models()
    if cfg.model_filter:
        all_models = {k: v for k, v in all_models.items() if k.lower() == cfg.model_filter.lower()}
    if not all_models:
        raise ValueError(f"No ML model selected (filter={cfg.model_filter}).")

    predictions_all = []

    # Pre-build multivariate frame if needed (other columns at 'B' freq)
    df_multi = None
    if cfg.variant == "multi":
        df_multi = raw.copy()
        df_multi[cfg.date_col] = pd.to_datetime(df_multi[cfg.date_col], errors="coerce")
        df_multi = df_multi.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col).drop_duplicates(subset=[cfg.date_col])
        df_multi = df_multi.set_index(cfg.date_col)
        # reindex all numeric columns to 'B' like target
        for c in df_multi.select_dtypes(include=[np.number]).columns:
            if c == cfg.target:
                continue
            df_multi[c] = df_multi[c].reindex(s.index).ffill().fillna(0.0)

    # Iterate horizons & folds (here a single horizon from env)
    h = int(cfg.horizon)
    print(f"[pipeline] Starting ML pipeline: {len(folds)} folds, horizon={h}, models={list(all_models.keys())}")
    for fold_idx, (train_end, test_start, test_end) in enumerate(folds, 1):
        print(f"[pipeline] Processing fold {fold_idx}/{len(folds)}: train_end={train_end.date()}, test={test_start.date()}..{test_end.date()}")
        # test B-days within the year Y
        test_idx_full = s[(s.index >= test_start) & (s.index <= test_end)].index
        if test_idx_full.empty:
            continue

        # candidate positions where origin exists (pos - h)
        positions = np.where(np.isin(s.index, test_idx_full))[0]
        positions = [pos for pos in positions if pos - h >= 0]
        if not positions:
            continue

        # Build training slice strictly up to train_end
        train_mask = s.index <= train_end
        s_train_full = s.loc[train_mask]

        # Multivariate: pick top-K exog from train only
        exog_cols = []
        if cfg.variant == "multi" and df_multi is not None:
            temp = df_multi.loc[s_train_full.index, [cfg.target] + [c for c in df_multi.columns if c != cfg.target]]
            exog_cols = multivariate_exog(temp, cfg.target, cfg.exog_top_k)

        # ── FIT-ONCE-PER-FOLD approach ────────────────────────────────
        # The model is fit ONCE on training data (all data <= train_end),
        # then predicts at each test origin.  This is standard expanding-
        # window CV and makes fold-level comparison fair across families.
        #
        # Why this is correct: within a fold, every origin date is in the
        # test period (>= test_start > train_end).  The old code did
        # s_train = s[:origin]; s_train = s_train[s.index <= train_end],
        # which always yields s_train_full regardless of origin.  So the
        # per-origin re-fit was redundant — same training data every time.
        #
        # Per-origin work is limited to building the feature vector at the
        # origin date (using data up to that date) and calling predict().

        # PHASE 1: Build training features & target ONCE
        f_train_full = lag_window_features(s_train_full, lags, wins).join(cal.loc[s_train_full.index])
        if exog_cols:
            f_train_full = f_train_full.join(df_multi.loc[s_train_full.index, exog_cols])

        # Step-based h-step-ahead target (vectorized).
        # shift(-h) places s[i+h] at position i; last h rows become NaN.
        y_target_full = s_train_full.shift(-h)

        # Delta modeling for stock/level targets
        if cfg.use_delta_modeling and is_stock(cfg.target):
            y_target_full = y_target_full - s_train_full

        # Combine and drop rows with NaN (initial lags + trailing h rows)
        train_combined = pd.concat([f_train_full, y_target_full.rename("y")], axis=1).dropna()
        if len(train_combined) < 10:
            print(f"[pipeline] Fold {fold_idx}: too few training samples ({len(train_combined)}), skipping")
            continue
        X_all_fold, y_all_fold = train_combined.drop(columns=["y"]), train_combined["y"]

        # Pre-build feature matrix covering all possible origin dates for
        # prediction.  lag_window_features is purely backward-looking
        # (shift(k>=0), rolling().shift(1)), so features at any date only
        # depend on data at-or-before that date.  Building on the full
        # series and slicing a row is equivalent to building on s[:origin].
        last_origin_date = s.index[max(positions) - h] if positions else test_end
        s_for_pred_feats = s[s.index <= last_origin_date]
        feat_all_origins = lag_window_features(s_for_pred_feats, lags, wins).join(cal.loc[s_for_pred_feats.index])
        if exog_cols:
            feat_all_origins = feat_all_origins.join(df_multi.loc[s_for_pred_feats.index, exog_cols])

        for model_name, estimator in all_models.items():
            import time
            model_start_time = time.time()

            # PHASE 2: Train/val split ONCE
            n_total = len(X_all_fold)
            if n_total >= 50:
                n_val = max(10, int(n_total * 0.2))
                X_tr_fit, y_tr_fit = X_all_fold.iloc[:-n_val], y_all_fold.iloc[:-n_val]
                X_tr_val, y_tr_val = X_all_fold.iloc[-n_val:], y_all_fold.iloc[-n_val:]
            else:
                X_tr_fit, y_tr_fit = X_all_fold, y_all_fold
                X_tr_val, y_tr_val = None, None

            # PHASE 3: Fit model ONCE per fold
            estimator.fit(X_tr_fit, y_tr_fit)

            if model_name == "Lasso":
                lasso_est = estimator.named_steps.get("est") if hasattr(estimator, "named_steps") else estimator
                if hasattr(lasso_est, "n_iter_"):
                    n_iter = lasso_est.n_iter_
                    coef_norm = float(np.linalg.norm(lasso_est.coef_)) if hasattr(lasso_est, "coef_") else np.nan
                    print(f"[pipeline] Lasso convergence: n_iter={n_iter}, ||coef||={coef_norm:.2f} "
                          f"(fold={fold_idx}, n_features={X_tr_fit.shape[1]})")

            # PHASE 4: Compute conformal PI radius ONCE from validation set
            conformal_radius = np.nan
            train_mae_fold, val_mae_fold = np.nan, np.nan
            if X_tr_val is not None and len(X_tr_val) > 5:
                y_pred_val = estimator.predict(X_tr_val)
                abs_residuals = np.abs(y_tr_val.values - y_pred_val)
                val_mae_fold = float(np.mean(abs_residuals))
                train_mae_fold = float(np.mean(np.abs(y_tr_fit - estimator.predict(X_tr_fit))))
                conformal_radius = float(np.quantile(abs_residuals, cfg.nominal_pi))
                ratio = val_mae_fold / train_mae_fold if train_mae_fold > 0 else np.nan
                print(f"[pipeline] {model_name} fold {fold_idx}: "
                      f"train_MAE={train_mae_fold:.2f}, val_MAE={val_mae_fold:.2f}, "
                      f"ratio={ratio:.2f}, conformal_r={conformal_radius:.2f}")
                if not np.isnan(ratio) and ratio > 1.5:
                    print(f"[WARN] Generalization gap for {model_name}: "
                          f"val/train ratio={ratio:.2f} — possible overfitting")

            # PHASE 5: Predict at each test origin (lightweight per-point loop)
            preds, ytrues, target_dates, origin_dates, origin_values = [], [], [], [], []

            for pos_t in positions:
                t = s.index[pos_t]           # target date
                pos_origin = pos_t - h
                origin = s.index[pos_origin] # origin date

                # GUARD A: alignment check
                if pos_origin + h >= len(s.index):
                    continue
                if s.index[pos_origin + h] != t:
                    print(f"[ERROR] Alignment mismatch: origin={origin.date()}, "
                          f"expected={s.index[pos_origin + h].date()}, actual={t.date()}, h={h}")
                    continue

                # Build feature vector at origin from pre-computed matrix.
                # This is leakage-safe: features at origin depend only on
                # data <= origin (backward-looking lags and rolling windows).
                if origin in feat_all_origins.index:
                    x_pred = feat_all_origins.loc[[origin]].fillna(0.0)
                else:
                    # Origin not in pre-computed features (shouldn't happen)
                    s_up_to_origin = s[s.index <= origin]
                    feat_at_origin = lag_window_features(s_up_to_origin, lags, wins).join(cal.loc[s_up_to_origin.index])
                    if exog_cols:
                        feat_at_origin = feat_at_origin.join(df_multi.loc[[origin], exog_cols])
                    x_pred = feat_at_origin.loc[[origin]].fillna(0.0)

                delta_hat = float(estimator.predict(x_pred)[0])

                # Reconstruct level if delta modeling
                origin_val = float(s.loc[origin])
                if cfg.use_delta_modeling and is_stock(cfg.target):
                    y_hat = origin_val + delta_hat
                else:
                    y_hat = delta_hat

                # y_true from the actual series at target date
                if t not in s.index:
                    continue
                y_true_val = float(s.loc[t])

                preds.append(y_hat)
                ytrues.append(y_true_val)
                target_dates.append(t)
                origin_dates.append(origin)
                origin_values.append(origin_val)

            # PHASE 6: Build output DataFrame (identical schema)
            if target_dates:
                model_elapsed = time.time() - model_start_time
                print(f"[pipeline] Model {model_name}: 1 fit + {len(target_dates)} predictions "
                      f"in {model_elapsed:.2f}s (fold {fold_idx})")

                # Conformal PI: single radius for all predictions in this fold
                y_lo_list = [p - conformal_radius if np.isfinite(conformal_radius) else np.nan for p in preds]
                y_hi_list = [p + conformal_radius if np.isfinite(conformal_radius) else np.nan for p in preds]

                df_m = pd.DataFrame({
                    "date": target_dates,
                    "target_date": target_dates,
                    "origin_date": origin_dates,
                    "origin_value": origin_values,
                    "target": cfg.target,
                    "horizon": h,
                    "model": model_name,
                    "y_true": ytrues,
                    "y_pred": preds,
                    "y_lo": y_lo_list,
                    "y_hi": y_hi_list,
                    "split_id": f"{train_end.date()}→{test_start.date()}..{test_end.date()}",
                    "defn_variant": "ML-Uni" if cfg.variant == "uni" else "ML-Multi",
                })
                predictions_all.append(df_m)

    # Consolidate & evaluate
    if not predictions_all:
        raise RuntimeError("No predictions produced — check data span, horizon, or lags/windows.")

    pred_long = pd.concat(predictions_all, ignore_index=True).sort_values(["target", "horizon", "date", "model"])
    
    # ✅ GUARD B: Final alignment verification before saving
    # Verify all predictions have correct step-based alignment
    if "origin_date" in pred_long.columns and "target_date" in pred_long.columns and "horizon" in pred_long.columns:
        date_to_pos = {date: pos for pos, date in enumerate(s.index)}
        alignment_errors = []
        for idx, row in pred_long.iterrows():
            origin_date = pd.to_datetime(row["origin_date"])
            target_date = pd.to_datetime(row["target_date"])
            horizon_h = int(row["horizon"])  # Get horizon from row (may vary if multiple horizons)
            if origin_date in date_to_pos:
                pos_origin = date_to_pos[origin_date]
                if pos_origin + horizon_h < len(s.index):
                    expected_target = s.index[pos_origin + horizon_h]
                    if target_date != expected_target:
                        alignment_errors.append({
                            "idx": idx,
                            "origin": origin_date.date(),
                            "expected": expected_target.date(),
                            "actual": target_date.date(),
                            "horizon": horizon_h,
                        })
                else:
                    alignment_errors.append({
                        "idx": idx,
                        "origin": origin_date.date(),
                        "expected": "OUT_OF_BOUNDS",
                        "actual": target_date.date(),
                        "horizon": horizon_h,
                    })
        
        if alignment_errors:
            print(f"[ERROR] Found {len(alignment_errors)} alignment errors in final predictions:")
            for err in alignment_errors[:5]:
                print(f"  Row {err['idx']}: origin={err['origin']}, h={err['horizon']}, "
                      f"expected={err['expected']}, actual={err['actual']}")
            if len(alignment_errors) > 5:
                print(f"  ... and {len(alignment_errors) - 5} more")
            # In Thorough mode, this should fail
            is_thorough = (cfg.folds is not None and cfg.folds >= 5) or cfg.min_train_years >= 4
            if is_thorough:
                raise RuntimeError(f"Forecast alignment validation failed: {len(alignment_errors)} predictions misaligned")
    
    # ✅ Data sanity: Drop rows where y_true is NaN (outliers/zeros detected above)
    pred_long_clean = pred_long.dropna(subset=["y_true"]).copy()
    if len(pred_long_clean) < len(pred_long):
        n_dropped = len(pred_long) - len(pred_long_clean)
        print(f"[INFO] Dropped {n_dropped} predictions with NaN y_true (outliers/zeros)")
        pred_long = pred_long_clean
    
    pred_long.to_csv(out_root / "predictions_long.csv", index=False)

    # metrics & plots
    ops_series = None if (is_stock(cfg.target) or ops_daily is None) else ops_daily.loc[pred_long["date"].min(): pred_long["date"].max()]
    mdf = evaluate_block(pred_long, cfg.target, cfg.horizon, ops_series)
    mdf.to_csv(out_root / "metrics_long.csv", index=False)
    
    # ✅ Initialize glb early to avoid UnboundLocalError
    glb = pd.DataFrame()
    if len(mdf) > 0:
        glb = (mdf.groupby("model", as_index=False)["MAE"].mean()
                  .sort_values("MAE")
                  .assign(target=cfg.target, horizon=cfg.horizon, rank=lambda g: np.arange(1, len(g) + 1)))

        # ✅ FIX 11: Add persistence baseline row so users can compare directly
        if "origin_value" in pred_long.columns:
            _valid = pred_long.dropna(subset=["origin_value", "y_true"])
            if len(_valid) > 0:
                _mae_persist = float(np.mean(np.abs(_valid["y_true"].values - _valid["origin_value"].values)))
                persist_row = pd.DataFrame([{
                    "target": cfg.target, "horizon": cfg.horizon,
                    "model": "⚡ Persistence (baseline)", "MAE": _mae_persist, "rank": 0,
                }])
                glb = pd.concat([persist_row, glb], ignore_index=True)
                # Re-rank: persistence gets rank 0 (reference), models start at 1
                glb["rank"] = range(len(glb))
                print(f"[leaderboard] Persistence baseline MAE = {_mae_persist:.4f}")

        glb[["target", "horizon", "model", "MAE", "rank"]].to_csv(out_root / "leaderboard.csv", index=False)
    
    # ✅ Forecast integrity checks with HARD GATE - run ALWAYS when predictions exist
    try:
        from preprocessing.integrity import compute_integrity_report, leakage_sentinel
        
        # Compute integrity report — pick best *trained* model (skip baseline row)
        if glb is not None and len(glb) > 0:
            _trained_for_integrity = glb[~glb["model"].str.contains("baseline", case=False, na=False)]
            best_model = (_trained_for_integrity.iloc[0]["model"]
                          if len(_trained_for_integrity) > 0
                          else glb.iloc[0]["model"])
        else:
            # Fallback: use first model found in predictions
            models_in_preds = pred_long["model"].unique()
            best_model = models_in_preds[0] if len(models_in_preds) > 0 else None
        if best_model and len(pred_long) > 0:
            # ✅ FIX 3 + FIX 16: Use horizon-aware integrity checks.
            # Try both import paths: when cwd=backend/ (production via
            # subprocess) the module is at top-level; when imported as a
            # package it lives under backend.
            try:
                try:
                    from forecast_integrity import (
                        validate_alignment_step_based,
                        shift_diagnostic_horizon_aware,
                        compute_persistence_baseline,
                        compute_skill_score,
                    )
                except ImportError:
                    from backend.forecast_integrity import (
                        validate_alignment_step_based,
                        shift_diagnostic_horizon_aware,
                        compute_persistence_baseline,
                        compute_skill_score,
                    )
                
                # Alignment validation (step-based)
                alignment_check = validate_alignment_step_based(pred_long, s.index, cfg.horizon)
                
                # Shift diagnostic (horizon-aware)
                pred_model = pred_long[pred_long["model"] == best_model].copy()
                if len(pred_model) > 0:
                    shift_check = shift_diagnostic_horizon_aware(
                        pred_model["y_true"].values,
                        pred_model["y_pred"].values,
                        cfg.horizon,
                    )
                else:
                    shift_check = {}
                
                # Persistence baseline
                persistence_baseline = compute_persistence_baseline(pred_model)
                
                # Skill score
                mae_model = float(np.mean(np.abs(pred_model["y_true"] - pred_model["y_pred"])))
                mae_persistence = persistence_baseline.get("mae_persistence", np.nan)
                skill_pct = compute_skill_score(mae_model, mae_persistence)
                
                # Build integrity report with new horizon-aware checks
                integrity_report = {
                    "alignment_ok": alignment_check.get("alignment_ok", False),
                    "n_misaligned": alignment_check.get("n_misaligned", 0),
                    "n_total": alignment_check.get("n_total", 0),
                    "validation_method": "step_based",
                    "best_shift": shift_check.get("best_shift", 0),
                    "mae_shift0": shift_check.get("mae_shift0", np.nan),
                    "mae_shift_minus_h": shift_check.get("mae_shift_minus_h", np.nan),
                    "mae_shift_minus_h_plus_1": shift_check.get("mae_shift_minus_h_plus_1", np.nan),
                    "improvement_pct": shift_check.get("improvement_pct", 0.0),
                    "is_lag0_issue": shift_check.get("is_lag0_issue", False),
                    "is_persistence_like": shift_check.get("is_persistence_like", False),
                    "shift_interpretation": shift_check.get("interpretation", "unknown"),
                    "mae_model": mae_model,
                    "mae_persistence": mae_persistence,
                    "skill_pct": skill_pct,
                    "run_status": "SUCCESS",
                }
                
                # Also compute legacy integrity report for backward compatibility
                legacy_report = compute_integrity_report(pred_long, s, cfg.horizon, best_model, cfg.cadence, date_index=s.index)
                integrity_report.update(legacy_report)  # Merge legacy fields
            except ImportError:
                # Fallback to legacy integrity checks if new module not available
                print("[WARN] forecast_integrity module not found, using legacy checks")
                integrity_report = compute_integrity_report(pred_long, s, cfg.horizon, best_model, cfg.cadence, date_index=s.index)
            
            # Perform leakage check on a sample fold (use last fold for efficiency)
            if len(folds) > 0:
                # Get a sample of training data from the last fold
                last_fold = folds[-1]
                train_end, test_start, test_end = last_fold
                
                # Build training features for leakage test
                train_mask = s.index <= train_end
                s_train_sample = s.loc[train_mask]
                if len(s_train_sample) > 20:
                    f_train_sample = lag_window_features(s_train_sample, lags, wins).join(cal.loc[s_train_sample.index])
                    if exog_cols and df_multi is not None:
                        f_train_sample = f_train_sample.join(df_multi.loc[s_train_sample.index, exog_cols])
                    # ✅ FIX 3: Use index-position based target construction (not calendar-day shift)
                    y_train_sample = pd.Series(index=s_train_sample.index, dtype=float)
                    s_sample_array = s_train_sample.values
                    for i, t_idx in enumerate(s_train_sample.index):
                        target_pos = i + h
                        if target_pos < len(s_train_sample):
                            y_train_sample.loc[t_idx] = s_sample_array[target_pos]
                        else:
                            y_train_sample.loc[t_idx] = np.nan
                    y_train_sample = y_train_sample.dropna()
                    f_train_sample = f_train_sample.loc[y_train_sample.index]
                    
                    # Get test data
                    test_df = pred_long[
                        (pred_long["model"] == best_model) & 
                        (pd.to_datetime(pred_long["date"]) >= test_start) &
                        (pd.to_datetime(pred_long["date"]) <= test_end)
                    ]
                    
                    if len(test_df) > 5 and len(f_train_sample) > 10:
                        # Build test features (simplified - would need actual origin dates)
                        # For leakage test, we'll use a subset
                        X_train_leak = f_train_sample.iloc[:min(500, len(f_train_sample))].fillna(0.0)
                        y_train_leak = y_train_sample.iloc[:min(500, len(y_train_sample))]
                        X_test_leak = f_train_sample.iloc[-min(100, len(test_df)):].fillna(0.0)
                        y_test_leak = y_train_sample.iloc[-min(100, len(test_df)):]
                        
                        if len(X_train_leak) > 10 and len(X_test_leak) > 5:
                            leakage_check = leakage_sentinel(
                                X_train_leak, y_train_leak, X_test_leak, y_test_leak, cfg.horizon
                            )
                            integrity_report["mae_shuffled_target"] = leakage_check["mae_shuffled_target"]
                            integrity_report["leakage_warning"] = leakage_check["leakage_warning"]
                            integrity_report["shuffled_to_normal_ratio"] = leakage_check.get("shuffled_to_normal_ratio", np.nan)
        else:
            # No model found - create minimal report
            integrity_report = {
                "error": "No model found for integrity check" if not best_model else "No predictions available",
                "n_predictions": len(pred_long),
                "run_status": "ERROR"
            }
        
        # ✅ Always save integrity report
        with open(out_root / "artifacts" / "integrity_report.json", "w", encoding="utf-8") as f:
            json.dump(integrity_report, f, indent=2, default=str)
        
        # ✅ FIX 9: Print acceptance criteria summary
        print("\n" + "="*80)
        print("FORECAST INTEGRITY REPORT SUMMARY")
        print("="*80)
        print(f"Alignment validation: {'PASS' if integrity_report.get('alignment_ok', False) else 'FAIL'}")
        print(f"  - Misaligned predictions: {integrity_report.get('n_misaligned', 0)}/{integrity_report.get('n_total', 0)}")
        print(f"  - Validation method: {integrity_report.get('validation_method', 'unknown')}")
        
        leakage_warning = integrity_report.get("leakage_warning", False)
        print(f"Leakage checks: {'PASS' if not leakage_warning else 'WARN'}")
        if leakage_warning:
            print(f"  - Shuffled target ratio: {integrity_report.get('shuffled_to_normal_ratio', np.nan):.2f}")
        
        shift_check = {
            "best_shift": integrity_report.get("best_shift", 0),
            "improvement_pct": integrity_report.get("improvement_pct", integrity_report.get("improvement_pct_vs_shift0", 0.0)),
            "is_lag0_issue": integrity_report.get("is_lag0_issue", False),
            "interpretation": integrity_report.get("shift_interpretation", "unknown"),
        }
        print(f"Shift detection: best_shift={shift_check['best_shift']}, improvement={shift_check['improvement_pct']:.1f}%")
        print(f"  - Interpretation: {shift_check['interpretation']}")
        if shift_check['is_lag0_issue']:
            print(f"  - ⚠️  MISSING LAG_0: best_shift ≈ -(h+1) pattern suggests missing lag_0 feature")
        elif shift_check['best_shift'] == 0:
            print(f"  - ✓ Best alignment at shift=0 (correct)")
        
        skill_pct = integrity_report.get("skill_pct", np.nan)
        mae_model = integrity_report.get("mae_model", np.nan)
        mae_persistence = integrity_report.get("mae_persistence", np.nan)
        print(f"Baseline comparison:")
        print(f"  - Model MAE: {mae_model:.2f}")
        print(f"  - Persistence MAE: {mae_persistence:.2f}")
        print(f"  - Skill: {skill_pct:.2f}% {'(PASS)' if skill_pct >= 5.0 else '(FAILED_QUALITY)'}")
        
        run_status = integrity_report.get("run_status", "SUCCESS")
        print(f"Run status: {run_status}")
        print("="*80 + "\n")
        
        # ✅ HARD INTEGRITY GATE: Fail if alignment error or lag_warning in Thorough mode (only if report is valid)
        if best_model and len(pred_long) > 0:
            # Detect "Thorough" mode: typically means folds >= 5 or min_train_years >= 4
            is_thorough = (cfg.folds is None or cfg.folds >= 5) or (cfg.min_train_years >= 4)
            
            # Check alignment
            if not integrity_report.get("alignment_ok", True):
                n_misaligned = integrity_report.get("n_misaligned", 0)
                error_msg = (
                    f"Forecast alignment validation failed: {n_misaligned} predictions have "
                    f"origin_date + h != target_date. This indicates a timestamping bug."
                )
                if is_thorough:
                    print(f"[FAIL] {error_msg}")
                    print(f"[FAIL] Run aborted due to alignment failure in Thorough mode.")
                    raise RuntimeError(error_msg)
                else:
                    print(f"[WARN] {error_msg}")
            
            # ✅ FIX B: Check for critical timestamping bug (near-perfect improvement after shifting)
            # Only FAIL if it's a true timestamping bug, otherwise WARN (autocorrelation is normal)
            is_critical_bug = integrity_report.get("is_critical_timestamping_bug", False)
            lag_warning = integrity_report.get("lag_warning", False)
            best_shift = integrity_report.get("best_shift", 0)
            improvement = integrity_report.get("improvement_pct", 0.0)
            
            if is_critical_bug:
                # Critical bug: near-perfect improvement after shifting (>95% AND mae_best < 0.2*mae0)
                error_msg = (
                    f"CRITICAL timestamping bug detected: predictions appear shifted by {best_shift} "
                    f"with near-perfect improvement ({improvement:.1f}%). "
                    f"This indicates a true timestamping/alignment bug, not just autocorrelation."
                )
                if is_thorough:
                    print(f"[FAIL] {error_msg}")
                    print(f"[FAIL] Run aborted due to critical timestamping bug in Thorough mode.")
                    raise RuntimeError(error_msg)
                else:
                    print(f"[WARN] {error_msg}")
            elif lag_warning:
                # Normal autocorrelation/persistence effect - just WARN, don't fail
                is_lag0_issue = integrity_report.get("is_lag0_issue", False)
                shift_interpretation = integrity_report.get("shift_interpretation", "")
                if is_lag0_issue:
                    print(f"[WARN] ⚠️  MISSING LAG_0 DETECTED: best_shift={best_shift} ≈ -(h+1)={-(cfg.horizon+1)}. "
                          f"This indicates model is effectively using lag_1 instead of lag_0. "
                          f"Ensure lag_0 is included in features (current lags: {lags}).")
                elif integrity_report.get("is_persistence_like", False):
                    print(f"[WARN] Persistence-like behavior: best_shift={best_shift} ≈ -h={-cfg.horizon}. "
                          f"Compare model performance vs persistence baseline.")
                else:
                    print(f"[WARN] Predictions show shift improvement (best_shift={best_shift}, improvement={improvement:.1f}%). "
                          f"Interpretation: {shift_interpretation}")
            
            # ✅ QUALITY GATE: Flag if skill < 2% but don't abort (outputs are still valid)
            skill_pct = integrity_report.get("skill_pct", np.nan)
            if not np.isnan(skill_pct):
                _QUALITY_GATE_SKILL_PCT = 5.0   # Minimum acceptable skill %
                if skill_pct < _QUALITY_GATE_SKILL_PCT:
                    error_msg = (
                        f"Model does not beat persistence baseline at horizon {cfg.horizon}: "
                        f"skill={skill_pct:.2f}% (threshold={_QUALITY_GATE_SKILL_PCT}%). "
                        f"Outputs are real but not useful yet. "
                        f"MAE_model={integrity_report.get('mae_model', np.nan):.0f} vs "
                        f"MAE_persistence={integrity_report.get('mae_persistence', np.nan):.0f}."
                    )
                    print(f"[WARN] {error_msg}")
                    print(f"[WARN] Run status: FAILED_QUALITY (outputs still written)")
                    # ✅ FIX 3: Store status but don't raise - outputs are still valid
                    integrity_report["run_status"] = "FAILED_QUALITY"
                    integrity_report["quality_gate_failed"] = True
                    # Save updated report (but don't abort)
                    with open(out_root / "artifacts" / "integrity_report.json", "w", encoding="utf-8") as f:
                        json.dump(integrity_report, f, indent=2, default=str)
                else:
                    integrity_report["run_status"] = "SUCCESS"
                    integrity_report["quality_gate_failed"] = False
                    print(f"[OK] Quality gate passed: skill={skill_pct:.2f}% >= {_QUALITY_GATE_SKILL_PCT}%")
            else:
                integrity_report["run_status"] = "SUCCESS" if not integrity_report.get("lag_warning", False) else "WARNING"
                integrity_report["quality_gate_failed"] = False
            
            # Save final report with run_status
            with open(out_root / "artifacts" / "integrity_report.json", "w", encoding="utf-8") as f:
                json.dump(integrity_report, f, indent=2, default=str)
            
            if integrity_report.get("leakage_warning"):
                print(f"[WARN] Forecast integrity: leakage_warning=true (shuffled_ratio={integrity_report.get('shuffled_to_normal_ratio', np.nan):.2f})")
            
            # ✅ Log baseline comparisons with detailed reporting
            mae_model = integrity_report.get("mae_model", np.nan)
            mae_persistence = integrity_report.get("mae_persistence", np.nan)
            skill_pct = integrity_report.get("skill_pct", np.nan)
            
            if not np.isnan(mae_persistence) and not np.isnan(mae_model):
                print(f"[BASELINE] Baseline(persist) MAE={mae_persistence:.2f}, Model MAE={mae_model:.2f}, Skill={skill_pct:.2f}%")
                if skill_pct < 0:
                    print(f"[WARN] Model underperforms persistence baseline: skill={skill_pct:.2f}% (model is worse than naive)")
                elif skill_pct < 5.0:
                    print(f"[WARN] Model skill is low: skill={skill_pct:.2f}% (threshold=5.0% for Thorough mode)")
                else:
                    print(f"[OK] Model beats persistence baseline: skill={skill_pct:.2f}%")
            
            # ✅ Log offset/lag diagnostic
            best_shift = integrity_report.get("best_shift", 0)
            mae_shift0 = integrity_report.get("mae_shift0", np.nan)
            best_mae = integrity_report.get("mae_best", np.nan)
            improvement_ratio = integrity_report.get("improvement_ratio", np.nan)
            mae_shift_minus_h = integrity_report.get("mae_shift_minus_h", np.nan)
            
            if not np.isnan(mae_shift0) and not np.isnan(best_mae):
                if best_shift != 0 and improvement_ratio < 0.85:
                    print(f"[WARN] Predictions appear shifted/lagged (best_shift={best_shift}); "
                          f"possible target misalignment or persistence-like behavior. "
                          f"MAE(shift={best_shift})={best_mae:.2f} vs MAE(shift=0)={mae_shift0:.2f} "
                          f"(improvement_ratio={improvement_ratio:.3f})")
                elif best_shift != 0:
                    print(f"[INFO] Best alignment at shift={best_shift} (MAE={best_mae:.2f} vs shift=0 MAE={mae_shift0:.2f}), "
                          f"but improvement is minor (ratio={improvement_ratio:.3f})")
                else:
                    print(f"[OK] Shift diagnostic: best alignment at shift=0 (correct)")
                
                # ✅ GUARD B: Check if model ≈ naive baseline (persistence)
                if not np.isnan(mae_shift_minus_h):
                    naive_mae = mae_shift_minus_h  # MAE at shift=-h (persistence-like)
                    if abs(mae_model - naive_mae) / max(mae_model, naive_mae, 1.0) < 0.1:
                        print(f"[WARN] Model performance ≈ naive baseline (persistence): "
                              f"Model MAE={mae_model:.2f} vs Naive(shift=-h) MAE={naive_mae:.2f}. "
                              f"Model may be learning mostly persistence.")
            
            # Log alignment status
            if integrity_report.get("alignment_ok", True):
                print(f"[OK] Alignment validation passed: all {integrity_report.get('n_predictions', 0)} predictions correctly aligned")
            else:
                print(f"[WARN] Alignment validation: {integrity_report.get('n_misaligned', 0)} misaligned predictions")
            
            print(f"[OK] Forecast integrity report saved: {out_root / 'artifacts' / 'integrity_report.json'}")
        else:
            # No valid report - already saved error report above
            print(f"[WARN] Integrity check skipped: {integrity_report.get('error', 'Unknown error')}")
    except RuntimeError:
        # Re-raise integrity failures
        raise
    except Exception as e:
        import traceback
        print(f"[WARN] Forecast integrity checks failed: {e}")
        traceback.print_exc()
        # Still save a partial report
        try:
            error_report = {
                "error": str(e),
                "error_type": type(e).__name__,
                "n_predictions": len(pred_long) if 'pred_long' in locals() else 0,
                "run_status": "ERROR"
            }
            with open(out_root / "artifacts" / "integrity_report.json", "w", encoding="utf-8") as f:
                json.dump(error_report, f, indent=2, default=str)
        except Exception:
            pass

    # glb already initialized above — pick best *trained* model (skip baseline row)
    best_model = None
    if glb is not None and len(glb) > 0:
        _trained = glb[~glb["model"].str.contains("baseline", case=False, na=False)]
        best_model = _trained.iloc[0]["model"] if len(_trained) > 0 else glb.iloc[0]["model"]
    try:
        plot_overlay_all(pred_long, cfg.target, cfg.horizon, ops_series, out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_overlay_all.png")
        if best_model:
            plot_overlay_top(pred_long, cfg.target, cfg.horizon, best_model, ops_series,
                             out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_overlay_top.png")
            plot_monthly_bars(pred_long[pred_long["model"] == best_model], cfg.target, cfg.horizon, ops_series,
                              out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_monthly_bars_top_vs_ops.png")
            # Use glb (which includes persistence baseline row) for leaderboard chart
            _lb = glb if (glb is not None and len(glb) > 0) else mdf
            plot_leaderboard_bar(_lb, cfg.target, cfg.horizon, out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_leaderboard_mae.png")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    # ✅ Feature importance extraction (for tree-based models)
    try:
        fi_records = []
        for model_name, estimator in all_models.items():
            importances = None
            feature_names = None

            # Try to get feature names from last fit
            if hasattr(estimator, 'feature_names_in_'):
                feature_names = list(estimator.feature_names_in_)
            elif hasattr(estimator, 'named_steps'):
                last_step = list(estimator.named_steps.values())[-1]
                if hasattr(last_step, 'feature_names_in_'):
                    feature_names = list(last_step.feature_names_in_)

            # Get importances
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            elif hasattr(estimator, 'named_steps'):
                last_step = list(estimator.named_steps.values())[-1]
                if hasattr(last_step, 'feature_importances_'):
                    importances = last_step.feature_importances_
                elif hasattr(last_step, 'coef_'):
                    importances = np.abs(last_step.coef_)

            if importances is not None and len(importances) > 0:
                if feature_names is None:
                    feature_names = [f"feat_{i}" for i in range(len(importances))]
                # Ensure same length
                n = min(len(feature_names), len(importances))
                for i in range(n):
                    fi_records.append({
                        "model": model_name,
                        "feature": feature_names[i],
                        "importance": float(importances[i]),
                    })

        if fi_records:
            fi_df = pd.DataFrame(fi_records)
            fi_df.to_csv(out_root / "artifacts" / "feature_importance.csv", index=False)
            print(f"[OK] Feature importance saved: {len(fi_records)} entries across {fi_df['model'].nunique()} models")
    except Exception as e:
        print(f"[WARN] Feature importance extraction failed: {e}")

    # Artifacts
    with open(out_root / "artifacts" / "RUN.md", "w", encoding="utf-8") as f:
        f.write(f"# B · Machine Learning run\n\n")
        f.write(f"- timestamp: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- data_path: {cfg.data_path}\n")
        f.write(f"- target: {cfg.target}\n")
        f.write(f"- cadence: {cfg.cadence}\n")
        f.write(f"- horizon: {cfg.horizon}\n")
        f.write(f"- variant: {cfg.variant}\n")
        f.write(f"- model_filter: {cfg.model_filter or '(all)'}\n")
        f.write(f"- folds: {cfg.folds}\n")
        f.write(f"- min_train_years: {cfg.min_train_years}\n")
        f.write(f"- lags/windows: {choose_recipe(cfg)}\n")

    with open(out_root / "artifacts" / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"[OK] Full outputs in: {out_root}")
    return str(out_root)
