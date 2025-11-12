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

    def __post_init__(self):
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
    feats = pd.DataFrame(index=s.index)
    for L in lags:
        feats[f"lag_{L}"] = s.shift(L)
    for W in windows:
        feats[f"rmean_{W}"] = s.rolling(W, min_periods=1).mean().shift(1)
    return feats


def choose_recipe(cfg: ConfigBML) -> Tuple[List[int], List[int]]:
    cad = cfg.cadence.lower()
    if cad == "daily":
        return cfg.lags_daily, cfg.windows_daily
    if cad == "weekly":
        return cfg.lags_weekly, cfg.windows_weekly
    return cfg.lags_monthly, cfg.windows_monthly


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
                           ("est", Lasso(random_state=0))]),
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
        rows.append({
            "target": target,
            "horizon": horizon,
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smp,
            "MAPE": mape,
            "R2": r2,
            "PI_coverage@90": np.nan,
            "PI_width@90": np.nan,
            "Monthly_TOL10_Accuracy": tol10,
            "MAE_skill_vs_Ops": np.nan,
        })
    return pd.DataFrame(rows)


def plot_overlay_all(df_slice: pd.DataFrame, target: str, h: int, ops_series: Optional[pd.Series], out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for model, g in df_slice.groupby("model"):
        ax.plot(g["date"], g["y_pred"], label=model, alpha=0.7)
    ax.plot(df_slice["date"], df_slice["y_true"], label="Actual", linewidth=2, color="black")
    if ops_series is not None and (not is_stock(target)):
        ops_aligned = ops_series.reindex(pd.to_datetime(df_slice["date"]))
        ax.plot(df_slice["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | h=D+{h} | ALL vs Ops (ML)")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_overlay_top(df_slice: pd.DataFrame, target: str, h: int, best_model: str,
                     ops_series: Optional[pd.Series], out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    g = df_slice[df_slice["model"] == best_model].sort_values("date")
    ax.plot(g["date"], g["y_pred"], label=best_model, linewidth=2)
    ax.plot(g["date"], g["y_true"], label="Actual", linewidth=2)
    if ops_series is not None and (not is_stock(target)):
        ops_aligned = ops_series.reindex(pd.to_datetime(g["date"]))
        ax.plot(g["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | h=D+{h} | TOP={best_model} vs Ops (ML)")
    ax.legend(loc="best"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_leaderboard_bar(mdf: pd.DataFrame, target: str, h: int, out_png: Path):
    g = mdf.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(g["model"], g["MAE"]); ax.invert_yaxis()
    ax.set_title(f"{target} | h=D+{h} | Leaderboard by MAE (ML)")
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
    folds = build_yearly_folds(s.index, cfg.min_train_years, cfg.folds)
    lags, wins = choose_recipe(cfg)

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
            df_multi[c] = df_multi[c].reindex(s.index).fillna(method="ffill").fillna(0.0)

    # Iterate horizons & folds (here a single horizon from env)
    h = int(cfg.horizon)
    for (train_end, test_start, test_end) in folds:
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

        # For each model
        for model_name, estimator in all_models.items():
            preds, ytrues, dates = [], [], []

            for pos_t in positions:
                t = s.index[pos_t]
                pos_origin = pos_t - h
                origin = s.index[pos_origin]

                # enforce fold boundary: training must be ≤ train_end
                s_train = s[:origin]
                s_train = s_train[s_train.index <= train_end]
                if s_train.empty:
                    continue

                # features on train
                f_train = lag_window_features(s_train, lags, wins).join(cal.loc[s_train.index])
                if exog_cols:
                    f_train = f_train.join(df_multi.loc[s_train.index, exog_cols])
                y_train = s_train

                # ✅ features at the *origin* row for forecasting h steps ahead
                feat_all = lag_window_features(s, lags, wins).join(cal)
                x_pred = feat_all.loc[[origin]].fillna(0.0)
                if exog_cols:
                    x_pred = x_pred.join(df_multi.loc[[origin], exog_cols])

                # train dropna + fit
                train = pd.concat([f_train, y_train.rename("y")], axis=1).dropna()
                if len(train) < 10:
                    continue
                X_tr, y_tr = train.drop(columns=["y"]), train["y"]

                estimator.fit(X_tr, y_tr)
                y_hat = float(estimator.predict(x_pred)[0])

                preds.append(y_hat)
                ytrues.append(float(s.loc[t]))
                dates.append(t)

            if dates:
                df_m = pd.DataFrame({
                    "date": dates,
                    "target": cfg.target,
                    "horizon": h,
                    "model": model_name,
                    "y_true": ytrues,
                    "y_pred": preds,
                    "y_lo": [np.nan] * len(dates),
                    "y_hi": [np.nan] * len(dates),
                    "split_id": f"{train_end.date()}→{test_start.date()}..{test_end.date()}",
                    "defn_variant": "ML-Uni" if cfg.variant == "uni" else "ML-Multi",
                })
                predictions_all.append(df_m)

    # Consolidate & evaluate
    if not predictions_all:
        raise RuntimeError("No predictions produced — check data span, horizon, or lags/windows.")

    pred_long = pd.concat(predictions_all, ignore_index=True).sort_values(["target", "horizon", "date", "model"])
    pred_long.to_csv(out_root / "predictions_long.csv", index=False)

    # metrics & plots
    ops_series = None if (is_stock(cfg.target) or ops_daily is None) else ops_daily.loc[pred_long["date"].min(): pred_long["date"].max()]
    mdf = evaluate_block(pred_long, cfg.target, cfg.horizon, ops_series)
    mdf.to_csv(out_root / "metrics_long.csv", index=False)

    glb = (mdf.groupby("model", as_index=False)["MAE"].mean()
              .sort_values("MAE")
              .assign(target=cfg.target, horizon=cfg.horizon, rank=lambda g: np.arange(1, len(g) + 1)))
    glb[["target", "horizon", "model", "MAE", "rank"]].to_csv(out_root / "leaderboard.csv", index=False)

    best_model = glb.iloc[0]["model"] if len(glb) else None
    try:
        plot_overlay_all(pred_long, cfg.target, cfg.horizon, ops_series, out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_overlay_all.png")
        if best_model:
            plot_overlay_top(pred_long, cfg.target, cfg.horizon, best_model, ops_series,
                             out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_overlay_top.png")
            plot_monthly_bars(pred_long[pred_long["model"] == best_model], cfg.target, cfg.horizon, ops_series,
                              out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_monthly_bars_top_vs_ops.png")
            plot_leaderboard_bar(mdf, cfg.target, cfg.horizon, out_root / f"{cfg.target.replace(' ','_')}_h{cfg.horizon}_leaderboard_mae.png")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

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
