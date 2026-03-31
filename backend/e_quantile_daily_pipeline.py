# e_quantile_daily_pipeline.py
# Georgia Treasury — Quantile models (Daily cadence)
# Contract: same as A_STAT/B_ML — run_pipeline(CONFIG) and write standard outputs.

from __future__ import annotations
import os, json, time, math, pathlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ---------- configuration ----------

@dataclass
class Config:
    target: str
    cadence: str              # "Daily" (supported now). Monthly can be added similarly if needed.
    horizon: int
    data_path: str
    date_col: str = "date"
    folds: Optional[int] = 3   # None = use ALL possible folds (thorough mode)
    min_train_years: int = 4
    model_filter: Optional[str] = None   # "GBQuantile", "ResidualRF" | None => all
    quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)
    lags_daily: Tuple[int, ...] = (1, 5, 20)
    windows_daily: Tuple[int, ...] = (5, 20)
    exog_top_k: Optional[int] = None     # multivariate only: top-K features by abs corr to target
    out_root: str = "outputs"
    demo_clip_months: Optional[int] = None  # None => full data; int => keep last N months
    variant: str = "univariate"          # "univariate" | "multivariate"

# ---------- tiny utils ----------

def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    # q \in (0,1) ; lower is better
    diff = y_true - y_pred
    return float(np.maximum(q * diff, (q - 1) * diff).mean())

def _time_folds(n: int, horizon: int, folds: int, min_train: int) -> List[Tuple[int, int]]:
    """
    Expanding-window time-series cross-validation.

    Returns list of (train_end_index_exclusive, test_end_index_exclusive).
    At each fold: train = [0 : train_end), test = [train_end : test_end)
    where len(test) == horizon.

    Training always starts at index 0, so later folds always have at least
    as much training data as earlier ones (expanding window).  Test blocks
    are non-overlapping and placed from the end of the series backward.
    """
    indices = []
    last_test_end = n
    for _ in range(folds, 0, -1):
        test_end = last_test_end
        test_start = test_end - horizon
        if test_start < 0:
            break
        train_end = test_start          # train = [0 : train_end)
        # min_train is in years; convert to approximate rows (252 biz days/yr).
        # Also enforce at least max(horizon, 30) so very short horizons don't
        # create trivially small training sets.
        min_train_rows = max(min_train * 252, horizon, 30)
        if train_end <= min_train_rows:
            break
        indices.append((train_end, test_end))
        last_test_end = test_start
    indices.reverse()   # earliest fold first
    return indices

def _calendar_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "dow": idx.dayofweek,           # 0..6
        "dom": idx.day,                 # 1..31
        "week": idx.isocalendar().week.astype(int),
        "month": idx.month,             # 1..12
        "year": idx.year
    }, index=idx)

def _build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build feature frame and **h-step-ahead** target.

    Returns
    -------
    X : feature DataFrame (features at time t, backward-looking only)
    y_target : target Series — value at position t + horizon  (step-based)
    origin_dates : Series of origin dates aligned with X
    origin_values : Series of y-values at origin aligned with X

    ✅ FIX QUANT-1: The target is now y(t + h) instead of y(t).
    This makes the quantile pipeline a genuine h-step-ahead forecaster,
    consistent with ML pipeline semantics.
    """
    y = df[cfg.target].astype(float).copy()
    X = pd.DataFrame(index=df.index)

    # Target-derived features (all backward-looking: safe)
    for l in cfg.lags_daily:
        X[f"y_lag_{l}"] = y.shift(l)
    for w in cfg.windows_daily:
        X[f"y_roll_mean_{w}"] = y.rolling(w, min_periods=1).mean().shift(1)
        X[f"y_roll_std_{w}"] = y.rolling(w, min_periods=1).std(ddof=0).shift(1)

    # Calendar features
    X = pd.concat([X, _calendar_feats(df.index)], axis=1)

    # Multivariate exogenous
    if cfg.variant == "multivariate":
        exog_cols = [c for c in df.columns if c not in (cfg.target,) and c != cfg.date_col]
        exog = df[exog_cols].copy()
        exog = exog.ffill().bfill().shift(1)
        if cfg.exog_top_k is not None and cfg.exog_top_k > 0:
            corr = exog.join(y).corr(numeric_only=True)[cfg.target].drop(cfg.target, errors="ignore").abs()
            keep = corr.sort_values(ascending=False).head(cfg.exog_top_k).index.tolist()
            exog = exog[keep]
        X = pd.concat([X, exog], axis=1)

    # ✅ FIX QUANT-1: Construct h-step-ahead target using step-based indexing.
    # y_target[i] = y[i + horizon]  (positional offset, not calendar-day).
    h = cfg.horizon
    y_vals = y.values
    y_target = pd.Series(np.nan, index=y.index, dtype=float)
    for i in range(len(y) - h):
        y_target.iloc[i] = y_vals[i + h]

    # origin_dates = feature dates, origin_values = y at feature dates
    origin_dates = pd.Series(y.index, index=y.index)
    origin_values = y.copy()

    # Align: drop rows where features or target are NaN
    both = pd.concat([X, y_target.rename("__target__"),
                       origin_values.rename("__origin_val__")], axis=1).dropna()
    X_out = both.drop(columns=["__target__", "__origin_val__"])
    y_out = both["__target__"]
    ov_out = both["__origin_val__"]
    od_out = pd.Series(both.index, index=both.index)
    return X_out, y_out, od_out, ov_out

def _save_csv(df: pd.DataFrame, out_root: str, name: str) -> str:
    p = os.path.join(out_root, name)
    df.to_csv(p, index=False)
    return p

def _save_run_json(cfg: Config, out_root: str, elapsed: float) -> None:
    run = asdict(cfg)
    run["elapsed_sec"] = round(elapsed, 3)
    with open(os.path.join(out_root, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

def _plot_quantiles(df_fold: pd.DataFrame, out_dir: str, title: str) -> None:
    # Lightweight matplotlib plot: actual + P50 + ribbon P10–P90
    import matplotlib.pyplot as plt

    df_fold = df_fold.sort_values("date")
    plt.figure()
    plt.plot(df_fold["date"], df_fold["y_true"], label="Actual")
    if "yhat_p50" in df_fold:
        plt.plot(df_fold["date"], df_fold["yhat_p50"], label="P50")
    if "yhat_p10" in df_fold and "yhat_p90" in df_fold:
        plt.fill_between(df_fold["date"], df_fold["yhat_p10"], df_fold["yhat_p90"], alpha=0.2, label="P10–P90")
    plt.title(title)
    plt.legend()
    _ensure_dir(out_dir)
    fn = os.path.join(out_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(fn, bbox_inches="tight")
    plt.close()

# ---------- models ----------

def _fit_gb_quantile(X_tr, y_tr, X_te, q: float) -> np.ndarray:
    # Gradient Boosting quantile (pinball loss). Separate model per quantile.
    model = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def _fit_residual_rf_quantiles(X_tr, y_tr, X_te, quantiles: Tuple[float, ...]) -> Dict[float, np.ndarray]:
    """
    Generic 'residual quantile' wrapper:
    1) Fit a point model (GBR squared_error),
    2) Estimate residual quantiles on train via CV-like split (simple and fast),
    3) Shift the point prediction by residual quantiles.
    This is distribution-free and avoids extra deps; acts as a baseline.
    """
    from sklearn.ensemble import RandomForestRegressor

    # point model
    rf = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, max_depth=None
    )
    rf.fit(X_tr, y_tr)
    yhat_tr = rf.predict(X_tr)
    resid = y_tr - yhat_tr

    preds = {}
    for q in quantiles:
        shift = np.quantile(resid, q)
        preds[q] = rf.predict(X_te) + shift
    return preds

# ---------- main pipeline ----------

def run_pipeline(CONFIG: Config) -> None:
    t0 = time.time()
    print("[runner] START pipeline for "
          f"target='{CONFIG.target}' cadence={CONFIG.cadence} horizon={CONFIG.horizon} ({CONFIG.variant})")

    # Load
    df = pd.read_csv(CONFIG.data_path)
    if CONFIG.date_col not in df.columns:
        raise ValueError(f"date column '{CONFIG.date_col}' not found.")
    df[CONFIG.date_col] = pd.to_datetime(df[CONFIG.date_col])
    df = df.sort_values(CONFIG.date_col).set_index(CONFIG.date_col)

    if CONFIG.demo_clip_months:
        last = df.index.max()
        clip_start = last - pd.DateOffset(months=int(CONFIG.demo_clip_months))
        df = df.loc[df.index >= clip_start]

    # Guardrails
    if CONFIG.cadence != "Daily":
        print("[runner] WARNING: this pipeline file currently supports Daily cadence. "
              "You can duplicate it for Monthly with the same contract.")
    if CONFIG.target not in df.columns:
        raise ValueError(f"target column '{CONFIG.target}' not found.")

    # Features — now returns h-step-ahead targets + origin metadata
    X_all, y_all, od_all, ov_all = _build_features(df, CONFIG)
    n = len(y_all)
    if n < CONFIG.horizon + 50:
        print("[runner] WARNING: very short series after feature alignment.")

    # CV folds
    folds = _time_folds(n, CONFIG.horizon, CONFIG.folds, CONFIG.min_train_years)
    if not folds:
        raise ValueError("Unable to create CV folds — series too short for requested horizon/folds.")

    # Model registry (you can add more later without touching the bridge/UI)
    registry = {
        "GBQuantile": "GradientBoosting (quantile loss)",
        "ResidualRF": "RandomForest + residual quantiles (baseline)"
    }
    chosen = list(registry.keys()) if not CONFIG.model_filter or CONFIG.model_filter.strip() == "" else [CONFIG.model_filter]
    chosen = [m for m in chosen if m in registry]

    out_root = CONFIG.out_root
    _ensure_dir(out_root)
    _ensure_dir(os.path.join(out_root, "plots"))

    preds_rows = []
    metrics_rows = []
    leaderboard_rows = []

    for model_name in chosen:
        print(f"[runner]  Model={model_name}")
        fold_ix = 0
        pinballs: Dict[float, List[float]] = {q: [] for q in CONFIG.quantiles}
        coverages: List[float] = []

        for (tr_end, te_end) in folds:
            fold_ix += 1
            X_tr, y_tr = X_all.iloc[:tr_end], y_all.iloc[:tr_end]
            X_te, y_te = X_all.iloc[tr_end:te_end], y_all.iloc[tr_end:te_end]
            od_te = od_all.iloc[tr_end:te_end]  # origin dates for test
            ov_te = ov_all.iloc[tr_end:te_end]  # origin values for test
            dates_te = X_te.index  # these are origin dates; target dates are h steps forward

            # Fit/predict per model
            if model_name == "GBQuantile":
                q_preds = {}
                for q in CONFIG.quantiles:
                    q_preds[q] = _fit_gb_quantile(X_tr, y_tr, X_te, q)
            elif model_name == "ResidualRF":
                q_preds = _fit_residual_rf_quantiles(X_tr, y_tr, X_te, CONFIG.quantiles)
            else:
                raise ValueError(f"Unknown model '{model_name}'")

            # ✅ FIX QUANT-2: Compute target_dates from origin + h steps.
            # The origin dates (dates_te / od_te) are the feature dates; the
            # actual target is h positions forward in the original series.
            y_series = df[CONFIG.target].astype(float)
            y_idx = y_series.index
            date_to_pos = {d: i for i, d in enumerate(y_idx)}
            target_dates_list = []
            for orig_d in od_te.values:
                orig_d_ts = pd.Timestamp(orig_d)
                pos = date_to_pos.get(orig_d_ts)
                if pos is not None and pos + CONFIG.horizon < len(y_idx):
                    target_dates_list.append(y_idx[pos + CONFIG.horizon])
                else:
                    target_dates_list.append(pd.NaT)

            # Collect predictions_long rows
            row = pd.DataFrame({
                "date": target_dates_list,     # ✅ now = target_date (h-step-ahead)
                "target_date": target_dates_list,
                "origin_date": od_te.values,
                "origin_value": ov_te.values,
                "y_true": y_te.values,
                "model": model_name,
                "fold": fold_ix,
                "horizon": CONFIG.horizon,
                "target": CONFIG.target,
            })
            for q in CONFIG.quantiles:
                row[f"yhat_p{int(round(q*100))}"] = q_preds[q]
            preds_rows.append(row)

            # Metrics: pinball per quantile + coverage if both lower/upper present
            for q in CONFIG.quantiles:
                pl = _pinball_loss(y_te.values, q_preds[q], q)
                pinballs[q].append(pl)

            if 0.1 in CONFIG.quantiles and 0.9 in CONFIG.quantiles:
                lower = q_preds[0.1]
                upper = q_preds[0.9]
                cov = float(((y_te.values >= lower) & (y_te.values <= upper)).mean())
                coverages.append(cov)

            # Optional fold plot
            plot_df = row.rename(columns={
                "yhat_p10": "yhat_p10",
                "yhat_p50": "yhat_p50",
                "yhat_p90": "yhat_p90"
            })
            _plot_quantiles(
                plot_df[["date", "y_true"] + [c for c in plot_df.columns if c.startswith("yhat_")]],
                os.path.join(out_root, "plots"),
                title=f"{model_name} fold {fold_ix}"
            )

            # Fold-level console line
            pinball_str = ", ".join([f"q{int(q*100)}={np.mean(pinballs[q]):,.2f}" for q in sorted(pinballs.keys())])
            if coverages:
                print(f"[runner]   Fold {fold_ix}: test={len(y_te)}, {pinball_str}, "
                      f"coverage(P10–P90)~{np.mean(coverages):.2%}")
            else:
                print(f"[runner]   Fold {fold_ix}: test={len(y_te)}, {pinball_str}")

        # Aggregate metrics to leaderboard row
        agg = {"model": model_name}
        for q in sorted(CONFIG.quantiles):
            agg[f"pinball_q{int(q*100)}"] = float(np.mean(pinballs[q]))
        if coverages:
            agg["coverage_p10_p90"] = float(np.mean(coverages))
        leaderboard_rows.append(agg)

        # Long metrics for each fold/quantile
        for q in sorted(CONFIG.quantiles):
            for i, (tr_end, te_end) in enumerate(folds, 1):
                # using mean pinball per fold already computed; store it
                metrics_rows.append({
                    "model": model_name,
                    "fold": i,
                    "metric": "pinball",
                    "quantile": q,
                    "value": float(pinballs[q][i-1])
                })
        if coverages:
            for i, v in enumerate(coverages, 1):
                metrics_rows.append({
                    "model": model_name, "fold": i,
                    "metric": "coverage_p10_p90",
                    "quantile": None, "value": float(v)
                })

    # Write master outputs
    predictions_long = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()
    metrics_long = pd.DataFrame(metrics_rows)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(f"pinball_q50" if "pinball_q50" in leaderboard_rows[0] else list(leaderboard_rows[0].keys())[1])

    # ✅ FIX QUANT-3: Integrity checks and persistence baseline
    _quant_integrity = {"pipeline": "QUANTILE", "target": CONFIG.target, "horizon": CONFIG.horizon}
    if not predictions_long.empty and "origin_value" in predictions_long.columns and "yhat_p50" in predictions_long.columns:
        _valid = predictions_long.dropna(subset=["origin_value", "y_true", "yhat_p50"])
        if len(_valid) > 5:
            mae_persist = float(np.mean(np.abs(_valid["y_true"].values - _valid["origin_value"].values)))
            mae_p50 = float(np.mean(np.abs(_valid["y_true"].values - _valid["yhat_p50"].values)))
            skill_pct = ((mae_persist - mae_p50) / mae_persist * 100.0) if mae_persist > 0 else np.nan
            _quant_integrity.update({
                "mae_p50": mae_p50, "mae_persistence": mae_persist,
                "skill_pct": skill_pct,
                "quality_gate_passed": (skill_pct >= 5.0) if np.isfinite(skill_pct) else False,
                "run_status": "SUCCESS" if (np.isfinite(skill_pct) and skill_pct >= 5.0) else "FAILED_QUALITY",
            })
            print(f"[quantile] Persistence MAE={mae_persist:.2f}, P50 MAE={mae_p50:.2f}, Skill={skill_pct:.2f}%")

    _save_csv(predictions_long, out_root, "predictions_long.csv")
    _save_csv(metrics_long, out_root, "metrics_long.csv")
    _save_csv(leaderboard, out_root, "leaderboard.csv")

    # Save integrity report
    _ensure_dir(os.path.join(out_root, "artifacts"))
    import json as _json
    with open(os.path.join(out_root, "artifacts", "integrity_report.json"), "w") as _f:
        _json.dump(_quant_integrity, _f, indent=2, default=str)

    elapsed = time.time() - t0
    _save_run_json(CONFIG, out_root, elapsed)
    print(f"[OK] Master outputs in: {out_root}")
    print("[runner] DONE")
