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
    folds: int = 3
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
    Returns list of (train_end_index_exclusive, test_end_index_exclusive).
    At each fold: train = [0:train_end), test = [train_end: test_end) of length horizon.
    """
    # minimal train length in rows (approx years * 252 trading days if Daily; we’ll just treat min_train as months≈~21*years)
    # To keep it simple and robust, we require at least horizon points for each test.
    indices = []
    # place folds back-to-back ending at series end
    last_test_end = n
    for _ in range(folds, 0, -1):
        test_end = last_test_end
        test_start = test_end - horizon
        if test_start < 0:
            break
        # train must end at test_start
        train_end = test_start
        if train_end <= max(horizon, 30):  # guardrail for tiny series
            break
        indices.append((train_end, test_end))
        last_test_end = test_start
    indices.reverse()
    return indices

def _calendar_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "dow": idx.dayofweek,           # 0..6
        "dom": idx.day,                 # 1..31
        "week": idx.isocalendar().week.astype(int),
        "month": idx.month,             # 1..12
        "year": idx.year
    }, index=idx)

def _build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Make lag/roll/cals; if multivariate, bring exog and optionally select top_k."""
    y = df[cfg.target].astype(float).copy()
    X = pd.DataFrame(index=df.index)

    # Target-derived features
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
        # forward-fill and back-fill modestly, then lag by 1 to avoid leakage
        exog = exog.ffill().bfill().shift(1)
        if cfg.exog_top_k is not None and cfg.exog_top_k > 0:
            # select top abs-corr with target (safe for NaNs)
            corr = exog.join(y).corr(numeric_only=True)[cfg.target].drop(cfg.target, errors="ignore").abs()
            keep = corr.sort_values(ascending=False).head(cfg.exog_top_k).index.tolist()
            exog = exog[keep]
        X = pd.concat([X, exog], axis=1)

    # Align
    dfX = X
    dfy = y
    both = pd.concat([dfX, dfy], axis=1).dropna()
    return both.drop(columns=[cfg.target]), both[cfg.target]

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

    # Features
    X_all, y_all = _build_features(df, CONFIG)
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
            dates_te = X_te.index

            # Fit/predict per model
            if model_name == "GBQuantile":
                q_preds = {}
                for q in CONFIG.quantiles:
                    q_preds[q] = _fit_gb_quantile(X_tr, y_tr, X_te, q)
            elif model_name == "ResidualRF":
                q_preds = _fit_residual_rf_quantiles(X_tr, y_tr, X_te, CONFIG.quantiles)
            else:
                raise ValueError(f"Unknown model '{model_name}'")

            # Collect predictions_long rows
            row = pd.DataFrame({
                "date": dates_te,
                "y_true": y_te.values,
                "model": model_name,
                "fold": fold_ix
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

    _save_csv(predictions_long, out_root, "predictions_long.csv")
    _save_csv(metrics_long, out_root, "metrics_long.csv")
    _save_csv(leaderboard, out_root, "leaderboard.csv")

    elapsed = time.time() - t0
    _save_run_json(CONFIG, out_root, elapsed)
    print(f"[OK] Master outputs in: {out_root}")
    print("[runner] DONE")
