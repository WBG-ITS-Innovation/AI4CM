# e_quantile_daily_pipeline.py
# Georgia Treasury — Quantile models (Daily cadence)
# Contract: same as A_STAT/B_ML — run_pipeline(CONFIG) and write standard outputs.

from __future__ import annotations
import os, json, time, pathlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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
    eval_start: Optional[str] = None  # e.g. "2025-01-01": tile folds over
                                      # [eval_start .. eval_end] (shared benchmark
                                      # window; overrides `folds` count)
    # Upper bound on the evaluation window, INCLUSIVE, by target date.
    # Without this eval_start only set a floor, so pinning to DEV_START tiled folds
    # straight through DEV and on into the 2025 holdout: 418 target dates where DEV
    # has 262. Any "DEV" figure produced that way silently included TEST.
    eval_end: Optional[str] = None
    # Workstream 3 fiscal-calendar feature groups; None == pre-WS3 feature set.
    fiscal_groups: Optional[Tuple[str, ...]] = None
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

def _time_folds(n: int, horizon: int, folds: Optional[int], min_train: int,
                eval_start_idx: Optional[int] = None,
                eval_end_idx: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Expanding-window time-series cross-validation.

    Returns list of (train_end_index_exclusive, test_end_index_exclusive).
    At each fold: train = [0 : train_end), test = [train_end : test_end)
    where len(test) == horizon.

    Training always starts at index 0, so later folds always have at least
    as much training data as earlier ones (expanding window).  Test blocks
    are non-overlapping and placed from the end of the series backward.

    Window selection (one of two modes):
      * eval_start_idx given  -> tile test blocks backward until the block
        would start before eval_start_idx.  This pins the evaluation to a
        fixed window (the shared benchmark window used by all families),
        so results are comparable across families and across runs.  The
        `folds` count is ignored in this mode.
      * eval_start_idx None   -> legacy behaviour: `folds` blocks from the
        end of the series (folds=None means all possible blocks).
    """
    indices: List[Tuple[int, int]] = []
    min_train_rows = max(min_train * 252, horizon, 30)

    # Pinned mode: tile FORWARD from eval_start so the first block begins exactly
    # there. Tiling backward from the end (the original behaviour) left a remainder
    # at the START of the window whenever its length was not a multiple of horizon:
    # with 156 target dates and h=5 the family evaluated 150 of them, starting
    # 2025-01-09 instead of 2025-01-01. That is a different window from every other
    # family, so the one-ruler check reported different persistence numbers -- on
    # Expenditure a 5.8M difference (78,036,083 vs 83,839,124). The final block may
    # be shorter than `horizon`; a partial block is a smaller sample, not a wrong one.
    if eval_start_idx is not None:
        # Exclusive upper bound on the tiling. Capping the window is what keeps a
        # DEV run out of the TEST holdout.
        stop = n if eval_end_idx is None else min(n, eval_end_idx + 1)
        start = max(eval_start_idx, min_train_rows + 1)
        while start < stop:
            test_end = min(start + horizon, stop)
            if start <= min_train_rows:
                start = test_end
                continue
            indices.append((start, test_end))
            start = test_end
        return indices

    last_test_end = n
    remaining = folds  # None => no count limit
    while True:
        if remaining is not None and remaining <= 0:
            break
        test_end = last_test_end
        test_start = test_end - horizon
        if test_start < 0:
            break
        if eval_start_idx is not None and test_start < eval_start_idx:
            break
        train_end = test_start          # train = [0 : train_end)
        # min_train is in years; converted to approximate rows (252 biz days/yr).
        # Also enforce at least max(horizon, 30) so very short horizons don't
        # create trivially small training sets.
        if train_end <= min_train_rows:
            break
        indices.append((train_end, test_end))
        last_test_end = test_start
        if remaining is not None:
            remaining -= 1
    indices.reverse()   # earliest fold first
    return indices

def is_stock(target: str) -> bool:
    """Level (stock) targets vs flow targets.

    Kept byte-identical to b_ml_pipeline.is_stock and c_dl_pipeline.is_stock so the
    three families cannot disagree about what kind of series they are modelling.
    """
    return str(target).strip().lower() in {"state budget balance", "balance", "t0"}


def to_business_index(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Reindex to a business-day calendar so h=5 means 5 BUSINESS days.

    Why this matters (review §1.2, §4.3): E_QUANTILE was the only family that did
    not reindex, so it ran on the raw 7-day calendar index and its ``h=5`` meant 5
    *calendar* days -- a 5-day-ahead forecast against every other family's 7-day
    (5 business day) one, graded against a persistence baseline computed over the
    same shorter gap. Measured on the pre-Phase-1 data, that gave E_QUANTILE a
    9.8% weaker (easier to beat) ruler: 66,161,268 vs the shared 60,273,679 on the
    same 148 target dates, and 28% of its evaluation targets were weekends no
    other family scored.

    Flows are filled with 0.0 on non-business days (no trading day, no flow);
    levels are forward-filled (a balance persists). Same convention as
    b_ml_pipeline.to_business_index.
    """
    out = df.copy()
    bidx = pd.date_range(out.index.min().normalize(), out.index.max().normalize(), freq="B")
    out = out.reindex(bidx)
    out.index.name = df.index.name or "date"
    stock = is_stock(target)
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col].ffill() if stock else out[col].fillna(0.0)
    return out


def _calendar_feats(idx: pd.DatetimeIndex,
                    y: Optional[pd.Series] = None,
                    fiscal_groups: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Calendar features.

    ``year`` was REMOVED (workstream 3). A tree that splits on the calendar year puts
    every 2025 row into a terminal bucket learned from 2024, so it fits the trend rather
    than the mechanism and cannot extrapolate past the training range. It was a pure
    trend crutch and the only family still carrying it.
    """
    out = pd.DataFrame({
        "dow": idx.dayofweek,           # 0..6
        "dom": idx.day,                 # 1..31
        "week": idx.isocalendar().week.astype(int),
        "month": idx.month,             # 1..12
    }, index=idx)
    if fiscal_groups:
        from preprocessing.fiscal_calendar import build_fiscal_features, drop_raw_year
        fx = build_fiscal_features(idx, y=y, groups=list(fiscal_groups))
        fx = fx.loc[:, [c for c in fx.columns if c not in out.columns]]
        out = drop_raw_year(pd.concat([out, fx], axis=1))
    return out

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

    Stock targets (D12).  For a level series such as ``State budget balance`` the
    modelling target is the CHANGE, ``y(t+h) - y(t)``, not the level.  A level is
    dominated by its own current value, so predicting it directly makes the model a
    trivial persistence predictor that scores well while learning nothing -- the
    trap b_ml_pipeline documents at :103-110.  ``lag_0`` (the value at origin) is
    added as a feature only in this mode, where it supplies change-context rather
    than the answer: it is known at the origin by definition, so it is not leakage.
    Callers reconstruct the level as ``origin_value + predicted_delta``; see
    ``run_pipeline``.
    """
    y = df[cfg.target].astype(float).copy()
    X = pd.DataFrame(index=df.index)
    stock = is_stock(cfg.target)

    # lag_0 = the value AT the origin. Known at forecast time by construction.
    # Only useful (and only safe from the persistence trap) under delta modelling.
    if stock:
        X["y_lag_0"] = y

    # Target-derived features (all backward-looking: safe)
    for l in cfg.lags_daily:
        X[f"y_lag_{l}"] = y.shift(l)
    for w in cfg.windows_daily:
        X[f"y_roll_mean_{w}"] = y.rolling(w, min_periods=1).mean().shift(1)
        X[f"y_roll_std_{w}"] = y.rolling(w, min_periods=1).std(ddof=0).shift(1)

    # Calendar features
    X = pd.concat([X, _calendar_feats(df.index, y=y,
                                      fiscal_groups=cfg.fiscal_groups)], axis=1)

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

    # origin_values = y at feature dates. Always the LEVEL, never the delta:
    # downstream persistence and level reconstruction both need the real y(t).
    origin_values = y.copy()

    # Stock targets are modelled as the change from origin (D12). The level target
    # stays available to the caller via origin_value + delta.
    if stock:
        y_target = y_target - origin_values

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
    Distribution-free 'residual quantile' intervals around a RandomForest.

    1) Fit a RandomForest point model.
    2) Estimate residual quantiles from **out-of-bag** predictions.
    3) Shift the point prediction by those residual quantiles.

    ✅ M-3: the residuals must be out-of-bag.  An unpruned forest nearly
    memorises its training rows, so *in-sample* residuals are far smaller
    than real forecast errors and the resulting intervals collapse (measured:
    26.5% coverage where P10–P90 should cover ~80%).  With OOB predictions
    each training row is scored only by the trees that did not see it, so the
    residuals reflect genuine generalisation error.

    Falls back to in-sample residuals only if OOB predictions are unavailable
    (very small training sets can leave rows with no out-of-bag votes); in
    that case intervals are known to be optimistic, which the caller's
    coverage gate will catch.
    """
    from sklearn.ensemble import RandomForestRegressor

    # point model (oob_score=True makes sklearn store oob_prediction_)
    rf = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, max_depth=None,
        bootstrap=True, oob_score=True,
    )
    rf.fit(X_tr, y_tr)

    y_tr_arr = np.asarray(y_tr, dtype=float)
    resid = None
    oob_pred = getattr(rf, "oob_prediction_", None)
    if oob_pred is not None:
        oob_pred = np.asarray(oob_pred, dtype=float)
        ok = np.isfinite(oob_pred)          # rows that received OOB votes
        if ok.sum() >= max(10, int(0.5 * len(y_tr_arr))):
            resid = y_tr_arr[ok] - oob_pred[ok]
    if resid is None or resid.size == 0:
        # Fallback: in-sample residuals (optimistic — flagged by the gate).
        resid = y_tr_arr - rf.predict(X_tr)

    point = rf.predict(X_te)
    preds = {}
    for q in quantiles:
        preds[q] = point + float(np.quantile(resid, q))
    # Guarantee monotone quantiles even if the residual quantiles are noisy.
    for lo, hi in zip(sorted(quantiles), sorted(quantiles)[1:]):
        preds[hi] = np.maximum(preds[hi], preds[lo])
    return preds

def quantile_quality_gate(skill_pct: float, coverage: Optional[float],
                          min_skill: float = 5.0,
                          coverage_band: Tuple[float, float] = (0.70, 0.90),
                          ) -> Tuple[bool, List[str]]:
    """Quality gate for a quantile model: skill AND calibrated intervals.

    A quantile family exists to produce intervals a treasury can plan
    around, so miscalibrated coverage fails the gate even when P50 skill
    is excellent.  The band is the nominal 80% (P10–P90) ± 10pp, roughly
    a 3-sigma binomial tolerance at ~150 evaluation points.
    Returns (passed, reasons); reasons is empty when passed.
    """
    reasons: List[str] = []
    if not np.isfinite(skill_pct) or skill_pct < min_skill:
        reasons.append(f"skill {skill_pct:.2f}% < {min_skill:.1f}% required")
    if coverage is None or not np.isfinite(coverage):
        reasons.append("coverage not measurable (P10/P90 missing)")
    elif not (coverage_band[0] <= coverage <= coverage_band[1]):
        reasons.append(
            f"coverage {coverage:.1%} outside "
            f"[{coverage_band[0]:.0%}, {coverage_band[1]:.0%}] (nominal 80%)"
        )
    return (len(reasons) == 0, reasons)


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

    # Reindex to business days so h=CONFIG.horizon means h BUSINESS days, matching
    # every other family. Without this the family ran on a 7-day calendar index and
    # h=5 meant 5 calendar days -- a shorter forecast graded against an easier
    # ruler (review §4.3).
    _n_before = len(df)
    df = to_business_index(df, CONFIG.target)
    print(f"[runner] Reindexed to business days: {_n_before} -> {len(df)} rows "
          f"({'ffill' if is_stock(CONFIG.target) else 'fillna(0)'} for a "
          f"{'stock' if is_stock(CONFIG.target) else 'flow'} target)")

    # Features — now returns h-step-ahead targets + origin metadata.
    # For a stock target y_all holds the DELTA from origin; levels are
    # reconstructed below.
    X_all, y_all, od_all, ov_all = _build_features(df, CONFIG)
    _stock = is_stock(CONFIG.target)
    if _stock:
        print(f"[runner] Stock target '{CONFIG.target}': modelling delta "
              f"y(t+{CONFIG.horizon}) - y(t); predictions reconstructed as "
              f"origin_value + delta")
    n = len(y_all)
    if n < CONFIG.horizon + 50:
        print("[runner] WARNING: very short series after feature alignment.")

    # CV folds — pinned to the shared benchmark window when eval_start is set.
    eval_start_idx = None
    if CONFIG.eval_start:
        od_ts = pd.to_datetime(pd.Series(od_all.values))
        # Pin on TARGET dates, not origin dates. The other three families define
        # their window by target_date, so pinning origins >= eval_start put
        # E_QUANTILE's first target h steps LATER (2025-01-08 instead of
        # 2025-01-01) -- a different window, and therefore a different persistence
        # number, which is exactly what the one-ruler check exists to catch.
        # target[i] lies h positions after origin[i], so the origin index that
        # yields the first in-window target is (index of eval_start) - h.
        _k = int(np.searchsorted(od_ts.values, np.datetime64(pd.Timestamp(CONFIG.eval_start))))
        eval_start_idx = max(0, _k - int(CONFIG.horizon))
        if eval_start_idx >= len(od_ts):
            raise ValueError(
                f"eval_start {CONFIG.eval_start} is beyond the last origin date "
                f"{od_ts.iloc[-1].date()} — nothing to evaluate."
            )
        print(f"[runner] Evaluation window pinned: origins >= {CONFIG.eval_start} "
              f"({len(od_ts) - eval_start_idx} rows)")
    eval_end_idx = None
    if CONFIG.eval_end:
        od_ts2 = pd.to_datetime(pd.Series(od_all.values))
        # Same target-date convention as eval_start: the last origin whose target
        # still falls on or before eval_end.
        _j = int(np.searchsorted(od_ts2.values,
                                 np.datetime64(pd.Timestamp(CONFIG.eval_end)), side="right"))
        eval_end_idx = max(0, _j - int(CONFIG.horizon) - 1)
        print(f"[runner] Evaluation window capped at {CONFIG.eval_end} "
              f"(origin index <= {eval_end_idx})")
    folds = _time_folds(n, CONFIG.horizon, CONFIG.folds, CONFIG.min_train_years,
                        eval_start_idx, eval_end_idx)
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

            # Reconstruct levels for a stock target: the model predicted the
            # change from origin, so the reported forecast is origin + change, and
            # y_true is restored to the level. Every quantile is shifted by the
            # same origin, so the interval width is unchanged and monotonicity is
            # preserved.
            if _stock:
                _ov = np.asarray(ov_te.values, dtype=float)
                y_te_out = _ov + np.asarray(y_te.values, dtype=float)
                q_preds = {q: _ov + np.asarray(v, dtype=float) for q, v in q_preds.items()}
            else:
                y_te_out = y_te.values

            # Collect predictions_long rows
            row = pd.DataFrame({
                "date": target_dates_list,     # ✅ now = target_date (h-step-ahead)
                "target_date": target_dates_list,
                "origin_date": od_te.values,
                "origin_value": ov_te.values,
                "y_true": y_te_out,
                "model": model_name,
                "fold": fold_ix,
                "horizon": CONFIG.horizon,
                "target": CONFIG.target,
            })
            for q in CONFIG.quantiles:
                row[f"yhat_p{int(round(q*100))}"] = q_preds[q]
            if 0.5 in CONFIG.quantiles:
                # Alias the median as y_pred so shared diagnostics (e.g. the
                # daily summary's detect_lagged_copy check) treat it as the
                # family's point forecast.
                row["y_pred"] = q_preds[0.5]
            preds_rows.append(row)

            # Metrics: pinball per quantile + coverage if both lower/upper present.
            # Uses y_te_out / q_preds, i.e. LEVELS for a stock target, so the
            # metrics describe the quantity actually reported. (Both are shifted by
            # the same origin, so pinball diffs and coverage are invariant -- this
            # is for consistency, not to change the numbers.)
            for q in CONFIG.quantiles:
                pl = _pinball_loss(y_te_out, q_preds[q], q)
                pinballs[q].append(pl)

            if 0.1 in CONFIG.quantiles and 0.9 in CONFIG.quantiles:
                lower = q_preds[0.1]
                upper = q_preds[0.9]
                cov = float(((y_te_out >= lower) & (y_te_out <= upper)).mean())
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

    # ✅ M-2: per-model integrity, coverage-aware gate, shift diagnostic.
    # The old block pooled every model's rows into one frame, so the reported
    # skill was an average across models that belonged to none of them.
    _quant_integrity = {"pipeline": "QUANTILE", "target": CONFIG.target,
                        "horizon": CONFIG.horizon, "eval_start": CONFIG.eval_start}
    needed = {"origin_value", "y_true", "yhat_p50", "model"}
    if not predictions_long.empty and needed.issubset(predictions_long.columns):
        from forecast_integrity import (
            compute_persistence_baseline,
            shift_diagnostic_horizon_aware,
        )
        per_model: Dict[str, Dict] = {}
        valid_all = predictions_long.dropna(subset=["origin_value", "y_true", "yhat_p50"])
        for model_name, g in valid_all.groupby("model"):
            if len(g) <= 5:
                continue
            mae_persist = compute_persistence_baseline(g)["mae_persistence"]
            mae_p50 = float(np.mean(np.abs(g["y_true"].values - g["yhat_p50"].values)))
            skill_pct = ((mae_persist - mae_p50) / mae_persist * 100.0) if mae_persist > 0 else float("nan")
            coverage = None
            if {"yhat_p10", "yhat_p90"}.issubset(g.columns):
                coverage = float(np.mean((g["y_true"].values >= g["yhat_p10"].values)
                                         & (g["y_true"].values <= g["yhat_p90"].values)))
            gate_passed, gate_reasons = quantile_quality_gate(skill_pct, coverage)
            per_model[str(model_name)] = {
                "n_predictions": int(len(g)),
                "mae_p50": mae_p50,
                "mae_persistence": mae_persist,
                "skill_pct": skill_pct,
                "coverage_p10_p90": coverage,
                "gate_passed": gate_passed,
                "gate_reasons": gate_reasons,
            }
            cov_s = f"{coverage:.1%}" if coverage is not None else "n/a"
            print(f"[quantile] {model_name}: n={len(g)}, P50 MAE={mae_p50:,.2f}, "
                  f"Persistence MAE={mae_persist:,.2f}, Skill={skill_pct:.2f}%, "
                  f"Coverage(P10–P90)={cov_s}, gate={'PASS' if gate_passed else 'FAIL'}")

        if per_model:
            # Best model = lowest P50 MAE **among gate-passing models** —
            # calibrated intervals are the point of this family, so a model
            # with broken coverage cannot be "best" merely on median MAE.
            # If no model passes, fall back to lowest MAE (and the gate fails).
            passing = [m for m in per_model if per_model[m]["gate_passed"]]
            pool = passing if passing else list(per_model)
            best_model = min(pool, key=lambda m: per_model[m]["mae_p50"])
            best = per_model[best_model]
            # Independent shift diagnostic on the best model's median forecast
            # (same shared helper the other families use).
            g_best = valid_all[valid_all["model"] == best_model]
            shift = shift_diagnostic_horizon_aware(
                g_best["y_true"].values, g_best["yhat_p50"].values, CONFIG.horizon
            )
            _quant_integrity.update({
                "models": per_model,
                "best_model": best_model,
                "mae_p50": best["mae_p50"],
                "mae_persistence": best["mae_persistence"],
                "skill_pct": best["skill_pct"],
                "coverage_p10_p90": best["coverage_p10_p90"],
                "quality_gate_passed": best["gate_passed"],
                "quality_gate_reasons": best["gate_reasons"],
                "run_status": "SUCCESS" if best["gate_passed"] else "FAILED_QUALITY",
                # Fields the daily summary reads for its pipeline shift line.
                "best_shift": shift.get("best_shift"),
                "shift_interpretation": shift.get("interpretation"),
                "mae_model": shift.get("mae_shift0"),
                "mae_shift_minus_h": shift.get("mae_shift_minus_h"),
                "is_persistence_like": shift.get("is_persistence_like"),
                "is_lag0_issue": shift.get("is_lag0_issue"),
            })

    _save_csv(predictions_long, out_root, "predictions_long.csv")
    _save_csv(metrics_long, out_root, "metrics_long.csv")
    # MAE of the median forecast per model, so downstream tools (e.g. the
    # daily summary's best-model line) can rank quantile models too.
    if not predictions_long.empty and {"y_true", "yhat_p50"}.issubset(predictions_long.columns):
        _ae = predictions_long.dropna(subset=["y_true", "yhat_p50"]).copy()
        _ae["abs_err"] = (_ae["y_true"] - _ae["yhat_p50"]).abs()
        _mae = _ae.groupby("model")["abs_err"].mean().rename("MAE").reset_index()
        leaderboard = leaderboard.merge(_mae, on="model", how="left")
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
