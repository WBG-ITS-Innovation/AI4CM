# run_a_stat.py — FINAL (Statistical models + Treasury baseline + robust folds)

from __future__ import annotations

import json, os, time, warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.theta import ThetaModel


def _log(msg: str): print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)

from evaluation_windows import (PURPOSE_REPORT, require_test_access,  # noqa: E402
                                window_for as _window_for)

# The implementation that diverged: it alone treated "net" and "stock" as stock targets and
# "t0" as a flow, so a column named t0 would have been modelled as a level here and as a delta
# in the other three families. Now one shared definition, whose alias set is the UNION of all
# four -- see backend/target_kinds.py for why the union rather than a pick.
from target_kinds import is_stock as _is_stock  # noqa: E402,F401

def _resample(df: pd.DataFrame, target: str, cadence: str, date_col: str) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    y = df[target].astype(float)
    cad = cadence.lower()
    if cad == "daily":
        bidx = pd.date_range(y.index.min(), y.index.max(), freq="B")
        ser = y.reindex(bidx)
        ser = ser.ffill() if _is_stock(target) else ser.fillna(0.0)
        ser.index.freq = "B"
        return ser
    if cad == "weekly":
        ser = (y.resample("W-FRI").last() if _is_stock(target) else y.resample("W-FRI").sum())
        ser.index.freq = ser.index.freq or pd.infer_freq(ser.index)
        return ser
    # monthly
    ser = (y.resample("ME").last() if _is_stock(target) else y.resample("ME").sum())
    ser.index.freq = ser.index.freq or pd.infer_freq(ser.index)
    return ser

# ── the Ops baseline: delegated, not re-implemented ───────────────────────────
#
# This module used to carry its own copy of the Treasury planning method, and that copy had the
# same defect as C_DL's: the intraday profile was built from the same month in a PREVIOUS year and
# then mapped onto the current year's dates with ``.reindex(days, fill_value=0.0)``. Labels cannot
# match across years, so every weight became zero and the daily baseline was identically zero
# (measured here: 2000 of 2000 values exactly 0.00).
#
# Four independent copies of this arithmetic existed, and fixing one in the ops-baseline session
# left three wrong -- which is the argument for delegation rather than a fourth patch. These now
# call ``backend/ops_baseline``, which is the single construction the scorecard, the leaderboards
# and this runner all share.


def _ops_monthly_baseline(series_daily: pd.Series, years: int = 3) -> pd.Series:
    """Monthly Treasury baseline (3-year annual mean x month share), via ops_baseline."""
    from c_dl_pipeline import ops_monthly_baseline_treasury
    base = ops_monthly_baseline_treasury(series_daily, years_window=years)
    if len(base):
        base.index.freq = "ME"
    return base


def _ops_daily_from_monthly(daily_hist: pd.Series, monthly_forecast: pd.Series) -> pd.Series:
    """Spread the monthly baseline over working days, evenly.

    ``flat`` is the canonical spread: it is the method as stated ("spread across working days"),
    it emits no negative planning figures, and it is the harsher comparison. See
    ``backend/ops_baseline`` for the measured reasoning.
    """
    from c_dl_pipeline import ops_daily_from_monthly
    from ops_baseline import SPREAD_FLAT
    out = ops_daily_from_monthly(daily_hist, monthly_forecast, method=SPREAD_FLAT)
    return out.dropna()


def _yearly_folds(idx: pd.DatetimeIndex, min_years: int, want_folds: Optional[int],
                  eval_start: Optional[str] = None,
                  eval_end: Optional[str] = None) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    """Annual rolling-origin folds, optionally bounded to an evaluation window.

    ``eval_start`` / ``eval_end`` are INCLUSIVE bounds on the TARGET dates a fold scores.
    Both default to ``None``, which is the behaviour this function has always had: fold
    over every full year in the file. That default is correct for the reporting run this
    module was written for, and wrong for anything that compares models by eye, because
    the last year in the file is the sealed holdout.

    B_ML's ``build_yearly_folds`` already took these bounds and this is the same rule, so
    a run bounded to train and dev covers the same years in both families. A fold that
    falls entirely outside the bounds is dropped; one that straddles an edge is trimmed
    to it, because a partial block is a smaller sample rather than a wrong one.
    """
    years = sorted(set(idx.year))
    lo = pd.Timestamp(eval_start) if eval_start else None
    hi = pd.Timestamp(eval_end) if eval_end else None
    folds=[]
    for Y in years:
        if Y - years[0] < min_years: 
            continue
        tr_end_cand = idx[idx <= pd.Timestamp(f"{Y-1}-12-31")]
        if tr_end_cand.empty: continue
        tr_end = tr_end_cand[-1]
        ts_span = idx[(idx >= pd.Timestamp(f"{Y}-01-01")) & (idx <= pd.Timestamp(f"{Y}-12-31"))]
        if ts_span.empty: continue
        if lo is not None:
            ts_span = ts_span[ts_span >= lo]
        if hi is not None:
            ts_span = ts_span[ts_span <= hi]
        if ts_span.empty: continue
        folds.append((tr_end, ts_span[0], ts_span[-1]))
    # If want_folds is None, use ALL folds (thorough mode). Otherwise limit to last N folds.
    if want_folds is not None and want_folds > 0 and len(folds) > want_folds:
        folds = folds[-want_folds:]
    return folds

def _fallback_fold(idx: pd.DatetimeIndex, horizon: int,
                   eval_start: Optional[str] = None,
                   eval_end: Optional[str] = None) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    """Used when yearly folds cannot be built; always return at least one fold if there is enough history to test `horizon`.

    The bounds are applied to the index BEFORE the block is carved, not to the block
    afterwards. Trimming afterwards would be worse than useless: this fallback takes the
    last ``te_len`` rows of the file, and on a bounded run those rows are exactly the ones
    the bound exists to exclude, so the caller would silently get an empty fold list from
    a function whose contract is "always return at least one fold". Restricting the index
    first means the fallback returns a real fold inside the window, or an honest nothing.
    """
    if eval_end:
        idx = idx[idx <= pd.Timestamp(eval_end)]
    n = len(idx)
    if n <= horizon + 5: 
        return []
    te_len = max(horizon, min(12, n//4))
    te_end = idx[-1]; te_start = idx[-te_len]
    tr_end = idx[-(te_len + horizon)]
    if eval_start:
        # Only the scored block moves. ``tr_end`` is history, and history before the
        # evaluation start is exactly what the model is supposed to learn from.
        lo = pd.Timestamp(eval_start)
        if te_end < lo:
            return []
        if te_start < lo:
            te_start = idx[idx >= lo][0]
    return [(tr_end, te_start, te_end)]

# ══════════════════════════════════════════════════════════════════════════════
# THE MODELS THIS FAMILY OFFERS -- ONE SOURCE OF TRUTH
#
# Item 6 part 3. `model_reference.DESCRIPTIONS` described ETS and Theta while `model_pool()`
# enumerated only B_ML and E_QUANTILE, so two descriptions were unreachable from the Models page
# and no test could tell. The fix is not to delete the descriptions -- these are real production
# models, and this module (not the unreferenced `a_stat_models_pipeline.py`) is the one
# `scripts/run_daily_forecast.sh` invokes. The fix is to make the family enumerable, the same way
# E_QUANTILE already is via its `registry_models()`.
#
# `_fc` dispatches on these names and now REFUSES an unknown one. It used to fall through to a
# naive forecast, so `TG_MODEL_FILTER=XGBoost` would have produced a carried-forward last value
# published under the label "XGBoost".
# ══════════════════════════════════════════════════════════════════════════════

#: ``{NAME: {"summary": ..., "role": "forecast" | "baseline"}}``
#: ``role`` matters for counting: a reference baseline is not a competing model, and summing the
#: two would inflate any headline count shown to a client (see reports/gate_audit.md §4).
A_STAT_MODELS: Dict[str, Dict[str, str]] = {
    "NAIVE": {"role": "baseline",
              "summary": "Carry the last observed value forward. The reference every other "
                         "model is measured against, not a competitor."},
    "WEEKDAY_MEAN": {"role": "baseline",
                     "summary": "Predict each day with the historical average for that weekday. "
                                "A calendar-only reference."},
    "MOVAVG": {"role": "baseline",
               "summary": "Predict the mean of the last N observations (default 7). A smoothing "
                          "reference with no trend or seasonal term."},
    "ETS": {"role": "forecast",
            "summary": "Exponential smoothing — a weighted average of the past where recent "
                       "observations count for more, with optional trend and seasonal terms. "
                       "Uses only the target's own history."},
    "SARIMAX": {"role": "forecast",
                "summary": "Seasonal ARIMA with optional external regressors. Models the series "
                           "through its own autocorrelation and differencing, and is the only "
                           "A_STAT model that can take exogenous inputs."},
    "STL_ARIMA": {"role": "forecast",
                  "summary": "Split the series into trend, season and remainder (STL), forecast "
                             "the remainder with ARIMA, then recombine. Useful when the seasonal "
                             "shape is strong and stable."},
    "THETA": {"role": "forecast",
              "summary": "A classical decomposition method: de-trend the series, forecast the "
                         "pieces, recombine. Strong on smooth seasonal series and a well-known "
                         "competition benchmark."},
}


class UnknownAStatModel(ValueError):
    """A model name this family does not implement."""


def registry_models() -> Dict[str, str]:
    """The models this family offers, as ``{name: description}``.

    Mirrors ``e_quantile_daily_pipeline.registry_models()`` so both families are enumerable by the
    same contract, and so a model cannot be added to the dispatch without appearing in the
    reports, the Models page and the tests that enumerate it.
    """
    return {name: spec["summary"] for name, spec in A_STAT_MODELS.items()}


def model_roles() -> Dict[str, str]:
    """``{name: "forecast" | "baseline"}`` — so a count can exclude the references."""
    return {name: spec["role"] for name, spec in A_STAT_MODELS.items()}


# predictors — each returns (y_pred, y_lo, y_hi) as numpy arrays
def _nan_pi(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return NaN prediction interval arrays of length n."""
    return np.full(n, np.nan), np.full(n, np.nan)

def _fc(model: str, y_tr: pd.Series, idx: pd.DatetimeIndex,
        ov: Dict, cadence: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast with optional native prediction intervals.
    Returns (y_pred, y_lo, y_hi).  Models without native PIs return NaN for y_lo/y_hi.
    """
    m = model.upper()
    if m not in A_STAT_MODELS:
        raise UnknownAStatModel(
            f"A_STAT does not implement {model!r}. Known models: "
            f"{', '.join(sorted(A_STAT_MODELS))}. Refusing to forecast -- this used to fall "
            f"through to a carried-forward last value, which would be published under the "
            f"requested model's name.")
    n = len(idx)
    pi_alpha = float(ov.get("pi_alpha", 0.10))  # 90 % PI by default

    if m == "NAIVE":
        y_pred = np.repeat(y_tr.iloc[-1], n).astype(float)
        return y_pred, *_nan_pi(n)

    if m == "WEEKDAY_MEAN":
        if y_tr.index.freqstr and "B" in y_tr.index.freqstr:
            wd = y_tr.groupby(y_tr.index.dayofweek).mean()
            y_pred = np.array([wd.get(ts.dayofweek, y_tr.iloc[-1]) for ts in idx], dtype=float)
        else:
            y_pred = np.repeat(y_tr.iloc[-1], n).astype(float)
        return y_pred, *_nan_pi(n)

    if m == "MOVAVG":
        w = int(ov.get("MOVAVG", {}).get("window", 7))
        y_pred = np.repeat(float(y_tr.tail(w).mean()), n).astype(float)
        return y_pred, *_nan_pi(n)

    if m == "ETS":
        ets_ov = ov.get("ETS", {})
        trend = None if ets_ov.get("trend") in (None, "None") else ets_ov.get("trend", "add")
        seasonal = None if ets_ov.get("seasonal") in (None, "None") else ets_ov.get("seasonal", "add")
        periods = int(ets_ov.get("seasonal_periods", 12))
        damped = bool(ets_ov.get("damped_trend", False))
        if seasonal and len(y_tr) < 2 * periods:
            seasonal = None
        try:
            fit = ExponentialSmoothing(y_tr, trend=trend, seasonal=seasonal,
                                       seasonal_periods=periods if seasonal else None,
                                       damped_trend=damped, initialization_method="estimated").fit(optimized=True)
            y_pred = fit.forecast(n).values.astype(float)
            # Try to get native prediction intervals
            try:
                pi = fit.get_prediction(start=len(y_tr), end=len(y_tr) + n - 1)
                sf = pi.summary_frame(alpha=pi_alpha)
                return y_pred, sf["pi_lower"].values.astype(float), sf["pi_upper"].values.astype(float)
            except Exception:
                return y_pred, *_nan_pi(n)
        except Exception:
            y_pred = np.repeat(y_tr.iloc[-1], n).astype(float)
            return y_pred, *_nan_pi(n)

    if m == "SARIMAX":
        sar = ov.get("SARIMAX", {})
        ord_ = tuple(sar.get("order", [1,1,1]))
        sOrd = tuple(sar.get("seasonal_order", [0,0,0, 12 if cadence=='Monthly' else 5]))
        fit = SARIMAX(y_tr, order=ord_, seasonal_order=sOrd,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(n)
        y_pred = fc.predicted_mean.values.astype(float)
        sf = fc.summary_frame(alpha=pi_alpha)
        return y_pred, sf["mean_ci_lower"].values.astype(float), sf["mean_ci_upper"].values.astype(float)

    if m == "STL_ARIMA":
        stl_ov = ov.get("STL_ARIMA", {})
        sp = int(stl_ov.get("stl_period", 12 if cadence=='Monthly' else 5))
        ao = tuple(stl_ov.get("arima_order", [1,1,0]))
        comp = STL(y_tr, period=sp, robust=True).fit()
        fit = SARIMAX(comp.resid, order=ao, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(n)
        res_pred = fc.predicted_mean.values
        seas = np.resize(comp.seasonal[-sp:], n) if sp > 0 and len(comp.seasonal) >= sp else np.zeros(n)
        trend_vals = np.full(n, comp.trend.iloc[-1])
        y_pred = trend_vals + seas + res_pred
        # PI from residual ARIMA, shifted by trend+seasonal
        sf = fc.summary_frame(alpha=pi_alpha)
        y_lo = trend_vals + seas + sf["mean_ci_lower"].values.astype(float)
        y_hi = trend_vals + seas + sf["mean_ci_upper"].values.astype(float)
        return y_pred, y_lo, y_hi

    if m == "THETA":
        y_pred = ThetaModel(y_tr).fit().forecast(n).values.astype(float)
        return y_pred, *_nan_pi(n)

    # Unreachable: membership was checked above. Kept as an explicit failure rather than a
    # silent naive fallback, so a model listed in A_STAT_MODELS but never wired here is caught.
    raise UnknownAStatModel(
        f"{m} is listed in A_STAT_MODELS but has no branch in _fc(); it was added to the "
        f"registry without being implemented.")

def _plot_overlay(df_slice: pd.DataFrame, out_png: Path, ops: Optional[pd.Series]):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_slice["date"], df_slice["y_true"], color="black", lw=2, label="Actual")
    ax.plot(df_slice["date"], df_slice["y_pred"], color="#59a4ff", lw=2, label=df_slice["model"].iloc[0])
    if ops is not None and len(ops):
        ops_aligned = ops.reindex(pd.to_datetime(df_slice["date"]))
        ax.plot(df_slice["date"], ops_aligned.values, color="red", lw=1.6, ls="--", label="Ops baseline")
    ax.grid(True, ls=":"); ax.legend(loc="best")
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=110); plt.close(fig)

def main():
    _log("===== Georgia A·Stat runner =====")
    env = {k: os.environ.get(k,"") for k in
           ["TG_MODEL_FILTER","TG_TARGET","TG_CADENCE","TG_HORIZON","TG_DATA_PATH","TG_DATE_COL","TG_OUT_ROOT","TG_PARAM_OVERRIDES"]}
    ov = json.loads(env.get("TG_PARAM_OVERRIDES") or "{}")

    model   = (env["TG_MODEL_FILTER"] or "NAIVE").upper()
    target  = env["TG_TARGET"]
    cadence = (env["TG_CADENCE"] or "Monthly").capitalize()
    horizon = int(env["TG_HORIZON"] or 6)
    data    = env["TG_DATA_PATH"]
    dcol    = env["TG_DATE_COL"] or "date"
    outroot = Path(env["TG_OUT_ROOT"]).resolve()
    folds_raw = ov.get("folds", 3)
    folds = None if folds_raw is None else int(folds_raw)  # None = use ALL folds (thorough mode)
    minyrs  = int(ov.get("min_train_years", 4))
    demo    = ov.get("demo_clip_months")
    # Both default to None, so an unbounded reporting run is byte-identical to before.
    # The Lab sets eval_end so an exploratory run never folds into the sealed window.
    eval_start = ov.get("eval_start") or None
    eval_end   = ov.get("eval_end") or None

    for k,v in [("TG_FAMILY","A_STAT"),("TG_MODEL_FILTER",model),("TG_TARGET",target),
                ("TG_CADENCE",cadence),("TG_HORIZON",horizon),("TG_DATA_PATH",data),
                ("TG_DATE_COL",dcol),("TG_PARAM_OVERRIDES",json.dumps(ov)),("TG_OUT_ROOT",str(outroot))]:
        _log(f"{k} = {v}")

    _log("Loading data…")
    raw = pd.read_csv(data)
    y_all = _resample(raw, target, cadence, dcol)
    if demo:
        cut = y_all.index.max() - pd.DateOffset(months=int(demo))
        y_all = y_all[y_all.index >= cut]

    cad_dir = outroot/cadence.lower(); (cad_dir/"plots").mkdir(parents=True, exist_ok=True)

    ops_daily = pd.Series(dtype=float); ops_month = pd.Series(dtype=float)
    if not _is_stock(target):
        y_daily = _resample(raw, target, "Daily", dcol)
        ops_month = _ops_monthly_baseline(y_daily)
        ops_daily = _ops_daily_from_monthly(y_daily, ops_month)
        pd.DataFrame({"date": ops_daily.index, "forecast": ops_daily.values}).to_csv(
            cad_dir/f"{target}_ops_baseline_daily.csv", index=False)
        ops_month.rename("forecast").to_csv(cad_dir/f"{target}_ops_baseline_monthly.csv")

    idx = y_all.index
    folds_list = _yearly_folds(idx, minyrs, folds, eval_start=eval_start, eval_end=eval_end)
    if eval_start or eval_end:
        _log(f"Evaluation window bounded: [{eval_start or 'start'} .. {eval_end or 'end'}] "
             f"-> {len(folds_list)} fold(s)")
    if not folds_list:
        _log("WARNING: Not enough full-year coverage; using recent sliding-window fold.")
        folds_list = _fallback_fold(idx, horizon, eval_start=eval_start, eval_end=eval_end)
        if not folds_list:
            # Last-ditch block, still inside the bound. Reaching past the bound here was
            # the quiet way a bounded run could end up scoring the sealed window: the
            # last-horizon block of an unrestricted index is precisely the newest data.
            bounded = idx
            if eval_start:
                bounded = bounded[bounded >= pd.Timestamp(eval_start)]
            if eval_end:
                bounded = bounded[bounded <= pd.Timestamp(eval_end)]
            if len(bounded) < max(horizon, 2) + 2:
                raise ValueError(
                    f"No dates fall in window [{eval_start or 'start'} .. {eval_end or 'end'}] "
                    f"with enough history to test horizon {horizon}: "
                    f"{len(bounded)} row(s) available.")
            _log("WARNING: Minimal fallback — using last-horizon test block.")
            te_end = bounded[-1]; te_start = bounded[-max(horizon, 2)]
            tr_end = bounded[-(max(horizon, 2) + 1)]
            folds_list = [(tr_end, te_start, te_end)]

    # ── The holdout read, recorded ────────────────────────────────────────────
    # A_STAT folds over every full year, so its evaluation reaches 2025 -- the sealed holdout --
    # on an ordinary run. That is legitimate: this module's own discipline says "TEST is run at
    # the end of a milestone to report what would have happened", and reporting is not choosing.
    # What was wrong is that it happened through NEITHER path: no gate, and no log entry, so the
    # holdout was being read on every daily run with nothing recording it.
    #
    # purpose="report" therefore records without raising. No selection guard is added here,
    # because this family makes no selection: it runs one model per invocation via
    # TG_MODEL_FILTER and its leaderboard ranks that model against the persistence baseline.
    _eval_dates = [t for (_tr, ts, te) in folds_list
                   for t in idx[(idx >= ts) & (idx <= te)]]
    _test_dates = [t for t in _eval_dates if _window_for(t) == "test"]
    if _test_dates:
        require_test_access(
            f"A_STAT reporting evaluation for {target!r} at h={horizon} covers "
            f"{len(_test_dates)} holdout target date(s) from {min(_test_dates).date()} to "
            f"{max(_test_dates).date()}",
            caller="run_a_stat.main", purpose=PURPOSE_REPORT)

    # ── Rolling-origin, h-step-ahead evaluation (C-3) ──
    # For every target date t in the fold's test window we refit the model on
    # the history up to the ORIGIN (t - h steps back) and take the h-step-ahead
    # forecast.  The origin therefore advances one step per target, exactly like
    # the ML family, and origin_value = y(t-h) is a true h-step persistence (not
    # a flat last-value), so the baseline matches every other family.
    idx_all = y_all.index
    pos_of = {ts: i for i, ts in enumerate(idx_all)}
    recs=[]
    for (tr_end, ts_start, ts_end) in folds_list:
        idx_te = idx_all[(idx_all >= ts_start) & (idx_all <= ts_end)]
        n_pi = 0
        for t in idx_te:
            pos_t = pos_of[t]
            origin_pos = pos_t - horizon
            if origin_pos < 0:
                continue  # not enough history to form an h-step forecast
            origin = idx_all[origin_pos]
            y_hist = y_all.iloc[:origin_pos + 1]              # observed up to the origin
            idx_future = idx_all[origin_pos + 1:pos_t + 1]    # h dates, last one == t
            if len(y_hist) == 0 or len(idx_future) != horizon:
                continue
            y_pred, y_lo, y_hi = _fc(model, y_hist, idx_future, ov, cadence)
            yp = float(np.asarray(y_pred).ravel()[-1])        # the h-step-ahead point
            lo = float(np.asarray(y_lo).ravel()[-1])
            hi = float(np.asarray(y_hi).ravel()[-1])
            if np.isfinite(lo):
                n_pi += 1
            recs.append({
                "date": t,
                "target_date": t,
                "origin_date": origin,                         # advances one step per target
                "origin_value": float(y_all.iloc[origin_pos]), # y(t-h) -> h-step persistence
                "target": target,
                "horizon": horizon,
                "horizon_note": f"stat_rolling_origin_h{horizon}",
                "model": model,
                "y_true": float(y_all.loc[t]), "y_pred": yp,
                "y_lo": lo, "y_hi": hi,
                "split_id": f"{ts_start.date()}..{ts_end.date()}",
                "cadence": cadence,
            })
        if n_pi:
            _log(f"  PI produced for fold {ts_start.date()}..{ts_end.date()} ({n_pi} values)")

    preds = pd.DataFrame.from_records(recs).sort_values("date")
    preds.to_csv(outroot/"predictions_long.csv", index=False)

    def _mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))
    def _rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    rows=[]
    for m,g in preds.groupby("model", sort=False):
        rows.append({"target":target,"horizon":horizon,"cadence":cadence,"model":m,
                     "MAE":_mae(g['y_true'],g['y_pred']),"RMSE":_rmse(g['y_true'],g['y_pred'])})
    metr=pd.DataFrame(rows); metr.to_csv(outroot/"metrics_long.csv", index=False)
    # The identity columns must be carried onto EVERY row. This groupby used to aggregate MAE
    # alone, dropping target/horizon/cadence -- and the persistence row below was concatenated
    # *with* them, so the baseline row was identified and the model rows were not. A consumer
    # asking "which model won for target X" got NaN for the winner. RMSE was dropped the same way,
    # which is why it read as an all-null column despite being computed in metrics_long.
    # NO selection guard here, deliberately. This family runs ONE model per invocation via
    # TG_MODEL_FILTER and never chooses between models: the leaderboard ranks that model
    # against the persistence baseline, which is a report, not a choice. A guard was added
    # here in this session's first pass and immediately refused an ordinary run, because
    # A_STAT legitimately evaluates over the reporting window. Ranking a model against a
    # ruler is not selecting a model. (Separately: this path reads 2025 rows without going
    # through require_test_access -- a pre-existing hole recorded in the session notes, not
    # something a selection guard should paper over.)
    lb=(metr.groupby("model",as_index=False)[["MAE","RMSE"]].mean().sort_values("MAE")
           .assign(target=target, horizon=horizon, cadence=cadence)
           .assign(rank=lambda x: np.arange(1,len(x)+1)))
    lb=lb[["target","horizon","cadence","model","MAE","RMSE","rank"]]

    # ✅ FIX STAT-2 / C-2: Persistence baseline (shared h-step ruler) + quality gate
    _stat_integrity = {"pipeline": "STAT", "target": target, "horizon": horizon}
    if not preds.empty and "origin_value" in preds.columns:
        from forecast_integrity import compute_persistence_baseline
        _valid = preds.dropna(subset=["origin_value", "y_true"])
        if len(_valid) > 0:
            mae_persist = compute_persistence_baseline(_valid)["mae_persistence"]
            mae_model_all = float(np.mean(np.abs(_valid["y_true"].values - _valid["y_pred"].values)))
            skill_pct = ((mae_persist - mae_model_all) / mae_persist * 100.0) if mae_persist > 0 else np.nan
            persist_row = pd.DataFrame([{"target":target,"horizon":horizon,"cadence":cadence,
                                         "model":"Persistence (baseline)","MAE":mae_persist,"RMSE":np.nan}])
            lb = pd.concat([persist_row, lb], ignore_index=True).sort_values("MAE").reset_index(drop=True)
            lb["rank"] = range(len(lb))
            _stat_integrity.update({
                "mae_model": mae_model_all, "mae_persistence": mae_persist,
                "skill_pct": skill_pct, "quality_gate_passed": skill_pct >= 5.0 if np.isfinite(skill_pct) else False,
                "run_status": "SUCCESS" if (np.isfinite(skill_pct) and skill_pct >= 5.0) else "FAILED_QUALITY",
            })
            _log(f"Persistence MAE={mae_persist:.2f}, Model MAE={mae_model_all:.2f}, Skill={skill_pct:.2f}%")

    lb.to_csv(outroot/"leaderboard.csv", index=False)

    # Save integrity report
    (outroot/"artifacts").mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(outroot/"artifacts"/"integrity_report.json", "w") as _f:
        _json.dump(_stat_integrity, _f, indent=2, default=str)

    # Item 1e: one provenance record per run, from the shared helper.
    from provenance import record_run, verify_expected_sha
    verify_expected_sha(data)
    record_run(outroot, "A_STAT", data,
               config={"model": model, "target": target, "cadence": cadence,
                       "horizon": horizon, "folds": folds, "min_train_years": minyrs,
                       "overrides": ov},
               date_col=dcol, seed=None,
               extra={"note": "A_STAT sets no random seed (audit m-1, still open)"})

    ops_series=None if _is_stock(target) else (ops_daily if cadence=="Daily" else ops_month)
    if not preds.empty:
        _plot_overlay(preds[preds["model"]==model], outroot/cadence.lower()/"plots"/f"{target.replace(' ','_')}_overlay.png", ops_series)
    plt.close("all")
    _log(f"DONE. Master outputs in: {outroot}")

if __name__ == "__main__":
    try:
        _log("===== Georgia A·Stat runner =====")
        main()
    except Exception as e:
        _log(f"ERROR: {e}")
        # See the C_DL runners: one report shape across all four families. TG_OUT_ROOT is
        # read from the environment rather than from `main`'s locals, because the failure
        # may have happened before `main` bound anything.
        try:
            from runner_errors import write_error_report
            write_error_report(os.environ.get("TG_OUT_ROOT", "outputs"), e, context="A_STAT")
        except Exception:
            pass
        raise
