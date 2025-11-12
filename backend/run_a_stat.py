# run_a_stat.py — FINAL (Statistical models + Treasury baseline + robust folds)

from __future__ import annotations

import json, os, sys, time, warnings
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

def _is_stock(name: str) -> bool:
    return str(name).strip().lower() in {"state budget balance", "balance", "net", "stock"}

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

def _ops_monthly_baseline(series_daily: pd.Series, years: int = 3) -> pd.Series:
    """Monthly Treasury baseline from flows (3y annual mean × month share)."""
    m = series_daily.resample("ME").sum().astype(float)
    if m.empty: return m
    df = m.to_frame("val")
    df["year"] = df.index.year; df["month"] = df.index.month
    counts = df.groupby("year")["val"].size()
    full_years = counts.index[counts.eq(12)].tolist()

    out = []
    for ts, mo, Y in zip(df.index, df["month"], df["year"]):
        prev = [Y - k for k in range(1, years + 1)]
        if not all(py in full_years for py in prev):
            out.append(np.nan); continue
        annual_prev = df[df["year"].isin(prev)].groupby("year")["val"].sum().reindex(prev)
        annual_mean = annual_prev.mean()
        mon_vals = [df.loc[(df["year"] == py) & (df["month"] == mo), "val"].iloc[0] for py in prev]
        shares = [mv / annual_prev.loc[py] if annual_prev.loc[py] > 0 else np.nan for mv, py in zip(mon_vals, prev)]
        share_mean = np.nanmean(shares)
        out.append(annual_mean * share_mean if np.isfinite(share_mean) else np.nan)
    base = pd.Series(out, index=m.index).ffill()
    base.index.freq = "ME"
    return base

def _ops_daily_from_monthly(daily_hist: pd.Series, monthly_forecast: pd.Series) -> pd.Series:
    """Distribute monthly baseline to business days using recent daily profiles."""
    pieces: List[pd.Series] = []
    for ts, mv in monthly_forecast.dropna().items():
        days = pd.date_range(ts.replace(day=1), ts, freq="B")
        if not len(days): continue
        profiles = []
        for k in (1, 2, 3):
            d0 = (ts - pd.DateOffset(years=k)).replace(day=1)
            d1 = (ts - pd.DateOffset(years=k))
            hist = daily_hist[(daily_hist.index >= d0) & (daily_hist.index <= d1)].reindex(
                pd.date_range(d0, d1, freq="B"), fill_value=0.0
            )
            if hist.sum() > 0:
                p = (hist / hist.sum()).reindex(days, fill_value=0.0).values
                profiles.append(p)
        if profiles:
            prof = np.mean(np.vstack(profiles), axis=0)
            prof = prof / (prof.sum() if prof.sum() > 0 else 1.0)
        else:
            prof = np.ones(len(days)) / len(days)
        pieces.append(pd.Series(float(mv) * prof, index=days))
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)

def _yearly_folds(idx: pd.DatetimeIndex, min_years: int, want_folds: int) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    years = sorted(set(idx.year))
    folds=[]
    for Y in years:
        if Y - years[0] < min_years: 
            continue
        tr_end_cand = idx[idx <= pd.Timestamp(f"{Y-1}-12-31")]
        if tr_end_cand.empty: continue
        tr_end = tr_end_cand[-1]
        ts_span = idx[(idx >= pd.Timestamp(f"{Y}-01-01")) & (idx <= pd.Timestamp(f"{Y}-12-31"))]
        if ts_span.empty: continue
        folds.append((tr_end, ts_span[0], ts_span[-1]))
    if want_folds>0 and len(folds)>want_folds:
        folds = folds[-want_folds:]
    return folds

def _fallback_fold(idx: pd.DatetimeIndex, horizon: int) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    """Used when yearly folds cannot be built; always return at least one fold if there is enough history to test `horizon`."""
    n = len(idx)
    if n <= horizon + 5: 
        return []
    te_len = max(horizon, min(12, n//4))
    te_end = idx[-1]; te_start = idx[-te_len]
    tr_end = idx[-(te_len + horizon)]
    return [(tr_end, te_start, te_end)]

# predictors
def _fc(model: str, y_tr: pd.Series, idx: pd.DatetimeIndex, ov: Dict, cadence: str) -> np.ndarray:
    m = model.upper()
    if m == "NAIVE":
        return np.repeat(y_tr.iloc[-1], len(idx)).astype(float)
    if m == "WEEKDAY_MEAN":
        if y_tr.index.freqstr and "B" in y_tr.index.freqstr:
            wd = y_tr.groupby(y_tr.index.dayofweek).mean()
            return np.array([wd.get(ts.dayofweek, y_tr.iloc[-1]) for ts in idx], dtype=float)
        return np.repeat(y_tr.iloc[-1], len(idx)).astype(float)
    if m == "MOVAVG":
        w = int(ov.get("MOVAVG", {}).get("window", 7))
        return np.repeat(float(y_tr.tail(w).mean()), len(idx)).astype(float)
    if m == "ETS":
        ets = ov.get("ETS", {})
        trend = None if ets.get("trend") in (None, "None") else ets.get("trend", "add")
        seasonal = None if ets.get("seasonal") in (None, "None") else ets.get("seasonal", "add")
        periods = int(ets.get("seasonal_periods", 12))
        damped = bool(ets.get("damped_trend", False))
        # safety: need at least two seasonal cycles if seasonal is requested
        if seasonal and len(y_tr) < 2 * periods:
            seasonal = None
        try:
            fit = ExponentialSmoothing(y_tr, trend=trend, seasonal=seasonal,
                                       seasonal_periods=periods if seasonal else None,
                                       damped_trend=damped, initialization_method="estimated").fit(optimized=True)
            return fit.forecast(len(idx)).values.astype(float)
        except Exception:
            # final fallback
            return np.repeat(y_tr.iloc[-1], len(idx)).astype(float)
    if m == "SARIMAX":
        sar = ov.get("SARIMAX", {})
        ord_ = tuple(sar.get("order", [1,1,1]))
        sOrd = tuple(sar.get("seasonal_order", [0,0,0, 12 if cadence=='Monthly' else 5]))
        fit = SARIMAX(y_tr, order=ord_, seasonal_order=sOrd,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return fit.forecast(len(idx)).values.astype(float)
    if m == "STL_ARIMA":
        stl = ov.get("STL_ARIMA", {})
        sp = int(stl.get("stl_period", 12 if cadence=='Monthly' else 5))
        ao = tuple(stl.get("arima_order", [1,1,0]))
        comp = STL(y_tr, period=sp, robust=True).fit()
        fit = SARIMAX(comp.resid, order=ao, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        res = fit.forecast(len(idx)).values
        seas = np.resize(comp.seasonal[-sp:], len(idx)) if sp>0 and len(comp.seasonal)>=sp else np.zeros(len(idx))
        trend = np.full(len(idx), comp.trend.iloc[-1])
        return trend + seas + res
    if m == "THETA":
        return ThetaModel(y_tr).fit().forecast(len(idx)).values.astype(float)
    return np.repeat(y_tr.iloc[-1], len(idx)).astype(float)

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
    folds   = int(ov.get("folds", 3))
    minyrs  = int(ov.get("min_train_years", 4))
    demo    = ov.get("demo_clip_months")

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
    folds_list = _yearly_folds(idx, minyrs, folds)
    if not folds_list:
        _log("WARNING: Not enough full-year coverage; using recent sliding-window fold.")
        folds_list = _fallback_fold(idx, horizon)
        if not folds_list:
            # last-ditch: naive test on last horizon
            _log("WARNING: Minimal fallback — using last-horizon test block.")
            te_end = idx[-1]; te_start = idx[-max(horizon, 2)]
            tr_end = idx[-(max(horizon, 2) + 1)]
            folds_list = [(tr_end, te_start, te_end)]

    recs=[]
    for (tr_end, ts_start, ts_end) in folds_list:
        y_tr = y_all[y_all.index <= tr_end]
        idx_te = y_all.index[(y_all.index >= ts_start) & (y_all.index <= ts_end)]
        if len(y_tr)==0 or len(idx_te)==0: continue
        y_pred = _fc(model, y_tr, idx_te, ov, cadence)
        y_pred = np.asarray(y_pred).ravel()
        if len(y_pred)!=len(idx_te): y_pred = np.resize(y_pred, len(idx_te))
        y_true = y_all.reindex(idx_te).values.astype(float)
        for dt_i, yp, yt in zip(idx_te, y_pred, y_true):
            recs.append({"date":dt_i,"target":target,"horizon":horizon,"model":model,
                         "y_true":float(yt),"y_pred":float(yp),"y_lo":np.nan,"y_hi":np.nan,
                         "split_id":f"{tr_end.date()}→{ts_start.date()}..{ts_end.date()}","cadence":cadence})

    preds = pd.DataFrame.from_records(recs).sort_values("date")
    preds.to_csv(outroot/"predictions_long.csv", index=False)

    def _mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))
    def _rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    rows=[]
    for m,g in preds.groupby("model", sort=False):
        rows.append({"target":target,"horizon":horizon,"cadence":cadence,"model":m,
                     "MAE":_mae(g['y_true'],g['y_pred']),"RMSE":_rmse(g['y_true'],g['y_pred'])})
    metr=pd.DataFrame(rows); metr.to_csv(outroot/"metrics_long.csv", index=False)
    lb=(metr.groupby("model",as_index=False)["MAE"].mean().sort_values("MAE")
           .assign(rank=lambda x: np.arange(1,len(x)+1)))
    lb.to_csv(outroot/"leaderboard.csv", index=False)

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
        raise
