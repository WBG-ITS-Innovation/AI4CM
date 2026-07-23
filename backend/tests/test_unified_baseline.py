"""Regression tests for audit findings C-2 and C-3: one baseline ruler.

C-2: every family must measure skill against the SAME h-step persistence
     (y_hat(t) = y(t-h)), computed by ONE shared function.
C-3: A_STAT must forecast at horizon h from rolling origins (origin advances
     one step per target), not a one-shot full-window flat forecast with a flat
     last-value baseline.

These tests build a shared synthetic flow series and check:
- the shared persistence function equals a plain-numpy h-step persistence;
- A_STAT's origin_date now advances daily and its baseline is the correct
  h-step persistence;
- A_STAT and B_ML, run on the same series/window, report the identical
  baseline MAE (float tolerance);
- no family carries its own inline persistence reimplementation any more.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import compute_persistence_baseline

HORIZON = 5


def _make_spiky_flow_csv(csv_path: Path) -> pd.Series:
    """Nonnegative, spiky daily flow series with true zeros; ~7 years."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-01", "2023-08-06", freq="B")
    base = 7.0e7
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 5.0)
    values = base * weekly * rng.normal(1.0, 0.12, size=len(dates))
    values[pd.Series(dates).dt.is_month_end.to_numpy()] *= 2.2   # end-of-month spikes
    values[rng.random(len(dates)) < 0.03] = 0.0                  # true zeros (holidays)
    values = np.maximum(values, 0.0)
    pd.DataFrame({"date": dates, "Revenues": values}).to_csv(csv_path, index=False)
    ser = pd.Series(values, index=pd.DatetimeIndex(dates))
    return ser


def _numpy_h_step_persistence(preds: pd.DataFrame) -> float:
    """Plain-numpy h-step persistence: mean|y_true - origin_value|."""
    v = preds.dropna(subset=["origin_value", "y_true"])
    return float(np.mean(np.abs(v["y_true"].values - v["origin_value"].values)))


# ---------------------------------------------------------------------------

def test_shared_persistence_matches_numpy(tmp_path):
    """The one shared function equals a hand-rolled h-step persistence."""
    ser = _make_spiky_flow_csv(tmp_path / "s.csv")
    idx = ser.index
    rows = []
    for pos in range(HORIZON, len(idx)):
        rows.append({
            "target_date": idx[pos],
            "origin_value": float(ser.iloc[pos - HORIZON]),  # y(t-h)
            "y_true": float(ser.iloc[pos]),
        })
    frame = pd.DataFrame(rows)
    shared = compute_persistence_baseline(frame)["mae_persistence"]
    manual = _numpy_h_step_persistence(frame)
    assert shared == manual, f"shared {shared} != numpy {manual}"


def _run_a_stat(csv_path: Path, out_dir: Path, model: str = "NAIVE"):
    import importlib
    env = {
        "TG_MODEL_FILTER": model, "TG_TARGET": "Revenues", "TG_CADENCE": "Daily",
        "TG_HORIZON": str(HORIZON), "TG_DATE_COL": "date",
        "TG_DATA_PATH": str(csv_path), "TG_OUT_ROOT": str(out_dir),
        "TG_PARAM_OVERRIDES": '{"folds":1,"min_train_years":4}',
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        import run_a_stat
        importlib.reload(run_a_stat)
        run_a_stat.main()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pd.read_csv(out_dir / "predictions_long.csv")


def test_a_stat_origin_advances_and_baseline_is_h_step(tmp_path):
    """C-3: origins advance daily; C-2: baseline is the correct h-step persistence."""
    csv = tmp_path / "s.csv"
    ser = _make_spiky_flow_csv(csv)
    preds = _run_a_stat(csv, tmp_path / "a_stat")

    # C-3: the origin is no longer a single fixed date.
    assert preds["origin_date"].nunique() > 1, "origin_date did not advance (still one-shot)"

    # Each row's origin is exactly h business-day steps before its target.
    idx = ser.index
    pos = {ts: i for i, ts in enumerate(idx)}
    od = pd.to_datetime(preds["origin_date"]); td = pd.to_datetime(preds["target_date"])
    steps = [pos[t] - pos[o] for o, t in zip(od, td)]
    assert all(s == HORIZON for s in steps), f"origins are not h={HORIZON} steps back: {set(steps)}"

    # C-2: A_STAT's baseline equals the shared function and a numpy h-step persistence.
    shared = compute_persistence_baseline(preds)["mae_persistence"]
    manual = _numpy_h_step_persistence(preds)
    assert abs(shared - manual) < 1e-6


def _run_b_ml(csv_path: Path, out_dir: Path, model: str = "Ridge"):
    from b_ml_pipeline import ConfigBML, run_pipeline_ml
    cfg = ConfigBML(
        data_path=str(csv_path), date_col="date", target="Revenues",
        cadence="Daily", horizon=HORIZON, variant="uni", model_filter=model,
        out_root=str(out_dir), folds=1, min_train_years=4,
    )
    run_pipeline_ml(cfg)
    return pd.read_csv(out_dir / "predictions_long.csv")


def test_a_stat_and_b_ml_report_identical_baseline(tmp_path):
    """Two families, one series/window -> identical baseline MAE (float tolerance)."""
    csv = tmp_path / "s.csv"
    _make_spiky_flow_csv(csv)
    a = _run_a_stat(csv, tmp_path / "a_stat")
    b = _run_b_ml(csv, tmp_path / "b_ml")

    # Compare on the target dates both families actually forecast.
    a_key = a.assign(target_date=pd.to_datetime(a["target_date"]))[["target_date", "origin_value", "y_true"]]
    b_one = b[b["model"] == b["model"].iloc[0]] if "model" in b.columns else b
    b_key = b_one.assign(target_date=pd.to_datetime(b_one["target_date"]))[["target_date", "origin_value", "y_true"]]
    common = sorted(set(a_key["target_date"]) & set(b_key["target_date"]))
    assert len(common) > 30, f"too few shared target dates ({len(common)})"

    a_c = a_key[a_key["target_date"].isin(common)].sort_values("target_date")
    b_c = b_key[b_key["target_date"].isin(common)].sort_values("target_date")
    mae_a = compute_persistence_baseline(a_c)["mae_persistence"]
    mae_b = compute_persistence_baseline(b_c)["mae_persistence"]
    assert abs(mae_a - mae_b) < 1e-6, f"A_STAT baseline {mae_a} != B_ML baseline {mae_b}"


def test_no_inline_persistence_reimplementations():
    """One ruler: every family routes through the shared function, none inlines it."""
    old_inline = 'np.abs(_valid["y_true"].values - _valid["origin_value"].values)'
    families = [
        "run_a_stat.py", "b_ml_pipeline.py",
        "e_quantile_daily_pipeline.py", "c_dl_pipeline.py",
    ]
    for fname in families:
        src = (BACKEND_DIR / fname).read_text()
        assert "compute_persistence_baseline" in src, f"{fname} does not use the shared function"
        assert old_inline not in src, f"{fname} still has an inline persistence reimplementation"
