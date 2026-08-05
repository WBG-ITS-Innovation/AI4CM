"""Forecast integrity and anti-leakage checks, against the SHARED module.

These tests previously exercised `preprocessing.integrity`, which carried a second
copy of the alignment check, the shift diagnostic and the h-step persistence
baseline. That module was retired in Phase 2 item 1c (decision D7) because
`b_ml_pipeline` merged its output over the shared module's via
`integrity_report.update(legacy_report)` -- so the numbers reaching consumers came
from the duplicate rather than from the implementation the tests guarded
(review §1.2).

The coverage is deliberately preserved rather than deleted: every test below asserts
the same property as its predecessor, now against `forecast_integrity`. Where the
shared API differs the assertion is adapted, not weakened -- noted per test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import (  # noqa: E402
    compute_persistence_baseline,
    compute_point_metrics,
    shift_diagnostic_horizon_aware,
    validate_alignment_step_based,
)


def _aligned_predictions(h=6, start="2024-01-01", end="2024-02-15"):
    """Predictions whose origin is exactly h index steps before the target."""
    dates = pd.bdate_range(start, end, freq="B")
    rows = []
    for i in range(h, len(dates) - h):
        rows.append({
            "origin_date": dates[i],
            "target_date": dates[i + h],
            "horizon": h,
            "y_true": 100.0 + i,
            "y_pred": 100.0 + i + (1.0 if i % 2 else -1.0),
            "origin_value": 100.0 + i - h,
            "model": "M",
        })
    return pd.DataFrame(rows), dates


# ── alignment ─────────────────────────────────────────────────────────────

def test_alignment_passes_when_origin_plus_h_equals_target():
    preds, dates = _aligned_predictions()
    result = validate_alignment_step_based(preds, dates, horizon=6)
    assert result["alignment_ok"] is True, result.get("misaligned_examples", [])
    assert result["n_misaligned"] == 0
    assert result["n_total"] == len(preds)


def test_alignment_detects_a_shifted_target():
    """One target moved by two steps must be caught."""
    preds, dates = _aligned_predictions()
    pos = {d: i for i, d in enumerate(dates)}
    bad = preds.copy()
    victim = bad.index[5]
    bad.loc[victim, "target_date"] = dates[pos[bad.loc[victim, "target_date"]] + 2]

    result = validate_alignment_step_based(bad, dates, horizon=6)
    assert result["alignment_ok"] is False
    assert result["n_misaligned"] > 0
    assert len(result["misaligned_examples"]) > 0


# ── shift diagnostic ──────────────────────────────────────────────────────

def test_shift_diagnostic_reports_zero_shift_on_aligned_predictions():
    """Adapted assertion.

    The retired `shift_sanity_check` exposed a boolean `lag_warning`. The shared
    `shift_diagnostic_horizon_aware` instead returns an `interpretation` string plus
    `is_lag0_issue` / `is_persistence_like` flags, which is strictly more
    informative. Asserting the interpretation starts with "OK" is the same claim.
    """
    rng = np.random.default_rng(0)
    y_true = np.cumsum(rng.normal(0, 1, 200)) + 100.0
    y_pred = y_true + rng.normal(0, 0.5, 200)

    r = shift_diagnostic_horizon_aware(y_true, y_pred, horizon=6)
    assert r["best_shift"] == 0, r["best_shift"]
    assert r["interpretation"].startswith("OK"), r["interpretation"]
    assert r["is_lag0_issue"] is False
    assert r["mae_shift0"] > 0


def test_shift_diagnostic_detects_predictions_that_are_a_lagged_copy():
    h = 6
    rng = np.random.default_rng(1)
    y_true = np.cumsum(rng.normal(0, 1, 300)) + 100.0
    y_pred = np.roll(y_true, h)          # predictions are the target, h steps stale
    y_pred[:h] = y_true[0]

    r = shift_diagnostic_horizon_aware(y_true, y_pred, horizon=h)
    assert r["best_shift"] <= -h + 2, r["best_shift"]
    assert r["improvement_ratio"] < 0.90, (
        f"shifting improved MAE by only {1 - r['improvement_ratio']:.1%}; "
        f"a lagged copy should improve far more"
    )
    assert r["is_persistence_like"] or r["is_lag0_issue"], r["interpretation"]


# ── persistence baseline ──────────────────────────────────────────────────

def test_persistence_baseline_from_origin_value():
    preds, _ = _aligned_predictions()
    r = compute_persistence_baseline(preds)
    assert not np.isnan(r["mae_persistence"])
    assert r["mae_persistence"] > 0
    assert r["n_valid"] == len(preds)


def test_persistence_baseline_ignores_rows_with_missing_inputs():
    preds, _ = _aligned_predictions()
    preds.loc[preds.index[:3], "origin_value"] = np.nan
    r = compute_persistence_baseline(preds)
    assert r["n_valid"] == len(preds) - 3


# ── the full field set the Dashboard reads ────────────────────────────────

def test_shared_helpers_supply_every_field_the_dashboard_reads():
    """Guards the item-1c surgery at the level that matters.

    Retiring `compute_integrity_report` removed the code that produced six
    Dashboard-only fields. They are now assembled from the shared helpers in
    `b_ml_pipeline`; this asserts the helpers can in fact supply them, so a
    regression shows up here rather than as an empty panel in the UI.
    """
    preds, dates = _aligned_predictions()
    align = validate_alignment_step_based(preds, dates, horizon=6)
    shift = shift_diagnostic_horizon_aware(
        preds["y_true"].values, preds["y_pred"].values, horizon=6)
    persist = compute_persistence_baseline(preds)
    model = compute_point_metrics(preds["y_true"], preds["y_pred"])
    persist_pts = compute_point_metrics(preds["y_true"], preds["origin_value"])

    report = {
        "alignment_ok": align["alignment_ok"],
        "misaligned_examples": align["misaligned_examples"],
        "best_shift": shift["best_shift"],
        "lag_warning": bool(shift["best_shift"] != 0 and shift["improvement_pct"] > 10.0),
        "mae_model": model["mae"],
        "rmse_model": model["rmse"],
        "r2_model": model["r2"],
        "mae_persistence": persist["mae_persistence"],
        "rmse_persistence": persist_pts["rmse"],
        "r2_persistence": persist_pts["r2"],
        "horizon": 6,
    }

    for field in ("alignment_ok", "misaligned_examples", "best_shift", "lag_warning",
                  "mae_model", "rmse_model", "r2_model", "mae_persistence",
                  "rmse_persistence", "r2_persistence", "horizon"):
        assert field in report, f"{field} cannot be built from the shared helpers"

    # rmse_persistence must agree between the two routes that can produce it.
    assert persist_pts["rmse"] == pytest.approx(persist["rmse_persistence"], rel=1e-12)
    assert report["mae_model"] > 0 and not np.isnan(report["r2_model"])


def test_point_metrics_r2_is_nan_rather_than_dividing_by_zero():
    r = compute_point_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert np.isnan(r["r2"]), "a zero-variance target must give NaN R2, not a crash"
    assert r["mae"] > 0


# ── date arithmetic (unchanged; never used the retired module) ─────────────

def test_training_data_boundary():
    """Model training rows must end at origin_date, and target == origin + h steps.

    Step-based (positional) indexing: dates[i + h] is the target for an origin at
    dates[i]. Calendar-day arithmetic does not work for a business-day series.
    """
    dates = pd.bdate_range("2024-01-01", "2024-02-15", freq="B")
    h = 6

    for i in range(h, len(dates) - h):
        origin_date = dates[i]
        target_date = dates[i + h]
        train_dates = dates[: i + 1]

        assert train_dates[-1] == origin_date
        assert target_date == dates[i + h], (
            f"step-based target mismatch at i={i}: expected "
            f"dates[{i + h}]={dates[i + h].date()}, got {target_date.date()}"
        )
        expected_target = origin_date + pd.offsets.BDay(h)
        assert expected_target == target_date, (
            f"BDay offset mismatch at i={i}: origin={origin_date.date()} + {h}BDay "
            f"= {expected_target.date()}, but target={target_date.date()}"
        )
