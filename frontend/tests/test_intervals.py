"""Interval detection must read the advertised level from the artifact, never assume it.

These pin the two defects this module was written to remove: E_QUANTILE's columns being
invisible, and a correctly calibrated 80% band being scored against a hard-coded 90%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from intervals import (  # noqa: E402
    TERCILE_LABELS,
    calibration_verdict,
    coverage_by_model,
    coverage_by_tercile,
    detect_intervals,
    reliability_curve,
)


def _equantile(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = 1e8 + 2e7 * rng.normal(0, 1, n)
    return pd.DataFrame({
        "model": ["GBQuantile"] * n, "y_true": y,
        "yhat_p10": y - 3e7, "yhat_p50": y, "yhat_p90": y + 3e7,
    })


def _bml(n=200, seed=1):
    rng = np.random.default_rng(seed)
    y = 1e8 + 2e7 * rng.normal(0, 1, n)
    return pd.DataFrame({"model": ["LightGBM_L1"] * n, "y_true": y,
                         "y_pred": y, "y_lo": y - 3e7, "y_hi": y + 3e7})


# ── the two defects ───────────────────────────────────────────────────────────

def test_e_quantile_columns_are_detected():
    """Regression: the old check looked only for y_lo/y_hi, so the quantile family -- whose
    entire purpose is intervals -- rendered an empty panel."""
    spec = detect_intervals(_equantile())
    assert spec is not None, "E_QUANTILE interval columns were not detected"
    assert (spec.lo, spec.hi, spec.mid) == ("yhat_p10", "yhat_p90", "yhat_p50")


def test_nominal_is_read_from_the_quantile_names_not_hardcoded():
    """p10..p90 advertises 80%, not 90%. Scoring it against 90% called a good band broken."""
    spec = detect_intervals(_equantile())
    assert spec.nominal == pytest.approx(0.80)
    assert spec.nominal_known is True
    assert "p10" in spec.nominal_source and "p90" in spec.nominal_source


def test_a_well_calibrated_80pc_band_passes():
    spec = detect_intervals(_equantile())
    ok, why = calibration_verdict(0.80, spec.nominal, n=200)
    assert ok is True, why
    # ...and the same band judged against 90% would have failed, which was the old behaviour
    bad, _ = calibration_verdict(0.80, 0.90, n=200)
    assert bad is False


def test_generic_columns_report_the_level_as_unknown_not_as_90():
    """B_ML writes y_lo/y_hi with no advertised level (ConfigBML.nominal_pi is never
    persisted). Guessing 90% here is what produced a confident verdict about nothing."""
    spec = detect_intervals(_bml())
    assert spec is not None
    assert (spec.lo, spec.hi) == ("y_lo", "y_hi")
    assert spec.nominal is None
    assert spec.nominal_known is False
    assert "does not record" in spec.nominal_source


def test_unknown_nominal_yields_never_verified_not_a_pass():
    state, why = calibration_verdict(0.83, None, n=500)
    assert state is None, "an unknown advertised level must not produce a pass"
    assert "not recorded" in why


def test_small_sample_yields_never_verified_not_a_pass():
    state, why = calibration_verdict(0.80, 0.80, n=12)
    assert state is None
    assert "too few" in why


def test_no_interval_columns_returns_none():
    assert detect_intervals(pd.DataFrame({"y_true": [1.0], "y_pred": [1.0]})) is None


# ── the measurements ──────────────────────────────────────────────────────────

def test_coverage_by_model_counts_correctly():
    df = _equantile(n=100)
    df.loc[:19, "y_true"] = df.loc[:19, "yhat_p90"] + 1.0     # 20 rows outside
    spec = detect_intervals(df)
    out = coverage_by_model(df, spec)
    assert len(out) == 1
    assert out.iloc[0]["coverage"] == pytest.approx(0.80)
    assert out.iloc[0]["n"] == 100


def test_tercile_coverage_exposes_a_band_that_misses_the_big_days():
    """The headline defect: good average coverage, bad coverage where it matters."""
    # Magnitudes must be CONTINUOUS or qcut cannot form three buckets -- a two-value
    # fixture made the split degenerate and the assertion hit an empty frame instead of
    # demonstrating anything.
    n = 300
    y = np.linspace(1e8, 9e8, n)
    p50 = y.copy()
    p50[200:] = y[200:] * 0.5          # the largest third is badly predicted
    half = 0.05e8                       # ...and the band is narrow, so those days are missed
    df = pd.DataFrame({"model": ["m"] * n, "y_true": y,
                       "yhat_p10": p50 - half, "yhat_p50": p50, "yhat_p90": p50 + half})
    spec = detect_intervals(df)
    overall = coverage_by_model(df, spec).iloc[0]["coverage"]
    terc = coverage_by_tercile(df, spec)
    assert overall > 0.6, "fixture should look acceptable overall"
    biggest = terc[terc["tercile"] == TERCILE_LABELS[-1]].iloc[0]["coverage"]
    assert biggest == pytest.approx(0.0), (
        "tercile split must expose that the largest days are entirely missed"
    )


def test_tercile_returns_empty_rather_than_guessing_on_degenerate_data():
    df = pd.DataFrame({"model": ["m"] * 8, "y_true": [5.0] * 8,
                       "yhat_p10": [4.0] * 8, "yhat_p50": [5.0] * 8, "yhat_p90": [6.0] * 8})
    spec = detect_intervals(df)
    assert coverage_by_tercile(df, spec).empty


def test_reliability_curve_reports_mass_outside_the_band():
    df = _equantile(n=100)
    df.loc[:9, "y_true"] = df.loc[:9, "yhat_p90"] + 1.0    # 10 above
    df.loc[10:19, "y_true"] = df.loc[10:19, "yhat_p10"] - 1.0   # 10 below
    spec = detect_intervals(df)
    rc = reliability_curve(df, spec)
    below = rc[rc["position"] == "below band"].iloc[0]
    above = rc[rc["position"] == "above band"].iloc[0]
    assert below["n"] == 10 and above["n"] == 10
    assert below["share"] == pytest.approx(0.10)
    # shares over all rows sum to 1
    assert rc["share"].sum() == pytest.approx(1.0, abs=1e-9)


def test_reliability_curve_empty_on_zero_width_bands():
    df = pd.DataFrame({"model": ["m"] * 20, "y_true": [1.0] * 20,
                       "yhat_p10": [1.0] * 20, "yhat_p50": [1.0] * 20,
                       "yhat_p90": [1.0] * 20})
    spec = detect_intervals(df)
    assert reliability_curve(df, spec).empty
