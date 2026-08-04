"""Step 1: `clean_treasury` must fit its cleaning only on data from before each row.

The pre-Phase-1 implementation computed both statistics from the whole series::

    ref  = dow_values.tail(weekday_weeks).median()   # last N of all 11 years
    med  = dow_values.median(); mad = ...            # every occurrence

``.tail(N)`` takes the most RECENT N occurrences, so a 2016 gap was filled with
the median of eight Mondays in 2025, and the clipping thresholds applied to 2016
were computed partly from the locked 2025 holdout.

Measured before the fix (review §2.3): 223 of 2,763 business-day Revenues values
(8.1%) were touched by holdout-informed statistics — 118 imputed, 105 clipped —
20 of them inside the 2025 test window, with a median clipping change of
118,342,253, roughly three times the best model's MAE. Because the cleaning also
feeds the persistence baseline, this moved the yardstick every model is graded
against.

The property that makes this safe is simple and testable: **no value dated at or
after row t may influence row t.**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.preprocess import (  # noqa: E402
    KNOWN_ABSENT_BUSINESS_DAYS,
    MIN_CLIP_HISTORY,
    SUSPECT_JUMP_MULTIPLE,
    _clean_flow_column_causally,
    check_business_day_coverage,
    flow_validity_report,
)

WEEKDAY_WEEKS = 8


def _all_business(idx):
    return pd.Series(True, index=idx)


def _clean(series, weekday_weeks=WEEKDAY_WEEKS):
    return _clean_flow_column_causally(
        series.copy(), _all_business(series.index), weekday_weeks=weekday_weeks
    )


# ── the core property ─────────────────────────────────────────────────────

def test_a_later_value_cannot_change_an_earlier_row():
    """Perturbing row 150 must leave rows 0..149 bit-identical.

    This is the single assertion that would have caught the original bug: with
    `.tail(N)` on the whole series, changing a 2025 value changed what a 2016 gap
    was filled with.
    """
    idx = pd.bdate_range("2020-01-06", periods=200)
    rng = np.random.default_rng(0)
    base_values = pd.Series(rng.normal(100.0, 5.0, len(idx)), index=idx)
    base_values.iloc[40] = np.nan          # an early gap

    clean_base, _, _, _ = _clean(base_values)

    perturbed = base_values.copy()
    perturbed.iloc[150] = 1e9              # a huge value much later
    clean_pert, _, _, _ = _clean(perturbed)

    delta = (clean_base.iloc[:150] - clean_pert.iloc[:150]).abs().max()
    assert delta == 0.0, (
        f"a value at position 150 changed earlier rows by up to {delta} — "
        f"the cleaning is not causal"
    )


def test_imputation_uses_prior_same_weekday_values_not_later_ones():
    idx = pd.bdate_range("2020-01-06", periods=60)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(np.nan, index=idx)
    s.loc[mondays[:5]] = [10.0, 12.0, 14.0, 16.0, 18.0]
    s.loc[mondays[5]] = np.nan             # the gap
    s.loc[mondays[6:]] = 999.0             # later Mondays are wildly different

    out, n_imputed, _, dates = _clean(s)

    assert out.loc[mondays[5]] == 14.0, (
        f"gap filled with {out.loc[mondays[5]]}; expected the median of the five "
        f"PRIOR Mondays (14.0). A value near 999 means later data was used."
    )
    assert n_imputed >= 1
    assert str(mondays[5].date()) in dates, "imputed dates must be reported"


def test_imputed_values_never_enter_the_imputation_reference():
    """Otherwise one fabricated value shifts the reference used for the next.

    Eight observed Mondays, then a long run of missing ones. Every gap must be
    filled from the eight OBSERVED values. If imputed values re-entered the
    rolling window, the reference would drift as the window filled with copies of
    itself, and the later gaps would differ from the earlier ones.
    """
    idx = pd.bdate_range("2020-01-06", periods=400)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(100.0, index=idx)
    observed = [60.0, 80.0, 100.0, 120.0, 140.0, 90.0, 110.0, 130.0]
    s.loc[mondays[:8]] = observed
    s.loc[mondays[8:28]] = np.nan          # twenty consecutive gaps

    out, n_imputed, _, _ = _clean(s)

    assert n_imputed >= 20, f"expected the twenty gaps to impute, got {n_imputed}"
    expected = float(np.median(observed))
    filled = out.loc[mondays[8:28]].to_numpy()
    assert np.allclose(filled, expected), (
        f"imputed values drifted ({filled[:5]} ... vs a constant {expected}) — "
        f"fabricated values are re-entering the rolling reference window"
    )


# ── observed values are never rewritten ───────────────────────────────────

def test_month_end_spikes_are_left_alone():
    """The spikes are the signal, not noise.

    Causal MAD clipping was tried and abandoned on measurement: any causally
    estimated same-weekday pool frequently excludes the month-end and tax-deadline
    spikes, so MAD comes out small, 8*MAD is tight, and the spikes get clipped. On
    the real workbook that suppressed the 2024 mean of Revenues by 41%
    (98,123,411 -> 58,213,784) and touched 216 of 2,763 business days.

    The old whole-series version survived only BECAUSE it was leaky — including the
    spikes in the pool inflated MAD into a generous threshold. Fixing the causality
    exposed that MAD clipping was never appropriate here.
    """
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(7.0e7, 8.0e6, len(idx)), index=idx)
    month_ends = idx[idx.to_series().dt.is_month_end.to_numpy()]
    s.loc[month_ends] *= 3.0               # realistic month-end multiple

    out, _, n_clipped, _ = _clean(s)

    assert n_clipped == 0, f"{n_clipped} observed values were rewritten"
    observed = s.notna()
    assert np.allclose(out[observed].to_numpy(), s[observed].to_numpy()), (
        "observed flow values must pass through unchanged"
    )


def test_extreme_values_pass_through_unaltered():
    """Even a 1e6-sized outlier is reported, not silently rewritten."""
    idx = pd.bdate_range("2020-01-06", periods=300)
    rng = np.random.default_rng(1)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(rng.normal(100.0, 5.0, len(idx)), index=idx)
    late = mondays[MIN_CLIP_HISTORY + 3]
    s.loc[late] = 1e6

    out, _, n_clipped, _ = _clean(s)
    assert out.loc[late] == 1e6
    assert n_clipped == 0


# ── validity reporting: flag, do not alter ────────────────────────────────

def test_validity_report_flags_an_order_of_magnitude_jump():
    idx = pd.bdate_range("2020-01-06", periods=200)
    s = pd.Series(100.0, index=idx)
    spike = idx[100]
    s.loc[spike] = 100.0 * SUSPECT_JUMP_MULTIPLE * 2

    rep = flow_validity_report(s, _all_business(idx))
    assert rep["n_suspect_jumps"] >= 1
    assert any(j["date"] == str(spike.date()) for j in rep["suspect_jumps"])


def test_validity_report_flags_negative_flows():
    idx = pd.bdate_range("2020-01-06", periods=50)
    s = pd.Series(100.0, index=idx)
    s.iloc[10] = -5.0
    rep = flow_validity_report(s, _all_business(idx))
    assert rep["n_negative"] == 1
    assert str(idx[10].date()) in rep["negative_dates"]


def test_validity_report_does_not_flag_a_realistic_month_end_spike():
    """A 3x month-end multiple must not be reported as a data error."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(7.0e7, 8.0e6, len(idx)), index=idx)
    s.loc[idx[idx.to_series().dt.is_month_end.to_numpy()]] *= 3.0

    rep = flow_validity_report(s, _all_business(idx))
    assert rep["n_suspect_jumps"] == 0, rep["suspect_jumps"][:3]
    assert rep["n_negative"] == 0


def test_gap_with_no_prior_history_is_left_nan_not_invented():
    """Better an honest NaN than a number borrowed from a different weekday."""
    idx = pd.bdate_range("2020-01-06", periods=40)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(50.0, index=idx)
    s.loc[mondays[0]] = np.nan             # the very first Monday

    out, n_imputed, _, _ = _clean(s)
    assert pd.isna(out.loc[mondays[0]]), (
        f"first Monday was filled with {out.loc[mondays[0]]} despite no prior "
        f"Monday existing to fill it from"
    )
    assert n_imputed == 0


# ── coverage check ────────────────────────────────────────────────────────

def test_full_coverage_passes_and_reports_counts():
    reported = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-29"))
    cov = check_business_day_coverage(reported, holidays=set(), allow_list=())
    assert cov["absent_business_days"] == 0
    assert cov["reported_business_days"] == cov["expected_business_days"]
    assert cov["unexplained_absent"] == []


def test_unexplained_gap_aborts():
    """Silently zero-filling this moved the persistence baseline 10.1% (review §7.1)."""
    reported = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-29"))
    with_gap = reported.delete(30)
    with pytest.raises(ValueError, match="missing from the source"):
        check_business_day_coverage(with_gap, holidays=set(), allow_list=())


def test_gap_explained_by_a_public_holiday_is_accepted():
    reported = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-29"))
    missing = reported[30]
    cov = check_business_day_coverage(
        reported.delete(30), holidays={missing}, allow_list=()
    )
    assert cov["absent_but_holiday"] == 1
    assert cov["unexplained_absent"] == []


def test_gap_on_the_allow_list_is_accepted():
    reported = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-29"))
    missing = reported[30]
    cov = check_business_day_coverage(
        reported.delete(30), holidays=set(), allow_list=(str(missing.date()),)
    )
    assert cov["absent_but_allow_listed"] == 1


def test_the_one_real_historical_gap_is_allow_listed_with_a_reason():
    """2018-11-28 is the single business day absent from the workbook that is not
    a Georgian public holiday (117 of 118 absences are holidays). It must stay on
    the list or historical regeneration cannot run — and the list must stay short,
    so a NEW gap can never hide among known ones."""
    assert "2018-11-28" in KNOWN_ABSENT_BUSINESS_DAYS
    assert len(KNOWN_ABSENT_BUSINESS_DAYS) == 1, (
        "the allow-list grew; each entry must be a reviewed exception, not a "
        "convenient way to silence the coverage check"
    )
