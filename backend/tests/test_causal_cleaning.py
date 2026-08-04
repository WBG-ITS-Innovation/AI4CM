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
    _clean_flow_column_causally,
    check_business_day_coverage,
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


def test_imputed_values_never_enter_the_statistic_pool():
    """Otherwise one fabricated value shifts the reference used for the next.

    Fixture is built to discriminate: eight observed Mondays with a wide spread,
    then twenty consecutive missing Mondays. Every one imputes to the same median.
    If those twenty entered the clipping pool, the MAD would collapse toward zero
    (twenty identical values sitting at the centre) and the threshold would
    tighten enough to clip a later value that the observed-only pool accepts.
    """
    idx = pd.bdate_range("2020-01-06", periods=400)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(100.0, index=idx)
    observed = [60.0, 80.0, 100.0, 120.0, 140.0, 90.0, 110.0, 130.0]
    s.loc[mondays[:8]] = observed
    s.loc[mondays[8:28]] = np.nan          # twenty imputed Mondays, each -> 105.0
    probe = mondays[28]
    # Observed-only pool: median 105, MAD 20 -> accepts up to 105 + 8*20 = 265.
    # Pool including the twenty imputed 105s: MAD collapses to 0, the code falls
    # back to std (~13.4) -> accepts only up to ~212. A probe of 240 sits in that
    # gap, so it survives iff the imputed values were excluded from the pool.
    probe_value = 240.0
    s.loc[probe] = probe_value

    out, n_imputed, n_clipped, _ = _clean(s)

    assert n_imputed >= 20, f"expected the twenty gaps to impute, got {n_imputed}"
    med = float(np.median(observed))
    mad = float(np.median(np.abs(np.asarray(observed) - med)))
    assert probe_value <= med + 8 * mad, (
        "fixture no longer discriminates: the probe must sit INSIDE the "
        f"observed-only threshold ({med + 8 * mad})"
    )
    assert out.loc[probe] == probe_value, (
        f"probe was clipped to {out.loc[probe]}, so the twenty imputed values "
        f"tightened the threshold — they leaked into the statistic pool"
    )


# ── clipping guards ───────────────────────────────────────────────────────

def test_no_clipping_before_minimum_history_exists():
    """A median/MAD from two points is noise, not a threshold."""
    idx = pd.bdate_range("2020-01-06", periods=300)
    rng = np.random.default_rng(1)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(rng.normal(100.0, 5.0, len(idx)), index=idx)
    early = mondays[1]                     # only one prior Monday
    s.loc[early] = 1e6

    out, _, _, _ = _clean(s)
    assert out.loc[early] == 1e6, (
        "an extreme value was clipped with fewer than "
        f"{MIN_CLIP_HISTORY} prior same-weekday observations to judge it against"
    )


def test_clipping_fires_once_history_exists():
    idx = pd.bdate_range("2020-01-06", periods=300)
    rng = np.random.default_rng(1)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(rng.normal(100.0, 5.0, len(idx)), index=idx)
    late = mondays[MIN_CLIP_HISTORY + 3]
    s.loc[late] = 1e6

    out, _, n_clipped, _ = _clean(s)
    assert out.loc[late] < 1e6, "an extreme value survived despite sufficient history"
    assert n_clipped >= 1


def test_zero_spread_history_does_not_clip():
    """A constant history gives MAD = 0 and std = 0; there is no threshold to apply.

    Guards against a divide-by-noise regression that would clip everything to a
    constant the moment a metric has a flat stretch — and 24% of this workbook's
    cells sit in sparse line items that are flat for years at a time.
    """
    idx = pd.bdate_range("2020-01-06", periods=300)
    mondays = idx[idx.weekday == 0]
    s = pd.Series(100.0, index=idx)
    late = mondays[MIN_CLIP_HISTORY + 3]
    s.loc[late] = 1e6

    out, _, n_clipped, _ = _clean(s)
    assert n_clipped == 0 and out.loc[late] == 1e6


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
