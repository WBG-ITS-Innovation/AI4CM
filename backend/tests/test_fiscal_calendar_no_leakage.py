"""Workstream 3: the fiscal calendar must not read past the forecast origin.

The module's docstring claims every feature is knowable at the origin. A docstring is
not a control. These tests are, and the central one is
``test_future_target_values_cannot_change_past_features``: it mutates the target's
future and asserts the past features are byte-identical. That is the property that
matters, and it is the one a careless rolling window silently breaks.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.fiscal_calendar import (  # noqa: E402
    ALL_GROUPS,
    CALENDAR_ENTRIES,
    CALENDAR_ONLY_GROUPS,
    DISTANCE_CAP,
    GROUP_A,
    GROUP_D,
    GROUP_E,
    SERIES_GROUPS,
    build_fiscal_features,
    calendar_features,
    calendar_version,
    deadline_dates,
    drop_raw_year,
    is_business_day,
    series_features,
    shift_to_business_day,
)
from preprocessing.holidays import georgian_holidays_range  # noqa: E402

IDX = pd.bdate_range("2018-01-01", "2025-08-06")


def _y(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(1e8, 3e7, len(IDX)), index=IDX)


# ── the leak test that matters ─────────────────────────────────────────────────

def test_future_target_values_cannot_change_past_features():
    """Mutate the target's future; every feature before the cut must be unchanged.

    This is the operational definition of "no leakage" for groups D/E. A rolling window
    that forgot to shift, or an aligned lag that indexed forward, changes values here
    and nowhere else obvious.
    """
    y = _y()
    cut = len(y) // 2
    base = series_features(y, SERIES_GROUPS)

    tampered = y.copy()
    tampered.iloc[cut:] = tampered.iloc[cut:] * 1000.0 + 5e9
    after = series_features(tampered, SERIES_GROUPS)

    left_base, left_after = base.iloc[:cut], after.iloc[:cut]
    pd.testing.assert_frame_equal(left_base, left_after)

    # And the mutation must actually be visible after the cut, or the test proves nothing.
    assert not base.iloc[cut:].equals(after.iloc[cut:]), (
        "the tampering had no effect anywhere -- this test cannot detect leakage"
    )


def test_series_features_at_t_exclude_y_at_t():
    """lag_safety=1 means a feature at t uses t-1 and earlier.

    Checked directly: perturb exactly one observation and confirm the same-row features
    do not move.
    """
    y = _y()
    t = 500
    f0 = series_features(y, [GROUP_E])
    y2 = y.copy()
    y2.iloc[t] = y2.iloc[t] * 50.0
    f1 = series_features(y2, [GROUP_E])
    pd.testing.assert_series_equal(f0.iloc[t], f1.iloc[t], check_names=False)
    # t+1 must move -- otherwise the shift is too large and we are wasting information.
    assert not f0.iloc[t + 1].equals(f1.iloc[t + 1])


def test_aligned_lags_never_reference_a_future_row():
    y = pd.Series(np.arange(len(IDX), dtype=float), index=IDX)
    f = series_features(y, [GROUP_D])
    # y is the row number, so any value must be strictly below the row's own position.
    for col in ("y_aligned_prev_month", "y_aligned_prev_year"):
        v = f[col].to_numpy()
        pos = np.arange(len(v), dtype=float)
        ok = np.isnan(v) | (v < pos)
        assert ok.all(), f"{col} referenced a future row at {np.where(~ok)[0][:5]}"


def test_calendar_features_depend_only_on_the_index():
    """Groups A/B/C cannot leak because they never see the target.

    Verified by recomputing on a truncated index: shared dates must be identical, which
    also proves no column is normalised across the whole sample.
    """
    full = calendar_features(IDX, CALENDAR_ONLY_GROUPS)
    part = calendar_features(IDX[:800], CALENDAR_ONLY_GROUPS)
    pd.testing.assert_frame_equal(full.iloc[:800], part)


def test_series_features_are_not_normalised_across_the_sample():
    """A truncated series must give identical past values.

    Any global mean/std would make early rows depend on late data -- the scaling-leak
    class from the original review.
    """
    y = _y()
    full = series_features(y, SERIES_GROUPS)
    part = series_features(y.iloc[:900], SERIES_GROUPS)
    pd.testing.assert_frame_equal(full.iloc[:900], part)


# ── the shift rule (Tax Code Art. 3(6)) ───────────────────────────────────────

def test_shift_is_forward_only_and_lands_on_a_business_day():
    hol = {pd.Timestamp(d).date() for d in georgian_holidays_range(date(2023, 1, 1), date(2025, 12, 31))}
    for d in pd.date_range("2024-01-01", "2024-12-31"):
        dd = d.date()
        out = shift_to_business_day(dd, hol)
        assert out >= dd, "a deadline must never move earlier than the statutory date"
        assert is_business_day(out, hol)


def test_deadline_shift_actually_moves_some_months():
    """If the shift never fired, the calendar would add nothing over `dom`.

    2024 is the fixture: the 15th falls on a weekend or holiday in three months.
    """
    dd = deadline_dates(date(2024, 1, 1), date(2024, 12, 31))["monthly_15"]
    assert len(dd) == 12
    moved = [d for d in dd if d.day != 15]
    assert len(moved) == 3, f"expected 3 shifted months in 2024, got {[str(x) for x in moved]}"


def test_deadline_dates_are_all_business_days():
    hol = {pd.Timestamp(d).date() for d in georgian_holidays_range(date(2015, 1, 1), date(2026, 1, 1))}
    for kind, dates in deadline_dates(date(2015, 1, 1), date(2025, 8, 6)).items():
        for d in dates:
            assert is_business_day(d, hol), f"{kind} produced non-business day {d}"


# ── distances are capped, so they cannot encode the absolute date ─────────────

@pytest.mark.parametrize("col", ["bdays_to_deadline", "bdays_since_deadline",
                                 "days_to_holiday", "days_since_holiday"])
def test_distance_features_are_capped(col):
    """An uncapped counter lets a tree reconstruct position in the sample.

    That is the same failure mode as a raw `year` feature: it fits the trend rather than
    the mechanism, and cannot extrapolate.
    """
    f = calendar_features(IDX, CALENDAR_ONLY_GROUPS)
    assert f[col].min() >= 0
    assert f[col].max() <= DISTANCE_CAP


def test_no_raw_year_anywhere():
    f = build_fiscal_features(IDX, _y(), ALL_GROUPS)
    assert "year" not in f.columns
    # and the helper removes it if some other builder reintroduces it
    with_year = f.assign(year=IDX.year)
    assert "year" not in drop_raw_year(with_year).columns


def test_features_are_finite_where_defined():
    f = build_fiscal_features(IDX, _y(), ALL_GROUPS)
    for c in f.columns:
        v = f[c].to_numpy(dtype=float)
        assert not np.isinf(v).any(), f"{c} contains inf"


# ── provenance discipline ─────────────────────────────────────────────────────

def test_every_entry_carries_a_status_and_a_citation():
    for r in CALENDAR_ENTRIES:
        assert r.status in ("VERIFIED", "UNVERIFIED"), r.name
        assert r.source_tier in ("primary", "secondary", "none"), r.name
        assert r.citation.strip(), f"{r.name} has no citation text"


def test_unverified_entries_say_so_explicitly():
    """An UNVERIFIED entry must state that no source was found.

    The failure this prevents is an entry that looks sourced because the citation field
    is prose. Treasury signs off from docs/FISCAL_CALENDAR_SOURCES.md, which is generated
    from these fields, so silence here becomes a false claim there.
    """
    for r in CALENDAR_ENTRIES:
        if r.status == "UNVERIFIED":
            assert "NO SOURCE" in r.citation.upper(), (
                f"{r.name} is UNVERIFIED but its citation does not say so"
            )
            assert r.source_tier == "none"


def test_verified_entries_name_a_source():
    for r in CALENDAR_ENTRIES:
        if r.status == "VERIFIED":
            assert r.source_tier in ("primary", "secondary")
            assert "NO SOURCE" not in r.citation.upper()


def test_uncited_obligations_contribute_no_dates():
    """not_implemented entries must not invent dates.

    Salary, pension and debt-auction schedules are uncited. The honest handling is zero
    dates, not a plausible guess -- a guessed date that happens to help would be
    indistinguishable from a real finding.
    """
    ni = [r for r in CALENDAR_ENTRIES if r.kind == "not_implemented"]
    assert ni, "fixture assumes some obligations remain unimplemented"
    kinds = set(deadline_dates(date(2024, 1, 1), date(2024, 12, 31)))
    assert "not_implemented" not in kinds


def test_calendar_version_is_stable_and_content_sensitive():
    import dataclasses

    import preprocessing.fiscal_calendar as fc

    v1 = calendar_version()
    assert v1 == calendar_version(), "version must be deterministic"
    assert len(v1) == 12

    # Promoting an entry's status must change the version: the provenance of a result is
    # part of the result, so a re-run after Treasury sign-off must be distinguishable.
    orig = fc.CALENDAR_ENTRIES
    try:
        bumped = tuple(
            dataclasses.replace(r, status="VERIFIED", source_tier="primary")
            if r.name == "property_tax_individuals" else r
            for r in orig
        )
        fc.CALENDAR_ENTRIES = bumped
        assert calendar_version() != v1
    finally:
        fc.CALENDAR_ENTRIES = orig
    assert calendar_version() == v1


def test_groups_are_independently_selectable():
    """The ablation depends on this: asking for one group must yield only its columns."""
    a = build_fiscal_features(IDX, _y(), [GROUP_A])
    assert a.shape[1] > 0
    assert all(c.startswith(("is_deadline", "bdays_to_deadline", "bdays_since_deadline",
                             "deadline_shift")) for c in a.columns), list(a.columns)
    e = build_fiscal_features(IDX, _y(), [GROUP_E])
    assert all(c.startswith("y_") for c in e.columns)


def test_series_groups_require_the_target():
    with pytest.raises(ValueError, match="need the target"):
        build_fiscal_features(IDX, None, [GROUP_D])


# ── pipeline wiring ───────────────────────────────────────────────────────────

def test_e_quantile_no_longer_emits_raw_year():
    """Regression: E_QUANTILE was the last family carrying a raw `year` feature.

    A tree that splits on the calendar year puts every 2025 row into a terminal bucket
    learned from 2024, so it fits the trend rather than the mechanism and cannot
    extrapolate past the training range.
    """
    import e_quantile_daily_pipeline as eq

    cols = list(eq._calendar_feats(IDX).columns)
    assert "year" not in cols, cols
    assert {"dow", "dom", "week", "month"} <= set(cols)


def test_b_ml_default_feature_set_is_unchanged_without_fiscal_groups():
    """Passing no groups must reproduce the pre-WS3 frame exactly.

    Otherwise the workstream-1 numbers stop being comparable to anything measured after
    it, and the ablation's own baseline row would silently include calendar features.
    """
    from b_ml_pipeline import calendar_exog

    y = _y()
    base = calendar_exog(IDX)
    assert "is_deadline_any" not in base.columns
    with_groups = calendar_exog(IDX, y=y, fiscal_groups=ALL_GROUPS)
    assert "is_deadline_any" in with_groups.columns
    # the original columns survive untouched
    pd.testing.assert_frame_equal(base, with_groups[base.columns])


def test_both_families_accept_the_same_groups():
    """The module is shared; a group that works for one family must work for the other."""
    import e_quantile_daily_pipeline as eq
    from b_ml_pipeline import calendar_exog

    y = _y()
    a = calendar_exog(IDX, y=y, fiscal_groups=ALL_GROUPS)
    b = eq._calendar_feats(IDX, y=y, fiscal_groups=ALL_GROUPS)
    shared = set(a.columns) & set(b.columns)
    assert "is_deadline_any" in shared and "y_ewm_hl21" in shared
    for c in ("is_deadline_any", "deadline_shift_days", "bdays_to_eom"):
        pd.testing.assert_series_equal(a[c], b[c], check_names=False)
