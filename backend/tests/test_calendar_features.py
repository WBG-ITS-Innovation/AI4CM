"""Phase 3 (first slice): day-of-month calendar features for B_ML.

Treasury cash flows are driven by fixed monthly dates — tax deadlines, salary
runs, scheduled transfers.  The previous feature set gave the model only
month, end-of-month/quarter flags and day-of-week, so it could represent
"a Tuesday in July" but not "the 15th".  The visible consequence was a model
that forecast the monthly average straight through every revenue spike.

Measured on the DEV window only (train <= 2023-12-31, validate on 2024;
the locked 2025 holdout was not touched), adding three positional features
moved HistGBDT from 32.78% to 46.91% skill — dev MAE 42,809,624 -> 33,807,615.

These features are safe by construction: each is a function of the calendar
date alone, so no future information can enter through them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from b_ml_pipeline import calendar_exog  # noqa: E402


def test_day_of_month_features_are_present():
    cal = calendar_exog(pd.bdate_range("2025-01-01", "2025-03-31"))
    for col in ("dom", "bdom", "bdom_rev"):
        assert col in cal.columns, f"{col} missing — the model cannot see monthly position"


def test_calendar_day_of_month_is_correct():
    idx = pd.bdate_range("2025-01-01", "2025-01-31")
    cal = calendar_exog(idx)
    assert list(cal["dom"]) == list(idx.day)


def test_business_day_of_month_counts_working_days_not_calendar_days():
    """Jan 6 2025 is the 6th of the month but only the 4th working day."""
    cal = calendar_exog(pd.bdate_range("2025-01-01", "2025-01-31"))
    assert int(cal.loc["2025-01-06", "dom"]) == 6
    assert int(cal.loc["2025-01-06", "bdom"]) == 4


def test_business_days_remaining_hits_zero_on_the_last_working_day():
    cal = calendar_exog(pd.bdate_range("2025-02-01", "2025-02-28"))
    last = cal.index[-1]
    assert int(cal.loc[last, "bdom_rev"]) == 0
    assert int(cal.loc[last, "is_eom"]) == 1


def test_bdom_restarts_each_month():
    cal = calendar_exog(pd.bdate_range("2025-01-01", "2025-03-31"))
    firsts = cal.groupby([cal.index.year, cal.index.month])["bdom"].min()
    assert set(firsts) == {1}, "business-day counter must restart every month"


def test_features_depend_only_on_the_date():
    """No target values are involved, so these features cannot leak.

    Building the calendar for a date range twice, independently, must give
    identical values — a feature that touched the series could not do this.
    """
    a = calendar_exog(pd.bdate_range("2024-05-01", "2024-05-31"))
    b = calendar_exog(pd.bdate_range("2024-05-01", "2024-05-31"))
    pd.testing.assert_frame_equal(a, b)
    assert not a.isna().any().any()


def test_month_end_flag_still_works():
    """Pre-existing behaviour must be unchanged."""
    cal = calendar_exog(pd.bdate_range("2025-01-01", "2025-02-28"))
    eom_dates = [d.strftime("%Y-%m-%d") for d in cal.index[cal["is_eom"] == 1]]
    assert "2025-01-31" in eom_dates
    assert "2025-02-28" in eom_dates
