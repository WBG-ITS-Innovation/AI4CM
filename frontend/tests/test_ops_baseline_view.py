"""The Ops baseline on a chart must be the Treasury's method, or must be absent and say so.

Two defects are pinned here, both found on the live app and both proven before being fixed.

**A baseline was invented for the stock target.** The Treasury's method takes a flow's annual
total and splits it across months, so it does not apply to a balance, and
``backend/ops_baseline.py`` returns nothing rather than inventing something. The Dashboard filled
the gap with a day-of-week mean of the ACTUAL figures and drew that under the name "Ops
baseline". Measured on ``State budget balance``: five distinct values spanning 1.38% of their own
mean, on an axis reaching 2.87B, so a straight horizontal line a reader takes for the Treasury's
planning method. It is the average of the answers.

**For flows the file was empty, and empty became zero.** Every B_ML run writes
``<target>_ops_baseline_daily.csv`` with no numbers in it: 0 non-NaN of 2763 rows, in all 14 run
folders present. An all-NaN series is not an *empty* series, so it passed the old
``not ops.empty`` guard, and ``.resample(...).sum()`` maps all-NaN to ``0.0`` -- a flat line
along zero on any weekly or monthly view. The cause is in the writer and is recorded for a
scoped fix; scoring never read those files.

The tests that matter most are the two agreement tests: the series this module produces
reproduces the run's stored ``ops_MAE`` exactly, and the stock-alias set matches the backend's.
Those are what make "the chart and the leaderboard cannot disagree" a measured claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

import ops_baseline_view as obv  # noqa: E402

DASHBOARD = FRONTEND / "pages" / "04_Dashboard.py"
LAB = FRONTEND / "pages" / "03_Lab.py"

#: A run folder with real predictions for a flow target, and one for the stock target.
FLOW_RUN = FRONTEND / "runs" / "run_B_uni_Ridge_Revenues_Daily_h6_20260819_1538" / "outputs"
FLOW_TARGET = "Revenues"
STOCK_TARGET = "State budget balance"


# ---------------------------------------------------------------------------
# The stock case: no baseline, and a reason rather than a blank
# ---------------------------------------------------------------------------

def test_the_stock_alias_set_matches_the_backend_definition():
    """The duplication guard. This module answers `is this a stock` before importing anything.

    `target_kinds` explains why its set is the union of four families' sets and why treating a
    stock as a flow is an order-of-magnitude error. A private copy that drifts from it would
    reintroduce exactly the disagreement that module was written to end.
    """
    from target_kinds import STOCK_ALIASES

    assert obv.STOCK_ALIASES == STOCK_ALIASES


def test_is_stock_matches_the_backend_on_every_alias_and_on_a_flow():
    from target_kinds import is_stock

    for name in sorted(obv.STOCK_ALIASES) + [FLOW_TARGET, "Expenditure", "Taxes"]:
        assert obv._is_stock(name) == is_stock(name), name


def test_it_is_an_exact_match_not_a_substring_match():
    """The bug the first version of `_is_stock` had, kept as a case of its own.

    `"balance" in target` looks equivalent and is not: it misses three aliases and invents a
    stock out of any column whose name happens to contain the word.
    """
    assert obv._is_stock("balance")
    assert not obv._is_stock("Revenues balance sheet")
    assert obv._is_stock("t0"), "an alias a substring check on 'balance' would miss"
    assert obv._is_stock("net")


def test_no_baseline_is_invented_for_the_stock_target():
    series, why = obv.compute(FLOW_RUN, STOCK_TARGET)
    assert series is None, "a balance has no Ops baseline and none may be substituted"
    assert why == obv.REASON_STOCK


def test_the_stock_reason_explains_itself_without_jargon():
    """A reader meeting this has to learn why, not just that."""
    why = obv.REASON_STOCK
    assert "balance" in why and "level" in why
    assert "year" in why, "the reason the method does not apply is the annual total"
    for jargon in ("stock", "REASON_STOCK", "flow aggregat", "None"):
        assert jargon not in why, f"{jargon!r} is not a word for a reader"


def test_the_stock_case_is_answered_from_the_name_alone():
    """No file read and no import, so an obvious no costs nothing."""
    assert obv.unavailable_reason(STOCK_TARGET, FLOW_RUN) == obv.REASON_STOCK


# ---------------------------------------------------------------------------
# `usable`: the specific check that was missing
# ---------------------------------------------------------------------------

def test_an_all_nan_series_is_not_usable():
    """The exact defect. `not s.empty` was True for this, so it was plotted."""
    s = pd.Series([np.nan] * 30, index=pd.bdate_range("2025-01-01", periods=30))
    assert not s.empty, "the old guard's question, answered the way that let the bug through"
    assert not obv.usable(s), "the question that needed asking"


def test_an_all_nan_series_would_have_been_drawn_as_zeros():
    """Why the old behaviour looked like a real flat line rather than a gap.

    Resampling with `.sum()` treats all-NaN as zero, so a weekly or monthly view drew a
    comparator sitting exactly on the axis.
    """
    s = pd.Series([np.nan] * 60, index=pd.bdate_range("2025-01-01", periods=60))
    assert (s.resample("ME").sum() == 0.0).all()


def test_a_series_with_numbers_is_usable():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert obv.usable(s)


def test_a_partly_nan_series_is_usable():
    """Absence of SOME values is not absence of the baseline; the caller drops the gaps."""
    s = pd.Series([np.nan, 2.0], index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert obv.usable(s)


def test_none_and_empty_are_not_usable():
    assert not obv.usable(None)
    assert not obv.usable(pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# The run-folder CSVs this module stopped reading, still measurably empty
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FLOW_RUN.exists(), reason="no sample run folder in this checkout")
def test_the_run_folder_csv_really_is_empty():
    """Pins the evidence, so the fix cannot be mistaken for a cosmetic change.

    If a corrected writer ever fills this file in, this test fails and points at the note in
    `ops_baseline_view` that says the writer is broken. That is the right time to revisit it.
    """
    csv = FLOW_RUN / f"{FLOW_TARGET}_ops_baseline_daily.csv"
    if not csv.exists():
        pytest.skip("this run folder has no ops baseline CSV")
    values = pd.to_numeric(pd.read_csv(csv)["forecast"], errors="coerce")
    assert len(values) > 0
    assert values.notna().sum() == 0, (
        "the run-folder ops CSV now has numbers in it. b_ml_pipeline.ops_monthly_baseline may "
        "have been fixed; if so, revisit the note in frontend/ops_baseline_view.py")


# ---------------------------------------------------------------------------
# The flow case: the real series, and the one claim worth measuring
# ---------------------------------------------------------------------------

requires_backend = pytest.mark.skipif(
    obv._ops_module() is None or not FLOW_RUN.exists(),
    reason="needs backend/ops_baseline.py and a sample run folder")


@pytest.fixture(scope="module")
def flow_series():
    series, why = obv.compute(FLOW_RUN, FLOW_TARGET)
    assert series is not None, f"expected a baseline for a flow target, got: {why}"
    return series


@requires_backend
def test_the_data_file_is_taken_from_the_run_rather_than_assumed(flow_series):
    """A run launched on an uploaded file must be compared against THAT file.

    Measured on this run, whose config points at `frontend/runs_uploads/uploaded.csv`:
    defaulting to the canonical table instead moved ops_MAE by 886.71. Small enough to read as
    rounding, and it is not rounding.
    """
    recorded = obv.data_path_for_run(FLOW_RUN)
    assert recorded is not None and recorded.exists()


@requires_backend
def test_it_reproduces_the_leaderboard_ops_mae_exactly(flow_series):
    """The claim that justifies going to the backend at all.

    The chart's comparison line and the table's skill figure must be the same numbers. Not
    close: the same. Both come from `ops_baseline.vintage_cache` / `ops_prediction_for`, so any
    difference means this module is building something else.
    """
    leaderboard = FLOW_RUN / "leaderboard.csv"
    if not leaderboard.exists():
        pytest.skip("this run folder has no leaderboard")
    stored = pd.read_csv(leaderboard)["ops_MAE"].dropna()
    if stored.empty:
        pytest.skip("this run recorded no ops comparison")

    pred = pd.read_csv(FLOW_RUN / "predictions_long.csv")
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred = pred.dropna(subset=["y_true", "y_pred"])
    pred["_ops"] = pred["target_date"].map(flow_series)
    ok = pred["_ops"].notna()
    assert ok.any(), "the baseline covered none of the run's dates"

    recomputed = float(np.mean(np.abs(pred.loc[ok, "y_true"] - pred.loc[ok, "_ops"])))
    assert recomputed == pytest.approx(float(stored.iloc[0]), abs=1e-6)


@requires_backend
def test_the_baseline_holds_one_value_per_month_and_vintage(flow_series):
    """The shape the caption promises, and the reason the line looks flat.

    The method spreads a month's total evenly across that month's working days, so it is
    constant within a month. This is the test that says the flat stretches are the method
    working rather than a bug, which is the question that started all of this.

    The grouping includes the ORIGIN year, and that is not a fudge to make the test pass. Every
    January carries two values, because its first few days were forecast from an origin in the
    previous December, when that year was not yet complete. Those rows are compared against the
    older figure the Treasury actually had in hand at the time, which is what
    `ops_baseline.vintage_cache` exists to do. Grouping by month alone asserts something untrue:
    measured on this run it fails on 2020-01 through 2024-01, five Januaries out of five.
    """
    from ops_baseline import _vintage_year

    pred = pd.read_csv(FLOW_RUN / "predictions_long.csv")
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred["origin_date"] = pd.to_datetime(pred["origin_date"])
    pred = pred.drop_duplicates("target_date").set_index("target_date")

    frame = pd.DataFrame({"ops": flow_series})
    # The vintage is "the last calendar year complete at the origin", NOT the origin's own year.
    # Origins inside a single December straddle that boundary: 2019-12-30 has 2018 complete,
    # 2019-12-31 has 2019. Grouping on `origin.year` therefore still reports 2020-01, 2021-01 and
    # 2022-01 as non-constant, which is the same true-but-misread shape one level down.
    frame["vintage"] = pred["origin_date"].map(_vintage_year)
    frame = frame.dropna()

    offenders = {
        f"{y}-{m:02d} (origin year {v})": sorted(g["ops"].unique())
        for (y, m, v), g in frame.groupby(
            [frame.index.year, frame.index.month, frame["vintage"]])
        if g["ops"].nunique() != 1
    }
    assert not offenders, f"not piecewise-constant within a month and vintage: {offenders}"


@requires_backend
def test_january_carries_two_values_because_the_vintage_rolls_over(flow_series):
    """Pins the exception itself, so nobody 'fixes' it into being wrong.

    A future reader seeing two values in January would reasonably suspect the bug this whole
    module replaced. It is the opposite: flattening January to one value would compare December
    origins against a year that had not finished at the time.
    """
    pred = pd.read_csv(FLOW_RUN / "predictions_long.csv")
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred["origin_date"] = pd.to_datetime(pred["origin_date"])
    jan = pred[pred["target_date"].dt.month == 1].drop_duplicates("target_date")
    jan = jan.assign(ops=jan["target_date"].map(flow_series)).dropna(subset=["ops"])
    if jan.empty:
        pytest.skip("this run covers no January dates")

    for year, group in jan.groupby(jan["target_date"].dt.year):
        origin_years = group["origin_date"].dt.year.nunique()
        if origin_years > 1:
            assert group["ops"].nunique() == origin_years, (
                f"January {year} spans {origin_years} vintages, so it must carry that many "
                f"values, not {group['ops'].nunique()}")
            # And the earlier-origin rows must get the SMALLER figure: they are averaging an
            # older, lower three-year window on a series that grows.
            by_origin = group.groupby(group["origin_date"].dt.year)["ops"].first()
            assert list(by_origin) == sorted(by_origin), (
                f"January {year}: the older vintage should be the lower figure")


@requires_backend
def test_the_baseline_steps_between_months(flow_series):
    """Constant within a month is only half the claim; a single global mean would pass that."""
    monthly_levels = flow_series.groupby(
        [flow_series.index.year, flow_series.index.month]).first()
    assert monthly_levels.nunique() > 1, "a baseline with one level everywhere is a global mean"
    assert monthly_levels.nunique() >= 0.5 * len(monthly_levels), (
        "most months should carry their own level")


@requires_backend
def test_the_baseline_is_never_negative(flow_series):
    """The reason the flat spread is canonical rather than the intraday-profile variant.

    `ops_baseline` records that the profile spread emits negative daily revenue figures on some
    days, which is not defensible as a planning number. A negative here means this module picked
    up the sensitivity variant instead of the method.
    """
    assert (flow_series.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# The pages: the invented fallback is gone and the reader is told
# ---------------------------------------------------------------------------

def test_the_dashboard_no_longer_invents_a_weekday_mean_baseline():
    src = DASHBOARD.read_text(encoding="utf-8")
    assert "def _weekday_mean_baseline" not in src, (
        "a day-of-week mean of the actuals is not the Treasury's planning method")
    assert "def _read_ops_baseline" not in src, (
        "the run-folder CSV is written empty by every B_ML run")


def test_both_pages_ask_whether_the_baseline_has_numbers_in_it():
    """`not s.empty` is the question that let an all-NaN series through."""
    for page in (DASHBOARD, LAB):
        src = page.read_text(encoding="utf-8")
        assert "obv.usable(" in src, f"{page.name} must test the series for real numbers"


def test_both_pages_explain_the_baseline_or_its_absence():
    """A chart that quietly drops its comparator reads as 'nothing to beat'."""
    for page in (DASHBOARD, LAB):
        src = page.read_text(encoding="utf-8")
        assert "obv.CAPTION_WHY_FLAT" in src, f"{page.name} must say why the line looks flat"


def test_the_flatness_caption_says_why_rather_than_apologising():
    caption = obv.CAPTION_WHY_FLAT
    assert "three complete calendar years" in caption
    assert "working days" in caption
    assert "not a forecast" in caption, "the most important thing about it"


def test_every_reason_is_written_for_a_reader():
    """No identifiers, no file paths presented as an explanation, no bare exception text."""
    for name in ("REASON_STOCK", "REASON_NO_BACKEND", "REASON_NO_PREDICTIONS"):
        reason = getattr(obv, name)
        assert reason[0].isupper() and reason.rstrip().endswith("."), name
        assert len(reason.split()) >= 12, f"{name} is too terse to explain anything"
