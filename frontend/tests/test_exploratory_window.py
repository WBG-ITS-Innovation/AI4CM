"""A UI-launched run may never be measured on data that must not inform a choice.

The defect these tests pin, in full
-----------------------------------
Launching Ridge on "State budget balance" at h=6 from the Lab produced
``SelectionOnReportOnlyDataError`` and a traceback. The guard was right: the run really
was about to rank models using rows from the sealed holdout. The cause was upstream. The
Lab sent ``TG_PARAM_OVERRIDES`` containing only ``folds``, ``min_train_years`` and
``demo_clip_months``, and every family reads an absent evaluation bound as "fold forward
to the last year in the file". On the canonical dataset the last year is 2025, which is
the holdout, so the final fold was the holdout end to end.

So there are two claims to hold down, and they pull in opposite directions:

1. The UI path must not produce a configuration that reaches report-only data. Tests
   below run the *real* fold builders of all four families over the real business-day
   index and assert that every scored date is train or dev.
2. The guard must not have been weakened to achieve that. Tests below re-assert that
   ``assert_selection_free`` still refuses TEST and LIVE dates, that it is not
   overridable, and that the Lab's own escape hatch cannot be used to get around it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

from evaluation_windows import (  # noqa: E402
    DEV,
    LIVE_START,
    SelectionOnReportOnlyDataError,
    TEST_START,
    assert_selection_free,
    window_for,
)
from exploratory import (  # noqa: E402
    EXPLORATORY_EVAL_END,
    check_can_run,
    contains_report_only_dates,
    exploratory_overrides,
    is_exploratory_safe,
    selectable_index,
)

#: The canonical dataset's span: business days from the first train date to the seal.
IDX = pd.bdate_range("2015-01-05", "2025-08-06")

#: What the Lab builds from each run profile, before the exploratory bound is applied.
PROFILE_OVERRIDES = {
    "Demo (fast)": {"folds": 1, "min_train_years": 0, "demo_clip_months": 12},
    "Balanced": {"folds": 3, "min_train_years": 2, "demo_clip_months": None},
    "Thorough": {"folds": 5, "min_train_years": 4, "demo_clip_months": None},
}

FAMILIES = ("A_STAT", "B_ML", "C_DL", "E_QUANTILE")


def _scored_dates(folds) -> pd.DatetimeIndex:
    """Every target date the given ``(train_end, test_start, test_end)`` folds score."""
    dates = []
    for _train_end, test_start, test_end in folds:
        dates.extend(IDX[(IDX >= test_start) & (IDX <= test_end)])
    return pd.DatetimeIndex(dates)


def _windows_touched(folds) -> set:
    return {window_for(d) for d in _scored_dates(folds)}


# ---------------------------------------------------------------------------
# 1. The configuration the UI produces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("profile", sorted(PROFILE_OVERRIDES))
def test_ui_config_never_carries_a_report_only_date(family, profile):
    ov = exploratory_overrides(family, PROFILE_OVERRIDES[profile])
    assert not contains_report_only_dates(ov), (
        f"{family}/{profile} sent a date in the sealed window or later: {ov}"
    )
    assert ov["eval_end"] == EXPLORATORY_EVAL_END
    assert window_for(pd.Timestamp(ov["eval_end"])) == "dev"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("profile", sorted(PROFILE_OVERRIDES))
def test_demo_clipping_is_cleared_on_every_profile(family, profile):
    """The Demo profile clipped to the last 12 months, which is almost all holdout.

    Bounding evaluation without clearing this leaves a bounded window over a series that
    barely reaches into dev, so the run has nothing to measure and falls back to a block
    the bound was meant to exclude. Clearing it is why the bound actually holds.
    """
    ov = exploratory_overrides(family, PROFILE_OVERRIDES[profile])
    assert ov["demo_clip_months"] is None


@pytest.mark.parametrize("family", FAMILIES)
def test_eval_start_is_explicitly_null_not_merely_absent(family):
    """C_DL's runners read ``ov.get("eval_start", TEST_START)``.

    Omitting the key therefore *selects the holdout*. Only an explicit null overrides it,
    which is why this is asserted rather than left to look like a redundant assignment.
    """
    ov = exploratory_overrides(family, {})
    assert "eval_start" in ov and ov["eval_start"] is None


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="Unknown family"):
        exploratory_overrides("D_MAGIC", {})


# ---------------------------------------------------------------------------
# 2. What the real fold builders do with that configuration
# ---------------------------------------------------------------------------

def test_b_ml_folds_stay_inside_the_selectable_region():
    pytest.importorskip("sklearn", reason="b_ml_pipeline needs the modelling stack")
    from b_ml_pipeline import build_yearly_folds

    ov = exploratory_overrides("B_ML", PROFILE_OVERRIDES["Thorough"])
    folds = build_yearly_folds(IDX, 4, ov["folds"],
                               eval_start=ov["eval_start"], eval_end=ov["eval_end"])
    assert folds, "the bound must not remove every fold"
    assert _windows_touched(folds) <= {"train", "dev"}
    assert_selection_free(_scored_dates(folds), "test_b_ml_folds")


def test_b_ml_unbounded_folds_still_reach_the_holdout():
    """The hazard the bound exists to remove. If this stops holding, the data changed."""
    pytest.importorskip("sklearn", reason="b_ml_pipeline needs the modelling stack")
    from b_ml_pipeline import build_yearly_folds

    assert "test" in _windows_touched(build_yearly_folds(IDX, 4, None))


def test_a_stat_folds_stay_inside_the_selectable_region():
    pytest.importorskip("statsmodels", reason="run_a_stat needs the modelling stack")
    import run_a_stat

    ov = exploratory_overrides("A_STAT", PROFILE_OVERRIDES["Thorough"])
    folds = run_a_stat._yearly_folds(IDX, 4, ov["folds"],
                                     eval_start=ov["eval_start"], eval_end=ov["eval_end"])
    assert folds, "the bound must not remove every fold"
    assert _windows_touched(folds) <= {"train", "dev"}
    assert_selection_free(_scored_dates(folds), "test_a_stat_folds")


def test_a_stat_unbounded_folds_still_reach_the_holdout():
    pytest.importorskip("statsmodels", reason="run_a_stat needs the modelling stack")
    import run_a_stat

    assert "test" in _windows_touched(run_a_stat._yearly_folds(IDX, 4, None))


def test_a_stat_fallback_fold_respects_the_bound():
    """The quiet path. ``_fallback_fold`` takes the newest rows in the file.

    On a bounded run those are exactly the rows the bound excludes, so before this fix a
    configuration that produced no yearly folds silently scored the holdout instead.
    """
    pytest.importorskip("statsmodels", reason="run_a_stat needs the modelling stack")
    import run_a_stat

    folds = run_a_stat._fallback_fold(IDX, 6, eval_start=None, eval_end=EXPLORATORY_EVAL_END)
    assert folds
    assert _windows_touched(folds) <= {"train", "dev"}
    assert "test" in _windows_touched(run_a_stat._fallback_fold(IDX, 6))


def test_c_dl_folds_stay_inside_the_selectable_region():
    pytest.importorskip("torch", reason="c_dl_pipeline imports torch")
    from c_dl_pipeline import build_yearly_folds as dl_folds

    ov = exploratory_overrides("C_DL", PROFILE_OVERRIDES["Thorough"])
    folds = dl_folds(IDX, 4, eval_start=ov["eval_start"], eval_end=ov["eval_end"])
    assert folds, "the bound must not remove every fold"
    assert _windows_touched(folds) <= {"train", "dev"}
    assert_selection_free(_scored_dates(folds), "test_c_dl_folds")


def test_c_dl_default_eval_start_is_the_holdout():
    """Characterises why an explicit null matters for this family specifically."""
    src = (REPO / "backend" / "run_c_dl_univariate.py").read_text(encoding="utf-8")
    assert 'ov.get("eval_start", _TEST_START)' in src


# ---------------------------------------------------------------------------
# 3. The guard is unchanged
# ---------------------------------------------------------------------------

def test_guard_still_raises_on_a_holdout_date():
    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(pd.to_datetime([TEST_START]), "unit_test")


def test_guard_still_raises_on_a_post_seal_date():
    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(pd.to_datetime([LIVE_START]), "unit_test")


def test_guard_still_raises_on_a_bound_edited_past_the_dev_end():
    """One day past the bound is one day too far, and the guard says so.

    The Lab's advanced JSON box is an escape hatch by design. This asserts that using it
    to reach the holdout produces a refusal rather than a result.
    """
    pytest.importorskip("sklearn", reason="b_ml_pipeline needs the modelling stack")
    from b_ml_pipeline import build_yearly_folds

    tampered = dict(exploratory_overrides("B_ML", {"folds": 5}), eval_end="2025-03-31")
    assert not is_exploratory_safe(tampered)
    folds = build_yearly_folds(IDX, 4, 5,
                               eval_start=None, eval_end=tampered["eval_end"])
    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(_scored_dates(folds), "tampered_bound")


@pytest.mark.parametrize("bad", [
    {"eval_end": "2025-01-01"},
    {"eval_end": None},
    {"eval_end": ""},
    {"eval_end": "not-a-date"},
    {"eval_end": DEV.end, "demo_clip_months": 12},
    {"eval_end": DEV.end, "eval_start": "2025-06-01"},
])
def test_is_exploratory_safe_rejects_each_way_round_the_bound(bad):
    assert not is_exploratory_safe(bad)


def test_is_exploratory_safe_accepts_what_the_ui_builds():
    for family in FAMILIES:
        for profile in PROFILE_OVERRIDES.values():
            assert is_exploratory_safe(exploratory_overrides(family, profile))


# ---------------------------------------------------------------------------
# 4. The pre-flight, so an impossible configuration is a sentence not a traceback
# ---------------------------------------------------------------------------

def test_preflight_passes_on_the_canonical_span():
    assert check_can_run(IDX, 6, 4) is None


def test_preflight_refuses_a_file_that_starts_after_the_bound():
    reason = check_can_run(pd.bdate_range("2025-02-03", "2025-08-06"), 6, 0)
    assert reason and reason.endswith(".")
    assert "sealed window" in reason


def test_preflight_refuses_a_horizon_longer_than_the_usable_span():
    reason = check_can_run(pd.bdate_range("2024-12-20", "2025-08-06"), 30, 0)
    assert reason and reason.endswith(".")


def test_preflight_refuses_too_little_history_for_the_profile():
    reason = check_can_run(pd.bdate_range("2023-01-02", "2024-12-31"), 6, 4)
    assert reason and reason.endswith(".")


def test_preflight_messages_are_free_of_jargon_and_tracebacks():
    """A blocked configuration is explained, not dumped."""
    reasons = [
        check_can_run(pd.bdate_range("2025-02-03", "2025-08-06"), 6, 0),
        check_can_run(pd.bdate_range("2024-12-20", "2025-08-06"), 30, 0),
        check_can_run(pd.bdate_range("2023-01-02", "2024-12-31"), 6, 4),
    ]
    for reason in reasons:
        assert reason
        assert "Traceback" not in reason
        assert "Error" not in reason
        assert "{" not in reason


def test_selectable_index_stops_at_the_dev_end():
    usable = selectable_index(IDX)
    assert usable.max() <= pd.Timestamp(EXPLORATORY_EVAL_END)
    assert {window_for(d) for d in usable} <= {"train", "dev"}
