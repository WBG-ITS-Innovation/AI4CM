"""Item 1b: C_DL must report on the shared window, not a multi-year average.

C_DL folded over EVERY available year, so its published skill was an average across
2019-2025 while the other three families reported on the 2025 holdout. Measured
(review §4.3):

    C_DL reported  +10.84% skill   (2019-2025 average, n=1,722)
    C_DL actually   -5.19%         (shared 2025 window, n=156)

The gap is not model behaviour, it is the ruler: persistence over 2019-2025 is
52,957,744 while over 2025 it is far harder. All five architectures are worse than
doing nothing on the reporting window, and the published figure hid it.

C_DL stays parked for Phase 2 (decision Q6). This pin exists so the one-ruler check
can include C_DL rather than exempt it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from c_dl_pipeline import ConfigDL, build_yearly_folds  # noqa: E402
from evaluation_windows import TEST_START  # noqa: E402

LABELS = pd.DatetimeIndex(pd.bdate_range("2015-01-05", "2025-08-06"))
MIN_TRAIN_YEARS = 4


def test_unpinned_folds_span_many_years_the_pre_fix_behaviour():
    """Documents what the pin is for; not an endorsement."""
    folds = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=None)
    years = {ts.year for _, ts, _ in folds}
    assert len(folds) >= 6, folds
    assert min(years) < 2025, (
        "unpinned folds should reach back years before the holdout -- that is the "
        "behaviour that let a 2019-2025 average be published as a holdout result"
    )


def test_pinned_folds_start_no_earlier_than_the_eval_start():
    folds = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=TEST_START)
    assert folds, "pinning must not remove every fold"
    cutoff = pd.Timestamp(TEST_START)
    for train_end, test_start, test_end in folds:
        assert test_start >= cutoff, (
            f"fold test block starts {test_start.date()}, before the pinned "
            f"eval_start {cutoff.date()}"
        )
        assert test_end >= cutoff


def test_pinning_drops_the_pre_holdout_folds():
    unpinned = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=None)
    pinned = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=TEST_START)
    assert len(pinned) < len(unpinned), (
        f"pinning kept {len(pinned)} of {len(unpinned)} folds -- it dropped nothing"
    )
    assert len(pinned) >= 1


def test_training_still_ends_before_the_evaluation_block():
    """The pin must not accidentally let a fold train into its own test window."""
    for train_end, test_start, test_end in build_yearly_folds(
            LABELS, MIN_TRAIN_YEARS, eval_start=TEST_START):
        assert train_end < test_start, (
            f"train_end {train_end.date()} is not before test_start {test_start.date()}"
        )


def test_a_fold_straddling_the_cutoff_is_trimmed_not_dropped():
    """A test block that begins before the cutoff but ends after it keeps its tail."""
    cutoff = "2025-03-01"
    folds = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=cutoff)
    assert folds, "the straddling fold was dropped entirely"
    for _, test_start, test_end in folds:
        assert test_start >= pd.Timestamp(cutoff)
    # 2025's block runs Jan..Aug; trimming should start it on/after 1 March.
    assert any(ts.year == 2025 for _, ts, _ in folds)


def test_config_default_is_unpinned_so_the_pin_is_an_explicit_choice():
    """ConfigDL stays permissive; the RUNNERS opt in.

    Keeping the dataclass default None means an existing caller constructing
    ConfigDL directly is not silently re-scoped, while the shipped runners pin to
    TEST_START.
    """
    cfg = ConfigDL(data_path="x")
    assert cfg.eval_start is None


def test_the_runners_default_to_the_shared_test_start():
    """Source-level: the pin must actually be wired, not merely available.

    A pin that exists but is never passed is the failure mode found twice already
    (the M-5 sentinel split and the E_QUANTILE reindex both had green unit tests
    while the pipeline stopped calling the fixed code).
    """
    for runner in ("run_c_dl_univariate.py", "run_c_dl_multivariate.py"):
        src = (BACKEND_DIR / runner).read_text()
        assert "eval_start=ov.get(\"eval_start\", _TEST_START)" in src, (
            f"{runner} does not default eval_start to TEST_START"
        )
        assert "from evaluation_windows import TEST_START" in src, (
            f"{runner} hardcodes a date instead of importing TEST_START"
        )


def test_pin_can_be_disabled_explicitly():
    """An operator must be able to ask for the full history, deliberately."""
    folds = build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=None)
    assert len(folds) > len(
        build_yearly_folds(LABELS, MIN_TRAIN_YEARS, eval_start=TEST_START))
