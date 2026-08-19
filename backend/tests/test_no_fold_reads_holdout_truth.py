"""No fold, in any family, may read TRUTH from a window it is not allowed to read.

The invariant, and why it needed its own file
---------------------------------------------
A fold is bounded by ORIGIN — the day a forecast is made. Its truth is read at ``origin + h``. Those
are different dates, and every guard in the project was checking the first one.

So a DEV fold bounded to 2024 was scored against 4 target dates in 2025: inside the sealed holdout.
``assert_selection_free`` passed because all 250 origins were in DEV. Measured on all three targets,
and it meant every champion credential in ``registry/recipes.json`` was computed partly on holdout
truth, while the Optuna search read 5 rows of its own confirmation set at the TRAIN/DEV boundary.

Two tests used to *document* that absence, pinned to fail once it was fixed. They are gone. This
file replaces them with the property itself, asserted per fold builder, because a test that records
a defect protects one call site while a test that asserts the invariant protects the next one
somebody writes.

What "allowed" means
--------------------
* A **selection** path (searching, tuning, crowning) may read truth from ``train`` and ``dev`` only.
* A **reporting** path may read holdout truth, but must announce it through the ledger — enforced
  in each family's own tests, not here.
* No path may read ``live`` truth for selection: that data arrived after sealing.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

from evaluation_windows import (DEV, SELECTABLE_WINDOWS, TEST_END,        # noqa: E402
                               TEST_START, window_for)

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
TARGETS = ["Revenues", "Expenditure", "State budget balance"]
HORIZON = 5


def _target_windows(index, origins, horizon=HORIZON):
    """The windows the truth for these origins actually comes from."""
    pos = {d: i for i, d in enumerate(index)}
    out = []
    for d in pd.DatetimeIndex(origins):
        i = pos.get(d)
        if i is not None and i + horizon < len(index):
            out.append(window_for(index[i + horizon]))
    return out


# ── the selection path that leaked ───────────────────────────────────────────

@needs_data
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("window", ["train", "dev"])
def test_ws2_tune_folds_read_truth_only_from_selectable_windows(target, window):
    """The tuner is THE selection path: its credentials are what the registry records."""
    from ws2_tune import design, make_folds

    s, *_ = design(target)
    folds, _ = make_folds(target, window)
    assert folds, f"{target}/{window}: no folds built"
    seen = set()
    for f in folds:
        seen.update(_target_windows(s.index, f.X_te.index))
    assert seen, "no truth windows resolved"
    assert seen <= SELECTABLE_WINDOWS, (
        f"{target}/{window}: truth read from {sorted(seen - SELECTABLE_WINDOWS)}, which selection "
        f"may not touch")


@needs_data
@pytest.mark.parametrize("target", TARGETS)
def test_a_train_search_does_not_read_its_own_confirmation_set(target):
    """DEV confirms a search. A search that has already seen part of DEV is not confirmed by it.

    TRAIN fold 5 used to read 5 DEV target dates on every target — the same origin/target
    mismatch, one boundary earlier.
    """
    from ws2_tune import design, make_folds

    s, *_ = design(target)
    folds, _ = make_folds(target, "train")
    seen = set()
    for f in folds:
        seen.update(_target_windows(s.index, f.X_te.index))
    assert seen == {"train"}, (
        f"{target}: a TRAIN search reads truth from {sorted(seen)}; DEV must stay unseen")


@needs_data
@pytest.mark.parametrize("target", TARGETS)
def test_a_dev_fold_reads_no_holdout_truth(target):
    """The specific leak, stated as the property rather than as its absence."""
    from ws2_tune import design, make_folds

    s, *_ = design(target)
    folds, _ = make_folds(target, "dev")
    seen = set()
    for f in folds:
        seen.update(_target_windows(s.index, f.X_te.index))
    assert seen == {"dev"}, f"{target}: a DEV fold reads truth from {sorted(seen)}"


# ── the reporting harness, on both of its scopes ────────────────────────────

@needs_data
@pytest.mark.parametrize("target", TARGETS)
def test_the_sealed_harness_reads_only_the_window_it_was_asked_for(target):
    """Scoped to the holdout it reads holdout truth; scoped to DEV it reads DEV truth.

    The second half is what makes ``dev_reconstruction`` comparable to a corrected credential: both
    are measured on the same row set.
    """
    import sealed_window_report as swr

    sealed, _ = swr.sealed_folds(target, log=False)
    tds = [d for f in sealed for d in f.target_dates]
    assert tds and {window_for(d) for d in tds} == {"test"}

    dev, _ = swr.sealed_folds(target, eval_start=DEV.start, eval_end=DEV.end, log=False)
    tdd = [d for f in dev for d in f.target_dates]
    assert tdd and {window_for(d) for d in tdd} == {"dev"}, (
        f"{target}: a DEV-scoped report reads {sorted({window_for(d) for d in tdd})}")


@needs_data
def test_no_fold_anywhere_reads_live_truth():
    """LIVE arrived after sealing. Nothing may select on it, and today it is empty by construction,
    so any LIVE truth at all means a boundary moved."""
    import sealed_window_report as swr
    from ws2_tune import design, make_folds

    for target in TARGETS:
        s, *_ = design(target)
        for window in ("train", "dev"):
            folds, _ = make_folds(target, window)
            for f in folds:
                assert "live" not in _target_windows(s.index, f.X_te.index), (
                    f"{target}/{window} reads LIVE truth")
        for f in swr.sealed_folds(target, log=False)[0]:
            assert all(d <= pd.Timestamp(TEST_END) for d in f.target_dates), (
                f"{target}: a sealed-window fold reaches past {TEST_END} into LIVE")


# ── the guard itself must look at target dates, not origins ─────────────────

def test_the_selection_guard_is_applied_to_target_dates():
    """Checking origins is what let this through; the source must check both."""
    src = (REPO / "scripts" / "ws2_tune.py").read_text(encoding="utf-8")
    assert "evaluation TARGET dates" in src, (
        "assert_selection_free must be called on the TARGET dates, not only the origins")
    assert "ALLOWED_TARGET_WINDOWS" in src


@needs_data
def test_the_fix_costs_the_rows_the_record_says_it_costs():
    """A regression guard on the trade, so a future change cannot quietly widen or narrow it.

    Measured and identical on all three targets: DEV 250 -> 246, TRAIN 1259 -> 1254.
    """
    from ws2_tune import make_folds

    for target in TARGETS:
        dev, _ = make_folds(target, "dev")
        train, _ = make_folds(target, "train")
        n_dev = sum(len(f.y_te) for f in dev)
        n_train = sum(len(f.y_te) for f in train)
        assert n_dev == 246, f"{target}: DEV n={n_dev}, expected 246"
        assert n_train == 1254, f"{target}: TRAIN n={n_train}, expected 1254"
