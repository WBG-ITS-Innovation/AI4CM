"""Phase 2 ground rule 1: the TRAIN/DEV/TEST split is enforced, not just described.

Until now `evaluation_windows.py` was documentation. It stated the discipline --
"TEST ... never used to choose anything ... the number of such consultations should
stay at zero" -- and no pipeline imported it (review §2.4). `eval_start_for()` and
`window_for()` had zero callers, while the only hardcoded evaluation window in the
repo pointed straight at `TEST_START`.

These tests pin the enforcement so it cannot quietly become documentation again.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import evaluation_windows as ew  # noqa: E402

BIZ = pd.date_range("2015-01-05", "2025-08-06", freq="B")
H = 5


@pytest.fixture(autouse=True)
def _test_reads_blocked(monkeypatch):
    """Every test starts with TEST reads forbidden, whatever the outer env says."""
    monkeypatch.delenv(ew.TEST_ACCESS_ENV, raising=False)


# ── the TEST holdout is gated and loud ────────────────────────────────────

def test_reading_test_raises_by_default():
    with pytest.raises(ew.TestWindowAccessError, match="Refusing to read the TEST window"):
        ew.require_test_access("curiosity")


def test_restrict_to_test_is_gated():
    with pytest.raises(ew.TestWindowAccessError):
        ew.restrict(BIZ, "test")


def test_a_test_read_requires_a_stated_reason():
    with pytest.raises(ValueError, match="stated reason"):
        ew.require_test_access("")


def test_permitted_test_read_is_logged(monkeypatch, tmp_path, capsys):
    """Permission is not enough: the read must leave a durable trace.

    'How many times did we consult the holdout' should have a factual answer.
    """
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(ew, "TEST_ACCESS_LOG", log)
    monkeypatch.setenv(ew.TEST_ACCESS_ENV, "1")

    ew.require_test_access("final reporting, released by the user")

    err = capsys.readouterr().err
    assert "TEST WINDOW READ" in err, "a permitted read must still be loud on stderr"
    assert log.exists(), "the read was not logged"
    assert "final reporting" in log.read_text()


def test_restrict_to_test_works_once_released(monkeypatch, tmp_path):
    monkeypatch.setattr(ew, "TEST_ACCESS_LOG", tmp_path / "a.log")
    monkeypatch.setenv(ew.TEST_ACCESS_ENV, "1")
    got = ew.restrict(BIZ, "test")
    assert len(got) > 0
    assert got.min() >= pd.Timestamp(ew.TEST_START)


# ── window membership ─────────────────────────────────────────────────────

def test_windows_partition_the_index_without_overlap():
    counts = {w: len(ew.restrict(BIZ, w)) for w in ("train", "dev")}
    assert counts["train"] > 0 and counts["dev"] > 0
    assert ew.restrict(BIZ, "train").max() < pd.Timestamp(ew.DEV_START)
    assert ew.restrict(BIZ, "dev").min() >= pd.Timestamp(ew.DEV_START)
    assert ew.restrict(BIZ, "dev").max() < pd.Timestamp(ew.TEST_START)


def test_train_plus_dev_is_the_searchable_region():
    td = ew.restrict(BIZ, "train+dev")
    assert td.max() < pd.Timestamp(ew.TEST_START)
    assert len(td) == len(ew.restrict(BIZ, "train")) + len(ew.restrict(BIZ, "dev"))


def test_assert_within_catches_a_dev_leak_into_a_train_only_step():
    spanning = pd.date_range("2023-12-20", "2024-01-10", freq="B")
    with pytest.raises(ew.TestWindowAccessError, match=r"\['dev'\]"):
        ew.assert_within(spanning, "train", "hyperparameter search")


def test_assert_within_passes_on_a_clean_window():
    ew.assert_within(pd.date_range("2020-01-01", "2020-06-30", freq="B"),
                     "train", "hyperparameter search")


# ── rolling-origin folds for search ───────────────────────────────────────

def test_search_folds_stay_inside_train():
    """Search must never see DEV, let alone TEST."""
    folds = ew.rolling_origin_folds(BIZ, horizon=H, n_folds=5)
    assert folds, "no folds produced"
    for f in folds:
        assert ew.window_for(f.train_end) == "train"
        assert ew.window_for(f.eval_start) == "train"
        assert ew.window_for(f.eval_end) == "train"


def test_every_fold_has_a_horizon_sized_embargo():
    """No training label may be dated at or after the first evaluation origin.

    This is the gap the review found missing in E_QUANTILE (§2.1, 80% of
    predictions affected) and in B_ML's own validation split (§2.2, where it made
    the M-4 overfit gate flip on XGBoost).
    """
    folds = ew.rolling_origin_folds(BIZ, horizon=H, n_folds=5)
    for f in folds:
        gap = len(pd.bdate_range(f.train_end, f.eval_start)) - 1
        assert gap >= H, (
            f"fold {f.fold_id}: only {gap} business day(s) between train_end "
            f"{f.train_end.date()} and eval_start {f.eval_start.date()}; need >= {H}"
        )


def test_folds_do_not_overlap_and_advance_in_time():
    folds = ew.rolling_origin_folds(BIZ, horizon=H, n_folds=5)
    for a, b in zip(folds, folds[1:]):
        assert a.eval_end < b.eval_start, "fold evaluation blocks overlap"
        assert a.train_end < b.train_end, "training window must expand"


def test_default_geometry_gives_a_usable_search_signal():
    """Guards the sizing choice, not just the mechanics.

    Sizing the block as usable//n_folds produced ~1.4-year evaluation blocks with
    the earliest fold training on about two years -- a noisy basis for choosing
    hyperparameters. The fixed 6-month block over a >=4-year base is the point.
    """
    folds = ew.rolling_origin_folds(BIZ, horizon=H, n_folds=5)
    assert len(folds) == 5
    for f in folds:
        block = len(pd.bdate_range(f.eval_start, f.eval_end))
        assert 100 <= block <= 150, f"fold {f.fold_id} eval block is {block} rows"
        train_rows = len(pd.bdate_range(BIZ.min(), f.train_end))
        assert train_rows >= ew.DEFAULT_MIN_TRAIN, (
            f"fold {f.fold_id} trains on {train_rows} rows, below "
            f"DEFAULT_MIN_TRAIN={ew.DEFAULT_MIN_TRAIN}"
        )


def test_asking_for_test_folds_is_gated():
    with pytest.raises(ew.TestWindowAccessError):
        ew.rolling_origin_folds(BIZ, horizon=H, n_folds=2, window="test")


def test_impossible_geometry_fails_loudly_rather_than_silently_shrinking():
    with pytest.raises(ValueError, match="do not fit"):
        ew.rolling_origin_folds(BIZ, horizon=H, n_folds=40, eval_block=126)


# ── MASE scaling ──────────────────────────────────────────────────────────

def _series(window_end="2025-08-06"):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-05", window_end, freq="B")
    return pd.Series(rng.normal(7e7, 1e7, len(idx)), index=idx, name="Revenues")


def test_mase_scale_uses_train_only():
    """A DEV or TEST shift must not move the denominator."""
    s = _series()
    base = ew.seasonal_naive_scale(s, season=5)

    shifted = s.copy()
    shifted.loc[shifted.index >= pd.Timestamp(ew.DEV_START)] *= 100.0
    assert ew.seasonal_naive_scale(shifted, season=5) == base, (
        "changing DEV/TEST changed the MASE scale, so it is not TRAIN-only"
    )


def test_mase_of_a_perfect_forecast_is_zero_and_of_the_scale_is_one():
    s = _series()
    scale = ew.seasonal_naive_scale(s, season=5)
    y = np.array([100.0, 200.0, 300.0])
    assert ew.mase(y, y, scale) == 0.0
    assert ew.mase(y, y + scale, scale) == pytest.approx(1.0)


def test_mase_scale_rejects_a_degenerate_series():
    idx = pd.date_range("2015-01-05", "2023-12-29", freq="B")
    flat = pd.Series(1.0, index=idx, name="flat")
    with pytest.raises(ValueError, match="Degenerate MASE scale"):
        ew.seasonal_naive_scale(flat, season=5)
