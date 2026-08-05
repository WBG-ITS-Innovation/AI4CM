"""Workstream 1: B_ML's yearly folds must be bindable to a window.

The defect this pins is not subtle. `build_yearly_folds` folded over every year from
`first_year + min_train_years` to the LAST year in the data. On this dataset that made
the final fold 2025 -- the sealed holdout -- on any default run, and there was no
argument that could ask the family for a TRAIN-internal or DEV-only search. The Phase-2
ground rules require exactly that, so the bounds are load-bearing, not a convenience.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from b_ml_pipeline import ConfigBML, available_models, build_yearly_folds  # noqa: E402
from evaluation_windows import TEST_START, window_for  # noqa: E402

IDX = pd.bdate_range("2015-01-05", "2025-08-06")


def _windows(folds):
    """Every distinct split name touched by the folds' test blocks."""
    names = set()
    for _, test_start, test_end in folds:
        for d in IDX[(IDX >= test_start) & (IDX <= test_end)]:
            names.add(window_for(d))
    return names


def test_unbounded_folds_reach_the_holdout():
    """Characterises the hazard the bounds exist to remove."""
    folds = build_yearly_folds(IDX, 4, None)
    assert "test" in _windows(folds), (
        "if this ever stops being true the dataset changed, not the bug"
    )


def test_train_only_bound_excludes_dev_and_test():
    folds = build_yearly_folds(IDX, 4, None, eval_end="2023-12-31")
    assert _windows(folds) == {"train"}
    assert len(folds) == 5, f"expected 2019..2023, got {len(folds)}"


def test_dev_bound_is_dev_only():
    folds = build_yearly_folds(IDX, 4, None, eval_start="2024-01-01",
                               eval_end="2024-12-31")
    assert _windows(folds) == {"dev"}
    assert len(folds) == 1
    assert folds[0][1] == pd.Timestamp("2024-01-01")
    assert folds[0][2] < pd.Timestamp(TEST_START)


def test_straddling_fold_is_trimmed_not_dropped():
    """A bound mid-year must shrink the block, not discard the whole year.

    Dropping it would silently halve the evaluation sample; overshooting would read
    beyond the bound. Trim.
    """
    folds = build_yearly_folds(IDX, 4, None, eval_start="2024-01-01",
                               eval_end="2024-06-30")
    assert len(folds) == 1
    assert folds[0][2] <= pd.Timestamp("2024-06-30")
    assert folds[0][1] == pd.Timestamp("2024-01-01")


def test_contradictory_bounds_yield_no_folds():
    folds = build_yearly_folds(IDX, 4, None, eval_start="2024-01-01",
                               eval_end="2019-12-31")
    assert folds == []


def test_config_carries_the_bounds_and_defaults_to_none():
    kw = dict(target="y", cadence="Daily", horizon=5, data_path="unused.csv",
              date_col="date", model_filter=None, variant="univariate", out_root="unused")
    cfg = ConfigBML(**kw)
    assert cfg.eval_start is None and cfg.eval_end is None
    bounded = ConfigBML(**kw, eval_start="2024-01-01", eval_end="2024-12-31")
    assert bounded.eval_end == "2024-12-31"


# ── the L1 variants themselves ─────────────────────────────────────────────

@pytest.mark.parametrize("name,attr,expected", [
    ("HistGBDT_L1", "loss", "absolute_error"),
    ("LightGBM_L1", "objective", "l1"),
    ("XGBoost_L1", "objective", "reg:absoluteerror"),
])
def test_l1_variants_actually_carry_an_l1_objective(name, attr, expected):
    """An L1-named model fitting squared error is worse than no L1 model at all.

    It would put an L2 result in the comparison table under an L1 label, and the
    workstream's conclusion would be drawn from a mislabelled row.
    """
    models = available_models()
    if name not in models:
        pytest.skip(f"{name} unavailable (optional dependency)")
    assert getattr(models[name], attr) == expected


def test_l1_variants_differ_from_their_twins_only_in_the_objective():
    """The delta must be attributable to the objective and nothing else."""
    models = available_models()
    for l1, base in (("HistGBDT_L1", "HistGBDT"), ("LightGBM_L1", "LightGBM"),
                     ("XGBoost_L1", "XGBoost")):
        if l1 not in models or base not in models:
            continue
        a = models[l1].get_params()
        b = models[base].get_params()
        def same(x, y):
            # XGBoost's `missing` defaults to NaN, and NaN != NaN -- comparing raw
            # would report a difference that is not one.
            if isinstance(x, float) and isinstance(y, float):
                return x == y or (x != x and y != y)
            return x == y

        differing = {k for k in set(a) | set(b) if not same(a.get(k), b.get(k))}
        assert differing <= {"loss", "objective"}, (
            f"{l1} differs from {base} in more than the objective: {differing}"
        )
