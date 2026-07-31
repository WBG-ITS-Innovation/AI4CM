"""M-4: B_ML must not memorise, and must not crown a model that does.

Two problems observed on the real Revenues run:

  * ExtraTrees reported ``train_MAE=0.00`` — exact memorisation of every
    training row — with a val/train ratio of ~1.5e14.  LightGBM sat at 30.38,
    XGBoost at 10.45, RandomForest at 3.97.  All of this was printed as a
    ``[WARN]`` line and then ignored.
  * The best model was chosen purely by test MAE, so a model that memorised
    its training window could be crowned as long as it got lucky on the test
    window.

Fixes: capacity floors that make single-observation leaves impossible, and a
selection rule that disqualifies clearly overfitting models.  The thresholds
are principled rather than searched — actual tuning is Phase 4, on the DEV
window, never against the locked 2025 holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from b_ml_pipeline import (  # noqa: E402
    MIN_SAMPLES_PER_LEAF,
    OVERFIT_GATE_RATIO,
    OVERFIT_WARN_RATIO,
    available_models,
    select_best_model,
)


# ── capacity floors ───────────────────────────────────────────────────────

def _models():
    try:
        return available_models()
    except TypeError:
        pytest.skip("available_models requires arguments in this configuration")


def test_tree_ensembles_cannot_isolate_single_observations():
    """A leaf holding one row memorises that row; require several."""
    models = _models()
    for name in ("RandomForest", "ExtraTrees"):
        assert name in models, f"{name} missing from the registry"
        leaf = getattr(models[name], "min_samples_leaf", 1)
        assert leaf >= MIN_SAMPLES_PER_LEAF, (
            f"{name} allows leaves of {leaf} row(s); memorisation is possible"
        )


def test_hist_gbdt_is_regularised():
    models = _models()
    hgb = models["HistGBDT"]
    assert getattr(hgb, "min_samples_leaf", 1) >= 20
    assert getattr(hgb, "l2_regularization", 0.0) > 0.0


def test_boosters_have_capacity_limits_when_available():
    models = _models()
    if "XGBoost" in models:
        xgb = models["XGBoost"]
        assert getattr(xgb, "min_child_weight", 0) >= 5
        assert getattr(xgb, "max_depth", 99) <= 4
        assert getattr(xgb, "reg_lambda", 0) > 0
    if "LightGBM" in models:
        lgbm = models["LightGBM"]
        assert getattr(lgbm, "min_child_samples", 0) >= 20
        assert getattr(lgbm, "num_leaves", 999) <= 31
        assert getattr(lgbm, "reg_lambda", 0) > 0


def test_gate_threshold_is_looser_than_the_warning():
    """Warn early, disqualify only on strong evidence."""
    assert OVERFIT_GATE_RATIO > OVERFIT_WARN_RATIO


# ── gated selection ──────────────────────────────────────────────────────

def _leaderboard(rows):
    return pd.DataFrame(rows)


def test_overfitting_model_is_not_crowned_best():
    """Lowest test MAE must not win if the model failed to generalise."""
    lb = _leaderboard([
        {"model": "LightGBM", "MAE": 40_000_000.0},   # best MAE, ratio 30
        {"model": "Lasso", "MAE": 43_472_942.0},      # honest runner-up
        {"model": "Persistence (baseline)", "MAE": 60_976_736.0},
    ])
    best, excluded = select_best_model(lb, {"LightGBM": 30.38, "Lasso": 1.49})
    assert best == "Lasso"
    assert excluded == ["LightGBM"]


def test_memorising_model_is_excluded():
    """train_MAE of 0 is recorded as an infinite ratio and must disqualify."""
    lb = _leaderboard([
        {"model": "ExtraTrees", "MAE": 39_000_000.0},
        {"model": "Ridge", "MAE": 43_500_000.0},
    ])
    best, excluded = select_best_model(lb, {"ExtraTrees": float("inf"), "Ridge": 1.49})
    assert best == "Ridge"
    assert "ExtraTrees" in excluded


def test_best_model_unchanged_when_all_models_generalise():
    lb = _leaderboard([
        {"model": "Lasso", "MAE": 43_000_000.0},
        {"model": "Ridge", "MAE": 43_500_000.0},
    ])
    best, excluded = select_best_model(lb, {"Lasso": 1.49, "Ridge": 1.49})
    assert best == "Lasso"
    assert excluded == []


def test_falls_back_rather_than_naming_nothing():
    """If every model overfits, still name one — and report the exclusions."""
    lb = _leaderboard([
        {"model": "LightGBM", "MAE": 40_000_000.0},
        {"model": "XGBoost", "MAE": 42_000_000.0},
    ])
    best, excluded = select_best_model(lb, {"LightGBM": 30.0, "XGBoost": 10.0})
    assert best == "LightGBM"
    assert set(excluded) == {"LightGBM", "XGBoost"}


def test_missing_ratio_is_not_treated_as_overfitting():
    """A model with no recorded ratio should not be silently disqualified."""
    lb = _leaderboard([{"model": "Lasso", "MAE": 43_000_000.0}])
    best, excluded = select_best_model(lb, {})
    assert best == "Lasso"
    assert excluded == []


def test_empty_leaderboard_returns_none():
    best, excluded = select_best_model(pd.DataFrame(columns=["model", "MAE"]), {})
    assert best is None
    assert excluded == []
