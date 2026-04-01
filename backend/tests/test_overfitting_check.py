"""Tests for overfitting_check module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from overfitting_check import check_overfitting


def _make_predictions(n_folds: int = 3, n_per_fold: int = 50, noise_scale: float = 10.0) -> pd.DataFrame:
    """Generate synthetic predictions with controllable error."""
    np.random.seed(42)
    rows = []
    for fold in range(n_folds):
        y_true = 100 + np.random.randn(n_per_fold) * 20
        y_pred = y_true + np.random.randn(n_per_fold) * noise_scale
        for i in range(n_per_fold):
            rows.append({
                "date": f"2020-{fold+1:02d}-{i+1:02d}",
                "y_true": y_true[i],
                "y_pred": y_pred[i],
                "model": "TestModel",
                "split_id": f"fold_{fold}",
            })
    return pd.DataFrame(rows)


class TestLowRisk:
    """Well-behaved model: consistent errors across folds."""

    def test_stable_model_is_low_risk(self):
        df = _make_predictions(n_folds=3, noise_scale=10.0)
        result = check_overfitting(df)
        assert result["overfitting_risk"] == "low"
        assert len(result["fold_errors"]) == 3

    def test_fold_errors_are_populated(self):
        df = _make_predictions(n_folds=4, noise_scale=5.0)
        result = check_overfitting(df)
        assert len(result["fold_errors"]) == 4
        for fe in result["fold_errors"]:
            assert fe["test_mae"] > 0
            assert fe["n_points"] == 50


class TestOverfitting:
    """Deliberately overfit: low train error, high test error."""

    def test_high_train_test_ratio(self):
        df = _make_predictions(n_folds=3, noise_scale=10.0)
        train_errors = [
            {"fold": "fold_0", "model": "TestModel", "train_mae": 2.0, "test_mae": 10.0},
            {"fold": "fold_1", "model": "TestModel", "train_mae": 1.5, "test_mae": 8.0},
            {"fold": "fold_2", "model": "TestModel", "train_mae": 1.0, "test_mae": 12.0},
        ]
        result = check_overfitting(df, train_errors=train_errors)
        assert result["overfitting_risk"] == "high"
        assert any("overfitting" in d.lower() for d in result["details"])

    def test_borderline_ratio_is_medium_or_low(self):
        df = _make_predictions(n_folds=2, noise_scale=5.0)
        train_errors = [
            {"fold": "fold_0", "model": "TestModel", "train_mae": 5.0, "test_mae": 5.5},
            {"fold": "fold_1", "model": "TestModel", "train_mae": 5.0, "test_mae": 6.0},
        ]
        result = check_overfitting(df, train_errors=train_errors)
        assert result["overfitting_risk"] in ("low", "medium")


class TestInstability:
    """High variance across folds."""

    def test_high_variance_is_flagged(self):
        np.random.seed(42)
        rows = []
        # Fold 0: low error, Fold 1: high error, Fold 2: very high error
        noise_scales = [2.0, 20.0, 50.0]
        for fold, ns in enumerate(noise_scales):
            y_true = 100 + np.random.randn(50) * 10
            y_pred = y_true + np.random.randn(50) * ns
            for i in range(50):
                rows.append({
                    "y_true": y_true[i], "y_pred": y_pred[i],
                    "model": "TestModel", "split_id": f"fold_{fold}",
                })
        df = pd.DataFrame(rows)
        result = check_overfitting(df)
        assert result["overfitting_risk"] in ("medium", "high")
        assert np.isfinite(result["error_cv"])
        assert result["error_cv"] > 0.3


class TestDegradation:
    """Performance degrades on later folds (distribution shift)."""

    def test_increasing_error_detected(self):
        np.random.seed(42)
        rows = []
        for fold in range(4):
            noise = 5.0 + fold * 15.0  # error grows with each fold
            y_true = 100 + np.random.randn(50) * 10
            y_pred = y_true + np.random.randn(50) * noise
            for i in range(50):
                rows.append({
                    "y_true": y_true[i], "y_pred": y_pred[i],
                    "model": "TestModel", "split_id": f"fold_{fold}",
                })
        df = pd.DataFrame(rows)
        result = check_overfitting(df)
        assert result["trend_slope"] > 0.25, f"Expected positive slope, got {result['trend_slope']}"
        assert result["overfitting_risk"] in ("medium", "high")


class TestEdgeCases:

    def test_empty_dataframe(self):
        result = check_overfitting(pd.DataFrame())
        assert result["overfitting_risk"] == "low"
        assert "No valid predictions" in result["details"][0]

    def test_single_fold(self):
        df = _make_predictions(n_folds=1, noise_scale=10.0)
        result = check_overfitting(df)
        assert result["overfitting_risk"] == "low"
        assert len(result["fold_errors"]) == 1

    def test_baseline_rows_skipped(self):
        df = _make_predictions(n_folds=2)
        # Add baseline rows that should be skipped
        baseline = df.head(10).copy()
        baseline["model"] = "Persistence (baseline)"
        df = pd.concat([df, baseline], ignore_index=True)
        result = check_overfitting(df)
        # Should analyze TestModel, not baseline
        assert result["overfitting_risk"] == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
