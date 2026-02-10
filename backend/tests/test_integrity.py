"""Unit tests for forecast integrity and anti-leakage checks."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.integrity import (
    validate_alignment,
    shift_sanity_check,
    compute_persistence_baseline_from_origin,
    compute_integrity_report,
)


def test_validate_alignment_business_days():
    """Test that origin_date + h business days equals target_date."""
    # Create synthetic daily business-day series
    dates = pd.bdate_range("2024-01-01", "2024-02-15", freq="B")
    n = len(dates)
    
    # Create predictions with h=6 business days
    h = 6
    predictions = []
    for i in range(h, n - 5):  # Ensure we have enough data
        origin_date = dates[i]
        target_date = dates[i + h]  # 6 business days later
        
        predictions.append({
            "origin_date": origin_date,
            "target_date": target_date,
            "origin_value": 100.0 + i * 0.5,
            "y_true": 100.0 + (i + h) * 0.5 + np.random.randn() * 0.1,
            "y_pred": 100.0 + (i + h) * 0.5 + np.random.randn() * 0.2,
            "model": "TestModel",
            "date": target_date,  # target_date for compatibility
        })
    
    df = pd.DataFrame(predictions)
    
    # Test alignment validation
    result = validate_alignment(df, horizon=h, cadence="Daily")
    
    # All should be aligned (within 1 day tolerance for business days)
    assert result["alignment_ok"] is True, f"Alignment failed: {result.get('misaligned_examples', [])}"
    assert result["n_misaligned"] == 0, f"Found {result['n_misaligned']} misaligned predictions"
    assert result["n_total"] == len(predictions)


def test_validate_alignment_misaligned():
    """Test that misaligned predictions are detected."""
    dates = pd.bdate_range("2024-01-01", "2024-02-15", freq="B")
    h = 6
    
    predictions = []
    for i in range(h, min(20, len(dates) - h)):
        origin_date = dates[i]
        # Intentionally misalign: use origin_date + h+1 instead of h
        target_date = dates[i + h + 1]  # WRONG: should be i+h
        
        predictions.append({
            "origin_date": origin_date,
            "target_date": target_date,
            "origin_value": 100.0,
            "y_true": 100.0,
            "y_pred": 100.0,
            "model": "TestModel",
            "date": target_date,
        })
    
    df = pd.DataFrame(predictions)
    result = validate_alignment(df, horizon=h, cadence="Daily")
    
    # Should detect misalignment
    assert result["alignment_ok"] is False
    assert result["n_misaligned"] > 0
    assert len(result["misaligned_examples"]) > 0


def test_shift_sanity_check_correct_alignment():
    """Test that correctly aligned predictions have best_shift=0."""
    # Create synthetic aligned data: y_pred[t] predicts y_true[t] correctly
    n = 100
    y_true = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    y_pred = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.2  # Slightly noisier
    
    h = 6
    result = shift_sanity_check(y_true, y_pred, horizon=h, max_shift=10)
    
    # Best shift should be 0 (or very close) for correctly aligned data
    assert result["best_shift"] == 0, f"Expected best_shift=0, got {result['best_shift']}"
    assert result["lag_warning"] is False, "Should not warn for correctly aligned data"
    assert result["mae_shift0"] > 0  # Should have some error


def test_shift_sanity_check_misaligned():
    """Test that misaligned predictions (shifted by -h) are detected."""
    n = 100
    h = 6
    
    # Create data where y_pred[t] actually matches y_true[t-h] (persistence-like)
    y_true = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    y_pred = np.concatenate([y_true[:h], y_true[:-h]])  # Shifted by -h
    
    result = shift_sanity_check(y_true, y_pred, horizon=h, max_shift=10)
    
    # Best shift should be -h (or close to it)
    assert result["best_shift"] <= -h + 2, f"Expected best_shift <= {-h+2}, got {result['best_shift']}"
    assert result["lag_warning"] is True, "Should warn for misaligned data"
    assert result["improvement_ratio"] < 0.85, "Should show significant improvement from shifting"


def test_persistence_baseline():
    """Test persistence baseline computation."""
    n = 50
    predictions = []
    for i in range(n):
        origin_val = 100.0 + i * 0.5
        y_true = origin_val + np.random.randn() * 0.1
        y_pred = origin_val + np.random.randn() * 0.2
        
        predictions.append({
            "origin_value": origin_val,
            "y_true": y_true,
            "y_pred": y_pred,
        })
    
    df = pd.DataFrame(predictions)
    result = compute_persistence_baseline_from_origin(df)
    
    assert not np.isnan(result["mae_persistence"])
    assert result["mae_persistence"] > 0
    assert result["n_valid"] == n


def test_integrity_report_full():
    """Test full integrity report on synthetic correctly-aligned data."""
    # Create synthetic daily business-day series
    dates = pd.bdate_range("2024-01-01", "2024-03-31", freq="B")
    n = len(dates)
    h = 6
    
    # Create full series
    full_series = pd.Series(
        100.0 + np.arange(n) * 0.5 + np.sin(np.arange(n) * 2 * np.pi / 5) * 2.0,
        index=dates
    )
    
    # Create predictions
    predictions = []
    for i in range(h, n - 10):
        origin_date = dates[i]
        target_date = dates[i + h]
        origin_val = float(full_series.loc[origin_date])
        y_true = float(full_series.loc[target_date])
        y_pred = y_true + np.random.randn() * 0.5  # Small error
        
        predictions.append({
            "date": target_date,
            "target_date": target_date,
            "origin_date": origin_date,
            "origin_value": origin_val,
            "y_true": y_true,
            "y_pred": y_pred,
            "model": "TestModel",
        })
    
    df = pd.DataFrame(predictions)
    
    # Compute integrity report
    report = compute_integrity_report(df, full_series, horizon=h, model_name="TestModel", cadence="Daily")
    
    # Check alignment
    assert report["alignment_ok"] is True, f"Alignment failed: {report.get('misaligned_examples', [])}"
    
    # Check shift diagnostic
    assert report["best_shift"] == 0, f"Expected best_shift=0, got {report['best_shift']}"
    assert report["lag_warning"] is False, "Should not warn for correctly aligned data"
    
    # Check baseline
    assert not np.isnan(report["mae_persistence"])
    assert not np.isnan(report["mae_model"])
    
    # Check skill
    assert not np.isnan(report["skill_pct"])
    # For synthetic data with small error, skill should be positive
    assert report["skill_pct"] > -50, f"Skill too negative: {report['skill_pct']}%"


def test_training_data_boundary():
    """Test that model training rows end at origin_date (no target leakage)."""
    # This is more of a documentation test - the actual check happens in the pipeline
    # But we can verify the concept with synthetic data
    
    dates = pd.bdate_range("2024-01-01", "2024-02-15", freq="B")
    h = 6
    
    # Simulate training: features at time t should only use data up to t
    # Target should be y(t+h), not y(t)
    for i in range(h, len(dates) - h):
        origin_date = dates[i]
        target_date = dates[i + h]
        
        # Training data should be dates[0] to dates[i] (inclusive)
        train_dates = dates[:i+1]
        
        # Target values should be shifted: y_train[t] = y_true[t+h]
        # This means the last training target is at origin_date, predicting target_date
        assert train_dates[-1] == origin_date
        assert target_date == origin_date + pd.Timedelta(days=h)
        
        # Verify: origin_date + h business days = target_date
        expected_target = origin_date + pd.Timedelta(days=h)
        # For business days, we need to count actual business days
        # Using positional indexing: dates[i+h] should equal target_date
        assert dates[i + h] == target_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
