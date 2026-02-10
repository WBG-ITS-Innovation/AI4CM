"""
Unit tests for forecast integrity checks across multiple horizons.

✅ Tests alignment, leakage, and shift detection for h in [1,2,5,6,10].
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.forecast_integrity import (
    validate_alignment_step_based,
    shift_diagnostic_horizon_aware,
    compute_persistence_baseline,
    compute_skill_score,
    check_feature_leakage,
)


def test_alignment_step_based_calendar_daily():
    """Test alignment validation for calendar daily data (no missing dates)."""
    # Create synthetic daily series
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    s = pd.Series(np.random.randn(len(dates)), index=dates)
    
    for h in [1, 2, 5, 6, 10]:
        # Create predictions with correct alignment
        predictions = []
        for i in range(h, len(dates)):
            origin_date = dates[i - h]
            target_date = dates[i]
            predictions.append({
                "origin_date": origin_date,
                "target_date": target_date,
                "horizon": h,
                "y_true": s.iloc[i],
                "y_pred": s.iloc[i] + np.random.randn() * 0.1,
            })
        
        df = pd.DataFrame(predictions)
        result = validate_alignment_step_based(df, dates, h)
        
        assert result["alignment_ok"], f"Alignment failed for h={h}"
        assert result["n_misaligned"] == 0, f"Found {result['n_misaligned']} misaligned predictions for h={h}"


def test_alignment_step_based_business_daily():
    """Test alignment validation for business daily data (weekends missing)."""
    # Create synthetic business-day series
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    s = pd.Series(np.random.randn(len(dates)), index=dates)
    
    for h in [1, 5, 6, 10]:
        # Create predictions with correct alignment
        predictions = []
        for i in range(h, len(dates)):
            origin_date = dates[i - h]
            target_date = dates[i]
            predictions.append({
                "origin_date": origin_date,
                "target_date": target_date,
                "horizon": h,
                "y_true": s.iloc[i],
                "y_pred": s.iloc[i] + np.random.randn() * 0.1,
            })
        
        df = pd.DataFrame(predictions)
        result = validate_alignment_step_based(df, dates, h)
        
        assert result["alignment_ok"], f"Alignment failed for h={h} (business-day)"
        assert result["n_misaligned"] == 0, f"Found {result['n_misaligned']} misaligned predictions for h={h}"


def test_alignment_misaligned():
    """Test that misaligned predictions are detected."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    h = 6
    
    # Create misaligned predictions (target_date = origin_date + h+1 instead of h)
    predictions = []
    for i in range(h + 1, len(dates)):
        origin_date = dates[i - h]
        target_date = dates[i]  # Wrong: should be dates[i-h+h] = dates[i-h+h] but we use dates[i]
        # Actually make it wrong: use dates[i+1]
        target_date = dates[min(i + 1, len(dates) - 1)]
        predictions.append({
            "origin_date": origin_date,
            "target_date": target_date,
            "horizon": h,
            "y_true": 1.0,
            "y_pred": 1.0,
        })
    
    df = pd.DataFrame(predictions)
    result = validate_alignment_step_based(df, dates, h)
    
    assert not result["alignment_ok"], "Should detect misalignment"
    assert result["n_misaligned"] > 0, "Should find misaligned predictions"


def test_shift_diagnostic_with_lag0():
    """Test shift diagnostic when lag_0 is present (should NOT flag lag_0 issue)."""
    for h in [1, 2, 5, 6, 10]:
        # Create synthetic predictions with lag_0 (no shift artifact)
        n = 100
        y_true = np.random.randn(n) * 10 + 100
        # Predictions are close to true values (lag_0 available)
        y_pred = y_true + np.random.randn(n) * 2
        
        result = shift_diagnostic_horizon_aware(y_true, y_pred, h)
        
        # With lag_0, best_shift should be close to 0, not -(h+1)
        assert abs(result["best_shift"]) <= 1, f"best_shift={result['best_shift']} should be ≈0 for h={h} with lag_0"
        assert not result["is_lag0_issue"], f"Should NOT flag lag_0 issue for h={h} when lag_0 is present"


def test_shift_diagnostic_without_lag0():
    """Test shift diagnostic when lag_0 is missing (SHOULD flag lag_0 issue)."""
    for h in [1, 2, 5, 6, 10]:
        # Create synthetic predictions WITHOUT lag_0 (model uses lag_1)
        # This creates best_shift ≈ -(h+1) pattern
        n = 100
        y_true = np.random.randn(n) * 10 + 100
        
        # Simulate lag_1 behavior: predictions are shifted by -(h+1)
        shift = -(h + 1)
        if abs(shift) < n:
            y_pred = np.roll(y_true, shift)
            y_pred[:abs(shift)] = y_true[:abs(shift)]  # Keep first values
        else:
            y_pred = y_true.copy()
        
        result = shift_diagnostic_horizon_aware(y_true, y_pred, h)
        
        # Without lag_0, should detect lag_0 issue pattern
        assert result["is_lag0_issue"] or abs(result["best_shift"] - (-(h + 1))) <= 1, \
            f"Should detect lag_0 issue for h={h}: best_shift={result['best_shift']}, expected≈{-(h+1)}"


def test_leakage_detection():
    """Test that leakage is detected in features."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    target = pd.Series(np.random.randn(n), index=dates)
    h = 6
    
    # Create features WITHOUT leakage
    X_good = pd.DataFrame({
        "lag_0": target,
        "lag_1": target.shift(1),
        "lag_7": target.shift(7),
        "rmean_7": target.rolling(7).mean().shift(1),
    }, index=dates)
    
    # Create features WITH leakage (negative shift)
    X_bad = pd.DataFrame({
        "lag_0": target,
        "lag_1": target.shift(1),
        "future_feature": target.shift(-h),  # LEAKAGE: uses future values
    }, index=dates)
    
    result_good = check_feature_leakage(X_good, target, h)
    result_bad = check_feature_leakage(X_bad, target, h)
    
    assert not result_good["leakage_detected"], "Should NOT detect leakage in good features"
    assert result_bad["leakage_detected"], "Should detect leakage in bad features"


def test_persistence_baseline():
    """Test persistence baseline computation."""
    predictions = pd.DataFrame({
        "origin_value": [100, 110, 105, 120],
        "y_true": [102, 112, 107, 122],
        "y_pred": [101, 111, 106, 121],
    })
    
    result = compute_persistence_baseline(predictions)
    
    assert not np.isnan(result["mae_persistence"]), "Should compute persistence MAE"
    assert result["mae_persistence"] > 0, "Persistence MAE should be positive"
    assert result["n_valid"] == 4, "Should have 4 valid rows"


def test_skill_score():
    """Test skill score computation."""
    # Model better than baseline
    skill = compute_skill_score(mae_model=10.0, mae_baseline=20.0)
    assert skill == 50.0, f"Expected skill=50%, got {skill}%"
    
    # Model worse than baseline
    skill = compute_skill_score(mae_model=20.0, mae_baseline=10.0)
    assert skill == -100.0, f"Expected skill=-100%, got {skill}%"
    
    # Model equal to baseline
    skill = compute_skill_score(mae_model=10.0, mae_baseline=10.0)
    assert skill == 0.0, f"Expected skill=0%, got {skill}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
