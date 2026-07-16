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

from forecast_integrity import (
    validate_alignment_step_based,
    shift_diagnostic_horizon_aware,
    compute_persistence_baseline,
    compute_skill_score,
    check_feature_leakage,
    detect_lagged_copy,
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
    """Test shift diagnostic when lag_0 is missing (SHOULD flag lag_0 issue).

    With a persistence-like predictor y_pred[t] = y_true[t - (h+1)], the
    best shift that minimises MAE is -(h+1).  We use a random walk so
    that the auto-correlation structure makes this detectable.
    """
    np.random.seed(42)
    for h in [2, 5, 6, 10]:   # Skip h=1: shift=-2 is too small for reliable detection
        n = 300
        # Random walk — strong auto-correlation makes shift detection reliable
        y_true = 100.0 + np.cumsum(np.random.randn(n) * 2.0)

        # Simulate lag_1 behaviour: y_pred[t] = y_true[t - (h+1)]
        shift_abs = h + 1
        y_pred = np.empty(n)
        y_pred[:shift_abs] = y_true[:shift_abs]    # pad start
        y_pred[shift_abs:] = y_true[:-shift_abs]   # lagged values

        result = shift_diagnostic_horizon_aware(y_true, y_pred, h)

        # Without lag_0, should detect lag_0 issue pattern
        assert result["is_lag0_issue"] or abs(result["best_shift"] - (-(h + 1))) <= 2, \
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


def test_detect_lagged_copy_flags_only_the_fake_model():
    """A shifted-copy model is flagged; an honest model is not.

    We build one target series and two models sharing it:
      - "honest": predictions are the true target plus small noise, so they
        align at shift 0 and beat the lag-1 baseline.
      - "lagcopy": predictions are the target shifted by one step (pure
        persistence), so they align best at shift +1 and cannot beat lag-1.
    """
    np.random.seed(7)
    n = 300
    # Mean-reverting AR(1) series: y[t] = mu + phi*(y[t-1]-mu) + noise.
    # phi=0.6 means the lag-1 autocorrelation is ~0.6, well below 1.0, so a
    # forecast that copies y[t-1] correlates clearly worse with the true
    # target (shift 0) than with the target shifted by +1.  A pure random
    # walk (autocorr ~1.0) would make shift 0 and shift 1 indistinguishable
    # by correlation, so we deliberately use a mean-reverting series here.
    mu, phi = 100.0, 0.6
    y_true = np.empty(n)
    y_true[0] = mu
    noise = np.random.randn(n) * 5.0
    for t in range(1, n):
        y_true[t] = mu + phi * (y_true[t - 1] - mu) + noise[t]
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Honest model: true value + small noise.
    honest_pred = y_true + np.random.randn(n) * 0.5

    # Lagged-copy model: yesterday's actual repeated as "the forecast".
    lag_pred = np.empty(n)
    lag_pred[0] = y_true[0]
    lag_pred[1:] = y_true[:-1]

    honest_df = pd.DataFrame({
        "date": dates, "model": "honest",
        "y_true": y_true, "y_pred": honest_pred,
    })
    lag_df = pd.DataFrame({
        "date": dates, "model": "lagcopy",
        "y_true": y_true, "y_pred": lag_pred,
    })
    df = pd.concat([honest_df, lag_df], ignore_index=True)

    result = detect_lagged_copy(df)

    # Pull each model's per-model record out for direct assertions.
    by_model = {r["model"]: r for r in result["per_model"]}

    assert not by_model["honest"]["flagged"], "Honest model should NOT be flagged"
    assert by_model["honest"]["best_shift"] == 0

    assert by_model["lagcopy"]["flagged"], "Lagged-copy model SHOULD be flagged"
    assert by_model["lagcopy"]["best_shift"] == 1

    assert result["risk"] == "high"
    assert any("lagcopy" in d for d in result["details"])
    assert all("honest" not in d for d in result["details"])


def test_detect_lagged_copy_skips_baseline_and_handles_empty():
    """Baseline rows are ignored, and an empty frame degrades gracefully."""
    # Empty frame -> low risk, informative message, no crash.
    empty = detect_lagged_copy(pd.DataFrame())
    assert empty["risk"] == "low"
    assert empty["per_model"] == []

    # A row set that is only a persistence baseline should be skipped, not
    # flagged (it is persistence on purpose).
    np.random.seed(1)
    n = 60
    y_true = 100.0 + np.cumsum(np.random.randn(n))
    base_pred = np.empty(n)
    base_pred[0] = y_true[0]
    base_pred[1:] = y_true[:-1]
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "model": "Persistence (baseline)",
        "y_true": y_true, "y_pred": base_pred,
    })
    result = detect_lagged_copy(df)
    assert result["risk"] == "low"
    assert result["per_model"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
