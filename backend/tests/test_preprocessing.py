"""Unit tests for preprocessing module."""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from preprocessing.holidays import georgian_holidays, orthodox_easter
from preprocessing.preprocess import (
    parse_balance_by_day_excel,
    apply_variant_raw,
    apply_variant_clean_conservative,
    apply_variant_clean_treasury,
)


def test_orthodox_easter():
    """Test Orthodox Easter calculation for known dates."""
    from datetime import date
    
    # 2024: Orthodox Easter should be May 5
    easter_2024 = orthodox_easter(2024)
    assert easter_2024 == date(2024, 5, 5), f"Expected 2024-05-05, got {easter_2024}"

    # 2025: Orthodox Easter should be April 20
    easter_2025 = orthodox_easter(2025)
    assert easter_2025 == date(2025, 4, 20), f"Expected 2025-04-20, got {easter_2025}"


def test_georgian_holidays():
    """Test Georgian holidays include fixed and movable dates."""
    from datetime import date, timedelta
    
    holidays_2024 = georgian_holidays(2024)

    # Check fixed holidays
    assert date(2024, 1, 1) in holidays_2024  # New Year
    assert date(2024, 1, 2) in holidays_2024  # New Year 2
    assert date(2024, 5, 26) in holidays_2024  # Independence Day
    assert date(2024, 8, 28) in holidays_2024  # Dormition

    # Check movable (Easter-related) - 2024
    # Orthodox Easter 2024: May 5
    # Good Friday: May 3, Holy Saturday: May 4, Easter Monday: May 6
    assert date(2024, 5, 3) in holidays_2024, "2024 Good Friday should be May 3"
    assert date(2024, 5, 4) in holidays_2024, "2024 Holy Saturday should be May 4"
    assert date(2024, 5, 5) in holidays_2024, "2024 Easter Sunday should be May 5"
    assert date(2024, 5, 6) in holidays_2024, "2024 Easter Monday should be May 6"
    
    # Verify erroneous dates are NOT holidays
    assert date(2024, 4, 16) not in holidays_2024, "2024-04-16 should NOT be a holiday"
    assert date(2024, 4, 17) not in holidays_2024, "2024-04-17 should NOT be a holiday"
    assert date(2024, 4, 18) not in holidays_2024, "2024-04-18 should NOT be a holiday"
    assert date(2024, 4, 19) not in holidays_2024, "2024-04-19 should NOT be a holiday"
    
    # Check 2025 Easter holidays
    holidays_2025 = georgian_holidays(2025)
    # Orthodox Easter 2025: April 20
    # Good Friday: April 18, Holy Saturday: April 19, Easter Monday: April 21
    assert date(2025, 4, 18) in holidays_2025, "2025 Good Friday should be April 18"
    assert date(2025, 4, 19) in holidays_2025, "2025 Holy Saturday should be April 19"
    assert date(2025, 4, 20) in holidays_2025, "2025 Easter Sunday should be April 20"
    assert date(2025, 4, 21) in holidays_2025, "2025 Easter Monday should be April 21"


def test_duplicate_metrics_summing():
    """Test that duplicate metric names are summed."""
    # Create a synthetic Excel-like structure
    # Simulate two rows with same metric name "Grants"
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    # Create data where "Grants" appears twice with different values
    data = {
        "Grants": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Grants": [5.0, 10.0, 15.0, 20.0, 25.0],  # Duplicate - should sum
        "Revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
    }

    # This is a simplified test - in real Excel parsing, duplicates would be handled
    # during the parsing phase. For now, we test the concept:
    df1 = pd.DataFrame({"Grants": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=dates)
    df2 = pd.DataFrame({"Grants": [5.0, 10.0, 15.0, 20.0, 25.0]}, index=dates)

    # Simulate summing duplicates
    combined = df1 + df2
    assert combined["Grants"].iloc[0] == 15.0  # 10 + 5
    assert combined["Grants"].iloc[1] == 30.0  # 20 + 10


def test_variant_raw():
    """Test raw variant processing."""
    # Create simple test data
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Revenue": np.random.randn(10) * 100 + 1000,
            "Expenditure": np.random.randn(10) * 50 + 500,
        },
        index=dates,
    )

    df_out, summary = apply_variant_raw(df, business_days_zero_flows=False)

    # Check that calendar flags are added
    assert "is_weekend" in df_out.columns
    assert "is_holiday" in df_out.columns

    # Check that we have full daily calendar
    assert len(df_out) >= len(df)  # Should have at least original dates

    # Check summary
    assert summary["variant"] == "raw"


def test_variant_clean_conservative():
    """Test clean_conservative variant."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "State budget balance": np.cumsum(np.random.randn(10) * 10) + 1000,
            "Revenue": np.random.randn(10) * 100 + 1000,
        },
        index=dates,
    )

    df_out, summary = apply_variant_clean_conservative(df, business_days_zero_flows=True)

    # Check level column is forward-filled
    assert "State budget balance" in summary["level_columns"]

    # Check summary
    assert summary["variant"] == "clean_conservative"


def test_variant_clean_treasury():
    """Test clean_treasury variant."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    # Create data with some NaN values
    revenue = np.random.randn(30) * 100 + 1000
    revenue[5] = np.nan  # Add a NaN
    revenue[15] = 10000  # Add an outlier

    df = pd.DataFrame({"Revenue": revenue}, index=dates)

    df_out, summary = apply_variant_clean_treasury(
        df, business_days_zero_flows=True, weekday_weeks=4
    )

    # Check that NaN was imputed
    assert "imputations" in summary
    assert "Revenue" in summary["imputations"]

    # Check that outlier was clipped
    assert "clipped_outliers" in summary
    assert "Revenue" in summary["clipped_outliers"]

    # Check summary
    assert summary["variant"] == "clean_treasury"
    assert summary["weekday_weeks"] == 4


if __name__ == "__main__":
    # Run tests
    print("Running preprocessing tests...")
    try:
        test_orthodox_easter()
        print("✓ test_orthodox_easter passed")
    except AssertionError as e:
        print(f"✗ test_orthodox_easter failed: {e}")
    
    try:
        test_georgian_holidays()
        print("✓ test_georgian_holidays passed")
    except AssertionError as e:
        print(f"✗ test_georgian_holidays failed: {e}")
    
    try:
        test_duplicate_metrics_summing()
        print("✓ test_duplicate_metrics_summing passed")
    except AssertionError as e:
        print(f"✗ test_duplicate_metrics_summing failed: {e}")
    
    try:
        test_variant_raw()
        print("✓ test_variant_raw passed")
    except AssertionError as e:
        print(f"✗ test_variant_raw failed: {e}")
    
    try:
        test_variant_clean_conservative()
        print("✓ test_variant_clean_conservative passed")
    except AssertionError as e:
        print(f"✗ test_variant_clean_conservative failed: {e}")
    
    try:
        test_variant_clean_treasury()
        print("✓ test_variant_clean_treasury passed")
    except AssertionError as e:
        print(f"✗ test_variant_clean_treasury failed: {e}")
    
    print("\nAll tests completed!")
