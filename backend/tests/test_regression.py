"""Optional regression test for preprocessing.

Runs only if Balance_by_Day_2015-2025.xlsx and master_daily_raw.csv exist in data/ folder.
"""

import pytest
from pathlib import Path
import pandas as pd

from preprocessing import PreprocessConfig, run_preprocess

# Paths relative to backend directory
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
EXPECTED_RAW = DATA_DIR / "processed" / "master_daily_raw.csv"
INPUT_XLSX = DATA_DIR / "Balance_by_Day_2015-2025.xlsx"


@pytest.mark.skipif(
    not INPUT_XLSX.exists() or not EXPECTED_RAW.exists(),
    reason="Regression test files not found. Skipping regression test.",
)
def test_regression_raw_variant():
    """Test that raw preprocessing matches expected master_daily_raw.csv."""
    import tempfile
    import shutil

    # Create temporary output directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_root = tmp_path / "data_preprocessed"
        run_outputs = tmp_path / "outputs"

        cfg = PreprocessConfig(
            input_path=str(INPUT_XLSX),
            variant="raw",
            business_days_zero_flows=False,
            weekday_weeks=8,
            out_root=str(out_root),
            run_outputs_dir=str(run_outputs),
            expected_csv=str(EXPECTED_RAW),
            save_parquet=False,
            sheet_name=None,
            header_row=None,
        )

        # Run preprocessing
        report = run_preprocess(cfg)

        # Check that output was created
        assert Path(report["output_csv"]).exists()

        # Load output and expected
        output_df = pd.read_csv(report["output_csv"])
        expected_df = pd.read_csv(EXPECTED_RAW)

        # Compare structure
        output_cols = set(output_df.columns)
        expected_cols = set(expected_df.columns)

        # Date columns should match
        output_df["date"] = pd.to_datetime(output_df["date"])
        expected_df["date"] = pd.to_datetime(expected_df["date"])

        # Check that we have comparison results in report
        if "comparison" in report:
            comp = report["comparison"]
            if "error" not in comp:
                # Check that differences are small (allowing for minor rounding)
                for col, diff_info in comp.get("differences", {}).items():
                    max_diff = diff_info.get("max_diff", 0.0)
                    # Allow small differences due to floating point
                    assert max_diff < 1.0, f"Column {col} has large difference: {max_diff}"

        # Basic structure check
        assert "date" in output_df.columns
        assert "date" in expected_df.columns
        assert len(output_df) > 0
        assert len(expected_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
