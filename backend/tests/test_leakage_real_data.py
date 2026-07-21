"""
Leakage detection test on real Georgia Treasury data.

Runs the stat pipeline (NAIVE — fastest model) on the actual processed
CSV, then verifies every prediction row for temporal leakage signals.
Skips gracefully if the data file is not available.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent
DATA_FILE = BACKEND_DIR / "data" / "processed" / "master_daily_clean_conservative.csv"

pytestmark = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason=f"Real data file not found: {DATA_FILE}",
)


@pytest.fixture(scope="module")
def run_output(tmp_path_factory):
    """Run the stat pipeline once and return the predictions DataFrame."""
    out_root = tmp_path_factory.mktemp("leakage_test")
    env = os.environ.copy()
    env.update({
        "TG_FAMILY": "A_STAT",
        "TG_MODEL_FILTER": "NAIVE",
        "TG_TARGET": "Revenues",
        "TG_CADENCE": "Daily",
        "TG_HORIZON": "5",
        "TG_DATA_PATH": str(DATA_FILE),
        "TG_DATE_COL": "date",
        "TG_OUT_ROOT": str(out_root),
        "TG_PARAM_OVERRIDES": json.dumps({"folds": 1, "min_train_years": 4}),
    })

    runner = BACKEND_DIR / "run_a_stat.py"
    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stdout}\n{result.stderr}"

    pred_path = out_root / "predictions_long.csv"
    assert pred_path.exists(), "No predictions_long.csv produced"
    df = pd.read_csv(pred_path, parse_dates=["date", "origin_date", "target_date"])
    assert len(df) > 0, "Predictions file is empty"
    return df


class TestTemporalOrdering:

    def test_origin_before_target(self, run_output):
        """Every origin_date must be strictly before target_date."""
        df = run_output
        violations = (df["origin_date"] >= df["target_date"]).sum()
        assert violations == 0, (
            f"{violations} rows have origin_date >= target_date"
        )

    def test_origin_at_least_one_day_before_target(self, run_output):
        """origin_date should be at least 1 business day before target_date.

        Note: the stat pipeline forecasts the full test window from a single
        origin (train_end), so origin-to-target gaps range from 1 day to a
        full year.  The h-step-ahead guarantee (gap >= horizon) applies only
        to ML/DL/quantile pipelines that produce per-origin predictions.
        For the stat pipeline we verify the weaker but still critical
        invariant: no prediction references future data (origin < target).
        """
        df = run_output
        for _, row in df.head(100).iterrows():
            bdays_between = len(
                pd.bdate_range(start=row["origin_date"], end=row["target_date"])
            ) - 1  # bdate_range is inclusive on both ends
            assert bdays_between >= 1, (
                f"Only {bdays_between} bdays between origin {row['origin_date'].date()} "
                f"and target {row['target_date'].date()} (need >= 1)"
            )


class TestPredictionQuality:

    def test_predictions_not_exact_match(self, run_output):
        """y_pred should not exactly equal y_true (would indicate leakage)."""
        df = run_output
        exact_matches = (df["y_pred"] == df["y_true"]).sum()
        match_rate = exact_matches / len(df)
        assert match_rate < 0.05, (
            f"{exact_matches}/{len(df)} predictions ({match_rate:.1%}) exactly "
            f"equal y_true — likely leakage"
        )

    def test_no_shift_leakage(self, run_output):
        """
        Check that y_pred isn't just y_true shifted by the horizon.

        If corr(y_pred, y_true.shift(h)) is suspiciously close to 1.0,
        the model is likely copying future actuals with a shift.
        """
        df = run_output.sort_values("date").reset_index(drop=True)
        horizon = int(df["horizon"].iloc[0])

        for shift in range(0, min(horizon + 3, 11)):
            y_true_shifted = df["y_true"].shift(shift)
            valid = y_true_shifted.notna() & df["y_pred"].notna()
            if valid.sum() < 20:
                continue
            corr = np.corrcoef(
                df.loc[valid, "y_pred"].values,
                y_true_shifted[valid].values,
            )[0, 1]
            if shift == 0:
                assert corr < 0.99, (
                    f"corr(y_pred, y_true) = {corr:.4f} — suspiciously high, "
                    f"model may be copying actuals"
                )
            else:
                # A shift of exactly horizon with corr > 0.98 is suspicious
                if shift == horizon and corr > 0.98:
                    pytest.fail(
                        f"corr(y_pred, y_true.shift({shift})) = {corr:.4f} — "
                        f"predictions appear to be y_true shifted by horizon"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
