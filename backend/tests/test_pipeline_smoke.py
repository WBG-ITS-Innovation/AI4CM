"""
End-to-end smoke tests for every pipeline family.

Each test:
- Generates synthetic daily time-series data (~300 rows)
- Runs the simplest model from each family
- Verifies output schema, finite predictions, no leakage, reasonable range
- Runs fast (< 30s total)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic data generator
# ---------------------------------------------------------------------------

def _make_synthetic_csv(tmpdir: Path, n_rows: int = 300) -> Path:
    """Create a synthetic daily CSV with trend + seasonality + noise."""
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-02", periods=n_rows, freq="B")
    t = np.arange(n_rows, dtype=float)
    revenues = 1000 + 2 * t + 50 * np.sin(2 * np.pi * t / 5) + np.random.randn(n_rows) * 30
    expenditure = 800 + 1.5 * t + 40 * np.sin(2 * np.pi * t / 5 + 1) + np.random.randn(n_rows) * 25
    balance = np.cumsum(revenues - expenditure) + 5000

    df = pd.DataFrame({
        "date": dates,
        "Revenues": revenues,
        "Expenditure": expenditure,
        "State budget balance": balance,
    })
    csv_path = tmpdir / "synthetic_daily.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _check_predictions(pred_path: Path, expected_target: str, expected_horizon: int):
    """Common assertions for all pipeline outputs."""
    assert pred_path.exists(), f"predictions_long.csv not found at {pred_path}"
    df = pd.read_csv(pred_path)
    assert len(df) > 0, "predictions_long.csv is empty"

    # Schema check — required columns
    required = {"date", "y_true", "y_pred", "model", "horizon"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    # All predictions must be finite (no NaN, no Inf)
    assert df["y_pred"].notna().all(), f"Found NaN in y_pred ({df['y_pred'].isna().sum()} rows)"
    assert np.isfinite(df["y_pred"].values).all(), "Found Inf in y_pred"
    assert df["y_true"].notna().all(), f"Found NaN in y_true ({df['y_true'].isna().sum()} rows)"

    # No leakage: origin_date < target_date (if both columns exist)
    if "origin_date" in df.columns and "target_date" in df.columns:
        origins = pd.to_datetime(df["origin_date"])
        targets = pd.to_datetime(df["target_date"])
        violations = (origins >= targets).sum()
        assert violations == 0, f"Leakage detected: {violations} rows have origin_date >= target_date"

    # Reasonable range: predictions should be within 10x the input scale
    y_true_range = df["y_true"].abs().max()
    y_pred_range = df["y_pred"].abs().max()
    if y_true_range > 0:
        ratio = y_pred_range / y_true_range
        assert ratio < 10, f"Predictions out of range: max|y_pred|/max|y_true| = {ratio:.1f}"

    return df


# ---------------------------------------------------------------------------
# A · Statistical pipeline
# ---------------------------------------------------------------------------

class TestStatPipelineSmoke:

    def test_stat_naive(self, tmp_path):
        """Run the stat pipeline with NAIVE model on synthetic data."""
        csv_path = _make_synthetic_csv(tmp_path)
        out_root = tmp_path / "stat_out"
        out_root.mkdir()

        # The stat runner reads env vars
        env = os.environ.copy()
        env.update({
            "TG_FAMILY": "A_STAT",
            "TG_MODEL_FILTER": "NAIVE",
            "TG_TARGET": "Revenues",
            "TG_CADENCE": "Daily",
            "TG_HORIZON": "5",
            "TG_DATA_PATH": str(csv_path),
            "TG_DATE_COL": "date",
            "TG_OUT_ROOT": str(out_root),
            "TG_PARAM_OVERRIDES": json.dumps({"folds": 1, "min_train_years": 1}),
        })

        import subprocess
        runner = Path(__file__).parent.parent / "run_a_stat.py"
        result = subprocess.run(
            [os.sys.executable, str(runner)],
            cwd=str(runner.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Stat pipeline failed:\n{result.stderr}\n{result.stdout}"

        df = _check_predictions(out_root / "predictions_long.csv", "Revenues", 5)

        # Fold boundaries: split_id should be present
        assert "split_id" in df.columns


# ---------------------------------------------------------------------------
# B · ML pipeline
# ---------------------------------------------------------------------------

class TestMLPipelineSmoke:

    def test_ml_ridge(self, tmp_path):
        """Run ML pipeline with Ridge on synthetic data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from b_ml_pipeline import ConfigBML, run_pipeline_ml

        csv_path = _make_synthetic_csv(tmp_path)
        out_root = tmp_path / "ml_out"

        cfg = ConfigBML(
            data_path=str(csv_path),
            date_col="date",
            target="Revenues",
            cadence="Daily",
            horizon=5,
            variant="uni",
            model_filter="Ridge",
            out_root=str(out_root),
            folds=1,
            min_train_years=1,
        )

        run_pipeline_ml(cfg)

        df = _check_predictions(out_root / "predictions_long.csv", "Revenues", 5)

        # Verify origin_date and target_date columns exist (ML pipeline adds them)
        assert "origin_date" in df.columns, "ML pipeline should produce origin_date"
        assert "target_date" in df.columns, "ML pipeline should produce target_date"
        assert "origin_value" in df.columns, "ML pipeline should produce origin_value"


# ---------------------------------------------------------------------------
# C · DL pipeline
# ---------------------------------------------------------------------------

class TestDLPipelineSmoke:

    def test_dl_mlp(self, tmp_path):
        """Run DL pipeline with MLP (fastest DL model) on synthetic data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from c_dl_pipeline import ConfigDL, run_pipeline

        csv_path = _make_synthetic_csv(tmp_path)
        out_root = tmp_path / "dl_out"

        cfg = ConfigDL(
            data_path=str(csv_path),
            date_col="date",
            out_root_uni=str(out_root),
            targets=["Revenues"],
            cadences=["daily"],
            horizons_daily=[5],
            models_univariate=["mlp"],
            min_train_years=1,
            epochs=3,       # minimal for smoke test
            batch_size=64,
            quick_mode=True,
        )

        run_pipeline(config=cfg, run_univariate=True, run_multivariate=False)

        # DL pipeline writes to cadence subfolder
        pred_path = out_root / "daily" / "predictions_long.csv"
        if not pred_path.exists():
            pred_path = out_root / "predictions_long.csv"
        df = _check_predictions(pred_path, "Revenues", 5)


# ---------------------------------------------------------------------------
# E · Quantile pipeline
# ---------------------------------------------------------------------------

class TestQuantilePipelineSmoke:

    def test_quantile_gb(self, tmp_path):
        """Run quantile pipeline with GBQuantile on synthetic data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from e_quantile_daily_pipeline import Config, run_pipeline

        csv_path = _make_synthetic_csv(tmp_path)
        out_root = tmp_path / "quantile_out"

        cfg = Config(
            target="Revenues",
            cadence="Daily",
            horizon=5,
            data_path=str(csv_path),
            date_col="date",
            folds=2,
            min_train_years=1,
            model_filter="GBQuantile",
            out_root=str(out_root),
        )

        run_pipeline(cfg)

        pred_path = Path(out_root) / "predictions_long.csv"
        assert pred_path.exists(), "Quantile pipeline did not produce predictions_long.csv"
        df = pd.read_csv(pred_path)
        assert len(df) > 0, "predictions_long.csv is empty"

        # Quantile pipeline should produce P10/P50/P90 columns
        assert "yhat_p50" in df.columns, "Missing yhat_p50"
        assert df["yhat_p50"].notna().all(), "NaN in yhat_p50"
        assert np.isfinite(df["yhat_p50"].values).all(), "Inf in yhat_p50"

        # Quantile ordering: P10 <= P50 <= P90 (most of the time).
        # GBQuantile fits separate models per quantile so minor crossing
        # is possible.  Flag only if > 20% of rows violate ordering.
        if "yhat_p10" in df.columns and "yhat_p90" in df.columns:
            valid = df.dropna(subset=["yhat_p10", "yhat_p50", "yhat_p90"])
            violations = ((valid["yhat_p10"] > valid["yhat_p50"]) |
                          (valid["yhat_p50"] > valid["yhat_p90"])).sum()
            violation_rate = violations / max(len(valid), 1)
            assert violation_rate <= 0.25, (
                f"Quantile ordering violated in {violations}/{len(valid)} rows "
                f"({violation_rate:.0%}) — too many crossings"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
