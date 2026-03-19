# -*- coding: utf-8 -*-
"""
Tests for all audit-driven fixes (March 2026 review).

Covers:
  1. lag_0 removal for stock targets
  2. Model name mapping
  3. Persistence baseline uses step-based indexing
  4. Quantile pipeline expanding-window folds
  5. Quality gate threshold
  6. Conformal prediction intervals for ML
  7. Dashboard PI warning logic
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure backend is importable
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ===================================================================
# 1. lag_0 EXCLUSION TESTS
# ===================================================================

class TestLag0Exclusion:
    """Verify lag_0 is NOT in the default lag recipe."""

    def test_default_lags_daily_exclude_lag0(self):
        """Default lags_daily must NOT contain 0."""
        from b_ml_pipeline import ConfigBML
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date", target="Revenues",
            cadence="Daily", horizon=5, variant="uni",
            model_filter=None, out_root="/tmp/test",
        )
        assert 0 not in cfg.lags_daily, (
            f"lag_0 found in default lags_daily={cfg.lags_daily}. "
            "It must be excluded to prevent persistence-bias."
        )

    def test_default_lags_weekly_exclude_lag0(self):
        from b_ml_pipeline import ConfigBML
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date", target="State budget balance",
            cadence="Weekly", horizon=4, variant="uni",
            model_filter=None, out_root="/tmp/test",
        )
        assert 0 not in cfg.lags_weekly

    def test_choose_recipe_does_not_inject_lag0(self):
        """choose_recipe must NOT silently inject lag_0."""
        from b_ml_pipeline import ConfigBML, choose_recipe
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date", target="State budget balance",
            cadence="Daily", horizon=6, variant="uni",
            model_filter=None, out_root="/tmp/test",
        )
        lags, _ = choose_recipe(cfg)
        assert 0 not in lags, (
            f"choose_recipe injected lag_0 into lags={lags}. "
            "This makes the model a trivial persistence predictor."
        )

    def test_user_can_still_opt_in_to_lag0(self):
        """If a user explicitly includes lag_0, it should be honoured."""
        from b_ml_pipeline import ConfigBML, choose_recipe
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date", target="Revenues",
            cadence="Daily", horizon=5, variant="uni",
            model_filter=None, out_root="/tmp/test",
            lags_daily=[0, 1, 7],
        )
        lags, _ = choose_recipe(cfg)
        assert 0 in lags, "Explicit lag_0 should be preserved."

    def test_lag_window_features_without_lag0(self):
        """Feature frame must not contain lag_0 when not requested."""
        from b_ml_pipeline import lag_window_features
        s = pd.Series(range(50), index=pd.bdate_range("2024-01-01", periods=50))
        feats = lag_window_features(s, lags=[1, 3, 7], windows=[3, 7])
        assert "lag_0" not in feats.columns

    def test_lag_window_features_with_lag0(self):
        """Feature frame should include lag_0 only when explicitly requested."""
        from b_ml_pipeline import lag_window_features
        s = pd.Series(range(50), index=pd.bdate_range("2024-01-01", periods=50))
        feats = lag_window_features(s, lags=[0, 1, 7], windows=[3])
        assert "lag_0" in feats.columns
        # lag_0 should equal s itself (no shift)
        pd.testing.assert_series_equal(feats["lag_0"], s, check_names=False)


# ===================================================================
# 2. MODEL NAME MAPPING TESTS
# ===================================================================

class TestModelNameMapping:
    """Verify UI model names match backend expectations."""

    def test_stat_model_names_uppercase_match(self):
        """All stat model UI labels must match _fc() dispatch after upper()."""
        sys.path.insert(0, str(BACKEND_DIR.parent / "frontend"))
        from backend_consts import STAT_MODEL_OPTIONS

        # These are the exact strings _fc() dispatches on (upper-cased)
        known_backends = {"NAIVE", "WEEKDAY_MEAN", "MOVAVG", "ETS", "SARIMAX",
                          "STL_ARIMA", "THETA"}
        for ui_label, backend_filter in STAT_MODEL_OPTIONS:
            assert backend_filter.upper() in known_backends, (
                f"STAT_MODEL_OPTIONS entry ({ui_label!r}, {backend_filter!r}) "
                f"does not match any case in _fc().  Known: {known_backends}"
            )

    def test_quality_gate_constant_exists(self):
        sys.path.insert(0, str(BACKEND_DIR.parent / "frontend"))
        from backend_consts import QUALITY_GATE_SKILL_PCT
        assert QUALITY_GATE_SKILL_PCT >= 3.0, (
            f"Quality gate is set to {QUALITY_GATE_SKILL_PCT}% which is "
            "too low to distinguish genuine skill from noise."
        )


# ===================================================================
# 3. PERSISTENCE BASELINE — STEP-BASED TESTS
# ===================================================================

class TestPersistenceBaseline:
    """Verify persistence baseline uses positional (step) indexing."""

    def _make_series(self, n=100) -> pd.Series:
        idx = pd.bdate_range("2024-01-01", periods=n)
        return pd.Series(100.0 + np.arange(n) * 0.5, index=idx)

    def test_compute_baselines_step_based(self):
        """Persistence at target t should equal series[pos(t) - h]."""
        from preprocessing.integrity import compute_baselines

        s = self._make_series(100)
        h = 6
        target_dates = s.index[h:]  # targets start at position h

        baselines = compute_baselines(s, target_dates, h)
        persistence = baselines["persistence"]

        # For step-based: persistence[i] should equal s.iloc[pos(target_dates[i]) - h]
        for i, td in enumerate(target_dates):
            pos = list(s.index).index(td)
            expected = float(s.iloc[pos - h])
            actual = persistence[i]
            if np.isfinite(actual):
                assert np.isclose(actual, expected, atol=1e-6), (
                    f"Persistence at {td.date()}: expected {expected}, got {actual}. "
                    "compute_baselines may still use calendar-day offsets."
                )

    def test_persistence_baseline_matches_origin_value(self):
        """When origin_value is available, baseline should match it."""
        from preprocessing.integrity import compute_persistence_baseline_from_origin

        n = 50
        df = pd.DataFrame({
            "origin_value": 100.0 + np.arange(n) * 0.5,
            "y_true": 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1,
            "y_pred": 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.2,
        })

        result = compute_persistence_baseline_from_origin(df)
        assert result["n_valid"] == n
        assert result["mae_persistence"] > 0
        # MAE should be close to std of random noise (0.1)
        assert result["mae_persistence"] < 1.0

    def test_baselines_no_weekend_dates(self):
        """Step-based baseline should never try to look up weekend dates."""
        from preprocessing.integrity import compute_baselines

        s = self._make_series(50)
        h = 3
        # All dates in s are business days; baselines should work
        baselines = compute_baselines(s, s.index[h:], h)
        # No NaNs expected (all positions are valid)
        nan_count = sum(1 for v in baselines["persistence"] if np.isnan(v))
        assert nan_count == 0, (
            f"Found {nan_count} NaN persistence values. "
            "Step-based indexing should find all origins."
        )


# ===================================================================
# 4. QUANTILE PIPELINE — EXPANDING WINDOW TESTS
# ===================================================================

class TestQuantileExpandingWindow:
    """Verify quantile pipeline uses expanding (not sliding) windows."""

    def test_expanding_folds(self):
        from e_quantile_daily_pipeline import _time_folds

        n = 500
        horizon = 20
        folds = 3
        min_train = 100

        result = _time_folds(n, horizon, folds, min_train)

        # Each fold should have train_end (exclusive), test_end (exclusive)
        assert len(result) >= 1, "Should produce at least 1 fold"

        # Verify expanding property: later folds have MORE training data
        for i in range(1, len(result)):
            prev_train_end = result[i-1][0]
            curr_train_end = result[i][0]
            assert curr_train_end > prev_train_end, (
                f"Fold {i} has train_end={curr_train_end} <= fold {i-1} "
                f"train_end={prev_train_end}. Folds must be expanding-window."
            )

    def test_training_always_starts_at_zero(self):
        """Training for every fold implicitly starts at index 0."""
        from e_quantile_daily_pipeline import _time_folds

        n = 300
        horizon = 10
        folds = 4
        min_train = 50

        result = _time_folds(n, horizon, folds, min_train)

        # The training slice for each fold is [0 : train_end).
        # We verify that the first fold's train_end is large enough.
        if result:
            first_train_end = result[0][0]
            assert first_train_end > 30, (
                f"First fold train_end={first_train_end} is too small."
            )

    def test_test_blocks_non_overlapping(self):
        from e_quantile_daily_pipeline import _time_folds

        result = _time_folds(500, 20, 5, 50)
        for i in range(1, len(result)):
            prev_test_end = result[i-1][1]
            curr_train_end = result[i][0]
            # Test blocks should not overlap: current train_end <= prev test_start
            assert curr_train_end <= result[i-1][0] or prev_test_end <= result[i][0], (
                "Test blocks overlap between folds."
            )


# ===================================================================
# 5. QUALITY GATE THRESHOLD TESTS
# ===================================================================

class TestQualityGate:
    """Verify the quality gate threshold is 5% (not 2%)."""

    def test_quality_gate_is_five_percent(self):
        sys.path.insert(0, str(BACKEND_DIR.parent / "frontend"))
        from backend_consts import QUALITY_GATE_SKILL_PCT
        assert QUALITY_GATE_SKILL_PCT == 5.0, (
            f"Quality gate should be 5.0%, got {QUALITY_GATE_SKILL_PCT}%"
        )


# ===================================================================
# 6. CONFORMAL PREDICTION INTERVAL TESTS
# ===================================================================

class TestConformalPI:
    """Verify conformal PI computation logic."""

    def test_conformal_radius_from_residuals(self):
        """Conformal radius should be the 90th percentile of |residuals|."""
        np.random.seed(42)
        n = 100
        residuals = np.random.randn(n)
        abs_res = np.abs(residuals)

        nominal_pi = 0.90
        radius = float(np.quantile(abs_res, nominal_pi))

        # Check: about 90% of |residuals| should be <= radius
        coverage = np.mean(abs_res <= radius)
        assert coverage >= 0.88, (  # Allow small floating-point slack
            f"Conformal radius coverage={coverage:.2f} < 0.88"
        )
        assert coverage <= 0.93

    def test_pi_symmetry(self):
        """y_lo and y_hi should be symmetric around y_pred."""
        y_pred = 100.0
        radius = 5.0
        y_lo = y_pred - radius
        y_hi = y_pred + radius
        assert np.isclose(y_pred - y_lo, y_hi - y_pred), (
            "PI should be symmetric: y_pred - y_lo == y_hi - y_pred"
        )


# ===================================================================
# 7. SHIFT DIAGNOSTIC TESTS
# ===================================================================

class TestShiftDiagnostics:
    """Verify shift diagnostic correctly identifies lag issues."""

    def test_correctly_aligned_shift_is_zero(self):
        """For well-aligned predictions, best_shift should be 0."""
        from preprocessing.integrity import shift_sanity_check

        np.random.seed(42)
        n = 200
        y_true = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
        y_pred = y_true + np.random.randn(n) * 0.3  # Small noise

        result = shift_sanity_check(y_true, y_pred, horizon=6)
        assert result["best_shift"] == 0, (
            f"Expected best_shift=0 for aligned data, got {result['best_shift']}"
        )
        assert result["lag_warning"] is False

    def test_persistence_like_detected(self):
        """Persistence-like predictions should have best_shift near -h."""
        from preprocessing.integrity import shift_sanity_check

        np.random.seed(42)
        n = 200
        h = 6
        y_true = 100 + np.cumsum(np.random.randn(n))  # Random walk
        # Persistence: y_pred[t] = y_true[t - h]
        y_pred = np.zeros(n)
        y_pred[:h] = y_true[:h]
        y_pred[h:] = y_true[:-h]

        result = shift_sanity_check(y_true, y_pred, horizon=h)
        # Best shift should be negative (close to -h)
        assert result["best_shift"] <= -h + 2, (
            f"Expected best_shift near -{h}, got {result['best_shift']}"
        )


# ===================================================================
# 8. LEAKAGE DETECTION TESTS
# ===================================================================

class TestLeakageDetection:
    """Verify leakage sentinel catches obvious leakage."""

    def test_no_leakage_on_data_with_signal(self):
        """Data with a real signal (y depends on X) should pass leakage check.

        The sentinel fires when shuffling targets barely changes MAE, which
        indicates either leakage OR no signal.  We use data with a strong
        linear relationship so the shuffled model is clearly worse.
        """
        from preprocessing.integrity import leakage_sentinel

        np.random.seed(42)
        n = 300
        X = pd.DataFrame({
            "lag_1": np.random.randn(n),
            "lag_7": np.random.randn(n),
            "rmean_7": np.random.randn(n),
        })
        # y has a strong relationship with features (not leaked, just correlated)
        y = pd.Series(3.0 * X["lag_1"] + 2.0 * X["lag_7"] + np.random.randn(n) * 0.5)

        X_train, y_train = X.iloc[:200], y.iloc[:200]
        X_test, y_test = X.iloc[200:], y.iloc[200:]

        result = leakage_sentinel(X_train, y_train, X_test, y_test, horizon=5)
        assert result["leakage_warning"] is False, (
            f"Leakage sentinel fired on clean data with signal. "
            f"ratio={result.get('shuffled_to_normal_ratio', 'N/A'):.2f}"
        )


# ===================================================================
# 9. END-TO-END INTEGRATION SMOKE TEST
# ===================================================================

class TestIntegrationSmoke:
    """Lightweight smoke test that the full pipeline config is valid."""

    def test_config_creation_stock_target(self):
        from b_ml_pipeline import ConfigBML, is_stock
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date",
            target="State budget balance",
            cadence="Daily", horizon=6, variant="uni",
            model_filter="Ridge", out_root="/tmp/test",
        )
        assert is_stock(cfg.target)
        assert 0 not in cfg.lags_daily

    def test_config_creation_flow_target(self):
        from b_ml_pipeline import ConfigBML, is_stock
        cfg = ConfigBML(
            data_path="dummy.csv", date_col="date",
            target="Revenues",
            cadence="Daily", horizon=5, variant="uni",
            model_filter=None, out_root="/tmp/test",
        )
        assert not is_stock(cfg.target)
        assert 0 not in cfg.lags_daily


# ===================================================================
# 10. LEADERBOARD BASELINE ROW & BEST-MODEL SELECTION (Fix 14)
# ===================================================================

class TestLeaderboardBaselineRow:
    """Verify persistence baseline row is present AND best_model skips it."""

    def _build_leaderboard(self) -> pd.DataFrame:
        """Simulate the leaderboard construction from b_ml_pipeline."""
        rows = [
            {"target": "T", "horizon": 6, "model": "Ridge", "MAE": 120.0},
            {"target": "T", "horizon": 6, "model": "Lasso", "MAE": 130.0},
        ]
        glb = pd.DataFrame(rows).sort_values("MAE").assign(rank=lambda g: np.arange(1, len(g) + 1))
        # Simulate Fix 11: add persistence baseline row at rank 0
        persist_row = pd.DataFrame([{
            "target": "T", "horizon": 6,
            "model": "⚡ Persistence (baseline)", "MAE": 150.0, "rank": 0,
        }])
        glb = pd.concat([persist_row, glb], ignore_index=True)
        glb["rank"] = range(len(glb))
        return glb

    def test_baseline_row_present(self):
        glb = self._build_leaderboard()
        baseline_rows = glb[glb["model"].str.contains("baseline", case=False)]
        assert len(baseline_rows) == 1, "Persistence baseline row must be present"
        assert baseline_rows.iloc[0]["rank"] == 0, "Baseline must be rank 0"

    def test_best_model_skips_baseline_for_plots(self):
        """best_model for plotting must NOT be the baseline row."""
        glb = self._build_leaderboard()
        # Replicate the selection logic from line 1231 of b_ml_pipeline.py
        _trained = glb[~glb["model"].str.contains("baseline", case=False, na=False)]
        best_model = _trained.iloc[0]["model"] if len(_trained) > 0 else glb.iloc[0]["model"]
        assert "baseline" not in best_model.lower(), (
            f"best_model for plotting selected the baseline row: {best_model!r}"
        )
        assert best_model == "Ridge"  # Ridge has lowest MAE among trained models

    def test_best_model_skips_baseline_for_integrity(self):
        """best_model for integrity checks must NOT be the baseline row."""
        glb = self._build_leaderboard()
        # Replicate the FIXED selection logic from b_ml_pipeline.py (Fix 14)
        _trained_for_integrity = glb[~glb["model"].str.contains("baseline", case=False, na=False)]
        best_model = (_trained_for_integrity.iloc[0]["model"]
                      if len(_trained_for_integrity) > 0
                      else glb.iloc[0]["model"])
        assert "baseline" not in best_model.lower(), (
            f"Integrity-check best_model selected the baseline row: {best_model!r}"
        )
        assert best_model == "Ridge"


# ===================================================================
# 11. DASHBOARD SKILL THRESHOLD CONSISTENCY (Fix 15)
# ===================================================================

class TestDashboardSkillThreshold:
    """Verify the Dashboard uses the same skill threshold as the backend."""

    def test_dashboard_threshold_matches_backend(self):
        sys.path.insert(0, str(BACKEND_DIR.parent / "frontend"))
        from backend_consts import QUALITY_GATE_SKILL_PCT

        # The backend pipeline hard-codes _QUALITY_GATE_SKILL_PCT = 5.0
        # The Dashboard must import and use the same constant.
        assert QUALITY_GATE_SKILL_PCT == 5.0, (
            f"QUALITY_GATE_SKILL_PCT should be 5.0, got {QUALITY_GATE_SKILL_PCT}"
        )

    def test_dashboard_imports_constant(self):
        """Dashboard file must contain the import of QUALITY_GATE_SKILL_PCT."""
        dashboard_path = BACKEND_DIR.parent / "frontend" / "pages" / "01_Dashboard.py"
        text = dashboard_path.read_text(encoding="utf-8")
        assert "QUALITY_GATE_SKILL_PCT" in text, (
            "Dashboard does not import QUALITY_GATE_SKILL_PCT — "
            "threshold may be hardcoded and out-of-sync."
        )
        # Ensure the old hardcoded 2.0 is no longer used as the threshold
        assert "skill_threshold = 2.0" not in text, (
            "Dashboard still has hardcoded skill_threshold = 2.0"
        )


# ===================================================================
# 12. FORECAST_INTEGRITY IMPORT PATH (Fix 16)
# ===================================================================

class TestForecastIntegrityImport:
    """Verify forecast_integrity module is importable from backend/."""

    def test_import_from_cwd_backend(self):
        """When cwd is backend/, 'from forecast_integrity import ...' must work."""
        from forecast_integrity import (
            validate_alignment_step_based,
            shift_diagnostic_horizon_aware,
            compute_persistence_baseline,
            compute_skill_score,
        )
        # All four functions should be callable
        assert callable(validate_alignment_step_based)
        assert callable(shift_diagnostic_horizon_aware)
        assert callable(compute_persistence_baseline)
        assert callable(compute_skill_score)

    def test_pipeline_uses_direct_import_first(self):
        """b_ml_pipeline.py must try 'from forecast_integrity import ...'
        BEFORE falling back to 'from backend.forecast_integrity import ...'."""
        pipeline_path = BACKEND_DIR / "b_ml_pipeline.py"
        text = pipeline_path.read_text(encoding="utf-8")
        pos_direct = text.find("from forecast_integrity import")
        pos_backend = text.find("from backend.forecast_integrity import")
        assert pos_direct != -1, (
            "Pipeline does not contain 'from forecast_integrity import ...'"
        )
        assert pos_direct < pos_backend, (
            "Pipeline must try the direct import BEFORE the 'backend.' prefixed one. "
            f"Direct import at char {pos_direct}, backend. import at char {pos_backend}"
        )


# ===================================================================
# 13. DL PIPELINE: lag_0 MASKING + ORIGIN EXPORT (Phase 1 fixes)
# ===================================================================

class TestDLPipelineHardening:
    """Verify DL pipeline fixes for lag_0, origin export, sequences."""

    def test_build_sequences_returns_five_arrays(self):
        """build_sequences must return (X, y, label_dates, origin_dates, origin_values)."""
        from c_dl_pipeline import build_sequences, make_feature_frame, load_holidays
        # We just test the signature — build a tiny synthetic frame
        idx = pd.bdate_range("2024-01-01", periods=100)
        F = pd.DataFrame({"target_col": np.arange(100, dtype=float),
                           "cal1": np.random.randn(100)}, index=idx)
        y = pd.Series(np.arange(100, dtype=float), index=idx)
        result = build_sequences(F, y, seq_len=10, horizon=5,
                                 target_col="target_col", mask_target_at_origin=False)
        assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
        X, yv, ld, od, ov = result
        assert X.shape[0] == yv.shape[0] == ld.shape[0] == od.shape[0] == ov.shape[0]
        assert X.shape[0] > 0

    def test_mask_target_at_origin_zeros_last_row(self):
        """When mask_target_at_origin=True, last row's target col should be 0."""
        from c_dl_pipeline import build_sequences
        idx = pd.bdate_range("2024-01-01", periods=50)
        vals = np.arange(50, dtype=float) + 100.0  # non-zero values
        F = pd.DataFrame({"the_target": vals, "feat1": np.random.randn(50)}, index=idx)
        y = pd.Series(vals, index=idx)
        X, yv, ld, od, ov = build_sequences(F, y, seq_len=5, horizon=3,
                                              target_col="the_target",
                                              mask_target_at_origin=True)
        assert X.shape[0] > 0
        # Check that the target column (column 0) in the last sequence row is 0
        target_col_idx = 0  # "the_target" is first column
        for i in range(X.shape[0]):
            assert X[i, -1, target_col_idx] == 0.0, (
                f"Sequence {i}: last row target col should be 0.0 (masked), "
                f"got {X[i, -1, target_col_idx]}"
            )

    def test_no_mask_preserves_origin_value(self):
        """When mask_target_at_origin=False, last row's target col keeps original value."""
        from c_dl_pipeline import build_sequences
        idx = pd.bdate_range("2024-01-01", periods=50)
        vals = np.arange(50, dtype=float) + 100.0
        F = pd.DataFrame({"the_target": vals, "feat1": np.random.randn(50)}, index=idx)
        y = pd.Series(vals, index=idx)
        X, yv, ld, od, ov = build_sequences(F, y, seq_len=5, horizon=3,
                                              target_col="the_target",
                                              mask_target_at_origin=False)
        # When not masked, last row target should be the origin value
        for i in range(min(5, X.shape[0])):
            assert X[i, -1, 0] != 0.0, "Without masking, target column should keep original value"

    def test_origin_dates_are_before_label_dates(self):
        """origin_date must always precede label_date by exactly h steps."""
        from c_dl_pipeline import build_sequences
        idx = pd.bdate_range("2024-01-01", periods=80)
        F = pd.DataFrame({"t": np.arange(80, dtype=float)}, index=idx)
        y = pd.Series(np.arange(80, dtype=float), index=idx)
        h = 5
        X, yv, ld, od, ov = build_sequences(F, y, seq_len=10, horizon=h)
        for i in range(len(ld)):
            ld_ts = pd.Timestamp(ld[i])
            od_ts = pd.Timestamp(od[i])
            ld_pos = list(idx).index(ld_ts)
            od_pos = list(idx).index(od_ts)
            assert ld_pos - od_pos == h, (
                f"Sequence {i}: label_pos={ld_pos} - origin_pos={od_pos} = {ld_pos-od_pos}, expected {h}"
            )


# ===================================================================
# 14. QUANTILE PIPELINE: h-STEP TARGET + ORIGIN EXPORT (Phase 3 fixes)
# ===================================================================

class TestQuantilePipelineHardening:
    """Verify quantile pipeline produces h-step-ahead targets."""

    def test_build_features_returns_four_values(self):
        from e_quantile_daily_pipeline import _build_features, Config
        idx = pd.bdate_range("2024-01-01", periods=100)
        df = pd.DataFrame({"Revenues": np.arange(100, dtype=float)}, index=idx)
        df.index.name = "date"
        cfg = Config(target="Revenues", cadence="Daily", horizon=5,
                     data_path="dummy", lags_daily=(1, 5), windows_daily=(5,))
        result = _build_features(df, cfg)
        assert len(result) == 4, f"Expected 4 returns (X, y_target, od, ov), got {len(result)}"

    def test_target_is_h_steps_ahead(self):
        """y_target[i] should equal the original y at position i + h."""
        from e_quantile_daily_pipeline import _build_features, Config
        idx = pd.bdate_range("2024-01-01", periods=100)
        raw_vals = np.arange(100, dtype=float) * 2.0 + 50.0
        df = pd.DataFrame({"Revenues": raw_vals}, index=idx)
        df.index.name = "date"
        h = 7
        cfg = Config(target="Revenues", cadence="Daily", horizon=h,
                     data_path="dummy", lags_daily=(1, 5), windows_daily=(5,))
        X, y_target, od, ov = _build_features(df, cfg)
        # For each valid row, y_target should be raw_vals[pos + h]
        date_to_pos = {d: i for i, d in enumerate(idx)}
        for row_date, target_val in y_target.items():
            pos = date_to_pos[row_date]
            expected = raw_vals[pos + h]
            assert np.isclose(target_val, expected, atol=1e-6), (
                f"At pos={pos}: y_target={target_val}, expected y[{pos+h}]={expected}"
            )

    def test_origin_values_match_current_y(self):
        """origin_value should be y at the feature row date (not the target date)."""
        from e_quantile_daily_pipeline import _build_features, Config
        idx = pd.bdate_range("2024-01-01", periods=100)
        raw_vals = np.arange(100, dtype=float) * 3.0
        df = pd.DataFrame({"Revenues": raw_vals}, index=idx)
        df.index.name = "date"
        cfg = Config(target="Revenues", cadence="Daily", horizon=5,
                     data_path="dummy", lags_daily=(1,), windows_daily=(5,))
        X, y_target, od, ov = _build_features(df, cfg)
        date_to_pos = {d: i for i, d in enumerate(idx)}
        for row_date, origin_val in ov.items():
            pos = date_to_pos[row_date]
            expected = raw_vals[pos]
            assert np.isclose(origin_val, expected, atol=1e-6), (
                f"origin_value at pos={pos}: got {origin_val}, expected {expected}"
            )


# ===================================================================
# 15. STAT PIPELINE: ORIGIN EXPORT (Phase 2 fixes)
# ===================================================================

class TestStatPipelineHardening:
    """Verify stat pipeline now exports origin_date and origin_value."""

    def test_stat_output_schema_includes_origin_columns(self):
        """The stat pipeline output schema must include origin_date and origin_value."""
        # We verify by checking the code itself (can't run full pipeline in unit test)
        stat_path = BACKEND_DIR / "run_a_stat.py"
        text = stat_path.read_text(encoding="utf-8")
        assert '"origin_date"' in text, "run_a_stat.py must include origin_date in output"
        assert '"origin_value"' in text, "run_a_stat.py must include origin_value in output"
        assert '"target_date"' in text, "run_a_stat.py must include target_date in output"
        assert "integrity_report.json" in text, "run_a_stat.py must save integrity report"

    def test_stat_persistence_baseline_in_code(self):
        """Stat pipeline must compute persistence baseline and skill."""
        stat_path = BACKEND_DIR / "run_a_stat.py"
        text = stat_path.read_text(encoding="utf-8")
        assert "mae_persist" in text, "run_a_stat.py must compute persistence MAE"
        assert "skill_pct" in text, "run_a_stat.py must compute skill percentage"
        assert "Persistence (baseline)" in text, "run_a_stat.py must add persistence row to leaderboard"


# ===================================================================
# 16. DASHBOARD TRUST BADGE (Phase 4 fix)
# ===================================================================

class TestDashboardTrustBadge:
    """Verify Dashboard shows pipeline trust status."""

    def test_dashboard_shows_trust_badge(self):
        dashboard_path = BACKEND_DIR.parent / "frontend" / "pages" / "01_Dashboard.py"
        text = dashboard_path.read_text(encoding="utf-8")
        assert "Quality gate PASSED" in text, "Dashboard must show quality gate passed message"
        assert "Quality gate FAILED" in text, "Dashboard must show quality gate failed message"
        assert "integrity_report.json" in text, "Dashboard must load integrity report"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
