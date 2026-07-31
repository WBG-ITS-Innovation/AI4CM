"""M-3: ResidualRF's prediction intervals must be honest.

The old implementation derived interval widths from the *in-sample* residuals
of an unpruned 400-tree forest.  Such a forest nearly memorises its training
data, so those residuals are far too small and the intervals collapse — M-2
measured 26.5% coverage where P10–P90 should cover ~80%.

The fix is to build residuals from out-of-bag (OOB) predictions: for each
training row, only the trees that did *not* see that row vote.  Those
residuals reflect genuine generalisation error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from e_quantile_daily_pipeline import _fit_residual_rf_quantiles  # noqa: E402

QUANTILES = (0.10, 0.50, 0.90)


def _noisy_linear(n=300, noise=50.0, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })
    y = pd.Series(10.0 * X["a"] + 3.0 * X["b"] + rng.normal(0, noise, n))
    return X, y


def test_intervals_are_wider_than_in_sample_residuals():
    """OOB-based widths must exceed the memorised in-sample widths."""
    from sklearn.ensemble import RandomForestRegressor

    X, y = _noisy_linear()
    X_tr, y_tr, X_te = X.iloc[:250], y.iloc[:250], X.iloc[250:]

    preds = _fit_residual_rf_quantiles(X_tr, y_tr, X_te, QUANTILES)
    oob_width = float(np.mean(preds[0.90] - preds[0.10]))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    in_sample = y_tr.values - rf.predict(X_tr)
    in_sample_width = float(np.quantile(in_sample, 0.90) - np.quantile(in_sample, 0.10))

    assert oob_width > in_sample_width * 1.5, (
        f"OOB width {oob_width:,.1f} should clearly exceed in-sample "
        f"{in_sample_width:,.1f}; intervals still look memorised"
    )


def test_quantiles_are_monotonic():
    X, y = _noisy_linear(seed=1)
    preds = _fit_residual_rf_quantiles(X.iloc[:250], y.iloc[:250], X.iloc[250:], QUANTILES)
    assert np.all(preds[0.10] <= preds[0.50] + 1e-9)
    assert np.all(preds[0.50] <= preds[0.90] + 1e-9)


def test_coverage_is_calibrated_on_holdout():
    """P10–P90 should cover ~80% of unseen points (allow a wide band for n)."""
    X, y = _noisy_linear(n=600, seed=2)
    X_tr, y_tr = X.iloc[:450], y.iloc[:450]
    X_te, y_te = X.iloc[450:], y.iloc[450:]

    preds = _fit_residual_rf_quantiles(X_tr, y_tr, X_te, QUANTILES)
    covered = float(np.mean((y_te.values >= preds[0.10]) & (y_te.values <= preds[0.90])))
    assert 0.65 <= covered <= 0.95, f"coverage {covered:.1%} is not near the nominal 80%"


def test_works_when_oob_unavailable():
    """Tiny training sets can leave rows with no OOB votes — must not crash."""
    X, y = _noisy_linear(n=20, seed=3)
    preds = _fit_residual_rf_quantiles(X.iloc[:15], y.iloc[:15], X.iloc[15:], QUANTILES)
    for q in QUANTILES:
        assert len(preds[q]) == 5
        assert np.all(np.isfinite(preds[q]))
