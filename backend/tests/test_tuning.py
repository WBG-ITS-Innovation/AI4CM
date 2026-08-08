"""WS2: crossing safety and h-gapped early stopping.

The two properties that make a tuned quantile model trustworthy rather than merely lower-loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from tuning import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    QUANTILES,
    FoldData,
    count_crossings,
    crossing_safe,
    fit_point,
    fit_quantiles,
    gapped_split,
    objective_mae,
    suggest_params,
)


# ── crossing safety ───────────────────────────────────────────────────────────

def test_crossing_safe_repairs_inverted_quantiles():
    """A p90 below a p50 is an invalid interval, not a pessimistic one."""
    bad = {0.10: np.array([5.0, 1.0]), 0.50: np.array([3.0, 2.0]),
           0.90: np.array([1.0, 3.0])}
    out = crossing_safe(bad)
    assert np.allclose(out[0.10], [1.0, 1.0])
    assert np.allclose(out[0.50], [3.0, 2.0])
    assert np.allclose(out[0.90], [5.0, 3.0])
    for a, b in ((0.10, 0.50), (0.50, 0.90)):
        assert (out[a] <= out[b]).all()


def test_crossing_safe_leaves_valid_quantiles_untouched():
    good = {0.10: np.array([1.0]), 0.50: np.array([2.0]), 0.90: np.array([3.0])}
    out = crossing_safe(good)
    for q in good:
        assert np.allclose(out[q], good[q])


def test_count_crossings_reports_rather_than_hides():
    """A model that crosses constantly is misconfigured; sorting alone would mask that."""
    q = {0.10: np.array([5.0, 1.0, 0.0]), 0.50: np.array([3.0, 2.0, 1.0]),
         0.90: np.array([1.0, 3.0, 2.0])}
    assert count_crossings(q) == 1
    assert count_crossings(crossing_safe(q)) == 0
    assert count_crossings({0.5: np.array([1.0])}) == 0


# ── h-gapped early stopping ───────────────────────────────────────────────────

def test_gapped_split_leaves_exactly_h_rows_out():
    """Row t holds y(t+h), so the last h fit rows would otherwise have answers inside
    the validation block -- early stopping would be tuned against partly-seen data."""
    h = 5
    fit_ix, val_ix = gapped_split(1000, h)
    assert fit_ix[-1] + 1 + h == val_ix[0], (
        f"gap is {val_ix[0] - fit_ix[-1] - 1}, expected {h}"
    )
    assert len(set(fit_ix) & set(val_ix)) == 0
    assert val_ix[-1] == 999


@pytest.mark.parametrize("h", [1, 5, 10, 21])
def test_gap_scales_with_horizon(h):
    fit_ix, val_ix = gapped_split(800, h)
    assert val_ix[0] - fit_ix[-1] - 1 == h


def test_gapped_split_refuses_impossible_geometry():
    with pytest.raises(ValueError, match="cannot build a gapped split"):
        gapped_split(6, 5)


def test_validation_block_is_the_most_recent_rows():
    """Early stopping must be judged on the latest data, not a random slice -- the series
    is non-stationary and a shuffled split would be optimistic."""
    fit_ix, val_ix = gapped_split(500, 5)
    assert val_ix[-1] == 499
    assert (np.diff(val_ix) == 1).all()


# ── the models actually run and stay ordered ──────────────────────────────────

def _folds(n=500, seed=0, h=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    X = pd.DataFrame({"f0": rng.normal(0, 1, n), "f1": np.arange(n) % 21,
                      "f2": rng.normal(0, 1, n)}, index=idx)
    y = 1e8 + 2e7 * X["f0"].to_numpy() + 5e6 * (X["f1"].to_numpy() > 15)
    cut = int(n * 0.8)
    return [FoldData(X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:])]


@pytest.mark.parametrize("model", ["LGBMQuantile", "CatBoostQuantile"])
def test_fitted_quantiles_are_never_crossed(model):
    f = _folds()[0]
    params = {"learning_rate": 0.1}
    if model == "CatBoostQuantile":
        params["depth"] = 4
    preds, _ = fit_quantiles(model, f.X_tr, f.y_tr, f.X_te, params, horizon=5)
    assert set(preds) == set(QUANTILES)
    assert (preds[0.10] <= preds[0.50]).all()
    assert (preds[0.50] <= preds[0.90]).all()


def test_fit_point_returns_one_prediction_per_row():
    f = _folds()[0]
    p = fit_point("LGBMQuantile", f.X_tr, f.y_tr, f.X_te, {"learning_rate": 0.1}, horizon=5)
    assert p.shape == (len(f.X_te),)
    assert np.isfinite(p).all()


def test_objective_is_mae_of_the_median():
    """We publish MAE, so we optimise MAE. Optimising pinball and reporting MAE would be
    optimising a different quantity than the one shown."""
    folds = _folds()
    v = objective_mae(folds, "LGBMQuantile", {"learning_rate": 0.1}, horizon=5)
    pred = fit_point("LGBMQuantile", folds[0].X_tr, folds[0].y_tr, folds[0].X_te,
                     {"learning_rate": 0.1}, horizon=5)
    assert v == pytest.approx(float(np.mean(np.abs(folds[0].y_te - pred))))


def test_unknown_model_is_rejected():
    f = _folds()[0]
    with pytest.raises(ValueError, match="unknown quantile model"):
        fit_quantiles("MagicQuantile", f.X_tr, f.y_tr, f.X_te, {}, horizon=5)


def test_search_space_covers_both_families():
    class _T:
        def suggest_float(self, n, lo, hi, log=False): return lo
        def suggest_int(self, n, lo, hi, log=False): return lo
        def suggest_categorical(self, n, c): return c[0]

    for m in ("LGBMQuantile", "CatBoostQuantile", "CatBoost_L1"):
        p = suggest_params(_T(), m)
        assert "learning_rate" in p
    with pytest.raises(ValueError, match="no search space"):
        suggest_params(_T(), "Nope")


def test_objective_inverts_before_measuring_error():
    """MAE must be measured in original units, never in transformed space.

    On Revenues the ratio transform divides by a ~5e7 level, so an error of 0.1 in ratio
    space and 0.1 in lari differ by seven orders of magnitude. Optimising the wrong one would
    pick a different model and report a number that means nothing.
    """
    f = _folds()[0]
    scale = 1e8
    scaled = FoldData(X_tr=f.X_tr, y_tr=f.y_tr / scale, X_te=f.X_te, y_te=f.y_te,
                      inverse=lambda p: p * scale)
    v_scaled = objective_mae([scaled], "LGBMQuantile", {"learning_rate": 0.1}, horizon=5)
    v_raw = objective_mae([f], "LGBMQuantile", {"learning_rate": 0.1}, horizon=5)
    # Both are in lari, so they must be the same order of magnitude.
    assert 0.2 < v_scaled / v_raw < 5.0, (
        f"inverse not applied: scaled objective {v_scaled:.3g} vs raw {v_raw:.3g}"
    )
    # And without the inverse the objective would be absurd (truth ~1e8 vs preds ~1)
    unscaled = FoldData(X_tr=f.X_tr, y_tr=f.y_tr / scale, X_te=f.X_te, y_te=f.y_te)
    assert objective_mae([unscaled], "LGBMQuantile", {"learning_rate": 0.1},
                         horizon=5) > v_raw * 10


# ── the E_QUANTILE port ───────────────────────────────────────────────────────

def test_e_quantile_registry_offers_lgbm_quantile():
    """WS2 port: the quantile family must be able to run LightGBM, not just sklearn GB."""
    src = (BACKEND / "e_quantile_daily_pipeline.py").read_text()
    assert '"LGBMQuantile"' in src
    assert "crossing-safe" in src
    assert "from tuning import fit_quantiles" in src


def test_e_quantile_config_carries_tuned_params():
    import e_quantile_daily_pipeline as eq

    c = eq.Config(target="y", cadence="Daily", horizon=5, data_path="x.csv")
    assert c.lgbm_params is None, "default must be library defaults, not a tuned set"
    c2 = eq.Config(target="y", cadence="Daily", horizon=5, data_path="x.csv",
                   lgbm_params={"learning_rate": 0.05})
    assert c2.lgbm_params["learning_rate"] == 0.05


def test_lgbm_quantile_runs_end_to_end_in_the_family(tmp_path):
    """A registered model that cannot actually run is worse than an absent one."""
    import e_quantile_daily_pipeline as eq

    n = 260
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(3)
    y = 1000.0 + 200.0 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 25, n)
    data = tmp_path / "d.csv"
    pd.DataFrame({"date": idx, "y": y}).to_csv(data, index=False)

    cfg = eq.Config(target="y", cadence="Daily", horizon=3, data_path=str(data),
                    date_col="date", folds=None, min_train_years=0,
                    eval_start=str(idx[-12].date()), model_filter="LGBMQuantile",
                    out_root=str(tmp_path / "out"))
    eq.run_pipeline(cfg)
    preds = pd.read_csv(tmp_path / "out" / "predictions_long.csv")
    assert len(preds) > 0
    ok = preds.dropna(subset=["yhat_p10", "yhat_p50", "yhat_p90"])
    assert (ok["yhat_p10"] <= ok["yhat_p50"] + 1e-9).all()
    assert (ok["yhat_p50"] <= ok["yhat_p90"] + 1e-9).all()
