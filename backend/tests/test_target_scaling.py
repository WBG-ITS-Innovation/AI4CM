"""Workstream 4: target transforms must round-trip, fail loudly, and not move the ruler.

The ruler check is the important one. The unified persistence benchmark is computed from
`origin_value` and `y_true`; if a transform ever leaked into those columns, every skill
figure in the project would shift and nothing would raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forecast_integrity import compute_persistence_baseline  # noqa: E402
from target_scaling import (  # noqa: E402
    SCALE_SANITY_FACTOR,
    TRANSFORMS,
    ScaledRegressor,
    assert_round_trip,
    sanity_check_prediction_scale,
    trailing_level,
)

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
H = 5

#: The unified ruler on the 2025 window definition, as published in phase-6 §4.
#: NOTE: this is a MODEL-FREE data statistic -- persistence against truth, no model fitted,
#: nothing selected. Recomputing it does not evaluate anything on the holdout.
PUBLISHED_RULER = {
    "Revenues": 83_534_152.85,
    "Expenditure": 83_839_124.43,
    "State budget balance": 189_930_653.98,
}


def _synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    y = 1e8 + 3e7 * rng.normal(0, 1, n)
    y[::21] *= 6.0            # month-end spikes
    y[::97] *= -1.0           # signed flows, as in Revenues
    X = pd.DataFrame({"f0": np.arange(n), "f1": rng.normal(0, 1, n)}, index=idx)
    return X, pd.Series(y, index=idx)


# ── round trip, per transform ─────────────────────────────────────────────────

@pytest.mark.parametrize("transform", TRANSFORMS)
def test_round_trip_recovers_the_original(transform):
    X, y = _synth()
    lvl = trailing_level(y) if transform == "ratio" else None
    m = ScaledRegressor(base=_Identity(), transform=transform, level=lvl)
    m.fit(X, y)
    fwd = m._forward(y.to_numpy(), m._rows_level(X) if transform == "ratio" else None)
    back = m._inverse(fwd, m._rows_level(X) if transform == "ratio" else None)
    assert np.allclose(back, y.to_numpy(), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_round_trip_holds_on_negative_values(transform):
    """Revenues prints negative on 72 business days; a transform that cannot represent
    them is unusable regardless of how it scores."""
    X, y = _synth()
    y = y.copy()
    y.iloc[:50] = -np.abs(y.iloc[:50])
    lvl = trailing_level(y) if transform == "ratio" else None
    m = ScaledRegressor(base=_Identity(), transform=transform, level=lvl).fit(X, y)
    p = m.predict(X)
    assert np.isfinite(p).all()


def test_assert_round_trip_rejects_a_broken_inverse():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="does not round-trip"):
        assert_round_trip("broken", lambda v: v * 2, lambda v: v * 3, y)


# ── failing loudly ────────────────────────────────────────────────────────────

def test_un_inverted_prediction_raises_rather_than_reporting():
    """The failure that matters: plausible numbers in the wrong units.

    Every downstream metric would compute happily on these, which is why the check lives at
    the point of prediction rather than in a review.
    """
    _, y = _synth()
    # asinh space is O(1) against a 1e8 target -- the classic un-inverted prediction
    with pytest.raises(ValueError, match="implausible"):
        sanity_check_prediction_scale("asinh", y.to_numpy(), np.full(50, 1.9))


def test_doubly_inverted_prediction_raises():
    _, y = _synth()
    with pytest.raises(ValueError, match="implausible"):
        sanity_check_prediction_scale("asinh", y.to_numpy(),
                                      np.full(50, float(np.median(np.abs(y))) * 1e4))


def test_reasonable_predictions_pass_the_scale_check():
    _, y = _synth()
    ref = float(np.median(np.abs(y)))
    sanity_check_prediction_scale("raw", y.to_numpy(), np.full(50, ref * 1.3))


def test_ratio_without_a_level_refuses_to_fit():
    X, y = _synth()
    with pytest.raises(ValueError, match="needs a `level` series"):
        ScaledRegressor(base=_Identity(), transform="ratio").fit(X, y)


def test_ratio_refuses_a_feature_matrix_with_no_index():
    """A bare numpy array cannot be aligned to a per-row divisor.

    Guessing the alignment would silently mismatch rows to divisors -- the worst version of
    this bug, because the output looks fine.
    """
    X, y = _synth()
    m = ScaledRegressor(base=_Identity(), transform="ratio", level=trailing_level(y))
    with pytest.raises(ValueError, match="needs a row index"):
        m.fit(X.to_numpy(), y)


def test_ratio_refuses_rows_it_has_no_divisor_for():
    X, y = _synth()
    lvl = trailing_level(y).iloc[:-40]          # divisor missing for the last 40 rows
    m = ScaledRegressor(base=_Identity(), transform="ratio", level=lvl)
    with pytest.raises(ValueError, match="trailing level is missing"):
        m.fit(X, y)


def test_unknown_transform_raises():
    X, y = _synth()
    with pytest.raises(ValueError, match="unknown transform"):
        ScaledRegressor(base=_Identity(), transform="loglog").fit(X, y)


# ── the trailing level is causal ──────────────────────────────────────────────

def test_trailing_level_is_causal():
    """Mutating the future must not change any past divisor."""
    _, y = _synth()
    cut = len(y) // 2
    base = trailing_level(y)
    tampered = y.copy()
    tampered.iloc[cut:] = tampered.iloc[cut:] * 1000
    after = trailing_level(tampered)
    pd.testing.assert_series_equal(base.iloc[:cut], after.iloc[:cut])
    assert not base.iloc[cut:].equals(after.iloc[cut:])


def test_trailing_level_is_positive_and_finite():
    _, y = _synth()
    lvl = trailing_level(y)
    assert (lvl > 0).all()
    assert np.isfinite(lvl).all()


# ── the ruler must not move ───────────────────────────────────────────────────

@pytest.mark.parametrize("target,expected", sorted(PUBLISHED_RULER.items()))
def test_unified_ruler_is_unchanged_by_workstream_4(target, expected):
    """The guard the whole workstream hangs on.

    Target scaling is implemented as an estimator wrapper, so the pipeline never emits a
    transformed value and `origin_value`/`y_true` stay in original units. That makes the
    ruler identical *by construction* -- and this asserts it anyway, because "by
    construction" is a claim, not evidence.

    This recomputes a MODEL-FREE statistic (persistence against truth). No model is fitted
    and nothing is selected, so it is not an evaluation of the sealed window.
    """
    if not DATA.exists():
        pytest.skip("canonical data not present")
    from b_ml_pipeline import to_business_index

    s = to_business_index(pd.read_csv(DATA), "date", target)
    idx = s.index[s.index >= "2025-01-01"]
    df = pd.DataFrame({
        "target_date": idx,
        "y_true": s.reindex(idx).to_numpy(),
        "origin_value": s.shift(H).reindex(idx).to_numpy(),
    }).dropna()
    got = compute_persistence_baseline(df)["mae_persistence"]
    assert abs(got - expected) < 0.01, (
        f"{target}: ruler moved to {got:,.6f} from the published {expected:,.2f}"
    )


def test_scaling_does_not_touch_the_prediction_path():
    """b_ml's prediction path must contain no target-transform code.

    Recorded as a test because the safety argument is structural: if a future change threads
    a transform through the pipeline instead of wrapping the estimator, the ruler guarantee
    weakens from 'impossible' to 'checked', and this is where that shows up.
    """
    src = (BACKEND / "b_ml_pipeline.py").read_text()
    for token in ("arcsinh", "np.sinh", "target_transform", "inverse_transform",
                  "ScaledRegressor"):
        assert token not in src, (
            f"b_ml_pipeline.py now references {token!r}; target scaling was supposed to "
            f"stay outside the prediction path"
        )


class _Identity:
    """Minimal estimator: predicts the target it was trained on, so the wrapper's own
    forward/inverse behaviour is what is under test rather than a model's fit quality."""

    def fit(self, X, y):
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        return self._y[: len(X)]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


# ── the guard must not fire on batches it cannot judge ────────────────────────

def test_single_row_predictions_are_not_scale_checked():
    """Regression: the magnitude check aborted a healthy run.

    Measured: b_ml predicts one origin at a time -- 1,304 of 1,314 predict() calls in a
    five-fold Expenditure run pass a single row. The median magnitude of a one-row batch is
    just that row, so the check fired on a legitimate holiday-zeroed day (predicted 307,414
    against a training magnitude of 2.79e7) and raised.

    A units error is systematic and therefore visible on the in-sample batch at fit time.
    One row cannot distinguish it from an unusual day, so one row is not judged.
    """
    from target_scaling import MIN_SANITY_BATCH

    _, y = _synth()
    tiny = np.array([307_414.0])          # the value that actually triggered it
    sanity_check_prediction_scale("asinh", y.to_numpy(), tiny,
                                  min_batch=MIN_SANITY_BATCH)   # must not raise
    # ...but a representative batch of the same wrong magnitude still must raise
    with pytest.raises(ValueError, match="implausible"):
        sanity_check_prediction_scale("asinh", y.to_numpy(),
                                      np.full(MIN_SANITY_BATCH, 307_414.0),
                                      min_batch=MIN_SANITY_BATCH)


def test_fit_time_check_catches_a_broken_inverse_through_the_model():
    """A transform whose inverse is wrong must fail at fit, on the in-sample batch.

    This is the check that replaced the per-row one: it sees thousands of rows, so a
    systematic units error cannot hide in it.
    """
    X, y = _synth()

    class _HalfInverse(ScaledRegressor):
        def _inverse(self, z, lvl):          # forgets to undo the asinh scaling
            return np.sinh(z)

    with pytest.raises(ValueError, match="does not round-trip|implausible"):
        _HalfInverse(base=_Identity(), transform="asinh").fit(X, y)


def test_predict_still_returns_original_units_after_the_fix():
    X, y = _synth()
    for transform in TRANSFORMS:
        lvl = trailing_level(y) if transform == "ratio" else None
        m = ScaledRegressor(base=_Identity(), transform=transform, level=lvl).fit(X, y)
        p = m.predict(X)
        ratio = float(np.nanmedian(np.abs(p))) / float(np.nanmedian(np.abs(y)))
        assert 0.5 < ratio < 2.0, f"{transform}: predictions are not in original units"


def test_ratio_divides_the_h_step_target_by_the_ORIGIN_level_not_the_target_level():
    """The alignment that would be leakage if it were wrong.

    The target is y(t+h); the divisor must be L(t), the trailing level at the ORIGIN, which
    is knowable when the forecast is made. Dividing by L(t+h) would use a statistic computed
    from data after the origin -- and it would improve every metric, because the divisor
    would then carry information about the answer.

    Checked structurally: the wrapper aligns its divisor to the FEATURE FRAME's index, which
    is the origin date, so a target shifted by h is divided by the origin-dated level.
    """
    n = 300
    idx = pd.bdate_range("2021-01-01", periods=n)
    y_series = pd.Series(np.arange(1.0, n + 1.0) * 1e6, index=idx)
    lvl = pd.Series(np.arange(1.0, n + 1.0), index=idx)     # level == row number
    H = 5
    # X is indexed by ORIGIN; y is the h-step-ahead target aligned to those origins.
    X = pd.DataFrame({"f": np.zeros(n - H)}, index=idx[: n - H])
    y_target = y_series.shift(-H).dropna()
    y_target.index = idx[: n - H]

    m = ScaledRegressor(base=_Identity(), transform="ratio", level=lvl)
    used = m._rows_level(X)
    # Must equal the level at the ORIGIN rows (1..n-H), never the level h rows later.
    assert np.allclose(used, np.arange(1.0, n - H + 1.0)), "divisor is not origin-aligned"
    assert not np.allclose(used, np.arange(1.0 + H, n + 1.0)), (
        "divisor is aligned to the TARGET date -- that is lookahead"
    )

    z = m._forward(y_target.to_numpy(), used)
    back = m._inverse(z, used)
    assert np.allclose(back, y_target.to_numpy())


def test_ratio_level_for_a_stock_target_uses_the_delta_scale():
    """A level series' trailing median is ~1e9 while its 5-day change is ~1e8.

    Dividing a delta by a level-scale divisor would shrink the target by an order of
    magnitude and make the transform look far worse than it is. The harness derives the
    level from the delta for stock targets; this pins the reason.
    """
    idx = pd.bdate_range("2021-01-01", periods=400)
    level = pd.Series(1.7e9 + np.arange(400) * 1e5, index=idx)
    lvl_of_level = trailing_level(level)
    lvl_of_delta = trailing_level(level.diff(5))
    assert lvl_of_level.median() > 50 * lvl_of_delta.median(), (
        "fixture no longer demonstrates the scale mismatch"
    )


def test_revenues_records_that_its_scaling_gain_is_drift_dependent():
    """Anyone quoting Revenues' 55.92% DEV skill must meet the caveat with it.

    The WS4 robustness study measured a +0.987 correlation between the evaluation window's
    level drift and the ratio transform's advantage. The DEV figure therefore describes a
    high-drift period, not a general property, and the registry has to say so -- otherwise
    the number travels without its condition.
    """
    from registry import recipe_for

    r = recipe_for("Revenues")
    assert r["params"].get("target_transform") == "ratio"
    cav = r.get("scaling_caveat")
    assert cav, "Revenues must carry the drift-dependence caveat"
    assert "DRIFT-DEPENDENT" in cav["finding"].upper()
    assert "0.987" in cav["evidence"]
    assert "conditional" in cav["how_to_quote"]
    # and it must state why adoption is still safe, not merely that there is a caveat
    assert "POSITIVE in every window" in cav["why_still_adopted"]


def test_targets_that_kept_raw_carry_no_scaling_caveat():
    """A caveat on a recipe that does not use the transform would be noise."""
    from registry import recipe_for

    for target in ("Expenditure", "State budget balance"):
        r = recipe_for(target)
        assert r["params"].get("target_transform", "raw") == "raw"
        assert "scaling_caveat" not in r
