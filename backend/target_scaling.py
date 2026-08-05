"""Workstream 4 — target scaling, without touching the prediction path.

The obvious way to compare target transforms is to transform the target inside the
pipeline, fit, and invert the predictions on the way out. That route runs straight through
the code that emits ``origin_value`` and ``y_true`` — the two columns the unified
persistence ruler is computed from — and a mistake there does not raise, it silently
changes the benchmark every model in the project is measured against.

So this module takes the other route. ``ScaledRegressor`` is an ordinary scikit-learn
regressor that transforms ``y`` inside ``fit`` and inverts inside ``predict``. The pipeline
hands it a target and gets back a prediction in the **original units**; it never sees a
transformed value, so there is nothing to invert downstream and no line of the prediction
path changes. The ruler is therefore identical by construction, not by inspection — and
``test_target_scaling.py`` still asserts it, because "by construction" is a claim.

--------------------------------------------------------------------------------
THE THREE CANDIDATES
--------------------------------------------------------------------------------
``raw``
    Identity. The incumbent, and the thing the others must beat.

``asinh``
    ``z = asinh(y / s)``, inverse ``y = s * sinh(z)``. Defined on the whole real line, so
    it survives the negative ``Revenues`` days, and it behaves like a logarithm for large
    magnitudes — it compresses the month-end and tax-deadline spikes without discarding
    their sign. ``log1p`` is unavailable while Treasury question **T1** is open, because it
    is undefined below −1 and 72 business days of ``Revenues`` print negative.
    ``s`` is a robust scale **fitted on the training fold only**.

``ratio``
    ``z = y / L(t)``, inverse ``y = z * L(t)``, where ``L(t)`` is a causal trailing level:
    a rolling median of |y| over the preceding window, shifted so row *t* uses history
    strictly before *t*. This turns the target into "how big is today relative to recent
    days", which is scale-free and stationary in a way the raw series is not. It is the
    only one of the three whose divisor varies per row, which is also what makes it the
    one that can fail silently — hence ``strict`` below.

--------------------------------------------------------------------------------
FAILING LOUDLY
--------------------------------------------------------------------------------
A mis-inverted prediction is the failure mode that matters: it produces plausible numbers
in the wrong units, and every metric downstream still computes. Three guards:

* ``ScaledRegressor.predict`` refuses to run if the per-row divisor cannot be aligned to
  the rows it was asked to predict. It raises rather than falling back to a global
  constant, because a silent fallback is exactly how a units error survives.
* ``assert_round_trip`` is called at fit time on the training targets, so a transform whose
  inverse does not recover its input fails at the point of use rather than at the point of
  reporting.
* ``sanity_check_prediction_scale`` compares a prediction batch against the magnitude of
  the training targets and raises when they differ by more than a configurable factor —
  the signature of an un-inverted or double-inverted prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

#: Window for the trailing level in the ``ratio`` transform, in business days (~3 months).
#: Long enough to be stable across a month-end, short enough to track a regime shift.
TRAILING_WINDOW = 63

#: Floor on the trailing level, as a fraction of the training median. Without it a quiet
#: stretch drives the divisor toward zero and the transformed target explodes.
LEVEL_FLOOR_FRACTION = 0.10

#: A prediction more than this many times larger (or smaller) than the training targets'
#: typical magnitude is treated as a units error, not a bold forecast.
SCALE_SANITY_FACTOR = 50.0

#: Minimum batch size for the magnitude check to mean anything.
#:
#: Measured: b_ml predicts ONE ORIGIN AT A TIME -- 1,304 of 1,314 predict() calls in a
#: five-fold Expenditure run pass a single row. The median magnitude of a one-row batch is
#: just that row, so the check fired on a legitimate holiday-zeroed day (predicted 307,414
#: against a training magnitude of 2.79e7) and aborted an otherwise healthy run.
#:
#: A units error is *systematic*: it affects every prediction, so it shows up on the
#: in-sample batch at fit time, which is thousands of rows. A single row cannot distinguish
#: a units error from an unusual day, so it is not checked.
MIN_SANITY_BATCH = 30

TRANSFORMS = ("raw", "asinh", "ratio")


def trailing_level(y: pd.Series, window: int = TRAILING_WINDOW,
                   floor_fraction: float = LEVEL_FLOOR_FRACTION) -> pd.Series:
    """Causal trailing level: rolling median of |y| over the *preceding* window.

    ``shift(1)`` is what makes it causal — the level at row *t* is computed from rows
    strictly before *t*, so it is knowable at the forecast origin. The floor is applied
    from the expanding median rather than a global one, so it too uses only history.
    """
    a = pd.Series(y).astype(float).abs()
    lvl = a.rolling(window, min_periods=max(5, window // 6)).median().shift(1)
    expanding_floor = a.expanding(min_periods=5).median().shift(1) * floor_fraction
    lvl = lvl.where(lvl > expanding_floor, expanding_floor)
    # Rows with no history yet fall back to the first usable level rather than NaN, so the
    # transform never silently drops training rows.
    return lvl.bfill().replace(0.0, np.nan).bfill().fillna(1.0)


def assert_round_trip(name: str, forward, inverse, y: np.ndarray,
                      rtol: float = 1e-9, atol: float = 1e-6) -> None:
    """A transform whose inverse does not recover its input must fail at fit time."""
    z = forward(y)
    back = inverse(z)
    if not np.allclose(back, y, rtol=rtol, atol=atol, equal_nan=True):
        worst = float(np.nanmax(np.abs(np.asarray(back) - np.asarray(y))))
        raise ValueError(
            f"target transform {name!r} does not round-trip: max absolute error {worst:.6g}. "
            f"Refusing to fit, because an inverse that loses information produces plausible "
            f"predictions in the wrong units."
        )


def sanity_check_prediction_scale(name: str, y_train: np.ndarray, pred: np.ndarray,
                                  factor: float = SCALE_SANITY_FACTOR,
                                  min_batch: int = 1) -> None:
    """Raise when predictions are implausibly far from the training magnitude.

    Catches the un-inverted and double-inverted cases, which are otherwise invisible: both
    yield finite numbers that every downstream metric happily consumes.

    ``min_batch`` guards against the opposite error. Batches smaller than this are skipped,
    because a median over a handful of rows says nothing about units -- see
    ``MIN_SANITY_BATCH``. Callers that hold a representative batch pass ``min_batch=1``.
    """
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    if len(pred) < min_batch:
        return
    ref = float(np.nanmedian(np.abs(np.asarray(y_train, dtype=float))))
    if not np.isfinite(ref) or ref == 0:
        return
    got = float(np.nanmedian(np.abs(np.asarray(pred, dtype=float))))
    if not np.isfinite(got):
        raise ValueError(f"transform {name!r}: predictions are not finite")
    if got > ref * factor or got < ref / factor:
        raise ValueError(
            f"transform {name!r}: predicted magnitude {got:.6g} is implausible against a "
            f"training magnitude of {ref:.6g} (factor {factor:g}). This is the signature of "
            f"an un-inverted or doubly-inverted prediction, so the fit is rejected rather "
            f"than reported."
        )


class ScaledRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor so it trains on a transformed target and predicts in original units.

    The pipeline sees an ordinary estimator. Nothing downstream of ``predict`` knows a
    transform happened, which is precisely why the ruler cannot move.

    Parameters
    ----------
    base : estimator
        Cloned before fitting, so one configured instance can be reused.
    transform : {"raw", "asinh", "ratio"}
    level : pandas.Series, optional
        Required for ``ratio``: the causal trailing level, indexed like the feature frame.
        Supplied from outside because it is a property of the target series, not of a fold.
    strict : bool
        When True (default) a divisor that cannot be aligned raises. Setting it False is
        only for tests that deliberately exercise the failure.
    """

    def __init__(self, base=None, transform: str = "raw",
                 level: Optional[pd.Series] = None, strict: bool = True):
        self.base = base
        self.transform = transform
        self.level = level
        self.strict = strict

    # ── divisor handling ──────────────────────────────────────────────────────
    def _rows_level(self, X) -> np.ndarray:
        if self.level is None:
            raise ValueError(
                "transform 'ratio' needs a `level` series; none was supplied. Refusing to "
                "substitute a global constant, which would change the units silently."
            )
        idx = getattr(X, "index", None)
        if idx is None:
            raise ValueError(
                "transform 'ratio' needs a row index to align its per-row divisor, but the "
                "feature matrix has none (it was probably converted to a bare array). "
                "Refusing to guess the alignment."
            )
        lv = self.level.reindex(idx)
        if lv.isna().any():
            missing = int(lv.isna().sum())
            msg = (f"transform 'ratio': the trailing level is missing for {missing} of "
                   f"{len(lv)} rows, so those predictions could not be returned to original "
                   f"units.")
            if self.strict:
                raise ValueError(msg + " Refusing to fit or predict.")
            lv = lv.ffill().bfill()
        return lv.to_numpy(dtype=float)

    # ── forward / inverse ─────────────────────────────────────────────────────
    def _forward(self, y: np.ndarray, lvl: Optional[np.ndarray]) -> np.ndarray:
        if self.transform == "raw":
            return y
        if self.transform == "asinh":
            return np.arcsinh(y / self._scale_)
        if self.transform == "ratio":
            return y / lvl
        raise ValueError(f"unknown transform {self.transform!r}; expected one of {TRANSFORMS}")

    def _inverse(self, z: np.ndarray, lvl: Optional[np.ndarray]) -> np.ndarray:
        if self.transform == "raw":
            return z
        if self.transform == "asinh":
            return np.sinh(z) * self._scale_
        if self.transform == "ratio":
            return z * lvl
        raise ValueError(f"unknown transform {self.transform!r}; expected one of {TRANSFORMS}")

    # ── sklearn API ───────────────────────────────────────────────────────────
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_train_ref_ = y

        # The asinh scale is fitted HERE, i.e. once per fold, on training targets only.
        if self.transform == "asinh":
            s = float(np.nanmedian(np.abs(y)))
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            self._scale_ = s
        else:
            self._scale_ = 1.0

        lvl = self._rows_level(X) if self.transform == "ratio" else None

        assert_round_trip(self.transform,
                          lambda v: self._forward(v, lvl),
                          lambda v: self._inverse(v, lvl),
                          y)

        z = self._forward(y, lvl)
        if not np.all(np.isfinite(z)):
            raise ValueError(
                f"transform {self.transform!r} produced non-finite training targets; "
                f"refusing to fit")
        self.estimator_ = clone(self.base)
        self.estimator_.fit(X, z)

        # Verify the round trip through the FITTED MODEL, not just through the transform.
        # This is the batch on which a units error is detectable: thousands of in-sample
        # rows, where an un-inverted or doubly-inverted inverse is systematic. Doing it here
        # means a broken transform fails at fit time rather than surfacing as a metric.
        in_sample = self._inverse(
            np.asarray(self.estimator_.predict(X), dtype=float), lvl)
        sanity_check_prediction_scale(self.transform, y, in_sample, min_batch=1)
        return self

    def predict(self, X):
        lvl = self._rows_level(X) if self.transform == "ratio" else None
        z = np.asarray(self.estimator_.predict(X), dtype=float)
        out = self._inverse(z, lvl)
        # Small batches are not informative about units (see MIN_SANITY_BATCH); the
        # load-bearing check already ran at fit time on the in-sample batch.
        sanity_check_prediction_scale(self.transform, self.y_train_ref_, out,
                                      min_batch=MIN_SANITY_BATCH)
        return out

    def __sklearn_tags__(self):  # pragma: no cover - sklearn plumbing
        return self.base.__sklearn_tags__()


def make_scaled_models(base_models: dict, transform: str,
                       level: Optional[pd.Series] = None) -> dict:
    """Wrap every model in a registry-style dict with the given transform.

    Used by the workstream-4 harness in place of ``available_models()``, so the comparison
    runs through the unmodified pipeline.
    """
    return {name: ScaledRegressor(base=est, transform=transform, level=level)
            for name, est in base_models.items()}
