"""Conformalised quantile regression, and a gate that scores coverage where it matters.

The defect this exists to fix, measured on DEV: nominal-80% bands cover **9.6% to 69.9%** of the
largest third of days depending on the target (`reports/ws2_tuning.md`). The overall figures look
far healthier — 57–83% — because the small and middle days carry the average. The largest days
are the ones a cash buffer exists for, so an average that hides them is the wrong number to gate
on.

--------------------------------------------------------------------------------
CQR — WHAT IT DOES AND WHAT IT ASSUMES
--------------------------------------------------------------------------------
Split-conformal quantile regression (Romano, Patterson & Candès 2019). Fit the quantile models on
a proper training subset, then on a held-out **calibration** slice measure how far outside its own
band each actual fell:

    E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

A positive score means the band missed by that much; a negative one means it had that much room to
spare. Take the ``ceil((n+1)(1-alpha))``-th smallest score and widen the band by it in both
directions.

What this buys: **finite-sample marginal coverage of at least 1-alpha**, with no assumption about
the model, the noise, or the shape of the distribution. It is a correction applied to whatever the
quantile model produced, not a better quantile model.

What it does **not** buy: conditional coverage. The guarantee is *on average over the calibration
distribution*. A single global widening will lift the overall figure while leaving the largest days
under-covered, because it adds the same absolute amount everywhere.

That is why ``cqr_calibrate`` supports **grouped** calibration: compute a separate correction per
magnitude tercile, so the widening is allowed to differ where the miss is concentrated. Grouping
costs statistical efficiency — each group calibrates on a third of the rows — and the honest
trade-off is stated in the report rather than hidden.

--------------------------------------------------------------------------------
THE CALIBRATION SLICE MUST BE CAUSAL
--------------------------------------------------------------------------------
Information set: the calibration rows are the **last** rows of the training window, and the
scores are computed from targets that were already known at the point the band is issued. No
calibration row's target falls at or after any evaluation origin. ``test_conformal.py`` asserts
this by construction and by mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Default nominal coverage. Matches every interval elsewhere in the project.
DEFAULT_ALPHA = 0.20          # -> 80% nominal

#: Fraction of the training window held back to calibrate on.
CALIBRATION_FRACTION = 0.25

#: Bucket labels, shared with the frontend's interval view so the two never drift.
TERCILES = ("smallest third", "middle third", "largest third")

#: The conditional-coverage floor. Deliberately below the 80% nominal: a band that must hit 80%
#: in *every* bucket on a 262-row window would be gated on noise. 60% is the level below which a
#: band stops being usable for planning at all -- it fails more often than two days in five.
CONDITIONAL_FLOOR = 0.60


# ── conformity scores and the correction ──────────────────────────────────────

def conformity_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """How far outside its own band each actual fell. Positive = missed by that much."""
    y = np.asarray(y, dtype=float)
    return np.maximum(np.asarray(lo, dtype=float) - y, y - np.asarray(hi, dtype=float))


def conformal_width(scores: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """The split-conformal correction: the ceil((n+1)(1-alpha))-th smallest score.

    The ``(n+1)`` is what makes the guarantee finite-sample rather than asymptotic. When the rank
    exceeds ``n`` -- too few calibration rows for the requested level -- there is no finite
    correction that can promise the coverage, and this returns ``inf`` rather than silently
    falling back to the largest observed score and implying a guarantee it cannot make.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(s)[k - 1])


@dataclass
class CQRCalibration:
    """A fitted correction: one global width, and optionally one per bucket."""

    alpha: float
    width: float
    n_calibration: int
    per_bucket: Dict[str, float] = field(default_factory=dict)
    bucket_edges: Optional[Tuple[float, float]] = None
    grouped: bool = False
    note: str = ""

    def apply(self, lo: np.ndarray, hi: np.ndarray,
              magnitude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Widen a band. Grouped calibration needs ``magnitude`` to pick the bucket."""
        lo = np.asarray(lo, dtype=float).copy()
        hi = np.asarray(hi, dtype=float).copy()
        if not self.grouped or magnitude is None or not self.per_bucket:
            return lo - self.width, hi + self.width
        w = np.full(len(lo), self.width, dtype=float)
        labels = assign_terciles_by_edges(np.asarray(magnitude, dtype=float),
                                          self.bucket_edges)
        for name, wi in self.per_bucket.items():
            w[labels == name] = wi
        return lo - w, hi + w


def tercile_edges(magnitude: np.ndarray) -> Tuple[float, float]:
    """The two cut points, taken from CALIBRATION data only.

    Returned and reused so evaluation rows are bucketed by the calibration window's edges rather
    than their own -- otherwise the buckets shift between fit and use, and the correction lands in
    the wrong one.
    """
    m = np.abs(np.asarray(magnitude, dtype=float))
    m = m[np.isfinite(m)]
    return (float(np.quantile(m, 1 / 3)), float(np.quantile(m, 2 / 3)))


def assign_terciles_by_edges(magnitude: np.ndarray,
                             edges: Optional[Tuple[float, float]]) -> np.ndarray:
    out = np.full(len(magnitude), TERCILES[1], dtype=object)
    if edges is None:
        return out
    m = np.abs(np.asarray(magnitude, dtype=float))
    out[m <= edges[0]] = TERCILES[0]
    out[m > edges[1]] = TERCILES[2]
    return out


def cqr_calibrate(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                  alpha: float = DEFAULT_ALPHA, grouped: bool = False,
                  min_per_bucket: int = 20,
                  bucket_by: Optional[np.ndarray] = None) -> CQRCalibration:
    """Fit the conformal correction on a calibration slice.

    ``grouped=True`` fits one correction per magnitude tercile. It falls back to the global width
    for any bucket with fewer than ``min_per_bucket`` rows and says so in ``note`` -- a correction
    estimated from a handful of scores is noise dressed as a guarantee.
    """
    scores = conformity_scores(y_cal, lo_cal, hi_cal)
    width = conformal_width(scores, alpha)
    cal = CQRCalibration(alpha=alpha, width=width, n_calibration=int(len(scores)),
                         grouped=False)
    if not grouped:
        cal.note = "global correction (marginal guarantee)"
        return cal

    # Bucket by an OBSERVABLE quantity, defaulting to the band midpoint.
    #
    # This was originally bucketed by the actual `y_cal`. At prediction time the actual is not
    # known, so the correction had to be assigned by predicted magnitude instead -- and fitting on
    # one quantity while applying by another puts corrections in the wrong buckets. Measured: on
    # Expenditure that made grouped CQR WORSE than no calibration at all (largest-third coverage
    # 9.6% -> 10.8% while overall fell 57.2% -> 43.2%). Both sides now use the same observable.
    basis = (np.asarray(bucket_by, dtype=float) if bucket_by is not None
             else (np.asarray(lo_cal, dtype=float) + np.asarray(hi_cal, dtype=float)) / 2.0)
    edges = tercile_edges(basis)
    labels = assign_terciles_by_edges(basis, edges)
    per, thin = {}, []
    for name in TERCILES:
        sel = labels == name
        if int(sel.sum()) < min_per_bucket:
            per[name] = width
            thin.append(f"{name} (n={int(sel.sum())})")
            continue
        per[name] = conformal_width(scores[sel], alpha)
    cal.grouped = True
    cal.per_bucket = per
    cal.bucket_edges = edges
    cal.note = ("per-tercile corrections"
                + (f"; fell back to the global width for {', '.join(thin)}" if thin else ""))
    return cal


def causal_calibration_split(n: int, horizon: int,
                             fraction: float = CALIBRATION_FRACTION) -> Tuple[np.ndarray, np.ndarray]:
    """(fit, calibration) indices, with an ``horizon``-row gap between them.

    Row *t* carries the target *y(t+h)*, so without the gap the last *h* fit rows have answers
    inside the calibration slice and the correction is measured partly against data the model has
    already seen -- which makes the band look better calibrated than it is.
    """
    if n <= horizon + 4:
        raise ValueError(f"cannot split {n} rows for calibration at horizon {horizon}")
    n_cal = max(horizon + 1, int(round(n * fraction)))
    n_cal = min(n_cal, n - horizon - 2)
    cal_start = n - n_cal
    fit_end = cal_start - horizon
    if fit_end <= 1:
        raise ValueError(f"gap of {horizon} leaves no fitting rows out of {n}")
    return np.arange(fit_end), np.arange(cal_start, n)


# ── the conditional-coverage gate (replaces the marginal one) ─────────────────

def coverage_by_bucket(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       bucket: np.ndarray) -> Dict[str, Dict[str, float]]:
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    inside = (y >= lo) & (y <= hi)
    out: Dict[str, Dict[str, float]] = {}
    for name in pd.unique(pd.Series(bucket)):
        sel = np.asarray(bucket) == name
        if not sel.any():
            continue
        out[str(name)] = {"coverage": float(inside[sel].mean()),
                          "n": int(sel.sum()),
                          "mean_width": float(np.mean(hi[sel] - lo[sel]))}
    return out


def volatility_terciles(vol: np.ndarray) -> np.ndarray:
    """Bucket rows by trailing volatility, using their own edges.

    Volatility is a second, independent axis: a band can be well calibrated across magnitudes and
    still fail on the days when the series is moving most. GBQuantile's documented inverted
    response -- widest at LOW volatility -- is exactly this failure, and a magnitude-only gate
    cannot see it.
    """
    v = np.abs(np.asarray(vol, dtype=float))
    finite = v[np.isfinite(v)]
    if len(finite) < 6:
        return np.full(len(v), TERCILES[1], dtype=object)
    e1, e2 = float(np.quantile(finite, 1 / 3)), float(np.quantile(finite, 2 / 3))
    out = np.full(len(v), TERCILES[1], dtype=object)
    out[v <= e1] = TERCILES[0]
    out[v > e2] = TERCILES[2]
    return out


def conditional_coverage_gate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                              magnitude: np.ndarray,
                              volatility: Optional[np.ndarray] = None,
                              floor: float = CONDITIONAL_FLOOR,
                              nominal: float = 1.0 - DEFAULT_ALPHA,
                              min_bucket_n: int = 20) -> Dict:
    """Gate on the WORST bucket, not the average.

    The marginal gate this replaces asked only whether overall coverage was near nominal. A band
    covering 83% overall and 9.6% of the largest days passed it -- and the largest days are the
    ones the band exists for.

    Every bucket is reported whether it passes or not, on both axes. Buckets thinner than
    ``min_bucket_n`` are reported and excluded from the verdict rather than allowed to fail it on
    a handful of rows; if that leaves nothing to judge, the verdict is ``None`` (never verified),
    never a pass.
    """
    mag_b = assign_terciles_by_edges(np.asarray(magnitude, dtype=float),
                                     tercile_edges(magnitude))
    buckets = {"magnitude": coverage_by_bucket(y, lo, hi, mag_b)}
    if volatility is not None:
        buckets["volatility"] = coverage_by_bucket(y, lo, hi,
                                                   volatility_terciles(volatility))

    judged, failures, thin = [], [], []
    for axis, per in buckets.items():
        for name, st in per.items():
            if st["n"] < min_bucket_n:
                thin.append(f"{axis}/{name} (n={st['n']})")
                continue
            judged.append((axis, name, st["coverage"]))
            if st["coverage"] < floor:
                failures.append((axis, name, st["coverage"], st["n"]))

    inside = ((np.asarray(y, float) >= np.asarray(lo, float)) &
              (np.asarray(y, float) <= np.asarray(hi, float)))
    overall = float(inside.mean()) if len(inside) else float("nan")

    if not judged:
        passed: Optional[bool] = None
        reason = ("No bucket had enough rows to judge, so conditional coverage is not verified. "
                  f"Thin buckets: {', '.join(thin)}." if thin else
                  "No rows available to judge conditional coverage.")
    elif failures:
        passed = False
        worst = min(failures, key=lambda f: f[2])
        reason = (f"{len(failures)} of {len(judged)} buckets fall below the {floor:.0%} floor. "
                  f"Worst: {worst[0]} / {worst[1]} at {worst[2]:.1%} on {worst[3]} days. "
                  f"Overall coverage is {overall:.1%} against a nominal {nominal:.0%}, which is "
                  f"why a marginal gate would have passed this band.")
    else:
        passed = True
        low = min(judged, key=lambda j: j[2])
        reason = (f"Every judged bucket clears the {floor:.0%} floor. Weakest is "
                  f"{low[0]} / {low[1]} at {low[2]:.1%}. Overall {overall:.1%} against a "
                  f"nominal {nominal:.0%}.")

    return {"passed": passed, "floor": float(floor), "nominal": float(nominal),
            "overall_coverage": overall, "buckets": buckets,
            "n_failing_buckets": len(failures), "n_judged_buckets": len(judged),
            "thin_buckets": thin, "reason_plain": reason}


# ── per-target selection rule ─────────────────────────────────────────────────

def select_per_target(candidates: Sequence[Dict], *,
                      mae_tie_pct: float = 1.0) -> Dict:
    """Formal per-target selection, with the tie-break that keeps L2 candidates in the pool.

    Each candidate needs ``name``, ``dev_mae``; optionally ``sentinel``, ``conditional_gate``
    (the dict above), ``objective`` ("L1"/"L2"), ``n_train_folds``.

    Rules, in order, and the reason for each:

    1. **Never select a candidate whose conditional-coverage gate FAILED** when another passes.
       An accurate point forecast with an unusable band is not the better product.
    2. Otherwise rank by DEV MAE.
    3. **Tie-break within ``mae_tie_pct``:** candidates whose DEV MAE is within that band of the
       leader are treated as tied, and the tie is broken by TRAIN-fold evidence, then by the
       sentinel. This is what keeps squared-error candidates in the pool: on Expenditure the
       DEV-best model is an **L2** model, 1.0% ahead of the promoted L1 recipe, and a strict
       argmin on one DEV fold would have silently swapped a recipe on noise. WS2 measured that a
       sub-1% margin on one fold does not predict anything.
    """
    if not candidates:
        return {"selected": None, "reason": "no candidates supplied", "ranked": []}

    def gate_state(c):
        g = c.get("conditional_gate") or {}
        return g.get("passed", None)

    scored = [c for c in candidates if c.get("dev_mae") is not None
              and np.isfinite(float(c["dev_mae"]))]
    if not scored:
        return {"selected": None, "reason": "no candidate has a DEV MAE", "ranked": []}

    passing = [c for c in scored if gate_state(c) is not False]
    pool = passing if passing else scored
    pool_note = ("" if passing else
                 " No candidate passed the conditional-coverage gate, so selection fell back to "
                 "the full pool and the winner ships with its band defect stated.")

    ranked = sorted(pool, key=lambda c: float(c["dev_mae"]))
    leader = ranked[0]
    thr = float(leader["dev_mae"]) * (1.0 + mae_tie_pct / 100.0)
    tied = [c for c in ranked if float(c["dev_mae"]) <= thr]

    if len(tied) == 1:
        chosen, why = leader, (
            f"lowest DEV MAE ({float(leader['dev_mae']):,.0f}), and no other candidate is within "
            f"{mae_tie_pct:.1f}%.")
    else:
        chosen = max(tied, key=lambda c: (int(c.get("n_train_folds") or 0),
                                          float(c.get("sentinel") or 0.0),
                                          -float(c["dev_mae"])))
        names = ", ".join(f"{c['name']} ({float(c['dev_mae']):,.0f})" for c in tied)
        why = (f"{len(tied)} candidates are within {mae_tie_pct:.1f}% on DEV [{names}], which one "
               f"fold cannot separate — WS2 measured that a sub-1% DEV margin predicts nothing. "
               f"Broken on TRAIN-fold evidence, then the signal reading: {chosen['name']}.")

    return {"selected": chosen["name"], "reason": why + pool_note,
            "gate_state": gate_state(chosen),
            "tied_within_pct": [c["name"] for c in tied],
            "ranked": [{"name": c["name"], "dev_mae": float(c["dev_mae"]),
                        "conditional_gate": gate_state(c),
                        "objective": c.get("objective")} for c in ranked]}
