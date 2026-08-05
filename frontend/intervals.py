"""Interval detection and calibration, read from the artifacts rather than assumed.

This module exists because of a real defect in the dashboard it replaces logic for. That
version looked only for ``y_lo``/``y_hi`` and hard-coded a 90% target, which produced two
silent failures at once:

* **E_QUANTILE showed nothing.** It writes ``yhat_p10``/``yhat_p50``/``yhat_p90``, so the
  column check failed and the interval tab rendered an empty panel for the one family whose
  entire purpose is intervals.
* **A correctly calibrated 80% band was scored as broken.** Measured against a hard-coded
  90% target, a band covering 80% of outcomes looks 10 points short.

So the nominal level is **read as data** here. Where an artifact does not record it, this
module returns ``None`` and the page says *not reported* — it does not fall back to a guess.
Scoring a range against the wrong advertised level yields a confident verdict about nothing,
which is worse than no verdict.

Known artifact gap, deliberately surfaced rather than papered over: B_ML's conformal
intervals are configured by ``ConfigBML.nominal_pi`` (0.90) but that value is **never written
to any artifact**, so ``y_lo``/``y_hi`` arrive with no advertised level attached. Until the
pipeline records it, B_ML coverage is reported as a measurement without a pass/fail verdict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Quantile-column pattern, e.g. yhat_p10 / yhat_p90.
_QPAT = re.compile(r"^yhat_p(\d{1,2})$")

TERCILE_LABELS = ("smallest third", "middle third", "largest third")


@dataclass
class IntervalSpec:
    """Which columns carry the interval, and what level it advertises."""

    lo: str
    hi: str
    mid: Optional[str]
    #: Nominal coverage as a fraction (0.80), or None when no artifact records it.
    nominal: Optional[float]
    #: Human-readable provenance for the nominal, or why it is unknown.
    nominal_source: str
    #: e.g. "P10–P90" for hover labelling.
    band_label: str

    @property
    def nominal_known(self) -> bool:
        return self.nominal is not None


def detect_intervals(df: pd.DataFrame) -> Optional[IntervalSpec]:
    """Find interval columns in either schema and read the advertised level from them.

    Quantile columns name their own level: ``yhat_p10``/``yhat_p90`` advertise 80%. Generic
    ``y_lo``/``y_hi`` do not, so the level comes back unknown.
    """
    cols = set(df.columns)

    # Preferred: explicit quantile columns, which carry their own level.
    qs = sorted(int(m.group(1)) for c in cols for m in [_QPAT.match(c)] if m)
    if len(qs) >= 2:
        lo_q, hi_q = qs[0], qs[-1]
        mid = "yhat_p50" if "yhat_p50" in cols else None
        return IntervalSpec(
            lo=f"yhat_p{lo_q}", hi=f"yhat_p{hi_q}", mid=mid,
            nominal=(hi_q - lo_q) / 100.0,
            nominal_source=f"read from the quantile column names (p{lo_q} to p{hi_q})",
            band_label=f"P{lo_q}–P{hi_q}",
        )

    if {"y_lo", "y_hi"} <= cols:
        return IntervalSpec(
            lo="y_lo", hi="y_hi",
            mid="y_pred" if "y_pred" in cols else None,
            nominal=None,
            nominal_source=("this artifact does not record the advertised level, so no "
                            "pass/fail verdict is given"),
            band_label="lower–upper",
        )
    return None


def _valid(df: pd.DataFrame, spec: IntervalSpec) -> pd.DataFrame:
    need = [spec.lo, spec.hi, "y_true"]
    out = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if out.empty:
        return out
    out["_covered"] = ((out["y_true"] >= out[spec.lo]) &
                       (out["y_true"] <= out[spec.hi])).astype(float)
    out["_width"] = (out[spec.hi] - out[spec.lo]).astype(float)
    return out


def coverage_by_model(df: pd.DataFrame, spec: IntervalSpec) -> pd.DataFrame:
    """Per-model empirical coverage, mean width and n."""
    v = _valid(df, spec)
    if v.empty:
        return pd.DataFrame(columns=["model", "coverage", "mean_width", "n"])
    grp = "model" if "model" in v.columns else None
    if grp is None:
        v = v.assign(model="(all)")
        grp = "model"
    out = (v.groupby(grp)
             .agg(coverage=("_covered", "mean"), mean_width=("_width", "mean"),
                  n=("_covered", "size"))
             .reset_index())
    return out.sort_values("coverage", ascending=False)


def coverage_by_tercile(df: pd.DataFrame, spec: IntervalSpec,
                        model: Optional[str] = None) -> pd.DataFrame:
    """Coverage split by |y_true| tercile — the project's biggest known product defect.

    A band can look well calibrated on average while missing most of the largest days, and
    the largest days are the ones a cash buffer exists for. So this is reported alongside the
    overall figure rather than behind it.
    """
    v = _valid(df, spec)
    if model is not None and "model" in v.columns:
        v = v[v["model"] == model]
    if len(v) < 6:      # three buckets need enough rows to be meaningful at all
        return pd.DataFrame(columns=["tercile", "coverage", "n", "mean_magnitude"])
    try:
        v = v.assign(_t=pd.qcut(v["y_true"].abs(), 3, labels=list(TERCILE_LABELS)))
    except ValueError:
        # Degenerate magnitudes (too many ties) cannot be split into three buckets.
        return pd.DataFrame(columns=["tercile", "coverage", "n", "mean_magnitude"])
    out = (v.groupby("_t", observed=False)
             .agg(coverage=("_covered", "mean"), n=("_covered", "size"),
                  mean_magnitude=("y_true", lambda s: float(np.mean(np.abs(s)))))
             .reset_index().rename(columns={"_t": "tercile"}))
    return out


def reliability_curve(df: pd.DataFrame, spec: IntervalSpec,
                      model: Optional[str] = None, bins: int = 8) -> pd.DataFrame:
    """Where inside the band the actual value tends to fall.

    For each row we compute the actual's position within its own predicted band, then bin
    those positions. A well-calibrated band puts roughly equal mass in each bin; mass piling
    up at the edges (or outside) means the band is the wrong shape, not merely the wrong
    width — a distinction the overall coverage number cannot make.
    """
    v = _valid(df, spec)
    if model is not None and "model" in v.columns:
        v = v[v["model"] == model]
    if v.empty:
        return pd.DataFrame(columns=["position", "share", "n"])
    width = v["_width"].to_numpy(dtype=float)
    ok = width > 0
    if not ok.any():
        return pd.DataFrame(columns=["position", "share", "n"])
    pos = ((v["y_true"].to_numpy(float)[ok] - v[spec.lo].to_numpy(float)[ok]) / width[ok])
    edges = np.linspace(0.0, 1.0, bins + 1)
    inside = pos[(pos >= 0) & (pos <= 1)]
    counts, _ = np.histogram(inside, bins=edges)
    total = len(pos)
    rows = [{"position": f"{edges[i]:.2f}–{edges[i+1]:.2f}",
             "share": counts[i] / total if total else np.nan,
             "n": int(counts[i])} for i in range(bins)]
    below = int((pos < 0).sum())
    above = int((pos > 1).sum())
    rows.insert(0, {"position": "below band", "share": below / total, "n": below})
    rows.append({"position": "above band", "share": above / total, "n": above})
    return pd.DataFrame(rows)


def calibration_verdict(coverage: Optional[float], nominal: Optional[float],
                        n: int, tolerance: float = 0.05) -> Tuple[Optional[bool], str]:
    """Tri-state verdict: True / False / None (never verified).

    Returns ``None`` whenever the advertised level is unknown or the sample is too small to
    judge — never a pass. An unmeasured band must not read as a calibrated one.
    """
    if coverage is None or nominal is None:
        return None, ("The advertised coverage level is not recorded in this run's artifact, "
                      "so coverage is reported as a measurement without a verdict.")
    if n < 30:
        return None, (f"Only {n} predictions available — too few to judge calibration. "
                      f"Measured coverage is shown for information.")
    gap = coverage - nominal
    if abs(gap) <= tolerance:
        return True, (f"Coverage {coverage:.1%} is within {tolerance:.0%} of the advertised "
                      f"{nominal:.0%}.")
    if gap < 0:
        return False, (f"Coverage {coverage:.1%} is {abs(gap):.1%} below the advertised "
                       f"{nominal:.0%}. The range is too narrow and understates risk.")
    return False, (f"Coverage {coverage:.1%} is {gap:.1%} above the advertised {nominal:.0%}. "
                   f"The range is wider than necessary, which makes it less informative.")
