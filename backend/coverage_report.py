"""Per-family interval coverage, measured from materialised run artifacts.

Why this module exists
----------------------
Coverage was being *computed* in three places and *gated* in none. Each family wrote its own
figure into its own artifact against its own advertised level, the frontend recomputed a fourth,
and ``publication_gates.Measured.coverage`` -- the field that actually decides whether a band is
publishable -- was never populated by anything. Every recipe in ``registry/recipes.json``
therefore carries the line *"This model reports no prediction intervals"* while publishing
``p10``/``p90`` from GBQuantile.

So this is the single reader. It scores bands off ``predictions_long.csv`` and hands
``publication_gates`` a number, so a miscalibrated band can change a verdict instead of being
silently absent from one.

Two rules it will not bend
--------------------------
**1. Each family is scored against its own advertised level.** E_QUANTILE's ``yhat_p10``/
``yhat_p90`` advertise 80%; B_ML and C_DL are configured at 90% (``nominal_pi``). Scoring an 80%
band against 90% manufactures a 10-point defect, which is the bug ``frontend/intervals.py`` was
written to fix. Where a run records no level, coverage comes back with ``nominal=None`` and no
verdict -- never a guess.

**2. "A big day" is defined before the day happens.** Conditional coverage is bucketed on the
forecast, never on the actual. See ``conformal`` for the control experiment showing that a band
which is correct by construction scores 37.8% when bucketed on the outcome.

Windows
-------
This module only ever *reads* rows that a pipeline already produced and wrote to disk; it fits
nothing and chooses nothing. It reports which window each scored row falls in, so a caller can
see whether a figure is DEV or holdout without inferring it from a path. Because publication is a
choice, :func:`coverage_for_publication` refuses to hand back a figure computed on report-only
rows unless the caller states it is reporting rather than selecting.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation_windows import window_for

REPO = Path(__file__).resolve().parent.parent

#: Interval columns and advertised level per family.
#:
#: The level is a fact about how the family was configured, not a preference: E_QUANTILE's band is
#: p10-p90 by construction, and B_ML/C_DL set ``nominal_pi = 0.90``. A_STAT emits ``y_lo``/``y_hi``
#: columns but never populates them (measured: 0 of 156 rows non-null in both logged runs), so it
#: is listed with ``None`` and reports as having no intervals rather than as failing.
FAMILY_INTERVALS: Dict[str, Dict] = {
    "e_quantile": {"lo": "yhat_p10", "hi": "yhat_p90", "mid": "yhat_p50", "nominal": 0.80,
                   "nominal_source": "read from the quantile column names (p10 to p90)"},
    "b_ml": {"lo": "y_lo", "hi": "y_hi", "mid": "y_pred", "nominal": 0.90,
             "nominal_source": "ConfigBML.nominal_pi, captured in artifacts/config.json"},
    "c_dl": {"lo": "y_lo", "hi": "y_hi", "mid": "y_pred", "nominal": 0.90,
             "nominal_source": "ConfigCDL.nominal_pi, captured in artifacts/config.json"},
    "a_stat": {"lo": "y_lo", "hi": "y_hi", "mid": "y_pred", "nominal": None,
               "nominal_source": "A_STAT writes the interval columns but leaves them empty"},
}

#: Fraction defining the "large day" tail for the headline conditional figure.
LARGE_DAY_QUANTILE = 0.90


@dataclass
class CoverageMeasurement:
    """One family/target/model band, scored. ``None`` means not measured, never "passed"."""

    family: str
    target: str
    model: str
    n: int
    nominal: Optional[float]
    nominal_source: str
    overall: Optional[float] = None
    large_day: Optional[float] = None
    n_large_day: int = 0
    mean_width: Optional[float] = None
    basis: str = ""
    windows: Tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> Dict:
        return {"family": self.family, "target": self.target, "model": self.model,
                "n": self.n, "nominal": self.nominal, "nominal_source": self.nominal_source,
                "overall_coverage": self.overall, "large_day_coverage": self.large_day,
                "n_large_day": self.n_large_day, "mean_width": self.mean_width,
                "magnitude_basis": self.basis, "windows": list(self.windows),
                "note": self.note}


def _magnitude_at_origin(g: pd.DataFrame, mid: str) -> Tuple[Optional[np.ndarray], str]:
    """The day-size signal, taken from what was knowable at the origin.

    Preference order is the forecast, then the level the day opened at. ``y_true`` is never a
    candidate: bucketing on it conditions coverage on the outcome.
    """
    for col, label in ((mid, "the forecast for that day"),
                       ("origin_value", "the level the day opened at")):
        if col and col in g.columns and g[col].notna().any():
            return np.abs(g[col].to_numpy(dtype=float)), label
    return None, ("no forecast or opening level in the artifact, so day size cannot be measured "
                  "from what was known at the origin")


def score_band(g: pd.DataFrame, *, family: str, target: str, model: str,
               spec: Dict) -> CoverageMeasurement:
    """Score one band. Never raises on thin or absent data -- it reports why instead."""
    lo_c, hi_c, mid = spec["lo"], spec["hi"], spec.get("mid")
    m = CoverageMeasurement(family=family, target=target, model=model, n=0,
                            nominal=spec["nominal"], nominal_source=spec["nominal_source"])
    if lo_c not in g.columns or hi_c not in g.columns:
        m.note = f"artifact has no {lo_c}/{hi_c} columns, so it carries no intervals"
        return m
    g = g.dropna(subset=[c for c in (lo_c, hi_c, "y_true") if c in g.columns])
    m.n = int(len(g))
    if m.n == 0:
        m.note = (f"{lo_c}/{hi_c} are present but empty in every row, so this family published "
                  f"no intervals for this run")
        return m

    y = g["y_true"].to_numpy(dtype=float)
    lo = g[lo_c].to_numpy(dtype=float)
    hi = g[hi_c].to_numpy(dtype=float)
    inside = (y >= lo) & (y <= hi)
    m.overall = float(inside.mean())
    m.mean_width = float(np.mean(hi - lo))
    if "target_date" in g.columns:
        m.windows = tuple(sorted({window_for(t) for t in pd.to_datetime(g["target_date"])}))

    mag, label = _magnitude_at_origin(g, mid)
    m.basis = label
    if mag is None:
        m.note = "overall coverage measured; day-size split unavailable"
        return m
    ok = np.isfinite(mag)
    if ok.sum() < 10:
        m.note = "overall coverage measured; too few rows with a usable day-size signal"
        return m
    sel = ok & (mag >= np.quantile(mag[ok], LARGE_DAY_QUANTILE))
    m.n_large_day = int(sel.sum())
    if m.n_large_day:
        m.large_day = float(inside[sel].mean())
    return m


def measure_run_dir(run_dir: Path) -> List[CoverageMeasurement]:
    """Score every family/target/model band under one ``forecast_runs/<date>/`` directory."""
    out: List[CoverageMeasurement] = []
    for path in sorted(glob.glob(str(run_dir / "*" / "predictions_long.csv"))
                       + glob.glob(str(run_dir / "*" / "*" / "predictions_long.csv"))):
        p = Path(path)
        family = next((part for part in p.parts if part in FAMILY_INTERVALS), None)
        if family is None:
            continue
        spec = FAMILY_INTERVALS[family]
        df = pd.read_csv(path)
        if "target" not in df.columns:
            continue
        keys = ["target", "model"] if "model" in df.columns else ["target"]
        for key, g in df.groupby(keys):
            target, model = key if isinstance(key, tuple) else (key, "(all)")
            out.append(score_band(g, family=family, target=str(target), model=str(model),
                                  spec=spec))
    return out


def measure_all(root: Optional[Path] = None) -> List[CoverageMeasurement]:
    """Score every band in every logged run under ``backend/forecast_runs/``."""
    root = root or (REPO / "backend" / "forecast_runs")
    out: List[CoverageMeasurement] = []
    for run_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        out.extend(measure_run_dir(run_dir))
    return out


def to_frame(ms: List[CoverageMeasurement]) -> pd.DataFrame:
    return pd.DataFrame([m.as_dict() for m in ms])


class SelectionOnReportedCoverageError(RuntimeError):
    """A coverage figure from report-only rows was about to inform a choice."""


def coverage_for_publication(ms: List[CoverageMeasurement], target: str,
                             *, interval_model: Optional[str] = None,
                             purpose: str = "report") -> Optional[CoverageMeasurement]:
    """The coverage figure that describes THIS recipe's band, for the publication gate.

    ``interval_model`` must be the recipe's declared interval producer (``GBQuantile`` for all
    three live recipes). Matching on it is not a nicety: without it this returned whichever
    measurement had the most rows, which selected C_DL's ``DCNN`` band for revenues and B_ML's
    ``CatBoost_L1`` for the stock target -- both healthy, neither the band the recipe actually
    ships. Gating a recipe on another model's interval reports a calibration the client will
    never receive, and it would have read as a pass for exactly the two targets whose own
    quantile bands measure worst.

    Returns ``None`` rather than substituting a different model when no measurement of the
    declared producer exists. An absent figure is reported as absent; a wrong one is worse.

    Among matching measurements, a run confined to the sealed reporting window is preferred over
    one that mixes windows, because a figure pooled across train, dev and test does not describe
    out-of-sample calibration at all.

    ``purpose`` mirrors ``evaluation_windows``: publication is a *report* on what the holdout
    would have shown, never a selection.
    """
    cands = [m for m in ms if m.target == target and m.nominal is not None
             and m.overall is not None]
    if interval_model:
        cands = [m for m in cands if m.model == interval_model]
    if not cands:
        return None

    def rank(m: CoverageMeasurement):
        single_window = len(m.windows) == 1
        return (single_window, m.n)

    best = max(cands, key=rank)
    if purpose == "selection" and not set(best.windows) <= {"train", "dev"}:
        raise SelectionOnReportedCoverageError(
            f"coverage for {target!r} was measured on {sorted(best.windows)} rows, which are "
            f"report-only. Publication may report this figure; nothing may be chosen from it.")
    return best
