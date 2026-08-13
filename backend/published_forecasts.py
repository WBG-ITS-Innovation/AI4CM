"""Retain what was published, then score it when reality arrives.

Until now the forward artifacts were gitignored, so **nothing recorded what we told anyone**.
That is the gap this closes. A forecast that is not retained cannot be scored, and a system
that cannot be scored has to be trusted on assertion — which is the opposite of how the rest
of this project works.

Two halves:

``publish()``
    Copies a forward run into ``forecasts/published/<issue_date>/`` — **tracked**, unlike the
    working artifacts — with the predictions, the intervals, the gate verdicts in force at
    issue time, the ``recipe_id``, and full provenance. Immutable once written: re-publishing
    the same issue date requires ``overwrite=True``, because silently rewriting a published
    forecast destroys the only record of what was actually said.

``score_published()``
    Once truth arrives for a published date, computes realized absolute error, skill against
    the **same unified persistence ruler** the rest of the project uses, and interval hit
    rate. Writes ``forecasts/scorecard.csv``.

    Two rules hold that number honest, both added after the 2026-08-13 diagnostic found it
    wrong in two independent ways (see docs/sessions/2026-08-13-p0-scoring-correctness.md):
    the ruler is **read** from the published artifact's ``origin_value`` rather than
    recomputed here, and truth is read from the **raw** actuals rather than the zero-filled
    modelling series, so a day the data does not carry stays pending instead of being scored
    as an observation of zero.

--------------------------------------------------------------------------------
WHY THIS IS NOT A HOLDOUT READ
--------------------------------------------------------------------------------
The distinction is worth stating precisely, because it looks superficially like the thing we
have spent nine sessions refusing to do.

The sealed 2025 window is sealed against **model selection and evaluation-before-commitment**:
you must not look at it, then choose. Scoring a *published* forecast is the opposite ordering.
The prediction was committed, in writing, with a data fingerprint and a git SHA, **before**
the truth existed. Nothing can be tuned in response to it without that being visible as a
new recipe and a new issue date.

The hard rule this module enforces: **a published date is scored only once its truth is
present in the canonical dataset.** ``score_one()`` raises rather than returning a partial
result for a date whose truth has not arrived, and ``score_published()`` records such dates as
``pending``. So the scorer cannot reach forward into data we do not have, and it cannot be
pointed at the holdout to manufacture an accuracy number — the truth simply is not there to
read until the data file itself moves forward.

This is how accuracy gets demonstrated over time without spending the one-shot holdout.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PUBLISHED_ROOT = REPO / "forecasts" / "published"
SCORECARD = REPO / "forecasts" / "scorecard.csv"

#: Nominal interval width, for hit-rate reporting.
NOMINAL_COVERAGE = 0.80

SCORECARD_COLUMNS: Sequence[str] = (
    "issue_date", "target", "recipe_id", "horizon", "target_date",
    "p10", "p50", "p90", "y_true", "abs_error",
    "persistence_pred", "persistence_abs_error", "skill_vs_ruler_pct",
    "persistence_source",
    "inside_interval", "publication_verdict", "point_model", "target_transform",
    "data_sha_at_issue", "git_sha_at_issue", "scored_at_data_sha",
)

#: How close the recomputed ruler must sit to the artifact's own `origin_value`.
#: They are the same quantity by definition, so this is a float-round-trip
#: tolerance, not a margin: on the live issue the two agree exactly (0.000000 on
#: all three targets). Sized for a float64 through CSV text and nothing more --
#: at rtol 1e-6 this would have accepted a 46-lari divergence on a 46.5M figure,
#: which is a margin of acceptable disagreement, and there is no such thing here.
BASELINE_RTOL = 1e-9
BASELINE_ATOL = 0.01         # one tetri


class TruthNotAvailable(RuntimeError):
    """Raised when a published date is scored before its truth exists."""


# ── publishing ────────────────────────────────────────────────────────────────

def publish(forward_dir: Path, issue_date: Optional[str] = None,
            published_root: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Retain a forward run as an immutable published forecast.

    ``issue_date`` defaults to the origin date of the run, which is the honest label: it is
    the last date whose data informed the forecast.
    """
    forward_dir = Path(forward_dir)
    fc_path = forward_dir / "forward_forecast.csv"
    if not fc_path.exists():
        raise FileNotFoundError(f"no forward run at {forward_dir}")
    fc = pd.read_csv(fc_path)

    if issue_date is None:
        issue_date = str(pd.to_datetime(fc["origin_date"]).max().date())

    root = Path(published_root or PUBLISHED_ROOT)
    dest = root / issue_date
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists. A published forecast is the only record of what was "
            f"actually said, so rewriting it requires overwrite=True."
        )
    dest.mkdir(parents=True, exist_ok=True)

    fc.to_csv(dest / "forecast.csv", index=False)
    for name in ("forward_provenance.json", "forward_gates.json"):
        src = forward_dir / name
        if src.exists():
            shutil.copyfile(src, dest / name.replace("forward_", ""))

    # A manifest so a reader does not have to parse three files to learn what this is.
    prov = {}
    p = dest / "provenance.json"
    if p.exists():
        prov = json.loads(p.read_text())
    manifest = {
        "issue_date": issue_date,
        "targets": sorted(fc["target"].unique().tolist()),
        "horizons": sorted(int(h) for h in fc["horizon"].unique()),
        "target_dates": sorted(str(pd.to_datetime(d).date())
                               for d in fc["target_date"].unique()),
        "recipes": prov.get("recipes", []),
        "data_sha_at_issue": prov.get("data", {}).get("sha256"),
        "git_sha_at_issue": prov.get("code", {}).get("git_sha"),
        "calendar_version": prov.get("calendar_version"),
        "test_window_touched": prov.get("test_window_touched"),
        "note": ("Immutable record of a forecast issued before its truth existed. Scored by "
                 "backend/published_forecasts.score_published() once truth arrives."),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def list_published(published_root: Optional[Path] = None) -> List[Path]:
    root = Path(published_root or PUBLISHED_ROOT)
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "forecast.csv").exists())


# ── scoring ───────────────────────────────────────────────────────────────────

def _truth_series(data_path: Path, target: str) -> pd.Series:
    """Actuals **as recorded**, on the business-day index, with gaps left as gaps.

    Deliberately NOT ``b_ml_pipeline.to_business_index``, which is the *modelling*
    series: it fills a missing flow day with ``0.0`` and forward-fills a missing
    stock day. Both are right for fitting a model, which needs a dense index, and
    both are catastrophic here. A zero-filled gap is scored as a real observation
    of zero: the diagnostic removed 2025-08-11 from the actuals and the scorer
    reported ``y_true = 0.00`` with a fabricated absolute error of 77.6M instead of
    reporting the date as pending. ``TruthNotAvailable`` could never fire for a flow
    inside the data range, because after the fill every business day is finite.

    So: reindex, never fill. A day the file does not carry is NaN, and NaN reaches
    ``score_one`` as "truth has not arrived" — which is what it is. Carrying
    yesterday's balance forward would be a modelling assumption, not an observation,
    so stocks are not forward-filled here either.
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(subset=["date"]))
    s = df.set_index("date")[target].astype(float)
    bidx = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="B")
    s = s.reindex(bidx)
    s.index.freq = "B"
    return s


def _persistence_for(row: Dict, truth: pd.Series, td: pd.Timestamp,
                     horizon_steps: int) -> tuple[float, str, float]:
    """The h-step ruler for one published row: read it, recompute only as a fallback.

    ``ŷ(t+h) = y(t)`` is the project's single ruler, and ``forward_forecast.csv``
    already carries it per row as ``origin_value`` — ``forward_forecast.py`` documents
    that column as exactly this quantity and the Forecast page plots it as the
    benchmark. So the scorer reads it.

    It used to recompute it, from ``truth.iloc[pos - horizon_steps]`` with
    ``horizon_steps`` fixed at 5 for every row while ``row["horizon"]`` sat unread two
    lines away. A published issue has ONE origin, so the ruler is the same number for
    all five horizons; the scorer instead used five different values and only h=5 was
    right. 12 of 15 realized skills and all three per-target aggregates were wrong.

    The recomputation survives as a fallback for rows with no ``origin_value``, and is
    returned alongside so callers can cross-check the two. They are the same quantity,
    so a divergence means the actuals were revised under an already-issued forecast, or
    the artifact is wrong. ``score_published`` reports those rows; it does not drop
    them, because the number that was *published* is the ruler that was committed to,
    and a data revision should not silently erase a track record.
    """
    pos = truth.index.get_loc(td)
    h = int(row["horizon"]) if row.get("horizon") is not None else horizon_steps

    recomputed = np.nan
    if pos - h >= 0:
        cand = truth.iloc[pos - h]
        if np.isfinite(cand):
            recomputed = float(cand)

    raw_origin = row.get("origin_value")
    published = np.nan
    if raw_origin is not None:
        try:
            candidate = float(raw_origin)
        except (TypeError, ValueError):
            candidate = np.nan          # a malformed column is not a ruler
        if np.isfinite(candidate):
            published = candidate

    if not np.isfinite(published):
        return recomputed, "recomputed: truth[target_date - h business days]", recomputed
    return published, "artifact: origin_value", recomputed


def score_one(row: Dict, truth: pd.Series, horizon_steps: int = 5) -> Dict:
    """Score a single published prediction. Raises if its truth is not yet available.

    The persistence comparator is the same one the rest of the project uses,
    ``ŷ(t+h) = y(t)`` — and it is READ from the published artifact rather than rebuilt
    here. See ``_persistence_for``. ``horizon_steps`` remains only as the last-resort
    step count for a row that carries neither ``origin_value`` nor ``horizon``.
    """
    td = pd.Timestamp(row["target_date"]).normalize()
    if td not in truth.index or not np.isfinite(truth.get(td, np.nan)):
        arrived = truth.dropna()
        ends = arrived.index.max().date() if len(arrived) else "never"
        raise TruthNotAvailable(
            f"{row['target']} {td.date()}: truth is not in the canonical dataset yet "
            f"(data ends {ends}). A published forecast is "
            f"scored only once reality has arrived -- never by reaching into data we do "
            f"not have."
        )
    y = float(truth.loc[td])

    pers, source, recomputed = _persistence_for(row, truth, td, horizon_steps)

    ae = abs(y - float(row["p50"]))
    pae = abs(y - pers) if np.isfinite(pers) else np.nan
    skill = ((pae - ae) / pae * 100.0) if (np.isfinite(pae) and pae > 0) else np.nan
    return {
        "y_true": y,
        "abs_error": ae,
        "persistence_pred": pers,
        "persistence_abs_error": pae,
        "skill_vs_ruler_pct": skill,
        "persistence_source": source,
        # Not a scorecard column: the cross-check value, for callers that want to
        # verify the artifact's ruler against the actuals. See score_published.
        "persistence_recomputed": recomputed,
        "inside_interval": bool(float(row["p10"]) <= y <= float(row["p90"])),
    }


def baseline_agrees(published: float, recomputed: float) -> bool:
    """Do the two routes to the h-step ruler agree?

    ``origin_value`` and ``y(target_date - h business days)`` are the same quantity,
    and on the live issue they agree exactly — 0.000000 on all three targets. So this
    is a float-round-trip tolerance, not a margin of acceptable disagreement.
    """
    if not (np.isfinite(published) and np.isfinite(recomputed)):
        return True          # nothing to compare is not a disagreement
    return bool(np.isclose(published, recomputed,
                           rtol=BASELINE_RTOL, atol=BASELINE_ATOL))


def score_published(data_path: Path,
                    published_root: Optional[Path] = None,
                    scorecard_path: Optional[Path] = None,
                    horizon_steps: int = 5) -> Dict:
    """Score every published prediction whose truth has arrived; report the rest as pending.

    Never raises for a pending date -- that is the normal state of a fresh forecast. It
    raises only if a published directory is malformed, because that is a real defect.
    """
    data_path = Path(data_path)
    scored: List[Dict] = []
    pending: List[Dict] = []
    disputed: List[Dict] = []
    truth_cache: Dict[str, pd.Series] = {}

    from provenance import sha256_of
    current_sha = sha256_of(data_path)

    for d in list_published(published_root):
        fc = pd.read_csv(d / "forecast.csv")
        man = {}
        mp = d / "manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text())
        gates = {}
        gp = d / "gates.json"
        if gp.exists():
            gates = json.loads(gp.read_text())
        recipe_by_target = {r["target"]: r for r in man.get("recipes", [])}

        for _, row in fc.iterrows():
            target = row["target"]
            if target not in truth_cache:
                truth_cache[target] = _truth_series(data_path, target)
            rec = recipe_by_target.get(target, {})
            rid = rec.get("recipe_id", "")
            verdict = ""
            for g in gates.values():
                if isinstance(g, dict) and g.get("target") == target:
                    verdict = g.get("status", "")
            base = {
                "issue_date": man.get("issue_date", d.name),
                "target": target,
                "recipe_id": rid,
                "horizon": int(row["horizon"]),
                "target_date": str(pd.Timestamp(row["target_date"]).date()),
                "p10": float(row["p10"]), "p50": float(row["p50"]),
                "p90": float(row["p90"]),
                "publication_verdict": verdict,
                "point_model": row.get("point_model", rec.get("point_model", "")),
                "target_transform": row.get("target_transform",
                                            rec.get("target_transform", "raw")),
                "data_sha_at_issue": man.get("data_sha_at_issue"),
                "git_sha_at_issue": man.get("git_sha_at_issue"),
                "scored_at_data_sha": current_sha,
            }
            try:
                got = score_one(row, truth_cache[target], horizon_steps)
                base.update(got)
                # One ruler, one implementation: the artifact's origin_value and the
                # actuals at target_date - h business days are the same quantity. A
                # divergence means the actuals were revised under an issued forecast,
                # or the artifact is wrong. Reported rather than swallowed -- and the
                # row still scores against what was published, because that is the
                # comparator the forecast was committed against.
                if not baseline_agrees(got["persistence_pred"],
                                       got["persistence_recomputed"]):
                    disputed.append({
                        "target": target, "horizon": base["horizon"],
                        "target_date": base["target_date"],
                        "artifact_origin_value": got["persistence_pred"],
                        "recomputed_from_actuals": got["persistence_recomputed"],
                        "delta": abs(got["persistence_pred"]
                                     - got["persistence_recomputed"]),
                    })
                scored.append(base)
            except TruthNotAvailable:
                pending.append(base)

    out = Path(scorecard_path or SCORECARD)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scored, columns=list(SCORECARD_COLUMNS)) if scored else \
        pd.DataFrame(columns=list(SCORECARD_COLUMNS))
    df.to_csv(out, index=False)

    return {
        "scored": len(scored),
        "pending": len(pending),
        "issues": len(list_published(published_root)),
        "scorecard": str(out),
        "summary": summarize_scorecard(df),
        "pending_dates": sorted({(p["target"], p["target_date"]) for p in pending}),
        "baseline_disagreements": disputed,
    }


def summarize_scorecard(df: pd.DataFrame) -> Dict[str, Dict]:
    """Per-target realized performance. Empty until truth arrives, which is honest."""
    out: Dict[str, Dict] = {}
    if df.empty:
        return out
    for target, g in df.groupby("target"):
        valid = g.dropna(subset=["abs_error"])
        if valid.empty:
            continue
        mae = float(valid["abs_error"].mean())
        pmae = float(valid["persistence_abs_error"].mean()) \
            if valid["persistence_abs_error"].notna().any() else float("nan")
        out[str(target)] = {
            "n": int(len(valid)),
            "realized_mae": mae,
            "persistence_mae": pmae,
            "skill_vs_ruler_pct": ((pmae - mae) / pmae * 100.0)
            if (np.isfinite(pmae) and pmae > 0) else float("nan"),
            "interval_hit_rate": float(valid["inside_interval"].mean()),
            "nominal_coverage": NOMINAL_COVERAGE,
            "issues_covered": int(valid["issue_date"].nunique()),
        }
    return out
