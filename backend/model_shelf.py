"""What else was measured for a target, and how the champion compares to it.

Why this module exists
----------------------
The Forecast page showed one model per target and nothing else. A reader could see that
the champion beat a persistence benchmark, but not whether anything else had been tried,
how close the runner-up came, or how the champion compares to the method the Treasury
uses today. That reads as "here is our model" rather than "here is the best of what we
measured, and here is what came second".

``experiments/log.csv`` and ``experiments/runs/*.json`` already hold every run ever made,
with a MASE, a skill figure and a sentinel ratio each. This module reads that ledger and
answers three questions per target:

* Who is the champion, and on what measured evidence? (from ``registry/recipes.json``)
* What else was measured, and would it have cleared the gates? (from the ledger, judged by
  ``publication_gates``, which is the same code that decided the champion's own verdict)
* How does the champion compare to the Treasury's current planning method? (from
  ``ops_baseline``, the one construction the scorecard and the leaderboards share)

**This module chooses nothing.** Ranking runs for display is not selection: the champion
comes from the registry, which is hand-edited and which nothing here writes. Every row it
reads is a DEV-window measurement, so there is no window question either. What it removes
is the reader's need to take "this is the best model" on trust.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_CSV = REPO_ROOT / "experiments" / "log.csv"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
DATA_DEFAULT = REPO_ROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"

#: The window every ledger row was measured on, and the window this module reports from.
MEASURED_WINDOW = "2024 (dev)"

#: How many alternatives the Forecast page shows beside the champion.
DEFAULT_ALTERNATIVES = 2


@dataclass
class ModelEvidence:
    """One model's measured record for one target. Every field is read, never derived."""

    target: str
    model: str
    run_id: str
    dev_mae: Optional[float] = None
    mase: Optional[float] = None
    skill_vs_ruler_pct: Optional[float] = None
    sentinel_ratio: Optional[float] = None
    ruler_mae: Optional[float] = None
    #: Would this model clear the publication gates on what was measured?
    gate_eligible: bool = False
    verdict: str = "unknown"
    #: The gates as ``publication_gates`` evaluates them, so a page can show reasons.
    gates: Dict = field(default_factory=dict)
    #: Plain sentence naming why it is or is not gate-eligible.
    eligibility_plain: str = ""
    is_champion: bool = False
    #: Can this model be run as a forward point forecast in this build?
    #:
    #: Not every measured model can. The ledger holds quantile-family runs whose estimators
    #: are not in ``b_ml_pipeline.available_models()``, so offering them in a comparison
    #: would produce a failure instead of a forecast. They still belong on the shelf with
    #: their measured evidence; what they cannot do is be re-run from a button.
    runnable: Optional[bool] = None
    note: str = ""


def _float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _read_ledger() -> List[dict]:
    if not LOG_CSV.exists():
        return []
    with LOG_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run_detail(run_id: str) -> dict:
    """The run's JSON sidecar, which carries the model name and the ruler MAE.

    ``log.csv`` records neither, so a ledger row on its own cannot say which model produced
    it. Falling back to parsing the run_id would be guessing at a string format; an absent
    sidecar returns an empty dict and the row is skipped rather than mislabelled.
    """
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _judge(target: str, row: dict, detail: dict) -> tuple:
    """Run one ledger row through the real publication gates.

    Uses ``publication_gates`` rather than a threshold comparison written here, because a
    second copy of the gate policy is a second thing to drift. Only the measurements the
    ledger actually holds are passed; everything else stays ``None``, which the gate code
    reads as "not measured" and never as "passed".
    """
    from publication_gates import Measured, decide

    measured = Measured(
        target=target,
        mase=_float(row.get("mase")),
        sentinel_ratio=_float(row.get("sentinel_ratio")),
        skill_vs_ruler_pct=_float(row.get("skill_vs_ruler")),
        coverage=_float(row.get("coverage_mid")),
        has_intervals=bool(detail.get("coverage_terciles")) or None,
    )
    result = decide(measured)
    gates = result["gates"]
    accuracy = gates.get("accuracy_vs_naive", {})
    eligible = accuracy.get("passed") is True
    if eligible:
        plain = ("This model beat the naive benchmark on data it never saw, so it is a "
                 "candidate the gates would consider.")
    elif accuracy.get("passed") is False:
        plain = ("This model was less accurate than simply repeating the same weekday last "
                 "week, so the gates rule it out whatever else it does well.")
    else:
        plain = ("Accuracy against the naive benchmark was never measured for this run, so "
                 "the gates cannot consider it.")
    return eligible, result["verdict"], gates, plain


#: Suffixes the ledger appends to a model name when a run is re-logged rather than re-fitted.
#: They name the bookkeeping, not a different model, so they are stripped before deduplication.
_RELOG_SUFFIXES = ("_recomputed", "_relogged", "_reproduction")


def base_model_name(model: str) -> str:
    """The model itself, with run bookkeeping stripped off.

    ``LightGBM_L1+ratio`` and ``LightGBM_L1`` are the same estimator under two target
    transforms, and ``GBQuantile_tuned_recomputed`` is one run of ``GBQuantile_tuned`` whose
    metrics were recomputed. Without this the shelf shows the same model twice and looks
    twice as deep as it is.
    """
    name = str(model or "").split("+")[0].strip()
    for suffix in _RELOG_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip("_ ") or str(model or "").strip()


def measured_models_for(target: str) -> List[ModelEvidence]:
    """Every distinct model measured for ``target``, best first.

    Deduplicated by model name, keeping the best MASE. The ledger holds repeated runs of the
    same configuration (reproductions, re-logs after a schema change), and listing a model
    five times would make the shelf look deeper than it is.
    """
    best: Dict[str, ModelEvidence] = {}
    for row in _read_ledger():
        if row.get("target") != target:
            continue
        run_id = row.get("run_id") or ""
        detail = _run_detail(run_id)
        model = str(detail.get("model") or "").strip()
        if not model:
            continue
        mase = _float(row.get("mase"))
        if mase is None:
            # A run with no MASE cannot be compared to anything on the shared ruler, so it
            # would appear on the shelf as a name with no evidence beside it.
            continue
        eligible, verdict, gates, plain = _judge(target, row, detail)
        candidate = ModelEvidence(
            target=target, model=model, run_id=run_id,
            dev_mae=_float(row.get("dev_mae")), mase=mase,
            skill_vs_ruler_pct=_float(row.get("skill_vs_ruler")),
            sentinel_ratio=_float(row.get("sentinel_ratio")),
            ruler_mae=_float(detail.get("ruler")),
            gate_eligible=eligible, verdict=verdict, gates=gates,
            eligibility_plain=plain, note=str(row.get("note") or ""),
        )
        key = base_model_name(model)
        held = best.get(key)
        if held is None or (candidate.mase or 9e9) < (held.mase or 9e9):
            best[key] = candidate
    return sorted(best.values(), key=lambda e: e.mase if e.mase is not None else 9e9)


def champion_model(target: str) -> Optional[str]:
    """The registry's champion point model for ``target``, or ``None`` if it has no recipe."""
    from registry import load_registry

    for rec in load_registry()["recipes"]:
        if rec["target"] == target:
            return rec["point_model"]
    return None


def alternatives_for(target: str, k: int = DEFAULT_ALTERNATIVES,
                     runnable_models: Optional[set] = None) -> List[ModelEvidence]:
    """The ``k`` best gate-eligible models for ``target`` that are not the champion.

    Gate-eligible means the accuracy gate passed on what was measured. A model that loses to
    the naive benchmark is excluded rather than listed with a caveat: showing it as an
    "alternative" would invite a comparison the gates already refuse to entertain.

    Returns fewer than ``k``, including none at all, when the ledger holds no more. That is a
    real answer about this target and the page states it as one.
    """
    champ = (champion_model(target) or "").strip()
    out = []
    for ev in measured_models_for(target):
        if not ev.gate_eligible:
            continue
        # The champion's own model, in any of its logged variants ("LightGBM_L1",
        # "LightGBM_L1+ratio"), is the champion rather than an alternative to it.
        if champ and base_model_name(ev.model) == base_model_name(champ):
            continue
        if runnable_models is not None:
            ev.runnable = base_model_name(ev.model) in set(runnable_models)
        out.append(ev)
        if len(out) >= int(k):
            break
    return out


# ---------------------------------------------------------------------------
# The Treasury's current method, on the same window
# ---------------------------------------------------------------------------

def ops_comparison(target: str, data_path: Optional[Path] = None,
                   horizon: int = 5) -> Dict:
    """The Treasury planning method's own error on 2024, beside the champion's.

    Both figures are mean absolute errors over the same calendar year, so the comparison is
    like for like. The method's predictions come from ``ops_baseline``, which is the single
    construction the scorecard, the leaderboards and the runners all share; nothing is
    re-derived here.

    On a stock target the method is not defined at all -- it aggregates a flow to an annual
    total, and a balance level has no annual total -- so this returns ``available: False``
    with that reason stated. A missing comparison that says why is usable; a blank is not.
    """
    import numpy as np
    import pandas as pd

    from ops_baseline import (REASON_STOCK, ops_prediction_for, skill_vs,
                              vintage_cache)
    from target_kinds import is_stock

    path = Path(data_path or DATA_DEFAULT)
    out = {"target": target, "available": False, "window": MEASURED_WINDOW,
           "ops_mae": None, "model_mae": None, "skill_pct": None, "n": 0, "reason": ""}

    if is_stock(target):
        out["reason"] = REASON_STOCK
        return out
    if not path.exists():
        out["reason"] = f"the data file was not found at {path}"
        return out

    try:
        df = pd.read_csv(path, usecols=["date", target], parse_dates=["date"])
    except (OSError, ValueError) as exc:
        out["reason"] = f"the data file could not be read ({exc})"
        return out

    dev = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].dropna()
    dev = dev[dev["date"].dt.weekday < 5]
    if dev.empty:
        out["reason"] = "no 2024 rows are present in this data file"
        return out

    origins = dev["date"] - pd.tseries.offsets.BDay(int(horizon))
    cache = vintage_cache(path, target, origins, dev["date"])
    preds = np.array([ops_prediction_for(t, o, cache)[0]
                      for t, o in zip(dev["date"], origins)], dtype=float)
    ok = np.isfinite(preds)
    if not ok.any():
        out["reason"] = ("no complete three-year window existed for 2024, so the method "
                         "cannot speak for this year")
        return out

    ops_mae = float(np.mean(np.abs(dev[target].to_numpy()[ok] - preds[ok])))
    out.update(available=True, ops_mae=ops_mae, n=int(ok.sum()))

    from registry import load_registry
    for rec in load_registry()["recipes"]:
        if rec["target"] == target:
            model_mae = rec["dev_credentials"].get("dev_mae")
            out["model_mae"] = model_mae
            if model_mae is not None:
                out["skill_pct"] = float(skill_vs(model_mae, ops_mae))
            break
    return out


# ---------------------------------------------------------------------------
# The champion, in one sentence
# ---------------------------------------------------------------------------

def champion_sentence(target: str, ops: Optional[Dict] = None) -> str:
    """One plain sentence saying what the champion is and what earned it that place.

    Written to be read aloud. It names the evidence, not the algorithm: a reader deciding
    whether to trust a number needs to know it was measured on data the model never saw,
    and does not need to know what gradient boosting is.
    """
    from registry import load_registry

    rec = next((r for r in load_registry()["recipes"] if r["target"] == target), None)
    if rec is None:
        return (f"There is no champion for {target}, because no model has been measured "
                f"against the benchmark on this line yet.")

    cred = rec["dev_credentials"]
    mase = cred.get("mase")
    ruler = cred.get("skill_vs_ruler_pct")
    parts = [
        f"{rec['point_model']} is the champion for {target}. It was chosen by measured "
        f"evidence on {MEASURED_WINDOW}, a year it was never fitted on."
    ]
    if mase is not None:
        better = (1.0 - float(mase)) * 100.0
        if better >= 0:
            parts.append(f"It is {better:.1f}% more accurate than repeating the same "
                         f"weekday from the previous week.")
        else:
            parts.append(f"It is {abs(better):.1f}% less accurate than repeating the same "
                         f"weekday from the previous week, which is why it is withheld.")
    if ruler is not None:
        parts.append(f"Against carrying the last known value forward it is {float(ruler):.1f}% "
                     f"more accurate.")
    ops = ops if ops is not None else {}
    if ops.get("available") and ops.get("skill_pct") is not None:
        skill = float(ops["skill_pct"])
        if skill >= 0:
            parts.append(f"Against the Treasury's current planning method it is {skill:.1f}% "
                         f"more accurate over the same year.")
        else:
            parts.append(f"Against the Treasury's current planning method it is "
                         f"{abs(skill):.1f}% less accurate over the same year.")
    elif ops.get("reason"):
        parts.append(f"It has no comparison against the Treasury's current planning method, "
                     f"because that method is {ops['reason']}.")
    return " ".join(parts)


def shelf_for(target: str, k: int = DEFAULT_ALTERNATIVES,
              data_path: Optional[Path] = None,
              runnable_models: Optional[set] = None) -> Dict:
    """Everything the Forecast page needs for one target's Models panel.

    ``runnable_models`` is the set of estimator names this build can actually fit, read live
    from the backend interpreter by the caller. Passing it marks each alternative, so the
    comparison button offers only models that will produce a forecast rather than an error.
    """
    ops = ops_comparison(target, data_path)
    champ = champion_model(target)
    alts = alternatives_for(target, k, runnable_models=runnable_models)
    return {
        "target": target,
        "champion_model": champ,
        "champion_sentence": champion_sentence(target, ops),
        "ops": ops,
        "alternatives": alts,
        "no_alternatives_reason": _no_alternatives_reason(target, alts),
        "measured_window": MEASURED_WINDOW,
        "comparable": [a.model for a in alts if a.runnable],
    }


def _no_alternatives_reason(target: str, alternatives: List[ModelEvidence]) -> str:
    """Why the alternatives list is empty, when it is. Never left blank."""
    if alternatives:
        return ""
    measured = measured_models_for(target)
    if not measured:
        return (f"No other model has been measured on {target} yet, so there is nothing to "
                f"compare the champion against on this line.")
    eligible = [m for m in measured if m.gate_eligible]
    if not eligible:
        return (f"{len(measured)} other models have been measured on {target} and none of "
                f"them beat the naive benchmark, so none is a candidate the gates would "
                f"consider. That is a fact about this line, not about the models.")
    return (f"Every gate-eligible model measured on {target} is the champion itself, so "
            f"there is no runner-up to show.")
