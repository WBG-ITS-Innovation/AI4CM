"""Minimal model registry — what is promoted, on what evidence, and what it is not.

One recipe per target champion. The registry's job is to make a claim auditable: every
number in it is linked by ``run_id`` to a row in ``experiments/log.csv``, and every status
is one that can be defended.

Three deliberate design rules, because a registry that flatters is worse than none:

* **Honest statuses only.** Nothing here is ``approved``. ``approved_by`` is ``null`` on
  every recipe, because no human has approved anything. Status is ``candidate --
  pre-tuning``: workstream 2 (hyperparameter search) has not run, and workstream 4 (target
  scaling) has not run, so no recipe here is a tuned final model.
* **Gates are reported, never summarised away.** Where a gate fails the recipe says so and
  says why in plain language. The signal gate fails on both flow targets; that is the most
  important fact in this file.
* **A recipe records when it is NOT the best measured option.** On Expenditure the
  DEV-best model is a squared-error ``HistGBDT`` on the pre-WS3 feature set, not the
  promoted recipe. Suppressing that would make the registry a sales document.

``recipes.json`` is the data; this module is the loader plus the integrity checks that keep
the two from drifting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
RECIPES_JSON = REGISTRY_DIR / "recipes.json"

SCHEMA_VERSION = 1

#: Statuses a recipe may carry. Nothing stronger than `candidate` exists yet, and
#: `approved` is intentionally absent from the code until an approval workflow exists --
#: a status that cannot be reached cannot be claimed by accident.
VALID_STATUSES = (
    "candidate -- pre-tuning",
    "candidate -- tuned",
    "withdrawn",
)

#: Publication verdicts. `withheld_as_forecast` is the honest label for a model whose
#: point predictions are usable as a central-tendency estimate but which fails the signal
#: gate -- i.e. we cannot claim it anticipates events. The numbers are still shown; what is
#: withheld is the claim, not the data.
PUBLICATION_VERDICTS = (
    "publishable",
    "withheld_as_forecast",
    "withheld",
)


def load_registry(path: Optional[Path] = None) -> Dict:
    p = Path(path or RECIPES_JSON)
    if not p.exists():
        raise FileNotFoundError(
            f"registry not found at {p}. It is version-controlled; a missing file means "
            f"the checkout is incomplete."
        )
    reg = json.loads(p.read_text(encoding="utf-8"))
    validate_registry(reg)
    return reg


def validate_registry(reg: Dict) -> None:
    """Fail loudly on a malformed or dishonest registry.

    The checks that matter are not schema checks. They are: no recipe claims approval it
    does not have, every quoted metric is traceable to a logged run, and a failing gate is
    accompanied by a reason a non-technical reader can act on.
    """
    problems: List[str] = []
    if reg.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {reg.get('schema_version')} != {SCHEMA_VERSION}")

    seen = set()
    for r in reg.get("recipes", []):
        rid = r.get("id", "<missing id>")
        if rid in seen:
            problems.append(f"duplicate recipe id {rid}")
        seen.add(rid)
        for key in ("id", "target", "family", "point_model", "feature_groups",
                    "calendar_version", "scaling", "status", "approved_by",
                    "dev_credentials", "publication"):
            if key not in r:
                problems.append(f"{rid}: missing {key!r}")
        if r.get("status") not in VALID_STATUSES:
            problems.append(f"{rid}: status {r.get('status')!r} not in {VALID_STATUSES}")
        if r.get("approved_by") is not None:
            problems.append(
                f"{rid}: approved_by is set, but no approval workflow exists. A recipe "
                f"cannot be approved until one does."
            )
        cred = r.get("dev_credentials", {})
        if not cred.get("run_id"):
            problems.append(f"{rid}: dev_credentials has no run_id -- the metrics are "
                            f"not traceable to experiments/log.csv")
        pub = r.get("publication", {})
        if pub.get("verdict") not in PUBLICATION_VERDICTS:
            problems.append(f"{rid}: publication verdict {pub.get('verdict')!r} invalid")
        if pub.get("verdict") != "publishable" and not pub.get("reason_plain"):
            problems.append(f"{rid}: withheld without a plain-language reason")
        for g in cred.get("gates", {}).values():
            if isinstance(g, dict) and g.get("passed") is False and not g.get("reason_plain"):
                problems.append(f"{rid}: a failing gate has no plain-language reason")

    if problems:
        raise ValueError("registry validation failed:\n  - " + "\n  - ".join(problems))


def recipe_for(target: str, path: Optional[Path] = None) -> Dict:
    for r in load_registry(path)["recipes"]:
        if r["target"] == target:
            return r
    raise KeyError(f"no registry recipe for target {target!r}")


def verify_against_log(path: Optional[Path] = None) -> Dict:
    """Check every recipe's quoted DEV metrics against experiments/log.csv.

    This is the check that makes the registry auditable rather than decorative: a metric
    that cannot be matched to its logged run is reported, not trusted.
    """
    from experiment_log import read_log

    rows = {r["run_id"]: r for r in read_log()}
    problems: List[str] = []
    checked = 0
    for rec in load_registry(path)["recipes"]:
        cred = rec["dev_credentials"]
        rid = cred["run_id"]
        row = rows.get(rid)
        if row is None:
            problems.append(f"{rec['id']}: run_id {rid} not in experiments/log.csv")
            continue
        for field, key in (("dev_mae", "dev_mae"), ("mase", "mase"),
                           ("skill_vs_ruler_pct", "skill_vs_ruler"),
                           ("sentinel_ratio", "sentinel_ratio")):
            if cred.get(field) is None:
                continue
            logged = row.get(key)
            if logged in (None, ""):
                problems.append(f"{rec['id']}: log row has no {key}")
                continue
            if abs(float(logged) - float(cred[field])) > 1e-3:
                problems.append(
                    f"{rec['id']}: {field} {cred[field]} != logged {key} {logged}")
            checked += 1
    return {"recipes": len(load_registry(path)["recipes"]),
            "metrics_checked": checked, "ok": not problems, "problems": problems}


# ══════════════════════════════════════════════════════════════════════════════
# CHAMPION POLICY — what a refit does and does not change
#
# P1(2). There is no champion reselection in this project and none is being built. What
# was missing is that the *absence* was invisible: a client loading new data sees a run
# complete, numbers move, and would reasonably assume the model was re-chosen for the new
# period. It was not. Only the fitted parameters changed.
#
# That is a defensible design -- reselecting on client data would mean selecting on the
# LIVE window, which `evaluation_windows.assert_selection_free` now forbids -- but it has
# a consequence the registry already records and nothing surfaced: a champion chosen in
# one regime may be the wrong choice in another. The Revenues recipe carries the measured
# version of exactly that risk.
#
# So this is written as a field the Agent can read, sourced FROM the recipe rather than
# retyped here. A caveat transcribed into a second place is a caveat that will disagree
# with itself.
# ══════════════════════════════════════════════════════════════════════════════

#: What a run does when new data arrives.
RESELECTION_NONE = "none"
REFIT_ONLY = "refit_only"


def champion_policy(path: Optional[Path] = None) -> Dict:
    """Machine-readable statement of champion fixity, with the registry's own caveats.

    Every field is derived from ``recipes.json``; nothing is hardcoded except the policy
    names themselves. A consumer should render ``statement`` verbatim and must not infer
    from a completed run that a champion was re-chosen for the period it covers.
    """
    reg = load_registry(path)
    recipes = reg.get("recipes", [])

    caveats: List[Dict] = []
    for r in recipes:
        cav = r.get("scaling_caveat")
        if not cav:
            continue
        caveats.append({
            "target": r["target"],
            "recipe_id": r["id"],
            "applies_to": r.get("params", {}).get("target_transform"),
            "finding": cav.get("finding"),
            "evidence": cav.get("evidence"),
            "how_to_quote": cav.get("how_to_quote"),
            "study": cav.get("study"),
        })

    return {
        "reselection": RESELECTION_NONE,
        "on_new_data": REFIT_ONLY,
        "statement": (
            "Champions are fixed. New data changes the fitted parameters of the recipe "
            "already chosen for each target; it never changes which model, which features "
            "or which target transform is used. Those were chosen once, on TRAIN and DEV, "
            "and are recorded in registry/recipes.json. No run re-selects them."
        ),
        "why": (
            "Re-selecting on data that arrived after the holdout was sealed would mean "
            "choosing a model on the LIVE window, which is scored but never selected on "
            "(evaluation_windows.assert_selection_free). Reselection would require a fresh "
            "TRAIN/DEV pass, which is a deliberate exercise and not a side effect of "
            "loading a file."
        ),
        "risk": (
            "A champion chosen in one regime can be the wrong choice in another, and "
            "nothing in a completed run flags that. The caveats below are the measured "
            "instances the registry already records."
        ),
        "reselection_requires": [
            "a TRAIN/DEV selection pass (never LIVE)",
            "a hand-edited registry/recipes.json — nothing writes it",
        ],
        "caveats": caveats,
        "recipes_fixed": [
            {"target": r["target"], "recipe_id": r["id"], "point_model": r["point_model"],
             "target_transform": r.get("params", {}).get("target_transform", "raw"),
             "status": r.get("status"), "approved_by": r.get("approved_by")}
            for r in recipes
        ],
    }
