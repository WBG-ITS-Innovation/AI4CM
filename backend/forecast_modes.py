"""Two forecast modes, separated at the boundary rather than by convention.

**OFFICIAL** — the target's registry champion recipe, refitted on all data through the data end,
published immutably. The model is deliberately **not** selectable: the point of a champion is that
it was chosen on recorded evidence, and letting an analyst swap it while keeping the "official"
label would make the label meaningless.

**EXPLORATORY** — any model, any target, any horizon. Runs and displays. It must never reach
``forecasts/published/``, never reach the scorecard, and never export as if official.

The separation is enforced *here*, not in the UI. A page can forget a flag; ``publish_official``
refuses an exploratory result outright, and ``ExploratoryResult`` has no publish path at all. One
test (``test_forecast_modes.py``) is the completion criterion for this work.

--------------------------------------------------------------------------------
HORIZON HONESTY
--------------------------------------------------------------------------------
Everything that makes a forecast trustworthy on this project is measured at **h=5 business days**:
the shared persistence ruler, recipe selection, and every gate. A different horizon has no
selected recipe and no measured gate, so it is exploratory **whatever mode is requested** — and
``official_run`` refuses rather than quietly relabelling. That refusal is the honest behaviour: an
official-looking forecast at h=1 would carry credentials that were never earned at h=1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

#: The only horizon at which the ruler, recipe selection and gates were measured.
VALIDATED_HORIZON = 5

MODE_OFFICIAL = "official"
MODE_EXPLORATORY = "exploratory"

EXPLORATORY_LABEL = "exploratory — not gated, not published"


class NotOfficial(RuntimeError):
    """Raised when something exploratory is asked to behave as official."""


class NoRecipe(RuntimeError):
    """Raised when an official run is asked for a target with no registry recipe."""


@dataclass
class ExploratoryResult:
    """An exploratory forecast. Deliberately has no publish method.

    Carries its own banner text so a caller cannot render it without the caveat being available,
    and ``is_official`` is a hard ``False`` rather than a flag someone can set.
    """

    target: str
    model: str
    horizon: int
    forecasts: pd.DataFrame
    reasons: List[str] = field(default_factory=list)

    mode: str = MODE_EXPLORATORY

    @property
    def is_official(self) -> bool:
        return False

    @property
    def banner(self) -> str:
        why = " ".join(self.reasons)
        return (f"**{EXPLORATORY_LABEL}.** Model `{self.model}` chosen by hand at horizon "
                f"{self.horizon}. No gate was measured for this combination and nothing here is "
                f"published or scored. {why}").strip()


@dataclass
class OfficialResult:
    """An official forecast: a registry champion at the validated horizon."""

    target: str
    recipe_id: str
    model: str
    horizon: int
    forecasts: pd.DataFrame
    provenance: Dict
    gates: Dict

    mode: str = MODE_OFFICIAL

    @property
    def is_official(self) -> bool:
        return True


def targets_available(data_path: Path) -> List[str]:
    """Every column in the canonical file that could be a forecast target.

    All 41 are offered, not only the three with recipes — but see ``recipe_status``: a target
    without a recipe is refused in official mode rather than silently given someone else's.
    """
    df = pd.read_csv(data_path, nrows=1)
    drop = {"date", "is_weekend", "is_holiday"}
    return [c for c in df.columns if c not in drop]


def recipe_status(target: str) -> Dict:
    """Whether ``target`` has a registry recipe, and what to say if not."""
    from registry import load_registry

    for r in load_registry()["recipes"]:
        if r["target"] == target:
            return {"has_recipe": True, "recipe_id": r["id"],
                    "model": r["point_model"], "recipe": r}
    return {
        "has_recipe": False, "recipe_id": None, "model": None, "recipe": None,
        "explanation": (
            f"**{target} has no champion recipe, so no official forecast can be issued for it.** "
            f"A champion is a model, feature set and target scaling chosen on recorded evidence — "
            f"five training folds and a confirmation on 2024 — and only three budget lines have "
            f"been through that. Substituting another target's recipe would attach evidence to a "
            f"model it was never measured on. You can still forecast this line in exploratory "
            f"mode, where nothing is published and no gate is claimed."),
    }


def horizon_status(horizon: int) -> Dict:
    """Whether ``horizon`` is the validated one, and what to say if not."""
    if int(horizon) == VALIDATED_HORIZON:
        return {"validated": True, "explanation": ""}
    return {
        "validated": False,
        "explanation": (
            f"**Horizon {horizon} is exploratory.** Everything that makes a forecast trustworthy "
            f"here was measured at {VALIDATED_HORIZON} business days: the benchmark it is scored "
            f"against, the recipe selection, and every quality gate. At horizon {horizon} no "
            f"recipe was selected and no gate was measured, so the result is shown for "
            f"exploration and is not published."),
    }


def official_run(target: str, data_path: Path, horizon: int = VALIDATED_HORIZON):
    """Run the target's champion recipe. Raises rather than degrading.

    Refuses on two grounds, both of which would otherwise produce an official-looking forecast
    carrying credentials it never earned:
      * no registry recipe for the target;
      * a horizon other than the validated one.
    """
    st = recipe_status(target)
    if not st["has_recipe"]:
        raise NoRecipe(st["explanation"])
    hz = horizon_status(horizon)
    if not hz["validated"]:
        raise NotOfficial(hz["explanation"])

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_forecast import Champion, build_provenance, run_forward

    r = st["recipe"]
    champ = Champion(
        target=target, point_model=r["point_model"],
        fiscal_groups=tuple(r["feature_groups"]),
        exog_blocks=tuple(r.get("exog_blocks") or ()),
        recipe_id=r["id"], scaling=r["scaling"],
        transform=r.get("params", {}).get("target_transform", "raw"),
    )
    raw = pd.read_csv(data_path)
    fc = run_forward(raw, champ)
    prov = build_provenance(str(data_path), [champ])
    gates = {r["id"]: {"target": target,
                       "gates": r.get("dev_credentials", {}).get("gates", {}),
                       "status": r["status"],
                       "approved_by": r["approved_by"]}}
    return OfficialResult(target=target, recipe_id=r["id"], model=r["point_model"],
                          horizon=horizon, forecasts=fc, provenance=prov, gates=gates)


def exploratory_run(target: str, model: str, data_path: Path,
                    horizon: int = VALIDATED_HORIZON,
                    fiscal_groups: Optional[Sequence[str]] = None) -> ExploratoryResult:
    """Run any model on any target at any horizon. Never publishable."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_forecast import Champion, run_forward

    reasons: List[str] = []
    st = recipe_status(target)
    if not st["has_recipe"]:
        reasons.append(f"{target} has no champion recipe.")
    elif model != st["model"]:
        reasons.append(f"The champion for {target} is {st['model']}, not {model}.")
    hz = horizon_status(horizon)
    if not hz["validated"]:
        reasons.append(hz["explanation"])

    groups = tuple(fiscal_groups) if fiscal_groups else (
        tuple(st["recipe"]["feature_groups"]) if st["has_recipe"] else ())
    champ = Champion(target=target, point_model=model, fiscal_groups=groups,
                     recipe_id="", scaling="exploratory", transform="raw")
    fc = run_forward(pd.read_csv(data_path), champ,
                     horizons=tuple(range(1, int(horizon) + 1)))
    return ExploratoryResult(target=target, model=model, horizon=int(horizon),
                             forecasts=fc, reasons=reasons)


def publish_official(result, *, published_root: Optional[Path] = None,
                     forward_dir: Optional[Path] = None) -> Path:
    """Publish an OfficialResult. Refuses anything else.

    The type check is the boundary: an exploratory result cannot be published by passing a flag,
    because it is a different type with no publish path.
    """
    if not getattr(result, "is_official", False):
        raise NotOfficial(
            "Refusing to publish an exploratory forecast. Exploratory runs are not gated, carry "
            "no recipe credentials, and must never enter forecasts/published/ or the scorecard.")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_forecast import DEFAULT_OUT, write_artifacts
    from published_forecasts import publish

    src = Path(forward_dir or DEFAULT_OUT)
    write_artifacts(src, result.forecasts, result.provenance, result.gates)
    return publish(src, published_root=published_root)


def next_issue_date(published_root: Optional[Path] = None) -> str:
    """An issue date that does not collide with an existing one.

    Retention is immutable, so a same-day re-issue takes a suffixed date rather than overwriting
    the only record of what was previously said.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from published_forecasts import PUBLISHED_ROOT, list_published

    root = Path(published_root or PUBLISHED_ROOT)
    existing = {p.name for p in list_published(root)}
    base = pd.Timestamp.now(tz="UTC").date().isoformat()
    if base not in existing:
        return base
    n = 2
    while f"{base}-r{n}" in existing:
        n += 1
    return f"{base}-r{n}"
