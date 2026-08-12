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

    #: ``estimator_store.FittedEstimator`` per fit, retained by ``publish_official``. Exploratory
    #: results deliberately have no equivalent: nothing unpublished needs re-deriving, and the
    #: blobs carry Treasury data.
    estimators: List = field(default_factory=list)

    mode: str = MODE_OFFICIAL

    @property
    def is_official(self) -> bool:
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TARGET ELIGIBILITY
#
# `targets_available` used to read `nrows=1` and return all 41 columns, so a column could be
# offered as a forecast target while being non-numeric, entirely null, or far too short to fit a
# single fold. Reading one row makes a length check impossible by construction.
#
# Eligibility is now measured, and **the reason a column is ineligible travels with the verdict**.
# That matters more than the verdict: a consumer told only "no" has to invent an explanation, and
# an invented explanation shown to a treasury is worse than a blank.
#
# The threshold is derived from the evaluation windows, not chosen here:
#     DEFAULT_MIN_TRAIN (1008, ~4y) + horizon (the embargo) + DEFAULT_EVAL_BLOCK (126, ~6m)
# i.e. enough history to train on a realistic base, keep a horizon-sized embargo, and still have
# one complete evaluation block left. Below that a number could be produced but not evaluated,
# which on this project is the same as not having one.
# ══════════════════════════════════════════════════════════════════════════════

#: Columns that are never forecast targets: the index itself and the two calendar flags.
NON_TARGET_COLUMNS = frozenset({"date", "is_weekend", "is_holiday"})

#: Machine-readable ineligibility codes. A consumer should branch on these, not on the prose.
INELIGIBLE_NOT_NUMERIC = "not_numeric"
INELIGIBLE_ALL_NULL = "all_null"
INELIGIBLE_INSUFFICIENT_HISTORY = "insufficient_history"
INELIGIBLE_UNUSABLE_DATE_INDEX = "unusable_date_index"


def min_history_rows(horizon: int = VALIDATED_HORIZON) -> int:
    """Non-null rows a target needs to be evaluable at ``horizon``.

    ``DEFAULT_MIN_TRAIN + horizon + DEFAULT_EVAL_BLOCK``. Imported from ``evaluation_windows`` so
    the two cannot drift: if the fold sizing changes, this moves with it.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluation_windows import DEFAULT_EVAL_BLOCK, DEFAULT_MIN_TRAIN

    return int(DEFAULT_MIN_TRAIN) + int(horizon) + int(DEFAULT_EVAL_BLOCK)


@dataclass
class Eligibility:
    """Whether one column can be forecast, and if not, why — in both forms."""

    target: str
    eligible: bool
    #: ``None`` when eligible; otherwise one of the ``INELIGIBLE_*`` codes.
    code: Optional[str] = None
    #: Plain language a consumer may quote verbatim. Never assemble your own.
    reason: Optional[str] = None
    n_usable: int = 0
    n_required: int = 0
    dtype: str = ""
    first_date: Optional[str] = None
    last_date: Optional[str] = None

    def as_dict(self) -> Dict:
        return {"target": self.target, "eligible": self.eligible, "code": self.code,
                "reason": self.reason, "n_usable": self.n_usable,
                "n_required": self.n_required, "dtype": self.dtype,
                "first_date": self.first_date, "last_date": self.last_date}


def date_index_status(data_path: Path, date_col: str = "date") -> Dict:
    """Whether the file's date index is parseable, unique and sortable.

    A file-level property: if it fails, **no** target in the file is eligible, because a
    horizon measured in business-day positions is meaningless over an index with holes or
    duplicates. Reported separately so a consumer can say "the file is unusable" rather than
    listing 41 identical column failures.
    """
    try:
        raw = pd.read_csv(data_path, usecols=[date_col])
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "reason": f"could not read {date_col!r} from the file: {exc}",
                "n": 0, "n_unique": 0, "n_unparseable": 0, "monotonic": None}

    parsed = pd.to_datetime(raw[date_col], errors="coerce")
    n = int(len(parsed))
    n_bad = int(parsed.isna().sum())
    n_unique = int(parsed.dropna().nunique())
    monotonic = bool(parsed.dropna().is_monotonic_increasing)

    if n == 0:
        reason = "the file has no rows, so there is no date index"
    elif n_bad:
        reason = (f"{n_bad} of {n} values in {date_col!r} could not be parsed as dates, so rows "
                  f"cannot be placed in time")
    elif n_unique != n - n_bad:
        dupes = (n - n_bad) - n_unique
        reason = (f"{dupes} duplicate date(s) in {date_col!r}; a horizon counted in index "
                  f"positions is ambiguous when one date appears twice")
    else:
        reason = None

    return {"ok": reason is None, "reason": reason, "n": n, "n_unique": n_unique,
            "n_unparseable": n_bad, "monotonic": monotonic}


def target_eligibility(data_path: Path, horizon: int = VALIDATED_HORIZON,
                       date_col: str = "date") -> Dict[str, Eligibility]:
    """Every candidate column with a verdict and the reason behind it.

    Checks, in the order a consumer would want them reported:
      1. the file's date index is usable at all (file-level; fails every column);
      2. the column is numeric;
      3. it has any non-null values;
      4. it has enough non-null history for one evaluable fold at ``horizon``.
    """
    required = min_history_rows(horizon)
    idx = date_index_status(data_path, date_col)

    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else None

    out: Dict[str, Eligibility] = {}
    for col in df.columns:
        if col in NON_TARGET_COLUMNS:
            continue
        s = df[col]
        e = Eligibility(target=col, eligible=False, n_required=required, dtype=str(s.dtype))

        if not idx["ok"]:
            e.code, e.reason = INELIGIBLE_UNUSABLE_DATE_INDEX, (
                f"the file's date index is unusable, so no column in it can be forecast: "
                f"{idx['reason']}")
            out[col] = e
            continue

        numeric = pd.to_numeric(s, errors="coerce")
        # A column that is not numeric at all -- as opposed to numeric with gaps -- is not a
        # series. Distinguished by whether coercion destroyed values that were there.
        if not pd.api.types.is_numeric_dtype(s) and numeric.notna().sum() == 0:
            e.code, e.reason = INELIGIBLE_NOT_NUMERIC, (
                f"{col!r} is {s.dtype} and none of its values parse as numbers, so it cannot be "
                f"forecast as a series")
            out[col] = e
            continue

        ok = numeric.notna()
        e.n_usable = int(ok.sum())
        if dates is not None and e.n_usable:
            d = dates[ok].dropna()
            if len(d):
                e.first_date, e.last_date = str(d.min().date()), str(d.max().date())

        if e.n_usable == 0:
            e.code, e.reason = INELIGIBLE_ALL_NULL, (
                f"{col!r} has no values at all in this file, so there is nothing to learn from")
            out[col] = e
            continue

        if e.n_usable < required:
            e.code, e.reason = INELIGIBLE_INSUFFICIENT_HISTORY, (
                f"{col!r} has {e.n_usable:,} usable observations; {required:,} are needed at "
                f"horizon {horizon} — about four years to train on, a {horizon}-day gap so no "
                f"training answer falls inside the evaluation, and one complete six-month block "
                f"left to score against. Below that a forecast could be produced but not "
                f"evaluated, and an unevaluated number is not a forecast")
            out[col] = e
            continue

        e.eligible = True
        out[col] = e
    return out


def targets_available(data_path: Path, horizon: int = VALIDATED_HORIZON,
                      include_ineligible: bool = False) -> List[str]:
    """Forecast targets in the canonical file.

    By default only the **eligible** ones: a column that cannot be evaluated should not be
    offered, because offering it invites a number nobody can score. Pass
    ``include_ineligible=True`` to get every candidate — a page that greys out the rejects and
    shows ``target_eligibility()``'s reason beside each is more useful than one that hides them.

    Eligibility is about the data. A target can be eligible here and still have no recipe (see
    ``recipe_status``) or no family that supports its kind (see ``family_supports_target``).
    """
    elig = target_eligibility(data_path, horizon=horizon)
    return [t for t, e in elig.items() if include_ineligible or e.eligible]


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
    sink: List = []
    fc = run_forward(raw, champ, estimator_sink=sink)
    prov = build_provenance(str(data_path), [champ])
    gates = {r["id"]: {"target": target,
                       "gates": r.get("dev_credentials", {}).get("gates", {}),
                       "status": r["status"],
                       "approved_by": r["approved_by"]}}
    return OfficialResult(target=target, recipe_id=r["id"], model=r["point_model"],
                          horizon=horizon, forecasts=fc, provenance=prov, gates=gates,
                          estimators=sink)


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
    dest = publish(src, published_root=published_root)

    # Retain what produced the numbers. The blobs are gitignored and the manifest is not -- see
    # estimator_store's module docstring for why this one published artifact is not tracked.
    if getattr(result, "estimators", None):
        from estimator_store import save_estimators
        origin = pd.DatetimeIndex(pd.to_datetime(result.forecasts["origin_date"]).unique())
        save_estimators(dest, result.estimators, keep_index=origin,
                        provenance=result.provenance)
    return dest


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


# ══════════════════════════════════════════════════════════════════════════════
# CLI — so a frontend can run a forecast WITHOUT importing the modelling stack
#
# The Streamlit venv has neither matplotlib nor sklearn, and it should not: it renders with
# Plotly and the models belong to the backend. Importing the pipeline from the page crashed on
# matplotlib and would then have crashed on sklearn, xgboost, lightgbm and catboost in turn.
# Deferring pyplot (backend/lazy_plot.py) was worth doing on its own merits but was never going to
# be sufficient.
#
# So the page dispatches here, the same pattern the Lab page and the model-pool lookup already use.
# One interpreter owns the models; the frontend reads JSON.
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Run a forecast in official or exploratory mode.")
    ap.add_argument("--mode", choices=[MODE_OFFICIAL, MODE_EXPLORATORY], required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=VALIDATED_HORIZON)
    ap.add_argument("--model", default="")
    ap.add_argument("--publish", action="store_true",
                    help="official mode only; refused otherwise by publish_official()")
    a = ap.parse_args()

    try:
        if a.mode == MODE_OFFICIAL:
            res = official_run(a.target, Path(a.data), horizon=a.horizon)
            out = {"ok": True, "mode": res.mode, "target": res.target,
                   "recipe_id": res.recipe_id, "model": res.model,
                   "approved_by": list(res.gates.values())[0].get("approved_by"),
                   "forecasts": _json.loads(res.forecasts.to_json(orient="records",
                                                                  date_format="iso"))}
            if a.publish:
                out["published_to"] = str(publish_official(res))
        else:
            res = exploratory_run(a.target, a.model, Path(a.data), horizon=a.horizon)
            out = {"ok": True, "mode": res.mode, "target": res.target, "model": res.model,
                   "banner": res.banner,
                   "forecasts": _json.loads(res.forecasts.to_json(orient="records",
                                                                 date_format="iso"))}
    except (NoRecipe, NotOfficial) as exc:
        out = {"ok": False, "refused": True, "reason": str(exc)}
    except Exception as exc:                        # pragma: no cover - surfaced to the page
        out = {"ok": False, "refused": False, "reason": f"{type(exc).__name__}: {exc}"}

    print(_json.dumps(out, default=str))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
