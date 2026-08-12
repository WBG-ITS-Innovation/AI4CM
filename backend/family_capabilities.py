"""What each model family can actually forecast, as data rather than as a note in a changelog.

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS, AND A CORRECTION
--------------------------------------------------------------------------------
`CHANGELOG.md` carried, under the Phase-0/1 **"Known gaps"** heading:

    E_QUANTILE has no stock-target path, so `State budget balance` cannot yet be forecast by the
    family that three workstreams depend on.

**That is no longer true, and this module was written after measuring it rather than after reading
it.** E_QUANTILE has a full stock path: ``_build_features`` adds ``y_lag_0``, models the change from
origin, and ``run_pipeline`` reconstructs the level from ``origin_value + delta``. Verified end to
end on 2026-08-12 against DEV 2024 (TEST untouched): 262 predictions, P50 MAE 168,565,455,
persistence MAE 242,653,025, **skill 30.53%**, coverage 65.6%, gate **FAILED** on coverage.

Its sibling bullet in the same block -- "E_QUANTILE is still on a calendar-day index" -- was also
fixed long ago by ``to_business_index``. The whole block describes a state the code has left.

So the honest reason `State budget balance` is not forecast by E_QUANTILE today is **not** that the
path is missing. It is that the path exists, has never been logged as a selection run, has no
registry recipe, and the one DEV run measured here fails the coverage gate. That is a materially
different statement: "unbuilt" invites building it, "built and failing its gate" invites fixing the
calibration.

A capability recorded only in prose drifts, because nothing fails when it goes stale. This module is
the machine-readable form, and ``test_family_capabilities.py`` re-measures the claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

BACKEND = Path(__file__).resolve().parent

#: The kinds of series a target can be.
KIND_FLOW = "flow"
KIND_STOCK = "stock"

#: How a family handles a stock (level) target.
STOCK_DELTA = "delta"          # models y(t+h) - y(t), reconstructs the level from origin_value
STOCK_LEVEL = "level"          # forecasts the level directly
STOCK_NONE = "unsupported"


@dataclass
class FamilyCapability:
    """What one family supports, and the evidence for it."""

    family: str
    supports_flow: bool
    supports_stock: bool
    #: ``STOCK_DELTA`` | ``STOCK_LEVEL`` | ``STOCK_NONE``
    stock_method: str
    #: How the claim was established. "measured" beats "read from a docstring".
    evidence: str
    #: Whether a *published* forecast exists for a stock target from this family.
    has_stock_recipe: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"family": self.family, "supports_flow": self.supports_flow,
                "supports_stock": self.supports_stock, "stock_method": self.stock_method,
                "evidence": self.evidence, "has_stock_recipe": self.has_stock_recipe,
                "notes": list(self.notes)}


#: The capability matrix. Every `supports_stock` value here was checked against the code, and
#: E_QUANTILE's was checked by running the family end to end (see the module docstring).
FAMILY_CAPABILITIES: Dict[str, FamilyCapability] = {
    "B_ML": FamilyCapability(
        family="B_ML", supports_flow=True, supports_stock=True, stock_method=STOCK_DELTA,
        evidence="is_stock() drives delta modelling; the three live recipes include the stock "
                 "target and it is the only one whose forecast is published",
        has_stock_recipe=True,
        notes=["The stock target's champion (HistGBDT_L1) is the only recipe that passes its "
               "gates, so it is the only published forecast of the three."]),
    "C_DL": FamilyCapability(
        family="C_DL", supports_flow=True, supports_stock=True, stock_method=STOCK_DELTA,
        evidence="is_stock() drives delta modelling and disables log1p scaling for stock targets "
                 "(c_dl_pipeline: mode = 'none' if stock else 'log1p_std')",
        has_stock_recipe=False,
        notes=["Parked since Phase 2; no recipe promotes a C_DL model."]),
    "E_QUANTILE": FamilyCapability(
        family="E_QUANTILE", supports_flow=True, supports_stock=True, stock_method=STOCK_DELTA,
        evidence="MEASURED end to end on 2026-08-12 against DEV 2024: 262 predictions, "
                 "skill 30.53%, coverage 65.6%, gate FAILED on coverage. The CHANGELOG's "
                 "'no stock-target path' entry is stale -- see the module docstring",
        has_stock_recipe=False,
        notes=["The path exists; what blocks a published stock forecast here is the coverage "
               "gate (65.6% against a required [70%, 90%]), not a missing implementation.",
               "No logged selection run on the stock target: 0 rows in experiments/log.csv."]),
    "A_STAT": FamilyCapability(
        family="A_STAT", supports_flow=True, supports_stock=True, stock_method=STOCK_LEVEL,
        evidence="_is_stock() drives resampling only (ffill vs fillna(0); last vs sum). The "
                 "family forecasts the LEVEL directly and never models a delta",
        has_stock_recipe=False,
        notes=["Different method from the other three, and legitimately so for a statistical "
               "model on a level series -- but it means a stock forecast from A_STAT is not "
               "comparable to one from B_ML without saying which was modelled how.",
               "Writes no shift-diagnostic fields, so it has no effective persistence-mimicry "
               "check at h=5 (reports/gate_audit.md §2.6)."]),
}


def target_kind(target: str) -> str:
    """``KIND_STOCK`` or ``KIND_FLOW``, per the shared ``is_stock`` definition."""
    import sys
    sys.path.insert(0, str(BACKEND))
    from b_ml_pipeline import is_stock

    return KIND_STOCK if is_stock(target) else KIND_FLOW


def family_supports_target(family: str, target: str) -> Dict:
    """Whether ``family`` can forecast ``target``, with the reason either way.

    "Supported" means the family has a code path for this kind of series. It does **not** mean a
    forecast from it is publishable: that additionally needs a registry recipe and passing gates.
    Both are reported so a consumer cannot read one as the other.
    """
    fam = str(family).strip().upper()
    cap = FAMILY_CAPABILITIES.get(fam)
    if cap is None:
        return {"family": fam, "target": target, "supported": False,
                "code": "unknown_family",
                "reason": (f"{family!r} is not a model family in this project. Known families: "
                           f"{', '.join(sorted(FAMILY_CAPABILITIES))}.")}

    kind = target_kind(target)
    supported = cap.supports_stock if kind == KIND_STOCK else cap.supports_flow
    out = {"family": fam, "target": target, "target_kind": kind, "supported": bool(supported),
           "stock_method": cap.stock_method if kind == KIND_STOCK else None,
           "publishable": bool(supported and cap.has_stock_recipe) if kind == KIND_STOCK
           else None,
           "evidence": cap.evidence, "notes": list(cap.notes)}

    if not supported:
        out["code"] = "kind_unsupported"
        out["reason"] = (f"{fam} has no code path for a {kind} target, so {target!r} cannot be "
                         f"forecast by it. {cap.evidence}")
    elif kind == KIND_STOCK and not cap.has_stock_recipe:
        out["code"] = "supported_but_not_published"
        out["reason"] = (
            f"{fam} can model {target!r} (a {kind} target, handled as "
            f"{cap.stock_method}), but no registry recipe promotes a {fam} model for it, so "
            f"nothing from this family is published for that target. Supported is not the same "
            f"as approved.")
    else:
        out["code"] = None
        out["reason"] = None
    return out


def stock_alias_divergence() -> Dict:
    """Where the four families disagree about which names are stock targets.

    ``e_quantile_daily_pipeline.is_stock`` documents itself as "byte-identical to
    b_ml_pipeline.is_stock and c_dl_pipeline.is_stock", which is true of those three.
    ``run_a_stat._is_stock`` is a fourth implementation with a **different alias set**, so for a
    column named ``t0`` three families would model a delta and A_STAT would model a level.

    Currently **latent**: none of the disputed names is a column in the canonical file. It becomes
    live the moment one is added or a column is renamed, which is exactly when nobody would think
    to check. Reported as data so a test can hold it.
    """
    import sys
    sys.path.insert(0, str(BACKEND))
    from b_ml_pipeline import is_stock as bml
    from c_dl_pipeline import is_stock as cdl
    from e_quantile_daily_pipeline import is_stock as eq
    from run_a_stat import _is_stock as astat

    impls = {"B_ML": bml, "C_DL": cdl, "E_QUANTILE": eq, "A_STAT": astat}
    names = sorted({"state budget balance", "balance", "t0", "net", "stock"})
    disputed: Dict[str, Dict[str, bool]] = {}
    for n in names:
        verdicts = {f: bool(fn(n)) for f, fn in impls.items()}
        if len(set(verdicts.values())) > 1:
            disputed[n] = verdicts
    return {
        "agree": not disputed,
        "disputed": disputed,
        "n_implementations": len(impls),
        "note": ("A_STAT uses a separate alias set ({'net', 'stock'} instead of {'t0'}). Latent "
                 "while none of the disputed names is a column in the canonical file."),
    }


def capability_matrix() -> Dict:
    """The whole matrix as plain data, for an artifact or a page."""
    return {
        "families": {f: c.as_dict() for f, c in sorted(FAMILY_CAPABILITIES.items())},
        "stock_alias_divergence": stock_alias_divergence(),
        "note": ("`supports_stock` means a code path exists. Publishing additionally requires a "
                 "registry recipe and passing gates -- see family_supports_target()."),
    }
