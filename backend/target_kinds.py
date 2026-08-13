"""Is a target a stock (a level) or a flow? One definition, imported by all four families.

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
There were **four** implementations of this question, and they disagreed:

    b_ml_pipeline.is_stock              {"state budget balance", "balance", "t0"}
    c_dl_pipeline.is_stock              {"state budget balance", "balance", "t0"}
    e_quantile_daily_pipeline.is_stock  {"state budget balance", "balance", "t0"}
    run_a_stat._is_stock                {"state budget balance", "balance", "net", "stock"}

``e_quantile``'s docstring claimed it was "kept byte-identical to b_ml_pipeline.is_stock and
c_dl_pipeline.is_stock so the three families cannot disagree about what kind of series they
are modelling" -- which was true of those three and silent about the fourth.

The consequence of a disagreement is not cosmetic. The answer decides whether a family
models the **change from origin** and reconstructs the level, or models the level directly;
and whether a missing day is forward-filled (a stock persists) or filled with zero (no money
moved). For a column named ``t0``, three families would have modelled a delta and A_STAT a
level, and nothing would have flagged it.

--------------------------------------------------------------------------------
THE UNION, NOT A PICK
--------------------------------------------------------------------------------
``STOCK_ALIASES`` is the **union** of the four sets, deliberately. Choosing one family's set
would silently reclassify names another family currently treats as stock, and the failure is
asymmetric: treating a stock as a flow means zero-filling gaps in a level series and
modelling a level directly where a delta was intended, which is an order-of-magnitude error.
Treating a flow as a stock is wrong but visible. When the two mistakes are unequal, take the
union.

**This reclassifies nothing that exists today.** Only ``State budget balance`` appears in the
canonical file; ``balance``, ``t0``, ``net`` and ``stock`` are not columns in it.
``test_target_kinds.py`` asserts that emptiness, so if a client ever loads a column with one
of those names the test fails and the reclassification is a deliberate discovery rather than
a silent one.
"""
from __future__ import annotations

from typing import FrozenSet

#: Every name any family has ever treated as a stock (level) target, unioned.
#:
#: * ``state budget balance`` / ``balance`` — the live stock target and its short form.
#: * ``t0`` — B_ML, C_DL and E_QUANTILE's convention for an opening-balance column.
#: * ``net`` / ``stock`` — A_STAT's own additions.
STOCK_ALIASES: FrozenSet[str] = frozenset({
    "state budget balance",
    "balance",
    "t0",
    "net",
    "stock",
})

KIND_STOCK = "stock"
KIND_FLOW = "flow"


def is_stock(target: str) -> bool:
    """True when ``target`` names a level (stock) series rather than a flow.

    A stock persists between observations, so a gap is forward-filled and the pipelines model
    the change from origin and rebuild the level. A flow is zero when nothing moved, so a gap
    is filled with zero and the level is modelled directly.
    """
    return str(target).strip().lower() in STOCK_ALIASES


def target_kind(target: str) -> str:
    """``"stock"`` or ``"flow"`` — the same question, as a label for artifacts."""
    return KIND_STOCK if is_stock(target) else KIND_FLOW
