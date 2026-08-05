"""Exogenous Treasury-line features — workstream 5, leak-safe by construction.

Why this workstream exists, and what it is testing:

Workstreams 1 (L1 objectives) and 3 (fiscal calendar) both improved MAE and both **failed
to move the flow sentinel ratio** — Revenues 1.226, Expenditure 1.088, against the 1.50
threshold. So the univariate feature set, however well engineered, does not carry
information about the flow targets. Multivariate features are the last modelling lever.

The hypothesis is specific, not a fishing expedition. ``docs/DATA_SEMANTICS.md`` §1 measured
that the 72 business days on which ``Revenues`` prints negative are driven by netting in the
debt-operation lines:

    Increase in liabilities   negative on 64/72 of those days, correlation 0.971
    Domestic                  negative on 65/72,               correlation 0.969

Those are precisely the days the flow targets cannot predict, and they are not on a fixed
calendar day, so no ``dom``-style feature can reach them. ``DEBT_OPS_BLOCK`` is therefore
tested **on its own, before** any broad pool: if the mechanism is real, a four-column block
should show it, and a 100-column pool would only obscure whether it did.

--------------------------------------------------------------------------------
LEAK SAFETY
--------------------------------------------------------------------------------
Three properties, each enforced by a test rather than asserted here:

1. **Every feature is lagged by at least one step.** A feature at origin *t* uses exogenous
   values dated *t-1* and earlier. Same-day values of a flow line are not reliably known at
   the origin, and assuming otherwise is the classic multivariate leak.
2. **No statistic is fit.** These are plain lags and calendar-aligned lags — no scaling, no
   encoding, no imputation fit across rows. Nothing is fit, so there is nothing that could
   be fit on the wrong window. That is a deliberate design choice: it makes the leak
   argument structural rather than procedural.
3. **No feature selection.** No top-K, no correlation screen, no importance filter. Any such
   screen would have to be fit per fold to be honest, and this module does not fit anything.
   If the broad pool ever needs pruning, per-fold selection is the follow-up — and it must be
   inside the fold loop, never on the whole series.

The target's own column is always excluded from the exogenous set: it enters through the
pipeline's own lag recipe, and duplicating it here would double-count it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The hypothesised mechanism for the unpredictable days. Tested alone, first.
DEBT_OPS_BLOCK: Tuple[str, ...] = (
    "Increase in liabilities",
    "Decrease in liabilities",
    "Domestic",
    "Foreign",
)

#: Major tax components. These are the mechanical constituents of `Revenues`, so they are
#: the natural second candidate: a tax line that moves a day early is a leading indicator
#: of the aggregate.
TAX_BLOCK: Tuple[str, ...] = (
    "Taxes",
    "Income tax",
    "Profit tax",
    "Value added tax",
    "Excise duty",
    "Import tax",
)

#: The Revenues <-> Expenditure cross. Each is a candidate predictor of the other, and both
#: are components of the stock target.
CROSS_BLOCK: Tuple[str, ...] = ("Revenues", "Expenditure")

#: Major expenditure components, the Expenditure-side analogue of TAX_BLOCK.
SPEND_BLOCK: Tuple[str, ...] = (
    "Compensation of employees",
    "Goods and services",
    "Social security",
    "Subsidies",
    "Interest",
)

#: Columns that are calendar flags, not economic series -- already covered by the fiscal
#: calendar, and including them here would silently duplicate features.
NON_ECONOMIC: Tuple[str, ...] = ("is_weekend", "is_holiday")

BLOCKS: Dict[str, Tuple[str, ...]] = {
    "debt_ops": DEBT_OPS_BLOCK,
    "tax": TAX_BLOCK,
    "cross": CROSS_BLOCK,
    "spend": SPEND_BLOCK,
}

#: Lags applied to every exogenous column. 1 = yesterday; 5 = one business week, the
#: forecast horizon; 21 ~ one month, so a monthly-cycle line aligns with itself.
DEFAULT_EXOG_LAGS: Tuple[int, ...] = (1, 5, 21)


def resolve_block(name: str, available: Iterable[str], target: str) -> List[str]:
    """Columns of a named block that exist in the data, excluding the target.

    Missing columns are dropped silently rather than raising: the block definitions name
    the Treasury chart of accounts, and a dataset that lacks one line should still be
    usable. What must NOT happen silently is a block resolving to nothing, which
    ``build_exog_features`` treats as an error.
    """
    if name == "broad":
        cols = [c for c in available
                if c != target and c not in NON_ECONOMIC]
        return cols
    if name not in BLOCKS:
        raise KeyError(f"unknown exog block {name!r}; known: {sorted(BLOCKS)} + 'broad'")
    avail = set(available)
    return [c for c in BLOCKS[name] if c in avail and c != target]


def _aligned_prev_month(v: pd.Series, idx: pd.DatetimeIndex) -> List[float]:
    """Same business-day-of-month, previous month, from strictly earlier rows only.

    Mirrors the fiscal calendar's group D for exogenous columns: two 15ths are the
    comparable pair, and "21 business days ago" lands on a different bdom in most months.
    """
    bdom = idx.to_series().groupby([idx.year, idx.month]).cumcount().to_numpy() + 1
    lookup = {(y, m, b): i for i, (y, m, b) in enumerate(zip(idx.year, idx.month, bdom))}
    arr = v.to_numpy()
    out: List[float] = []
    for i, (y, m, b) in enumerate(zip(idx.year, idx.month, bdom)):
        py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
        j = lookup.get((py, pm, int(b)))
        out.append(arr[j] if (j is not None and j < i) else np.nan)
    return out


def build_exog_features(df: pd.DataFrame,
                        target: str,
                        blocks: Sequence[str],
                        index: pd.DatetimeIndex,
                        lags: Sequence[int] = DEFAULT_EXOG_LAGS,
                        aligned: bool = True,
                        date_col: str = "date") -> pd.DataFrame:
    """Lagged exogenous features for the requested blocks, on ``index``.

    Parameters mirror the information-set discipline: ``lags`` must all be >= 1, and the
    frame is reindexed onto the model's own business-day index before any shifting, so a
    shift of 1 means one business day rather than one calendar day.

    Flow columns are filled with 0.0 on absent business days and level columns
    forward-filled, matching the target-side convention in the pipelines. The distinction
    matters: a zero flow is an observation ("no transactions"), a zero level is not.
    """
    if any(int(L) < 1 for L in lags):
        raise ValueError(
            f"exogenous lags must all be >= 1 (got {sorted(lags)}); a lag of 0 would use "
            "the exogenous value dated at the forecast origin, which is not reliably "
            "known at that time"
        )

    work = df.copy()
    if date_col in work.columns:
        work = work.set_index(date_col)
    work.index = pd.DatetimeIndex(work.index).normalize()

    wanted: List[str] = []
    for b in blocks:
        cols = resolve_block(b, work.columns, target)
        if not cols:
            raise ValueError(f"exog block {b!r} resolved to no usable columns")
        wanted.extend(c for c in cols if c not in wanted)

    idx = pd.DatetimeIndex(index).normalize()
    out = pd.DataFrame(index=idx)
    for c in wanted:
        v = pd.to_numeric(work[c], errors="coerce").reindex(idx)
        # `State budget balance` is the only level among the candidate blocks; everything
        # else in the Treasury chart of accounts here is a daily flow.
        v = v.ffill() if c == "State budget balance" else v.fillna(0.0)
        safe = f"x_{c}".replace(" ", "_").replace("(", "").replace(")", "")
        for L in lags:
            out[f"{safe}_lag{int(L)}"] = v.shift(int(L))
        if aligned:
            out[f"{safe}_aligned_prev_month"] = _aligned_prev_month(v.shift(1), idx)
    return out


def exog_spec_hash(blocks: Sequence[str], lags: Sequence[int], aligned: bool) -> str:
    """Short hash of the exogenous specification, for the experiments log."""
    payload = {"blocks": sorted(blocks), "lags": sorted(int(L) for L in lags),
               "aligned": bool(aligned)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
