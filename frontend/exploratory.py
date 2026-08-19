"""What an exploratory run is, in one place.

Why this module exists
----------------------
The Lab launches whatever the user configures, and until now it sent no evaluation
bound at all. Every pipeline treats an absent bound the same way: fold forward to the
last year present in the file. On the canonical dataset that makes the final fold
``train <= 2024-12-31 / evaluate 2025-01-01 .. 2025-08-06``, which is the sealed
holdout end to end. ``b_ml_pipeline`` then crowns a best model from those rows,
``assert_selection_free`` refuses, and the user sees a traceback.

The guard was right. The caller was wrong: a run launched from the UI to satisfy
curiosity is a *choice-shaped* act (the person is comparing models by eye), so it must
be measured on data that choices are allowed to touch. That is train and dev.

So this module answers one question -- "what configuration does an exploratory run
get?" -- and every UI path that launches a backend run asks it rather than assembling
overrides of its own. The dates come from ``backend/evaluation_windows`` rather than
being repeated here, because a second copy of the split is a second thing to get wrong.

What this module deliberately does NOT do
-----------------------------------------
It does not weaken, wrap, or bypass any guard. ``assert_selection_free`` still refuses
report-only data exactly as before; this simply stops handing it report-only data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pandas as pd  # noqa: E402

from evaluation_windows import DEV, TEST, window_for  # noqa: E402

#: Inclusive last date an exploratory run may evaluate on. Everything after it is the
#: sealed holdout (TEST) or data that arrived after sealing (LIVE), and neither may
#: inform a choice.
EXPLORATORY_EVAL_END: str = DEV.end

#: The sentence shown next to every exploratory result. Plain words on purpose: the
#: reader should not need to know what "dev" or "holdout" mean to understand the limit.
EXPLORATORY_NOTE: str = (
    "Exploratory results are measured on train and dev data, never on the sealed "
    "window. That keeps the final holdout unspent, so the official numbers stay honest."
)

#: Shown where the run profile is chosen, because the profile no longer shortens data.
DEMO_CLIP_NOTE: str = (
    "The Demo profile runs fewer folds. It no longer shortens the data, because the "
    "most recent months sit inside the sealed window and exploratory runs never read it."
)

#: Families the Lab can launch. C_DL is listed separately below because its runners
#: default the evaluation start to the holdout, so an exploratory run has to say "no".
FAMILIES = ("A_STAT", "B_ML", "C_DL", "E_QUANTILE")


def exploratory_overrides(family: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return ``overrides`` bound to the region a choice may be made on.

    Three things are forced, and each one closes a way the previous behaviour reached
    report-only data:

    * ``eval_end`` is pinned to the last dev date. Without it every family folds to the
      end of the file.
    * ``eval_start`` is set to ``None`` explicitly rather than omitted. C_DL's runners
      read ``ov.get("eval_start", TEST_START)``, so *omitting* the key selects the
      holdout; only an explicit null overrides it.
    * ``demo_clip_months`` is cleared. It clipped the series to the last N months of the
      file, which on this dataset is almost entirely holdout and post-seal data, so a
      bounded evaluation over a clipped series has nothing left to evaluate.

    Everything else the caller passed survives untouched.
    """
    fam = str(family).strip().upper()
    if fam not in FAMILIES:
        raise ValueError(
            f"Unknown family {family!r}. Expected one of {', '.join(FAMILIES)}."
        )
    out: Dict[str, Any] = dict(overrides or {})
    out["eval_start"] = None
    out["eval_end"] = EXPLORATORY_EVAL_END
    out["demo_clip_months"] = None
    return out


def is_exploratory_safe(overrides: Optional[Dict[str, Any]]) -> bool:
    """True when ``overrides`` cannot cause evaluation past the last selectable date.

    Used after the advanced JSON box is parsed. The box is an escape hatch by design,
    and someone can type a later ``eval_end`` into it; this is how the Lab notices
    rather than discovering it from a traceback several minutes later.
    """
    ov = dict(overrides or {})
    end = ov.get("eval_end")
    if end in (None, ""):
        return False
    try:
        end_ts = pd.Timestamp(end)
    except (ValueError, TypeError):
        return False
    if end_ts > pd.Timestamp(EXPLORATORY_EVAL_END):
        return False
    if ov.get("demo_clip_months") not in (None, 0):
        return False
    start = ov.get("eval_start")
    if start not in (None, ""):
        try:
            if pd.Timestamp(start) > end_ts:
                return False
        except (ValueError, TypeError):
            return False
    return True


def contains_report_only_dates(config: Dict[str, Any]) -> bool:
    """True if any date-looking value in ``config`` falls in TEST or LIVE.

    A blunt check over the whole payload, which is the point: it does not need to know
    which key a date arrived under, so a future key that carries a date is covered the
    day it is added rather than the day someone remembers to update a list.
    """
    for value in dict(config or {}).values():
        if not isinstance(value, str) or len(value) < 8:
            continue
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError):
            continue
        if pd.isna(ts):
            continue
        if window_for(ts) not in ("train", "dev"):
            return True
    return False


def selectable_index(index) -> pd.DatetimeIndex:
    """The part of a date index an exploratory run is allowed to evaluate on."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index), errors="coerce")).dropna()
    return idx[idx <= pd.Timestamp(EXPLORATORY_EVAL_END)]


def check_can_run(index, horizon: int, min_train_years: int = 0) -> Optional[str]:
    """Return a plain-language reason this configuration cannot run, or ``None``.

    Called before launching, so an impossible configuration produces one readable
    sentence instead of a backend traceback. The three refusals below are the three
    ways the bound can leave nothing to measure, and they are checked in the order a
    reader would ask about them: is there any usable data, is there enough of it, and
    is there enough history in front of it.
    """
    usable = selectable_index(index)
    if len(usable) == 0:
        return (
            "This data file has no rows on or before "
            f"{EXPLORATORY_EVAL_END}, so there is nothing an exploratory run may "
            "measure on. Exploratory runs never read the sealed window. Load a file "
            "that includes earlier history, or publish an official forecast instead."
        )

    horizon = int(horizon)
    if len(usable) <= horizon + 1:
        return (
            f"This data file has only {len(usable)} rows on or before "
            f"{EXPLORATORY_EVAL_END}, which is not enough to forecast "
            f"{horizon} steps ahead and still have something to score. Choose a "
            "shorter horizon, or load a file with more history."
        )

    years = sorted({int(y) for y in usable.year})
    if int(min_train_years) > 0 and len(years) <= int(min_train_years):
        return (
            f"This data file covers {len(years)} year(s) on or before "
            f"{EXPLORATORY_EVAL_END}, and the chosen profile asks for "
            f"{int(min_train_years)} year(s) of training history before the first "
            "evaluation. Choose the Demo profile, or load a file with more history."
        )
    return None


def describe_bound() -> str:
    """One sentence naming the exact dates, for the run configuration panel."""
    return (
        f"Evaluation is bounded to {DEV.start[:4]} and earlier, ending "
        f"{EXPLORATORY_EVAL_END}. The sealed window opens {TEST.start} and stays "
        "closed to exploratory runs."
    )
