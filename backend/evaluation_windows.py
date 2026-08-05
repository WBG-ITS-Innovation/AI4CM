"""Evaluation windows — one definition, used by every family and every tuner.

Why this module exists
----------------------
The audit's most expensive finding was that each family measured itself
against its own baseline on its own window, so the numbers were not
comparable and "skill" meant something different in each report.  Windows
have the same failure mode: if B_ML tunes on 2025 while A_STAT tunes on
2024, no cross-family comparison is meaningful, and if *everything* tunes
on the same window we later report as the final result, that result is no
longer out-of-sample in any useful sense.

The three-way split
-------------------
    TRAIN   2015-01-05 .. 2023-12-31    model fitting
    DEV     2024-01-01 .. 2024-12-31    all tuning, feature selection,
                                        threshold choices, model comparison
    TEST    2025-01-01 .. (data end)    LOCKED — final reporting only

Discipline: every decision that could be made differently (hyperparameters,
which features to keep, which models to ship, where to set a gate) is made
by looking at DEV only.  TEST is run at the end of a milestone to report
what would have happened, and is never used to choose anything.  Each time
TEST is consulted to make a choice, it stops being a clean holdout, so the
number of such consultations should stay at zero.

Note on rolling origins: TRAIN is a floor, not a cap.  When evaluating on
DEV or TEST the pipelines roll their origins forward, so a fold predicting
2025-06-01 legitimately trains on everything up to 2025-05-31.  What never
happens is training on data at or after the origin it is predicting from.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

TRAIN_START = "2015-01-05"
DEV_START = "2024-01-01"
TEST_START = "2025-01-01"

# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------
# Until Phase 2 this module was documentation: it stated the discipline and
# nothing imported it, so nothing was enforced (review §2.4).  The helpers below
# make the split checkable in code, and make reading TEST an explicit, logged act
# rather than something that can happen by accident.

#: Env var that must be set to read the TEST window.  Deliberately awkward.
TEST_ACCESS_ENV = "AI4CM_ALLOW_TEST_READ"

#: Every TEST read is appended here, so "how many times did we consult the
#: holdout" has a factual answer instead of a recollection.
TEST_ACCESS_LOG = Path(__file__).resolve().parent.parent / "experiments" / "test_access.log"


class TestWindowAccessError(RuntimeError):
    """Raised when code touches the TEST window without explicit permission."""


def is_test_read_allowed() -> bool:
    return os.environ.get(TEST_ACCESS_ENV, "").strip().lower() in {"1", "true", "yes"}


def require_test_access(reason: str, caller: Optional[str] = None) -> None:
    """Gate and loudly log any read of the locked TEST window.

    Phase 2's rule is that TEST is untouched until explicitly released.  A quiet
    boolean would be too easy to flip, so this raises by default and, when
    permitted, writes a banner to stderr *and* appends to
    ``experiments/test_access.log``.  The count of consultations should stay at
    zero during model search; if it does not, the log says exactly when and why.
    """
    if not reason or not reason.strip():
        raise ValueError("A TEST read requires a stated reason.")

    if not is_test_read_allowed():
        raise TestWindowAccessError(
            f"Refusing to read the TEST window ({TEST_START} onward): {reason}. "
            f"TEST is the locked holdout and consulting it during search invalidates "
            f"it as a holdout. If this read is genuinely intended, set "
            f"{TEST_ACCESS_ENV}=1 for that one command -- it will be logged to "
            f"{TEST_ACCESS_LOG}."
        )

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "caller": caller or "unspecified",
        "argv": " ".join(sys.argv[:4]),
    }
    banner = "!" * 78
    print(f"\n{banner}\n!! TEST WINDOW READ ({TEST_START} onward): {reason}\n"
          f"!! logged to {TEST_ACCESS_LOG}\n{banner}\n", file=sys.stderr, flush=True)
    try:
        TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TEST_ACCESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass   # never let logging failure mask the read itself


def window_of(ts) -> str:
    """Alias of :func:`window_for`, kept for readability at call sites."""
    return window_for(ts)


def restrict(obj, window: str):
    """Return only the rows of a Series/DataFrame/DatetimeIndex inside ``window``.

    ``window`` is 'train', 'dev', 'test', or 'train+dev' (the searchable region).
    Reading 'test' goes through :func:`require_test_access`.
    """
    window = window.strip().lower()
    if window == "test":
        require_test_access("restrict(..., 'test')", caller="evaluation_windows.restrict")

    idx = obj if isinstance(obj, pd.DatetimeIndex) else pd.DatetimeIndex(obj.index)
    names = {"train", "dev", "test"} if window == "train+dev" else {window}
    if window == "train+dev":
        names = {"train", "dev"}
    keep = np.array([window_for(t) in names for t in idx], dtype=bool)

    if isinstance(obj, pd.DatetimeIndex):
        return obj[keep]
    return obj.loc[keep]


def assert_within(dates: Iterable, window: str, context: str) -> None:
    """Raise if any date falls outside ``window``.

    Used to assert that a search or fit never saw a window it should not have.
    ``context`` names the call site so the error is actionable.
    """
    allowed = {"train", "dev"} if window.strip().lower() == "train+dev" else {window.strip().lower()}
    offenders = sorted({window_for(t) for t in pd.DatetimeIndex(list(dates))} - allowed)
    if offenders:
        raise TestWindowAccessError(
            f"{context}: touched window(s) {offenders} but only {sorted(allowed)} "
            f"is permitted here. Split is {describe_split()}."
        )


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold. Dates are origins/targets, not row positions."""
    fold_id: int
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_end": str(self.train_end.date()),
            "eval_start": str(self.eval_start.date()),
            "eval_end": str(self.eval_end.date()),
        }


#: Default rolling-origin geometry for search, in business-day rows.
#: A 6-month evaluation block over a >=4-year training base: long enough that a
#: fold's MAE is not dominated by one month-end, short enough that five folds fit
#: inside TRAIN while every fold still trains on a realistic amount of history.
#: Sizing the block as usable//n_folds instead gives ~1.4-year blocks and leaves
#: the earliest fold training on two years, which makes the search signal noisy.
DEFAULT_EVAL_BLOCK = 126     # ~6 months of business days
DEFAULT_MIN_TRAIN = 1008     # ~4 years of business days


def rolling_origin_folds(
    index: pd.DatetimeIndex,
    horizon: int,
    n_folds: int = 5,
    min_train_rows: int = DEFAULT_MIN_TRAIN,
    window: str = "train",
    eval_block: Optional[int] = DEFAULT_EVAL_BLOCK,
) -> List[Fold]:
    """Rolling-origin folds carved out of ONE window, for search and selection.

    All hyperparameter search happens on folds inside TRAIN, so DEV stays a
    confirmation set and TEST is never involved.  Each fold's evaluation block is
    separated from its training data by ``horizon`` rows, so no training label is
    dated at or after the first evaluation origin -- the embargo the review found
    missing in E_QUANTILE (§2.1) and in B_ML's own validation split (§2.2).

    Blocks are contiguous, equal-sized and non-overlapping, taken from the end of
    the window backwards so the most recent regime is always represented.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if window.strip().lower() == "test":
        require_test_access("rolling_origin_folds(window='test')",
                            caller="evaluation_windows.rolling_origin_folds")

    idx = restrict(pd.DatetimeIndex(index), window)
    n = len(idx)
    if n == 0:
        raise ValueError(f"No dates fall in window '{window}'.")

    usable = n - min_train_rows - horizon * n_folds
    if usable <= 0:
        raise ValueError(
            f"Window '{window}' has {n} rows: too few for {n_folds} folds at "
            f"horizon {horizon} with min_train_rows={min_train_rows}."
        )
    # Fixed block by default; fall back to filling the window when asked.
    block = max(horizon, eval_block if eval_block else usable // n_folds)
    if block * n_folds > usable:
        raise ValueError(
            f"Window '{window}' has {n} rows: {n_folds} folds of {block} rows plus "
            f"min_train_rows={min_train_rows} and {horizon}-row embargoes do not fit. "
            f"Reduce n_folds or eval_block."
        )

    folds: List[Fold] = []
    end = n
    for k in range(n_folds):
        eval_end_pos = end - 1
        eval_start_pos = end - block
        train_end_pos = eval_start_pos - horizon
        if train_end_pos < min_train_rows:
            break
        folds.append(Fold(
            fold_id=n_folds - k,
            train_end=idx[train_end_pos],
            eval_start=idx[eval_start_pos],
            eval_end=idx[eval_end_pos],
        ))
        end = eval_start_pos
    folds.reverse()
    for i, f in enumerate(folds, 1):
        object.__setattr__(f, "fold_id", i)
    return folds


def seasonal_naive_scale(
    series: pd.Series,
    season: int = 5,
    window: str = "train",
) -> float:
    """MASE denominator: in-sample one-step seasonal-naive MAE on TRAIN only.

    The standard Hyndman scaling.  Computed on TRAIN alone so it never carries DEV
    or TEST information, and once per target so DEV numbers are comparable across
    targets whose magnitudes differ by orders of magnitude.

    ``season=5`` is one business week on this index.
    """
    s = restrict(series.dropna(), window)
    if len(s) <= season:
        raise ValueError(f"Need more than {season} points in '{window}' to scale MASE.")
    diffs = np.abs(s.to_numpy()[season:] - s.to_numpy()[:-season])
    scale = float(np.mean(diffs))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Degenerate MASE scale ({scale}) for '{series.name}'.")
    return scale


def mase(y_true, y_pred, scale: float) -> float:
    """Mean absolute scaled error against a precomputed TRAIN-only scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0 or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[ok] - y_pred[ok])) / scale)


@dataclass(frozen=True)
class Window:
    name: str
    start: str
    end: Optional[str]      # None = open-ended (to end of data)
    purpose: str

    def contains(self, ts) -> bool:
        ts = pd.Timestamp(ts)
        if ts < pd.Timestamp(self.start):
            return False
        return True if self.end is None else ts <= pd.Timestamp(self.end)


TRAIN = Window("train", TRAIN_START, "2023-12-31", "model fitting")
DEV = Window("dev", DEV_START, "2024-12-31",
             "tuning, feature selection, thresholds, model comparison")
TEST = Window("test", TEST_START, None,
              "LOCKED holdout — final reporting only, never used to choose")

WINDOWS = (TRAIN, DEV, TEST)


def window_for(ts) -> str:
    """Return the window name a timestamp falls in ('train'/'dev'/'test')."""
    ts = pd.Timestamp(ts)
    if ts < pd.Timestamp(DEV_START):
        return "train"
    if ts < pd.Timestamp(TEST_START):
        return "dev"
    return "test"


def eval_start_for(purpose: str) -> str:
    """Return the eval_start date a pipeline should use.

    purpose="tuning"  -> DEV_START   (choose things here)
    purpose="report"  -> TEST_START  (report from here, choose nothing)
    """
    p = purpose.strip().lower()
    if p in ("tuning", "tune", "dev", "development"):
        return DEV_START
    if p in ("report", "reporting", "final", "test", "holdout"):
        return TEST_START
    raise ValueError(
        f"purpose must be 'tuning' or 'report', got {purpose!r} — "
        "being explicit here is what keeps the holdout clean."
    )


def describe_split(data_end: Optional[str] = None) -> str:
    """One-paragraph description for reports and slides."""
    test_end = data_end or "end of data"
    return (
        f"Train {TRAIN.start}..{TRAIN.end} · "
        f"Dev {DEV.start}..{DEV.end} (tuning) · "
        f"Test {TEST.start}..{test_end} (locked holdout, reporting only)"
    )
