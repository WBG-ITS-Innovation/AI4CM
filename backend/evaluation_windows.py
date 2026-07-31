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

from dataclasses import dataclass
from typing import Optional

import pandas as pd

TRAIN_START = "2015-01-05"
DEV_START = "2024-01-01"
TEST_START = "2025-01-01"


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
