"""The Treasury's current planning method, for a chart, from the one construction that defines it.

What this is for
----------------
Two pages draw a comparison line labelled "Ops baseline": the Dashboard and the Lab. It is the
method the Treasury actually plans with today, so it is the line that answers "is this model
better than what we already do". Getting it wrong is worse than omitting it, because a reader
compares against it and believes the answer.

Both pages used to read ``<target>_ops_baseline_daily.csv`` out of the run folder. That was wrong
in two different ways at once, and this module exists to replace both readings.

Defect 1: a baseline was invented for the stock target
------------------------------------------------------
The method takes a flow's **annual total** and splits it up. A balance is a level, so it has no
annual total and the method does not apply. ``backend/ops_baseline.py`` says so and returns
nothing rather than inventing something (``REASON_STOCK``).

The Dashboard filled the gap anyway: with no CSV present it fell through to a day-of-week mean of
the actual figures and drew *that* under the name "Ops baseline". Measured on
``State budget balance`` it has five distinct values spanning 1.38% of its own mean, so on an axis
reaching 2.87B it reads as a straight horizontal line. A reader would take it for the Treasury's
planning method. It is the average of the answers.

Defect 2: for flows the file was empty, and empty became zero
-------------------------------------------------------------
Every B_ML run writes that CSV with no numbers in it: 0 non-NaN of 2763 rows, in all 14 run
folders present. An all-NaN series is not an *empty* series, so it passed the caller's
``not ops.empty`` guard and was plotted; and ``.resample(...).sum()`` maps all-NaN to ``0.0``, so
a weekly or monthly view drew a flat line along zero.

The cause is in the writer, ``b_ml_pipeline.ops_monthly_baseline``, not here, and is recorded for
a scoped fix rather than patched from the frontend. Scoring never read those files, so no
published figure was affected.

What this module does instead
-----------------------------
It builds the comparator from ``backend/ops_baseline.py``: ``vintage_cache`` and
``ops_prediction_for``, which is the same pair the leaderboard's ``skill_vs_ops_pct`` is computed
from. So the line on the chart and the skill figure in the table cannot disagree, and
``test_ops_baseline_view.py`` holds them to it by recomputing a run's stored ``ops_MAE`` from this
series and requiring an exact match.

Imported in-process rather than dispatched to a subprocess. That was the first design, on the
assumption that ``ops_baseline`` needs the modelling stack. It does not: its one
``c_dl_pipeline`` import is lazy and sits inside ``ops_daily_series``, which nothing here calls.
The whole path runs on pandas alone, and importing pure-pandas backend modules from a page is
already how the Scorecard, Forecast and Models pages work.

Why it reads the run's own ``predictions_long.csv``. The value depends on the forecast **origin**,
not only on the target date: a January date forecast from late December has an origin at which the
previous year was not yet complete, so the method in force was the older one. That is the
``vintage`` in ``vintage_cache``, and the origins are recorded per row in the run's predictions.

On the sealed window
--------------------
Plotting this comparator over holdout dates is a read of the sealed window, so it is recorded
through ``log_sealed_window_read`` with ``PURPOSE_REPORT``, exactly as A_STAT, B_ML, C_DL and
E_QUANTILE do on their reporting paths. Without that call the frontend would be the one path that
reads the holdout and says nothing. In practice it is usually a no-op: the sealed window starts
2025-01-01 and UI-launched runs are bound to train and dev.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"

#: The default Treasury table, and the only input the method needs.
DEFAULT_DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ══════════════════════════════════════════════════════════════════════════════
# WHY THE READER IS TOLD, RATHER THAN SHOWN A GAP
#
# Every path that cannot produce the line returns a reason written for a reader, not a code. A
# chart that silently loses its comparator invites the reader to assume the model had nothing to
# beat. These are the reasons there are.
# ══════════════════════════════════════════════════════════════════════════════

#: Plain-words form of ``ops_baseline.REASON_STOCK``. The wording follows that module's reasoning
#: rather than restating its sentence, because this one is read by a non-specialist.
def _t(text: str) -> str:
    """The selected language's version of ``text``, or ``text`` itself.

    These five strings reached the reader untranslated: they are module constants returned to
    a caller that renders them, so nothing on the page had a chance to translate them and the
    Georgian coverage figure could not see them either.

    Translated here, at the point of return, rather than at each call site. Two pages render
    them and both would have had to remember; one place cannot be forgotten. Imported inside
    the function, so this module still imports without streamlit or the i18n layer present.
    """
    try:
        from i18n import t

        return t(text)
    except Exception:                              # noqa: BLE001 - copy, not logic
        return text


REASON_STOCK = (
    "There is no Ops baseline for a balance. The Treasury's planning method works by taking a "
    "year's total and splitting it across the months, and a balance is a level on a day rather "
    "than something you can total up over a year. So the method does not apply here, and nothing "
    "is drawn in its place."
)

REASON_NO_BACKEND = (
    "The Ops baseline could not be worked out, because the pipeline code under `backend/` was not "
    "found. Everything else on this chart is unaffected: it is read from files the run already "
    "wrote."
)

REASON_NO_PREDICTIONS = (
    "The Ops baseline needs the run's predictions file to know which day each forecast was made "
    "from, and that file was not found in this run folder."
)

REASON_NO_WINDOW = (
    "The Ops baseline has no value for the dates in this run. The method needs three complete "
    "calendar years of history before the day a forecast was made from."
)

#: Shown under the chart whenever the line IS drawn, because the shape looks like a mistake.
#:
#: The "turn of the year" sentence is not a detail. Measured on a real run, every January holds
#: TWO values rather than one: the first few days were forecast from an origin in December, when
#: the year had not finished, so they are compared against the older figure the Treasury actually
#: had in hand. Claiming one value per month without that would be wrong five times over in this
#: run alone.
CAPTION_WHY_FLAT = (
    "**Why the Ops baseline looks flat.** It is the Treasury's current planning method, not a "
    "forecast. It averages the last three complete calendar years, splits that total across the "
    "months, then spreads each month evenly across its working days. So it holds one value for a "
    "whole month and steps to a new value at the start of the next. At the turn of the year it "
    "changes only once the new year's figures are complete, so the first few days of January are "
    "compared against the figure that was actually available then. It cannot react to news, and "
    "it is not meant to."
)


# ══════════════════════════════════════════════════════════════════════════════
# Is this target a level or a flow?
# ══════════════════════════════════════════════════════════════════════════════

#: Every name the backend treats as a level (stock) rather than a flow.
#:
#: Copied from ``backend/target_kinds.STOCK_ALIASES`` so the stock case can be answered before
#: importing anything. ``test_ops_baseline_view.py`` asserts the two sets are equal, so the
#: duplication cannot drift; that module explains why the set is a union of four families' sets
#: and why getting it wrong is an order-of-magnitude error rather than a cosmetic one.
#:
#: An earlier version of this checked ``"balance" in target``, which is a different question: it
#: would have missed ``t0``, ``net`` and ``stock``, and would have invented a level out of any
#: column whose name merely contained the word "balance".
STOCK_ALIASES = frozenset({"state budget balance", "balance", "t0", "net", "stock"})


def _is_stock(target: str) -> bool:
    """Whether ``target`` names a level (stock) series rather than a flow.

    Exact match on the stripped, lowercased name, which is what ``target_kinds.is_stock`` does.
    """
    return str(target).strip().lower() in STOCK_ALIASES


# ══════════════════════════════════════════════════════════════════════════════
# Building the series
# ══════════════════════════════════════════════════════════════════════════════

def _ops_module():
    """``backend/ops_baseline`` or None. Imported lazily so a chart cannot break a page.

    Pure pandas on the path used here, despite the module's lazy ``c_dl_pipeline`` import: that
    one lives inside ``ops_daily_series``, which nothing here calls.
    """
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    try:
        import ops_baseline

        return ops_baseline
    except Exception:                                  # noqa: BLE001 - a chart, not logic
        return None


def data_path_for_run(base_dir: Path) -> Optional[Path]:
    """The table THIS run was scored against, from its own ``artifacts/config.json``.

    Not a detail. A run launched from the Lab on an uploaded file records that file here, and the
    comparator has to be built from the same numbers the run was scored against or the line on
    the chart is not the line the skill figure used. Measured on one real run whose config
    pointed at ``frontend/runs_uploads/uploaded.csv``, defaulting to the canonical table instead
    moved ``ops_MAE`` by 886.71, which is small enough to look like rounding and is not.

    Returns None when the run recorded no readable path, and the caller falls back.
    """
    cfg = Path(base_dir) / "artifacts" / "config.json"
    if not cfg.exists():
        return None
    try:
        recorded = json.loads(cfg.read_text(encoding="utf-8")).get("data_path")
    except (OSError, ValueError):
        return None
    if not recorded:
        return None
    p = Path(recorded)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p if p.exists() else None


def unavailable_reason(target: str, base_dir: Path) -> Optional[str]:
    """The reader-facing reason there is no Ops baseline, or None when there should be one.

    Checked before any work, so the cases that need no computation do not pay for it.
    """
    if _is_stock(target):
        return _t(REASON_STOCK)
    if not (Path(base_dir) / "predictions_long.csv").exists():
        return _t(REASON_NO_PREDICTIONS)
    if _ops_module() is None:
        return _t(REASON_NO_BACKEND)
    return None


def compute(base_dir: Path, target: str,
            data_path: Optional[Path] = None) -> Tuple[Optional[pd.Series], Optional[str]]:
    """``(series, None)`` on success, ``(None, reason)`` otherwise. Never raises.

    ``base_dir`` is the run's ``outputs`` directory, the one holding ``predictions_long.csv``. The
    data file is taken from the run's own config unless one is passed explicitly.
    """
    reason = unavailable_reason(target, base_dir)
    if reason:
        return None, reason

    ops = _ops_module()
    data = Path(data_path) if data_path else (data_path_for_run(base_dir) or DEFAULT_DATA)
    if not data.exists():
        return None, (f"The Ops baseline needs the Treasury table at `{data}`, which was not "
                      f"found.")

    try:
        frame = pd.read_csv(Path(base_dir) / "predictions_long.csv",
                           usecols=lambda c: c in ("target_date", "origin_date", "target"))
        if "target" in frame.columns:
            frame = frame[frame["target"].astype(str) == str(target)]

        target_dates = pd.to_datetime(frame["target_date"], errors="coerce")
        origins = (pd.to_datetime(frame["origin_date"], errors="coerce")
                   if "origin_date" in frame.columns else target_dates)
        origins = origins.fillna(target_dates)
        keep = target_dates.notna()
        target_dates, origins = target_dates[keep], origins[keep]
        if not len(target_dates):
            return None, _t(REASON_NO_PREDICTIONS)

        # Recorded, not refused: a report over the holdout is allowed and is logged. Usually a
        # no-op, because UI-launched runs are bound to train and dev.
        ops.log_sealed_window_read(str(target), target_dates,
                                   caller="frontend.ops_baseline_view")

        cache = ops.vintage_cache(data, str(target), origins, target_dates)

        # One value per target date. The comparator depends on (target date, origin), and every
        # model in a run shares both, so duplicates across models are identical by construction
        # rather than averaged away.
        values, seen = {}, set()
        for td, origin in zip(target_dates, origins):
            if td in seen:
                continue
            seen.add(td)
            value, _why = ops.ops_prediction_for(td, origin, cache)
            if pd.notna(value):
                values[td] = float(value)
    except Exception as exc:                           # noqa: BLE001 - a chart, not logic
        return None, (_t("The Ops baseline could not be worked out.")
                      + f" {type(exc).__name__}: {exc}")

    if not values:
        return None, _t(REASON_NO_WINDOW)
    return pd.Series(values).sort_index(), None


def usable(series: Optional[pd.Series]) -> bool:
    """Whether a series actually carries numbers.

    The specific check the old readers were missing. An all-NaN series is not empty, so
    ``not s.empty`` let one through, and ``.resample(...).sum()`` then turned it into a line of
    zeros. "Has at least one real number" is the question that needed asking.
    """
    return series is not None and not series.empty and bool(series.notna().any())
