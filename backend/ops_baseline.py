"""The Treasury's current planning method, as a comparison the Lab can be measured against.

What the method is
------------------
Take the average annual total over the **three preceding complete calendar years**, split it
across months by each month's average share of those years' totals, then spread each month
evenly across its working days. It is a *planning* baseline, refreshed annually — not a forecast.
It cannot react to news, and it is not meant to.

It matters because it is what the Treasury actually uses today. "How much better than the naive
ruler" answers a modelling question; "how much better than the current method" answers theirs.

Why this module exists
----------------------
The arithmetic already lived in ``c_dl_pipeline`` and was reachable from four places, none of
them the publication or scoring path. This is the single reader for the reporting path, so an
ops figure on a leaderboard, in the scorecard, and in a report all come from one construction.

--------------------------------------------------------------------------------
CAUSALITY: WHY THIS IS FAIR AT A DAILY h-STEP ORIGIN
--------------------------------------------------------------------------------
The obvious objection is that a monthly planning average cannot honestly be compared against a
five-business-day-ahead forecast. Measured, it can, and for a specific reason: **every figure
depends only on complete prior calendar years.** The whole 2025 baseline is therefore knowable
on 2025-01-01 — it is constant within the year and fixed before the year begins.

That was verified by truncation rather than argued. Recomputing the baseline from data cut at
four separate origins inside the sealed window reproduces the full-history values exactly:
monthly 354/354 and daily 7618/7618 identical on both flow targets. ``test_ops_baseline.py``
pins it.

**The one place it is not automatically true, and what is done about it.** A target date in
early January needs the year that ended days earlier. At h=5 four sealed-window dates
(2025-01-01, 01-02, 01-03, 01-06) have origins in December 2024, when 2024 was not yet complete
— so the 2025 baseline did not exist at those origins. The Treasury had the *2024* baseline,
built from 2021-2023. So the comparator is **vintage-correct**: each row is scored against the
baseline in force at *its own origin*, which is what the Treasury actually had in hand. See
:func:`ops_prediction_for`.

--------------------------------------------------------------------------------
WHY DAILY GRANULARITY IS NOT A RIGGED COMPARISON
--------------------------------------------------------------------------------
The fear is that scoring a monthly planning total on individual days flatters anything that
reacts to daily variation. Measured on the sealed window, the opposite holds — models beat the
ops method by **more** when both are aggregated to months, which is the method's home ground:

    Revenues, sealed window        skill vs ops (flat)
    XGBoost                                      7.75%
    LightGBM                                    -2.81%
    ElasticNet                                  -7.82%

Five of eleven models measured are WORSE than the current method on this window. That is the
number to take to the Treasury, and it is nothing like the figure the previous implementation
implied -- see the note below.

--------------------------------------------------------------------------------
WHY THE PREVIOUS FIGURES WERE FOUR TIMES TOO GENEROUS
--------------------------------------------------------------------------------
``ops_daily_from_monthly(method="profile")`` returned **identically zero** for every date: 1983
of 1983 non-NaN values exactly 0.00, while the monthly totals were correct. It built each month's
shape from the same month in a previous year, then mapped it onto the current year's dates with
``.reindex(days, fill_value=0.0)`` -- labels that never match across years, so every weight
became zero.

So every ``MAE_skill_vs_Ops`` figure in the repository was skill against a *zero forecast*, i.e.
against ``mean|y|``. On revenues that read 55-62% where the truth against the flat baseline is
-8% to +8%. The bug is fixed at source in ``c_dl_pipeline``; this module exists so the reporting
path has one construction that cannot drift from it.

--------------------------------------------------------------------------------
FLOWS ONLY
--------------------------------------------------------------------------------
The method aggregates a flow to an annual total. A balance *level* has no annual total, so there
is no ops baseline for the stock target and none is invented: :func:`ops_daily_series` returns
``None`` and the caller records the reason. ``c_dl_pipeline`` already made the same call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from target_kinds import is_stock                                        # noqa: E402

#: Years of history the method averages over. The Treasury's own window.
YEARS_WINDOW = 3

#: How a month's total is spread across its working days.
#:
#: ``"flat"`` -- total / working days. This is canonical, for three reasons: it is the method as
#: stated ("spread across working days"), it produces no negative daily figures, and it is the
#: harsher comparison, so no margin is claimed that a softer construction manufactured.
#:
#: ``"profile"`` -- the average intraday shape of that month over the window's years. Available
#: as a sensitivity. It inherits the 72 negative days in the raw revenues series, so it can emit
#: a negative daily *revenue* baseline (measured: 6 of 156 sealed-window days, min -184M), which
#: is not defensible as a planning figure even though the arithmetic is faithful.
SPREAD_FLAT = "flat"
SPREAD_PROFILE = "profile"
DEFAULT_SPREAD = SPREAD_FLAT

#: Why an ops figure is absent, when it is. Recorded per row rather than left blank, so a NaN
#: is never mistaken for "the model tied with the current method".
REASON_STOCK = ("not defined: the Treasury method aggregates a flow to an annual total, and a "
                "balance level has no annual total")
REASON_NO_VINTAGE = ("not defined: no complete three-year window existed at this forecast's "
                     "origin")

#: What the figure IS, when it is present. Travels with the number.
SOURCE_VINTAGE = ("Treasury planning method (3-year average annual total, split by month share, "
                  "spread evenly over working days), at the vintage in force at this row's "
                  "origin")


@lru_cache(maxsize=32)
def _flow_series_cached(path_str: str, mtime: float, target: str,
                        date_col: str) -> pd.Series:
    """Parsed once per (file, target). Keyed on mtime so an edited file is never served stale.

    Without this the CSV was re-read and re-indexed for every vintage of every target, which made
    the test suite roughly eight times slower -- the reporting path is called from many tests.
    """
    df = pd.read_csv(path_str, usecols=[date_col, target], parse_dates=[date_col])
    s = df.dropna(subset=[date_col]).set_index(date_col)[target].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index().asfreq("D").fillna(0.0)


def _flow_series(data_path: Path, target: str, date_col: str = "date") -> pd.Series:
    p = Path(data_path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _flow_series_cached(str(p), mtime, target, date_col)


def _monthly_frame(s: pd.Series) -> pd.DataFrame:
    """Month-end totals with year/month columns. Computed ONCE per vintage, not per month."""
    m = s.resample("ME").sum().astype(float)
    df = m.to_frame("val")
    df["year"], df["month"] = df.index.year, df.index.month
    return df


def ops_daily_series(data_path: Path, target: str,
                     as_of: Optional[pd.Timestamp] = None) -> Optional[pd.Series]:
    """The daily ops baseline for ``target``, or None when the method does not apply.

    ``as_of`` truncates the input, so the result is the series the method would have produced
    from data available at that moment. Truncation is provably inert for dates whose three-year
    window was already complete (module docstring).

    This is the whole-series form, for plots and reports. The **scoring** path uses
    :func:`vintage_cache` / :func:`ops_prediction_for` instead, because those can also value a
    month lying beyond the window complete at the origin -- which this form cannot, since the
    month would be absent from the truncated index entirely.
    """
    if is_stock(target):
        return None
    from c_dl_pipeline import ops_daily_from_monthly, ops_monthly_baseline_treasury

    s = _flow_series(Path(data_path), target)
    if as_of is not None:
        s = s.loc[:pd.Timestamp(as_of)]
    if s.empty:
        return None
    monthly = ops_monthly_baseline_treasury(s, years_window=YEARS_WINDOW)
    if monthly.dropna().empty:
        return None
    return ops_daily_from_monthly(s, monthly, method="profile")


def _vintage_year(origin: pd.Timestamp) -> int:
    """The last calendar year complete as of ``origin``.

    An origin on 2024-12-30 has 2023 as its latest complete year; an origin on 2025-01-02 has
    2024. This is the window a planner standing at that origin could actually have averaged.
    """
    o = pd.Timestamp(origin)
    return o.year if o >= pd.Timestamp(year=o.year, month=12, day=31) else o.year - 1


def ops_figure_for_month(s: pd.Series, year: int, month: int, latest_complete_year: int,
                         years_window: int = YEARS_WINDOW,
                         spread: str = DEFAULT_SPREAD,
                         monthly: Optional[pd.DataFrame] = None) -> Tuple[float, pd.Series]:
    """The method, applied explicitly to one month using a stated three-year window.

    Returns ``(monthly_total, daily_profile_weights)``.

    This exists because a target month can lie *beyond* the window that was complete at the
    origin. In late December 2024 a planner forecasting January 2025 had 2021-2023 closed, not
    2024 — and the method still applies, just with the older window. Deriving the figure from the
    truncated series' own index instead would return nothing for such a month, which would drop
    the comparison rather than make it fairly.

    The window is ``latest_complete_year`` and the ``years_window - 1`` years before it, stated
    by the caller rather than inferred, so the vintage is auditable.
    """
    years = [latest_complete_year - k for k in range(years_window)]
    df = _monthly_frame(s) if monthly is None else monthly

    annual, shares, profiles = [], [], []
    for py in years:
        yr = df[df["year"] == py]
        if len(yr) < 12:                      # not a complete year: the window is unusable
            return float("nan"), pd.Series(dtype=float)
        total = float(yr["val"].sum())
        annual.append(total)
        cell = yr[yr["month"] == month]["val"]
        if cell.empty or total <= 0:
            return float("nan"), pd.Series(dtype=float)
        shares.append(float(cell.iloc[0]) / total)

        # Intraday shape of the same month in that year, over its working days.
        hist = pd.date_range(pd.Timestamp(year=py, month=month, day=1),
                             pd.Timestamp(year=py, month=month, day=1) + pd.offsets.MonthEnd(0),
                             freq="B")
        d = s.reindex(hist).fillna(0.0)
        if float(d.sum()) > 0:
            profiles.append((d / d.sum()).to_numpy())

    if not annual or not shares:
        return float("nan"), pd.Series(dtype=float)

    total = float(np.mean(annual)) * float(np.mean(shares))

    days = pd.date_range(pd.Timestamp(year=year, month=month, day=1),
                         pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
                         freq="B")
    n = len(days)
    if spread == SPREAD_PROFILE and profiles:
        # Working-day counts differ between months, so shapes are mapped onto this month's
        # length by position -- never by date label, which is the bug this module documents.
        resampled = [np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pr)), pr)
                     for pr in profiles]
        w = np.mean(np.vstack(resampled), axis=0)
        w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    else:
        w = np.ones(n) / n          # canonical: spread evenly across working days
    return total, pd.Series(w, index=days)


def ops_series_for_vintage(data_path: Path, target: str, latest_complete_year: int,
                           months, spread: str = DEFAULT_SPREAD) -> Optional[pd.Series]:
    """Daily ops values covering ``months``, all built from one stated vintage window."""
    if is_stock(target):
        return None
    s = _flow_series(Path(data_path), target)
    s = s.loc[:pd.Timestamp(year=latest_complete_year, month=12, day=31)]
    if s.empty:
        return None
    monthly = _monthly_frame(s)
    pieces = []
    for (y, mo) in sorted({(pd.Timestamp(m).year, pd.Timestamp(m).month) for m in months}):
        total, w = ops_figure_for_month(s, y, mo, latest_complete_year, spread=spread,
                                       monthly=monthly)
        if np.isfinite(total) and len(w):
            pieces.append(w * total)
    if not pieces:
        return None
    return pd.concat(pieces).sort_index()


def vintage_cache(data_path: Path, target: str, origins, target_dates,
                  spread: str = DEFAULT_SPREAD) -> Dict[int, Optional[pd.Series]]:
    """One ops series per distinct vintage, covering the months actually being scored.

    Keyed by vintage rather than by row: rows sharing an origin year share a baseline, and
    recomputing it per row would repeat the same arithmetic thousands of times.
    """
    o = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(origins))).dropna())
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(target_dates))).dropna())
    out: Dict[int, Optional[pd.Series]] = {}
    for oi, ti in zip(o, t):
        vy = _vintage_year(oi)
        out.setdefault(vy, [])
        out[vy].append(ti)
    return {vy: ops_series_for_vintage(data_path, target, vy, months, spread=spread)
            for vy, months in out.items()}


def ops_prediction_for(target_date, origin, cache: Dict[int, Optional[pd.Series]]
                       ) -> Tuple[float, str]:
    """The ops figure for one prediction, and where it came from.

    Returns ``(nan, reason)`` rather than raising when the method cannot speak for this row: a
    stated absence is usable, a fabricated number is not.
    """
    td = pd.Timestamp(target_date).normalize()
    series = cache.get(_vintage_year(origin))
    if series is None:
        return float("nan"), REASON_NO_VINTAGE
    v = series.reindex([td]).iloc[0]
    if not np.isfinite(v):
        return float("nan"), REASON_NO_VINTAGE
    return float(v), SOURCE_VINTAGE


def skill_vs(model_abs_error, comparator_abs_error) -> float:
    """Percent by which the model's error is below the comparator's. Same form as the ruler's.

    ``nan`` when the comparator has no error to improve on, rather than a division blowing up
    into an infinite skill.
    """
    m, c = float(model_abs_error), float(comparator_abs_error)
    if not (np.isfinite(m) and np.isfinite(c)) or c <= 0:
        return float("nan")
    return (1.0 - m / c) * 100.0


def log_sealed_window_read(target: str, dates, caller: str) -> int:
    """Record any holdout dates this comparison read. Returns how many there were.

    Measuring a model against the ops baseline over the sealed window is a *report* on what
    would have happened, so this records rather than refuses — the same call A_STAT, C_DL,
    E_QUANTILE and B_ML make on their reporting paths.
    """
    from evaluation_windows import PURPOSE_REPORT, require_test_access, window_for

    holdout = [pd.Timestamp(d) for d in pd.DatetimeIndex(dates)
               if window_for(d) == "test"]
    if holdout:
        require_test_access(
            f"Ops-baseline comparison for {target!r} covers {len(holdout)} holdout target "
            f"date(s) from {min(holdout).date()} to {max(holdout).date()}",
            caller=caller, purpose=PURPOSE_REPORT)
    return len(holdout)
