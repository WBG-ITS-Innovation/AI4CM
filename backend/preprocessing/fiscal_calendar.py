"""Georgian fiscal calendar — shared feature source for B_ML and E_QUANTILE.

Workstream 3. Purpose, stated precisely because it determines what belongs here:

The pipelines already carry ``dom`` (calendar day of month) and ``bdom`` (business day
of month), so a tree can already learn "the 15th is a big day". A fiscal calendar is
therefore **not** valuable for telling the model that the 15th matters. Its additive
content is:

1. **The shift rule.** Georgian Tax Code Article 3(6): a deadline falling on a
   non-business day extends to the next business day. So the *effective* VAT deadline is
   the 15th in some months, the 16th or 17th in others, and ``dom`` cannot represent
   that — the spike moves relative to every fixed calendar feature.
2. **Holiday interaction.** Proximity to public holidays, bridge days, and the movable
   Orthodox Easter window, which shifts by up to five weeks between years.
3. **Alignment.** "Same business day of month, last month" is a different quantity from
   "20 business days ago", and only the former lines up two 15ths.

Everything here is computable at the forecast origin from the calendar alone (groups
A/B/C) or from target history strictly before the origin (groups D/E). No feature reads
a value dated at or after the origin; ``test_fiscal_calendar_no_leakage.py`` enforces it
rather than trusting this paragraph.

--------------------------------------------------------------------------------
CITATIONS AND HONESTY
--------------------------------------------------------------------------------
Every entry in ``CALENDAR_ENTRIES`` carries a ``status`` and a ``source_tier``:

* ``VERIFIED`` / ``primary``   — confirmed against the legislation on matsne.gov.ge,
  with the article number.
* ``VERIFIED`` / ``secondary`` — consistent across multiple independent professional
  sources, but the governing article was not located in the primary text. Treated as
  reliable for modelling, still listed for Treasury confirmation.
* ``UNVERIFIED`` / ``none``    — **no source was found.** Where a date appears anyway it
  is an explicit HYPOTHESIS included so the ablation can test whether the data supports
  it and so Treasury can correct it. It is labelled as such in
  ``docs/FISCAL_CALENDAR_SOURCES.md``.

No citation in this file was written without being fetched. Where a search returned
nothing usable — Georgian public-sector salary dates, state pension payment dates, and
domestic debt auction/redemption schedules — the entry says so. An UNVERIFIED entry with
honest provenance is worth more than a plausible-looking URL that was never opened,
because the first can be fixed by one email and the second silently corrupts the audit
trail.

``calendar_version()`` is a content hash over the entries. Every experiment row records
it, so any result can be re-run after Treasury's sign-off changes an entry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .holidays import georgian_holidays_range

# ── the shift rule ─────────────────────────────────────────────────────────────
#
# Tax Code of Georgia, Article 3(6):
#   "If the last day of the performance of the action coincides with a non-business
#    day, the timeframe for the action shall be extended to the end of the next
#    business day."
#
# Source (fetched): https://matsne.gov.ge/en/document/view/1043717
# This is the single most important rule in this module: it is the reason a fiscal
# calendar carries information that `dom` does not.
SHIFT_RULE_CITATION = "Tax Code of Georgia Art. 3(6) — matsne.gov.ge/en/document/view/1043717"


@dataclass(frozen=True)
class CalendarRule:
    """One dated fiscal obligation, with its provenance attached.

    ``kind`` groups rules whose dates coincide (all the monthly 15th-of-next-month
    taxes) so the feature set gets one flag per economically distinct event rather than
    four identical columns.
    """

    name: str
    kind: str                    # "monthly_15" | "annual" | "not_implemented"
    description: str
    status: str                  # "VERIFIED" | "UNVERIFIED"
    source_tier: str             # "primary" | "secondary" | "none"
    citation: str
    # Rule parameters. monthly_15: day-of-month. annual: (month, day) pairs.
    day_of_month: Optional[int] = None
    annual_dates: Tuple[Tuple[int, int], ...] = ()
    note: str = ""


#: The calendar. Order is not significant; the content hash sorts by name.
CALENDAR_ENTRIES: Tuple[CalendarRule, ...] = (
    CalendarRule(
        name="vat_return",
        kind="monthly_15",
        day_of_month=15,
        description="VAT return filed and VAT liability paid; reporting period is the "
                    "calendar month, due by the 15th of the following month.",
        status="VERIFIED",
        source_tier="secondary",
        citation="Consistent across independent professional sources (Andersen Georgia, "
                 "Modern Consulting, TPsolution, Legalese Georgia); filed via rs.ge. "
                 "Governing article not located in the fetched primary text.",
        note="Shifted per " + SHIFT_RULE_CITATION,
    ),
    CalendarRule(
        name="pit_withholding",
        kind="monthly_15",
        day_of_month=15,
        description="Personal income tax withheld at source by employers, remitted "
                    "monthly by the 15th of the following month.",
        status="VERIFIED",
        source_tier="secondary",
        citation="Same secondary sources as vat_return; monthly withholding regime "
                 "administered through rs.ge. Primary article not located.",
        note="Coincides with vat_return, so it contributes no separate date — retained "
             "as a named entry for Treasury sign-off.",
    ),
    CalendarRule(
        name="profit_tax",
        kind="monthly_15",
        day_of_month=15,
        description="Corporate profit tax under the distributed-profit ('Estonian') "
                    "model in force since 2017: monthly declaration by the 15th of the "
                    "following month, payable on distribution rather than on accrual.",
        status="VERIFIED",
        source_tier="secondary",
        citation="Secondary sources as above. NOTE: the 2017 switch to the "
                 "distributed-profit model means the pre-2017 advance-payment schedule "
                 "differs from the post-2017 one — see `note`.",
        note="ADVANCE PAYMENTS: the brief asks for profit-tax advance-payment dates. "
             "Under the post-2017 distributed-profit model there is no quarterly "
             "advance-payment schedule of the classical kind. The pre-2017 regime did "
             "have one, which would make this rule REGIME-DEPENDENT across a "
             "2015-2025 training sample. Not implemented as a separate date because no "
             "citable pre-2017 schedule was found; flagged for Treasury (T2).",
    ),
    CalendarRule(
        name="excise",
        kind="monthly_15",
        day_of_month=15,
        description="Excise tax return and payment, monthly, by the 15th of the "
                    "following month.",
        status="VERIFIED",
        source_tier="secondary",
        citation="Secondary sources as above. Primary article not located.",
        note="Coincides with vat_return.",
    ),
    CalendarRule(
        name="property_tax_individuals",
        kind="annual",
        annual_dates=((11, 1), (11, 15)),
        description="Property tax for individuals: annual declaration around 1 November "
                    "and payment around 15 November.",
        status="UNVERIFIED",
        source_tier="none",
        citation="NO SOURCE FOUND. These two dates are a HYPOTHESIS carried from general "
                 "knowledge and were not confirmed against rs.ge or matsne.gov.ge.",
        note="Included so the ablation can test whether November shows an effect and so "
             "Treasury can correct or delete it. Do not cite this entry.",
    ),
    CalendarRule(
        name="public_sector_salaries",
        kind="not_implemented",
        description="Public-sector salary payment dates.",
        status="UNVERIFIED",
        source_tier="none",
        citation="NO SOURCE FOUND. Searches returned no Georgian public-sector salary "
                 "schedule.",
        note="Deliberately NOT given a hypothesised date. Any monthly salary date is "
             "already representable by `dom`/`bdom`, which the models carry, so inventing "
             "one would add no information while polluting the calendar with an uncited "
             "date. The additive content would be the shift rule applied to it — which "
             "requires knowing the actual date. Highest-priority item for Treasury (T2).",
    ),
    CalendarRule(
        name="state_pensions",
        kind="not_implemented",
        description="State pension payment dates.",
        status="UNVERIFIED",
        source_tier="none",
        citation="NO SOURCE FOUND. A search for the Georgian state pension schedule "
                 "returned results for the US state of Georgia, not the country; no "
                 "citable Georgian source was located.",
        note="Same reasoning as public_sector_salaries: not hypothesised. Pensions are a "
             "large, highly regular expenditure line, so the actual date is likely to be "
             "one of the most valuable single facts Treasury can supply.",
    ),
    CalendarRule(
        name="domestic_debt_operations",
        kind="not_implemented",
        description="Domestic debt auction and redemption dates (Treasury securities).",
        status="UNVERIFIED",
        source_tier="none",
        citation="NO SOURCE FOUND in this session. NBG (nbg.gov.ge) and MoF (mof.ge) "
                 "publish auction calendars, but no specific schedule was fetched and "
                 "verified, so no dates are asserted here.",
        note="DIRECTLY RELEVANT AND HIGHEST VALUE. docs/DATA_SEMANTICS.md §1 measured "
             "that the 72 negative-`Revenues` business days are driven by netting in "
             "`Increase in liabilities` (negative on 64/72, correlation 0.971) and "
             "`Domestic` (65/72, 0.969) — i.e. debt operations. Those are precisely the "
             "days the flow targets cannot predict. An auction/redemption calendar is "
             "the single most promising unexploited input, and it is not derivable from "
             "`dom`. Blocked on a citable source.",
    ),
)


def calendar_version() -> str:
    """12-hex content hash of the calendar.

    Recorded on every experiment row so that a result produced before Treasury's
    sign-off can be told apart from one produced after, and re-run. Covers the fields
    that change what dates come out — including ``status``, so that promoting an entry
    from UNVERIFIED to VERIFIED changes the version even if the dates do not. That is
    deliberate: the provenance of a result is part of the result.
    """
    payload = sorted(
        (
            {
                "name": r.name,
                "kind": r.kind,
                "status": r.status,
                "source_tier": r.source_tier,
                "day_of_month": r.day_of_month,
                "annual_dates": list(r.annual_dates),
            }
            for r in CALENDAR_ENTRIES
        ),
        key=lambda d: d["name"],
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


# ── date generation ───────────────────────────────────────────────────────────

def _holiday_set(start: date, end: date) -> Set[date]:
    return {pd.Timestamp(d).date() for d in georgian_holidays_range(start, end)}


def is_business_day(d: date, holidays: Set[date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def shift_to_business_day(d: date, holidays: Set[date]) -> date:
    """Apply Tax Code Art. 3(6): extend to the end of the next business day.

    Forward only. A deadline never moves earlier -- shifting backward would invent an
    obligation before it legally exists, and would also leak, because the shifted date
    could precede information that determines it.
    """
    out = d
    guard = 0
    while not is_business_day(out, holidays):
        out = out + timedelta(days=1)
        guard += 1
        if guard > 30:  # pragma: no cover - a month of consecutive holidays is impossible
            raise RuntimeError(f"could not find a business day after {d}")
    return out


def deadline_dates(start: date, end: date,
                   holidays: Optional[Set[date]] = None) -> Dict[str, List[date]]:
    """Effective (shift-applied) dates per rule kind, over [start, end].

    Returns one list per ``kind`` that produces dates. Rules of kind
    ``not_implemented`` are absent by construction -- an uncited obligation contributes
    no dates rather than a guessed one.
    """
    if holidays is None:
        # Pad so a December 15th shifting into January is still resolvable.
        holidays = _holiday_set(start - timedelta(days=40), end + timedelta(days=40))

    out: Dict[str, List[date]] = {}
    kinds = {r.kind for r in CALENDAR_ENTRIES}

    if "monthly_15" in kinds:
        dom = next(r.day_of_month for r in CALENDAR_ENTRIES if r.kind == "monthly_15")
        dates = []
        cur = date(start.year, start.month, 1)
        while cur <= end + timedelta(days=40):
            try:
                nominal = date(cur.year, cur.month, dom)
            except ValueError:  # pragma: no cover - dom<=28 in this calendar
                cur = (date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1))
                continue
            shifted = shift_to_business_day(nominal, holidays)
            if start <= shifted <= end:
                dates.append(shifted)
            cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        out["monthly_15"] = sorted(set(dates))

    annual = [r for r in CALENDAR_ENTRIES if r.kind == "annual"]
    if annual:
        dates = []
        for r in annual:
            for y in range(start.year, end.year + 1):
                for (m, d) in r.annual_dates:
                    shifted = shift_to_business_day(date(y, m, d), holidays)
                    if start <= shifted <= end:
                        dates.append(shifted)
        out["annual"] = sorted(set(dates))

    return out


# ── feature groups ────────────────────────────────────────────────────────────
#
# Each group is ablated independently (see reports/ws3_fiscal_calendar.md). Group letters
# are the brief's.
GROUP_A = "A_deadline"      # calendar-deadline proximity
GROUP_B = "B_holiday"       # holiday proximity, bridges, Easter window
GROUP_C = "C_month"         # month/quarter/year structure
GROUP_D = "D_aligned_lags"  # calendar-aligned lags of the target
GROUP_E = "E_rolling"       # rolling / EWMA statistics of the target
ALL_GROUPS: Tuple[str, ...] = (GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E)

#: How far ahead/behind a distance feature is allowed to see. Beyond ~2 weeks the
#: distance to the next monthly deadline is just a relabelling of `bdom`, and an
#: uncapped counter lets a tree reconstruct the absolute date, which is a trend crutch.
DISTANCE_CAP = 10

CALENDAR_ONLY_GROUPS = (GROUP_A, GROUP_B, GROUP_C)
SERIES_GROUPS = (GROUP_D, GROUP_E)


def _bday_distance(idx: pd.DatetimeIndex, events: Sequence[date],
                   cap: int) -> Tuple[pd.Series, pd.Series]:
    """Business days to the next / since the previous event, capped.

    Measured in business days along ``idx`` itself, not calendar days, because the
    quantity that matters operationally is "how many working days of cash movement
    remain", and a weekend is zero of those.
    """
    pos = {d: i for i, d in enumerate(idx.date)}
    ev = sorted({e for e in events if e in pos})
    ev_pos = np.array([pos[e] for e in ev], dtype=np.int64)
    n = len(idx)
    if ev_pos.size == 0:
        return (pd.Series(cap, index=idx, dtype="int64"),
                pd.Series(cap, index=idx, dtype="int64"))
    all_pos = np.arange(n)
    nxt = np.searchsorted(ev_pos, all_pos, side="left")
    prv = nxt - 1
    to_next = np.where(nxt < ev_pos.size, ev_pos[np.clip(nxt, 0, ev_pos.size - 1)] - all_pos, cap)
    since = np.where(prv >= 0, all_pos - ev_pos[np.clip(prv, 0, ev_pos.size - 1)], cap)
    return (pd.Series(np.clip(to_next, 0, cap), index=idx, dtype="int64"),
            pd.Series(np.clip(since, 0, cap), index=idx, dtype="int64"))


def calendar_features(idx: pd.DatetimeIndex,
                      groups: Iterable[str] = CALENDAR_ONLY_GROUPS) -> pd.DataFrame:
    """Groups A/B/C — derived from the index alone.

    Information set: a calendar is known years ahead, so every column here is knowable
    at any origin by construction. This is the one part of the feature set that cannot
    leak, and saying so is not a substitute for the test that checks it.
    """
    groups = set(groups)
    idx = pd.DatetimeIndex(idx)
    out = pd.DataFrame(index=idx)
    if len(idx) == 0:
        return out

    start, end = idx.date.min(), idx.date.max()
    holidays = _holiday_set(start - timedelta(days=400), end + timedelta(days=400))

    # ── A: deadline proximity ────────────────────────────────────────────────
    if GROUP_A in groups:
        dd = deadline_dates(start, end, holidays)
        monthly = dd.get("monthly_15", [])
        annual = dd.get("annual", [])
        dset, aset = set(monthly), set(annual)
        # Justification: the effective deadline date is fixed by statute and the holiday
        # calendar, both known in advance -- nothing here consults the target.
        out["is_deadline_monthly"] = [1 if d in dset else 0 for d in idx.date]
        out["is_deadline_annual"] = [1 if d in aset else 0 for d in idx.date]
        out["is_deadline_any"] = ((out["is_deadline_monthly"] + out["is_deadline_annual"]) > 0).astype(int)
        to_next, since = _bday_distance(idx, monthly, DISTANCE_CAP)
        out["bdays_to_deadline"] = to_next
        out["bdays_since_deadline"] = since
        # Whether the statutory 15th actually moved this month, and by how much. This is
        # the column that carries information `dom` cannot: it is nonzero only in months
        # where Art. 3(6) displaced the deadline.
        shift_amt = []
        for d in idx.date:
            nominal = date(d.year, d.month, 15)
            shift_amt.append((shift_to_business_day(nominal, holidays) - nominal).days)
        out["deadline_shift_days"] = shift_amt

    # ── B: holidays ──────────────────────────────────────────────────────────
    if GROUP_B in groups:
        hol_in_range = sorted(h for h in holidays if start <= h <= end)
        hset = set(hol_in_range)
        out["is_holiday"] = [1 if d in hset else 0 for d in idx.date]
        # Distances measured in CALENDAR days here, not business days: a holiday's cash
        # effect is about the closure itself, and business-day distance would collapse
        # to ~0 for every holiday (holidays are not in a business-day index).
        hol_arr = np.array([pd.Timestamp(h).value for h in hol_in_range]) if hol_in_range else np.array([])
        if hol_arr.size:
            iv = idx.asi8
            nxt_i = np.searchsorted(hol_arr, iv, side="left")
            to_next = np.where(nxt_i < hol_arr.size,
                               (hol_arr[np.clip(nxt_i, 0, hol_arr.size - 1)] - iv) // 86_400_000_000_000,
                               DISTANCE_CAP)
            prv_i = nxt_i - 1
            since = np.where(prv_i >= 0,
                             (iv - hol_arr[np.clip(prv_i, 0, hol_arr.size - 1)]) // 86_400_000_000_000,
                             DISTANCE_CAP)
            out["days_to_holiday"] = np.clip(to_next, 0, DISTANCE_CAP)
            out["days_since_holiday"] = np.clip(since, 0, DISTANCE_CAP)
        else:  # pragma: no cover - the Georgian calendar is never empty over a year
            out["days_to_holiday"] = DISTANCE_CAP
            out["days_since_holiday"] = DISTANCE_CAP
        # Bridge day: a lone business day wedged between two non-business days. Treasury
        # activity on such a day is typically thin, and neither dow nor is_holiday marks it.
        bridge = []
        for d in idx.date:
            prev_nb = not is_business_day(d - timedelta(days=1), holidays)
            next_nb = not is_business_day(d + timedelta(days=1), holidays)
            bridge.append(1 if (prev_nb and next_nb and is_business_day(d, holidays)) else 0)
        out["is_bridge_day"] = bridge
        # Easter window. Orthodox Easter moves up to five weeks between years, so no
        # fixed-date feature can represent it; verified against seven known years.
        from .holidays import orthodox_easter
        easters = {y: orthodox_easter(y) for y in range(start.year - 1, end.year + 2)}
        win = []
        for d in idx.date:
            e = easters[d.year]
            win.append(int(np.clip((d - e).days, -10, 10)))
        out["days_from_easter"] = win
        out["in_easter_week"] = [1 if abs(v) <= 3 else 0 for v in win]

    # ── C: month / quarter / year structure ──────────────────────────────────
    if GROUP_C in groups:
        # Business days remaining to the end of the TRUE calendar period.
        #
        # This must be derived from the calendar, NOT from position within the observed
        # index. Counting rank-from-end inside each observed period looked equivalent and
        # was not: in the final (incomplete) period of any fold it measured "business days
        # until my data runs out", so the value at row t depended on how many rows existed
        # after t. That is lookahead, and it was caught by
        # test_calendar_features_depend_only_on_the_index rather than by reading the code.
        #
        # Built over a padded true business-day calendar (weekends and Georgian public
        # holidays removed) so truncating the sample cannot change any value.
        pad_lo = date(start.year - 1, 1, 1)
        pad_hi = date(end.year + 1, 12, 31)
        cal = pd.DatetimeIndex(
            [d for d in pd.date_range(pad_lo, pad_hi, freq="B")
             if d.date() not in holidays]
        )
        cal_pos = {d: i for i, d in enumerate(cal.date)}
        for period, attr, cap in (("M", "eom", 25), ("Q", "eoq", 70), ("Y", "eoy", 260)):
            per_end = idx.to_period(period).to_timestamp(how="end").normalize()
            vals = []
            for d, pe in zip(idx.date, per_end.date):
                i = cal_pos.get(d)
                if i is None:  # a holiday/weekend row in the input index
                    vals.append(0)
                    continue
                # last calendar business day at or before the true period end
                j = int(np.searchsorted(cal.date, pe, side="right")) - 1
                vals.append(int(np.clip(j - i, 0, cap)))
            out[f"bdays_to_{attr}"] = vals
        out["week_of_month"] = ((idx.day - 1) // 7 + 1).astype("int64")
        out["month"] = idx.month.astype("int64")
        out["dow"] = idx.dayofweek.astype("int64")
        # NOTE: raw `year` is deliberately absent. See drop_raw_year().

    return out


def series_features(y: pd.Series,
                    groups: Iterable[str] = SERIES_GROUPS,
                    lag_safety: int = 1) -> pd.DataFrame:
    """Groups D/E — functions of the target's own history.

    ``lag_safety=1`` shifts every column by one row, so a feature at origin *t* uses
    values dated *t-1* and earlier. That is one step more conservative than necessary
    for a stock target (where y(t) is known at t) and exactly right for a flow reported
    with a lag. Being uniformly conservative costs one day of information and removes a
    whole class of argument about whether same-day values are available.
    """
    groups = set(groups)
    y = pd.Series(y).astype(float)
    idx = pd.DatetimeIndex(y.index)
    out = pd.DataFrame(index=idx)
    if len(y) == 0:
        return out

    ys = y.shift(lag_safety)

    # ── D: calendar-aligned lags ─────────────────────────────────────────────
    if GROUP_D in groups:
        # Same business-day-of-month, previous month; and same (month, bdom), previous
        # year. Justification: two 15ths are the comparable pair, and "21 business days
        # ago" lands on a different bdom in most months. Built by an explicit index map
        # over PAST rows only.
        bdom = pd.Series(idx.to_series().groupby([idx.year, idx.month]).cumcount() + 1,
                         index=idx)
        key = pd.DataFrame({"y": idx.year, "m": idx.month, "b": bdom.to_numpy()}, index=idx)
        lookup = {(r.y, r.m, r.b): i for i, r in enumerate(key.itertuples())}
        vals_1m, vals_1y = [], []
        arr = ys.to_numpy()
        for i, r in enumerate(key.itertuples()):
            py, pm = (r.y, r.m - 1) if r.m > 1 else (r.y - 1, 12)
            j = lookup.get((py, pm, r.b))
            vals_1m.append(arr[j] if (j is not None and j < i) else np.nan)
            k = lookup.get((r.y - 1, r.m, r.b))
            vals_1y.append(arr[k] if (k is not None and k < i) else np.nan)
        out["y_aligned_prev_month"] = vals_1m
        out["y_aligned_prev_year"] = vals_1y

    # ── E: rolling / EWMA ────────────────────────────────────────────────────
    if GROUP_E in groups:
        # Medians and upper quantiles rather than means: the series is spiky, and the
        # point of workstream 1 was that the mean is the wrong summary here.
        for w in (5, 21, 63):
            out[f"y_roll_med_{w}"] = ys.rolling(w, min_periods=max(2, w // 3)).median()
            out[f"y_roll_max_{w}"] = ys.rolling(w, min_periods=max(2, w // 3)).max()
            out[f"y_roll_q90_{w}"] = ys.rolling(w, min_periods=max(2, w // 3)).quantile(0.90)
        for hl in (5, 21):
            out[f"y_ewm_hl{hl}"] = ys.ewm(halflife=hl, min_periods=2).mean()

    return out


def build_fiscal_features(idx: pd.DatetimeIndex,
                          y: Optional[pd.Series] = None,
                          groups: Iterable[str] = ALL_GROUPS,
                          lag_safety: int = 1) -> pd.DataFrame:
    """Assemble the requested groups. ``y`` is required only for D/E."""
    groups = list(groups)
    parts = [calendar_features(idx, [g for g in groups if g in CALENDAR_ONLY_GROUPS])]
    wanted_series = [g for g in groups if g in SERIES_GROUPS]
    if wanted_series:
        if y is None:
            raise ValueError(f"groups {wanted_series} need the target series; y was None")
        parts.append(series_features(y, wanted_series, lag_safety=lag_safety))
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def drop_raw_year(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any raw calendar-year column.

    A tree that splits on ``year`` cannot extrapolate past its training range: every
    2025 row falls in one terminal bucket learned from 2024. It is a trend crutch that
    looks like skill in-sample and degrades the moment the window moves. E_QUANTILE's
    feature builder carried ``year``; this exists so removing it is one call and is
    testable.
    """
    return df.drop(columns=[c for c in ("year", "yr") if c in df.columns])
