# Fiscal Calendar — Sources and Treasury Sign-off Request

**Calendar version:** `4b480eae9c8f`  
**Module:** [backend/preprocessing/fiscal_calendar.py](backend/preprocessing/fiscal_calendar.py)  
**Generated from** `CALENDAR_ENTRIES` — this file cannot drift from the code that uses it.

---

## Please confirm

This is the fiscal-calendar input to the Treasury daily cash-flow forecasting model. The
model uses these dates to anticipate the large, regular movements in `Revenues` and
`Expenditure` that a purely statistical forecast misses.

**We are asking for confirmation, correction, or deletion of each row below.** Rows marked
**UNVERIFIED** are the priority: we could not find a citable public source for them, so we
have either left them out entirely or included them as an explicitly labelled hypothesis.
None of them should be read as established fact.

Three things would be most useful:

1. **Correct anything wrong.** A wrong date is worse than a missing one, because the model
   will learn around it.
2. **Supply the three schedules we could not source at all** — public-sector salary dates,
   state pension payment dates, and domestic debt auction/redemption dates. The last of
   these is the most valuable: see the note at the end.
3. **Tell us where a rule changed over time.** Our training sample runs 2015–2025, and a
   rule that changed mid-sample (the 2017 profit-tax reform is the known case) needs its
   effective dates, not just the current version.

Every model result records the calendar version above, so once you send corrections we can
re-run and show exactly what changed.

---

## The one rule we confirmed against the legislation

> **Georgian Tax Code, Article 3(6):** *"If the last day of the performance of the action
> coincides with a non-business day, the timeframe for the action shall be extended to the
> end of the next business day."*
>
> Source: <https://matsne.gov.ge/en/document/view/1043717>

This rule does most of the work. It means the effective monthly tax deadline is **not**
always the 15th — in 2024 it moved in three of twelve months:

```
2024 effective monthly deadline dates (statutory day = 15th):
  2024-01-15
  2024-02-15
  2024-03-15
  2024-04-15
  2024-05-15
  2024-06-17   <- shifted
  2024-07-15
  2024-08-15
  2024-09-16   <- shifted
  2024-10-15
  2024-11-15
  2024-12-16   <- shifted
```

Shifted in **3 of 12** months. Across the full 2015–2025 sample the effective deadline differs from the 15th on **27.5%** of business days, which is why the calendar carries information that a plain day-of-month feature cannot.

**Please confirm this shift is forward-only** (a deadline never moves earlier) and that it applies to all the monthly taxes below.

---

## Entries

### `excise` — 🟡 VERIFIED (secondary sources only)

**What it is:** Excise tax return and payment, monthly, by the 15th of the following month.

**Dates used:** monthly, statutory day **15** of the following month, shifted forward per Art. 3(6).

**Source status:** Secondary sources as above. Primary article not located.

**Note:** Coincides with vat_return.

**Please confirm:** dates and that the rule applies unchanged across 2015–2025.

### `pit_withholding` — 🟡 VERIFIED (secondary sources only)

**What it is:** Personal income tax withheld at source by employers, remitted monthly by the 15th of the following month.

**Dates used:** monthly, statutory day **15** of the following month, shifted forward per Art. 3(6).

**Source status:** Same secondary sources as vat_return; monthly withholding regime administered through rs.ge. Primary article not located.

**Note:** Coincides with vat_return, so it contributes no separate date — retained as a named entry for Treasury sign-off.

**Please confirm:** dates and that the rule applies unchanged across 2015–2025.

### `profit_tax` — 🟡 VERIFIED (secondary sources only)

**What it is:** Corporate profit tax under the distributed-profit ('Estonian') model in force since 2017: monthly declaration by the 15th of the following month, payable on distribution rather than on accrual.

**Dates used:** monthly, statutory day **15** of the following month, shifted forward per Art. 3(6).

**Source status:** Secondary sources as above. NOTE: the 2017 switch to the distributed-profit model means the pre-2017 advance-payment schedule differs from the post-2017 one — see `note`.

**Note:** ADVANCE PAYMENTS: the brief asks for profit-tax advance-payment dates. Under the post-2017 distributed-profit model there is no quarterly advance-payment schedule of the classical kind. The pre-2017 regime did have one, which would make this rule REGIME-DEPENDENT across a 2015-2025 training sample. Not implemented as a separate date because no citable pre-2017 schedule was found; flagged for Treasury (T2).

**Please confirm:** dates and that the rule applies unchanged across 2015–2025.

### `vat_return` — 🟡 VERIFIED (secondary sources only)

**What it is:** VAT return filed and VAT liability paid; reporting period is the calendar month, due by the 15th of the following month.

**Dates used:** monthly, statutory day **15** of the following month, shifted forward per Art. 3(6).

**Source status:** Consistent across independent professional sources (Andersen Georgia, Modern Consulting, TPsolution, Legalese Georgia); filed via rs.ge. Governing article not located in the fetched primary text.

**Note:** Shifted per Tax Code of Georgia Art. 3(6) — matsne.gov.ge/en/document/view/1043717

**Please confirm:** dates and that the rule applies unchanged across 2015–2025.

### `domestic_debt_operations` — 🔴 UNVERIFIED — no source found

**What it is:** Domestic debt auction and redemption dates (Treasury securities).

**Dates used:** **none.** This obligation contributes no dates to the model.

**Source status:** NO SOURCE FOUND in this session. NBG (nbg.gov.ge) and MoF (mof.ge) publish auction calendars, but no specific schedule was fetched and verified, so no dates are asserted here.

**Note:** DIRECTLY RELEVANT AND HIGHEST VALUE. docs/DATA_SEMANTICS.md §1 measured that the 72 negative-`Revenues` business days are driven by netting in `Increase in liabilities` (negative on 64/72, correlation 0.971) and `Domestic` (65/72, 0.969) — i.e. debt operations. Those are precisely the days the flow targets cannot predict. An auction/redemption calendar is the single most promising unexploited input, and it is not derivable from `dom`. Blocked on a citable source.

**Please confirm:** **whether this is right at all**, and supply the correct schedule.

### `property_tax_individuals` — 🔴 UNVERIFIED — no source found

**What it is:** Property tax for individuals: annual declaration around 1 November and payment around 15 November.

**Dates used:** annually on **11/1**, **11/15**, shifted forward per Art. 3(6).

**Source status:** NO SOURCE FOUND. These two dates are a HYPOTHESIS carried from general knowledge and were not confirmed against rs.ge or matsne.gov.ge.

**Note:** Included so the ablation can test whether November shows an effect and so Treasury can correct or delete it. Do not cite this entry.

**Please confirm:** **whether this is right at all**, and supply the correct schedule.

### `public_sector_salaries` — 🔴 UNVERIFIED — no source found

**What it is:** Public-sector salary payment dates.

**Dates used:** **none.** This obligation contributes no dates to the model.

**Source status:** NO SOURCE FOUND. Searches returned no Georgian public-sector salary schedule.

**Note:** Deliberately NOT given a hypothesised date. Any monthly salary date is already representable by `dom`/`bdom`, which the models carry, so inventing one would add no information while polluting the calendar with an uncited date. The additive content would be the shift rule applied to it — which requires knowing the actual date. Highest-priority item for Treasury (T2).

**Please confirm:** **whether this is right at all**, and supply the correct schedule.

### `state_pensions` — 🔴 UNVERIFIED — no source found

**What it is:** State pension payment dates.

**Dates used:** **none.** This obligation contributes no dates to the model.

**Source status:** NO SOURCE FOUND. A search for the Georgian state pension schedule returned results for the US state of Georgia, not the country; no citable Georgian source was located.

**Note:** Same reasoning as public_sector_salaries: not hypothesised. Pensions are a large, highly regular expenditure line, so the actual date is likely to be one of the most valuable single facts Treasury can supply.

**Please confirm:** **whether this is right at all**, and supply the correct schedule.

---

## The single most valuable thing you can send us

**Domestic debt auction and redemption dates.**

We measured that the 72 business days on which `Revenues` prints **negative** are driven
almost entirely by netting in the debt-operation lines, not by tax revenue:

| Component | Negative on | Correlation with `Revenues` on those days |
|---|---:|---:|
| `Increase in liabilities` | 64 / 72 | **0.971** |
| `Domestic` | 65 / 72 | **0.969** |

Those days are precisely the ones the model cannot currently anticipate, and no
day-of-month feature can represent them, because auctions are not on a fixed calendar day.
An auction and redemption schedule — even a historical one — would be the highest-value
input we could add.

*(This also relates to open question **T1**: whether those negative values are a genuine
netting convention or a data-quality issue. That question is still outstanding.)*

---

## How to reply

Marking up this file directly is ideal. Otherwise, per row: **confirmed** / **corrected
to X** / **delete** / **rule changed on DATE**.
