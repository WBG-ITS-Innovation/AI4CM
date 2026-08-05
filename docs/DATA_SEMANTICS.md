# Data Semantics

What the numbers in `master_daily_clean_treasury.csv` mean, where that is uncertain, and
what has been asked of Treasury. Written so the audit trail carries the open questions
rather than leaving them in a chat log.

**Dataset:** `backend/data/processed/master_daily_clean_treasury.csv`
**SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**Source:** `backend/data/Balance_by_Day_2015-2025.xlsx`
**Coverage:** 2015-01-05 … 2025-08-06 · 3,867 calendar rows · 2,763 business days · 41 metrics

---

## 1 · Negative values in `Revenues`

### Characterization (measured)

| Fact | Value |
|---|---|
| Business days with `Revenues < 0` | **72** of 2,763 (2.6%) |
| Range of those values | −443,977,588 … −249,438 |
| Distribution by year | 2015: 5 · 2016: 5 · 2017: 10 · 2018: 9 · 2019: 10 · 2020: 8 · 2021: 9 · 2022: 7 · 2023: 3 · 2024: 4 · 2025: 2 |

They are **not** a cleaning artifact. On those same 72 days:

| Component | Also negative | Correlation with `Revenues` |
|---|---:|---:|
| `Domestic` | 65 / 72 | **0.969** |
| `Increase in liabilities` | 64 / 72 | **0.971** |
| `Other taxes` | 29 / 72 | — |
| `Taxes` | 12 / 72 | — |

Mean value on those days: `Increase in liabilities` −85,582,305, `Domestic` −87,625,400.

**Reading:** the negatives are driven by netting in the financing lines
(`Increase in liabilities` / `Domestic`), not by anomalies in tax revenue. The near-unity
correlations mean `Revenues` goes negative essentially *because* those components do.

### One correction to an earlier characterization

The negatives were described as "identical in raw". **They are not, by exactly one day.**

```
negatives in raw only  : ['2021-01-19']
negatives in clean only: []

  2021-01-19  raw = -783,716.11   clean = 0.00
    Georgian public holiday? True
```

Raw has **73** negative business days; cleaned has **72**. The single difference is
2021-01-19, a Georgian public holiday carrying a reported negative value, which
`business_days_zero_flows` correctly zeroed. **On genuine business days the two sets are
identical** — so the substance of the characterization holds, and the cleaning is not
introducing or removing negatives.

Worth noting for its own sake: the source reports a non-zero value on a public holiday.
That is either a legitimate back-dated adjustment or a data-entry issue, and it is the
kind of thing the validity report now surfaces rather than silently absorbing.

### Consequences for modelling

Interim position (**decision Q1**): treat negatives as **legitimate signed flows**,
pending confirmation.

- `log1p` is **inapplicable** — it is undefined below −1.
- Workstream 4 target-scale candidates are therefore **raw**, **asinh** (signed,
  defined on the whole real line, ≈ log for large magnitudes) and
  **ratio-to-trailing-level**.
- `flow_validity_report()` continues to flag negatives without altering them. It
  currently reports 39 of 41 columns as having at least one negative or
  order-of-magnitude jump. Flagged, never rewritten.

---

## 2 · Questions put to Treasury

| # | Question | Status | What it changes |
|---|---|---|---|
| **T1** | Are negative `Revenues` values a **refund/reversal or netting convention** (i.e. genuine signed flows), or a data-quality problem? Specifically: is it expected that `Revenues` nets `Increase in liabilities` / `Domestic`, so that gross revenue can print negative on 72 of 2,763 business days? | **Sent.** Interim position per Q1: legitimate signed flows | Confirms or overturns the workstream-4 transform set. If they are errors, they need correcting at source rather than modelling around |
| **T2** | Confirmation of the **fiscal calendar**: statutory tax deadlines (VAT, income, profit, excise) with weekend-shift rules, public-sector salary and pension payment dates, and debt-service dates where public | **Not yet sent** — `docs/FISCAL_CALENDAR_SOURCES.md` must be drafted first (workstream 3, decision Q3) so Treasury has a concrete list to confirm rather than an open question | Every date in the fiscal-calendar feature set. Until confirmed, entries are marked **UNVERIFIED** and the experiments log records a calendar version hash so results can be re-run after sign-off |

**Also worth raising when convenient (not yet asked):** the single non-holiday business
day absent from the workbook, `2018-11-28`. It is allow-listed in
`KNOWN_ABSENT_BUSINESS_DAYS` so historical regeneration can run, but nobody has
confirmed *why* it is missing.

---

## 3 · Other semantics worth recording

### Blank cells are not all the same thing

Measured on the source workbook: **27,693 of 115,428 cells (24.0%) are blank**,
overwhelmingly in sparse line items:

| Metric | Blank share |
|---|---:|
| `Valuables` | 100.0% |
| `Shares and other equity` | 99.2% |
| `Inventories` | 99.2% |
| `Dividends` | 91.2% |
| … | … |
| `Revenues`, `Expenditure`, `Taxes`, all tax lines | 1 cell each |
| `State budget balance` | 2 cells |

For a sparse **flow** line, blank means "no transaction of this type that day" → `0.0`.
For a **level**, a blank is not a zero balance. The parser therefore reports blanks
faithfully as NaN and the variant layer applies flow-vs-level semantics. A blanket
blank→NaN would have fabricated 27k values via imputation.

### Weekends and holidays

Weekends are genuinely unobserved: 1,103 of 1,104 weekend `Revenues` cells are NaN in
raw. Flows are set to `0.0` on non-business days (no trading day, no flow); levels are
forward-filled. Of 118 business-day NaNs, **117 are Georgian public holidays** and one
is `2018-11-28`.

> The committed CSV previously contained **1,095 fabricated non-zero weekend
> `Revenues` values**, produced by a superseded code path. Regeneration removed them.
> Any analysis predating SHA `0b009fd0…` inherited them.

### Outliers are reported, never rewritten

`clean_treasury` performs **no** outlier clipping. Causal 8×MAD clipping was
implemented and abandoned on measurement: it suppressed the 2024 `Revenues` mean by 41%
(98,123,411 → 58,213,784), because any causally estimated same-weekday pool excludes the
month-end and tax-deadline spikes — and those spikes are the signal the day-of-month
features exist to predict. `flow_validity_report()` flags negatives and jumps above 10×
the prior observed maximum without altering anything.

---

## 4 · How to reproduce every figure here

```bash
# dataset identity
shasum -a 256 backend/data/processed/master_daily_clean_treasury.csv
# expect 0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1

# negative-Revenues characterization (§1)
./backend/.venv/bin/python - <<'PY'
import pandas as pd, numpy as np
P="backend/data/processed/"
clean=pd.read_csv(P+"master_daily_clean_treasury.csv",parse_dates=["date"]).set_index("date")
raw=pd.read_csv(P+"master_daily_raw.csv",parse_dates=["date"]).set_index("date")
bd=clean.index.dayofweek<5
neg=clean.index[bd & (clean["Revenues"]<0)]
print("negative business days:", len(neg))
for c in ("Domestic","Increase in liabilities"):
    sub=clean.loc[neg,c]
    print(f"  {c}: negative on {int((sub<0).sum())}/{len(neg)}, "
          f"corr {np.corrcoef(clean.loc[neg,'Revenues'],sub)[0,1]:.3f}")
rn=raw.index[(raw.index.dayofweek<5)&(raw["Revenues"]<0)]
print("raw negatives:", len(rn), "| raw-only:", [str(d.date()) for d in sorted(set(rn)-set(neg))])
PY
```

Blank-cell shares (§3) are reproduced by the script in
`reports/phase1_session_record.md` §3; the clipping measurement is in `CHANGELOG.md`
under Phase 1 → Removed.
