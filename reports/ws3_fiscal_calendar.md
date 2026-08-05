# Workstream 3 — Georgian fiscal calendar

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Data SHA-256:** `0b009fd0…5361f1`
**Calendar version:** `4b480eae9c8f` · **Suite:** 315 passing · **TEST (2025) reads: 0**
**All 34 figures below are logged** in `experiments/log.csv` with `calendar_version` recorded
on every row; integrity `{'n_rows': 72, 'ok': True, 'problems': []}`.

---

## The headline, stated first

**The fiscal calendar improves accuracy on all three targets. It does not give the flow
targets detectable signal.**

| Target | DEV MAE improvement | Sentinel before → after | Verdict at threshold 1.50 |
|---|---:|---|---|
| Revenues | **6.32%** | 1.138 → **1.226** | **NO SIGNAL** |
| Expenditure | 1.10% | 1.088 → **1.088** | **NO SIGNAL** |
| State budget balance | 4.26% | 5.566 → **6.926** | SIGNAL (already had it) |

The brief named the sentinel ratio as the number to lift on the flows, "not just MAE". On
that measure the workstream **did not succeed**. Revenues moved 0.087 of the 0.36 it needed;
Expenditure moved **0.000** on DEV. The largest gain went to the one target that already had
signal.

MAE did improve, consistently and on both windows, and that is worth keeping. But a 6%
error reduction on a target whose features still fail a shuffled-target control is the same
situation workstream 1 described, only slightly better: the model is fitting the shape of
the central tendency, not anticipating events.

---

## 1 · What was built

`backend/preprocessing/fiscal_calendar.py`, consumed by **both** B_ML (`calendar_exog`) and
E_QUANTILE (`_calendar_feats`). 31 features in five ablatable groups.

The design point that determined the contents: **the pipelines already carry `dom` and
`bdom`**, so a tree could already learn "the 15th is a big day". A fiscal calendar is
therefore not valuable for saying the 15th matters. Its additive content is:

1. **The shift rule** — Tax Code Art. 3(6), the one rule confirmed against the primary
   source. A deadline on a non-business day extends to the next business day, so the
   *effective* deadline is the 15th in some months and the 16th or 17th in others.
   **Measured: the effective date differs from the 15th on 27.5% of business days**
   (2015–2025); in 2024 it moved in 3 of 12 months. No fixed calendar feature can represent
   that.
2. **Holiday interaction** — proximity, bridge days, and the movable Orthodox Easter window
   (verified against seven known years).
3. **Alignment** — "same business day of month, last month" is not "21 business days ago".

`year` was **removed** from E_QUANTILE, the last family carrying it. A tree that splits on
the calendar year puts every 2025 row in a terminal bucket learned from 2024: it fits the
trend, not the mechanism, and cannot extrapolate.

### Information-set justification, one line per group

| Group | Features | Why it is knowable at the origin |
|---|---|---|
| **A** deadline | `is_deadline_{monthly,annual,any}`, `bdays_{to,since}_deadline`, `deadline_shift_days` | Statutory dates plus the public holiday calendar; both fixed years in advance |
| **B** holiday | `is_holiday`, `days_{to,since}_holiday`, `is_bridge_day`, `days_from_easter`, `in_easter_week` | The Georgian holiday calendar is deterministic; Easter is computable |
| **C** month structure | `bdays_to_{eom,eoq,eoy}`, `week_of_month`, `month`, `dow` | Pure calendar arithmetic over the true calendar |
| **D** aligned lags | `y_aligned_prev_{month,year}` | Target values from strictly earlier rows, shifted one step; index map asserted to reference only past positions |
| **E** rolling/EWMA | `y_roll_{med,max,q90}_{5,21,63}`, `y_ewm_hl{5,21}` | Rolling statistics of the target shifted one step, so a feature at *t* uses *t−1* and earlier |

Distances are **capped at 10**. An uncapped counter lets a tree reconstruct absolute
position in the sample, which is the same failure mode as a raw `year`.

---

## 2 · Marginal contribution of each group (TRAIN-internal folds, 2019–2023, n=1,304)

Vehicles: `LightGBM_L1` for the flows, `HistGBDT_L1` for the stock — the WS1 winners.
`dMAE` is the reduction versus the no-fiscal baseline; `d_sent` the sentinel change.

### Revenues — ruler 67,062,951

| Group | MAE | Skill | dMAE | Sentinel | d_sent |
|---|---:|---:|---:|---:|---:|
| *(baseline, no fiscal)* | 41,517,099 | 38.09% | | 1.0703 | |
| **A** deadline | 40,978,790 | 38.90% | **+1.30%** | 1.0715 | +0.0011 |
| B holiday | 41,488,674 | 38.13% | +0.07% | 1.0882 | +0.0178 |
| C month-structure | 41,125,918 | 38.68% | +0.94% | 1.0843 | +0.0140 |
| D aligned lags | 41,383,104 | 38.29% | +0.32% | 1.0680 | −0.0024 |
| E rolling/EWMA | 41,140,060 | 38.65% | +0.91% | 1.1337 | **+0.0633** |

### Expenditure — ruler 54,170,043

| Group | MAE | Skill | dMAE | Sentinel | d_sent |
|---|---:|---:|---:|---:|---:|
| *(baseline, no fiscal)* | 36,766,991 | 32.13% | | 1.1304 | |
| A deadline | 36,626,158 | 32.39% | +0.38% | 1.1133 | −0.0171 |
| B holiday | 36,361,230 | 32.88% | +1.10% | 1.1288 | −0.0015 |
| C month-structure | 35,797,193 | 33.92% | +2.64% | 1.1517 | +0.0213 |
| **D** aligned lags | 35,650,943 | 34.19% | **+3.04%** | 1.1321 | +0.0017 |
| E rolling/EWMA | 37,359,689 | 31.03% | **−1.61%** | 1.1640 | +0.0337 |

### State budget balance — ruler 153,623,500

| Group | MAE | Skill | dMAE | Sentinel | d_sent |
|---|---:|---:|---:|---:|---:|
| *(baseline, no fiscal)* | 138,414,554 | 9.90% | | 2.2843 | |
| A deadline | 136,861,248 | 10.91% | +1.12% | 2.5767 | **+0.2923** |
| B holiday | 137,094,128 | 10.76% | +0.95% | 2.3051 | +0.0208 |
| C month-structure | 137,358,070 | 10.59% | +0.76% | 2.2923 | +0.0079 |
| **D** aligned lags | 131,389,367 | 14.47% | **+5.08%** | 2.3168 | +0.0325 |
| E rolling/EWMA | 140,232,334 | 8.72% | **−1.31%** | 2.2815 | −0.0028 |

**Group E hurts both non-Revenues targets.** Rolling medians and upper quantiles of the
target are close to a smoothed persistence signal; on Expenditure and the stock they crowd
out the calendar structure that actually pays. It is kept only for Revenues, where it helps.

**Group D is the strongest single group on two of three targets** — and it is not a fiscal
calendar feature at all, it is calendar-*aligned lagging*. The genuinely fiscal groups (A, B)
are the weakest. That is a finding, not a disappointment: it says the exploitable structure
in this data is "this month resembles last month at the same point", not "this is a tax
deadline".

---

## 3 · Best subset, then one DEV confirmation per target

Individual marginals ignore interaction, so three candidate subsets were tested per target
(union of paying groups, top-2, and all five). Winners by TRAIN MAE:

| Target | Winning subset | TRAIN MAE | TRAIN skill | TRAIN sentinel |
|---|---|---:|---:|---:|
| Revenues | **A+B+C+D+E** | 39,862,167 | 40.56% | 1.1810 |
| Expenditure | **A+B+C+D** | 35,069,403 | 35.26% | 1.1524 |
| State budget balance | **A+B+C+D** | 131,168,230 | 14.62% | 2.6854 |

Rejected subsets, for the record: Revenues A+C (40,823,819); Expenditure C+D (35,454,362)
and all-five (35,539,285); stock A+D (131,984,797) and all-five (132,716,026).

### The single DEV confirmation

| Target | Window | MAE base | MAE WS3 | dMAE | Skill base | Skill WS3 | Sent base | Sent WS3 | d_sent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Revenues | TRAIN | 41,517,099 | 39,862,167 | 3.99% | 38.09% | 40.56% | 1.070 | 1.181 | +0.111 |
| Revenues | **DEV** | 55,952,324 | **52,417,152** | **6.32%** | 36.65% | **40.65%** | 1.138 | **1.226** | +0.087 |
| Expenditure | TRAIN | 36,766,991 | 35,069,403 | 4.62% | 32.13% | 35.26% | 1.130 | 1.152 | +0.022 |
| Expenditure | **DEV** | 52,179,454 | **51,602,951** | **1.10%** | 28.64% | **29.42%** | 1.088 | **1.088** | **+0.000** |
| Stock | TRAIN | 138,414,554 | 131,168,230 | 5.24% | 9.90% | 14.62% | 2.284 | 2.685 | +0.401 |
| Stock | **DEV** | 203,596,374 | **194,926,464** | **4.26%** | 16.10% | **19.67%** | 5.566 | **6.926** | +1.360 |

The MAE gains hold on DEV, which matters — a feature set that only helps in-sample would be
worth discarding. Expenditure's DEV gain (1.10%) is much smaller than its TRAIN gain (4.62%),
so its calendar benefit is partly fold-specific.

---

## 4 · Does the sentinel understate nonlinear signal? An open measurement question

`signal_sentinel` fits **Ridge** — a linear probe. Group A's content is a small-cardinality
categorical effect (a deadline moved by 1–3 days in some months), which a linear model can
barely use but a tree can. That would explain the pattern observed here: MAE improved by 4–6%
while the ratio barely moved.

**This must not be used to wave the result through.** It is a hypothesis about the
*instrument*, and it needs its own test: swap the Ridge probe for a shallow tree probe and see
whether the ratio moves on the same feature set. If it does, the current gate systematically
understates tree-exploitable signal and the threshold needs rethinking on its own merits — not
retroactively, and not because a model we like is failing it. If it does not, the honest
reading stands unchanged: these features carry little information about the flows.

Until that test is run, **the flow targets are recorded as having no detectable signal.**

---

## 5 · What most needs Treasury confirmation

`docs/FISCAL_CALENDAR_SOURCES.md` is the sign-off artifact, generated from `CALENDAR_ENTRIES`
so it cannot drift from the code. In priority order:

1. **Domestic debt auction and redemption dates — UNVERIFIED, no dates in the model.** This
   is the highest-value gap and the reason to expect it: `docs/DATA_SEMANTICS.md` §1 measured
   that the 72 negative-`Revenues` business days are driven by netting in `Increase in
   liabilities` (negative on 64/72, correlation **0.971**) and `Domestic` (65/72, **0.969**).
   Those are debt operations, they are exactly the days the flow targets cannot predict, and
   they are **not** on a fixed calendar day, so no `dom`-style feature can reach them. If any
   single input turns the flow sentinel, this is the candidate.
2. **State pension payment dates — UNVERIFIED, no dates.** A large, highly regular
   expenditure line. A search returned results for the US state of Georgia, not the country;
   no Georgian source was found, so nothing was asserted.
3. **Public-sector salary dates — UNVERIFIED, no dates.**
4. **Profit-tax advance payments** — the 2017 switch to the distributed-profit model means
   this rule is regime-dependent across a 2015–2025 sample. No citable pre-2017 schedule was
   found. Treasury's confirmation of *when* the rule changed matters as much as what it is.
5. **`property_tax_individuals` (1 Nov / 15 Nov)** — the only entry carrying **hypothesised**
   dates, explicitly labelled. Confirm or delete.
6. **The monthly 15th deadlines** (VAT, PIT withholding, profit tax, excise) are consistent
   across several independent professional sources but the governing article was not located
   in the primary text. Recorded as VERIFIED/secondary, not promoted.

No citation in the module was written without being fetched. Where nothing was found, the
entry says `NO SOURCE FOUND` and contributes **zero dates** — a test enforces that UNVERIFIED
entries say so, because the sign-off document is generated from those fields and silence
there would become a false claim to Treasury.

---

## 6 · Answering the brief's closing question

> *whether the flow targets now show detectable signal*

**No.** Revenues 1.226 and Expenditure 1.088 against a threshold of 1.50. Revenues moved
0.087; Expenditure did not move at all on DEV. The fiscal calendar is worth keeping for its
4–6% accuracy gain, and it is not the answer to the signal problem.

The two remaining levers are **WS5 (multivariate)** — the 38 other Treasury lines, including
the debt-operation components that the negative-`Revenues` analysis points straight at — and
**an actual auction calendar from Treasury**. On present evidence the second is more promising
than anything in the modelling backlog, because it addresses the specific days that are
unpredictable rather than the average day.

---

## 7 · Reproduction

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q          # expect 315 passed
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from experiment_log import verify_log_integrity as v;print(v())"
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from preprocessing.fiscal_calendar import calendar_version as c;print(c())"  # 4b480eae9c8f
```

Runs are `ConfigBML(target=…, horizon=5, min_train_years=4, variant="univariate",
model_filter=<vehicle>, fiscal_groups=<groups>)` with `eval_end="2023-12-31"` for TRAIN and
`eval_start="2024-01-01", eval_end="2024-12-31"` for DEV. Each asserted its own window
membership (`assert seen == {window}`) before any number was recorded.
