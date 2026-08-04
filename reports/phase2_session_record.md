# Phase 2 — Session Record

Record of the Phase-2 modelling session: the instruction as given, the decisions taken, and the state
at the point work paused.

**Branch:** `model/excellence` (1 commit, based on `55b58c9` on `fix/trust-phase`)
**Date:** 2026-08-04
**Test suite:** 207 → 225 passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**TEST window reads to date:** 0 (`experiments/test_access.log` empty)

---

## 1 · The instruction

> Phase 1 is merged; data and yardstick are canonical. Begin the modeling phase on branch
> model/excellence. Targets from the start: Revenues, Expenditure, State budget balance (the stock path
> — flag anything it breaks). Goal: genuinely excellent, trustworthy forecasts, not leaderboard
> arbitrage.
>
> Ground rules (non-negotiable, build them first):
> - Enforce TRAIN ≤2023 / DEV 2024 / TEST 2025 in code via evaluation_windows.py. All search and
>   selection on TRAIN-internal rolling-origin folds; DEV consulted sparingly for confirmation; TEST
>   untouched until I say so, and any TEST read must be logged loudly.
> - Append-only experiments log (experiments/log.csv + one JSON per run): timestamp, git SHA, data
>   SHA-256, target, feature-set hash, params, seed, fold scheme, DEV MAE, MASE, skill vs unified
>   ruler, sentinel ratio, per-tercile coverage. Every number in your reports must be reproducible
>   from a logged run.
> - All leak checks, gates, and positive controls stay on. Any new feature ships with a one-line
>   information-set justification (what is known at origin) in code comments.
>
> Workstreams, in order, each ending with a short ablation report in reports/:
> 1. Objective alignment: L1 objectives for B_ML trees (LightGBM l1, XGBoost reg:absoluteerror,
>    HistGBDT absolute_error). Per-model DEV deltas.
> 2. Port E_QUANTILE to LightGBM quantile (p10/50/90, crossing-safe) and run Optuna (~100 trials; lr,
>    num_leaves, min_child_samples, subsample, colsample, reg_alpha/lambda, n_estimators via early
>    stopping on an h-gapped tail). Report best config vs the sklearn-default incumbent.
> 3. External calendar module backend/preprocessing/fiscal_calendar.py: Georgian tax deadlines (VAT,
>    income, profit, excise — weekend-shifted), public-sector salary/pension dates, and debt-service
>    dates where public. Cite the official source for every date in the module docstring and mark
>    unverified entries; produce docs/FISCAL_CALENDAR_SOURCES.md for Treasury to confirm. Features:
>    is_deadline, days_to/since_deadline, is_holiday, days_to/since_holiday, bridge days, Easter
>    window, true-calendar days_to_eom/eoq/eoy, week_of_month, calendar-aligned lags (same bdom last
>    month / last year), rolling median/max/quantiles at 5/21/63 (shifted), EWMAs. Drop raw year.
>    Ablate feature groups per target; keep only what pays on DEV.
> 4. Target-scale experiment per target: raw vs log1p vs ratio-to-trailing-level (fit per fold); stock
>    target uses the delta path.
> 5. Multivariate, leak-safe: lagged cross-target features among the three priority targets and their
>    components, selection fit on train folds only.
> 6. Ensemble via ensemble_postprocess.py: median of (tuned quantile P50, tuned LGBM-L1, ETS); report
>    vs best single.
> 7. Per-target model selection on DEV gated by: skill vs unified ruler, sentinel ratio ≥ 1.5,
>    interval coverage in band including a top-tercile floor. Then CQR on the winning quantile model
>    (review §3 P2.1) and the conditional-coverage gate (P3.1).
>
> C_DL stays parked. Do not touch TEST. End with reports/phase2_modeling.md: per-target DEV results
> table, chosen recipes, and what you'd expect on TEST with uncertainty stated honestly.

---

## 2 · Decisions taken

Three items in the brief rested on premises that did not hold, so they were put to the user.

| # | Question | Chosen | Rejected, and why |
|---|---|---|---|
| D11 | Phase 1's Step 2 (unified yardstick) was never done, and both ground rule 2 and workstream 7 depend on "skill vs unified ruler" | **Fold Step 2 into the Phase-2 ground rules and build it before any modelling** | Finishing all of Phase 1 first defers modelling substantially; a "ruler only, defer the rest" option leaves the published integrity report still coming from the duplicate implementation |
| D12 | E_QUANTILE has no stock-target path at all, but `State budget balance` is a priority target and workstreams 2, 6 and 7 all route through the quantile model | **Add a stock path to E_QUANTILE** (ffill, delta modelling with level reconstruction, `lag_0` as change-context), shipping with its own leak test | Excluding the stock target from the quantile workstreams leaves it with no interval model and unable to pass the workstream-7 coverage gate |
| D13 | Ground rule 2 requires a DEV MASE; no MASE existed and the denominator was unspecified | **In-sample seasonal naive on TRAIN** (Hyndman), season = 5 business days | h-step persistence on TRAIN would keep MASE and "skill vs ruler" on one denominator but departs from the published definition reviewers expect |

### Premise corrections recorded at the time

The brief opened "Phase 1 is merged; data and yardstick are canonical." Measured:

- **Data: canonical.** ✅ SHA `0b009fd0…`, causal cleaning, coverage abort, suite green.
- **Merged: no.** `git branch --contains 4830d4d` returned only `fix/trust-phase`. Nothing on `main`.
- **Yardstick: not built.** Step 2 was never started — E_QUANTILE still on a calendar-day index
  (`h=5` means 5 *calendar* days), C_DL unpinned, `integrity_report.update(legacy_report)` still
  present, and no pipeline imported `evaluation_windows`.

---

## 3 · Output: what was built

### Ground rule 1 — `68ae723`

`evaluation_windows.py` turned from documentation into enforcement. Until this commit the module
stated the discipline and nothing imported it (review §2.4); `eval_start_for()` and `window_for()` had
zero callers, while the only hardcoded evaluation window in the repo pointed at `TEST_START`.

| Added | Behaviour |
|---|---|
| `require_test_access(reason)` | Raises `TestWindowAccessError` unless `AI4CM_ALLOW_TEST_READ=1`; when permitted, prints a stderr banner **and** appends to `experiments/test_access.log`. Refuses an empty reason |
| `restrict(obj, window)` | `'train'` / `'dev'` / `'train+dev'` / `'test'` — the last routes through the gate |
| `assert_within(dates, window, ctx)` | Raises if a step touched a window it should not have |
| `rolling_origin_folds(...)` | TRAIN-internal folds for all search, each with a horizon-sized embargo |
| `seasonal_naive_scale()`, `mase()` | MASE on a TRAIN-only denominator, per D13 |

Measured windows:

```
train      n=2345  2015-01-05 .. 2023-12-29
dev        n= 262  2024-01-01 .. 2024-12-31
train+dev  n=2607  2015-01-05 .. 2024-12-31
test       BLOCKED by default

MASE scale (TRAIN-only, season=5):
  Revenues              51,364,210
  Expenditure           46,747,996
  State budget balance 122,982,009
```

**A design correction made during the work.** The first fold geometry sized evaluation blocks as
`usable // n_folds`, which on this index gives ~1.4-year blocks with the earliest fold training on
about two years — a noisy basis for choosing hyperparameters:

```
fold 1: train<= 2017-01-02  eval 2017-01-09..2018-05-31  gap=5
fold 5: train<= 2022-08-02  eval 2022-08-09..2023-12-29  gap=5
```

Replaced with a fixed `DEFAULT_EVAL_BLOCK=126` (~6 months) over `DEFAULT_MIN_TRAIN=1008` (~4 years);
an impossible geometry now raises rather than silently shrinking. A test pins the *sizing*, not just
the embargo.

Verification: suite 207 → 225 passed (18 new). `test_evaluation_windows_enforced.py` asserts TEST
raises by default; a permitted read is still loud and still logged; search folds never leave TRAIN;
every fold has a ≥ horizon embargo; folds do not overlap and training expands; the default geometry
gives 5 folds of 100–150 rows over ≥1008 training rows; and the MASE scale does not move when
DEV/TEST values are multiplied by 100.

### Supporting

`experiments/README.md` documents the log layout and why `test_access.log` is untracked (a local audit
trail whose value is that it exists at all). `.gitignore` keeps `test_access.log` and `runs/` out while
tracking `log.csv`.

---

## 4 · Issues flagged, not yet resolved

| # | Finding | Consequence |
|---|---|---|
| 1 | **`Revenues` contains negative values** (min −443,977,588); the Phase-1 validity report flags 39 of 41 columns | Whether these are refund/reversal conventions or data problems decides whether `log1p` in workstream 4 is applicable at all |
| 2 | **E_QUANTILE has no stock path** | On the critical path before `State budget balance` can be forecast by the family workstreams 2, 6 and 7 depend on |
| 3 | **Phase 1 is unmerged and Step 2 undone** | "Skill vs unified ruler" has no unified ruler yet; four families still produce four different baselines |

---

## 5 · State at pause

**Built:** ground rule 1 only.

**Not built:** the yardstick (E_QUANTILE reindex, stock path, `.update()` removal), ground rule 2
(`experiments/log.csv`), and all seven workstreams. Measured at pause:

```
W1 L1 objectives                          0 hits
W2 LightGBM quantile port + Optuna        0 files
W3 fiscal_calendar.py                     ABSENT
W3 docs/FISCAL_CALENDAR_SOURCES.md        ABSENT
GR2 experiments/log.csv                   ABSENT
yardstick: E_QUANTILE on freq=B           0 hits
yardstick: .update(legacy_report) removed NO - still present
E_QUANTILE stock path                     0 hits
```

**No per-target candidates exist**, and no DEV selection has been run. `experiments/test_access.log`
is empty: the TEST window has not been read.

The stated next steps were: (1) the yardstick, (2) ground rule 2, (3) workstream 1 — deliberately in
that order, because every number workstream 1 produces has to be logged and measured against the
unified ruler.
