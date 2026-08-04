# Changelog

All notable changes to AI4CM. Newest first.

Numbers quoted here were produced by running code, not by reading it. Where a measurement
contradicted an expectation, both are recorded — see `docs/reviews/2026-08-04_review.md` for the
underlying evidence and `reports/phase1_session_record.md` / `reports/phase2_session_record.md` for the
decisions taken.

---

## Unreleased — Phase 2, modelling (`model/excellence`)

**Status: in progress.** Ground rule 1 only. No modelling has been done; no per-target candidates
exist; the TEST window has not been read (`experiments/test_access.log` is empty).

### Added

- **TRAIN/DEV/TEST enforced in code** (`68ae723`). `backend/evaluation_windows.py` was documentation —
  it stated the discipline and nothing imported it, while the only hardcoded evaluation window in the
  repo pointed at `TEST_START`. Now:
  - `require_test_access(reason)` raises `TestWindowAccessError` unless `AI4CM_ALLOW_TEST_READ=1`;
    when permitted, prints a stderr banner **and** appends to `experiments/test_access.log`.
  - `restrict()`, `assert_within()` make window membership checkable at any call site.
  - `rolling_origin_folds()` gives TRAIN-internal folds for search, each with a horizon-sized embargo.
  - `seasonal_naive_scale()` / `mase()` — MASE on a TRAIN-only seasonal-naive denominator.
- `experiments/` with a documented layout for the append-only run log.

### Measured

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

### Known gaps

- Unified yardstick not built: E_QUANTILE is still on a calendar-day index, so its `h=5` means 5
  *calendar* days against every other family's 5 business days.
- E_QUANTILE has no stock-target path, so `State budget balance` cannot yet be forecast by the family
  that three workstreams depend on.
- `Revenues` contains negative values (min −443,977,588) and the validity report flags 39 of 41
  columns. Whether these are refund conventions or data problems decides whether `log1p` target
  scaling is applicable at all.

---

## Phase 1 — trust (`fix/trust-phase`)

Suite 158 → 207 passing. Data regenerated; no model behaviour changed except as a consequence of the
data.

### Fixed

- **Cleaning is now causal** (`2d37b07`). `clean_treasury` fitted its imputation and clipping on the
  whole series — `.tail(N)` takes the most *recent* N occurrences, so a 2016 gap was filled with the
  median of eight Mondays in 2025, and thresholds applied to 2016 were computed partly from the locked
  2025 holdout. This sat upstream of every split, so no pipeline check could see it. Measured before
  the fix: 223 of 2,763 business-day Revenues values (8.1%) touched by holdout-informed statistics,
  20 inside the 2025 window, median clipping change 118,342,253 — roughly three times the best model's
  MAE. Every statistic is now built from same-weekday observed rows strictly before the row in
  question, and imputed values never re-enter the reference pool.

- **Absent rows are no longer confused with true zeros** (`2d37b07`). The parser turned every blank
  Excel cell into `0.0`, so "not reported" and "zero flow" were indistinguishable from the first step.
  The parser now reports faithfully (blank = NaN) and the variant layer applies flow-vs-level
  semantics. This split was chosen on measurement: 27,693 of 115,428 cells (24.0%) are blank,
  overwhelmingly in sparse line items (Valuables 100%, Shares and other equity 99.2%, Inventories
  99.2%, Dividends 91.2%) where blank *does* mean zero — a blanket blank→NaN would have fabricated 27k
  values. Every headline aggregate has one blank; `State budget balance`, where a zero balance is
  implausible, has two.

- **Missing business days abort instead of being silently zero-filled** (`2d37b07`).
  `check_business_day_coverage()` refuses to preprocess when a business day is absent from the source
  and is not a Georgian public holiday or on a one-entry allow-list. Silently filling such a gap was
  measured to move the h=5 persistence baseline by 10.1% with no warning anywhere. Of 118
  business-day NaNs, 117 are public holidays and exactly one is not (`2018-11-28`); a test asserts the
  allow-list stays that length so a new gap cannot hide among known ones.

### Removed

- **Flow outlier clipping** (`4830d4d`). Making the 8×MAD threshold causal was implemented, measured,
  and abandoned: any causally estimated same-weekday pool is a *sample* that frequently excludes the
  month-end and tax-deadline spikes, so MAD comes out small, 8×MAD is tight, and the spikes get
  clipped. It suppressed the 2024 mean of Revenues by 41% (98,123,411 → 58,213,784) and touched 216 of
  2,763 business days. The old version survived only *because* it was leaky — including the spikes
  inflated MAD into a generous threshold, which is why it clipped just 105 values. MAD clipping was
  never appropriate for this series; the spikes are the signal the day-of-month features exist to
  predict. Replaced by `flow_validity_report()`, which reports negatives and order-of-magnitude jumps
  **without altering anything**.

### Changed — the number that matters

```
h=5 persistence MAE, 2025 window, business-day index:
   raw                83,534,152.85
   NEW                83,534,152.85   <-- cleaning no longer distorts the yardstick
   OLD (leaky)        60,976,736.58
```

The `60,976,736.58` baseline anchoring every skill figure in
`docs/reviews/2026-08-04_review.md` was an artifact of leaky clipping suppressing the spikes. The
honest baseline is **37% higher**, and the cleaned series now matches `raw` exactly. **Every skill
number in that review is superseded and will fall.** The re-run that quantifies this per model has not
been done yet.

Other effects:

| Quantity | Before | After |
|---|---:|---:|
| Observed business-day Revenues altered vs raw | 223 | **2** (both public holidays zeroed, correctly) |
| Imputations across all 41 metrics | 1,326 | **40** |
| Fabricated non-zero weekend Revenues | 1,095 | **0** |

`master_daily_clean_treasury.csv`
`9c04149706946d10e6da3ce2…` → `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`

### Added — regression tests

Twelve tests for causal cleaning plus five locking in earlier fixes the suite could not detect a revert
of (`f0c47d4`). Each was mutation-validated: the fix was reverted, the test confirmed to fail, and the
revert undone. Notably, two existing test files **passed** under mutations that the new tests caught,
confirming the coverage gaps were real:

| Mutation | Result |
|---|---|
| sentinel embargo `gap = 0` | 1 failed |
| pre-M-5 `iloc[:500]` / `iloc[-100:]` slices | 2 failed — `[short] 84 row(s) appear in both slices` |
| `leakage_warning = (ratio < 1.5)` (the inversion) | 3 failed |
| duplicate persistence impl → `median` | 2 failed (`test_unified_baseline.py` passed) |
| remove `train_mae_fold == 0.0` branch | 3 failed (`test_b_ml_overfitting.py` passed) |
| stop publishing `signal_detected` | 3 failed |
| imputation reference from the whole series | 2 failed |
| clipping pool from the whole series | 2 failed |
| coverage abort removed | 1 failed |

### Not done in Phase 1

Steps 2–7 of `docs/EXECUTION_PLAN.md`: the unified yardstick, the four known-unfixed bugs, ops P0
(staging + promote, flock, run log, minimum evaluation points), the Agent contract and validator, and
intervals P1. Phase 1 is **not merged**.

---

## 2026-07-21 — Model audit

`reports/model_audit_2026-07-21.md`. Diagnosis only. Established the M-series findings (C-1 target
scaling, C-2 baseline definitions, C-3 A_STAT horizon, M-1..M-5). Its metrics rest on the pre-Phase-1
data and are superseded.
