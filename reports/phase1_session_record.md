# Phase 1 — Session Record

Record of the Phase-1 execution session: the instruction as given, every decision taken and the
evidence behind it, and the status at the point work paused.

**Branch:** `fix/trust-phase` (5 commits, based on `4c521d0`)
**Date:** 2026-08-04
**Test suite:** 193 → 207 passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`

---

## 1 · The instruction

> The consolidated seven-part review is at docs/reviews/2026-08-04_review.md. All four open decisions
> in its §8.3 are now made: (1) regenerate the data first — before anything else; (2) Option B,
> staging + atomic promote; (3) fix the four known-unfixed bugs and land each fix with its regression
> test in the same commit; (4) interval P1 now, CQR after the data regen.
>
> Work on branch fix/trust-phase. First write docs/EXECUTION_PLAN.md with a checkbox list of the steps
> below, then execute in order, one commit per step, checking boxes as you go. After each step: full
> test suite + the specific verification you built in the review, with before/after evidence pasted
> into the commit message.
>
> 1. Data: make cleaning causal (imputation/clipping fit on data strictly before each row, or strictly
>    pre-2025), distinguish absent rows from true zeros at ingestion and abort on unexpected missing
>    business days, regenerate master_daily_clean_treasury.csv with current code, select input by
>    explicit name + recorded SHA-256 (never mtime). Re-run your §4 backtest and paste the new vs old
>    tier table.
> 2. One yardstick: reindex E_QUANTILE to freq="B", pin C_DL to TEST_START, verify all four
>    persistence MAEs are literally one number; delete integrity_report.update(legacy_report) and
>    retire the duplicate integrity module; remove/alias mae_seasonal_naive.
> 3. Fix the four known-unfixed bugs, each with a mutation-validated regression test: (a)
>    leaderboard.csv and plots respect select_best_model with exclusions marked, (b) seasonal-naive
>    duplication, (c) bdom_rev built from the true Georgian business calendar via
>    preprocessing/holidays.py, (d) detect_lagged_copy made horizon-aware with a margin that can fire
>    on smooth series. Also fix the C_DL integrity-report filename so Dashboard and Lab find it, and
>    fix E_QUANTILE's multivariate ffill/bfill + whole-series feature selection.
> 4. Ops P0 (§7.6): staging dir + validate + atomic promote, flock, tee to <run_dir>/run.log, minimum-
>    evaluation-points enforcement, wire data_preflight.py into the batch path, pin RUN_DATE timezone,
>    exit non-zero when all families are withheld.
> 5. Contract (§6): AGENT_CONTRACT.md, schema_version 2 additive keys (data_file{name,path,sha256,
>    latest_data_date,n_rows}, status, notes, numeric twins, eval_window, git_sha, package versions,
>    seeds, per-family wall-clock), declarative validator + CLI wired into the staging gate, regression
>    tests for C1/C2. Run the validator over all existing forecast_runs folders and report findings.
>    CSV-content changes (contract Phase 2) go in a separate commit clearly marked "pending Agent-team
>    sign-off".
> 6. Intervals P1 (§3): split-conformal for ResidualRF, GBQuantile monotonicity, interval_method +
>    n_calibration recording, finite-sample conformal quantile with h-gapped calibration in B_ML and
>    C_DL, fix or remove A_STAT's ETS intervals with a logged targeted except.
> 7. Cleanup: mark LEAKAGE_AUDIT.md superseded by the review, delete the dead files
>    (a_stat_models_pipeline.py, overfitting_check.py, app/registry/db stack, unreferenced scripts),
>    untrack .venv and .env (do not print their contents).
>
> Finish with: run the whole daily pipeline twice and diff; write reports/phase1_trust.md summarizing
> every change, every number that moved and why, and how a third party re-runs each verification.
> Anything ambiguous: stop and ask, don't assume.

---

## 2 · Decisions taken

### Asked before starting

Four decisions genuinely forked the work, so they were put to the user rather than assumed.

| # | Question | Chosen | Rejected, and why |
|---|---|---|---|
| D5 | Cleaning causality: "strictly before each row" or "strictly pre-2025"? | **Expanding window** — per row, fit only on strictly prior data | A frozen pre-2025 fit cannot serve a 2027 production run and hardcodes a year |
| D6 | How far should the Georgian business calendar reach? | **`bdom` counter + month-length denominator** | Excluding holidays from the modelling index too would redefine "h=5 business days" for all four families and re-open the yardstick Step 2 exists to close |
| D7 | "Retire the duplicate integrity module" — but `signal_sentinel` lives in it, 3 test files import it, and the Dashboard reads 16 fields `compute_integrity_report` supplies | **Move `signal_sentinel` to `forecast_integrity.py`, delete the duplicate implementations, leave a deprecated re-export** | Full delete drops 6 Dashboard fields; keeping the module leaves two integrity modules and the ambiguity returns |
| D8 | The cited review path `docs/reviews/2026-08-04_review.md` did not exist; `docs/` did not exist | **Move it there** | `docs/` = durable documentation (plan, contract, reviews); `reports/` = generated analysis |

### Stated rather than asked

- **`mae_seasonal_naive` aliased, not removed.** The Dashboard reads it, so the key survives; it gains
  `seasonal_naive_season_steps` and becomes `NaN` with `seasonal_naive_degenerate: true` when the
  season equals the horizon — the h=5 case where it was silently identical to persistence.
- **Minimum evaluation points = 30 target dates.** Below it, `run_status = "INSUFFICIENT_DATA"` and
  `gate_passed = None` (never verified), *not* `FAILED_QUALITY`. The model did not fail; we could not
  measure it. Hard-failing a whole run for one thin family would be worse.

### Asked mid-flight, after measurement contradicted the plan

| # | Question | Chosen |
|---|---|---|
| D9 | Causal MAD clipping suppressed 2024's Revenues mean by 41%. What should happen to flow outliers? | **Stop clipping flows entirely**; keep causal imputation plus validity checks that report rather than alter |
| D10 | The regenerated data differs from everything in the review. How to report the effect? | **Re-run the §4 backtest on the new data** and paste the new-vs-old tier table, flagging the review as superseded |

---

## 3 · Measurements that changed the plan

Three findings altered the approach mid-step. Each came from running code, not reading it.

**Blank cells are not all the same thing.** Step 1 asked to "distinguish absent rows from true zeros
at ingestion". The parser turned every blank Excel cell into `0.0`. But measurement showed 27,693 of
115,428 cells (24.0%) are blank, overwhelmingly in sparse line items — Valuables 100%, Shares and
other equity 99.2%, Inventories 99.2%, Dividends 91.2%. For those, blank genuinely means "no
transaction of this type", so a blanket blank→NaN would have fabricated 27k values. Every headline
aggregate has exactly one blank; `State budget balance`, where a zero balance is implausible, has two.

Resolution: parse faithfully (blank = NaN), then apply flow-vs-level semantics in the variant layer
where that distinction already exists. This also yields what was actually needed — after parsing, NaN
inside the reported dates is a blank cell, and NaN introduced by reindexing is an absent date. Those
were the two cases being conflated.

**The abort rule is viable because the data is nearly clean.** Of 118 business-day NaNs, **117 are
Georgian public holidays** and exactly one is not: `2018-11-28`. So the coverage check can be strict,
with a one-entry allow-list, and a test asserts the list stays that length so a new gap cannot hide
among known ones.

**Causal clipping was worse than the leak it fixed.** Making the 8×MAD threshold causal was
implemented, measured, and abandoned:

```
causal clipping, Revenues annual mean (business days)
  2016: raw 39,305,237 -> 25,700,983
  2020: raw 68,463,225 -> 36,723,560
  2024: raw 98,123,411 -> 58,213,784   (-41%)
216 of 2,763 business days altered; 11,135 clips across all 41 metrics
```

Any causally estimated same-weekday pool is a *sample* that frequently excludes the month-end and
tax-deadline spikes, so MAD comes out small, 8×MAD is tight, and the spikes get clipped. The old
whole-series version survived only *because* it was leaky: including the spikes inflated MAD into a
generous threshold, which is why it clipped just 105 values. **MAD clipping was never appropriate for
this series** — the spikes are the signal the day-of-month features exist to predict — and the leak
was masking that.

---

## 4 · Output: state at pause

### Commits landed

| Commit | What |
|---|---|
| `f0c47d4` | Five mutation-validated regression tests from the review |
| `71b375f` | Review → `docs/reviews/2026-08-04_review.md`; `docs/EXECUTION_PLAN.md` with all decisions |
| `2d37b07` | Causal cleaning, faithful blank-cell parsing, coverage abort (+12 tests) |
| `4830d4d` | Flow clipping removed, `clean_treasury` regenerated (+2 tests) |
| `8194267` | Checkbox tick |

### The headline number

```
h=5 persistence MAE, 2025 window, business-day index:
   raw                83,534,152.85
   NEW                83,534,152.85   <-- cleaning no longer distorts the yardstick
   OLD (leaky)        60,976,736.58
```

The `60,976,736.58` anchoring every skill figure in the review was an artifact of leaky clipping
suppressing the spikes. The honest baseline is **37% higher**, and NEW now matches `raw` exactly.
**Every skill number in the review is superseded and will fall.**

### Other effects measured

| Quantity | Before | After |
|---|---:|---:|
| Observed business-day Revenues altered vs raw | 223 | **2** (both public holidays zeroed, correctly) |
| Imputations across all 41 metrics | 1,326 | **40** |
| Fabricated non-zero weekend Revenues | 1,095 | **0** |
| Business-day coverage on regeneration | — | 2645/2763 reported, 117 holidays, 1 allow-listed |

### Step 1 progress

Five of seven boxes. Outstanding: input selection by explicit name + recorded SHA-256, and the §4
backtest re-run with the new-vs-old tier table.

---

## 5 · Two mistakes, recorded

**Causal clipping shipped nothing.** I implemented it before measuring its effect on the real data. It
was caught before the file was installed — the regeneration went to a scratchpad and the canonical CSV
was untouched — but the right order was measure first.

**I destroyed my own uncommitted work.** Mutation-testing the causal cleaner, I ran
`git checkout backend/preprocessing/preprocess.py` to revert the mutation while the Phase-1 edits were
still uncommitted, which discarded all four of them. They were reapplied from context and re-verified.
Earlier mutation tests were safe only because those files happened to be committed. For the rest of the
work, mutations are reverted from a scratchpad backup copy, never with `git checkout` on a dirty file.
The incident is recorded in `2d37b07`'s commit message.

---

## 6 · How a third party reproduces this

```bash
git checkout fix/trust-phase
./backend/.venv/bin/python -m pytest -q                    # expect 207 passed

# regenerate the canonical data and confirm the SHA
cd backend && PP_INPUT_PATH=data/Balance_by_Day_2015-2025.xlsx \
  PP_VARIANT=clean_treasury PP_OUT_ROOT=/tmp/regen PP_RUN_OUTPUTS=/tmp/regen/run \
  ../backend/.venv/bin/python run_preprocess.py
shasum -a 256 /tmp/regen/clean_treasury/Balance_by_Day_2015-2025__clean_treasury.csv
# expect 0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1
```

The causality property is asserted directly by
`backend/tests/test_causal_cleaning.py::test_a_later_value_cannot_change_an_earlier_row`: perturbing
row 150 must leave rows 0–149 bit-identical. Mutation evidence for every new test is in the commit
messages of `2d37b07` and `4830d4d`.
