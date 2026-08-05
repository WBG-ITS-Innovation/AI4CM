# Phase 1 — Trust Execution Plan

**Branch:** `fix/trust-phase`
**Source review:** [`docs/reviews/2026-08-04_review.md`](reviews/2026-08-04_review.md)
**Started:** 2026-08-04

One commit per step. After each step: full test suite **plus** the specific verification built in
the review, with before/after evidence in the commit message.

---

## Decisions taken before starting

Recorded here so the reasoning is not lost, and so a reviewer can see what was chosen over what.

| # | Decision | Chosen | Rejected, and why |
|---|---|---|---|
| D1 | Order of work | **Regenerate the data first, before anything else** | Every metric in the review rests on `master_daily_clean_treasury.csv`; fixing anything else first means re-measuring it twice |
| D2 | Publication gate | **Option B — staging directory + validate + atomic promote** | Option A (validate in place) leaves a malformed run sitting at the path the Agent reads |
| D3 | Bug fixes | **Each fix lands with its regression test in the same commit** | Batching fixes then tests loses the mutation evidence that the test actually catches the bug |
| D4 | Intervals | **P1 now, CQR after the data regeneration** | Calibrating CQR against fabricated weekend targets would tune to noise |
| D5 | Cleaning causality | **Expanding window** — for each row, fit imputation/clipping only on rows strictly before it | A frozen pre-2025 fit cannot serve a 2027 production run and hardcodes a year |
| D6 | Georgian calendar scope | **`bdom` counter + month-length denominator** use the true calendar | Excluding holidays from the modelling index too would redefine "h=5 business days" for all four families and re-open the yardstick Step 2 exists to close |
| D7 | Duplicate integrity module | **Move `signal_sentinel` into `forecast_integrity.py`; delete the duplicate implementations; leave `preprocessing/integrity.py` as a deprecated re-export** | A full delete drops 6 fields the Dashboard reads; keeping the module leaves two integrity modules and the ambiguity returns |
| D8 | Canonical review location | **`docs/reviews/2026-08-04_review.md`** | `docs/` = durable documentation (plan, contract, reviews); `reports/` = generated analysis output |

### Assumptions stated rather than asked

- **`mae_seasonal_naive` is aliased, not removed.** The Dashboard reads it, so the key survives. It
  gains a companion `seasonal_naive_season_steps`, and the value becomes `NaN` with
  `seasonal_naive_degenerate: true` whenever the season equals the horizon — the h=5 case where it was
  silently identical to persistence.
- **Minimum evaluation points = 30 target dates**, as a named constant. Below it,
  `run_status = "INSUFFICIENT_DATA"` and `gate_passed = None` (never verified) — *not*
  `FAILED_QUALITY`. The model did not fail; we could not measure it. Hard-failing the whole run for one
  thin family would be worse than reporting honestly.

---

## Step 0 — Setup

- [x] Branch `fix/trust-phase` created off `feat/calendar-features`
- [x] Land the five mutation-validated regression tests from the review (`f0c47d4`)
- [x] Move the review to `docs/reviews/2026-08-04_review.md`
- [x] Write this plan

---

## Step 1 — Data: make cleaning causal and regeneration reproducible

- [x] Make imputation causal: weekday reference fitted on rows **strictly before** the imputed row
      (currently `dow_values.tail(weekday_weeks).median()` takes the *last* N occurrences in the whole
      series — 2025 values used to fill a 2016 gap)
- [x] Make clipping causal: median/MAD thresholds fitted on rows **strictly before** each row
      (currently whole-series per-weekday statistics, including the 2025 holdout)
- [x] Distinguish absent rows from true zeros at ingestion; keep NaN as NaN through to the modelling
      layer rather than `fillna(0.0)` erasing the distinction
- [x] Abort on unexpected missing business days, with Georgian holidays treated as expected
      (measured: 117 of 118 business-day NaNs are holidays; the single exception is **2018-11-28**,
      which needs an explicit allow-list entry or the historical regeneration cannot run)
- [x] Regenerate `master_daily_clean_treasury.csv` with current code
- [ ] Select the input by **explicit name + recorded SHA-256**, never mtime
      (`run_daily_forecast.sh:60` currently uses `ls -t | head -1`)
- [ ] Re-run the §4 backtest; paste the new-vs-old tier table

**Verification:** §2.3 measurement script (imputed/clipped counts, how many fall in the 2025 window);
weekend-value check (expect 1,095 fabricated non-zero weekend values to become 0); §4 shared-window
tier table; full suite.

---

## Step 2 — One yardstick

- [ ] Reindex E_QUANTILE to `freq="B"` so `h=5` means 5 business days, not 5 calendar days
- [ ] Pin C_DL to `TEST_START` so it reports on the shared window, not 2019–2025
- [ ] Verify all four persistence MAEs are **literally one number**
- [ ] Delete `integrity_report.update(legacy_report)` (`b_ml_pipeline.py:1015`)
- [ ] Retire the duplicate integrity module per D7
- [ ] Alias `mae_seasonal_naive` per the assumption above

**Verification:** the §4 four-source baseline-identity table, extended to require cross-family
equality; §1.2 `season_steps == horizon` degeneracy check; full suite.

---

## Step 3 — The four known-unfixed bugs, each with a mutation-validated test

One commit per bug (D3).

- [ ] **(a)** `leaderboard.csv` and the top-model plots respect `select_best_model`, with excluded
      models marked (`is_overfit_excluded`). Today `leaderboard.csv` ranks excluded XGBoost #1 and
      `*_overlay_top.png` is drawn for it while the report names RandomForest
- [ ] **(b)** Seasonal-naive duplication (aliased in Step 2; regression test lands here)
- [ ] **(c)** `bdom_rev` built from the true Georgian business calendar via `preprocessing/holidays.py`
      per D6. Re-measure the Phase-3 skill claim on **DEV only** and record the new figure
- [ ] **(d)** `detect_lagged_copy` made horizon-aware, with a margin that can fire on smooth series
      (today the hardcoded `0.05` correlation margin cannot be cleared when `corr@0 = 0.965`, so a
      *perfect* h-step persistence copy goes unflagged at every `max_shift`)
- [ ] C_DL integrity-report filename discoverable by Dashboard and Lab
- [ ] E_QUANTILE multivariate: remove `bfill()`, make top-K feature selection train-only

**Verification:** per-bug mutation test (revert the fix, confirm the new test fails, restore); full
suite.

---

## Step 4 — Ops P0 (§7.6)

- [ ] Staging directory + validate + atomic promote (D2); extend the `rm -rf` safety guard to the
      staging root
- [ ] `flock` concurrency guard
- [ ] `tee` to `<run_dir>/run.log`
- [ ] Minimum-evaluation-points enforcement per the assumption above
- [ ] Wire `data_preflight.py` into the batch path as a hard blocker
- [ ] Pin `RUN_DATE` timezone
- [ ] Exit non-zero when all families are withheld

**Verification:** simulated mid-run failure leaves the previous good run intact; concurrent-trigger
test; truncated-input test (§7.1 — A_STAT currently publishes a 10-point verdict); full suite.

---

## Step 5 — Agent contract (§6)

- [ ] `docs/AGENT_CONTRACT.md`
- [ ] `schema_version: 2` additive keys: `data_file{name,path,sha256,latest_data_date,n_rows}`,
      `status`, per-family `notes`, numeric twins (`skill_pct_value`, `horizon_steps`),
      `eval_window`, `git_sha`, package versions, seeds, per-family wall-clock
- [ ] Declarative validator + CLI, wired into the staging gate
- [ ] Regression tests for **C1** (`data_file` present) and **C2** (`notes` present when `ok` is false)
- [ ] Run the validator over all existing `forecast_runs/` folders and report findings
      (expect failures: `2026-07-31` has no `SUMMARY.json`; older runs predate the M-fixes)
- [ ] **Separate commit, marked "pending Agent-team sign-off":** contract Phase 2 CSV-content changes
      (leaderboard join keys, `rank` semantics, `is_baseline`, `y_lo`/`y_hi` aliases, `pi_nominal`,
      C_DL empty-stub column set)

---

## Step 6 — Intervals P1 (§3)

- [ ] Split-conformal for ResidualRF (measured: 69.8% → 82.4% coverage)
- [ ] GBQuantile monotonicity (2 observed `p50 > p90` crossings)
- [ ] Record `interval_method` and `n_calibration`
- [ ] Finite-sample conformal quantile `⌈(n+1)(1−α)⌉/n` with h-gapped calibration in B_ML and C_DL
- [ ] Fix or remove A_STAT's ETS intervals with a **logged, targeted** except
      (`HoltWintersResults` has no `get_prediction`; a bare `except` has always swallowed this, so
      every ETS interval has always been NaN)

**Verification:** the §3.2 coverage table — marginal, conditional by magnitude and volatility tercile,
and block-bootstrap CI; full suite.

---

## Step 7 — Cleanup

- [ ] Mark `backend/LEAKAGE_AUDIT.md` superseded by the review (it asserts "PASS — no data leakage
      found" and cites a line the C-3 fix changed)
- [ ] Delete dead files: `a_stat_models_pipeline.py`, `overfitting_check.py`, the
      `app/` + `models/registry.py` + `core/db.py` Dash stack, `make_ml_heatmaps.py`,
      `make_weekly_from_daily_stat.py`
- [ ] Untrack `.venv` and `.env` (**contents never printed**)

---

## Finish

- [ ] Run the whole daily pipeline **twice** and diff
- [ ] Write `reports/phase1_trust.md`: every change, every number that moved and why, and how a third
      party re-runs each verification

---

## Standing constraints

- Thresholds are set on **DEV**. The 2025 TEST window is consulted for reporting only. Any step that
  would tune against TEST stops and asks.
- B_ML and E_QUANTILE are not byte-reproducible (thread-reduction order under `n_jobs=-1`, relative
  ~1e-15). Diffs use numeric tolerance, not `cmp`.
- Mutation-validate every new regression test: revert the fix, confirm failure, restore, confirm the
  tree is clean.
