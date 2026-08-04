# AI4CM — Handoff

**Single self-contained document for resuming work.** Everything needed is inlined; no other file is
required. Share this one file.

**Generated:** 2026-08-04 · **Branch:** `model/excellence` @ `b1b7ae1` · **`main`** @ `a03b1bc`
**Suite:** 237 passing · **TEST (2025) reads: 0**
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`

---

## 0 · How to use this

The project has been through an audit (findings still valid, **metrics superseded**), a data-trust phase
(merged), and the start of a modelling phase (in progress). Three rules have governed the work and
should continue to:

1. **Verify every premise with a measurement before acting on it.** Three separate instructions have
   opened with a premise that did not hold; each time the implied work would have been wrong. §2 gives
   the commands.
2. **TEST (2025) is sealed.** Reading it requires `AI4CM_ALLOW_TEST_READ=1`, raises otherwise, and is
   logged to `experiments/test_access.log`. It must stay empty until final reporting.
3. **Mutation-validate every regression test** — revert the fix, confirm the test fails, restore **from
   a backup copy, never `git checkout` on a file with uncommitted work**. That mistake destroyed four
   edits once.

---

## 1 · Where the work stands

| Phase | Status |
|---|---|
| Audit (`docs/reviews/2026-08-04_review.md`) | Complete. Findings valid; **all its metrics superseded** by the Phase-1 data fix |
| Phase 1 — data trust | **Merged to `main`** (`a03b1bc`). Step 1 done; Steps 2–7 open |
| Phase 2 — modelling | In progress on `model/excellence`. Ground rule 1 done; item 1 partly done; workstreams 1–7 not started |

### Commits on `model/excellence` (6 ahead of `main`)

```
b1b7ae1  docs: yardstick session record
15fb6ee  fix: E_QUANTILE on business days + an honest stock path (yardstick, D12)
b69c060  docs: trust-pack session record + complete open-issue register
cc60073  docs: trust pack -- CHANGELOG, VERIFICATION, and honest stubs
c3c2fd8  docs: Phase-2 session record
489c9e0  feat: enforce TRAIN/DEV/TEST in code with a gated, logged TEST holdout
```

### Reference numbers

```
h=5 persistence ruler, 2025 window, business-day index:
  83,534,152.85   <-- the honest ruler (post-Phase-1)
  60,976,736.58   <-- what the review used; an artifact of leaky clipping. SUPERSEDED

MASE scale (TRAIN-only in-sample seasonal naive, season=5):
  Revenues              51,364,210
  Expenditure           46,747,996
  State budget balance 122,982,009

Windows: train n=2345 (2015-01-05..2023-12-29) | dev n=262 (2024) | test SEALED

Latest DEV results (E_QUANTILE GBQuantile, eval_start=2024-01-01, n=410):
  Revenues              P50 MAE  44,640,347  persistence  85,807,340  skill 47.98%  cov 76.3%
  State budget balance  P50 MAE 151,986,202  persistence 224,174,233  skill 32.20%  cov 70.2%
```

---

## 2 · Verify before acting

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q                      # expect 237 passed
shasum -a 256 backend/data/processed/master_daily_clean_treasury.csv
                                                             # expect 0b009fd031ad...
cat experiments/test_access.log 2>/dev/null || echo "TEST reads: 0"
git rev-list --count main..HEAD                              # expect 6

# still-open issues -- these SHOULD be non-zero / present
grep -c 'update(legacy_report)' backend/b_ml_pipeline.py               # expect 1
grep -c 'TEST_START\|eval_start' backend/c_dl_pipeline.py              # expect 0
grep -c 'season_steps = 5' backend/preprocessing/integrity.py          # expect 1
grep -c 'ls -t' scripts/run_daily_forecast.sh                          # expect 1
ls experiments/log.csv 2>/dev/null || echo "ground rule 2 not built"
ls backend/preprocessing/fiscal_calendar.py 2>/dev/null || echo "W3 not built"

# already fixed -- these should now pass
grep -c 'freq="B"' backend/e_quantile_daily_pipeline.py                # expect >=1
grep -ci 'is_stock' backend/e_quantile_daily_pipeline.py               # expect >=1
```

---

## 3 · Resume point

**Item 1 of the Phase-2 plan, sub-task 1c.** Do them in this order:

| # | Task | Notes |
|---|---|---|
| **1c** | Retire the duplicate integrity module (D7) and delete `integrity_report.update(legacy_report)` | Move `signal_sentinel` into `forecast_integrity.py`; delete `compute_persistence_baseline_from_origin`, `compute_baselines`, `shift_sanity_check`; leave `preprocessing/integrity.py` as a deprecated re-export so 3 test files and 16 Dashboard fields survive. Compute `rmse_model`, `r2_model`, `misaligned_examples`, `lag_warning`, `r2_persistence` from the shared helpers |
| 1d | Alias `mae_seasonal_naive` (A2) | Add `seasonal_naive_season_steps`; set value `NaN` + `seasonal_naive_degenerate: true` when season == horizon |
| 1b | Pin C_DL to `TEST_START` | Small. C_DL is parked, but the pin is needed for the one-ruler check |
| 1e | Input selection by explicit name + recorded SHA-256 | `run_daily_forecast.sh:60` still uses `ls -t \| head -1` |
| 1f | Backtest re-run + one-ruler verification + new-vs-old tier table | The verification for item 1. Needs 1b–1e first |

**Then:** item 2 (ground rule 2 — `experiments/log.csv`), item 3 (workstream 1 — L1 objectives), items
4–7 (workstreams 2–7).

**The one-ruler check has not been performed.** E_QUANTILE is on the right index now, but C_DL is
unpinned and B_ML's published baseline still comes from the duplicate implementation.

---

## 4 · Decisions already taken — do not re-litigate

| # | Decision |
|---|---|
| D1 | Regenerate the data first, before anything else |
| D2 | Publication gate = staging dir + validate + **atomic promote** (not validate-in-place) |
| D3 | Each fix lands with its regression test in the same commit |
| D4 | Intervals P1 now; CQR after the data regen |
| D5 | Cleaning causality = **expanding window** (per row, strictly prior data), not a frozen pre-2025 fit |
| D6 | Georgian calendar = `bdom` counter + month-length denominator. **Not** excluding holidays from the modelling index (that would redefine "h=5 business days") |
| D7 | Move `signal_sentinel` to `forecast_integrity.py`, delete duplicate implementations, keep a deprecated re-export |
| D8 | `docs/` = durable docs; `reports/` = generated analysis |
| D9 | **Stop clipping flow outliers entirely.** Causal MAD clipping suppressed 2024 Revenues by 41%; the old code survived only because it was leaky. `flow_validity_report` flags without altering |
| D10 | Re-run the backtest on the new data and paste a new-vs-old tier table |
| D11 | Fold Phase-1 Step 2 (yardstick) into the Phase-2 ground rules, before modelling |
| D12 | Add a stock path to E_QUANTILE (delta modelling + level reconstruction) — **done**, `15fb6ee` |
| D13 | MASE = TRAIN-only in-sample seasonal naive, season 5 (Hyndman) |
| D14 | Build the candidates before reading TEST |
| D15 | Blocked trust-pack items ship as stubs stating the blocker |
| Q1 | Negatives are **legitimate signed flows** pending Treasury. Workstream 4 candidates: **raw, asinh, ratio-to-trailing-level**. log1p dropped |
| Q2 | Merge Phase 1 into `main`, rebase `model/excellence` — **done** |
| Q3 | Draft `fiscal_calendar.py` with a citation per date (rs.ge / matsne.gov.ge / MoF); mark uncited entries **UNVERIFIED**; write `docs/FISCAL_CALENDAR_SOURCES.md`. UNVERIFIED dates usable, but the experiments log must record a **calendar version hash** |
| Q4 | **Additive** `SUMMARY.json` keys only. CSV-content changes in a marked unmerged commit pending Agent-team sign-off |
| Q5 | Stale feed **blocks** publication (fail closed); overridable only by an explicit env flag recorded in provenance |
| Q6 | C_DL parked through Phase 2 |
| Q7 | CQR proceeds in workstream 7 |
| A1 | Min evaluation points = 30. Below it: `run_status = "INSUFFICIENT_DATA"`, `gate_passed = None` — **not** `FAILED_QUALITY` |
| A2 | `mae_seasonal_naive` aliased, not removed |
| A3 | MASE denominator per D13 |

---

## 5 · Open questions

| # | Question | Blocks |
|---|---|---|
| **OQ1** | **Treasury confirmation on negative flow values** (Revenues min −443,977,588; validity report flags 39 of 41 columns). Question sent; Q1 is the interim position | Nothing immediately, but confirms or overturns workstream 4's transform set |
| **OQ2** | Should `main` / `model/excellence` be **pushed**? Nothing has been pushed; all merges are local | Sharing with the team |
| **OQ3** | Treasury sign-off on `docs/FISCAL_CALENDAR_SOURCES.md` once drafted | Whether workstream 3's features are UNVERIFIED or confirmed |
| **OQ4** | Agent-team sign-off on contract Phase-2 CSV-content changes | Phase-1 Step 5 second commit |
| **OQ5** | When to release TEST | Final reporting |

---

## 6 · Unresolved technical issues

All identified and evidenced in the audit. **None fixed unless marked.** Grouped by owning step.

### Yardstick (Phase-1 Step 2) — blocks "skill vs unified ruler" everywhere

| Issue |
|---|
| ~~E_QUANTILE on a calendar-day index~~ — **fixed** `15fb6ee` |
| ~~E_QUANTILE has no stock path~~ — **fixed** `15fb6ee` |
| C_DL unpinned: folds 2019–2025, reported +10.84% while being −5.19% on the 2025 window |
| `integrity_report.update(legacy_report)` lets a duplicate implementation overwrite the shared one |
| `mae_seasonal_naive` identical to `mae_persistence` at h=5 (`season_steps` hardcoded 5) |

### The four known-unfixed bugs (Phase-1 Step 3)

| Issue |
|---|
| `leaderboard.csv` and top-model plots ignore `select_best_model`: excluded XGBoost ranks #1 and its plots are drawn while the report names RandomForest |
| `bdom_rev` derived from the observed index, so the final partial month is mis-featured (2025-08: `max(bdom)=4` vs a true 21) |
| `detect_lagged_copy` cannot flag h-step persistence at any `max_shift`; the hardcoded 0.05 correlation margin is unclearable when `corr@0 = 0.965`. It caught C_DL by 0.01 of margin |
| C_DL `integrity_<Target>_h<H>.json` not discoverable by Dashboard or Lab, so a Lab-launched C_DL run shows no quality gate |
| E_QUANTILE multivariate `bfill()` + whole-series top-K selection (reachable via a shipped runner) |

### Detection layer

| Issue |
|---|
| **No active leakage detector.** `check_feature_leakage` has zero production callers; `leakage_warning` is a hardwired `False`; only `origin_date >= target_date` is live |
| `signal_sentinel` has no upper bound: an oracle feature equal to `y(t+h)` scores 447× and reports "signal present" — a clean pass |
| `check_feature_leakage` silently misses leakage when given the h-step label (what every pipeline builds) |
| `is_persistence_like` computed correctly by three families and **never read** by the summary |
| C_DL `"alignment_ok": True` is a hardcoded literal |
| A_STAT writes no shift fields, so it has no effective persistence-mimicry check |

### Intervals (Phase-1 Step 6 / P1)

| Issue | Measured |
|---|---|
| ResidualRF coverage 69.8% vs 80% nominal; block-bootstrap CI [61.9, 77.1] excludes 80% | audit §3.2 |
| GBQuantile passes marginally at 78% but covers **51.5%** of the top magnitude tercile; ResidualRF 41.2% | §3.2 |
| GBQuantile quantile crossing (2 rows `p50 > p90`); monotonicity applied only to ResidualRF | §3.2 |
| A_STAT ETS intervals have **always** been NaN — `HoltWintersResults` has no `get_prediction`, swallowed by a bare `except` | §3.4 |
| B_ML and C_DL under-cover 2–10pp; plug-in quantile instead of `⌈(n+1)(1−α)⌉/n`; calibration split has no h-gap | §3.4 |
| No `pi_nominal` recorded; Dashboard hardcodes 90% and cannot read E_QUANTILE's intervals at all | §3.4 |
| **New:** E_QUANTILE stock-target coverage 70.2% — inside the `[0.70, 0.90]` band by 0.2pp. Passes, but on the same knife-edge criticised for ResidualRF. The bootstrap-CI gate (P3.2) would judge it properly | this session |

### Ops P0 (Phase-1 Step 4)

| Issue |
|---|
| No run-log capture at all (`run_daily_forecast.sh` has no `tee`) |
| `rm -rf "$RUN_DIR"` runs **before** any family, so a failed re-run destroys the last good run |
| No `flock`; concurrent triggers corrupt the same directory |
| Input selected by **mtime**, not name + hash |
| `RUN_DATE` timezone unpinned |
| No minimum-evaluation-points enforcement (A_STAT published a verdict off 10 points) |
| `data_preflight.py` not wired into the batch path |
| Exit code 0 when every family is withheld |
| Provenance for 1 of 4 families; no git SHA, package versions or seeds recorded |

### Contract (Step 5) and cleanup (Step 7)

| Issue |
|---|
| 12 contract defects C1–C12: notably missing `data_file`, missing per-family `notes`, A_STAT's NaN leaderboard join keys, three meanings of `rank`, C_DL's empty-stub column set |
| No artifact validator anywhere |
| `MAPE` ≈ 9.4e14 for flow targets; `MAE_skill_vs_Ops` permanently NaN |
| ~1,950 lines dead code (`a_stat_models_pipeline.py`, `overfitting_check.py`, the Dash `app/`+`registry`+`db` stack, two unreferenced scripts) |
| `.venv` and `backend/.env` tracked in git |
| `backend/LEAKAGE_AUDIT.md` asserts "PASS — no data leakage found" and cites a line the C-3 fix changed |

---

## 7 · Standing constraints

- **B_ML and E_QUANTILE are not byte-reproducible** (thread-reduction order under `n_jobs=-1`, relative
  ~1e-15). Diff with numeric tolerance, never `cmp`. A_STAT and C_DL are bit-identical.
- Thresholds are set on **DEV**. TEST is for reporting only.
- **Direct-call unit tests systematically miss wiring.** This has now bitten twice — the M-5 sentinel
  split and the E_QUANTILE reindex both had green unit tests while the pipeline stopped calling the
  fixed code. Every fix needs an assertion on the published artifact, not just the helper.
- Two fixtures have been corrected rather than the code they tested: a constant series has MAD = 0 so
  "no clipping" is correct, and a leak test needed twenty imputed values to discriminate. Check whether
  a failing new test is finding a bug or is simply a bad fixture.

---

## 8 · Document map

| File | Contains |
|---|---|
| **`reports/HANDOFF.md`** | **This file. Self-contained; start here** |
| `docs/reviews/2026-08-04_review.md` | The seven-part audit with `file:line` evidence. Metrics superseded; findings stand |
| `docs/EXECUTION_PLAN.md` | Phase-1 checkbox plan, D1–D8 |
| `reports/phase1_session_record.md` | Phase-1 instruction, decisions, the three measurements that changed the plan, two mistakes |
| `reports/phase2_session_record.md` | Phase-2 instruction, D11–D13, ground rule 1 |
| `reports/phase2_trust_pack_record.md` | Trust-pack instruction, D14–D15, why the TEST read was declined |
| `reports/phase3_session_record.md` | Yardstick instruction, premise verification, Q2 merge, E_QUANTILE fix |
| `CHANGELOG.md` | Every change across both phases with the numbers that moved |
| `VERIFICATION.md` | How a third party re-runs each check, with tolerances |
| `reports/BACKTEST_REPORT.md`, `reports/VALIDATION.json`, `reports/treasury_report.html` | Stubs stating their blockers |
