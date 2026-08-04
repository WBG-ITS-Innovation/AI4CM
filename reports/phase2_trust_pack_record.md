# Phase 2 — Trust Pack Request: Session Record & Handoff

This file is written to be **self-contained enough to resume from**. It records the instruction, what
was delivered, what was declined and why, and — most importantly — a complete register of every open
question and unresolved issue across the whole engagement, so nothing has to be rediscovered.

**Date:** 2026-08-04
**Branch:** `model/excellence` (3 commits) on top of `fix/trust-phase` (6 commits)
**Test suite:** 225 passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**TEST (2025) window reads to date: 0** — `experiments/test_access.log` is empty

---

## 1 · The instruction

> Approved: run the single TEST (2025) evaluation of the final per-target candidates. Then produce the
> trust pack:
> 1. Tier table per target vs the unified persistence ruler, with sentinel ratio and per-tercile
>    coverage — same format as review §4.4 — plus DEV-vs-TEST comparison so any gap is visible.
> 2. Regenerated BACKTEST_REPORT.md whose "one shared baseline" sentence is now true, with eval windows
>    stated per family.
> 3. The Treasury HTML report per review §7.5: verdict first, plain-language GEL, no acronyms, ranges as
>    behaviour with the big-day caveat measured, withheld results explained, scope line, self-contained
>    file.
> 4. VERIFICATION.md: exact commands for a third party to re-run preprocessing, the backtest, the
>    validator, the positive controls, and the independent metric recomputation, with expected outputs
>    and tolerances.
> 5. CHANGELOG.md across both phases; validator run over the final artifacts attached as VALIDATION.json.
>
> If any final model fails a gate, ship it as WITHHELD with the reason in plain language — do not relax
> a gate to make a model pass.

---

## 2 · What was declined, and why

**The TEST (2025) evaluation was not run.** The instruction was to evaluate "final per-target
candidates". None exist. Measured at the time:

```
W1 L1 objectives (LightGBM l1 / reg:absoluteerror)    0 hits
W2 LightGBM quantile port + Optuna                    0 files
W3 fiscal_calendar.py                                 ABSENT
W3 docs/FISCAL_CALENDAR_SOURCES.md                    ABSENT
GR2 experiments/log.csv                               ABSENT
yardstick: E_QUANTILE on freq=B                       0 hits
yardstick: .update(legacy_report) removed             NO - still present
E_QUANTILE stock path                                 0 hits
```

`model/excellence` contained ground rule 1 and two documentation commits. No workstream ran. No DEV
selection happened.

Reading TEST then would have graded the **pre-Phase-2 models** — the same ones the review already
measured — spending a holdout that can only be spent once, while trust-pack item 1's DEV-vs-TEST column
would have stayed empty because there was no DEV selection behind it. Item 2 could not be satisfied
either: the "one shared baseline" sentence is still false, so regenerating it would have restated the
exact falsehood the review caught.

**Pattern worth carrying forward.** Three consecutive instructions opened with a premise that did not
hold: "Phase 1 is merged" (it is not, still), "the yardstick is canonical" (never built), "final
per-target candidates" (none). In each case the implied work would have been wrong rather than merely
early, and the third would have destroyed the holdout. Checking state against the session records before
issuing the next instruction is cheap insurance.

---

## 3 · Decisions taken this turn

| # | Question | Chosen | Rejected, and why |
|---|---|---|---|
| D14 | No candidates and no DEV selection exist; how to proceed on the TEST read? | **Build the candidates first** — yardstick, ground rule 2, workstreams 1–7, DEV selection, then one TEST read on genuine candidates | Spending TEST on the incumbents now yields a real 2025 number but permanently dirties the holdout for the eventual Phase-2 candidates; a docs-only trust pack leaves the gap undocumented |
| D15 | How to handle trust-pack items that depend on unbuilt infrastructure? | **Write them as stubs stating the blocker** | Omitting them leaves the gap invisible; building the validator early was considered but does not unblock items 1–3 |

---

## 4 · What was delivered

Commit `9bd952c`.

**Real deliverables**

| File | Contents |
|---|---|
| `CHANGELOG.md` | Both phases. Full mutation-evidence table, including the two cases where an existing test file **passed** under a mutation a new test caught. Records the baseline change 60,976,736.58 → 83,534,152.85 and that every review skill figure is superseded |
| `VERIFICATION.md` | Commands, expected outputs and tolerances for the four checks runnable today; precisely what is missing for the three that are not. States the leakage-detector gap explicitly so a green suite is not misread as "no leakage detected" |

**Stubs, written deliberately rather than omitted**

| File | Says |
|---|---|
| `reports/BACKTEST_REPORT.md` | Why "one shared baseline" cannot be made true yet, with the four-family table (two agreeing baselines, one calendar-day horizon, one 2019–2025 window) |
| `reports/VALIDATION.json` | `status: NOT_RUN`, what must be built, and the 12 contract defects C1–C12 the validator must catch |
| `reports/treasury_report.html` | Self-contained, verdict first, no acronyms. Tells a Treasury reader nothing is being published today and why, including the measured big-day interval caveat and the one-of-41-series scope line |

Also this turn: `reports/phase2_session_record.md` (`ec3a2d4`).

---

## 5 · Current state, precisely

### Branches — **neither is merged anywhere**

```
model/excellence     (3 commits, current)
  9bd952c  docs: trust pack -- CHANGELOG, VERIFICATION, and honest stubs
  ec3a2d4  docs: Phase-2 session record
  68ae723  feat: enforce TRAIN/DEV/TEST in code with a gated, logged TEST holdout

fix/trust-phase      (6 commits)
  55b58c9  docs: Phase-1 session record
  8194267  docs: tick Step 1 regeneration checkbox
  4830d4d  data: stop clipping flow outliers; regenerate clean_treasury
  2d37b07  data: make clean_treasury cleaning causal; abort on unexplained gaps
  71b375f  docs: trust review + Phase-1 execution plan
  f0c47d4  test: five mutation-validated regression tests

base: 4c521d0 (feat/calendar-features)
```

### What exists

- Causal cleaning + coverage abort; regenerated canonical data (SHA `0b009fd0…`)
- Flow clipping removed; `flow_validity_report()` reports without altering
- TRAIN/DEV/TEST enforced in code with a gated, logged TEST holdout; TRAIN-internal rolling-origin folds
  with horizon embargo; MASE on a TRAIN-only denominator
- 225 tests, 19 of them mutation-validated regression tests

### What does not exist

- Phase 1 Steps 2–7 entirely (yardstick, four bug fixes, ops P0, contract + validator, intervals P1,
  cleanup)
- Phase 2 ground rule 2 (`experiments/log.csv`) and workstreams 1–7
- Any backtest on the regenerated data, so **no current metrics exist for any model**

### Reference numbers

```
h=5 persistence MAE, 2025 window, business-day index:
  raw / NEW (post-fix)   83,534,152.85     <-- the honest ruler
  OLD (leaky)            60,976,736.58     <-- what the review used; superseded

MASE scale (TRAIN-only, seasonal naive, season=5):
  Revenues              51,364,210
  Expenditure           46,747,996
  State budget balance 122,982,009

Windows: train n=2345 (2015-01-05..2023-12-29) | dev n=262 (2024) | test SEALED
```

---

## 6 · Open questions requiring a decision

Ordered by what blocks the most work.

| # | Question | Blocks | Notes |
|---|---|---|---|
| Q1 | **`Revenues` contains negatives** (min −443,977,588); the validity report flags 39 of 41 columns. Are these refund/reversal conventions or data problems? | Workstream 4 (target scaling) — if negatives are legitimate, `log1p` is inapplicable and a signed transform is needed | Needs Treasury confirmation, not a guess |
| Q2 | **Should `fix/trust-phase` and `model/excellence` be merged, and into what?** (`main`? `feat/calendar-features`?) | Nothing technically, but the repo state does not match the stated premise | Both branches have been described as merged; neither is |
| Q3 | **Fiscal calendar dates need official sources.** Workstream 3 requires citing an official source per date and marking unverified entries | Workstream 3, and any feature derived from tax deadlines | `docs/FISCAL_CALENDAR_SOURCES.md` is the intended artifact for Treasury to confirm; not yet written |
| Q4 | **Contract Phase 2 CSV-content changes need Agent-team sign-off** (leaderboard join keys, `rank` semantics, `is_baseline`, `y_lo`/`y_hi` aliases, `pi_nominal`, C_DL stub columns) | Phase 1 Step 5 second commit | Additive `SUMMARY.json` keys are safe without sign-off; file-content changes are not |
| Q5 | **Stale-data policy: does a stale feed block publication or only warn?** | Ops P0 item 11 | Currently warns and publishes |
| Q6 | **Is C_DL permanently parked, or revisited after the data fix?** | Nothing immediately | On the shared 2025 window all five architectures were −5% to −30%; the C-1 fix improved but did not rescue them |
| Q7 | **CQR (review §3 P2.1) was deferred "until after the data regen".** The regen is done, so it is now unblocked | Workstream 7's conditional-coverage gate | Confirm whether to proceed as originally planned |

### Assumptions I made rather than asking — confirm or overrule

| # | Assumption |
|---|---|
| A1 | **Minimum evaluation points = 30 target dates.** Below it, `run_status = "INSUFFICIENT_DATA"` and `gate_passed = None`, not `FAILED_QUALITY` — the model did not fail, we could not measure it |
| A2 | **`mae_seasonal_naive` aliased, not removed.** Key survives for the Dashboard; gains `seasonal_naive_season_steps` and becomes `NaN` with `seasonal_naive_degenerate: true` when season == horizon |
| A3 | **MASE denominator** = in-sample seasonal naive on TRAIN, season = 5 business days (Hyndman). Confirmed as D13 but worth re-checking against how the Agent will consume it |

---

## 7 · Unresolved technical issues

All identified and evidenced in `docs/reviews/2026-08-04_review.md`. **None is fixed.** Grouped by the
Phase-1 step that owns it.

### Step 2 — unified yardstick (blocks "skill vs unified ruler" everywhere)

| Issue | Evidence |
|---|---|
| E_QUANTILE on a calendar-day index: `h=5` means 5 **calendar** days | review §1.2, §4.3 |
| C_DL unpinned: folds 2019–2025, reported +10.84% while −5.19% on the 2025 window | §4.3 |
| `integrity_report.update(legacy_report)` lets a duplicate implementation overwrite the shared one | §1.2 |
| `mae_seasonal_naive` identical to `mae_persistence` at h=5 (`season_steps` hardcoded 5) | §1.2 |

### Step 3 — the four known-unfixed bugs

| Issue | Evidence |
|---|---|
| `leaderboard.csv` and top-model plots ignore `select_best_model`: excluded XGBoost ranks #1 and its plots are drawn while the report names RandomForest | §1.3 |
| `bdom_rev` derived from the observed index, so the final partial month is mis-featured (2025-08: `max(bdom)=4` vs a true 21) | §2.2 |
| `detect_lagged_copy` cannot flag h-step persistence at any `max_shift`; the hardcoded 0.05 correlation margin is unclearable when `corr@0 = 0.965`. Caught C_DL by 0.01 of margin | §2.5 |
| C_DL `integrity_<Target>_h<H>.json` not discoverable by Dashboard or Lab, so a Lab-launched C_DL run shows no quality gate | §1.4 |
| E_QUANTILE multivariate `bfill()` + whole-series top-K selection (reachable via a shipped runner) | §2.2 |
| E_QUANTILE has **no stock path** — blocks `State budget balance` in workstreams 2, 6, 7 | this session |

### Detection layer

| Issue | Evidence |
|---|---|
| **No active leakage detector.** `check_feature_leakage` has zero production callers; `leakage_warning` is a hardwired `False`; only `origin_date >= target_date` is live | §2.5 |
| `signal_sentinel` has no upper bound: an oracle feature scores 447× and reports "signal present" | §2.5 |
| `check_feature_leakage` silently misses leakage when given the h-step label (what every pipeline builds) | §2.5 |
| `is_persistence_like` computed correctly by three families and **never read** by the summary | §2.5 |
| C_DL `"alignment_ok": True` is a hardcoded literal | §1.4 |
| A_STAT writes no shift fields, so it has no effective persistence-mimicry check | §2.5 |

### Intervals (Step 6 / P1)

| Issue | Measured |
|---|---|
| ResidualRF marginal coverage 69.8% vs 80% nominal; block-bootstrap CI [61.9, 77.1] excludes 80% | §3.2 |
| GBQuantile passes marginally at 78% but covers **51.5%** of the top magnitude tercile; ResidualRF 41.2% | §3.2 |
| GBQuantile quantile crossing (2 rows `p50 > p90`); monotonicity applied only to ResidualRF | §3.2 |
| A_STAT ETS intervals have **always** been NaN — `HoltWintersResults` has no `get_prediction`, swallowed by a bare `except` | §3.4 |
| B_ML and C_DL under-cover 2–10pp; plug-in quantile instead of `⌈(n+1)(1−α)⌉/n`; calibration split has no h-gap | §3.4 |
| No `pi_nominal` recorded; Dashboard hardcodes 90% and cannot read E_QUANTILE's intervals at all | §3.4 |

### Ops P0 (Step 4)

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

### Contract (Step 5) & cleanup (Step 7)

| Issue |
|---|
| 12 contract defects C1–C12, notably missing `data_file`, missing per-family `notes`, A_STAT's NaN leaderboard join keys, three meanings of `rank` |
| No artifact validator anywhere |
| `MAPE` ≈ 9.4e14 for flow targets; `MAE_skill_vs_Ops` permanently NaN |
| ~1,950 lines dead code (`a_stat_models_pipeline.py`, `overfitting_check.py`, the Dash `app/`+`registry`+`db` stack, two unreferenced scripts) |
| `.venv` and `backend/.env` tracked in git |
| `backend/LEAKAGE_AUDIT.md` asserts "PASS — no data leakage found" and cites a line the C-3 fix changed |

### Standing constraints

- Thresholds are set on **DEV**; TEST is consulted for reporting only. Any TEST read must go through
  `require_test_access()` and is logged.
- B_ML and E_QUANTILE are **not byte-reproducible** (thread-reduction order under `n_jobs=-1`, relative
  ~1e-15). Diff with numeric tolerance, never `cmp`.
- Mutation-validate every new regression test, and **revert mutations from a backup copy, never
  `git checkout` on a file with uncommitted work** — that mistake destroyed four edits once already.

---

## 8 · Next steps as agreed (D14)

1. **Yardstick** — reindex E_QUANTILE to `freq="B"`, add its stock path, delete `.update(legacy_report)`,
   retire the duplicate integrity module per D7, alias `mae_seasonal_naive` per A2. Verify all four
   families report literally one persistence number.
2. **Ground rule 2** — `experiments/log.csv` + per-run JSON with the specified columns.
3. **Workstream 1** — L1 objectives, per-model DEV deltas. First real DEV numbers.
4. Workstreams 2–7 in order, each with an ablation report in `reports/`.
5. DEV selection per target, gated on skill vs the unified ruler, sentinel ratio ≥ 1.5, and interval
   coverage in band including a top-tercile floor.
6. **One** TEST read, then the full trust pack.

---

## 9 · The document set

To resume, these are the files that carry state:

| File | Contains |
|---|---|
| `docs/reviews/2026-08-04_review.md` | The seven-part review: all findings with `file:line` evidence. **Its metrics are superseded by the Phase-1 data fix**; the findings are not |
| `docs/EXECUTION_PLAN.md` | Phase 1 checkbox plan, decisions D1–D8. Step 1 is 5/7; Steps 2–7 untouched |
| `reports/phase1_session_record.md` | Phase 1 instruction, decisions D5–D10, the three measurements that changed the plan, both mistakes |
| `reports/phase2_session_record.md` | Phase 2 instruction, decisions D11–D13, ground rule 1, open issues |
| `reports/phase2_trust_pack_record.md` | **This file** — instruction, D14–D15, and the complete open-issue register |
| `CHANGELOG.md` | Every change across both phases with the numbers that moved |
| `VERIFICATION.md` | How a third party re-runs each check, with tolerances |
| `reports/model_audit_2026-07-21.md` | The original audit that started this. Superseded metrics; findings stand |

Sending §5 (current state), §6 (open questions) and §7 (unresolved issues) of this file is enough to
resume without re-deriving anything.
