# AI4CM — Handoff

**Single self-contained document for resuming work.** Everything needed is inlined; no other file is
required. Share this one file.

**Generated:** 2026-08-04, updated 2026-08-05 (phase 7) · **Branch:** `model/excellence` @ `6c22009`
**`origin/main`** @ `6c22009` (**Phase 1 via PR #23, Phase 2a via PR #24 — both merged**)
**Suite:** 441 passing · **TEST (2025) gated reads: 0** — but see §0a, there are two
retrospective disclosure
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

## 0a · TEST-window accounting — read this before quoting any 2025 number

`experiments/test_access.log` records **0 gated reads**, and that is misleading on its own.
**Item 1f's one-ruler verification did evaluate the incumbent models on the 2025 window** —
all four families across three targets, n=156 each — and it did so *outside* the gate,
because the pipelines never route their fold construction through `require_test_access()`.
The gate was therefore never consulted and the ledger stayed at zero while a 2025
evaluation had in fact happened.

What is and is not true about it:

- It was **necessary**: the new-vs-old tier table exists to show how much the Phase-1 data
  fix moved every previously published figure, which cannot be shown without measuring the
  same window.
- **No selection was made from it.** No model was chosen, no hyperparameter set, no
  threshold moved on the strength of those numbers. They were reported, not acted on.
- It is **not** a clean-holdout result for the Phase-2 candidates, which do not exist yet.

**Policy from this point: no evaluation of any model on 2025 until the single final TEST
read**, which must go through `require_test_access()` so it lands in the ledger. All model
search and confirmation happens on TRAIN-internal rolling-origin folds and DEV (2024).

A retrospective entry recording this is appended to `experiments/test_access.log`. That file
is gitignored, so **this section is the durable record**.

### Second disclosure — the phase-3 "DEV" figures were DEV+TEST (found 2026-08-05)

Writing the experiments logger surfaced a defect in item 1f's own fix. `eval_start` set a
**floor** on the evaluation window and nothing set a ceiling, so pinned folds tiled forward
from `eval_start` to the **end of the series**. That is harmless for the 2025 benchmark,
whose window genuinely ends at the series end — and wrong for anything else.

Pinning to `DEV_START` therefore evaluated 2024-01-01 … **2025-08-06**: 418 target dates
where DEV has 262. So these previously reported "DEV" figures were computed over DEV *plus
the whole available holdout*:

| Figure, as reported | Reported | Actually measured over | Honest DEV-only |
|---|---:|---|---:|
| GBQuantile Revenues, DEV skill (phase 3) | 47.98% | 2024-01-01 … 2025-08-06, n=410 | **46.37%** |
| GBQuantile State budget balance, DEV skill (phase 3) | 32.20% | same | **33.81%** |

The correction is small in magnitude and that is not the point: the numbers were not
DEV numbers. **`eval_end` now caps the window**, four regression tests pin it
(`test_eval_end_caps_the_window` and neighbours), and both figures have been re-measured
DEV-only and logged as reproductions in `experiments/log.csv`.

Selection consequence: **none.** Those two figures were a fold-scheme sanity check, and no
model, hyperparameter or threshold was chosen from them. Every subsequent number in the
log is DEV-capped by construction.

This was caught by an assertion written into the logging script *because* the sealed-holdout
rule had just been tightened — `assert wins == {"dev"}` — not by review. Cheap guards on the
window are worth more than care.

---

## 1 · Where the work stands

| Phase | Status |
|---|---|
| Audit (`docs/reviews/2026-08-04_review.md`) | Complete. Findings valid; **all its metrics superseded** by the Phase-1 data fix |
| Phase 1 — data trust | **Merged to `origin/main`** via PR #23. Step 1 done; Steps 2–7 open |
| Phase 2a — yardstick + provenance | **Merged via PR #24.** Ground rule 1 and **all of item 1** complete: one persistence ruler per target across all four families |
| Phase 2b — modelling | Not started. **Item 2** (experiments log) is next, then workstreams 1–7 |

### Commits on `model/excellence` (11 ahead of `main`)

Phase-1 and Phase-2a are **merged to `origin/main`** via PR #24. Seven commits sit on top,
carried by **PR #25** (open, ready for review, not merged). Recent history:

```
9ba1144  WS2 infrastructure: quantile port, crossing-safe, h-gapped early stopping
9cd7e9b  WS4 robustness: Revenues scaling gain is drift-dependent in magnitude, not sign
98c93e1  docs: phase-10 session record
4b4a515  Published-forecast retention + realized scoring
250f501  WS4: target scaling. ratio-to-trailing-level wins Revenues by 25.7% on DEV
90213a2  Decisions 1-4: install catboost, record dual-probe and model-agreement rulings
33e1130  Demo item 7: CatBoost hooks behind an optional guard; docs/ROADMAP.md
bb173ae  Demo: Treasury HTML report, Streamlit polish, demo runbook
9228f58  Demo: forward forecast, model registry, plain-language insights, Forecast page
863f967  Merge Phase 2c (WS5 + probe study) into main   <-- PR #25 MERGED
68f2cda  Sentinel probe study: the negative result survives a nonlinear instrument
0c5d664  WS5: multivariate, leak-safe. The debt-ops hypothesis fails, for an instructive reason
246f9be  docs: HANDOFF phase-8 commits; resume -> WS5 (second reorder)
136ae76  docs: phase-8 session record; HANDOFF resume point -> workstream 2
2ab1c3e  WS3 step 2: wire the fiscal calendar into both families; ablate; it does NOT lift the flows
7b8da25  WS3 step 1: shared Georgian fiscal calendar module + Treasury sign-off artifact
efafa02  docs: HANDOFF phase-7 commits; resume -> WS3 (reordered ahead of WS2)
fa6dec6  docs: phase-7 session record; HANDOFF resume point -> workstream 2
db27cd4  Item 3 / workstream 1: absolute-error objectives for B_ML (L1 wins 17 of 18)
5f2990c  Item 2: append-only experiments log; fix the unbounded evaluation window it exposed
40d391b  docs: HANDOFF -- phase-6 commits, resume at item 2, and an honest TEST-window disclosure
6c22009  Merge pull request #24 from WBG-ITS-Innovation/model/excellence  (Phase 2a)
e9c775b  docs: one-ruler verification session record (item 1f complete)
87fc971  fix: one persistence ruler per target across all four families (item 1f)
e413cd1  docs: update HANDOFF with phase-5 commits; resume point -> item 1f
e7b3159  docs: yardstick-completion session record (items 1b, 1e, DATA_SEMANTICS)
94db732  docs: DATA_SEMANTICS.md -- negatives characterization + Treasury questions
db06426  feat: input by name + SHA-256; provenance in all four families (1e)
b7bccd7  fix: pin C_DL to TEST_START so it reports on the shared window (1b)
7c89e5d  docs: update HANDOFF with phase-4 commits and PR #23 merge
98a1da4  docs: integrity-consolidation session record (items 1c, 1d, PR #23)
904a520  fix: alias mae_seasonal_naive so it stops impersonating persistence (1d, A2)
154efbe  refactor: retire the duplicate integrity module; one implementation (1c, D7)
9f24b22  docs: single self-contained handoff
b1b7ae1  docs: yardstick session record
15fb6ee  fix: E_QUANTILE on business days + an honest stock path (yardstick, D12)
b69c060  docs: trust-pack session record + complete open-issue register
cc60073  docs: trust pack -- CHANGELOG, VERIFICATION, and honest stubs
c3c2fd8  docs: Phase-2 session record
489c9e0  feat: enforce TRAIN/DEV/TEST in code with a gated, logged TEST holdout
```

**Phase 1 is now on `origin/main`** via PR #23 (merged 2026-08-05). A direct push of the local
`main` was rejected as non-fast-forward -- `origin/main` had received PRs #21 and #22 -- so the
work went through `merge/phase1-trust` based on `origin/main`.

### Reference numbers

UNIFIED RULERS (2025 window, h=5 business days) — one number per target, verified
identical across all four families (spread 0.000000):

```
Revenues              83,534,152.85
Expenditure           83,839,124.43
State budget balance 189,930,653.98
```

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

**Workstream 2 — hyperparameter tuning.** Workstreams 3, 4 and 5, the sentinel-probe study,
the demo tranche and published-forecast scoring are complete.

> **Standing rule adopted 2026-08-05:** never state a test count in a commit message that was
> not produced by a `pytest` run from the repo root **in that same command** — and **check the
> exit code, not the tail**. Piping `pytest` to `tail` makes the pipeline's status that of
> `tail`, so an `&&` guard does not short-circuit on failure. This bit once: a commit claimed
> 441 while the suite was 440 passed + 1 failed (`9ba1144`, amended).
>
> **WS4 robustness (`9cd7e9b`):** Revenues' ratio advantage tracks the evaluation window's
> level drift at **corr +0.987** (+1.30% at 13.5% drift → +24.14% at 84.1%). A line fitted on
> TRAIN windows alone predicts +21.83% at DEV's 81.7% drift against a measured +25.73%, so the
> DEV gain is the relationship doing what it does, not an anomaly. **Quote the 55.92% DEV
> skill as conditional on a high-drift period** — enforced by `scaling_caveat` in the registry
> plus two tests. Adoption stands on the weaker claim that the advantage was positive in every
> window tested. Expenditure and the stock target are *hurt* by ratio and hurt more as drift
> rises, so drift alone does not determine whether it helps — an untested hypothesis
> (fluctuations scaling with level) is recorded for WS7.
>
> **WS4 landed.** Revenues uses `ratio`-to-trailing-level (DEV error 52,417,152 → 38,931,956,
> skill 40.65% → **55.92%**); Expenditure and the stock target stay raw. Implemented as an
> **estimator wrapper**, so no line of the prediction path changed and the unified ruler is
> bit-identical — asserted against the published 2025-definition constants.
>
> **CatBoost is installed** and in the pool for WS2, promoted nowhere.
>
> **Published forecasts are now retained and scored.** `forecasts/published/<issue_date>/` is
> tracked; the scorer refuses any date whose truth is not yet in the canonical data. Currently
> 15 pending / 0 scored, which is correct — every published date is still in the future.

> **A forward forecast now exists** (`backend/forward_forecast.py`): the next five business
> days beyond the data end, P10/P50/P90, no truth read, `test_window_touched: false`. It is
> the first non-backtesting code in the project. Gate verdicts are carried from DEV by
> `recipe_id` and never recomputed on forward dates.
>
> **Three recipes are registered as candidates** (`registry/recipes.json`), reconciled to
> `experiments/log.csv` by `run_id`. Both flow targets ship as **WITHHELD as a forecast**
> because the signal gate fails; the stock target is publishable. Nothing is approved and
> `validate_registry()` refuses to let anything claim otherwise.

> **WS4 touches the PREDICTION PATH** — it must invert the transform before
> `predictions_long.csv` is written, and that file's `origin_value`/`y_true` columns are what
> the unified persistence ruler is computed from. An error there fails silently. Verify the
> ruler is byte-identical before and after.

> **Second reorder, 2026-08-05: WS5 and WS4 now precede WS2.** Same tune-once reason as the
> first reorder. WS3 changed the feature set and WS5/WS4 will change it again (exogenous
> columns, target scaling); ~100 Optuna trials spent before those land would be spent on a
> recipe that no longer exists. WS2 tunes once, on the final recipe.
>
> The substantive reason WS5 goes first: **WS1 and WS3 have both failed to move the flow
> sentinel** (Revenues 1.226, Expenditure 1.088, threshold 1.50). WS5 is the last modelling
> lever on it, and the debt-operation lines are the specific hypothesis — `docs/DATA_SEMANTICS.md`
> §1 measured the 72 negative-`Revenues` days as netting in `Increase in liabilities`
> (corr 0.971) and `Domestic` (0.969).

> **Reorder, 2026-08-05: WS3 now precedes WS2.** Phase 7 established that the sentinel ratio is
> measured by a fixed Ridge probe **on the feature set**, so only WS3 (fiscal calendar) and WS5
> (multivariate) can move it — no objective, hyperparameter or ensemble work can. Optuna is
> expensive and should therefore be spent **once, on the winning feature set**, not on the
> current one and again afterwards. WS2 tunes on top of whatever WS3 leaves standing, still on
> **L1** (pinball at τ=0.5 *is* absolute error, so the objectives are already consistent).

Remaining, in order:

| # | Task | Notes |
|---|---|---|
| ~~1c~~ | ~~Retire the duplicate integrity module (D7)~~ | **DONE** `154efbe`. 649 -> 59 lines; `.update(legacy_report)` gone; 10 tests migrated not deleted; mutation-validated |
| ~~1d~~ | ~~Alias `mae_seasonal_naive` (A2)~~ | **DONE** `904a520`. 16/16 Dashboard fields; `NaN` + `seasonal_naive_degenerate` at season == horizon |
| ~~1b~~ | ~~Pin C_DL to `TEST_START`~~ | **DONE** `b7bccd7`. 7 folds -> 1, test block 2025-01-01..2025-08-06 |
| ~~1e~~ | ~~Input by explicit name + recorded SHA-256~~ | **DONE** `db06426`. `ls -t` gone; one shared provenance module for all four families; pinning fails closed |
| ~~1f~~ | ~~Backtest re-run + one-ruler verification~~ | **DONE** `87fc971`. Spread 0.000000 across families on all three targets |
| ~~2~~ | ~~Ground rule 2 — experiments log~~ | **DONE** `5f2990c`. 38 rows, integrity OK, `runs/` now tracked. Exposed the unbounded `eval_start` (see §0a, second disclosure). **"Never report an unlogged number" is now in force** |
| ~~3~~ | ~~Workstream 1 — L1 objectives for B_ML~~ | **DONE** `db27cd4`. L1 wins **17 of 18**; also added `eval_start`/`eval_end` to B_ML, whose yearly folds previously ran into the 2025 holdout on any default run |
| ~~4~~ | ~~Workstream 3 — fiscal calendar~~ | **DONE** `7b8da25`+`2ab1c3e`. MAE +1.1–6.3% on DEV across all three targets; **the flow sentinel did NOT move into signal territory** (Revenues 1.138→1.226, Expenditure 1.088→1.088, threshold 1.50). `docs/FISCAL_CALENDAR_SOURCES.md` ready — **T2 can be sent**. Winning subsets: Revenues A+B+C+D+E, Expenditure A+B+C+D, stock A+B+C+D |
| ~~4b~~ | ~~Workstream 3 — fiscal calendar (original row)~~ | Shared `backend/preprocessing/fiscal_calendar.py` consumed by B_ML *and* E_QUANTILE; `docs/FISCAL_CALENDAR_SOURCES.md` as the Treasury sign-off artifact so T2 can be sent; five feature groups ablated on TRAIN folds, one DEV confirmation per target | **Next.** The headline metric is the **sentinel ratio**, not MAE. Every entry cites rs.ge / matsne.gov.ge / mof.ge / nbg.gov.ge or is marked **UNVERIFIED** — never a fabricated citation. `calendar_version` (content hash) recorded on every experiment row |
| ~~5~~ | ~~Workstream 5 — multivariate~~ | **DONE** `0c5d664`. **The debt-ops hypothesis failed** — the 0.971 correlation is contemporaneous, and lagged realised values cannot anticipate a future auction. Nothing adopted for the flows; `cross` adopted for the stock (+0.42% DEV) |
| ~~5b~~ | ~~Sentinel probe study~~ | **DONE** `68f2cda`. Ridge's blind spot is real (pure interactions) but is **not** the one I hypothesised in WS3. Flows read 1.421 / 1.396 under a tree probe — still no signal. Default probe unchanged; dual-probe reporting **deferred to the user** |
| ~~5c~~ | ~~(original WS5 row)~~ | **Next.** The last remaining lever on the flow sentinel. Debt-ops block tested on its own first — it is the hypothesised mechanism for the unpredictable days. Lags ≥ 1 step, transforms and any top-K fit per fold only |
| ~~6~~ | ~~Workstream 4 — target scaling~~ | **DONE** `250f501`. Revenues → `ratio` (+25.73% DEV); others raw. Ruler bit-identical. **Signal unchanged** — Revenues still 1.2255 and still `withheld_as_forecast` |
| ~~6b~~ | ~~Published-forecast retention + scoring~~ | **DONE** `4b4a515`. First scoreable date 2025-08-07, once the data moves past it |
| ~~6c~~ | ~~(original WS4 row)~~ | raw vs asinh vs ratio-to-trailing-level, statistics fit per fold; stock keeps its delta path. `log1p` remains inapplicable while T1 is open (negatives) |
| 7 | Workstream 2 — LightGBM quantile + Optuna ~100 trials | Now **after** WS5/WS4, on the final recipe |
| 8 | Send **T2**; then the sentinel-probe question (shallow-tree probe vs Ridge on an identical feature set — a measurement study on its own merits, **not** a way to rescue a failing model) | The auction calendar is the top ask, and on present evidence more promising than anything left in the modelling backlog |
| 9 | Workstreams 6, 7 | 7 owns the top-tercile coverage fix (CQR) |
| 7 | Phase-1 Steps 4/5/7 — ops P0, Agent contract + validator, cleanup | Deferred since Phase 1 |

**The central finding to carry forward:** the two flow targets have **no detectable feature
signal** (sentinel ratio 1.07–1.14 vs 1.50 required), and the sentinel probes the *feature set*
with a fixed Ridge — so no objective, hyperparameter or ensemble work can move it. Only
workstreams 3 and 5 can. Meanwhile `LightGBM_L1` reports 36.65% DEV skill on Revenues: real
error reduction against a spiky ruler, but not a forecast. Never quote one number without the
other.

**Four attempts have now failed to find signal in the flows** (WS1 objectives, WS3 calendar,
WS5 multivariate, plus a three-instrument probe study): Revenues 1.167–1.421, Expenditure
1.071–1.396, threshold 1.50. `reports/ws5_multivariate.md` §6 states the option that deserves
stating: a 5-business-day-ahead forecast of daily Treasury flows may not exist in this data
beyond central tendency, and the negative results are evidence *for* that. The one untried
input that could overturn it is Treasury's **forward** auction calendar — the only candidate
knowable in advance, targeting the specific unpredictable days. That is now the highest-value
item in the backlog and it is not a modelling task.

**And keep the L1 claim exact.** L1 wins 17 of 18 *paired* comparisons against its own twin.
It does **not** supply the best model on every target: on Expenditure the DEV-best is the
squared-error `HistGBDT` at **30.13%**, ahead of `LightGBM_L1`'s 28.64% — the same single cell
where the paired comparison also went against L1 (−4.44%). Per-target selection (WS7) must
therefore keep L2 candidates in the pool rather than assume L1 dominates.

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
| **OQ1** | **Treasury confirmation on negative flow values.** Characterized in `docs/DATA_SEMANTICS.md`: 72 business days, driven by `Increase in liabilities`/`Domestic` netting (corr 0.971/0.969). **Question T1 sent; answer OUTSTANDING.** Proceeding under Q1's interim policy: legitimate signed flows | Confirms or overturns workstream 4's transform set (raw / asinh / ratio-to-trailing-level; log1p inapplicable) |
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
| ~~`integrity_report.update(legacy_report)` lets a duplicate overwrite the shared one~~ — **fixed** `154efbe` |
| ~~`mae_seasonal_naive` identical to `mae_persistence` at h=5~~ — **fixed** `904a520` |
| ~~C_DL unpinned: reported +10.84% while being −5.19% on the 2025 window~~ — **fixed** `b7bccd7` |

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
| ~~Input selected by **mtime**, not name + hash~~ — **fixed** `db06426` |
| `RUN_DATE` timezone unpinned |
| No minimum-evaluation-points enforcement (A_STAT published a verdict off 10 points) |
| `data_preflight.py` not wired into the batch path |
| Exit code 0 when every family is withheld |
| ~~Provenance for 1 of 4 families; no git SHA, versions or seeds~~ — **fixed** `db06426` |

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
