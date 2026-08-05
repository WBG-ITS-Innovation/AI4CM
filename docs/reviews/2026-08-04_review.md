# AI4CM — State Assessment, Audits and Readiness Review

**Date:** 2026-08-04
**Branch:** `feat/calendar-features` @ `4c521d0`
**Data file under audit:** `backend/data/processed/master_daily_clean_treasury.csv` (2015-01-05 … 2025-08-06)
**Test suite at start:** 158 passed · **at end:** 193 passed
**Scope of code changes made:** five new test files under `backend/tests/`. No production code was modified.

## How to read this document

This is a transcript-style record of a seven-part review. Each section reproduces the request
verbatim, then the findings that answered it. Every quantitative claim was produced by running
the code — pipelines, tests and one-off measurement scripts — rather than by reading it. Where a
measurement contradicted an initial hypothesis, both are recorded; those corrections are marked.

Mutation testing was used to validate new tests: the fix under test was temporarily reverted, the
test was confirmed to fail, and the revert was undone via `git checkout`. Every mutation is
recorded with its outcome.

## Contents

1. [Repository state assessment](#1--repository-state-assessment)
2. [Dedicated leakage audit](#2--dedicated-leakage-audit)
3. [E_QUANTILE and prediction-interval review](#3--e_quantile-and-prediction-interval-review)
4. [Full backtest run and trust verification](#4--full-backtest-run-and-trust-verification)
5. [Quality gates, test suite, and five new regression tests](#5--quality-gates-test-suite-and-five-new-regression-tests)
6. [Agent artifact contract and validation plan](#6--agent-artifact-contract-and-validation-plan)
7. [Production readiness for automated daily runs](#7--production-readiness-for-automated-daily-runs)
8. [Appendix: files added, and cross-cutting themes](#8--appendix)

---

# 1 · Repository state assessment

## Prompt

> Read the entire AI4CM repo. Do not change anything yet. Produce a written state assessment
> covering: (1) the full pipeline from raw data to SUMMARY.json, predictions_long.csv, and
> leaderboards; (2) every place a baseline is computed — confirm whether the h-step persistence
> benchmark is a single shared implementation or duplicated across model families; (3) the current
> status of the M-series fixes (quality gates, E_QUANTILE, ResidualRF intervals, overfitting caps,
> leakage-sentinel semantics) — verify each fix is actually present in the code, not just in commit
> messages; (4) anything that looks unfinished, dead, or inconsistent. Give me a plain-language
> verdict on what works, what's fragile, and what's broken, with file/line references.

## Output

18.4k lines of tracked Python read in full. Test suite: **158 passed in 28s**.

### 1.1 The pipeline, raw data → outputs

**Stage A — ingestion (Excel → CSV).** `backend/preprocessing/preprocess.py` parses
`Balance_by_Day_*.xlsx` into three variants (`raw` / `clean_conservative` / `clean_treasury`),
driven by `backend/run_preprocess.py` reading `PP_*` env vars, launched from
`frontend/pages/00_Data_Preprocessing.py` via `frontend/backend_bridge.py`. Output lands in
`backend/data/processed/master_daily_*.csv`.

The shipped CSV is **7-day calendar-daily**, not business-daily — 3,867 rows with 552–553 rows for
every weekday including Saturday and Sunday. This matters in §1.2.

**Stage B — the four families.** Two entry paths, same runners:

- Interactive: `frontend/pages/00_Lab.py` → `launch_backend()` → subprocess with `TG_*` env vars → `frontend/runs/run_<...>/`
- Batch: `scripts/run_daily_forecast.sh` → each family into `backend/forecast_runs/<date>/<family>/`

| Family | Runner → pipeline | Fold scheme | Index |
|---|---|---|---|
| A_STAT | `run_a_stat.py` (self-contained) | yearly, rolling origin per target (`:286-321`) | `freq="B"` (`:35`) |
| B_ML | `run_b_ml_univariate.py` → `b_ml_pipeline.py` | yearly, fit-once-per-fold (`:640-832`) | `freq="B"` (`:145`) |
| C_DL | `run_c_dl_quick_univariate.py` → `c_dl_pipeline.py` | yearly on label dates (`:727`) | `freq="B"` (`:399`) |
| E_QUANTILE | `run_e_quantile_daily_univariate.py` → `e_quantile_daily_pipeline.py` | `_time_folds`, blocks of size `h` tiled backward (`:46-92`) | **raw CSV index — no reindex** |

Each family writes `predictions_long.csv` (schema `date, target_date, origin_date, origin_value,
y_true, y_pred, y_lo, y_hi, model, horizon, split_id`), `metrics_long.csv`, `leaderboard.csv`,
plots, and `artifacts/integrity_report.json`.

**Stage C — reporting.** `scripts/daily_summary.py` walks each family dir, reads
`predictions_long.csv` + leaderboard + integrity report, runs two *independent* checks
(`origin_date >= target_date`; `detect_lagged_copy`), computes gate reasons at `:83-116`, and writes
`SUMMARY.txt` + `SUMMARY.json`. In `--mode backtest`, `scripts/backtest_report.py` then reads
`SUMMARY.json` into `BACKTEST_REPORT.md`.

Structurally this layer is the best part of the repo. The gating logic is honest: a gate-failing
family is displayed as `WITHHELD — <reasons>, not usable; <model> for diagnosis only` (`:376-384`),
never dropped. `gate_passed` is tri-state — `None` when never verified, so an unverified family
cannot masquerade as a pass (`:289-292`).

One gap: `main()` returns non-zero **only** when a family produced no output (`:458-461`).
Quality-gate failures and leakage flags exit 0, so `run_daily_forecast.sh` reports success on a run
where every family was withheld. The module docstring claims otherwise.

### 1.2 Baselines — the central finding

**The h-step persistence: one function, four different measurements.**

The shared ruler is `backend/forecast_integrity.py:260` `compute_persistence_baseline`. All four
families call it: `run_a_stat.py:341`, `b_ml_pipeline.py:909` and `:975`, `c_dl_pipeline.py:889`,
`e_quantile_daily_pipeline.py:475`. `backend/tests/test_unified_baseline.py` enforces this by
grepping each family's source.

But a shared function is not a shared measurement. From the real 2026-07-30 run:

| Family | n (1 model) | Target-date window | Persistence MAE |
|---|---:|---|---:|
| A_STAT | 156 | 2025-01-01 … 2025-08-06 | **60,976,736.58** |
| B_ML | 156 | 2025-01-01 … 2025-08-06 | **60,976,736.58** |
| E_QUANTILE | 205 | 2025-01-10 … 2025-08-06 | **65,888,163.54** |
| C_DL | 1,722 | **2019-01-01** … 2025-08-06 | **52,957,744.22** |

A_STAT and B_ML agree to the cent — audit finding C-2 is genuinely fixed *for those two*. The other
two measure different things:

- **C_DL never restricts to the reporting window.** It folds over every available year, so its
  `skill_pct = 10.84%` is an average over 2019–2025, not the 2025 holdout.
- **E_QUANTILE's horizon is a different unit.** It is the only family that does not reindex to
  `freq="B"`, so `h=5` means 5 rows of a 7-day calendar index. Verified: every row has
  `target_date - origin_date == 5` **calendar** days, versus 7 calendar days (5 business days) for
  B_ML. 57 of its 205 target dates (28%) are weekends; B_ML has 0. Weekend skill 41.9% vs weekday
  51.0%, so weekends are not the inflation source — the problem is horizon semantics and window.

**Duplication: the shared function is not the one that writes the number.**

`compute_persistence_baseline` has a second implementation —
`backend/preprocessing/integrity.py:480` `compute_persistence_baseline_from_origin`, called from
`compute_integrity_report:581`. In B_ML:

```python
# b_ml_pipeline.py:975  — shared ruler
persistence_baseline = compute_persistence_baseline(pred_model)
...
# b_ml_pipeline.py:1014-1015
legacy_report = compute_integrity_report(pred_long, s, cfg.horizon, best_model, ...)
integrity_report.update(legacy_report)     # ← legacy OVERWRITES the shared values
```

`.update()` means the number that actually reaches `integrity_report.json` — and therefore the
Dashboard, the summary, and the backtest report — comes from the **duplicate**. The two agree
arithmetically today, so nothing is currently wrong; but `test_unified_baseline.py`'s "one ruler"
guarantee does not cover the path that produces the artifact.

A **third** persistence lives in `preprocessing/integrity.py:124-175` `compute_baselines`, which
also emits `mae_seasonal_naive` with `season_steps = 5` hardcoded at `:147`. At the production
horizon `h=5` this makes seasonal-naive *identical to persistence*:

```
h=3: persistence == seasonal_naive ? False
h=5: persistence == seasonal_naive ? True     ← production horizon
h=7: persistence == seasonal_naive ? False
```

Real artifact confirms `mae_seasonal_naive: 60976736.58082051` == `mae_persistence`. It is a
redundant metric that looks like independent corroboration.

A **fourth** notion is the lag-1 baseline inside `forecast_integrity.py:414-417`
`detect_lagged_copy`, which is `h=1` regardless of the configured horizon.

**Ops/Treasury baseline: triplicated with different algorithms.** Three live implementations,
materially different: `b_ml_pipeline.py:199` (3y same-month rolling mean), `run_a_stat.py:49` (3y
annual mean × month share), `c_dl_pipeline.py:192`. The dashed "Ops baseline" line the Dashboard
plots is therefore a different series depending on which family produced the run. And it is inert
where it matters: `b_ml_pipeline.evaluate_block` (`:374`) takes `ops_series` and never uses it — the
`MAE_skill_vs_Ops` column at `:424` is hardcoded `np.nan`, confirmed empty in the real
`metrics_long.csv`.

### 1.3 M-series fixes — verified against code, not commits

B_ML was re-run fresh with current code because the **stored artifacts in
`backend/forecast_runs/` predate the M-4 and M-5 commits** (runs dated 07-29/30/31; `ad327ce` M-5
and `f9a0324` M-4 both landed 07-31). Anyone reading those folders as current state would be misled.

| Fix | Present? | Evidence |
|---|---|---|
| **Quality gates** | ✅ per-family | B_ML `:1193-1219` (skill ≥ 5%, `FAILED_QUALITY`, outputs still written); A_STAT `:350-351`; C_DL `:909-920`; E_QUANTILE `quantile_quality_gate:248-270` — skill **and** coverage in [0.70, 0.90]. Summary-level gate at `daily_summary.py:83-116` |
| **E_QUANTILE (M-2)** | ✅ | `eval_start` pinning (`:305-317`), wired at `run_daily_forecast.sh:108`. Real run: n=205 per model, not 10. Per-model skill/coverage (`:470-495`). Best model chosen among gate-passers only (`:502-504`) — verified: ResidualRF had the *lower* MAE (33.25M vs 33.96M) but failed coverage at 69.8%, so GBQuantile was correctly crowned |
| **ResidualRF intervals (M-3)** | ✅ | OOB residuals via `oob_score=True` (`:198-246`), documented in-sample fallback, monotonicity enforced. Real coverage 69.8% (was 26.5%). Still marginally outside the gate band — the gate says so rather than hiding it |
| **Overfitting caps (M-4)** | ⚠️ present, partly wired | `MIN_SAMPLES_PER_LEAF=5` on RF/ExtraTrees (`:308-315`), `OVERFIT_GATE_RATIO=3.0`, `select_best_model()` (`:335-367`), zero-train-MAE → `inf` (`:754-756`). Works — fresh run logged `[M-4] Excluded from best-model selection (val/train ratio > 3.0): LightGBM (8.82), XGBoost (3.01)`, and ExtraTrees no longer reaches train_MAE 0.00 (now 14.8M). But see the gap below |
| **C_DL target scaling (C-1)** | ✅ | `log1p_std` with fit-slice-only stats and matching inverse (`:778-814`). MLP MAE 47.2M vs the audit's 73.4M; skill −38.65% → +10.84% |
| **Leakage-sentinel semantics (M-5)** | ✅ and correct | `preprocessing/integrity.py:213-318` `signal_sentinel` renamed, `leakage_warning` hardwired `False` with reasoning documented, `leakage_sentinel` kept as deprecated alias. Train-statistics-only standardisation. Held-out split with an `h`-sized gap at `b_ml_pipeline.py:1054-1075` — the substantive half of the fix. Summary gates on the accurate reason (`daily_summary.py:110-113`) |

**What M-5's fix actually revealed** — old (stale) artifact vs fresh run, same data:

| | stored 07-30 | current code |
|---|---|---|
| `leakage_warning` | `true` | `false` |
| `shuffled_to_normal_ratio` | 0.83 | 1.13 |
| `signal_detected` | *(absent)* | `false` |
| `signal_verdict` | *(absent)* | `"WEAK SIGNAL: shuffling the targets barely hurt (ratio 1.13 < 1.50 required)"` |

B_ML still fails the gate — now for the true reason ("no signal beyond shuffled targets") rather
than a false leakage alarm.

**M-4's real gap: the leaderboard and the plots ignore the gate.** `select_best_model()` is used for
the integrity report, but `best_model` is recomputed without it for plotting at
`b_ml_pipeline.py:1299-1303`:

```python
best_model = None
if glb is not None and len(glb) > 0:
    _trained = glb[~glb["model"].str.contains("baseline", case=False, na=False)]
    best_model = _trained.iloc[0]["model"]      # ← overfit_ratios not consulted
```

The fresh run makes the divergence concrete:

```
leaderboard.csv:  rank 1 = XGBoost      MAE 38,554,577   ← in overfit_excluded_models
                  rank 2 = RandomForest MAE 38,565,591
integrity_report: best_model = RandomForest
```

So `*_overlay_top.png` and `*_monthly_bars_top_vs_ops.png` are drawn for **XGBoost**, the
disqualified model, while the report names RandomForest. `leaderboard.csv` ranks XGBoost first with
no exclusion marker — and that CSV feeds the Dashboard Leaderboard tab and Compare page.
`daily_summary.py` is safe (it prefers `report["best_model"]`, fixed by `c69a991`), but nothing else
is. `overfit_ratios` / `overfit_excluded_models` are written to JSON and read by **zero** consumers.

### 1.4 Unfinished, dead, inconsistent

**Dead code (~1,950 lines):**

| File | LOC | Status |
|---|---:|---|
| `backend/a_stat_models_pipeline.py` | 713 | Referenced only by docs. A second, divergent A_STAT that refits ETS per test point |
| `backend/overfitting_check.py` | 172 | `check_overfitting()` called only by its own test. B_ML has separate inline logic |
| `backend/models/registry.py` + `backend/app/app.py` + `backend/core/db.py` | ~870 | A parallel Dash demo stack with its own 6 models. `_simple_metrics` (`:44-50`) reports **in-sample** MAE/RMSE as "a sanity metric" — no baseline, no holdout. Nothing imports it |
| `backend/make_ml_heatmaps.py`, `backend/make_weekly_from_daily_stat.py` | 202 | Standalone, unreferenced |

**`evaluation_windows.py` is documentation, not enforcement.** It defines TRAIN/DEV/TEST with a
strong discipline statement — "TEST … never used to choose anything … the number of such
consultations should stay at zero." No pipeline imports it. Its only consumer is `describe_split()`
for a prose line in `backtest_report.py:99`. `eval_start_for()` and `window_for()` have zero
callers. Meanwhile the one family that does pin a window pins it to `TEST_START`:
`eval_start: "2025-01-01"` hardcoded in `run_daily_forecast.sh:108`.

**Contradictory shift flags (latent).** The legacy and new `is_lag0_issue` formulas disagree, and
legacy wins the merge. Constructed and verified:

```
=== when best_shift == -h (h=5) ===
 new: best_shift=-5  is_persistence_like=True   is_lag0_issue=False
 old: best_shift=-5                             is_lag0_issue=True   ← overwrites
 merged -> {'is_persistence_like': True, 'is_lag0_issue': True,
            'shift_interpretation': 'Persistence-like (best_shift≈-h=-5, ...)'}
```

Both flags true simultaneously. `b_ml_pipeline.py:1182-1185` branches on `is_lag0_issue` first, so a
genuine persistence-like result is reported as `⚠️ MISSING LAG_0 DETECTED` and the operator is told
to add a feature that is already there.

**M-1 was fixed in the summary but not in the UI.** `daily_summary.py:67-80` falls back to
`integrity_*.json` to find C_DL's oddly-named report. The Dashboard does not —
`frontend/pages/01_Dashboard.py:204-206` and `:813-815` only try
`artifacts/integrity_report.json`, and C_DL writes `artifacts/integrity_<Target>_h<H>.json`
(`c_dl_pipeline.py:932-934`). Same for the Lab's post-run check at
`frontend/pages/00_Lab.py:669-674`. **A C_DL run launched from the Lab shows no quality gate at
all.**

**Other confirmed items:**

- **C_DL alignment is never checked.** `"alignment_ok": True` is a literal at `c_dl_pipeline.py:905`.
- **C_DL master leaderboard drops the baseline row** (`:968-973` rebuilds from `metr`).
- **E_QUANTILE multivariate leakage (audit m-5) unfixed and reachable.** `:134` `exog.ffill().bfill()`
  pulls future values backward; `:135-138` selects top-K features by correlation over the whole series.
  Reachable via `run_e_quantile_daily_multivariate.py:29`.
- **A_STAT still has no seed** (audit m-1) — zero `seed`/`random_state` occurrences in the file.
- **A_STAT's C-3 fix traded fit-once for a full refit per origin.** `run_a_stat.py:299` calls `_fc`
  inside the per-target loop — 156 SARIMAX/STL fits per fold where the audit's check #1 expected one.
- **A_STAT's integrity report is a stub** — 7 keys, no alignment check, no shift diagnostic, no sentinel.
- **`MAPE` is garbage for flow targets.** Real value `942150668337156.5`, from dividing by
  `max(|y_true|, 1e-9)` on a series with true zeros (`b_ml_pipeline.py:396`). Surfaced in the Dashboard.
- **`backend/forecast_runs/2026-07-31/` is a half-finished run** — no `SUMMARY.json`, no C_DL. No marker.
- **`backend/LEAKAGE_AUDIT.md` is stale and now wrong.** Verdict "PASS — no data leakage found"; its
  A_STAT table asserts `origin_date = train_end` at line 299, which the C-3 fix changed.
- **`.venv` directories and `backend/.env` are committed**, contradicting the README's own warning.

### 1.5 Verdict

**What works.** The reporting and gating layer is genuinely trustworthy: it withholds failing
families instead of dropping them, distinguishes "failed" from "never verified", and runs two checks
independent of the pipelines' self-assessment. Re-running it on the 07-30 artifacts with current code
correctly downgraded C_DL from `PASSED` to `WITHHELD — forecast is persistence-like`. Five of six
M-series fixes are real and do what the commits claim. A_STAT and B_ML agree on the persistence
baseline to the cent across two independently-written pipelines. M-4's capacity floors changed real
outcomes, not just logs.

**What's fragile.** "One shared ruler" is true of the function and false of the measurement — four
families, four windows, and E_QUANTILE measuring a 5-calendar-day horizon while everyone else
measures 5 business days. M-4 is wired into the report but not into anything a user looks at.
`integrity_report.update(legacy_report)` at `b_ml_pipeline.py:1015` is the single most load-bearing
fragile line in the repo. The DEV/TEST discipline is a comment. Stale artifacts in `forecast_runs/`
disagree with current code and carry no version marker.

**What's broken.** C_DL has no alignment validation. The Lab and Dashboard cannot find C_DL's
integrity report. `mae_seasonal_naive` is identical to `mae_persistence` at the production horizon.
E_QUANTILE multivariate leaks. `MAE_skill_vs_Ops` is permanently NaN and `MAPE` is ~9.4e14. ~1,950
lines of dead code including a second A_STAT and a registry reporting in-sample metrics.

**Suggested sequence.** (1) Force one evaluation window on all four families. (2) Delete
`integrity_report.update(legacy_report)`; retire one integrity module. (3) Route
`select_best_model()` through the leaderboard and plots. (4) Make `evaluation_windows` enforcing.
(5) Give C_DL a real alignment check and a discoverable report name. (6) Delete dead files, refresh
`LEAKAGE_AUDIT.md`. Items 1 and 2 decide whether the numbers in `BACKTEST_REPORT.md` can be defended
in a room.

---

# 2 · Dedicated leakage audit

## Prompt

> Do a dedicated leakage audit. For every model family, trace exactly what data is available at
> training time vs prediction time for each forecast horizon. Check: feature construction (including
> dom/bdom/bdom_rev calendar features), any rolling/lag features, target encoding,
> scaling/normalization fit points, and train/test split logic in backend/evaluation_windows.py.
> Confirm the leakage sentinel's semantics are correct (a sentinel that SHOULD fail must fail, and
> for the right reason — leakage, no-signal, and persistence-like behavior must be flagged
> separately). Run the sentinel tests and paste the actual output. Report findings before proposing
> any fix.

## Output

**Method.** For each family the criterion was: *for a prediction with origin `o` and target `τ`, does
any data dated after `o` influence the prediction?* That covers three channels — feature values,
training labels, and fitted transform parameters. Positive controls were then run: injected leakage,
injected no-signal, injected persistence.

### 2.1 Information-set trace, per family

| Family | Features at origin | Training labels | Transform fit | Verdict |
|---|---|---|---|---|
| **A_STAT** | `y_hist = y_all.iloc[:origin_pos+1]` — data ≤ origin only (`:295`) | refit per origin; no labels beyond origin | none | **Clean.** Strictest of the four |
| **B_ML** | backward-only lags/rolling; built on `s[≤ last_origin]` (`:702-706`) | `s_train_full.shift(-h)` on the **truncated** train slice → max label = `train_end` (`:684`) | Scaler+Imputer inside Pipeline, fit on `X_tr_fit` (`:723`) | **Clean except a 4-prediction fold boundary** |
| **C_DL** | sequences from `F`; train/test selected by **label date** (`:731-732`) | `tr_mask = (ld <= tr_end)` → max label = `tr_end` by construction | feature scaler on `X_fit` (`:772`); target `ty_mu/ty_sd` from `y_fit_log` (`:799-800`) | **Clean** on labels and scaling |
| **E_QUANTILE** | backward-only lags/rolling (`:121-125`) | `y_target` built on the **full** series, then sliced positionally (`:345`) → labels reach `h-1` past the split | none (trees) | **No embargo** |

The decisive difference is one line. B_ML applies `shift(-h)` *after* truncating to `train_end`, so
trailing rows become NaN and drop out — a gap appears for free. E_QUANTILE builds the label column on
the whole series first and slices by position afterward.

```
E_QUANTILE  predictions whose fold-model was trained on labels dated AFTER the prediction's own origin:
            168/210 (80%)  folds: 42
B_ML        4/156 (3%) (confined to the fold boundary: origins that fall inside the training year)
```

**Measured impact: none.** Re-running E_QUANTILE on real data with a proper embargo:

```
folds=41  h=5

AS SHIPPED  (train = [0:tr_end), no embargo)
   n=205  P50 MAE=33,964,365  persistence MAE=65,888,164  skill=48.45%
WITH EMBARGO (train = [0:tr_end-4), all train labels <= origin)
   n=205  P50 MAE=33,808,751  persistence MAE=65,888,164  skill=48.69%
```

The as-shipped figure reproduces the stored artifact to the dollar, so the harness is sound.
Removing the leakage makes the model *marginally better*. **E_QUANTILE's 48.45% is not a leakage
artifact.** Inspection alone would have called this a serious leak; the measurement says otherwise.

### 2.2 Feature construction

**Lag and rolling features — clean.** `lag_window_features` (`b_ml_pipeline.py:221-238`): `shift(L)`
for L≥1, and `rolling(W, min_periods=1).mean().shift(1)` — the trailing `shift(1)` makes the window
end at `t-1`. E_QUANTILE (`:121-125`) does the same and additionally shifts `roll_std`. No centred
windows, no negative shifts anywhere. `lag_0` is added only when delta modelling is active
(`:264-267`), which is legitimate — it is the value *at* the origin.

The comment at `b_ml_pipeline.py:697-701` claiming that building on the full series and slicing is
equivalent to building on `s[:origin]` is **correct**, precisely because every operator is
backward-only.

**Calendar features `dom`, `bdom`, `bdom_rev`:**

```python
_s = idx.to_series()
df["dom"]      = idx.day
df["bdom"]     = _s.groupby([idx.year, idx.month]).cumcount() + 1
df["bdom_rev"] = df.groupby([idx.year, idx.month])["bdom"].transform("max") - df["bdom"]
```

`dom` and `bdom` are deterministic functions of the date — **no leakage**. `bdom_rev` needs the
month's business-day count, also a pure calendar fact, but the implementation derives it from
`transform("max")` over the **observed index**:

```
  2025-06: rows in index=21  true business days=21  max(bdom)=21  bdom_rev at month's last row=0
  2025-07: rows in index=23  true business days=23  max(bdom)=23  bdom_rev at month's last row=0
  2025-08: rows in index= 4  true business days=21  max(bdom)= 4  bdom_rev at month's last row=0   <-- TRUNCATED
```

**Not target leakage** — no future `y` is involved. It is a train/serve skew: for the final partial
month the feature says "0 business days remain" when 17 actually do, so those rows are mislabelled as
month-end. Given the Phase 3 commit credits these features with "+14pp dev skill" and month-end is
the dominant flow driver, the last month of every run is systematically mis-featured.

**Target encoding — none exists.** No mean/target/likelihood encoding anywhere. B_ML one-hots `dow`
(`:172`); E_QUANTILE passes raw ordinals `dow, dom, week, month, year` (`:94-101`). No
target-encoding leakage channel. (`year` as an ordinal is a hazard for tree extrapolation, not leakage.)

**Scaling / normalisation fit points — all correct:**

| Where | Fit on | Verdict |
|---|---|---|
| B_ML `StandardScaler` + `SimpleImputer` | inside sklearn `Pipeline`, fit in `estimator.fit(X_tr_fit, y_tr_fit)` | ✅ train-only |
| C_DL feature scaler | `fit_feature_scaler(X_fit)` (`:772`) | ✅ fit-slice only |
| C_DL target `log1p_std` | `ty_mu/ty_sd` from `y_fit_log` (`:799-800`) | ✅ fit-slice only |
| `signal_sentinel` | `mu/sigma` from `Xtr` only (`integrity.py:269-273`) | ✅ train-only |
| E_QUANTILE / A_STAT | no scaling | ✅ n/a |

This is the cleanest part of the codebase. One asymmetry remains (audit m-4): training imputes with
the pipeline median, inference forces `fillna(0.0)` (`b_ml_pipeline.py:778`).

**Validation splits have no horizon gap:**

```
X_tr_fit = X_all_fold.iloc[:-n_val] ; X_tr_val = X_all_fold.iloc[-n_val:]   -> gap = 0
Compare the M-5 sentinel (b_ml_pipeline.py:1064): gap = max(int(cfg.horizon), 1)  -> gap = 5
```

The M-5 fix gave the *sentinel* a horizon gap and left the *model's own validation split* ungapped.
Everything derived from it inherits the overlap: `conformal_radius`, `val_MAE`, and M-4's overfit
ratio. Measured consequence:

```
train rows=2588  n_val=517  gate=3.0

model          ratio (gap=0, as shipped)   ratio (gap=h-1=4)  gate decision change
------------------------------------------------------------------------------------
Ridge                               1.52                1.52
Lasso                               1.52                1.52
ElasticNet                          1.53                1.52
RandomForest                        2.28                2.28
ExtraTrees                          2.16                2.16
HistGBDT                            2.95                2.94
XGBoost                             3.01                2.98  *** FLIPS: excluded -> eligible
LightGBM                            8.82                8.99
```

XGBoost's disqualification turns on a 4-row change to the split. It is also the model ranked **#1** on
`leaderboard.csv` and the one whose plots are drawn.

### 2.3 Pre-split leakage — the most serious finding

`clean_treasury` — the variant the production run consumes — bakes whole-series statistics into the
CSV *before any split exists*, so no pipeline check can see it.

**Imputation**, `preprocess.py:494-503`:

```python
if len(dow_values) >= weekday_weeks:
    ref = dow_values.tail(weekday_weeks).median()
```

`dow_values` spans 2015–2025. `.tail(8)` takes the **last 8 occurrences of that weekday in the entire
dataset** — i.e. 2025 — and uses them to fill a gap anywhere, including 2016. The docstring one line
above says *"median of past N weeks of same weekday"*. The implementation has no notion of "past
relative to this row".

**Clipping**, `preprocess.py:524-534`: `median` and `mad` are computed over every occurrence of that
weekday across the full series — **including the 2025 test window** — then applied to clip the whole
series.

Measured on real data (fresh run of current code vs `master_daily_raw.csv`):

```
BUSINESS DAYS only (n=2763), Revenues, clean_treasury vs raw:
  imputed (raw NaN -> filled with whole-series weekday median): 118 (4.3%)
  clipped (altered by whole-series weekday MAD threshold)     : 105 (3.8%)
  TOTAL touched by whole-series statistics                    : 223 (8.1%)
  of which fall in the 2025 TEST window                       : 20
  median |clip change|=118,342,253  max=1,533,480,847
```

The median clipping change is **118M — roughly 3× the best model's MAE (38.6M)**.

**The committed production CSV is worse than current code produces:**

```
committed master_daily_clean_treasury.csv
   weekend Revenues: NaN=   8  nonzero=1095
FRESH run, current code
   weekend Revenues: NaN=   0  nonzero=   0
```

Weekends are genuinely unobserved (NaN in `master_daily_raw.csv`: 1,103 of 1,104). Current code zeroes
them. **The committed CSV contains 1,095 fabricated non-zero weekend Revenues values** (mean 66.8M) —
produced by an older code path and never regenerated. `run_daily_forecast.sh:60` selects that file by
modification time, and E_QUANTILE draws 28% of its evaluation targets from those values.

### 2.4 Split logic in `evaluation_windows.py`

```
  2010-06-01: window_for=train  TRAIN.contains=False DEV.contains=False TEST.contains=False   <-- disagree
  2015-01-05: window_for=train  TRAIN.contains=True  DEV.contains=False TEST.contains=False
  2023-12-31: window_for=train  TRAIN.contains=True  DEV.contains=False TEST.contains=False
  2024-01-01: window_for=dev    TRAIN.contains=False DEV.contains=True  TEST.contains=False
  2025-01-01: window_for=test   TRAIN.contains=False DEV.contains=False TEST.contains=True

  eval_start_for('tuning')=2024-01-01   eval_start_for('report')=2025-01-01

=== pipelines importing evaluation_windows ===
  (no output — no pipeline imports it)
```

1. **Not wired in.** No pipeline imports the module. Only `describe_split()` is used, for prose.
2. **`window_for` disagrees with `TRAIN.contains`** for dates before `TRAIN_START`.
3. **The only hardcoded evaluation window points at TEST** (`run_daily_forecast.sh:108`), against a
   docstring saying TEST is "LOCKED … never used to choose anything".

A related non-leakage asymmetry: **A_STAT never uses `tr_end` to cap its fit** (`:295`), so a July
2025 origin trains on data through July 2025 while B_ML's model is frozen at 2024-12-31. Both
defensible, but different information sets reported in one table.

### 2.5 Sentinel semantics

**The three concepts are correctly separated — the leakage slot is now empty.**

```
=== callers of check_feature_leakage ===
backend/forecast_integrity.py:105:def check_feature_leakage(       <- definition
backend/preprocessing/integrity.py:240:    (check_feature_leakage), alignment validation, ...   <- a docstring
backend/tests/test_forecast_integrity_multi_horizon.py:21,171,172   <- tests only
```

`check_feature_leakage` has **zero production callers**. The only live leakage check is
`summary_leakage_check` (`daily_summary.py:184`), which tests `origin_date >= target_date`.
`pipeline_leakage` reads `leakage_warning`, which M-5 hardwired to `False`. So the system's leakage
verdict is a constant.

**Test output — all green:**

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_signal_sentinel_semantics.py \
                                       backend/tests/test_leakage_real_data.py -v

platform darwin -- Python 3.13.11, pytest-9.1.1, pluggy-1.6.0
collected 13 items

test_signal_sentinel_semantics.py::test_signal_detected_when_features_predict_target PASSED [  7%]
test_signal_sentinel_semantics.py::test_no_signal_when_features_are_noise            PASSED [ 15%]
test_signal_sentinel_semantics.py::test_sentinel_never_claims_leakage                PASSED [ 23%]
test_signal_sentinel_semantics.py::test_ratio_is_scale_invariant                     PASSED [ 30%]
test_signal_sentinel_semantics.py::test_deprecated_name_still_works                  PASSED [ 38%]
test_signal_sentinel_semantics.py::test_insufficient_data_is_not_measurable_not_a_pass PASSED [ 46%]
test_signal_sentinel_semantics.py::test_gate_fails_for_no_signal_not_for_leakage      PASSED [ 53%]
test_signal_sentinel_semantics.py::test_gate_fails_a_persistence_like_forecast        PASSED [ 61%]
test_signal_sentinel_semantics.py::test_clean_family_still_passes                     PASSED [ 69%]
test_leakage_real_data.py::TestTemporalOrdering::test_origin_before_target            PASSED [ 76%]
test_leakage_real_data.py::TestTemporalOrdering::test_origin_at_least_one_day_before_target PASSED [ 84%]
test_leakage_real_data.py::TestPredictionQuality::test_predictions_not_exact_match    PASSED [ 92%]
test_leakage_real_data.py::TestPredictionQuality::test_no_shift_leakage               PASSED [100%]

============================== 13 passed in 8.33s ==============================
```

Plus `test_forecast_integrity_multi_horizon.py` + `test_integrity.py`: **20 passed in 1.17s**. Full
suite: **158 passed**.

**But every one of these is a negative control.** They assert that clean inputs pass and that the
sentinel *doesn't* claim leakage. Not one injects leakage and demands a failure. So leakage was
injected:

```
--- CONTROL L  (TRUE LEAKAGE): feature 'oracle' == y(t+h), the exact answer
  signal_sentinel : ratio=446.99  signal_detected=True  leakage_warning=False
                    verdict: signal present: shuffling the targets made error 446.99x worse
  check_feature_leakage (NOT CALLED IN PRODUCTION): leakage_detected=False

--- CONTROL L2 (SUBTLE LEAKAGE): feature = y(t+h) + 15% noise
  signal_sentinel : ratio=17.42  signal_detected=True  leakage_warning=False
  check_feature_leakage (NOT CALLED IN PRODUCTION): leakage_detected=False

--- CONTROL N  (NO SIGNAL): pure-noise features
  signal_sentinel : ratio=1.01  signal_detected=False  leakage_warning=False
                    verdict: WEAK SIGNAL: shuffling the targets barely hurt (ratio 1.01 < 1.50 required)

--- CONTROL S  (LEGIT SIGNAL): backward-only lag/rolling features
  signal_sentinel : ratio=5.38  signal_detected=True  leakage_warning=False

--- CONTROL P (PERSISTENCE-LIKE): y_pred = y(t-h), no leakage, no signal
  shift_diagnostic : best_shift=-5 is_persistence_like=True is_lag0_issue=False
  detect_lagged_copy: risk=low  flagged=[]
```

Controls N and S behave correctly. Control L — a feature that literally *is* the answer — produces a
clean pass at 447×. There is a discriminating gap (447× and 17× for leakage vs 5.4× for legitimate
signal) but **no upper bound exists**; `MIN_SIGNAL_RATIO = 1.5` has no `MAX_` counterpart.

`check_feature_leakage` returned `False` on the oracle. Traced:

```
=== A: check_feature_leakage — what does it expect as `target_series`? ===
  target_series = h-step LABEL y(t+h)  <- what pipelines actually build
     -> leakage_detected=False  details=[]
  target_series = RAW series y(t)       <- what the check assumes
     -> leakage_detected=True  details=["Feature 'oracle' has near-perfect correlation (r=1.000) with target at t+5 (suspicious)"]
```

The check compares `feature[t]` against `target_series[t+h]` (`:141-142`). Hand it the h-step label
the pipelines build and it compares against `y(t+2h)` and finds nothing. The parameter is documented
only as *"target_series: Target series (for reference)"*.

`detect_lagged_copy` failed to flag pure h-step persistence at every window size:

```
=== B: detect_lagged_copy vs h-step persistence, by max_shift ===
  max_shift= 3: risk=low  best_shift=+3 corr@0=0.965 corr_best=0.986 flagged=False <- DEFAULT used by daily_summary.py:200
  max_shift= 5: risk=low  best_shift=+5 corr@0=0.965 corr_best=1.000 flagged=False
  max_shift= 7: risk=low  best_shift=+5 corr@0=0.965 corr_best=1.000 flagged=False
  max_shift=10: risk=low  best_shift=+5 corr@0=0.965 corr_best=1.000 flagged=False
```

Even with `corr_best = 1.000` it does not fire. The blocker is the hardcoded margin at `:445-449`:
`(corr_best - corr_at_0) > 0.05`. On an autocorrelated series `corr@0` is already 0.965, so the
margin can never be cleared. It is also called with default `max_shift=3` and never told the horizon.
It *did* catch C_DL on Revenues (`corr@0=0.25`, `corr_best=0.31`) — clearing the margin by **0.01**.

Note also the **opposite sign convention**: `detect_lagged_copy` reports `+5`,
`shift_diagnostic_horizon_aware` reports `−5`, for the same phenomenon.

**`is_persistence_like` is computed correctly and never read** (`grep`: zero hits in
`daily_summary.py`). Coverage of persistence-like detection:

| Family | `shift_interpretation` | `mae_shift_minus_h` | `detect_lagged_copy` | Coverage |
|---|---|---|---|---|
| B_ML | ✅ | ✅ | blind at h=5 | adequate |
| E_QUANTILE | ✅ | ✅ | blind at h=5 | adequate |
| C_DL | ✗ | ✗ | razor-thin | one thin check |
| A_STAT | ✗ | ✗ | only path | none effective |

### 2.6 Findings, ranked

| # | Severity | Finding |
|---|---|---|
| **L-1** | High | `clean_treasury` imputes and clips using whole-series weekday statistics including the holdout. 8.1% of business-day Revenues touched, 20 inside the 2025 window, median change 118M ≈ 3× model MAE. Upstream of every split |
| **L-2** | High | The committed `master_daily_clean_treasury.csv` contains 1,095 fabricated non-zero weekend values that current code zeroes. Stale artifact, auto-selected by the daily run |
| **S-1** | High | **No active leakage detector exists.** `signal_sentinel` declines by design; `check_feature_leakage` has zero production callers; `leakage_warning` is a hardwired `False` |
| **S-2** | High | Injected oracle leakage yields `signal_detected=True`, "446.99× worse" — a clean pass. No upper ratio bound |
| **S-3** | Medium | `check_feature_leakage` silently misses leakage when given the h-step label — the thing every pipeline builds |
| **S-4** | Medium | `detect_lagged_copy` cannot flag h-step persistence at any `max_shift`; 0.05 margin unclearable on autocorrelated series. Opposite sign convention to the other diagnostic |
| **S-5** | Medium | `is_persistence_like` is computed, written by three families, and never read. A_STAT has no effective check |
| **L-3** | Medium | E_QUANTILE has no embargo — 80% of predictions affected. **Measured impact ≈ nil** (48.45% → 48.69%) |
| **L-4** | Medium | B_ML/C_DL train-val splits have no horizon gap; XGBoost's M-4 exclusion flips (3.01 → 2.98) |
| **F-1** | Medium | `bdom_rev` derives month length from the observed index; final partial month mis-featured |
| **W-1** | Low | `evaluation_windows.py` unenforced; `window_for` disagrees with `TRAIN.contains`; only hardcoded window points at TEST |
| **L-5** | Low | B_ML fold boundary: 4/156 predictions have origins inside the training year |
| **F-2** | Low | B_ML inference `fillna(0.0)` vs training median-impute; E_QUANTILE multivariate `bfill()` + whole-series top-K |

**What was *not* found — the pipelines are largely clean.** All lag and rolling features are
backward-only across all four families. Every scaling and normalisation fit point is train-only,
including C_DL's target transform. No target encoding exists. A_STAT and C_DL are clean on labels and
features; C_DL's label-date masking is the most robust split construction in the repo. B_ML's label
construction is structurally leak-proof. `signal_sentinel` correctly distinguishes no-signal from
legitimate signal, is scale-invariant, and returns `None` rather than a pass when it cannot measure.

**Bottom line.** The in-pipeline leakage story is good, and the one leak found there is worth 0.24pp.
The two serious problems are elsewhere — a data-cleaning stage that writes holdout-informed values
into the CSV before any split exists, and a detection layer whose leakage slot is empty, whose
feature-leakage check is unwired and would silently no-op if wired, and whose persistence detector
cannot see persistence at the configured horizon.

---

# 3 · E_QUANTILE and prediction-interval review

## Prompt

> Review the E_QUANTILE evaluation and all prediction intervals end to end. Verify: (1) quantile loss
> is computed on honest out-of-sample predictions only; (2) interval coverage is empirically measured
> — for a nominal 80% interval, what's the actual coverage on the shared test window? Compute and
> show it; (3) the ResidualRF interval fix is statistically sound — explain in plain language what
> the intervals now represent and what assumptions they rest on. If coverage is badly miscalibrated,
> say so plainly with numbers, don't soften it. Propose fixes as a plan first; do not implement until
> I approve.

## Output

### 3.1 Is quantile loss computed on honest out-of-sample predictions only?

**Yes. This part is clean.** Verified three ways.

Index disjointness: predictions come only from `X_te = X_all.iloc[tr_end:te_end]` (`:346`) while the
fit uses `X_all.iloc[:tr_end]` (`:345`).

```
duplicate (model,target_date) rows across folds: 0
unique target_dates per model: {'GBQuantile': 205, 'ResidualRF': 205}
```

Independent recomputation of every reported pinball figure:

```
model            q        reported   recomputed(pooled)  recomputed(mean-of-folds)  match
GBQuantile    0.10       8,368,405            8,368,405                  8,368,405   OK
GBQuantile    0.50      16,982,182           16,982,182                 16,982,182   OK
GBQuantile    0.90      12,428,656           12,428,656                 12,428,656   OK
ResidualRF    0.10       7,277,395            7,277,395                  7,277,395   OK
ResidualRF    0.50      16,622,930           16,622,930                 16,622,930   OK
ResidualRF    0.90      12,251,471           12,251,471                 12,251,471   OK
```

`_pinball_loss` (`:41-44`) is the correct formula. Aggregation is a mean of per-fold means (`:430`);
because every fold is exactly `h=5` rows this equals the pooled value — but it would silently
mis-weight if fold sizes ever differed.

Two caveats, neither affecting the numbers: the 80% embargo gap from §2 applies (measured impact
0.24pp), and the per-fold console line prints the *cumulative* mean while labelling it
`Fold {fold_ix}` (`:420-423`).

### 3.2 Empirical coverage

| family / model | nominal | empirical | n | mean width | verdict |
|---|---:|---:|---:|---:|---|
| E_QUANTILE / GBQuantile | 80% | **78.0%** | 205 | 98,880,044 | ok (marginally) |
| E_QUANTILE / ResidualRF | 80% | **69.8%** | 205 | 65,467,544 | **under by 10.2pp** |
| B_ML / ElasticNet | 90% | 87.8% | 156 | 204,324,582 | ok |
| B_ML / ExtraTrees | 90% | 87.8% | 156 | 144,956,134 | ok |
| B_ML / HistGBDT | 90% | **80.1%** | 156 | 139,099,721 | **under by 9.9pp** |
| B_ML / Lasso | 90% | 87.2% | 156 | 194,898,585 | ok |
| B_ML / LightGBM | 90% | 84.6% | 156 | 149,556,078 | under by 5.4pp |
| B_ML / RandomForest | 90% | 87.8% | 156 | 146,090,640 | ok |
| B_ML / Ridge | 90% | 87.2% | 156 | 195,032,488 | ok |
| B_ML / XGBoost | 90% | 85.3% | 156 | 147,364,756 | ok |
| C_DL / DCNN·GRU·LSTM·MLP·TRANSFORMER | 90% | 85.5–87.8% | 1722 | 142M–351M | ok |
| **A_STAT / ETS** | 90% | **no intervals at all** | 0 | — | **broken** |

On the shared business-day window (B_ML's 156 target dates):

```
GBQuantile    n=148 of 205  coverage=79.1%   (vs 78.0% on its own window)
ResidualRF    n=148 of 205  coverage=70.3%   (vs 69.8% on its own window)
```

The window is not the driver. E_QUANTILE's own window includes 57 weekend targets whose values are
the fabricated ones from §2 — coverage there is 75.4% / 68.4%, so they are not distorting the
headline either.

**Marginal coverage is hiding the real problem.** Decomposed by realised magnitude:

```
GBQuantile                                  nominal 80%
  low |y|    coverage= 82.6%  n= 69  width=  74,587,316
  mid |y|    coverage=100.0%  n= 68  width=  90,802,416
  high |y|   coverage= 51.5%  n= 68  width= 131,607,647

ResidualRF
  low |y|    coverage= 76.8%  n= 69  width=  65,419,813
  mid |y|    coverage= 91.2%  n= 68  width=  65,505,683
  high |y|   coverage= 41.2%  n= 68  width=  65,477,838
```

**Stated plainly: on the largest third of days, the advertised 80% interval contains the actual value
51.5% of the time for GBQuantile and 41.2% of the time for ResidualRF.** The 78.0% marginal figure is
100% over-coverage in the middle cancelling ~50% under-coverage at the top. ResidualRF's widths
(65,419,813 / 65,505,683 / 65,477,838) confirm the cause: **the band is the same width everywhere.**

Conditioning on origin-time information (the defensible test):

```
GBQuantile
  origin_value tercile     low: 82.6% (w=104.7M) | mid: 75.0% (w= 94.3M) | high: 76.5% (w= 97.5M)
  20d volatility at origin low: 66.7% (w=113.4M) | mid: 82.4% (w= 94.2M) | high: 85.3% (w= 88.8M)

ResidualRF
  origin_value tercile     low: 73.9% (w= 65.4M) | mid: 57.4% (w= 65.4M) | high: 77.9% (w= 65.6M)
  20d volatility at origin low: 68.1% (w= 65.5M) | mid: 69.1% (w= 65.4M) | high: 72.1% (w= 65.5M)
```

GBQuantile's width moves the **wrong way**: lowest-volatility origins get the *widest* bands (113.4M)
and the *worst* coverage (66.7%); highest-volatility origins get the narrowest (88.8M) and the best
(85.3%). Its conditional response to volatility is inverted.

Is the miscalibration real? Moving-block bootstrap (block = one fold = 5 points, 4000 resamples):

```
GBQuantile   pooled=78.0%  block-bootstrap 95% CI=[71.2%, 84.4%]  (naive binomial ±5.5%)
             per-fold coverage histogram: 100%×16  80%×11  60%×9  40%×4  20%×1
ResidualRF   pooled=69.8%  block-bootstrap 95% CI=[61.9%, 77.1%]  (naive binomial ±5.5%)
             per-fold coverage histogram: 100%×9  80%×15  60%×8  40%×6  20%×2  0%×1
```

- **ResidualRF: the CI excludes 80%.** Real, not sampling noise.
- **GBQuantile: the CI contains 80%.** Marginal calibration defensible; conditional is not.
- Dependence widens the interval ~25% over binomial. `quantile_quality_gate:248-270` justifies its
  ±10pp band as "roughly a 3-sigma binomial tolerance at ~150 evaluation points" — that reasoning is
  wrong because the points are not independent.
- The gate also decided ResidualRF's fate on a **0.2pp** margin. The verdict is correct; the route to
  it was a coin-flip.

**Quantile crossing:**

```
GBQuantile     p10>p50:   0   p50>p90:   2   p10>p90:   0   of n=205   *** CROSSED INTERVALS
ResidualRF     p10>p50:   0   p50>p90:   0   p10>p90:   0   of n=205   monotone
```

GBQuantile fits three independent `GradientBoostingRegressor`s (`:351-354`) with no monotonicity
constraint. The `np.maximum` sort at `:243-245` is applied **only inside**
`_fit_residual_rf_quantiles` — the model that structurally cannot cross.

### 3.3 Is the ResidualRF fix statistically sound?

**What the intervals represent, in plain language.** Fit one random forest to predict revenue five
days ahead. Every tree is grown on a bootstrap sample, so roughly a third of the training rows are
left out of any given tree. For each training row, average only the trees that did *not* see it —
the out-of-bag prediction — and record the error. Take the 10th and 90th percentiles of that pile and
add them to today's point forecast. The interval says: *"across the whole training history, 80% of
this model's mistakes fell between −X and +Y."* It is a **single historical error distribution,
shifted to sit around each forecast**. Every day gets the same width.

M-3 was a genuine improvement:

```
  ACTUAL test residual q10..q90 spread (what a calibrated 80% band needs):    105,794,497

  residual source                      mean band width   coverage   nominal 80%
  in-sample (pre-M-3)                       24,258,707      28.8%   UNDER by 51.2pp
  out-of-bag (as shipped, M-3)              65,467,544      69.8%   UNDER by 10.2pp
  split-conformal 20% tail                 115,034,103      82.4%   ok
```

**M-3 closed about two-thirds of the gap and the remaining third is systematic.** The OOB band is 62%
of the width the test residuals require.

**The assumptions — and which fail:**

1. **Exchangeability between training and test residuals.** *Fails.* OOB residual spread grows across
   the decade:

```
  era1  2015-01-25..2017-02-12   spread=    40,922,318
  era2  2017-02-13..2019-02-28   spread=    49,268,875
  era3  2019-03-01..2021-03-19   spread=    75,604,441
  era4  2021-03-20..2023-04-04   spread=    88,286,912
  era5  2023-04-05..2025-04-18   spread=    78,683,729
  POOLED (what the shipped band uses)              spread=    66,376,455
```

  Pooling over ten years lets the calm 2015–2018 era drag the band down to 66.4M, below even the most
  recent era's 78.7M. **This is the primary cause.**

2. **OOB error ≈ deployed-model error.** *Fails, in a direction that helps.* OOB uses ~37% of trees, so
   it measures a weaker predictor and should be pessimistically wide. It is not wide enough anyway.

3. **OOB error ≈ forecast-forward error.** *Fails.* An OOB row from 2018 has training neighbours on
   both temporal sides; the model interpolates. A genuine forecast extrapolates past the end of its
   data. This is why a temporally-adjacent split-conformal holdout succeeds where OOB does not.

4. **Homoscedasticity.** *Fails badly.* Flat 65.4M widths; high-|y| coverage 41.2%.

5. **Signed rather than symmetric residuals.** *Holds, and is a genuine strength* (`:242`) — better
   than B_ML's and C_DL's symmetric `±quantile(|resid|)`.

6. **The OOB fallback is safe.** *Partly.* Silent revert to in-sample residuals (`:235-237`) — the
   28.8% behaviour — with nothing recording which path ran.

**Verdict: directionally correct and honestly documented, but not sufficient.** It swapped a
badly-wrong residual sample for a moderately-wrong one. OOB solves memorisation; it does not address
non-stationarity, forecast-forward extrapolation, or heteroscedasticity.

Fixed-width split-conformal fixes marginal and volatility-conditional calibration but not the tail:

```
SPLIT-CONFORMAL: marginal vs conditional calibration (nominal 80%)
  MARGINAL coverage = 82.4%   mean width=115,034,103
    low vol    coverage= 78.3%      low |y|    coverage= 79.7%
    mid vol    coverage= 79.4%      mid |y|    coverage=100.0%
    high vol   coverage= 89.7%      high |y|   coverage= 67.6%
```

67.6% on the top third — much better than 41.2%, still not 80%. No fixed-width method will get there.

### 3.4 Other interval defects

**A_STAT produces no intervals at all, silently.** ETS is the default `STAT_MODEL`:

```
   get_prediction raised: AttributeError 'HoltWintersResults' object has no attribute 'get_prediction'
```

`run_a_stat.py:169-174` calls `fit.get_prediction(...)` inside a bare `except Exception: return
_nan_pi(n)`. The method does not exist on statsmodels' Holt-Winters results object, so **every ETS
interval has always been NaN** and the failure is swallowed. SARIMAX and STL_ARIMA use `get_forecast`
and do work. The README advertises *"Native Prediction Intervals: ETS/SARIMAX confidence intervals."*

**B_ML and C_DL systematically under-cover by 2–10pp.** Both use `np.quantile(|resid|, 0.90)`
(`b_ml:741`, `c_dl:655-658`) — the plug-in quantile, anti-conservative in finite samples; split
conformal requires the `⌈(n+1)(1−α)⌉/n` order statistic. B_ML's calibration set is also the ungapped
validation tail from §2.

**The frontend cannot read E_QUANTILE intervals and hardcodes the wrong nominal level.** E_QUANTILE
writes `yhat_p10/p50/p90` and no `y_lo`/`y_hi`; the Dashboard's Interval Diagnostics tab reads only
`y_lo`/`y_hi` (`:263`, `:270`), labels the chart "target: 90%" and draws a reference line at 0.90
(`:274-276`). Scorecard thresholds are `>=0.85` trust / `>=0.70` caution (`:375`). A *perfectly
calibrated* 80% band would be scored "caution".

### 3.5 Proposed plan (for approval, not implemented)

**Phase 1 — make the intervals honest.**
- **P1.1** Replace ResidualRF's OOB residuals with split-conformal on a recent holdout tail. Measured: 69.8% → 82.4%, width 65.5M → 115.0M.
- **P1.2** Enforce monotonicity on GBQuantile (fixes 2 observed crossings).
- **P1.3** Record `interval_method` and `n_calibration` so a silent fallback is visible.
- **P1.4** Use the finite-sample conformal quantile in B_ML and C_DL; give B_ML's calibration split the horizon gap.
- **P1.5** Fix or remove A_STAT's ETS intervals; replace the bare `except`.

**Phase 2 — fix conditional calibration.**
- **P2.1** Conformalised Quantile Regression (CQR) for GBQuantile *(recommended)* — keeps adaptivity, adds a finite-sample marginal guarantee.
- **P2.2** Investigate GBQuantile's inverted volatility response before adding machinery.
- **P2.3** Reposition ResidualRF as the baseline interval method, or normalise its residuals.

**Phase 3 — make the gate say what it means.**
- **P3.1** Gate on conditional coverage, not just marginal. This is the change that would have caught GBQuantile.
- **P3.2** Replace the point-estimate band with a bootstrap test; correct the docstring's "3-sigma binomial" justification.
- **P3.3** Carry `nominal_pi` as data; add `y_lo`/`y_hi` aliases for E_QUANTILE.

**What not to do:** widen bands to hit 80% marginal by construction — that hides the conditional
failure. **Standing caveat:** all numbers rest on `master_daily_clean_treasury.csv`, which §2 found to
be a stale artifact with 1,095 fabricated weekend values and 8% whole-series-clipped business days.
Regenerating it will move every coverage number.

---

# 4 · Full backtest run and trust verification

## Prompt

> Run the full backtest via --mode backtest and scripts/backtest_report.py against the shared test
> window. Then verify the backtest itself is trustworthy: confirm each backtest window only uses data
> available as of that date, that the persistence baseline in the backtest is identical to the one in
> normal evaluation, and that results are reproducible (run twice, diff the outputs). Paste actual
> before/after metrics per model family vs the honest persistence baseline. Verdict: which models
> genuinely beat persistence, on which targets, by how much?

## Output

**Command:** `MODE=backtest ./scripts/run_daily_forecast.sh` → `scripts/backtest_report.py`
**Output:** `backend/forecast_runs/2026-08-04/` · exit 0 both runs · 4/4 families produced output
**Scope:** Revenues, Daily, h=5

Preconditions checked first — the script `rm -rf`s its dated folder, and `2026-08-04` did not exist.

### 4.1 Is the backtest trustworthy?

**Temporal availability — yes:**

```
TEMPORAL AVAILABILITY — does each prediction use only data dated <= its own origin?
family           n  origin<target  step gap==h  origin_value==y(origin)  index
A_STAT         156           PASS     PASS h=5                     PASS   business-day
B_ML          1248           PASS     PASS h=5                     PASS   business-day
E_QUANTILE     410           PASS     PASS h=5                     PASS   calendar-day
C_DL          8610           PASS     PASS h=5                     FAIL   business-day
```

C_DL's "FAIL" is a precision artifact, not a leak:

```
max |origin_value - y(origin)| = 7.9200   max relative = 5.690e-08
rows deviating by >1e-6: 8520 of 8610
cause: c_dl_pipeline.py:467  ov = np.array(origin_values_list, dtype=np.float32)
float32 eps at 1e8 ~= 8.00  -> consistent with the deviation above
```

Immaterial to a 53M baseline, but C_DL's persistence baseline is not bit-comparable with the others'
float64 computation.

**Baseline identity — yes, and the mode flag changes nothing:**

```
family         integrity_report    leaderboard row          shared fn        plain numpy  agree(<1e-6)
A_STAT            60,976,736.58      60,976,736.58      60,976,736.58      60,976,736.58   YES
B_ML              60,976,736.58      60,976,736.58      60,976,736.58      60,976,736.58   YES
E_QUANTILE        65,888,163.54                n/a      65,888,163.54      65,888,163.54   YES
C_DL              52,957,744.22                n/a      52,957,744.22      52,957,744.22   YES
```

```
family        skill (backtest)  skill (production)   gate bt  gate prod  metrics identical
A_STAT                  27.51%              27.51%      True       True   YES
B_ML                    36.75%              36.75%     False      False   YES
E_QUANTILE              48.45%              48.45%      True       True   YES
C_DL                    10.84%              10.84%     False      False   YES

top-level SUMMARY.json keys that differ between modes: ['mode', 'freshness']
```

`--mode` touches only the freshness wording. But note: the four baselines are internally consistent
and **mutually different**.

**Reproducible — yes, to the last bit for reporting purposes:**

```
run1 vs run2:  IDENTICAL  a_stat/predictions_long.csv    IDENTICAL  SUMMARY.json
               IDENTICAL  c_dl/daily/predictions_long.csv IDENTICAL  BACKTEST_REPORT.md
               *** DIFFERS b_ml/predictions_long.csv
               *** DIFFERS e_quantile/predictions_long.csv

B_ML  (n=1248 rows, identical keys)
   y_pred     max|Δ|=0.000000   max relative Δ=1.013e-15   n rows differing=203
   y_lo       max|Δ|=0.000000   max relative Δ=2.856e-13   n rows differing=243
     RandomForest   MAE run1=38,565,591.4272  run2=38,565,591.4272  Δ=0.000000
     XGBoost        MAE run1=38,554,577.1712  run2=38,554,577.1712  Δ=0.000000

E_QUANTILE  (n=410 rows, identical keys)
   yhat_p50   max|Δ|=0.000000   max relative Δ=9.822e-16   n rows differing=125
     GBQuantile     MAE run1=33,964,364.7006  run2=33,964,364.7006  Δ=0.000000
```

Last-bit floating point from thread reduction order under `n_jobs=-1`. Every MAE identical to 4
decimals; all rankings, best-model choices, gate verdicts and coverage figures identical;
`SUMMARY.json` and `BACKTEST_REPORT.md` byte-identical. This extends the audit's `m-2` note
(previously E_QUANTILE only) to B_ML — CSV byte-comparison is not a valid regression test for those
two families.

### 4.2 Before / after

```
BEFORE (2026-07-30, pre M-4/M-5/Phase-3)
  family      best model          skill    gate   reason
  A_STAT      ETS                27.51%    True   -
  B_ML        Lasso              28.71%   False   leakage flag raised
  E_QUANTILE  GBQuantile         48.45%    True   -
  C_DL        MLP                10.84%    True   -

AFTER (2026-08-04, current code)
  A_STAT      ETS                27.51%    True   -
  B_ML        RandomForest       36.75%   False   no signal beyond shuffled targets (ratio 1.13)
  E_QUANTILE  GBQuantile         48.45%    True   -
  C_DL        MLP                10.84%   False   forecast is persistence-like (shift diagnostic)
```

Two verdicts changed for the better: B_ML's failure reason is now accurate, and C_DL is correctly
withheld instead of presented as clean.

```
  model               MAE before       MAE after  skill before  skill after   Δskill
  XGBoost             47,191,696      38,554,577        22.61%       36.77%   +14.16pp
  RandomForest        44,140,611      38,565,591        27.61%       36.75%    +9.14pp
  ExtraTrees          47,088,711      39,347,999        22.78%       35.47%   +12.69pp
  HistGBDT            49,594,250      41,978,520        18.67%       31.16%   +12.49pp
  LightGBM            53,586,169      42,969,803        12.12%       29.53%   +17.41pp
  Lasso               43,472,942      43,315,354        28.71%       28.96%    +0.26pp
  Ridge               43,475,386      43,333,157        28.70%       28.93%    +0.23pp
  ElasticNet          45,498,679      45,064,066        25.38%       26.10%    +0.71pp

  field                              before                     after
  best_model                       (absent)              RandomForest
  skill_pct                         28.7057                   36.7536
  best_shift                            -10                         0
  leakage_warning                      True                     False
  shuffled_to_normal_ratio           0.8279                    1.1278
  signal_detected                  (absent)                     False
  overfit_excluded_models          (absent)   ['LightGBM', 'XGBoost']
```

The tree models gained 9–17pp — that is the Phase-3 `dom`/`bdom`/`bdom_rev` features, not the M-4
capacity floors. `best_shift` went −10 → 0.

The other three families are numerically unchanged, confirming the fixes were scoped to B_ML:

```
  A_STAT       pooled MAE before= 44,199,748.0592  after= 44,199,748.0592  IDENTICAL
  E_QUANTILE   pooled MAE before= 33,605,112.3672  after= 33,605,112.3672  IDENTICAL
  C_DL         pooled MAE before= 52,677,862.7005  after= 52,677,862.7005  IDENTICAL
```

### 4.3 The report the pipeline produced is not on one window

`BACKTEST_REPORT.md` states: *"Skill vs persistence … measured on one shared baseline for all
families."* **That sentence is false as generated.**

```
SHARED TEST WINDOW: 2025-01-01 .. 2025-08-06   n=156 business days
ONE PERSISTENCE RULER  y_hat(t)=y(t-5 business days):  MAE = 60,976,737

family / model               n on shared             MAE  skill vs ONE ruler  beats persistence?
E_QUANTILE / ResidualRF              148      32,095,336              47.36%   YES
E_QUANTILE / GBQuantile              148      32,449,183              46.78%   YES
B_ML / XGBoost                       156      38,554,577              36.77%   YES
B_ML / RandomForest                  156      38,565,591              36.75%   YES
B_ML / ExtraTrees                    156      39,347,999              35.47%   YES
B_ML / HistGBDT                      156      41,978,520              31.16%   YES
B_ML / LightGBM                      156      42,969,803              29.53%   YES
B_ML / Lasso                         156      43,315,354              28.96%   YES
B_ML / Ridge                         156      43,333,157              28.93%   YES
A_STAT / ETS                         156      44,199,748              27.51%   YES
B_ML / ElasticNet                    156      45,064,066              26.10%   YES
C_DL / MLP                           156      64,143,423              -5.19%   NO
C_DL / TRANSFORMER                   156      69,633,516             -14.20%   NO
C_DL / DCNN                          156      76,556,282             -25.55%   NO
C_DL / GRU                           156      78,583,753             -28.87%   NO
C_DL / LSTM                          156      79,014,459             -29.58%   NO
```

**C_DL reports +10.84% and is actually −5.19% on the reporting window.** Its +10.84% is an average
over 2019–2025 where persistence is easier (53.0M vs 61.0M). All five architectures are worse than
doing nothing on the 2025 holdout. The C-1 fix took C_DL from −38.65% to −5.19% here — real progress,
but still unusable, and the shipped number hides it.

**Does E_QUANTILE's lead survive a fair horizon?**

```
n=148 shared target dates
persistence, 5 BUSINESS-day origin (shared definition):     60,273,679
persistence, 5 CALENDAR-day origin (E_QUANTILE's own)  :     66,161,268   (9.8% weaker ruler)
    GBQuantile   skill vs own ruler=50.95%   skill vs SHARED ruler=46.16%
    ResidualRF   skill vs own ruler=51.49%   skill vs SHARED ruler=46.75%
```

Feeding E_QUANTILE's own pipeline a business-day-reindexed series (same code, different input index):

```
APPLES-TO-APPLES: every family at h=5 BUSINESS days, 2025 window, ONE persistence ruler
model                          n             MAE     persistence     skill
E_QUANTILE / GBQuantile      150      31,875,452      60,578,809    47.38%
E_QUANTILE / ResidualRF      150      31,454,643      60,578,809    48.08%
B_ML / XGBoost               156      38,554,577      60,976,737    36.77%
B_ML / RandomForest          156      38,565,591      60,976,737    36.75%
B_ML / Lasso                 156      43,315,354      60,976,737    28.96%
A_STAT / ETS                 156      44,199,748      60,976,737    27.51%
```

**E_QUANTILE's lead is genuine.** At a true 5-business-day horizon it holds 47–48% skill, ~11pp ahead
of B_ML's best. The most likely reason: GBQuantile's P50 optimises pinball loss at q=0.5 — absolute
error, the metric being reported — while B_ML's estimators minimise squared error. On a spiky series,
fitting the conditional median beats fitting the conditional mean when scored on MAE.

### 4.4 Verdict — which models genuinely beat persistence

**Target: Revenues, Daily, h=5 business days, 2025-01-01 … 2025-08-06 (156 business days), one shared
ruler MAE 60,976,737.**

| Tier | Models | Skill | Assessment |
|---|---|---:|---|
| **Best** | E_QUANTILE ResidualRF, GBQuantile | **47–48%** | Genuinely beats persistence; lead survives horizon correction. GBQuantile is the only model that also passes its interval gate |
| **Solid** | B_ML XGBoost, RandomForest, ExtraTrees | 35–37% | Beat persistence substantially. XGBoost excluded by M-4 on a knife-edge |
| **Moderate** | B_ML HistGBDT, LightGBM, Lasso, Ridge, ElasticNet; A_STAT ETS | 26–31% | Beat persistence. ETS is the only gate-passing model in this tier |
| **Worse than nothing** | C_DL LSTM, GRU, DCNN, TRANSFORMER, MLP | **−5% to −30%** | All five fail. Do not use |

**Two things "beats persistence" does not mean.** First, B_ML's shuffled-target sentinel reports
`signal_detected=False` at ratio 1.13. A model with 36.75% skill and no detectable signal is beating
an *inflated* baseline — h-step persistence chases holiday and month-end spikes and gets punished,
while a smooth predictor sidesteps them. That is regression to the mean, not forecasting skill.
Second, only **1 of 41 candidate targets** was evaluated. This verdict covers Revenues only;
`State budget balance` in particular routes through entirely different code paths that nothing has
evaluated.

**Standing caveat:** every number rests on the stale, partly-fabricated
`master_daily_clean_treasury.csv`. This table should not go in front of the client before that is
resolved.

---

# 5 · Quality gates, test suite, and five new regression tests

## Prompt

> Review the quality gates and test suite. For each gate, tell me: what failure mode it catches, what
> would slip through, and whether it fails for the accurate reason (leakage vs no-signal vs
> persistence-mimicry must stay distinct). Then identify the 5 highest-value missing tests —
> prioritize regression tests for the bugs we already fixed (shuffled-target, inverted sentinel, zero
> train MAE overfitting, duplicated baselines) so they can never silently return. Plan first, then
> implement one test at a time with a passing run shown after each.

## Output

**Result: 193 passed** (was 158; +35). Five new files, zero production changes.

### 5.1 Gate review — pipeline gates

| Gate | Location | Catches | Slips through | Accurate reason? |
|---|---|---|---|---|
| **Skill gate** (skill < 5%) | `b_ml:1193`, `a_stat:350`, `c_dl:909` | Model worse than persistence on its own window | **Everything about *which* window.** C_DL reports +10.84%, is −5.19% on the shared holdout. Also passes a no-signal model and one missing half its largest days | ⚠️ Reason accurate but the *baseline* differs per family |
| **Overfit gate** (val/train > 3.0) | `select_best_model` (`:335`) | Memorising models being crowned best | **The leaderboard and the plots.** Verdict flips on a 4-row split change (3.01 vs 2.98) | ✅ Distinct |
| **Coverage gate** (P10–P90 ∈ [70%,90%]) | `quantile_quality_gate` (`:248`) | Marginally miscalibrated intervals | **Conditional miscalibration.** GBQuantile passes at 78% while covering 51.5% of the top tercile | ✅ Own category |
| **Alignment gate** | `b_ml:1139`, `b_ml:840` | `origin + h ≠ target` | **C_DL entirely** — `"alignment_ok": True` is a literal (`c_dl:905`). A_STAT writes no field | ✅ Distinct |
| **Critical timestamping** | `b_ml:1158` | Gross inversion | Anything subtler than 95%. `is_lag0_issue` contradicts `is_persistence_like` at `best_shift == −h` | ⚠️ Flag collision |
| **Signal sentinel** (ratio ≥ 1.5) | `signal_sentinel` (`:213`) | Features carrying no information | **Leakage** — an oracle scores 447× and reports "signal present". No upper bound | ✅ **Correct after M-5** |

### 5.2 Gate review — summary / reporting gates

| Gate | Catches | Slips through | Accurate reason? |
|---|---|---|---|
| `gate_reasons` (`:83`) | `FAILED_QUALITY`, leakage, no-signal, persistence-like as four separate strings | A family that never wrote the field is not gated on it. A_STAT writes no shift fields | ✅ **Three modes stay distinct** |
| Tri-state `gate_passed` | "Never verified" cannot look like a pass (`:289`) | — | ✅ |
| `summary_leakage_check` | `origin_date >= target_date` | **Every real leakage mechanism.** The only live detector | ⚠️ Slot nearly empty |
| `detect_lagged_copy` | Lagged copies where `corr_best − corr_at_0 > 0.05` | **h-step persistence at h=5.** Fails on a *perfect* copy at every `max_shift` | ⚠️ Right category, blind in the treasury case |
| Exit code | Only "no family produced output" (`:458`) | **All gate failures exit 0** | ❌ |

**Verdict on distinctness:** the three modes *are* properly separated — in the sentinel's semantics,
the four gate-reason strings, and ownership. The failure is not conflation, it is **coverage**: the
leakage owner is unwired and the persistence owner is blind at the production horizon.

### 5.3 The plan

| # | Test | Locks | Why the existing suite missed it |
|---|---|---|---|
| 1 | `test_sentinel_holdout_split.py` | M-5 **wiring**: sentinel scored on disjoint, embargoed, recent rows | Only the sentinel *function* was tested |
| 2 | `test_failure_mode_distinctness.py` | M-5 **inversion**: ratio direction + three separate gate reasons | Semantics tested one-way; distinctness invariant wasn't |
| 3 | `test_published_baseline_is_shared.py` | C-2: the *published* number is the shared function's | `test_unified_baseline` greps source, not the artifact |
| 4 | `test_overfit_ratio_recording.py` | M-4 **producer**: zero train MAE → infinite → excluded | Only `select_best_model` tested, ratios hand-fed |
| 5 | `test_no_signal_end_to_end.py` | M-5 **chain**: pipeline → artifact → summary → reason | Gate tested against hand-written JSON |

### 5.4 Implementation, with mutation proof

| # | Result | Mutation applied | Outcome |
|---|---|---|---|
| 1 | 8 passed | `gap = 0` | 1 failed ✓ |
| 1 | | pre-M-5 `iloc[:500]`/`iloc[-100:]` | 2 failed ✓ — `[short] 84 row(s) appear in both slices` |
| 2 | 10 passed | `leakage_warning = (ratio < 1.5)` | 3 failed ✓ |
| 3 | 5 passed | duplicate impl → `median` | 2 failed ✓ (`test_unified_baseline` **passed** — gap confirmed) |
| 4 | 6 passed | remove zero-error branch | 3 failed ✓ (`test_b_ml_overfitting` **passed** — gap confirmed) |
| 5 | 6 passed | stop publishing `signal_detected` | 3 failed ✓ |

**Two corrections that came from doing this rather than assuming.**

**My first fixture could not detect the bug it was written for.** With ~1,900 training rows,
`iloc[:500]` and `iloc[-100:]` happen to be disjoint — the original defect only bites below ~600
rows, which is the Demo/Balanced profile. A short-history parametrisation was added; it now reports
`84 row(s) appear in both slices`.

**A claim in test 5's docstring was wrong and was corrected.** I had written that reordering
`integrity_report.update(legacy_report)` after the sentinel would wipe `signal_detected` and silently
un-gate the run. The mutation passed, so it was measured:

```
keys legacy_report would OVERWRITE if merged after the sentinel:
   leakage_warning  -> False        (already its value)
   mae_shuffled_target  -> nan      (display only)
sentinel keys legacy_report does NOT contain (so they survive any merge):
   shuffled_to_normal_ratio, signal_detected, signal_verdict
```

The gate is *not* at risk from that ordering. The docstring now states the real fragility — five
nested guards around the sentinel block, any of which skips it silently and leaves no
`signal_detected` key, so "not measured" becomes "fine".

Also from test 5's mutation run: with `signal_detected` absent, the run reported `[OK] Quality gate
passed: skill=31.51%` and was still withheld — but for the *shift* reason. The specific-reason
assertion caught it; a "was it withheld?" test would have passed.

**Not covered, deliberately:** the leaderboard/plots ignoring the overfit gate,
`mae_seasonal_naive` duplicating persistence at h=5, `bdom_rev`'s truncated final month, and
`detect_lagged_copy`'s blindness at h=5 are all *unfixed*. Tests for those would fail today.

---

# 6 · Agent artifact contract and validation plan

## Prompt

> The AI4CM Agent (separate repo) reads SUMMARY.json, leaderboards, and predictions_long.csv. Treat
> these as a contract: document their exact schema as produced today, check for keys that can be
> missing (we already hit a missing data_file key bug), and add a validation step at the end of the
> pipeline that fails loudly if any artifact is malformed or incomplete before it's published. Plan
> first.

## Output

### 6.1 Artifact 1 · `SUMMARY.json`

Written by `daily_summary.py:418-454`. One file per run at `forecast_runs/<date>/SUMMARY.json`.

```
run_date     str    '2026-08-04'
target       str    'Revenues'
cadence      str    'Daily'
horizon      str    '5'          ← string, not int
mode         str    'backtest' | 'production'
families     list   (one object per requested family)
overall      dict   families_requested:int, families_ok:int, families_gate_passed:int,
                    leakage_flags:int, shift_flags:int, quality_gate_failures:int
freshness    dict   line:str, stale:bool, backtest:bool
```

`families[]` — 12 keys, all always present:

```
name                 str        'A_STAT'
ok                   bool       family produced usable output
models               str        'ETS, Persistence (baseline)'   ← comma-joined, not a list
best_model           str        'ETS (MAE 44,199,748)'          ← name+MAE fused
best_model_display   str        may carry 'WITHHELD — <reasons>...' or '(integrity not verified)'
skill_pct            str        '27.51%'  OR  'n/a (not produced)'
run_status           str        'SUCCESS' | 'FAILED_QUALITY' | 'ERROR' | 'n/a (not produced)'
integrity_verified   bool
gate_passed          bool|null  tri-state: null = never verified
gate_reasons         list[str]
leakage_flag         bool
shift_flag           bool
```

### 6.2 Artifact 2 · `leaderboard.csv` — four incompatible schemas

| Family | Columns | `rank` semantics |
|---|---|---|
| A_STAT | `target, horizon, cadence, model, MAE, RMSE, rank` | **0 = best model**, sorted by MAE |
| B_ML | `target, horizon, model, MAE, rank` | **0 = persistence baseline**, prepended unsorted |
| C_DL | `target, horizon, model, MAE, rank` | **1..N**, no baseline row |
| E_QUANTILE | `model, pinball_q10, pinball_q50, pinball_q90, coverage_p10_p90, MAE` | **no rank column** |

C_DL also writes `leaderboard_<target>_h<h>.csv` with 13 different columns.

### 6.3 Artifact 3 · `predictions_long.csv` — 9 common columns of 19

```
column          A_STAT   B_ML   E_QUANTILE   C_DL
date              yes     yes       yes       yes   ← == target_date in all four
target_date       yes     yes       yes       yes
origin_date       yes     yes       yes       yes
origin_value      yes     yes       yes       yes
target            yes     yes       yes       yes
horizon           yes     yes       yes       yes
model             yes     yes       yes       yes
y_true            yes     yes       yes       yes
y_pred            yes     yes       yes       yes
--- family-specific below ---
y_lo / y_hi       yes     yes        .        yes   ← 90% nominal
yhat_p10/50/90     .       .        yes        .    ← 80% nominal
split_id          yes     yes        .        yes
cadence           yes      .         .        yes
defn_variant       .      yes        .        yes
horizon_note      yes      .         .         .
fold               .       .        yes        .
```

### 6.4 Missing & unstable keys, ranked

| # | Defect | Evidence | Impact |
|---|---|---|---|
| **C1** | **`data_file` absent from SUMMARY.json** | `SUMMARY.txt` line 4 reads `Data file: master_daily_clean_treasury.csv`; the JSON has no such key | **The bug you hit.** Cannot tell which dataset produced a run |
| **C2** | **`notes` absent from SUMMARY.json** | `info["notes"]` holds `"no predictions_long.csv found"` / `"predictions_long.csv is empty"`; text only | A family reports `ok: false` with **no machine-readable reason** |
| **C3** | `skill_pct` is a string, sometimes `'n/a (not produced)'` | verified in degraded run | `float(s.rstrip('%'))` raises; never null |
| **C4** | `horizon` is `str` in SUMMARY.json, `int64` in predictions_long | `'5'` vs `np.int64(5)` | Joins fail silently |
| **C5** | **A_STAT's leaderboard has NaN `target`/`horizon`/`cadence` on model rows** | `ETS` row: `target=NaN, horizon=NaN`; only the persistence row carries them | A join on `(target, horizon)` **drops every real model** |
| **C6** | `rank` means three different things; E_QUANTILE has none | table above | `rank == 0` yields the baseline for B_ML and the **best model** for A_STAT |
| **C7** | E_QUANTILE's leaderboard has no `target`/`horizon` | verified | Rows cannot be attributed to a task |
| **C8** | Interval columns and nominal level differ; level recorded nowhere | 90% vs 80% | Coverage compared to the wrong target |
| **C9** | **C_DL's empty-run stub omits `target_date`, `origin_date`, `origin_value`** | `c_dl_pipeline.py:963` | A *failed* C_DL publishes a different schema than a successful one |
| **C10** | No `schema_version`, `git_sha`, or per-family `eval_window` | verified absent | Cannot pin a contract version or see C_DL's differing window |
| **C11** | SUMMARY.json published even when the run failed | exit 1, file still written | No top-level `status` |
| **C12** | `models` comma-joined; `best_model` fuses name+MAE | `'ETS (MAE 44,199,748)'` | Requires regex to recover either |

Headline: `git ls-files '*.py' | xargs grep -ln "validate_artifact\|REQUIRED_COLUMNS\|schema_version"`
returns nothing. **There is no artifact validation anywhere in the repo.**

### 6.5 Plan

**Phase 0 — freeze the contract in writing.** `docs/AGENT_CONTRACT.md` with the tables above plus
per-field type, nullability and stability tier. Mark the current shape `schema_version: 1` and the
intended `2` alongside it. A deliverable the Agent team can code against today.

**Phase 1 — additive SUMMARY.json fixes (backward compatible).** Nothing removed or retyped:

```
schema_version   int         2
status           str         'ok' | 'degraded' | 'failed'
data_file        object      {name, path, sha256, latest_data_date, n_rows}
families_requested  list[str]
eval_window      object      {start, end, n_target_dates}     top-level + per family
families[].notes            list[str]   ← fixes C2
families[].skill_pct_value  float|null  ← numeric twin, null not a sentinel string
families[].horizon_steps    int
families[].models_list      list[str]
families[].best_model_name  str
families[].best_model_mae   float|null
```

Retyping in place would break the Agent; add twins now, deprecate the string forms, remove in
`schema_version: 3`.

**Phase 2 — normalise the CSV contracts.** Require `target, horizon, model, MAE` on all four
leaderboards; fix A_STAT's NaN join keys; add `is_baseline: bool`; define `rank` as 1..N over trained
models. For predictions: declare the 9-column core as required, emit `y_lo`/`y_hi` aliases for
E_QUANTILE, add `pi_nominal`, fix the C_DL stub. These change file contents — needs the Agent team's
sign-off on timing.

**Phase 3 — the validator.** `backend/artifact_contract.py` (declarative spec + `validate_run`),
`scripts/validate_artifacts.py` (CLI, writes `VALIDATION.json`, non-zero on ERROR),
`backend/tests/test_artifact_contract.py` (one fixture per malformation, including permanent
regression tests for C1 and C2).

Checks, tiered — **ERROR:** file missing/unparseable; required column missing; wrong dtype; null in a
non-nullable column; empty predictions while the family reports `ok`; any `origin_date >= target_date`;
`origin_value` null where `y_true` present; leaderboard missing `MAE`; SUMMARY.json missing a required
key; a family in SUMMARY.json with no directory; `overall.families_ok` inconsistent with per-family
flags. **WARN:** declared-optional column absent; `pi_nominal` unknown; eval windows differ;
`gate_passed: null`.

**Phase 4 — wire into publication.** One decision needed:

- **Option A — validate in place (small).** Families → validate → `daily_summary` → validate →
  fail the shell script. ~20 lines. Downside: a malformed run still sits at the path the Agent reads.
- **Option B — staging and atomic promote (recommended).** Run into `forecast_runs/.staging/<date>/`,
  validate, then `mv` to `forecast_runs/<date>/`. **The Agent only ever sees a validated run.** Final
  path unchanged, so the Agent needs no modification. Failed runs stay in `.staging/` for debugging.
  Cost: `run_daily_forecast.sh` changes shape; its `rm -rf` guard needs extending.

Recommend **B** — the only version that makes the guarantee real. Also fix the exit-code gap while
there.

**Sequencing note.** Run the validator against the five existing `forecast_runs/` folders as its first
real test. Expect failures — `2026-07-31` has no `SUMMARY.json`; older runs predate the M-fixes. That
is the validator earning its keep.

---

# 7 · Production readiness for automated daily runs

## Prompt

> Assess production readiness for automated daily runs. Cover: what happens on missing/late/partial
> input data; idempotency if a run is triggered twice; where drift detection should hook in and what
> the simplest honest drift signal would be; logging sufficient to reconstruct any run after the
> fact; and what the HTML report needs to show a non-technical Treasury reader (verdicts and plain
> language, not acronyms). Output a prioritized readiness checklist — don't build anything yet.

## Output

**Verdict: not ready for unattended automation.** The modelling and gating layers are in reasonable
shape; the *operational* layer is largely absent. Three findings are hard blockers: a silent
data-gap corruption path, a destructive non-idempotent run directory, and no log capture at all.

### 7.1 Missing, late, and partial input data

| Scenario | What happens today | Verdict |
|---|---|---|
| **No data file** | Clear error, exit 1 (`run_daily_forecast.sh:57-65`) | ✅ fine |
| **Missing target column** | `KeyError: 'Revenues'`, exit 1, `artifacts/error.json`, `set -e` aborts | ✅ fine |
| **Stale data** | `check_freshness` warns; the run **proceeds and publishes**. `STALE_DAYS=3` | ⚠️ warns, doesn't stop |
| **Partial / late delivery** | `to_business_index` does `s.fillna(0.0)` for flows (`b_ml_pipeline.py:150`) | 🔴 **silent corruption** |
| **Truncated history** | `min_train_years` silently ignored by the fallback chain | 🔴 **publishes anyway** |
| **Pre-flight checks** | `frontend/data_preflight.py` exists — **only wired into the Streamlit Lab** | 🔴 **not wired** |

**The gap path is the most dangerous thing in the system.** Removing three months of target rows
(May–Jul 2025) from an otherwise clean file:

```
  full.csv -> May-Jul 2025: n=66, zeros=0,  mean=70,130,388
  gap.csv  -> May-Jul 2025: n=66, zeros=66, mean=0

  h=5 persistence MAE on 2025 window:  full=12,468,801   with gap=11,211,483   (-10.1%)
```

Absent rows become **66 business days of genuine-looking zero revenue**. No warning, no NaN, no
row-count check. And because the persistence baseline is computed from the same corrupted series,
**the yardstick every model is graded against moves by 10%**. Zeros are legitimate in this series
(holidays), so `fillna(0.0)` cannot be distinguished from real data after the fact.

**Truncated history degrades silently.** A_STAT on 40 business days with `min_train_years=4`:

```
WARNING: Not enough full-year coverage; using recent sliding-window fold.
Persistence MAE=6,765,719, Model MAE=6,501,001, Skill=3.91%
  -> PUBLISHED 10 prediction(s)
  integrity: {'skill_pct': 3.91, 'quality_gate_passed': False, 'run_status': 'FAILED_QUALITY'}
```

A full integrity report off **10 evaluation points**. The gate caught it at 3.91% against a 5%
threshold — had skill landed at 6%, this would have reported `SUCCESS`. There is no
minimum-evaluation-points check anywhere, and the fallback chain at `run_a_stat.py:267-276` ignores
`min_train_years` entirely, down to a "last-ditch" two-point test block.

### 7.2 Idempotency

**Repeatability is genuinely good** (§4: bit-identical A_STAT/C_DL, 10⁻¹⁵ for B_ML/E_QUANTILE, all
MAEs identical to four decimals, `SUMMARY.json` and `BACKTEST_REPORT.md` byte-identical).

**The run directory is not safe:**

1. **Destructive-first ordering.** `rm -rf "$RUN_DIR"` at line 73, *before* any family executes. A
   re-run failing on family 2 of 4 has already destroyed the previous good run and leaves a partial
   tree with no `SUMMARY.json`.
2. **No concurrency guard.** No `flock`, no PID file. Cron overlap means two processes `rm -rf` and
   write the same directory concurrently.
3. **Input selection isn't stable across time.** Data file chosen by *mtime*; two runs on the same
   `RUN_DATE` can use different inputs.
4. **Unpinned timezone.** `RUN_DATE="$(date +%F)"` uses local time.

### 7.3 Drift detection

**Where it should hook.** After predictions exist and the integrity report is computed, **before** the
gate verdict is finalised — so drift can contribute a gate reason. Two artifacts: a `drift` block in
each family's integrity report, and a run-level `drift.json`. Depends on the artifact contract work,
since drift compares against previously published runs.

**The simplest honest signal: two numbers on the trailing 60 business days, and their ratio.**

1. `persistence_mae_60d` — the h-step persistence baseline. **Model-free**; moves only when the
   data's volatility moves.
2. `model_mae_60d` — the deployed model's error over the same window.

| baseline | model MAE | reading | action |
|---|---|---|---|
| ↑ | ↑, ratio flat | the world got harder; model kept pace | inform |
| flat | ↑ | **model drift** | alarm |
| ↑ | ↑, ratio ↓ | model failing to adapt | alarm |
| ↓ | flat | baseline got easier; apparent skill will fall | inform |

This matters because the baseline moves enormously on this data:

```
  2015   persistence MAE =     25,468,890     +0.0%
  2019   persistence MAE =     39,346,247    +54.5%
  2022   persistence MAE =     56,900,468   +123.4%
  2024   persistence MAE =     63,966,230   +151.2%
  2025   persistence MAE =     60,976,737   +139.4%
```

A +139% shift, near-monotonic after 2016. Any monitor watching model error alone will fire constantly
for the wrong reason.

**Compare year-over-year, not against the previous window.** The 60-day rolling baseline has an IQR of
30.6M–56.7M within normal history (max/median 1.85×), so a naive "1.5× previous window" rule fires on
ordinary seasonality.

**Cheap input-side checks worth adding:** row count vs expected business days; date continuity; share
of zero business days (historically 0–1% per year — the gap fixture would have shown 100% for three
months); monthly total vs same month last year (median +12.2%, 5th/95th −16%/+43%).

**Two honest caveats.** Thresholds must be set on DEV; tuning against 2025 would burn the holdout. And
drift monitoring is close to meaningless for a family whose sentinel already reports no signal — you
cannot detect degradation in a model that never had skill.

### 7.4 Logging and reconstructability

**Cannot reconstruct a run today.**

| Need | Status |
|---|---|
| Run log | 🔴 **None.** `run_daily_forecast.sh` has no `tee` or redirect. All `[pipeline]`/`[WARN]`/`[M-4]` diagnostics — including every fold's train/val MAE and the overfit exclusions — are lost |
| Input identity | 🔴 No hash, row count or date range. `data_file` isn't even in `SUMMARY.json` |
| Code version | 🔴 No git SHA anywhere |
| Dependency versions | 🔴 Not captured (matters: `HAVE_XGB`/`HAVE_LGBM` silently change the model set) |
| Config | ⚠️ Uneven — B_ML writes `config.json` + `provenance.json`, C_DL `config.json`, E_QUANTILE `run.json`, **A_STAT only `integrity_report.json`** |
| Timing | ⚠️ Only E_QUANTILE records `elapsed_sec` |
| Seeds | ⚠️ In B_ML's config; A_STAT has no seed at all |

`provenance.json` exists for exactly one of four families. If a reviewer asks what produced the number
in the 2026-07-30 report, the honest answer is that we cannot fully say.

### 7.5 The HTML report for a non-technical Treasury reader

**No HTML output exists.** Published artifacts are `SUMMARY.txt`, `SUMMARY.json`,
`BACKTEST_REPORT.md`. The Dashboard is Streamlit — needs the app running, cannot be emailed.

Jargon inventory from today's published files:

```
  13 MAE      12 persistence   5 E_QUANTILE   4 target_date   4 origin_date
   4 detect_lagged_copy   4 MLP   4 GBQuantile   4 C_DL   4 B_ML   4 A_STAT
   3 shuffled   3 shift diagnostic   3 lag-1   2 WITHHELD   2 P10   2 P90 ...
```

29 distinct technical terms, plus raw eight-digit currency (`44,199,748`) with no unit named.

**What the report needs:**

- **Verdict first, in one sentence.** "Today's forecast should not be used" / "Usable, with the limits below."
- **Plain-language accuracy.** "On a typical day the forecast was off by about GEL 39 million. A simple rule — just repeat the figure from five working days ago — would have been off by about GEL 61 million. So the forecast is roughly a third more accurate than that rule." No "MAE", no "skill", no "persistence".
- **Rounded, named currency.** Millions, currency stated.
- **Descriptive model names.** "Range-based model" not "E_QUANTILE / GBQuantile".
- **Ranges explained as behaviour**, with the honest caveat: "on the busiest days it landed inside only about half the time, so the range understates risk when it matters most."
- **Withheld results explained, not hidden.**
- **What changed since the last run**, and **what to do / who to contact.**
- **Single self-contained file** (inline CSS, no external assets) so it survives email.
- **Verdict not conveyed by colour alone.**
- **Explicit scope line.** Today's run covers **1 of 41 candidate targets**.

### 7.6 Prioritized readiness checklist

**P0 — blocks unattended automation**

| # | Item | Why |
|---|---|---|
| 1 | **Fail on input gaps instead of zero-filling.** Distinguish "absent row" from "true zero" at ingestion | Silent 10% baseline shift; unrecoverable |
| 2 | **Wire `data_preflight.py` into the batch path** as a hard blocker | The checks exist and the automated path doesn't use them |
| 3 | **Stop destroying the previous run.** Staging directory, validate, atomic promote | A failed re-run deletes the last good run |
| 4 | **Add a concurrency lock** (`flock`) | Overlapping triggers corrupt the directory undetectably |
| 5 | **Capture the run log** to `<run_dir>/run.log` | Every diagnostic is currently lost |
| 6 | **Enforce minimum evaluation points and `min_train_years`** | A_STAT published a full report off 40 days |
| 7 | **Exit non-zero when all families are withheld** | Currently exits 0; cron sees success |

**P1 — before any Treasury reader sees output**

| # | Item |
|---|---|
| 8 | **Provenance for all four families**: git SHA, data-file SHA-256 + row count + date range, package versions, seeds, per-family wall-clock |
| 9 | **Pin the timezone** for `RUN_DATE`; record data-file identity rather than selecting by mtime |
| 10 | **Treasury HTML report** per §7.5 |
| 11 | **Stale-data policy decision**: block or warn? |
| 12 | **Artifact validation before promotion** (§6) — the staging gate in #3 is where it belongs |

**P2 — hardening and monitoring**

| # | Item |
|---|---|
| 13 | **Drift block**: `persistence_mae_60d`, `model_mae_60d`, rolling skill, year-over-year; plus the four input-side checks |
| 14 | **Drift thresholds calibrated on DEV**, never against the 2025 holdout |
| 15 | **Restrict drift monitoring** to families passing the signal check |
| 16 | **Retention and alerting policy** |
| 17 | **Record `n_jobs` float non-determinism** in the contract so byte-comparison isn't used as a regression test |
| 18 | **Extend coverage beyond Revenues** — 40 of 41 targets unevaluated, including the stock-target paths |

**Dependencies:** item 12 needs the §6 staging decision; item 13 needs stable published artifacts, so
it follows 12. Items 1, 2, 5 and 6 are independent and remove the sharpest edges on their own.

---

# 8 · Appendix

## 8.1 Files added during this review

Five test files under `backend/tests/`, all passing, each mutation-validated. No production code was
modified at any point; every mutation was reverted with `git checkout` and the working tree confirmed
clean.

| File | Tests | Locks in |
|---|---:|---|
| `test_sentinel_holdout_split.py` | 8 | M-5 wiring: sentinel scored on disjoint, embargoed, recent rows (long + short history) |
| `test_failure_mode_distinctness.py` | 10 | M-5 inversion: ratio direction and three separate gate reasons |
| `test_published_baseline_is_shared.py` | 5 | C-2: the published persistence number is the shared function's |
| `test_overfit_ratio_recording.py` | 6 | M-4 producer: zero train MAE → infinite ratio → excluded |
| `test_no_signal_end_to_end.py` | 6 | M-5 chain: pipeline → artifact → summary → accurate reason |

Suite: **158 → 193 passed.**

## 8.2 Cross-cutting themes

**One ruler, four measurements.** The most consequential finding, recurring in §1, §2, §3 and §4:
`compute_persistence_baseline` is genuinely shared, and the four families still produce four different
baseline numbers because they disagree on the evaluation window and on what a "step" means. Two of the
four agree to the cent; C_DL is off by a five-year window and E_QUANTILE by a calendar-vs-business-day
horizon. Every cross-family comparison in `BACKTEST_REPORT.md` inherits this.

**Detection layers that compute the right thing and publish it nowhere.** `is_persistence_like` is
computed correctly and never read. `overfit_ratios` and `overfit_excluded_models` are written to JSON
with zero consumers. `check_feature_leakage` works but has no production caller. `evaluation_windows.py`
states a discipline nothing enforces. In each case the analysis exists and the wiring does not.

**Marginal numbers hiding conditional failures.** GBQuantile's 78% coverage is 100% in the middle
cancelling 51.5% at the top. B_ML's 36.75% skill is regression to the mean against an inflated
baseline. C_DL's +10.84% is a five-year average concealing −5.19% on the reporting window. In all
three the headline is arithmetically correct and substantively misleading.

**Stale artifacts presented as current.** `backend/forecast_runs/` contains runs predating the M-4 and
M-5 commits with no version marker; `master_daily_clean_treasury.csv` was produced by an older
preprocessing path and contains 1,095 fabricated weekend values; `backend/LEAKAGE_AUDIT.md` asserts
"PASS — no data leakage found" and cites a line the C-3 fix changed. Nothing in the repo distinguishes
a current artifact from an obsolete one.

**Measurement beat inspection three times.** E_QUANTILE's missing embargo looked like a serious leak
and is worth 0.24pp. A test fixture written to catch the pre-M-5 sentinel bug could not detect it until
a short-history case was added. A documented hazard about merge ordering turned out not to exist. Each
correction is recorded in place above.

## 8.3 Open decisions

1. **§3** — which interval phases to implement (P1 alone, or P1 + P2.1 CQR).
2. **§6** — Option A (validate in place) or Option B (staging and atomic promote; recommended), and
   whether Phase 2's file-content changes need the Agent team's sign-off first.
3. **§5** — whether to add failing tests that pin the four known-unfixed bugs, or fix them first.
4. **Sequencing** — whether to regenerate `master_daily_clean_treasury.csv` before anything else, since
   every metric in this document moves when it changes.
