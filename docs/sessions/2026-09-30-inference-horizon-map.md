# Inference and horizon map: how new data becomes a forecast, and where h=5 lives

**Date:** 2026-09-30
**Scope:** read-only mapping of the working tree at `12d3555` (branch `fix/ui-rendering-and-script`).
No code was changed, no pipeline was run, no artifact was written. One read-only call was made in the
backend interpreter to list business days after a date with the repository's own calendar function
(§1.4); it imports the calendar modules only and writes nothing.
**Relies on:** `docs/sessions/2026-09-29-lab-audit.md` for the selection guard's behaviour, the B_ML
bound, the family order, E_QUANTILE's open decision, and the clean forward run. None of that is
re-demonstrated here.

> **Correction, 2026-09-30 (same day, later session).** This record originally stated in §1.4,
> step A3, that `forecasts/published/` is empty on this machine and that `list_published` returns
> nothing. **Both statements were false.** The store holds three issues (2025-08-06, 2026-08-13,
> 2026-08-16; 54 files; 25 forecast rows) and `list_published()` returns all three. The record
> asserted this without checking: the directory listing it was based on had been cut short by a
> `head` filter after the parent-directory line, and the truncated output was read as an empty
> directory. The wrong sentences are left in place below, struck through, with the correction
> beside them. The full inventory and the scoring-loop exercise are in
> `2026-09-30-scoring-loop-audit.md`.

Two questions were asked. The first is about a client with a trained champion and a newer data file.
The second is about the five-working-day horizon. Each answer names the file and line it rests on.

---

## Question 1 — new data to a forecast with an existing champion

### 1.1 How new data enters

There is one file every pipeline, the scorer and the forward runner read:
`backend/data/processed/master_daily_clean_treasury.csv` (`backend/ingest_actuals.py:52-53`,
`backend/run_forward_forecast.py:26`, `frontend/pages/07_Forecast.py:155`,
`frontend/pages/08_Scorecard.py:69`, `scripts/run_daily_forecast.sh:70-73`). Four entry points reach
it or sit beside it.

| # | Entry point | Where the upload lands | What happens next | Reaches the canonical file? |
|---|---|---|---|---|
| 1 | **Scorecard page upload** — `frontend/pages/08_Scorecard.py:522-530` | `frontend/runs_uploads/actuals/<uploaded name>` (`:70`, `:528-530`) | `ingest_actuals.validate` (`:532-534`); on a ticked confirmation and a button press, `ingest_actuals.install` (`:564-570`); then `score_published` (`:579-582`, via `load_scoring` at `:115-128`) | **Yes.** `install` copies the current file to `backend/data/processed/backups/<stem>.<UTC stamp>.csv` and then copies the upload over the canonical path, byte for byte (`backend/ingest_actuals.py:57`, `:288-292`, `:317-323`) |
| 2 | **Command line** — `backend/ingest_actuals.py --file X --install [--score]` (`:350-384`) | wherever `--file` points | same `validate` and `install`; `--score` calls `score_published` (`:372-378`) | **Yes**, same path as 1 |
| 3 | **Lab page upload** — `frontend/pages/03_Lab.py:406-410` | `frontend/runs_uploads/uploaded.csv` (`:408`); optionally a tail sample at `runs_uploads/quick_sample.csv` (`:425`) | the file is passed to the chosen runner as `TG_DATA_PATH` for that one exploratory run (`:429-432`, `:446`, `:794`) | **No.** The canonical file is never read or written by the Lab (`README.md:289-290`) |
| 4 | **Data Preprocessing page** — `frontend/pages/02_Data_Preprocessing.py:96-121` | `frontend/runs_uploads/<uploaded name>` (`:118-121`), or a path typed in (`:106-124`) | `backend/run_preprocess.py` writes the processed dataset to `data_preprocessed/<variant>/` (`02_Data_Preprocessing.py:233`, `:259`, `:268`; `backend/run_preprocess.py:35-37`) | **No.** Nothing under `backend/preprocessing/` names `backend/data` or `data/processed` (grep, no matches). A processed file has to go through entry point 1 or 2, or be copied by hand |

Two further routes bypass the ingest checks entirely, and are named here because they exist:

* The daily script reads `backend/data/processed/$DATA_FILE_NAME`, or any path in `TG_DATA_PATH`
  (`scripts/run_daily_forecast.sh:70-74`). It refuses a missing file and records the SHA-256, but
  does no schema or date check of its own (`:75-90`).
* `backend/forecast_modes.py --data <path>` accepts any CSV (`:505`, `:517`, `:532`). The Forecast
  page always passes the canonical path (`07_Forecast.py:416`, `:460`).

**What `validate` checks before anything is written** (`backend/ingest_actuals.py:166-285`): a
`date` column (`:189-195`); dates that parse (`:197-208`); no repeated date (`:210-215`); every
column the current file has (`:225-232`); a last date strictly later than the one held (`:242-248`);
different bytes (`:250-255`). Revised values on dates already held are counted and shown as a
warning, never refused (`:257-263`). It does not fit, choose or touch a window (`:33-38`).

**The landing paths are git-ignored.** Verified with `git check-ignore -v`:

```
.gitignore:114:data/                        backend/data/processed/master_daily_clean_treasury.csv
.gitignore:114:data/                        backend/data/processed/backups/x.csv
.gitignore:11:frontend/runs_uploads/        frontend/runs_uploads/actuals/x.csv
.gitignore:15:data_preprocessed/            data_preprocessed/x
.gitignore:24:backend/forecast_runs/**      backend/forecast_runs/forward/latest/forward_forecast.csv
.gitignore:72:forecasts/published/          forecasts/published/2026-09-30/forecast.csv
```

`git ls-files backend/data data_preprocessed frontend/runs_uploads` returns nothing. The README says
the same (`README.md:285-287`): any directory named `data` is ignored, so the file cannot be
committed by accident. The vault that keeps the real audit trail is ignored as a whole
(`.gitignore:113`), and so is the experiments log (`:93-94`) and the holdout ledger (`:99`).

### 1.2 Can the system forecast on newer data without retraining or reselecting?

**Reselecting: never.** Which model is champion is a hand-edited fact in `registry/recipes.json`;
nothing in the project writes that file (`backend/registry.py:157-160`, `:224-227`;
`docs/REFRESH_AND_RETRAIN.md:54-60`). `champion_policy()` states `reselection: "none"` and
`on_new_data: "refit_only"` (`backend/registry.py:173-175`, `:203-206`).

**Retraining: always.** There is no path that loads a saved model and predicts on newer data.
The official forward path refits from scratch on every run:

* `forecast_modes.official_run` (`backend/forecast_modes.py:320-357`) reads the recipe
  (`:339-346`), reads the CSV (`:347`) and calls `forward_forecast.run_forward` (`:349`).
* `run_forward` (`backend/forward_forecast.py:220-313`) is documented as *"Fit one model per
  horizon on all available history; predict the final origin"* (`:225`). For each of the five
  horizons it calls `_fit_predict_point`, which does `est.fit(X_tr, y_tr)` and then
  `est.predict(X_new)` in the same call (`:186-188`), and `_fit_predict_quantiles`, which fits
  three `GradientBoostingRegressor` objects the same way (`:210-215`). That is twenty fits per
  target per run, sixty for the three registry targets.
* `backend/run_forward_forecast.py:45-82` does the same for every registry recipe and writes to
  `backend/forecast_runs/forward/latest` (`:73`; `forward_forecast.py:53`).

**What a "champion" is.** A recipe: a model name, hyperparameters, feature groups, an exogenous block
list and a target transform (`registry/recipes.json:15-44`). It carries no fitted weights.

**The persisted estimators are not an inference path.** `backend/estimator_store.py` saves the fitted
objects beside a published issue, but its own docstring says *"Official runs still refit on current
data — nothing here serves a stale model"* (`:5`, `:288-289`). The only reader,
`reproduce_prediction` (`:391-469`), rebuilds the design matrix from the data file (`:422`),
refuses unless the issue's own origin date is present in it (`:434-438`), predicts at that one
origin (`:446`) and compares the result with the number that was published (`:448-455`). It cannot
be pointed at a later origin.

So the plain answer is: **no such path exists.** The closest thing is a refit at zero selection
cost, which is what an official run is.

### 1.3 How the official forward path avoids the selection guard

**Where the guard is called.** `assert_selection_free` is defined at
`backend/evaluation_windows.py:206-249` and is invoked from three places:

| Caller | Line | What is being chosen |
|---|---|---|
| B_ML backtest, immediately before `select_best_model` | `backend/b_ml_pipeline.py:1090-1094` | the family's best model |
| E_QUANTILE backtest, immediately before its best-model choice | `backend/e_quantile_daily_pipeline.py:1083-1087` | the family's best model |
| `rolling_origin_folds(window="live")` | `backend/evaluation_windows.py:348-354` | search folds |

`backend/forward_forecast.py` does not import `evaluation_windows` at all, and
`backend/forecast_modes.py` imports only two fold-size constants from it (`:137`). The forward path
therefore never meets the guard. It does not need to, and the reason is the guard's own contract:
*"Call this from any path that chooses something"* (`evaluation_windows.py:209-210`). The forward
path chooses nothing. The model is fixed by the registry (`forecast_modes.py:339-346`), the horizon
is fixed at five and any other is refused (`:331-333`), and the features come from the recipe.

**The three acts, kept apart.**

| Act | Where it happens on the official path | Guarded? | What the sealed and live rows are to it |
|---|---|---|---|
| **Selection** — choosing a model, recipe, hyperparameter or threshold | Nowhere. It happened once, on TRAIN and DEV, in the workstreams (`scripts/ws2_tune.py:73-80` bounds folds to those windows) | yes, where it does happen | forbidden as evidence |
| **Refit on the official run** — fitting the chosen recipe on everything through the data end | `forward_forecast.py:187` (point model) and `:213` (each quantile) | no | **training rows.** `usable = X.notna().all(axis=1) & y_h.notna()` (`:266-267`) keeps every row whose label is known, so the 2025-01-01..2025-08-06 rows, and any LIVE rows, become training labels |
| **Inference** — one prediction at the final origin | `forward_forecast.py:188` and `:214`, at `X_new = X.loc[[origin]]` (`:268`), where the origin is the last row with a complete feature set (`:248-251`) | no | not read. Every target date is strictly after the data end (`:255-256`, `:310`), and no truth column can exist (`:311-312`) |

Using sealed rows as training labels is within the rule the windows module states for itself: *"TRAIN
is a floor, not a cap … What never happens is training on data at or after the origin it is
predicting from"* (`evaluation_windows.py:35-38`). What the holdout protects against is a **choice**
informed by 2025, and the forward path makes none.

One thing to read as it is: `"test_window_touched": False` in the provenance is a literal
(`forward_forecast.py:335`), not a measurement. It is a statement of the design above, and nothing
computes it.

**Does this path serve "new CSV in, forward forecast out" for a user today? Yes, by two routes.**

1. **In the app.** Upload on the Scorecard page and confirm (§1.1, entry point 1). Then on the
   Forecast page choose *Official* (`07_Forecast.py:381-387`), pick a target, and press *Run the
   champion recipe* (`:413`). The page dispatches
   `backend/forecast_modes.py --mode official --target <T> --data <canonical csv>` in the backend
   interpreter (`:188-205`, `:416`). With the *Publish* box ticked it appends `--publish`
   (`:406`, `:417-418`), and `publish_official` writes `forecasts/published/<issue date>/`
   and mirrors it to `private_vault/published/` (`forecast_modes.py:441-442`, `:450-460`;
   `backend/published_forecasts.py:390-478`, `:471-473`; the vault directory is created if
   absent, `:347`).
2. **On the command line.** `backend/forecast_modes.py --mode official --target <T> --data <any csv>
   [--publish] [--issue-date D]` (`:498-543`). `--data` is any path, so the ingest checks can be
   skipped. Or `backend/run_forward_forecast.py [--publish]`, which reads the fixed canonical
   path (`:26`) and forecasts all three registry targets (`:46-56`).

Three facts about these routes that a user would otherwise discover the hard way:

* **The reading tab does not update from a page-launched run.** The Forecast page's *read* tab shows
  `backend/forecast_runs/forward/latest` (`07_Forecast.py:88-91` →
  `backend/insights.py:347-363` → `forward_forecast.DEFAULT_OUT`). Only
  `run_forward_forecast.py:73` writes there. The page's official run writes nothing without
  `--publish`, and with it stages under `forward/staging/<issue>--<target>` and removes the staging
  directory on success (`forecast_modes.py:435-446`). The artifact on disk today is dated
  2025-08-06 (`backend/forecast_runs/forward/latest/forward_provenance.json`, `latest_data_date`).
* **The two publish paths differ on verdicts.** `publish_official` refuses a target whose registry
  verdict is `withheld` (`forecast_modes.py:400-412`). Today that is Expenditure and State budget
  balance (`registry/recipes.json:229`, `:332`); only Revenues is `publishable` (`:110`).
  `run_forward_forecast.py --publish` calls `published_forecasts.publish` directly (`:95`) and
  makes no such check.
* **The daily script does not produce a forward forecast.** `scripts/run_daily_forecast.sh` runs four
  backtest runners (`:122-167`) and a summary (`:176`). The word "forward" does not appear in it or
  in `scripts/daily_summary.py`.

### 1.4 Walk-through: a CSV ending 2026-09-30, standard flow, to the first failure or a forecast

Assumed: a CSV with the same columns as the current file, business days through Wednesday
2026-09-30. Two flows are called "standard" in this repository, so both are walked. The dates in
step A6 come from the repository's own calendar function, called once read-only.

#### Flow A — upload on the Scorecard page, then an official run on the Forecast page

| Step | What happens | Where | Outcome |
|---|---|---|---|
| A1 | The file is saved under `frontend/runs_uploads/actuals/` and validated | `08_Scorecard.py:528-534`; `ingest_actuals.py:166-285` | passes: has `date`, unique dates, every current column, last date 2026-09-30 > 2025-08-06, different bytes. Revised historical values are counted and shown as a warning |
| A2 | Confirmation ticked, button pressed → `install` | `08_Scorecard.py:564-570`; `ingest_actuals.py:295-339` | the current file is copied to `backend/data/processed/backups/master_daily_clean_treasury.<UTC stamp>.csv`, then the upload replaces the canonical file |
| A3 | Published forecasts are scored | `08_Scorecard.py:579-582` → `published_forecasts.score_published` | ~~`forecasts/published/` is empty on this machine and absent in a clone (`.gitignore:72`), so `list_published` returns nothing (`published_forecasts.py:481-485`): 0 scored, 0 pending, the scorecard is rewritten header-only (`:757-761`).~~ **Corrected 2026-09-30:** the store holds three issues and 25 rows, all with target dates 2025-08-07..2025-08-13. With a file ending 2026-09-30 every one of those rows has its truth, so `score_published` scores all 25 and writes 25 rows to `forecasts/scorecard.csv` (`published_forecasts.py:757-761`), each labelled `scored_in_window: live` (`:735-736`). A fresh clone, which has no `forecasts/published/` (`.gitignore:72`), would score 0 and pend 0. No failure either way |
| A4 | Side effect on the test suite | `backend/tests/test_live_window.py:116-127` | this test asserts the canonical file has zero LIVE rows and ends on `TEST_END`. It now fails, by design; its docstring says the failure means the file grew and the record must say the LIVE figures are real (`:119-120`). The SHA pin is opt-in (`backend/provenance.py:218-220`), so runners do not refuse |
| A5 | Forecast page → Official → target → *Run the champion recipe* | `07_Forecast.py:413-419` → `forecast_modes._cli` → `official_run` | recipe found, horizon 5 validated (`forecast_modes.py:328-333`) |
| A6 | `run_forward` builds the design and the forward dates | `forward_forecast.py:235-256` | index reindexed to business days through 2026-09-30 (`b_ml_pipeline.py:178-188`). Holidays are computed per year by formula, valid 1900-2099 (`backend/preprocessing/holidays.py:4`, `:49-98`, `:113`); fiscal features pad one year either side (`fiscal_calendar.py:433`, `:454-455`). No table needs extending. Origin = 2026-09-30. Target dates = 2026-10-01, 10-02, 10-05, 10-06, 10-07 (repository calendar; Mtskhetoba on 10-14 is outside the window). `assert_forward_only` passes |
| A7 | Twenty fits per target, one prediction each | `forward_forecast.py:259-273` | **a forecast is produced**: five rows per target with P10/P50/P90, provenance, and the 2024 DEV gate verdicts attached by recipe id (`forecast_modes.py:351-354`) |
| A8 | If *Publish* was ticked | `forecast_modes.py:388-461` | Revenues publishes to `forecasts/published/2026-09-30/` (issue date is wall-clock, `:464-482`) and to the vault, with twenty estimator blobs under `estimators/`. Expenditure and State budget balance are **refused** as `withheld` (`:406-412`); the page shows the reason (`07_Forecast.py:420-422`) |

**Flow A ends in a produced forecast.** Nothing crashes. The two things that change state
unexpectedly are the failing `test_live_window` assertion and the reading tab still showing the
2025-08-06 artifact.

#### Flow B — the daily script, default settings

`scripts/run_daily_forecast.sh` picks the canonical file (`:70-74`), records its hash, wipes and
recreates `backend/forecast_runs/2026-09-30/` (`:105-112`), and runs the families in order
`A_STAT B_ML C_DL E_QUANTILE` under `set -euo pipefail` (`:29`, `:50`, `:170-172`).

| Family | Window it evaluates on with data to 2026-09-30 | Guarded? | Outcome |
|---|---|---|---|
| **A_STAT** (`folds=1, min_train_years=4`, no bound; `:130-134`) | `_yearly_folds` builds one fold per year 2019..2026 and keeps the last (`backend/run_a_stat.py:107-128`, `:126-127`): train ≤ 2025-12-31, evaluate 2026-01-01..2026-09-30. **Every evaluation row is LIVE.** | No. Only `test` dates are ledger-logged (`:471-479`), and there are none. The runner says why there is no selection guard: one model per invocation, ranked against persistence (`:468-470`) | **exit 0.** Numbers computed entirely on post-seal data; nothing in the family's own artifacts names the window |
| **B_ML** (`folds=1, min_train_years=4, eval_end=2024-12-31`; `:135-142`) | `build_yearly_folds` drops 2025 and 2026 because their start is after the bound (`b_ml_pipeline.py:262-263`) and keeps 2024 (`:273-274`): train ≤ 2023-12-31, evaluate 2024 | Yes, and it passes: every evaluation row is DEV | **exit 0.** The new rows never enter this run: training stops at the fold's train end (`:773-774`) and prediction features stop at the last 2024 origin (`:821-823`) |
| **C_DL** (`run_c_dl_quick_univariate.py:16` → `run_c_dl_univariate.py`; `eval_start` defaults to `TEST_START`, no `eval_end`; `:85`, `:88`) | `build_yearly_folds` keeps the 2025 fold trimmed to start 2025-01-01, and a 2026 fold (`c_dl_pipeline.py:398-430`) | No selection guard in this family (`:437-440`); the 2025 dates are logged as a report read (`:441-448`) | **exit 0** (a `FAILED_QUALITY` status is not an exit code, per the audit). 2026 rows are evaluated as LIVE and not labelled as such in this family's outputs |
| **E_QUANTILE** (`eval_start=2025-01-01`, no `eval_end`; `:143-148`) | `_time_folds` tiles five-row blocks forward from 2025-01-01 to the end of the file (`e_quantile_daily_pipeline.py:100-112`); the 2025 dates are logged as a report read (`:825-840`) | Yes: `assert_selection_free` runs before the best-model choice (`:1083-1087`) | **exit 1.** `SelectionOnReportOnlyDataError` from `evaluation_windows.py:242`, now naming both `test` and `live`. The script stops before `daily_summary.py` (`:176`), so no `SUMMARY.txt` or `SUMMARY.json` is written |

**Flow B's first failure is E_QUANTILE**, the same abort the audit recorded, with `live` added to
the message. Three families complete before it. Had all four completed, the run would still hold no
forward forecast (§1.3, last bullet): the script's product is four backtests and a summary.

---

## Question 2 — the horizon

### 2.5 Every place h = 5 working days is assumed

Grouped by layer. A place is listed where the number five is written down, or where a quantity is
derived from it. Places that merely pass `horizon` through as a parameter are not listed, with a few
exceptions marked because they are where a change would have to be made.

**Configuration and constants**

| File:line | What |
|---|---|
| `backend/forecast_modes.py:33` | `VALIDATED_HORIZON = 5`, "the only horizon at which the ruler, recipe selection and gates were measured" (`:16-22`). Official mode refuses any other (`:305-317`, `:331-333`) |
| `frontend/pages/07_Forecast.py:161` | a second, independent `VALIDATED_HORIZON = 5`. Drives the caption (`:395-396`), the exploratory slider's default and warning (`:450-454`) |
| `backend/forward_forecast.py:48` | `FORWARD_HORIZONS = (1, 2, 3, 4, 5)`; feeds the design config (`:123`), the stock level divisor (`:244`) and the date count (`:255`) |
| `scripts/run_daily_forecast.sh:53` | `TG_HORIZON` default 5 (`:19`) |
| `backend/run_b_ml_univariate.py:19`, `backend/run_e_quantile_daily_univariate.py:16`, `backend/run_c_dl_univariate.py:51`, `backend/run_foundation.py:67` | runner defaults of 5 when `TG_HORIZON` is unset |
| `backend/run_a_stat.py:398` | runner default of **6**, the odd one out. Harmless today because every caller sets `TG_HORIZON` |
| `registry/recipes.json:30`, `:143`, `:253` | `params.horizon: 5` on each recipe; `:56` labels the ruler "h=5 business-day persistence" |
| `backend/sealed_window_report.py:96` | `HORIZON = 5`, "the horizon the recipes are credentialed at" |
| `scripts/calibrate_sentinel.py:55` | `HORIZON = 5`: the sentinel null was drawn at this horizon (`:68`, `:93`, `:100`, `:106`) |
| `scripts/ws2_tune.py:36` | `H=5`, imported by `scripts/ws2_recompute.py:47` and `scripts/ws7_cqr.py:34` |
| `backend/c_dl_pipeline.py:100` | `horizons_daily = [1, 5, 20]` default, overridden by the runner to `[horizon]` (`run_c_dl_univariate.py:96`) |
| `backend/a_stat_models_pipeline.py:89` | `[1, 5, 20]` in a module the 2026-08-13 record lists as unreferenced |
| `frontend/backend_consts.py:80-84` | `HORIZON_PRESETS` Daily `[1, 5, 10, 20]`; `frontend/pages/03_Lab.py:486` slider default 6 |
| `frontend/pages/09_Documentation.py:346` | documents a C_DL daily horizon of 14, which matches neither the pipeline default nor the runner |

**Feature building and target construction** (parametric in `h`, tied to five by the constants above)

| File:line | What |
|---|---|
| `backend/forward_forecast.py:263-264` | `y_h = s.shift(-h)` per horizon; delta for stocks |
| `backend/b_ml_pipeline.py:801-803` | `y_target_full = s_train_full.shift(-h)`; origin at `pos_t - h` (`:885-888`) |
| `backend/e_quantile_daily_pipeline.py:250-256` | `y_target[i] = y[i + horizon]`; test blocks are `horizon` rows long (`:74`, `:106`); `min_train_rows = max(min_train*252, horizon, 30)` (`:90`) |
| `backend/c_dl_pipeline.py:536-545` | label at `end_i + horizon` |
| `backend/run_a_stat.py:495-504` | origin at `pos_t - horizon`; a `horizon`-step path is requested and its last value kept |
| Embargo and gap sizes | `evaluation_windows.py:361`, `:381` (fold embargo = horizon); `backend/tuning.py:86-101` (`gapped_split`); `backend/conformal.py:258-273` (`causal_calibration_split`); `scripts/ws2_tune.py:68-70`, `:81` (target-date map at H); `backend/sealed_window_report.py:117-119`, `:154-155` |
| `backend/forecast_modes.py:129-139` | `min_history_rows = DEFAULT_MIN_TRAIN + horizon + DEFAULT_EVAL_BLOCK` = 1008 + 5 + 126 = 1139 rows for a target to be eligible |

**Fives that are not the horizon, and are easy to confuse with it**

| File:line | What it is |
|---|---|
| `backend/evaluation_windows.py:397-409` | `seasonal_naive_scale(season=5)`: the MASE denominator is a one-business-week naive, "one business week on this index" |
| `backend/forecast_integrity.py:301-303` | `SEASONAL_NAIVE_SEASON_STEPS = 5`; `:313-340` handles the coincidence explicitly and reports the seasonal-naive MAE as degenerate when `season == horizon` (`:332`) |
| `backend/b_ml_pipeline.py:113` | `exog_lags = (1, 5, 21)`, a lag choice |
| `backend/run_e_quantile_daily_univariate.py:29-30` | lags `[1, 5, 20]`, windows `[5, 20]`, feature choices |
| `frontend/backend_consts.py:72` | `QUALITY_GATE_SKILL_PCT = 5.0`, a percentage |
| `frontend/data_preflight.py:74` | `horizon * 3 + 30` minimum rows, parametric |

**Training**

The official forward fits one model per horizon in 1..5 (`forward_forecast.py:16-20`, `:259-273`).
Each backtest runner trains at the single `TG_HORIZON` it is given (`b_ml_pipeline.py:757`,
`run_a_stat.py:398`, `run_c_dl_univariate.py:96`, `run_e_quantile_daily_univariate.py:16`).

**Evaluation and scoring**

| File:line | What |
|---|---|
| `registry/recipes.json:48-56` and the two other `dev_credentials` blocks | `dev_mae`, `mase`, `skill_vs_ruler_pct`, `sentinel_ratio` are h=5 measurements |
| `backend/forward_forecast.py:407-441` | `benchmark_mae_for_target` reads the h=5 ruler from the logged run; shown on the Forecast page (`07_Forecast.py:571-580`) |
| `backend/published_forecasts.py:565`, `:617-620` | `horizon_steps: int = 5` as a last-resort fallback; the scorer reads each row's own horizon (`:542`) and records the defect the fixed 5 once caused (`:528-532`) |
| `backend/ops_baseline.py:32-38` | the vintage rule is reasoned for "four sealed-window dates at h=5" |
| `docs/AGENT_ARTIFACT_CONTRACT.md:68` | `horizon` is a string field in `SUMMARY.json`, e.g. `"5"` |
| Tests | 32 test files name the horizon. Literal pins: `backend/tests/test_forecast_modes.py:147` (`range(1, 6)`), `test_artifact_validation.py:412` (`range(1, 6)`), and `HORIZON = 5` in `test_overfit_ratio_recording.py:48`, `test_failure_mode_distinctness.py:41`, `test_no_fold_reads_holdout_truth.py:47`, `test_no_signal_end_to_end.py:53`, `test_published_baseline_is_shared.py:46`, `test_sentinel_holdout_split.py:47`, `test_unified_baseline.py:30` |

**Gates**

| File:line | What |
|---|---|
| `backend/publication_gates.py:94` | `MASE_MAX = 1.0`, definitional and horizon-free (`:28-34`), but the measured MASE it is compared with is an h=5 number |
| `backend/publication_gates.py:97`, `:107-118` | `SENTINEL_MIN = 1.15`, chosen from a 360-draw null (`:50-75`) that `scripts/calibrate_sentinel.py` drew at `HORIZON = 5` (`:55`). The threshold has no measured false-positive rate at any other horizon |
| `backend/publication_gates.py:104` | `OVERFIT_MAX = 3.0`, horizon-free |
| `backend/publication_gates.py:306-308` | coverage band = nominal ± 0.10, horizon-free; the measured coverage is per-horizon |
| `registry/recipes.json:57-105` and the two other `gates` blocks | every verdict is an h=5 verdict, attached unchanged to forward runs (`forecast_modes.py:351-354`; `run_forward_forecast.py:65-71`) |

**User interface and copy**

| File:line | What |
|---|---|
| `frontend/pages/07_Forecast.py:1` | "the next five working days"; `:511` reads the count from the artifact; `:621`, `:647` "all five days"; `:742` "five working days ago" |
| `frontend/pages/08_Scorecard.py:103`, `:475` | "five working days earlier" |
| `frontend/ui_styles.py:875`, `:913`, `:1025`, `:1030-1031`, `:1072-1076` | glossary and help text; `:1075` "Official forecasts use h=5 and only h=5" |
| `frontend/translations_ka.py:285`, `:292-293`, `:463` | Georgian renderings of the same sentences, keyed on the English text |
| `frontend/pages/01_Start_here.py:220` | "the one horizon everything here was measured at" |
| `backend/insights.py:118`, `backend/forward_forecast.py:388-392` | narrative strings: "one of the five days", "all five days are forecast from the same origin" |
| `backend/run_forward_forecast.py:3` | "the next five Georgian business days" |
| `frontend/pages/03_Lab.py:147-152` | help text uses 14-day and 6-month examples |

### 2.6 Direct or recursive? The code

**The official forward is direct, one model per horizon.**

`backend/forward_forecast.py:16-20`:

```
**One model per horizon.** The rest of the project fixes h=5 and scores the fifth business
day. A forecast the Treasury can use has to cover every day between now and then, so for
each horizon h in 1..5 a separate model is fit on rows where ``y(t+h)`` is known and asked
for exactly one prediction from the final origin. Five fits, five dates. Reusing the h=5
model for nearer days would silently misstate what each number means.
```

and the loop that does it, `:259-273`:

```python
for h in horizons:
    y_h = s.shift(-h)
    y_h = (y_h - s) if stock else y_h
    usable = X.notna().all(axis=1) & y_h.notna()
    X_tr, y_tr = X[usable], y_h[usable]
    X_new = X.loc[[origin]]
    p50_raw, point_est = _fit_predict_point(X_tr, y_tr, X_new, champ.point_model, ...)
    qs, quantile_ests = _fit_predict_quantiles(X_tr, y_tr, X_new, ...)
```

No prediction is fed back as an input. `X_new` is the origin row and nothing else (`:268`).

**B_ML is direct at one horizon.** `backend/b_ml_pipeline.py:801-803`:

```python
# Step-based h-step-ahead target (vectorized).
# shift(-h) places s[i+h] at position i; last h rows become NaN.
y_target_full = s_train_full.shift(-h)
```

**E_QUANTILE is direct.** `backend/e_quantile_daily_pipeline.py:250-256`:

```python
# y_target[i] = y[i + horizon]  (positional offset, not calendar-day).
h = cfg.horizon
...
for i in range(len(y) - h):
    y_target.iloc[i] = y_vals[i + h]
```

**C_DL is direct.** `backend/c_dl_pipeline.py:6`: *"Direct forecasting per horizon"*, and
`:536-545`: the label is `y.iloc[end_i + horizon]` for a window ending at `end_i`.

**A_STAT is the exception: a multi-step path, with the h-th step kept.** `backend/run_a_stat.py`
asks the fitted statsmodels object for `n = len(idx)` steps (`:260`), where `idx` is the h future
dates (`:500`), and keeps the last (`:503-504`):

```python
y_pred, y_lo, y_hi = _fc(model, y_hist, idx_future, ov, cadence)
yp = float(np.asarray(y_pred).ravel()[-1])        # the h-step-ahead point
```

ETS does `fit.forecast(n)` (`:299`), SARIMAX and STL use `get_forecast(n)` (`:347`, `:358`),
Theta `forecast(n)` (`:370`). How each reaches step h is the model's own recursion inside
statsmodels. NAIVE and MOVAVG repeat one constant (`:264`, `:277`).

**Foundation models** likewise request `horizon` steps and keep the last
(`backend/run_foundation.py:13`, `:134-135`; `backend/foundation_models.py:282-315`).

### 2.7 What ten working days would require

**As an experiment: nothing.** Exploratory mode already runs any horizon from 1 to 10. The Forecast
page's slider goes to 10 (`07_Forecast.py:450`), `exploratory_run` builds horizons 1..h
(`forecast_modes.py:382-383`), the result carries a banner and has no publish path (`:50-74`,
`:414-417`). The daily script accepts `TG_HORIZON=10` for backtests.

**As an official forecast: a selection exercise, not a configuration change.** Everything that
makes a forecast official was measured at five: the ruler, the recipe selection and every gate
(`forecast_modes.py:16-22`). Re-choosing needs "a TRAIN/DEV selection pass (never LIVE)" and "a
hand-edited registry/recipes.json — nothing writes it" (`backend/registry.py:224-227`;
`docs/REFRESH_AND_RETRAIN.md:84-87`). The work, in the order it has to happen:

| Step | What it involves | Where it lands | Estimate |
|---|---|---|---|
| 1. Constants, copy, tests | change every entry in §2.5's first and last tables, including both `VALIDATED_HORIZON` copies; update the 32 test files that pin five; re-key the Georgian strings, which are looked up by their English text | the files named in §2.5 | 1–2 days |
| 2. Selection pass at h=10 on TRAIN and DEV | rerun the objective comparison (36 logged runs at h=5, `reports/ws1_objectives.md:5`), the feature-group ablation, the target-scaling and the exogenous-block studies, or an agreed subset, per target. The scripts pin `H=5` (`scripts/ws2_tune.py:36`) and write to `experiments/log.csv`, which a clone does not have (`.gitignore:93-94`). The DEV-fold rule that keeps an evaluation row's truth inside an allowed window (`ws2_tune.py:87-110`) drops `2h` rows at each fold edge, so DEV shrinks further at h=10 | new `run_id`s in `experiments/log.csv`; new `dev_credentials` | 2–4 days |
| 3. Sentinel null at h=10 | 120 draws per target per null construction (`reports/sentinel_calibration.md:4`, `:48`) with `HORIZON` changed at `scripts/calibrate_sentinel.py:55`; pick a threshold with a measured false-positive rate; update `SENTINEL_MIN` and `SENTINEL_NULL` (`publication_gates.py:97`, `:107-118`) | `reports/`, `publication_gates.py` | 0.5–1 day |
| 4. Gates and registry | measure MASE, coverage, overfit ratio, leakage and mimicry for each h=10 champion on DEV; write the verdicts and credentials into `recipes.json` by hand, with the `run_id` the loader demands (`registry.py:96-98`) | `registry/recipes.json` | 1 day |
| 5. Sealed-window report at h=10 | optional. `sealed_window_report.py` reports on TEST as a logged report read (`:122-130`); it chooses nothing, so it does not spend the one selection read. The ledger today holds no `purpose=selection` entry (`experiments/test_access.log`, 754 lines, 752 of them report reads) | `reports/` | 0.5 day |
| **Total** | | | **5–9 working days** for one person who knows the workstream scripts, on the same laptop-scale compute the workstreams used. This is an estimate from the steps listed, not a measurement. Step 2 dominates and is the one a reviewer would want to see planned before it starts |

Two things do not need changing. `business_days_after` looks 60 calendar days ahead for holidays
(`forward_forecast.py:64`), which covers ten business days. The scorer reads each row's own
horizon (`published_forecasts.py:542`), so a ten-row issue scores correctly.

**What breaks silently if someone changes only the number.** Suppose `FORWARD_HORIZONS` becomes
1..10 and `VALIDATED_HORIZON` becomes 10, and nothing else moves.

* **Credentials earned at five are attached to ten.** `official_run` attaches the recipe's DEV gates
  and status by id (`forecast_modes.py:351-354`) and reads only `params.target_transform` from the
  recipe (`:345`); nothing compares `params.horizon` (`recipes.json:30`) with the horizon requested.
  The result is exactly what the module's own docstring calls an official-looking forecast carrying
  credentials it never earned (`:18-22`).
* **The benchmark beside the forecast stays the five-day ruler.** `benchmark_mae_for_target`
  (`forward_forecast.py:407-441`) reads the logged h=5 persistence error; the page shows it next to
  a ten-day forecast (`07_Forecast.py:571-580`) with no note.
* **The sentinel threshold keeps a false-positive rate nobody measured.** 1.15 comes from a null at
  h=5 (`calibrate_sentinel.py:55`); at h=10 the null is unknown and pass/fail verdicts are
  uncalibrated.
* **A number appears that never existed before.** The seasonal-naive reference is reported as
  degenerate only when `season_steps == horizon` (`forecast_integrity.py:332`). At h=10 the
  Dashboard starts showing a seasonal-naive MAE that was never shown at five, with no caption.
* **Eligibility moves by five rows.** `min_history_rows` becomes 1144 (`forecast_modes.py:139`);
  a column near the margin can flip from eligible to ineligible with no message about why.
* **E_QUANTILE's coverage figures change for a structural reason.** Its test blocks are `horizon`
  rows long (`e_quantile_daily_pipeline.py:74`, `:106`), so the same window becomes half as many
  blocks, and `min_train_rows` shifts (`:90`).
* **Fold edges lose twice as many rows.** Every embargo and target-date rule in §2.5 doubles, so DEV
  credentials change even for an unchanged model.
* **The page and the backend can disagree.** `VALIDATED_HORIZON` is two literals
  (`forecast_modes.py:33`, `07_Forecast.py:161`). Changing one leaves the page's caption, slider
  default and warning at the other value, and nothing checks them against each other.
* **The Georgian copy goes silently English.** Translations are keyed on the English sentence
  (`translations_ka.py:285`, `:292-293`, `:463`); an edited "five" in the English drops the match
  and the page falls back to English for that sentence.
* **The narrative keeps saying five.** `insights.py:118` and `forward_forecast.py:388-392` would
  describe "five days" over ten rows.
* **Published issues mix horizons.** Each issue records its horizons in `manifest.json`
  (`published_forecasts.py:458`), so the mix is recoverable, but nothing flags it.

The one part that is not silent is the test suite: the 32 files that pin five fail loudly.

---

## What this record does not do

No fix is proposed and none was made. The findings that look like defects — the second
`VALIDATED_HORIZON` literal, the A_STAT runner's default of 6, the Documentation page's C_DL
horizon of 14, the two publish paths' different treatment of `withheld`, the reading tab that a
page-launched run never updates, and the A_STAT and C_DL families evaluating on LIVE rows without
labelling them — are recorded as the map found them. The E_QUANTILE window remains the open decision
the 2026-09-29 record left it as.

> **Update, 2026-10-01 (branch `fix/scoring-trust-batch`).** Three things above have since
> changed. The two publish paths now share one verdict guard, `forecast_modes.refuse_withheld`,
> and the runner refuses what the page refuses. `official_run` refuses a recipe whose
> `params.horizon` differs from `VALIDATED_HORIZON`, with a reason naming both numbers, which
> closes the first bullet of §2.7's "what breaks silently". And the E_QUANTILE window is decided
> as Option A: the daily script bounds the family to DEV (`2024-01-01 .. 2024-12-31`, 262
> evaluation points), so selection never reads the holdout. Details and clone evidence in
> `2026-09-30-scoring-loop-audit.md`, "Fixes applied 2026-10-01". The other findings in this
> section stand.
