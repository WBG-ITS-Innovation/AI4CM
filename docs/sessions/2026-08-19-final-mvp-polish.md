# Session record: crash fix, ops baseline verdict, registry widening, foundation forecasters

**Date:** 2026-08-19 · **Branch:** `model/excellence` · **PR:** #26 (open, not merged)
**Scope delivered:** Tasks 0, 1, 2, 3 · **Deferred:** Tasks 4 to 9

---

## 1. What this session was asked for, and what it did

Ten tasks were specified. Four were completed. Six were deferred by an agreed split, taken at
plan time on the grounds that ten tasks across ten pages, a README rewrite, eleven model
registrations with end-to-end smoke tests and three foundation-model installs would not fit one
session honestly. The split was placed after Task 3 so that all backend work landed together, and
because Task 7 renames the page files that Tasks 4, 5, 6 and 8 write copy into: doing 8 before 7
would mean writing the same copy twice.

| Task | State | One line |
|---|---|---|
| 0 · Forecast page crash | **Done** | Root cause was name shadowing, not i18n. Fixed at the root, plus an app-wide lint |
| 1 · Flat Ops baseline | **Done** | Verdict: display bug. Scoring unaffected. Two defects, both fixed in the frontend |
| 2 · Widen the registry | **Done** | 11 models across 3 families, 11/11 smoke-passed. Found 10 models the Lab could not reach |
| 3 · Foundation forecasters | **Done** | Chronos and TimesFM registered and pinned. Lag-Llama attempted and dropped |
| 4 · Forecast page tabs | **Deferred** | |
| 5 · Scorecard rework | **Deferred** | |
| 6 · Lab downloads | **Deferred** | Investigation done, see §8 |
| 7 · Nav reorder | **Deferred** | Order confirmed, no files renamed |
| 8 · Whole-app clarity | **Deferred** | 7 new strings bypass i18n, see §9 |
| 9 · README rewrite | **Deferred** | |

### Test counts

| Command | Before | After |
|---|---|---|
| `./backend/.venv/bin/python -m pytest -q` (repo root, both suites) | 1209 passed, 9 skipped | **1322 passed, 20 skipped** |
| `./frontend/.venv/bin/python -m pytest frontend/tests -q` | 482 passed, 7 skipped | **528 passed, 18 skipped** |
| `./backend/.venv/bin/python -m pytest backend/tests -q` (backend alone) | 1092 passed | 1114 passed |

A note on the baseline, because two numbers were in play. `pytest.ini` sets `testpaths` to both
`backend/tests` and `frontend/tests`, so the repo-root command collects both and the 1209 figure
from the previous record comes from there: 1092 backend plus 117 of the frontend tests that run
under the backend interpreter. The 482 figure is the frontend suite under its own interpreter,
where streamlit exists. Both were re-measured at the start of this session and both matched.

### Commits

| SHA | Subject |
|---|---|
| `82268fb` | Forecast page crashed on render: the translator was shadowed by a target name |
| `d239919` | Ops baseline: the Treasury's method or nothing, never a stand-in |
| `67d5cda` | Registry: eleven models across three families, and the Lab can finally reach them |
| `61cb415` | Snapshot the backend environment before any foundation-model install |
| `04e8748` | Foundation forecasters: two pinned checkpoints, exploratory by construction |

---

## 2. Task 0 — the Forecast page crash

### Root cause

Not an i18n fault. **Name shadowing of a module global.**

`frontend/pages/01_Forecast.py:39` imported the translator as `_t`:

```python
from i18n import t as _t
```

Three lines roughly 550 lines below reused `_t` as a variable holding a Treasury line name. Two of
them sit inside a module-level `if _modes_ok:` block, and Python does not open a new scope for an
`if`, so they rebound the module **global**:

| Line | Statement | What `_t` became |
|---|---|---|
| 606 | `for _t in [t for t in _sel if t not in _reg]:` | a target name, `str` |
| 622 | `for _t in _runnable:` | a target name, `str` |
| 648 | `_t = st.selectbox("Target", ...)` | a target name, `str` |

Line 648 runs on every render in Exploratory mode. After it, the verdict history at line 724
reached `_t(...)` at line 188 and Python raised `TypeError: 'str' object is not callable`, which is
the traceback reported. `01_Forecast.py` was the only file in `frontend/` that assigned to a name
it imported from `i18n`.

### Why the page's own render test slept through it

`test_verdict_history_render.py` already asserted `not at.exception` and passed. `AppTest` runs a
page at its **default** widget values, and at those values neither shadowing line executes: the
mode radio defaults to `"Official"`, so the Exploratory selectbox is never reached, and the default
target selection contains only targets that *have* a champion recipe, so the loop over recipe-less
targets has an empty body. Both landmines sat on paths a default render never walks. The crash
needed one radio click.

### Fix

Renaming the three variables would have fixed the crash and left the trap open, since `_t` is the
obvious short name for "target". So the translator no longer owns a name anybody would reach for:

- `from i18n import t as _translate`, and its 4 call sites (188, 327, 337, 344).
- the 3 target variables became `_tgt` (10 lines).
- a comment at the import explaining why the name is what it is.

### Evidence the tests catch it

Four new tests drive the widgets rather than trusting defaults, in **both** languages, because the
failing call was the translation call itself and an English-only test can pass while Georgian
raises. Run against the pre-fix file retrieved with `git show HEAD:...`:

```
FAILED test_exploratory_mode_renders_and_still_shows_verdict_words[en]
FAILED test_exploratory_mode_renders_and_still_shows_verdict_words[ka]
FAILED test_official_mode_with_a_recipeless_target_also_renders
FAILED test_the_translator_is_not_named_something_a_target_variable_would_reuse

  File ".../01_Forecast.py", line 188, in _verdict_words
    return _t(_VERDICT_WORDS_EN.get(code, code))
TypeError: 'str' object is not callable
```

That is the reported bug reproduced in a test. The `recipeless` failure confirms the second
shadowing site was live too.

`frontend/tests/test_no_i18n_shadowing.py` makes it a class of bug rather than one bug: an AST lint
over every page and helper, asserting no module rebinds a name it imported from `i18n`. Against the
pre-fix file it reports exactly the defect; against the fixed file, clean.

The lint's first version was **wrong** and reported `ui_styles.py`, which is correct code: its i18n
import is function-local and its two `t` bindings are comprehension variables, which in Python 3
have their own scope. It is now scope-aware, and that false positive is a test case of its own,
because a lint that cries wolf on correct code gets deleted.

### One thing found by accident

`test_i18n.py`'s orphan check scanned for calls named `("t", "_t")` to build the set of phrases the
app can translate. The rename made it blind to the whole Forecast page, so it reported the
`withheld_as_forecast` banner as an orphaned translation while the copy sat untouched on screen.
That hardcoded pair was a rot trap for anyone renaming the import. It now reads the alias from each
page's own `import` statement.

### Browser confirmation

The app starts clean and the Forecast page returns HTTP 200. The meaningful check is `AppTest`,
which executes the page script, and it now covers Official mode, Exploratory mode, both languages
and the recipe-less path. Worth noting for Task 7: the served URL is `/Forecast`, confirming page
URLs derive from page names rather than number prefixes.

---

## 3. Task 1 — the flat Ops baseline: **case (b), a display bug**

**Scoring is unaffected. The `+31.99%` and `+55.96%` figures do not depend on the broken series.**

Proven: the leaderboard's `ops_MAE` and `skill_vs_ops_pct` are computed at
`backend/b_ml_pipeline.py:1049-1068` from `ops_baseline.vintage_cache` and
`ops_prediction_for`, the truncation-verified construction. `MAE_skill_vs_Ops` in
`metrics_long.csv` is hardcoded `np.nan` at `backend/b_ml_pipeline.py:501`. The broken series fed
only the run-folder CSVs and the matplotlib plots.

Two display defects were found, not one.

### Defect 1 — a baseline was invented for the stock target. This is what was on screen.

The 0 to 3B axis identifies the run as `State budget balance`. Stocks correctly have no ops CSV, so
`frontend/pages/03_Dashboard.py:663-664` fell through to `_weekday_mean_baseline` and drew the
day-of-week mean **of the actuals** under the name "Ops baseline":

```
STOCK target: what the Dashboard labelled "Ops baseline"
  distinct values: 5      (one per weekday)
    dow 0: 1,577,344,665
    dow 1: 1,572,081,908
    dow 2: 1,562,181,291
    dow 3: 1,555,710,268
    dow 4: 1,574,842,388
  spread as % of mean: 1.38%          <-- the flat line
  y_true range: 0 .. 2,869,343,643    <-- the 0-3B axis
```

This contradicted the backend on purpose-built grounds. `ops_baseline.REASON_STOCK` states the
method aggregates a flow to an annual total, so it is undefined for a balance, and "none is
invented". The Dashboard invented one, under the wrong name. It is the average of the answers.

### Defect 2 — for flows the file was empty, and empty became zero.

Every B_ML run writes `<target>_ops_baseline_daily.csv` with no numbers in it:

```
run                                                rows  nonnan  nuniq
run_B_uni_Ridge_Revenues_Daily_h6_20260819_1538    2763       0      0
run_B_multi_Ridge_Revenues_Daily_h6_20260805_0954   243       0      0
run_B_uni_LightGBM_Revenues_Daily_h5_20260731_1126 2763       0      0
run_B_uni_Ridge_Revenues_Daily_h5_20260731_1125    2763       0      0
... 14 of 14 frontend run folders identical: 0 non-NaN
```

An all-NaN series is **not** an empty series, so it passed the caller's `not ops.empty` guard, and
`.resample(...).sum()` maps all-NaN to `0.0` — a flat line along the axis on any weekly or monthly
view.

For contrast, the correct construction is piecewise-constant by month exactly as expected:

```
backend/forecast_runs/2026-08-18/c_dl/daily/Revenues_ops_baseline_daily.csv
  1983 non-NaN, 92 unique values, 0 zeros
  year   distinct monthly levels    min .. max
  2018        12                    29,831,143 .. 55,726,453
  2024        12                    62,014,284 .. 120,418,833
  2025         8                    73,351,454 .. 93,453,838

  last months, one value each:
    2024-11  n=21  distinct=1  value=72,847,706
    2024-12  n=22  distinct=1  value=120,418,833
    2025-01  n=23  distinct=1  value=74,786,226
    2025-03  n=21  distinct=1  value=93,453,838
```

### What replaced both

`frontend/ops_baseline_view.py` builds the comparator from `ops_baseline.vintage_cache` /
`ops_prediction_for` — the same pair scoring uses — so the chart and the leaderboard cannot
disagree. Held to that by a test:

```
ops_MAE recomputed from the CHART series : 47,311,235.5240
ops_MAE stored in leaderboard.csv        : 47,311,235.5240
delta                                    : 0.000000
```

Getting that exact match found a third thing. The data file must come from the run's own
`artifacts/config.json`, not the canonical table: this run was launched on
`frontend/runs_uploads/uploaded.csv`, and defaulting to the canonical table instead moved `ops_MAE`
by **886.71** — small enough to read as rounding, and not rounding.

Stocks now show no line and a plain-words reason. Flows show the line plus a caption explaining the
flatness. Both verified live by rendering the pages: the flow run renders the flatness caption, the
stock run renders the no-baseline reason and draws nothing.

### Two corrections to my own first attempt

- **In-process, not a subprocess.** The first version dispatched to `backend/.venv` on the
  assumption that `ops_baseline` needs the modelling stack. It does not: the `c_dl_pipeline` import
  is lazy and sits inside `ops_daily_series`, which nothing on this path calls. Removing the
  subprocess removed about 60 lines and a failure mode.
- **`_is_stock` was a substring test.** I wrote `"balance" in target`. The real definition is an
  exact match against a five-alias set, so mine would have missed `t0`, `net` and `stock` and would
  have invented a level out of any column containing the word "balance". `target_kinds.py` has a
  long docstring about four families disagreeing on exactly this question; I had quietly become a
  fifth. It now copies `STOCK_ALIASES` with a test asserting the sets are equal.

### And the caption was wrong until a test corrected it

I asserted the baseline holds one value per month. It failed on five Januaries. My second guess,
grouping by the origin's calendar year, still failed on three. The real key is "the last calendar
year **complete** at the origin", so origins inside a single December straddle the boundary at
31 December:

```
January 2024: target date, the origin it was forecast FROM, and the comparator
 2024-01-01  2023-12-22   58,014,391.72
 2024-01-05  2023-12-28   58,014,391.72     <- 2023 vintage (built from 2020-2022)
 2024-01-09  2024-01-01   64,164,350.62     <- 2024 vintage (built from 2021-2023)
 2024-01-12  2024-01-04   64,164,350.62
```

January genuinely carries two values, because its first days were forecast from a December origin
when that year had not closed, so they are compared against the older figure the Treasury actually
had in hand. Flattening January to one value would be the real error. The caption now says so and a
test pins the exception so nobody "fixes" it into being wrong.

---

## 4. Scoring-side findings for a separate scoped session

**Both are recorded and neither was touched, per the agreed scope.** They belong together in one
scoping.

### Finding A — `b_ml_pipeline.ops_monthly_baseline` returns all-NaN

**File and lines:** `backend/b_ml_pipeline.py:278-281`, used at `backend/b_ml_pipeline.py:672-678`.

```python
def ops_monthly_baseline(series: pd.Series) -> pd.Series:
    m = series.resample("ME").sum().astype(float)
    # fallback: 3y same-month rolling mean
    return m.groupby(m.index.month).transform(lambda x: x.shift(12).rolling(36, min_periods=1).mean())
```

Two faults in one line. It uses **only** the fallback branch and never the real three-year
annual-total × month-share construction that `c_dl_pipeline.ops_monthly_baseline_treasury` and
`ops_baseline.py` implement. And the fallback is broken: inside a month-group, consecutive rows are
one **year** apart, so `.shift(12)` shifts twelve years.

Evidence:

```
target col: Revenues, 2015-01-05 .. 2025-08-06
monthly rows: 128, spanning 11 years
rows per month-group (all Januaries): 11   ->  .shift(12) needs >12
result: non-NaN 0 of 128

Demo, all Januaries:
      value        shift(12)
2015  7.135006e+08   NaN
2016  6.058889e+08   NaN
...
2025  2.284582e+09   NaN
```

**Blast radius.** The broken series feeds:
- `<target>_ops_baseline_daily.csv` and `_monthly.csv` in every B_ML run folder (14 of 14 present
  are empty), read by the Dashboard and Lab until this session.
- the matplotlib plots at `backend/b_ml_pipeline.py:521`, `:555`, `:598` — the "Ops baseline" line
  in `*_overlay_all.png`, `*_overlay_top.png` and `*_monthly_bars_top_vs_ops.png` is absent or
  zero in every B_ML run.
- `backend/forecast_runs/2026-08-04/b_ml/Revenues_ops_baseline_daily.csv`: 0 non-NaN of 2763.

It does **not** feed `leaderboard.csv`'s `ops_MAE` / `skill_vs_ops_pct`, which come from
`ops_baseline.py`. No published figure is affected.

Related but distinct, and already documented in `ops_baseline.py`: two older run folders carry the
*previous* `method="profile"` bug rather than this one —
`backend/forecast_runs/2026-08-04/a_stat/daily/` (2000 rows, all zero) and
`.../2026-08-04/c_dl/daily/` (1983 rows, all zero). The 2026-08-18 c_dl runs are correct.

**Proposed fix sketch.** Delete `b_ml_pipeline.ops_monthly_baseline` and
`ops_daily_from_monthly` and have the writer and the three plot calls read
`ops_baseline.ops_series_for_vintage` / `vintage_cache`, which is already the single construction
for the reporting path and is truncation-verified. That makes the CSVs and the PNGs agree with the
leaderboard by construction rather than by coincidence. Then regenerate the affected run folders,
or leave them and let `test_the_run_folder_csv_really_is_empty` fail as the prompt to do so — that
test exists and points here.

### Finding B — `GBQuantile` emits crossed intervals, and neither repairs nor reports

**File and lines:** `backend/e_quantile_daily_pipeline.py:340-341` (the `GBQuantile` branch) and
`:323-327` (`_fit_gb_quantile`). The unrepaired predictions flow to `:846-852` and into the
`yhat_p10` / `yhat_p50` / `yhat_p90` columns at `:907-910`.

Measured on 60 synthetic test rows, quantiles 0.1/0.5/0.9:

```
model            monotone_after   rows_repaired
GBQuantile       False            0        <-- crossed, and reports nothing
ResidualRF       True             0            repairs silently (line 548-550)
LGBMQuantile     True             24           repairs and reports
LinearQuantile   True             0            new this session
HistGBQuantile   True             15           new this session
XGBQuantile      True             14           new this session
```

`GBQuantile` fits each quantile as an independent model, and nothing downstream sorts them: the
`n_cross` the fold loop prints at `:848-852` is only ever *reported*, never used to repair.

**Scope of the harm is limited, and worth stating precisely.** Published intervals are safe:
`backend/forward_forecast.py:190-215` fits GBQuantile and then sorts, and its docstring says
"Independently fitted quantiles can cross -- p90 below p50 -- which is not a wide interval but an
invalid one." So the defect is confined to the E_QUANTILE evaluation pipeline's own
`predictions_long.csv` and any coverage metric computed from it. Note also that the same docstring
claims sorting "is what the E_QUANTILE family already does", which is true of `ResidualRF` and
`LGBMQuantile` and **not** of `GBQuantile`.

**Proposed fix sketch.** Route the `GBQuantile` branch through `_enforce_monotone`, added this
session at `backend/e_quantile_daily_pipeline.py`, which sorts each row and returns the count so
the existing report fires. One line. Then consider whether `ResidualRF`'s silent repair should also
report, since constant crossing is how a misconfigured quantile model looks from outside and a
silent repair means nobody learns. Both changes alter numbers in E_QUANTILE artifacts, which is why
they are not in this session.

---

## 5. Task 2 — eleven models across three families

Additive only. **Zero lines removed** from `model_catalog.py`, `model_reference.py`,
`run_a_stat.py` or `e_quantile_daily_pipeline.py`, verified by `git diff | grep "^-"` returning
nothing for all four, so every model with a measured number keeps the exact code path that
produced it.

Six of the originally suggested candidates were **already registered** — Theta, ElasticNet, Lasso,
ExtraTrees, HistGradientBoosting and damped ETS — so the list was redrawn from what sklearn 1.8,
XGBoost 3.2 and statsmodels 0.14.6 still offer unused.

### Smoke test: 11 of 11 passed

Through the Lab's own dispatch path — same runner per family, same `TG_*` environment contract,
same exploratory overrides — on Revenues, daily, h=5, Demo profile. Every one wrote a populated
`predictions_long.csv`.

| Family | Model | rc | rows | secs | What it adds |
|---|---|---|---|---|---|
| B_ML | `BayesianRidge` | 0 | 262 | 2.2 | Ridge that estimates its own shrinkage; nothing to tune |
| B_ML | `TheilSen` | 0 | 262 | 4.3 | Robust to a *run* of extreme days, where Huber is not |
| B_ML | `KNN` | 0 | 262 | 6.1 | No fitted form: have days like today happened before |
| B_ML | `KernelRidge` | 0 | 262 | 2.3 | Allowed to curve, between the linear models and trees |
| B_ML | `DecisionTree_L1` | 0 | 262 | 2.1 | The one model whose reasoning can be printed and read |
| B_ML | `AdaBoost` | 0 | 262 | 3.1 | Reweights hard rows, not residuals |
| E_QUANTILE | `LinearQuantile` | 0 | 5 | 2.5 | The only non-tree member of the family |
| E_QUANTILE | `HistGBQuantile` | 0 | 5 | 3.7 | Binned sibling of GBQuantile |
| E_QUANTILE | `XGBQuantile` | 0 | 5 | 3.2 | `reg:quantileerror`, XGBoost >= 2.0 |
| A_STAT | `SES` | 0 | 262 | 7.3 | Level only: how much of ETS is the level alone |
| A_STAT | `HOLT` | 0 | 262 | 13.8 | Undamped trend: what trend and damping are each worth |

Quantile outputs verified monotone in the real artifacts. `HistGBQuantile` row 0 came back with
p50 == p90, which is visible evidence the crossing repair fired on real data.

A_STAT additions sit in their **own** `_fc` branch rather than widening the `("ETS", "ETS_DAMPED")`
tuple, because those two have measured numbers and a new code path inside their branch would be a
new code path inside the thing that produced them.

### The counts, recomputed

| | Before | After Task 2 | After Task 3 |
|---|---|---|---|
| Shelf total | 31 | 42 | **44** |
| Measured (`evaluated_total`) | 8 | 8 | **8** |
| Untested | 20 | 31 | **33** |
| Competing | 25 | 33 | **33** |
| Champion pool (B_ML) | 15 | 21 | **21** |

`client_framing()` derives its sentence from these and updated itself:

> 21 machine-learning models, 5 deep-learning models and 7 statistical models compete on each
> target; prediction intervals come from 6 quantile methods; 3 further entries are reference
> baselines, not competitors. Of those, 8 have a recorded result on at least one target and 31 are
> registered candidates with no recorded result yet.

Nine pinned-count assertions across five test files were updated deliberately, each with a note
saying what changed. **The number to read is the middle row: the shelf grew by thirteen and the
evidence grew by nothing.**

### Task 2 found something worse than a missing model

**Ten registered models could not be run from the Lab at all.** `08_Lab.py` imported the option
lists from `backend_consts` and then hardcoded them **again** inline in the selectboxes, so there
were three diverging copies:

| Family | Registry offered | Lab named | Missing |
|---|---|---|---|
| B_ML | 15 | 8 | Huber, GBDT_L1, HistGBDT_L1, XGBoost_L1, LightGBM_L1, CatBoost_L1, CatBoost_Quantile |
| A_STAT | 8 | 7 | ETS_DAMPED |
| E_QUANTILE | 3 | 1 | ResidualRF, LGBMQuantile |

Meanwhile `06_Models.py:719` tells readers "You can run an untested model as an experiment from the
Lab". Fixed by derivation: `backend_consts` reads `model_catalog`, which is deliberately kept free
of heavy imports so the Streamlit interpreter can. Pickers verified rendering at **10 / 21 / 5 / 6**
where they showed 7 / 8 / 5 / 1. A test fails if a model list reappears in `08_Lab.py`.

One trap documented rather than tripped over: `ModelSpec.installed` calls `find_spec` in the
**calling** interpreter, and from Streamlit that is False for xgboost, lightgbm and catboost, which
live in `backend/.venv`. Filtering on it would have deleted XGBoost and LightGBM from the interface
while their runs work perfectly. The lists are unfiltered.

### Two guards had to be taught rather than silenced

Both were second copies of a list, which is the same defect one level down.

- `test_stat_model_names_uppercase_match` compared against a typed set that had never gained
  `ETS_DAMPED`. It now reads `run_a_stat.A_STAT_MODELS`, which is what `_fc` actually refuses
  against.
- `test_every_reported_parameter_has_a_plain_language_meaning_or_says_none` caught four parameters
  I had set with no explanation for a reader: `TheilSen.max_subpopulation`, `KNN.n_neighbors`,
  `KNN.weights`, `KernelRidge.kernel`. All four now have one.

And one of my own assertions was wrong: I required HOLT's forecast to move by 5% of the series
level, which no correct trend forecast could do — over ten steps the most a right answer travels is
nine increments, 90 against a level of 6,990. It now checks against the series' slope.

---

## 6. Task 3 — foundation forecasters

### What landed

| Model | Repo | Revision (pinned commit) | Download | Native quantiles |
|---|---|---|---|---|
| `Chronos_Bolt_Small` | `amazon/chronos-bolt-small` | `772f3d25d38aec6d914c8949dab4462e2d46f5d8` | small | yes |
| `TimesFM_2p5_200M` | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | 925.2 MB | yes (10-col block) |

Revisions are full 40-character commit hashes, never tags or `main`, and a test enforces that: a
forecast whose weights can move underneath it is not a record of anything.

**Cache location:** `~/.cache/huggingface/hub`. Nothing downloads at runtime after the first fetch —
a pinned revision already cached resolves from disk with no network call. Measured: Chronos 17.4s on
first load including download, 0.1s from cache; TimesFM 31.1s first, then cached.

### Package versions pinned

`backend/requirements-foundation.txt`, **not** the core requirements. `chronos-forecasting==2.3.1`,
`timesfm==2.0.2`, plus `accelerate==1.14.0`, `transformers==5.15.1`, `tokenizers==0.22.2`,
`safetensors==0.8.0`, `huggingface_hub==1.28.0`, `einops==0.8.2`, `hf-xet==1.6.0`, `httpx==0.28.1`,
`httpcore==1.0.9`, `h11==0.16.0`, `anyio==4.14.2`, `regex==2026.7.19`, `rich==15.0.0`,
`typer==0.27.1`, `shellingham==1.5.4`, `markdown-it-py==4.2.0`, `mdurl==0.1.2`,
`annotated-doc==0.0.5`, and `click==8.4.2`.

### The environment held

`pip install --dry-run` ran first and was diffed against the committed freeze (`61cb415`, 73
packages, Python 3.13.11). Chronos would move **none** of numpy 2.4.4, pandas 3.0.2, torch 2.11.0,
so it went into the core venv per the agreed rule. TimesFM added exactly one package. Verified
after both installs: all three unchanged, along with sklearn 1.8.0, lightgbm 4.6.0, xgboost 3.2.0,
catboost 1.2.10, statsmodels 0.14.6.

**One package was upgraded in place rather than added:** `click 8.3.1 -> 8.4.2`, required by typer,
which transformers pulls in. That is the only change this made to a package the core install
already had, and `requirements-foundation.txt` records it as such. The separate
`backend/.venv-foundation` was therefore not needed and was not created.

### Lag-Llama — one honest attempt, dropped

Not on PyPI (`No matching distribution found`). From GitHub it pins `gluonts<=0.14.4`, which forces
an **old pandas built from source**, and that pandas' Cython output does not compile against Python
3.13:

```
Collecting gluonts<=0.14.4 (from gluonts[torch]<=0.14.4->lag-llama==0.1.0)
Collecting pandas (from lag-llama==0.1.0)
  error: subprocess-exited-with-error
    pandas/_libs/tslibs/base.pyx.c:5399:70: error: too few arguments to function call, expected 6, have 5
    pandas/_libs/tslibs/dtypes.pyx.c:11412:7: error: call to undeclared function '_PyDict_SetItem_KnownHash'
  ninja: build stopped: subcommand failed.
error: metadata-generation-failed
```

It failed at the **dry-run**, so nothing was installed and nothing was touched; core stack
re-verified unchanged afterwards. Timeboxed and dropped, as agreed.

### Family placement: new additive `F_FOUNDATION`

Not folded into C_DL. Every C_DL model was trained on Treasury data by this pipeline, and its
numbers mean "this architecture learned this series". A Chronos number means "a model that has
never seen Georgian Treasury data guessed this". Counting them together would make "deep-learning
models: 5" describe two kinds of claim in a sentence a client reads.

Wiring was additive: an entry in `model_reference._CATEGORY_BY_PIPELINE`, a place in
`CATEGORY_ORDER`, a block in `model_pool()`, and descriptions. `client_category` raises on an
unknown pipeline **by design**, which is what forced the decision to be explicit.

### Champion lock verified

**Evidence, as required.** The lock is pre-existing and nothing was added to enforce it:

1. `model_reference.CHAMPION_POOL_CATEGORY` is `"machine-learning models"`, B_ML alone. Measured
   after registration: `champion_pool_size` is **21**, unchanged, and the foundation models are
   absent from `champion_pool`. `composition()` fails outright if a promoted model falls outside
   the category.
2. Status is derived from `experiments/log.csv`. Both are **UNTESTED**, so
   `publication_gates` has no measurement to pass.
3. A recipe cites a ledger `run_id` and `registry.verify_against_log` checks it. Neither has a
   ledger row, so neither has anything a recipe could cite.

They are also kept **out** of `COMPETING_CATEGORIES`, a separate decision: they do produce point
forecasts and could have been ranked, but a competing count containing unmeasured entries is the
overstatement `client_framing` exists to prevent. `competing_total` stayed 33.

### Smoke test, end to end through the Lab's runner

Revenues, h=5, 145 origins, bounded to train and dev:

| Model | Origins | Runtime (CPU) | MAE | 80% band coverage |
|---|---|---|---|---|
| `Chronos_Bolt_Small` | 145/145 | 7.8s | 51,093,509 | **80.7%** |
| `TimesFM_2p5_200M` | 145/145 | 11.3s | 49,706,714 | 77.2% |

A zero-shot 80% band landing on 80.7% of outcomes is a genuinely interesting reading. It is a
reading only: neither model is measured, neither is in the ledger, neither competes.

### Four things found by building it

1. **Weekends.** The Treasury table carries a zero for all 1,104 weekend rows. Handing those over
   spends the context window teaching a weekly zero pattern, and Chronos then forecast *into* it:
   two of five median steps came back **negative** (-4,184,616 and -1,143,048). On the business-day
   series the same call returns 74.9M to 78.9M, none negative, against a recent business-day mean of
   95.3M. `business_days_only` now applies for every caller.
2. **Reloading per forecast.** My first wrapper loaded the checkpoint inside the predict call. Fine
   for one forecast; for the 145-origin walk it re-read and recompiled TimesFM's 925 MB checkpoint
   at every origin and did not finish inside ten minutes. Cached behind `lru_cache` on
   `(kind, repo, revision)`: 11.3s. Kept as a test.
3. **TimesFM's quantile block has ten columns**, and which is which was checked rather than
   assumed. Column 0 is the mean; columns 1 to 9 are deciles q10 to q90. Column 5 came back
   byte-identical to the returned point forecast (60,286,384), so p10/p50/p90 are columns 1/5/9.
4. **`daily_best_model_families` was "every pipeline in the pool"**, which assumed every enumerable
   family also runs daily. F_FOUNDATION runs from the Lab only, so that derivation would have
   promised the Agent a per-family `best_model` the daily summary never writes, sending it after an
   artifact that does not exist. `EXPLORATORY_ONLY_FAMILIES` now excludes it and the list is back to
   the four it was. **This is the concrete reason the Agent smoke test must be rerun.**

`run_foundation.py` refuses outright if handed no `eval_end` rather than defaulting the window
open. The bound is about which dates a **result** may be measured on, not what a model trained on:
a zero-shot model scored over the holdout has still spent the holdout. `F_FOUNDATION` was added to
`frontend/exploratory.py`'s `FAMILIES` so the Lab applies the same bound.

---

## 7. Agent repo: rerun the smoke test

**Required, and now for two concrete reasons rather than one.**

1. The Agent lists Lab models. The shelf went 31 to 44, and the Lab's pickers went from 7 / 8 / 5 /
   1 to 10 / 21 / 5 / 6.
2. `daily_best_model_families` changed derivation. It reads the same four families as before, which
   is the point — but the derivation now filters `EXPLORATORY_ONLY_FAMILIES`, and any Agent-side
   copy of the old "every pipeline in the pool" logic would now include `F_FOUNDATION` and go
   looking for a daily best_model that is never written.

---

## 8. Task 6 investigation (deferred, but the findings stand)

Verified filenames in a real run folder
(`frontend/runs/run_B_uni_Ridge_Revenues_Daily_h6_20260819_1538/outputs`): `predictions_long.csv`,
`metrics_long.csv`, `leaderboard.csv`, `artifacts/config.json`, `artifacts/RUN.md`,
`artifacts/provenance.json`, `artifacts/integrity_report.json`, `artifacts/feature_importance.csv`.
**There is no `predictions.csv` or `metrics.csv`** — the task brief's names do not exist.

The page that browses run folders is **History** (`frontend/pages/05_History.py`), and it already
offers exactly the downloads Task 6 asks for: `predictions_long.csv` at line 211,
`metrics_long.csv` at line 213, and an all-artifacts zip at line 209. So there is **no gap to
report honestly in the UI** — Task 6 reduces to adding the same buttons to the Lab as a
convenience directly after a run, and pointing at History by name.

---

## 9. Open items and gaps

1. **Seven new user-facing strings bypass i18n.** Five reason/caption constants in
   `frontend/ops_baseline_view.py` and two notices in the Lab's foundation branch are raw strings,
   not routed through `t()`. Task 8 must route them. This also means the coverage figure below is
   measured over a domain that **excludes** them.
2. **Georgian coverage measures 80.5% (62 of 77 phrases)**, against 78.6% (55 of 70) in the
   previous record. This is not a like-for-like improvement: the figure is computed by rendering
   pages and depends on which branches render, and the seven strings above are invisible to it
   because they never enter the i18n layer. The "unreviewed" label is unchanged and still correct.
3. **The Language selector's empty pill was not fixed.** Confirmed *not* a data problem: the radio
   renders options `['English', 'ქართული']` correctly. The likely cause is the CSS rule at
   `frontend/ui_styles.py:607`, `[data-testid="stRadio"] label`, which borders every `label`
   inside the widget including the widget's own label wrapper. Scoping it to the option labels
   only, e.g. `[data-testid="stRadio"] [role="radiogroup"] label`, is the candidate fix. Not
   attempted, so this remains a hypothesis to verify in a browser.
4. **`GBQuantile` and the `b_ml_pipeline` fallback** — the two scoring findings in §4, awaiting
   authorization as one scoped session.
5. **`ResidualRF` repairs crossed quantiles silently** (`e_quantile_daily_pipeline.py:548-550`),
   returning 0 rather than the count. Worth folding into the same scoping.
6. **`DL_MODEL_OPTIONS` is still hardcoded** in `frontend/backend_consts.py`. C_DL is not in
   `model_catalog`, and the Lab's labels differ from `c_dl_registry`'s names (TCN/Transformer here,
   DCNN/TRANSFORMER there). Left as-is rather than half-migrated, and noted in the file.
7. **Older run folders carry the previous `method="profile"` ops bug**, distinct from Finding A:
   `backend/forecast_runs/2026-08-04/a_stat/daily/` and `.../c_dl/daily/` are all-zero. Regenerate
   or leave with the failing test as the prompt.
8. **Tasks 4, 5, 7, 8, 9 not started.** Task 7's confirmed order is: 1 Overview, 2 Start here,
   3 Data Preprocessing, 4 Lab, 5 Dashboard, 6 Compare, 7 History, 8 Forecast, 9 Scorecard,
   10 Documentation, with Models renamed to Documentation by `git mv` and absorbing the content.
   No files were renamed, so no cross-page link, test or doc reference has moved.

---

## 10. Manual smoke checklist

Launch:

```
./frontend/.venv/bin/python -m streamlit run frontend/Overview.py
```

| Page | Action | What you should see |
|---|---|---|
| Forecast | Open it, then switch Mode to **Exploratory** | The page renders with no error. This is the crash: it died here before |
| Forecast | Switch the sidebar to ქართული, switch Mode again | Still renders. The verdict history still shows verdicts in words |
| Dashboard | Select the newest Revenues run, view Weekly or Monthly | An "Ops baseline" line that steps month to month, with a caption explaining why it is flat within a month |
| Dashboard | Switch to the `State_budget_balance` run | **No** Ops baseline line, and a sentence saying the method does not apply to a balance |
| Lab | Family → B · Machine Learning | 21 models, including Huber and LightGBM_L1, which were unreachable before |
| Lab | Family → E · Quantile | 6 models, not 1 |
| Lab | Family → A · Statistical, pick **SES**, Demo profile, run it | Completes and writes a run folder (Task 2 model) |
| Lab | Family → **F · Foundation**, pick `Chronos_Bolt_Small`, run it | The exploratory notice appears above the picker; the run completes in roughly 10 seconds and reports MAE and band coverage (Task 3 model) |
| Models | Scroll the shelf | 44 entries, 8 measured, the rest badged UNTESTED |

---

## 11. Ground rules

Rule 5, terminal commands with no inline comments — followed. Rule 6, the writing voice — applied
to all new user-facing copy, though see gap 1: that copy is English-only for now. Rule 3, scope —
the two scoring findings were recorded rather than fixed, and `b_ml_pipeline.py` was not touched.
Rule 4, verify before asserting — every claim in this record has a measurement or a file reference
behind it, and three of my own assertions were wrong and corrected by tests along the way (the
substring `_is_stock`, the one-value-per-month caption, and the HOLT trend threshold).
