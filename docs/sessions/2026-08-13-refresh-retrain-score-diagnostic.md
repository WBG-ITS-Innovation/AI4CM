# Diagnostic: withheld flows, data refresh → retrain, realized scoring, moved edge

**Date:** 2026-08-13 · **Branch:** `model/excellence` @ `59709f1` · **Nothing changed.**
No source file, artifact, registry entry or test was modified. Every run below wrote to a
`tempfile.mkdtemp()`; `forecasts/published/` and the canonical CSV are untouched.

---

## 1. The exact prompt given

> Continue on model/excellence in the AI4CM repo. Diagnostic session — change nothing yet.
>
> The client will load their own data covering the period after 2025-08-06, retrain, forecast, and
> compare our forecasts to actuals. Four things must work and I need their real state before we touch
> anything.
>
> 1. WITHHELD FLOW TARGETS. The last record says flow targets remain withheld_as_forecast because the
> signal sentinel has not cleared 1.50 under any of three probes. Revenues and Expenditure are two of
> the three targets the client will test. Explain in plain language what the sentinel measures, why it
> isn't clearing, whether 1.50 is the right threshold or an assumption, and what a client actually
> sees today when they ask for a Revenues forecast. Is the model bad, or is the gate miscalibrated?
> Evidence, not opinion.
>
> 2. DATA REFRESH → RETRAIN. Trace what happens when master_daily_clean_treasury.csv gains rows past
> 2025-08-06: preprocessing, the business-day index, evaluation windows, refitting, champion
> reselection, publishing. Name every place that breaks, silently degrades, or needs a manual step.
> State whether champions are reselected on new data or reused from the last selection run.
>
> 3. SCORING. The scorecard reads 0 scored / 15 pending — that path has never run with real actuals.
> Trace how a pending forecast is matched to arrived actuals and how realized error and realized skill
> vs persistence are computed. Then EXERCISE it against a synthetic actuals file covering 2025-08-07
> to 2025-08-13 and paste the real output. I expect it to break; tell me how.
>
> 4. FORWARD FORECAST FROM A MOVED EDGE. Confirm run_forward works when the data edge moves — that
> origin_value, the benchmark series and the h-step horizons follow the new edge and nothing is pinned
> to 2025-08-06.
>
> Give me a prioritized fix list for a client to load new data, retrain, forecast, and see
> forecast-vs-actual. File and line references, plain language, no changes yet.
>
> Session logging: docs/sessions/2026-08-13-refresh-retrain-score-diagnostic.md, full narrative
> verbatim.

---

## 2. Plan

Measure each of the four, in the order that lets the later ones reuse the earlier fixtures. Run the
scoring and forward paths for real against a synthetic extension rather than reasoning about them.
Verify each premise before repeating it — including the prompt's expectation that scoring breaks.

---

## 3. Commands run and their real output

### Sentinel ratios, all 153 logged runs

```
  Expenditure              n= 35  min=1.0710  median=1.1304  max=1.1977  >=1.5: 0
  Revenues                 n= 36  min=1.0000  median=1.1382  max=1.2255  >=1.5: 0
  State budget balance     n= 38  min=0.9835  median=2.7007  max=7.0058  >=1.5: 37
```

### Three probes, from `reports/sentinel_probe_study.md`

```
| Target               | Feature set | ridge | tree  | forest | Verdict          |
| Revenues             | final       | 1.167 | 1.421 | 1.260  | neither detects  |
| Expenditure          | final       | 1.071 | 1.396 | 1.155  | neither detects  |
| State budget balance | final       | 3.992 | 3.182 | 3.580  | both detect      |
```

Highest flow reading across three instruments and two feature sets: **1.421**, against a threshold
of 1.50.

### The authoritative gate record

```
Revenues  [LightGBM_L1]
   logged dev_mae            38,931,955.83
   logged skill_vs_ruler  55.9181
   logged mase            0.757959
   logged sentinel_ratio  1.2255
      signal       passed=False measured=1.2255 threshold=1.5
      overfitting  passed=True  measured=< 3.0 threshold=3.0
      vs_ruler     passed=True  measured=55.92% threshold=> 0%

Expenditure  [LightGBM_L1]
   logged dev_mae            51,602,950.54
   logged skill_vs_ruler  29.4248
   logged mase            1.103854
   logged sentinel_ratio  1.0882
      signal       passed=False measured=1.0882 threshold=1.5

State budget balance  [HistGBDT_L1]
   logged dev_mae           194,104,921.84
   logged skill_vs_ruler  20.0072
   logged mase            1.57832
   logged sentinel_ratio  7.0058
      signal       passed=True  measured=7.0058 threshold=1.5
```

### What a featureless constant scores against the same ruler

```
target                    persistence     TRAIN mean  trailing-20 mean  skill(mean)
Revenues                   96,779,790     59,442,065        61,795,050       38.58%
Expenditure                87,003,666     55,954,448        56,375,552       35.69%
State budget balance      192,207,853    767,732,358       255,689,405     -299.43%
```

(My own row selection, so directional against the pipeline's logged figures rather than
apples-to-apples. The champions do beat the constant: Revenues by 34.50%, Expenditure by 7.78%.)

### What the client is told for Revenues

```
  status      : candidate -- pre-tuning
  approved_by : None
  signal gate : False | measured 1.2255
  plain reason: Shuffling the historical answers barely worsened this model's error (x1.23, we
                require x1.50). That means the inputs carry little information about what happens
                next, so the model is tracking the typical level rather than anticipating events.

  publication.verdict     : withheld_as_forecast
  publication.reason_plain: The numbers are shown and the error is 56% below the simple benchmark,
                but the signal check fails: this is a central-tendency estimate, not an event
                forecast. It should not be relied on to anticipate an unusual day.
  publication.named_fix   : Treasury's forward auction and redemption calendar
```

### Windows and holidays

```
TRAIN = 2015-01-05 .. 2023-12-31
DEV   = 2024-01-01 .. 2024-12-31
TEST  = 2025-01-01 .. (none)          <- every new client row lands here

Georgian WEEKDAY holidays 2025-01-01..2025-08-06: 10
   2025-01-01  row_in_file=True  Revenues=0
   2025-03-03  row_in_file=True  Revenues=0
   ...
is_* columns: ['is_weekend', 'is_holiday']      is_holiday rows set: 181

holiday coverage forward:
  2025: 17 holidays   2026: 17   2027: 17   2030: 17     (Easter-derived, no hardcoded table)
```

### Item 3 — scoring, exercised against synthetic actuals 2025-08-07 … 2025-08-13

```
real data ends: 2025-08-06
synthetic actuals: ['2025-08-07', '2025-08-08', '2025-08-11', '2025-08-12', '2025-08-13']

scorecard rows: 15
              target  horizon target_date          p50       y_true    abs_error  persistence_pred  persistence_abs_error  skill_vs_ruler_pct  inside_interval
            Revenues        1  2025-08-07 9.960823e+07 8.316463e+07 1.644360e+07      8.210340e+07           1.061232e+06        -1449.481283             True
            Revenues        2  2025-08-08 7.635415e+07 6.254337e+07 1.381078e+07      1.045604e+08           4.201704e+07           67.130516             True
            Revenues        3  2025-08-11 7.763747e+07 1.243290e+08 4.669149e+07      7.348653e+07           5.084243e+07            8.164318            False
            Revenues        4  2025-08-12 6.155727e+07 8.149867e+07 1.994140e+07      7.512562e+07           6.373051e+06         -212.901892             True
            Revenues        5  2025-08-13 5.831942e+07 3.026689e+07 2.805253e+07      4.649079e+07           1.622390e+07          -72.908653            False
         Expenditure        1  2025-08-07 6.495626e+07 1.043896e+08 3.943334e+07      7.036250e+07           3.402710e+07          -15.888033             True
         ...
State budget balance        4  2025-08-12 1.690622e+09 1.926327e+09 2.357049e+08      1.894389e+09           3.193810e+07         -638.005511            False
```

Per-target aggregates it produced:

```
 Expenditure          : n=5  realized_mae 19,224,604   persistence_mae 74,914,777   skill  74.34%  hit 1.0
 Revenues             : n=5  realized_mae 24,987,960   persistence_mae 23,303,531   skill  -7.23%  hit 0.6
 State budget balance : n=5  realized_mae 187,180,179  persistence_mae 184,518,075  skill  -1.44%  hit 0.2
```

The comparator, against what the artifact already records:

```
  target  horizon origin_date  origin_value target_date          p50
Revenues        1  2025-08-06   46490793.48  2025-08-07 9.960823e+07
Revenues        2  2025-08-06   46490793.48  2025-08-08 7.635415e+07
Revenues        3  2025-08-06   46490793.48  2025-08-11 7.763747e+07
Revenues        4  2025-08-06   46490793.48  2025-08-12 6.155727e+07
Revenues        5  2025-08-06   46490793.48  2025-08-13 5.831942e+07

scorecard persistence_pred by horizon:
   h=1:     82,103,400   (should be     46,490,793)
   h=2:    104,560,400   (should be     46,490,793)
   h=3:     73,486,530   (should be     46,490,793)
   h=4:     75,125,620   (should be     46,490,793)
   h=5:     46,490,790   (should be     46,490,793)   <- only this one is right
```

### A gap in the actuals file

```
2025-08-11 deliberately ABSENT from the actuals file.

 horizon target_date          p50       y_true    abs_error
       3  2025-08-11 7.763747e+07 0.000000e+00 7.763747e+07

>>> 2025-08-11 was SCORED with y_true=0.00
```

### Item 4 — forward forecast from a moved edge

```
original edge : 2025-08-06
extended edge : 2025-08-13

ORIGINAL:
   origin_date  2025-08-06
   origin_value 46,490,793.48
   target_dates ['2025-08-07', '2025-08-13']
   n_train_rows [2409, 2406]
EXTENDED:
   origin_date  2025-08-13
   origin_value 30,266,889.40
   target_dates ['2025-08-14', '2025-08-20']
   n_train_rows [2414, 2410]
```

---

## 4. Full narrative response, verbatim

Nothing was changed. Two of the four came back healthier than expected and two have real defects, one
of which will produce wrong numbers on the client's first comparison.

### 1. The withheld flow targets

**What the sentinel measures.** Take the same features, shuffle the historical answers so each row is
paired with somebody else's outcome, refit, and measure held-out error. Divide that by the error of
the honestly-fitted model. If the features genuinely inform the target, destroying the pairing should
hurt a lot. The reading is that ratio, and it must reach 1.50 — the error must get at least half again
worse.

**Why it isn't clearing.** Across 71 logged flow runs the ratio never once reaches 1.50: Revenues tops
out at **1.2255**, Expenditure at **1.1977**. The probe study widened the instrument from ridge to a
tree and a forest, and the best flow reading anywhere is **1.421** (tree, Revenues) — still short.

The mechanism is the shape of the series. A daily flow is a spiky sequence around a level. Almost all
of the achievable accuracy comes from knowing roughly what a typical day looks like, and a model
fitted on *shuffled* answers still learns that level — it just cannot say which days are the big ones.
So its error is only 10–20% worse, and the ratio sits near 1.1. The stock target behaves oppositely
(median 2.70, 37 of 38 clearing) because it is modelled as a change from the origin, and the change
genuinely depends on the features.

**Is 1.50 right, or an assumption?** It is an **assumption**, and the repo is honest about it.
`forecast_integrity.py:511` is a bare constant with a one-line comment. `docs/SIGNAL_FINDING.md`
rationalises it — "the error must get at least half again worse… the margin above 1.00 exists so that
noise cannot pass" — which is a judgement, not a calibration. **No null distribution was ever
estimated**: nobody has measured what ratio a genuinely signal-free feature set produces on this data,
so there is no false-positive rate behind 1.50. `reports/sentinel_probe_study.md` pre-registered that
the study would not move it, which was the right discipline for that study but leaves the number
unvalidated.

That matters because the best flow reading, 1.421, is **0.08 below the line**. A threshold nobody
calibrated is deciding a publication verdict on a margin of 5%.

**Is the model bad, or the gate miscalibrated?** On the evidence, neither exactly: **the gate set is
incomplete and internally inconsistent.** Three facts.

First, **MASE is computed and logged on every run and gated on nothing.** Revenues has MASE **0.758** —
24% better than the seasonal-naive benchmark — and is withheld. The stock target has MASE **1.578** —
58% *worse* than seasonal-naive — and is the one that publishes. The gate that would have caught that
is sitting in the log unused.

Second, the gate that does pass is nearly vacuous: `vs_ruler` has a threshold of **"> 0%"**. Beating
persistence by any margin at all satisfies it.

Third, on a flow, beating persistence is a weak claim. A single constant — the TRAIN mean, using no
features whatsoever — scores **38.58%** on Revenues and **35.69%** on Expenditure against the same
ruler. So the headline "56% below the benchmark" is partly a statement about how poor persistence is
on a spiky series. The champions do beat that constant (Revenues by 34.50%, Expenditure by only
7.78%), so Revenues has real content; but Expenditure is 8% better than a number that requires no
model, while its own MASE says it is worse than seasonal-naive.

My read: **the sentinel is telling the truth about Revenues and Expenditure**, and it is the right
instrument for the question it asks. What is wrong is that a single uncalibrated threshold is the sole
gate on publication while a better-grounded one (MASE) is ignored — which is how the project ends up
withholding its best-MASE model and publishing its worst.

**What a client sees today.** Not a refusal. `official_run` refuses only when a target has no recipe or
the horizon is not 5, and Revenues has a recipe — so it fits, forecasts, and publishes. The client gets
five dated numbers with P10/P90 intervals, labelled `withheld_as_forecast`, with this text: *"The
numbers are shown and the error is 56% below the simple benchmark, but the signal check fails: this is
a central-tendency estimate, not an event forecast. It should not be relied on to anticipate an unusual
day."* `approved_by` is null on all three recipes and status reads `candidate -- pre-tuning`. That is
honest and I would not soften it — but note the tension: the client is shown a number and told not to
rely on it for the thing they most likely want it for.

### 2. Data refresh → retrain

**The central blocker is the evaluation windows.** `evaluation_windows.py:295` defines
`TEST = 2025-01-01 .. (none)`. Every row the client adds after 2025-08-06 falls **inside the sealed
holdout**. Any code path that selects, tunes or compares models over that data calls
`require_test_access`, which raises unless `AI4CM_ALLOW_TEST_READ=1`. So the client cannot retrain-and-
reselect on their own recent data without deliberately unlocking the holdout — and once unlocked, the
2025 holdout stops being clean for the final read this project has been protecting for months. That is
a decision to be made consciously, not a bug to fix quietly.

**Champions are reused, never reselected.** `registry/recipes.json` is hand-maintained: there is no
`save_registry`, nothing writes recipes, and `scripts/` has no reselection entry point (`ws2_tune.py`
and `ws7_cqr.py` are workstream one-offs). `official_run` reads the recipe and refits *that* model on
current data. So new data changes the fitted parameters but never the choice of model, features or
target transform. If the client's period has a different character — and `registry/recipes.json` records
that Revenues' `ratio` transform advantage is drift-dependent, +1.30% at 13.5% drift versus +24.14% at
84.1% — the champion could be the wrong choice and nothing would say so.

**What actually works.** Refitting on all available history is clean (item 4). Holidays are derived
internally by `preprocessing/fiscal_calendar.py` from `georgian_holidays_range`, computed per year with
Easter movable feasts — 17 holidays in 2025, 2026, 2027 and 2030 alike, no hardcoded table — so the
client's file does **not** need to populate `is_holiday`. `verify_expected_sha` is opt-in and returns
early when `AI4CM_EXPECTED_DATA_SHA256` is unset, so it will not block a new file unless someone sets
it.

**What silently degrades.** `b_ml_pipeline.py:181` builds the modelling index with
`pd.date_range(..., freq="B")` — plain Mon–Fri, **not** the Georgian calendar — and then fills missing
flow days with `0.0`. That happens to match the existing convention (the canonical file already carries
its 10 weekday holidays in 2025 as `Revenues=0`, 181 `is_holiday` rows in total), so a client file that
simply omits holidays degrades into the same shape. But it means a genuine data-collection gap is
indistinguishable from a public holiday: both become a real observation of zero. There is no check that
the client's new rows are dense.

**What needs a manual step.** Preprocessing is driven by environment variables from Streamlit
(`run_preprocess.py` reads `PP_*`), so producing `master_daily_clean_treasury.csv` from a client upload
is a UI action, not a callable pipeline step. And any B_ML evaluation must have `eval_start`/`eval_end`
set explicitly — this is the bug that once made "DEV" figures silently include 2025, and the bounds are
not inferred from the data.

### 3. Scoring — it runs, and it is wrong

The prompt expected a break. **It does not break: 15 of 15 scored, no exception.** The matching logic
is sound — `score_one` looks up `target_date` in the truth index, refuses via `TruthNotAvailable` when
it is absent, computes absolute error against `p50` and an interval hit against P10/P90.

But look at the realized skill column: **−1449%**, **−212%**, **−638%**. Those are not noise, they are
a broken comparator.

`published_forecasts.py:146` declares `score_one(row, truth, horizon_steps: int = 5)` and line 166
computes the baseline as `truth.iloc[pos - horizon_steps]` — a **fixed 5 for every row**, ignoring
`row["horizon"]`, which line 226 of the same function reads and writes into the output. The project's
ruler is `ŷ(t+h) = y(t)`: the value h business days before the target. For a published issue all five
horizons share **one origin**, so the correct baseline for every row is the same number — the origin
value. The scorecard instead used five different values, and only h=5 is right. The proof is arithmetic:
h=5's `persistence_pred` is 46,490,790 and the artifact's `origin_value` is 46,490,793.48.

So **12 of 15 realized skill figures are wrong, and all three per-target aggregates are wrong.** The
Revenues headline of −7.23% and Expenditure's +74.34% are both computed against a baseline the project
does not use anywhere else.

The fix is smaller than the bug: **`origin_value` is already a column in `forecast.csv`.** It *is* the
persistence prediction — `forward_forecast.py` documents it as exactly that and the Forecast page plots
it as the benchmark. The scorer recomputes it from the truth series instead of reading it, which is the
one thing this project's "one ruler, read not recomputed" discipline exists to prevent, violated at the
place where it decides published accuracy.

There is a second defect. `_truth_series` (line 139) routes the actuals through `to_business_index`,
which fills missing flow days with `0.0`. I removed 2025-08-11 from the actuals file and re-scored:
that date was **scored with `y_true = 0.00`** and a fabricated error of 77.6M, rather than being
reported as pending. `TruthNotAvailable` can never fire for a flow target inside the data range,
because after the zero-fill every business day is present and finite. A client whose upload is missing
a day will see a large error they cannot explain.

(One thing in that output is my fixture's artifact, not a defect: the actual values appear shifted by a
day because my synthetic generator emitted one fewer row. The zero-fill is the real finding.)

### 4. Forward forecast from a moved edge — clean

Nothing is pinned. Extending the file to 2025-08-13 moved `origin_date` from 2025-08-06 to 2025-08-13,
`origin_value` from 46,490,793.48 to 30,266,889.40, and the target dates from
2025-08-07/2025-08-13 to 2025-08-14/2025-08-20. Training rows grew from 2409/2406 to 2414/2410, so the
refit genuinely used the new data. The benchmark series follows because it *is* `origin_value`.

Two caveats that are correct behaviour rather than problems. The origin is the last row whose features
are all present, not simply the last row — so a trailing row with an incomplete feature window is
skipped, and the origin can sit a few days behind the file's last date. And `assert_forward_only`
guarantees every target date is strictly beyond the data end, so a client cannot accidentally get a
"forecast" for a day they already have.

---

## 5. Prioritized fix list

Nothing below is implemented.

**P0 — wrong published numbers. Fix before the client scores anything.**

1. **`score_one` uses the wrong persistence baseline.**
   `backend/published_forecasts.py:146` (signature), `:165-166` (the lookup), `:239` (the call).
   Read `row["origin_value"]` as the baseline instead of recomputing it; fall back to
   `truth.iloc[pos - int(row["horizon"])]` only when the column is absent. Consequence today: 12 of 15
   realized skills and all three aggregates are wrong.
2. **A missing actual is scored as zero.**
   `backend/published_forecasts.py:139` → `b_ml_pipeline.py:181-186`. Scoring needs the *raw* series,
   not the zero-filled modelling series, so a genuine gap stays `TruthNotAvailable` and is reported as
   pending. Consequence: a client with one missing day sees a fabricated large error.

**P1 — blocks retrain-and-compare.**

3. **New data lands inside the sealed TEST window.** `backend/evaluation_windows.py:295`. Decide
   deliberately: either the client's period becomes a fourth window (a rolling "live" window that is
   scored but never used to select), or the holdout is consciously released. Not a code fix — a policy
   decision that then has a code shape.
4. **No champion reselection exists.** `registry/recipes.json` is hand-edited; no writer, no script.
   Either state plainly that champions are fixed and only refitted, or build a reselection path with
   the gate set from item 5 below. The drift caveat already recorded in the registry is the reason this
   matters for a new period.

**P2 — the gate set decides the wrong thing.**

5. **MASE is logged and not gated.** Revenues MASE 0.758 is withheld; the published stock target is
   1.578. Add MASE to the gate set, or state why skill-vs-persistence is preferred over it. This is the
   single highest-value change to what the client is told.
6. **`vs_ruler` threshold is "> 0%"** — nearly vacuous on a flow, where a featureless constant scores
   ~36–39%. Raise it, or replace it with MASE.
7. **Calibrate 1.50, or stop treating it as a bright line.** `forecast_integrity.py:511`. Estimate the
   null distribution by running the sentinel on deliberately signal-free features on this data, and set
   the threshold to a stated false-positive rate. The best flow reading is 1.421 against an uncalibrated
   1.50 — a 5% margin currently decides publication.

**P3 — robustness for a client-supplied file.**

8. **No density check on new rows.** A holiday and a collection gap are both zero. Add a check that
   reports gaps in the client's date range before anything trains on it.
9. **Preprocessing is only reachable through Streamlit env vars** (`backend/run_preprocess.py`). A
   callable entry point would make "load new data" scriptable.
10. **`eval_start`/`eval_end` must be set explicitly** or B_ML folds run into the new data — the bug
    that once made DEV figures include 2025.

---

## 6. Verdict

Two of the four are healthy. **`run_forward` follows a moved edge correctly** and **the scoring path
runs end to end** rather than breaking as expected.

But scoring produces **wrong numbers**: a fixed `horizon_steps=5` means 12 of 15 realized skill figures
and all three per-target aggregates are computed against a baseline the project uses nowhere else,
while the correct baseline sits unread in the artifact's own `origin_value` column. And a missing actual
is scored as a zero rather than reported as pending.

The withheld flows are the sentinel telling the truth, but the **gate set is inconsistent**: MASE is
logged and ignored, so the model with the best MASE (Revenues, 0.758) is withheld while the worst
(the stock target, 1.578) publishes. The 1.50 threshold is an uncalibrated assumption deciding that on
a 5% margin.

Retrain-and-reselect is **blocked by design**: all new client data falls inside the sealed TEST window,
and champions are reused rather than reselected because nothing writes the registry.

---

## 6b. Addendum: an untracked 2026-08-12 run, and what it proves

While confirming nothing had been modified, `git status` showed an untracked
`backend/forecast_runs/2026-08-12/` (created 2026-08-12 22:35 — before this session; every run here
wrote to a temp directory). It is a real `mode: production` run over
`State budget balance`, families A_STAT / B_ML / E_QUANTILE, and it is worth three observations.

**It carries all four fields.** `run_id: 2026-08-12`, `schema_version: 2`,
`data_file: master_daily_clean_treasury.csv`, `client_framing: present`. So the writer changes reach a
genuinely fresh run, not only the regenerated 2026-08-04 summary.

**It passes the artifact contract with 0 errors** (7 warnings, all previously catalogued: the derived
baseline row absent from `predictions_long`, the decorated join key, empty `sMAPE`/`MAPE` columns).

**It empirically confirms the previous session's verdict on the a_stat leaderboard.** That verdict was
"pre-existing CSV defect, writer fixed, the next A_STAT run will pass". Measured across the two runs:

```
  2026-08-04: identity fully populated=False   RMSE populated=False
  2026-08-12: identity fully populated=True    RMSE populated=True
```

Same columns, same writer, different output — because the second run was produced *after* the fix. That
is the independent data point the earlier reasoning lacked, and it settles the regression question by
demonstration rather than by inference.

**Left untouched.** I did not commit it and did not delete it. Now that
`backend/forecast_runs/*/SUMMARY.json` and `SUMMARY.txt` are un-ignored, this run's two summary files
show as untracked — so committing it is a decision worth making deliberately: it is the first run whose
summary is contract-clean, which makes it a better reference artifact for the Agent than 2026-08-04. Its
row-level CSVs remain ignored either way.

---

## 7. Outstanding

* Everything in §5 — nothing was implemented.
* The 1.50 null distribution has never been estimated on this data; until it is, "no signal" is a
  convention rather than a measurement.
* Whether Expenditure should be published at all is an open question the gates do not currently ask:
  MASE 1.104 (worse than seasonal-naive) and only 7.78% better than a featureless constant.
* The stock target publishes with MASE 1.578 and, on my own row selection, roughly matches h-step
  persistence on the level. Worth a dedicated look before a client compares it to actuals.
* `score_published`'s return keys were not what I guessed (`n_scored` / `n_pending` returned `None`);
  the summary shape should be checked against `AGENT_ARTIFACT_CONTRACT.md` before a consumer relies on
  it.
* Unchanged from earlier records: the four `is_stock` implementations diverge (latent);
  `conditional_coverage_gate` is not wired into `quantile_quality_gate`; `check_feature_leakage` is
  weak; A_STAT writes no shift fields; `a_stat_models_pipeline.py` is a second unreferenced A_STAT
  implementation; the committed 2026-08-04 a_stat leaderboard fails the contract (pre-existing);
  `⚡ Persistence (baseline)` uses decoration in a join key; `skill_pct` and `horizon` are strings;
  `metrics_long.csv` has two shapes under one filename; ops P0, Phase-1 cleanup, registry approval
  workflow, single TEST read.
