# AI4CM Model Audit — 2026-07-21

**Branch:** `audit/model-review`  **Scope:** families A_STAT, B_ML, E_QUANTILE, C_DL
**Type:** DIAGNOSIS ONLY — no fixes, no refactors applied.
**Target audited:** `Revenues`, Daily cadence, horizon `h = 5`.

This report explains *what the pipeline actually did*, in plain terms, and pins
down every symptom with `file:line` evidence plus independent numpy
re-derivations. Every metric quoted from the pipeline was recomputed by hand and
**matched to the cent** unless stated otherwise — so the problems below are about
*what is being compared and how the models behave*, not about arithmetic bugs in
the metric code.

---

## 0. How to read this report (beginner orientation)

A few terms used throughout:

- **Horizon `h`** — how far ahead we forecast. `h = 5` means "predict the value 5
  business days after the last data point we're allowed to see."
- **Origin date** — the last date whose value the model is allowed to use. The
  **target date** is `origin + h`. The value at the origin is `origin_value`.
- **Persistence baseline** — the dumbest honest forecast: "tomorrow (h steps
  ahead) will equal today (the origin)." Formally `ŷ(t) = y(t − h)`. Any real
  model must beat this to be worth anything.
- **Skill** — how much better than the baseline: `(MAE_baseline − MAE_model) /
  MAE_baseline × 100`. Positive = better than the dumb baseline; negative = worse.
- **MAE** — mean absolute error, the average size of the miss, in the same units
  as the data (here, currency ~10⁸).
- **Fold** — one train/test split. "One fold" = train on the past, test on one
  held-out block.
- **Coverage** — for an interval forecast (e.g. P10–P90), the fraction of actual
  values that actually fell inside the band. An 80% band should cover ~80%.

---

## 1. Test suite

Full suite (`pytest`, per `pytest.ini`): **97 passed in 12.70s**, exit 0. The
production-trust toolkit is green on this branch. The findings below are about
*model behaviour and metric definitions*, which the unit tests do not exercise.

---

## 2. Unified metrics table

One row per model, plus each family's baseline. **Read the "Eval window" and "n"
columns carefully — the four families were NOT evaluated on the same window, and
that is itself a finding (see §4).**

| Family | Model | MAE | Skill vs baseline | Baseline used | Eval window (target dates) | n test pts |
|---|---|---:|---:|---|---|---:|
| A_STAT | **ETS** (best) | 42,549,097 | **+3.81%** | flat last-value (*not* h-step) | 2025-01-01 → 2025-08-06, 1 fold | 156 |
| A_STAT | Persistence *(flat last-value)* | 44,233,643 | — (baseline) | — | same | 156 |
| B_ML | **Lasso** (best) | 43,472,942 | **+28.71%** | h-step persistence | 2025-01-01 → 2025-08-06, 1 fold | 156 |
| B_ML | Ridge | 43,475,386 | +28.70% | h-step persistence | same | 156 |
| B_ML | RandomForest | 44,140,611 | +27.61% | h-step persistence | same | 156 |
| B_ML | ElasticNet | 45,498,679 | +25.38% | h-step persistence | same | 156 |
| B_ML | ExtraTrees | 47,088,711 | +22.77% | h-step persistence | same | 156 |
| B_ML | XGBoost | 47,191,696 | +22.61% | h-step persistence | same | 156 |
| B_ML | HistGBDT | 49,594,250 | +18.67% | h-step persistence | same | 156 |
| B_ML | LightGBM | 53,586,169 | +12.12% | h-step persistence | same | 156 |
| B_ML | Persistence *(h-step)* | **60,976,736** | — (baseline) | — | same | 156 |
| E_QUANTILE | **GBQuantile** (P50) | 14,975,709 | **+57.5%** | h-step persistence | 2025-07-28 → 2025-08-06, 2 folds | 10 |
| E_QUANTILE | ResidualRF (P50) | 16,462,092 | +53.3% | h-step persistence | same | 10 |
| E_QUANTILE | Persistence *(h-step)* | 35,246,357 | — (baseline) | — | same | 10 |
| C_DL | **DCNN** (best) | 73,425,360 | **−38.65%** | h-step persistence | 2019-01-01 → 2025-08-06, multi-fold | 1,722 |
| C_DL | MLP | 73,426,555 | −38.65% | h-step persistence | same | 1,722 |
| C_DL | GRU | 73,426,609 | −38.65% | h-step persistence | same | 1,722 |
| C_DL | TRANSFORMER | 73,426,609 | −38.65% | h-step persistence | same | 1,722 |
| C_DL | LSTM | 73,426,609 | −38.65% | h-step persistence | same | 1,722 |
| C_DL | Persistence *(h-step)* | 52,957,744 | — (baseline) | — | same | 1,722 |

**Beginner reading of this table:** the families are *not on a level playing
field*. B_ML and C_DL both compare against a genuine h-step persistence, and both
compute the identical number on the shared 2025 window (60,976,736 — see §3), so
they *are* comparable to each other. A_STAT compares against a **different, easier
baseline** and forecasts a whole year in one shot (§4). E_QUANTILE looks best of
all (+57%) but was tested on just **10 days** (§5). C_DL's negative skill means
its five neural nets are all **worse than doing nothing** (§4, Critical-1).

---

## 3. The persistence discrepancy — resolved with arithmetic

The audit's headline puzzle: the same target/horizon produced three different
"persistence" numbers. Here is exactly what each one is, reconstructed
independently in numpy on **B_ML's exact single fold** (156 rows, target dates
2025-01-01 → 2025-08-06). All three reconstructions reproduce the reported
figures to the cent.

| # | Reported number | What the code actually computes | Reconstruction | Is it a correct h-step persistence? |
|---|---:|---|---:|---|
| 1 | **60,976,736** | `ŷ(t) = y(t − h)` = `origin_value`, evaluated on the 156-pt window (B_ML leaderboard "⚡ Persistence (baseline)") | `mean(\|y_true − origin_value\|)` = **60,976,736.58** | ✅ **YES — this is the correct h-step persistence** |
| 2 | **44,233,643** | A_STAT's baseline = the **single last training value held flat** for the whole year (`y(2024-12-31) ≈ 102.3M` repeated 156×) | `mean(\|y_true − 102.3M\|)` = **44,233,643.69** | ❌ NO — it's a *flat last-value* baseline, not `y(t−h)` |
| 3 | **43,507,476** | B_ML integrity `mae_shift_minus_h` = the **best model's own predictions** (Lasso) shifted back 5 steps | `mean(\|y_true[:-5] − y_pred[5:]\|)` = **43,507,476.12** | ❌ NO — it's a *diagnostic on the model*, not a baseline at all |

**Cross-check:** C_DL, evaluated on the same 2025 window, independently produces a
plain h-step persistence of **60,976,737** — matching B_ML to the dollar. So two
independent families agree on the one correct baseline.

**Verdict.** Only **60,976,736** is a correct h-step persistence for the shared
window. The premise "same fold → baselines should be near-identical" is right —
and B_ML and C_DL *do* agree. A_STAT diverges for two independent reasons:

1. It measures against a **different baseline** (flat last-value, not `y(t−h)`) —
   `run_a_stat.py:291` `origin_val = float(y_tr.iloc[-1])`, written as a constant
   `origin_value` for every test row (`run_a_stat.py:299-300`), then MAE'd at
   `run_a_stat.py:329`.
2. Its **baseline number is lower (44.2M < 60.9M)** because a flat line near the
   series mean is *easier to be near* than a value copied from 5 volatile days
   ago — this daily series has holiday spikes (revenue jumps 0 ↔ 133M), which the
   true h-step persistence chases and gets punished for.

The "43.5M" is a red herring: it is Lasso's own predictions time-shifted
(`forecast_integrity.py:215`, surfaced as `mae_shift_minus_h` at
`forecast_integrity.py:249`). Its near-equality to Lasso's real MAE (43.47M) is
precisely what triggers the misleading *"model ≈ naive baseline (shift=-h)"* text
— but the "naive" thing it's near is **the model itself shifted**, not a
persistence forecast (see §6, B_ML explanation).

---

## 4. Seven correctness checks — per family

Legend: ✅ pass · ⚠️ pass-with-caveat · ❌ fail · (evidence = `file:line`).

### A_STAT (ETS + baseline) — runner `run_a_stat.py`

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Fit once per fold | ✅ | `run_a_stat.py:282` calls `_fc` once/fold; ETS fits once then forecasts the block (`:164-167`) |
| 2 | Preprocessing fit on train only | ✅ | no scaler; only calendar fill (`:34-39`), constant `fillna(0)` for flows |
| 3 | Prediction target matches training target | ❌ | **horizon ignored**: forecasts `n = len(idx_te)` steps = the whole year, not 5 (`:135`, `:167`); `h` kept only as metadata (`:303` `"stat_models_forecast_full_test_window"`) |
| 4 | Baseline correct & same window | ❌ | baseline is flat last-value, not h-step persistence (`:291`, `:299-300`); *window* is the same as the model (✅ that part), math at `:329` |
| 5 | No tuning on test | ✅ | params from env overrides only (`:228`), ETS internally optimised on `y_tr` (`:166`) |
| 6 | Seeded & reproducible | ⚠️ | **no seed set anywhere in `run_a_stat.py`**; deterministic in practice — double-run was **bit-identical** (§7) |
| 7 | Inference features == training | ✅ | univariate, uses only `y_tr` (`:155-167`); SARIMAX uses no exog (`:183-184`) |

> Note: a separate `a_stat_models_pipeline.py` exists that *refits ETS per test
> point* (`:575`, `:604-606`) — this would FAIL check #1 — but it is **not** the
> runner used by the daily pipeline, so it is out of scope for this run.

### B_ML (8 sklearn/boosting models + baseline) — `b_ml_pipeline.py`

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Fit once per fold | ✅ | `estimator.fit` at `:642`, outside the per-origin predict loop `:672` |
| 2 | Preprocessing fit on train only | ✅ | `StandardScaler` inside sklearn `Pipeline`, fit on `X_tr_fit` only (`:265-267`, `:635`, `:642`) |
| 3 | Prediction target matches training target | ✅ | level target `y(t+h)` via `shift(-h)` (`:603`); no differencing for a flow; truth is level (`:710`) |
| 4 | Baseline correct & same window | ✅ | plain h-step persistence via `origin_value = y(t−h)` (`:674`, `:818`, `:821`); same window as models |
| 5 | No tuning on test | ✅ | no GridSearch/CV; hyperparams hard-coded (`:263-288`); val split is train-internal only |
| 6 | Seeded & reproducible | ✅ | global seed `:453`; every estimator has `random_state=0` incl. RF/ExtraTrees/XGB/LGBM (`:274-287`) |
| 7 | Inference features == training | ⚠️ | same builder (`:597`/`:623`); minor: inference `fillna(0.0)` (`:689`) vs training median-impute |

### E_QUANTILE (GBQuantile, ResidualRF) — `e_quantile_daily_pipeline.py`

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Fit once per fold | ✅ | fit inside fold loop `:262`, predicts whole test block (`:176-177`) |
| 2 | Preprocessing fit on train only | ⚠️/❌ | univariate ✅ (backward-only features `:102-106`); **multivariate leaks** — global `bfill` (`:115`) and full-series top-K feature selection (`:116-119`). *Daily run is univariate, so not exercised.* |
| 3 | Prediction target matches training target | ✅ | h-step target built once (`:122-128`); no transform, no inverse needed |
| 4 | Baseline correct & same window | ✅ | persistence = `origin_value` (`:131`, `:299`); same `_valid` rows (`:376`) |
| 5 | No tuning on test | ✅ | fixed hyperparams (`:175`, `:190-192`) |
| 6 | Seeded & reproducible | ⚠️ | seeds set (`:175`, `:191`); RF `n_jobs=-1` gives ~1e-8 run-to-run float noise (§7) |
| 7 | Inference features == training | ✅ | single `X_all`, sliced by fold (`:264-265`) |

### C_DL (LSTM/GRU/DCNN/Transformer/MLP) — `c_dl_pipeline.py`

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Fit once per fold | ✅ | fold→model→single `train_model` (`:746`, `:789`, `:796`) |
| 2 | Preprocessing fit on train only | ✅ | feature scaler fit on `X_fit` only (`:772-776`) |
| 3 | Prediction target matches training target | ❌ (see Critical-1) | transform "none" (`run_c_dl_univariate.py:79`); **features standardised but target left in raw ~10⁸ units → predictions collapse to ≈0** |
| 4 | Baseline correct & same window | ✅ | h-step persistence via `origin_value` (`:859`); skill computed (`:861`) |
| 5 | No tuning on test | ✅ | architectures hard-coded (`:569-576`); early-stop uses train-internal val |
| 6 | Seeded & reproducible | ⚠️ | `torch.manual_seed`+`np.random.seed` (`:665`); **no cuDNN determinism flags** (GPU only; CPU run here is reproducible) |
| 7 | Inference features == training | ✅ | single `build_sequences`, train/test are mask slices (`:717-748`) |

---

## 5. Independent metric audit (one fold per family, plain numpy)

Every pipeline number was recomputed from the stored `predictions_long.csv` with
plain numpy. **All matched.**

| Family | Quantity | Pipeline | Independent numpy | Match |
|---|---|---:|---:|:--:|
| A_STAT | ETS MAE | 42,549,097 | 42,549,096.59 | ✅ |
| A_STAT | Baseline MAE / skill | 44,233,643 / 3.81% | 44,233,643.69 / 3.81% | ✅ |
| B_ML | Lasso MAE | 43,472,942 | 43,472,942.15 | ✅ |
| B_ML | h-step persistence | 60,976,736 | 60,976,736.58 | ✅ |
| B_ML | `mae_shift_minus_h` | 43,507,476 | 43,507,476.12 | ✅ |
| E_QUANTILE | pooled skill | 55.40% | 55.40% (n=20) | ✅ |
| C_DL | DCNN MAE / skill | 73,425,360 / −38.65% | 73,425,360 / −38.65% | ✅ |

**Conclusion:** the metric *arithmetic* is correct across all four families. The
problems are in **baseline definitions, evaluation windows, and model behaviour**
— not in how MAE/skill are summed.

---

## 6. Reproducibility (fresh reruns)

The prior run folder `backend/forecast_runs/2026-07-17/` exists but is **empty**
(no `predictions_long.csv`), so the cross-day diff had nothing to compare. Instead
I ran the two cheapest families **twice each** into isolated scratch dirs (audited
artifacts untouched) and diffed predictions, ignoring run-date metadata.

| Family | run1 vs run2 | run1 vs stored 07-21 | Verdict |
|---|---|---|---|
| A_STAT | **bit-identical** (max\|Δ\|=0) | **bit-identical** (max\|Δ\|=0) | fully deterministic ✅ |
| E_QUANTILE | identical to ~6e-08 | identical to ~6e-08 | deterministic to float tolerance ✅ |

The E_QUANTILE ~6e-08 wobble (≈1e-15 relative, on ~10⁷ values) comes from
`RandomForestRegressor(..., n_jobs=-1)` (`e_quantile_daily_pipeline.py:191`)
summing across threads in nondeterministic order. The **seed is set**
(`random_state=42`), no metric changes — this is a Minor note, not an unseeded
source. **No family is meaningfully non-deterministic.** (B_ML is fully seeded per
§4; C_DL is reproducible on CPU, the mode used here.)

---

## 7. Findings, ranked

Each finding: severity, `file:line`, and a one-paragraph plain-English
explanation.

### 🔴 CRITICAL

**C-1 — C_DL neural nets are broken: they output ≈0 and are worse than doing
nothing.**
`run_c_dl_univariate.py:79` (`target_transform="none"`) + `c_dl_pipeline.py:772`
(feature scaler fit on `X` only; the *target* is never scaled).
All five architectures produce point forecasts averaging **~283** while the actual
revenue averages **~71,600,000**. The result: MAE ≈ the average size of the
target itself (73.4M), and skill is **−38.65%** on the full window (**−65%** on the
2025 window). In plain terms: the pipeline feeds the network inputs that have been
shrunk to roughly unit size (mean 0, std 1) but asks it to predict raw numbers in
the hundreds of millions. With only a few training epochs the network's outputs
stay near their small initial values (~0) and never climb to the right scale. The
integrity file correctly records this as `FAILED_QUALITY`, but the models as
shipped are unusable.

**C-2 — "Skill vs persistence" is not comparable across families because the
baseline is defined three different ways.**
`run_a_stat.py:291,299-300` (flat last-value) vs `b_ml_pipeline.py:674,818`
(h-step persistence) vs `forecast_integrity.py:215,249` (a model-shift
diagnostic). As shown in §3, A_STAT grades itself against a flat-line baseline
(44.2M) while B_ML/C_DL grade against genuine h-step persistence (60.9M). Because
the yardsticks differ, you cannot compare A_STAT's "+3.81%" to B_ML's "+28.71%" —
they are measured against different things. A trustworthy leaderboard needs **one**
persistence definition used identically everywhere.

**C-3 — A_STAT ignores the horizon and forecasts the entire test year in one
shot.**
`run_a_stat.py:135` (`n = len(idx)`), `:167` (`fit.forecast(n)`), `:303`
(`horizon_note="stat_models_forecast_full_test_window"`).
The config says `h = 5`, but A_STAT never uses it to build the forecast: it fits
ETS once at 2024-12-31 and predicts *every* business day of the following window
(156 steps) as essentially a flat line. So A_STAT's "MAE at h=5" is really the
error of a one-shot, 1-to-156-step-ahead flat forecast. That is a fundamentally
different quantity from B_ML/C_DL/E_QUANTILE, all of which forecast exactly 5 steps
ahead from rolling origins. This is why A_STAT's origin_date is constant
(2024-12-31) in its predictions while B_ML's advances daily.

### 🟠 MAJOR

**M-1 — `SUMMARY.txt` hides C_DL's failure.**
`daily_summary.py` (the summary writer) reports C_DL as *"Best model: DCNN (MAE
73,425,360) … Skill: n/a … Run status: n/a"*, even though C_DL's own
`c_dl/daily/artifacts/integrity_Revenues_h5.json` clearly recorded
`skill_pct = -38.65` and `run_status = "FAILED_QUALITY"`. The summary reads the
other families' `integrity_report.json` but not C_DL's differently-named
`integrity_Revenues_h5.json`, so it silently drops the one signal that would tell
an operator "the deep-learning models failed." Presenting a failed model as the
"best model" with a clean-looking MAE is dangerous.

**M-2 — E_QUANTILE's 55.4% skill and its coverage rest on only ~10 test points.**
`e_quantile_daily_pipeline.py:60` (test block = `horizon` = 5 points/fold),
`:67-69` (folds silently dropped when training data is short), `:23` (folds=3
requested → **2 survive**).
The impressive +55% skill is computed over **2 folds × 5 days = 10 target dates
per model** — specifically only the last two business weeks, 2025-07-28 →
2025-08-06. Ten points is far too few to claim a model "beats persistence by
55%"; a single lucky or unlucky day moves the number by several percent. The
headline figure also **pools both models' P50 predictions together**
(`:369,375-380`), so "55.4%" is not attributable to either model alone.

**M-3 — E_QUANTILE ResidualRF prediction intervals are far too narrow (50%
coverage vs 80% nominal).**
`e_quantile_daily_pipeline.py:193-195` (residuals from the *training* set the model
just fit), `:199-200` (band = those in-sample residual quantiles).
ResidualRF builds its P10/P90 band from the errors it made **on its own training
data**. But it's a 400-tree, unlimited-depth forest (`:190-192`) that nearly
memorises the training set, so its training errors are artificially tiny — giving
a band ~3.8× narrower than GBQuantile's (24.5M vs 92.8M) and covering only **50%**
of actual values instead of 80%. The docstring even claims a "CV-like split"
(`:182-185`) that the code never performs. An interval that covers half of reality
while advertising 80% will badly mislead any risk decision.

**M-4 — B_ML models have essentially no predictive signal at h=5 and overfit
heavily; the "+28.71%" skill is regression-to-the-mean, not learning.**
`b_ml_pipeline.py:655-667` (train vs val MAE), `:274-275` (unbounded trees).
ExtraTrees scores `train_MAE = 0.00` (it memorises training data perfectly) while
its validation error is far higher; val/train ratios reach 30.4× (LightGBM),
10.45× (XGBoost), 3.97× (RandomForest) — textbook overfitting. Meanwhile the
label-shuffle sentinel (M-5) shows the features carry almost no information about
revenue 5 days ahead. So how does Lasso "beat persistence by 28.7%"? Because the
h-step persistence baseline (60.9M) is *inflated by holiday spikes*, and Lasso
simply predicts something close to the recent average — a smoother line that
avoids those spikes. It beats a volatile baseline by regressing toward the mean,
not by discovering structure. The number is arithmetically real but should not be
read as "the model learned to forecast revenue."

**M-5 — B_ML's leakage detector is misconfigured and its verdict here is
misleading.**
`preprocessing/integrity.py:260` (`leakage_warning = ratio < 1.5`), `:247-249`
(shuffle), `b_ml_pipeline.py:950-953` (train-derived, overlapping slices).
The check shuffles the target labels, refits Ridge, and flags "leakage" if the
shuffled model isn't much worse (`ratio < 1.5`). Here `ratio = 0.83`, meaning the
**label-shuffled model was actually better** than the real one — that is the
*opposite* of leakage; it means the features are noise for this target. Reporting
this as `leakage_warning=true` is a false alarm caused by a threshold that
conflates "no signal" with "leakage." Worse, the sentinel's "test" set is sliced
from the **training** features (and overlaps the "train" slice when data is short),
so it never measures true held-out leakage at all. Related: the *"Suspicious
backward shift best_shift=-10"* message is the same no-signal story — the shift
search (`forecast_integrity.py:184-185`, range ±10) finds its "best" alignment at
the very edge of the window, which happens because a nearly flat/low-signal
prediction barely changes MAE under any shift; it is not evidence of a real
10-step lag bug.

### 🟡 MINOR

**m-1 — `run_a_stat.py` sets no random seed.** No `np.random.seed`/`random_state`
anywhere in the file; the statsmodels optimisers are deterministic in practice and
the double-run was bit-identical (§6), but explicit seeding is absent. (Contrast
`a_stat_models_pipeline.py:508`, which does seed — but is not the runner used.)

**m-2 — E_QUANTILE run-to-run float noise (~1e-8).** From
`RandomForestRegressor(n_jobs=-1)` (`e_quantile_daily_pipeline.py:191`) reducing
across threads out of order. Seeded and harmless; no metric changes.

**m-3 — C_DL missing cuDNN determinism flags.** No
`torch.backends.cudnn.deterministic`/`benchmark` or `torch.cuda.manual_seed`
(`c_dl_pipeline.py:665` seeds only CPU/numpy). Affects GPU only; this run was CPU
and reproducible.

**m-4 — B_ML inference vs training NaN handling differ.** Inference forces
`fillna(0.0)` on the feature row (`b_ml_pipeline.py:689`) while training relies on
the pipeline's median imputer (`:265`). Immaterial at well-populated origins, but
not strictly identical.

**m-5 — E_QUANTILE multivariate feature leakage (not exercised here).** Global
`bfill` (`:115`) and full-series top-K feature selection (`:116-119`) would leak
test information — but the daily run uses the univariate path, so this did not
affect the 2026-07-21 numbers. Flagged so it is fixed before multivariate is used.

---

## 8. Proposed fix order (for a later phase — nothing changed here)

Ordered by trust impact, cheapest-decisive-fix first within a tier:

1. **(C-1) Scale the C_DL target.** Normalise `y` for the neural nets (or model in
   log/relative space) and inverse-transform predictions; verify predictions land
   in the right order of magnitude before trusting any DL number.
2. **(C-2 + C-3) Unify the baseline and honour the horizon in A_STAT.** Adopt a
   single shared h-step persistence (`ŷ(t)=y(t−h)`) used identically by every
   family and the leaderboard; make A_STAT forecast at horizon `h` from rolling
   origins like the others, so all four families are on one window/definition.
3. **(M-1) Make `daily_summary.py` read C_DL's `integrity_Revenues_h5.json`** so a
   `FAILED_QUALITY`/negative-skill result can never be presented as "best model."
4. **(M-2 + M-3) Strengthen E_QUANTILE's evaluation and intervals.** Widen the
   test window / add folds so skill isn't resting on 10 days; rebuild ResidualRF's
   band from genuinely out-of-sample residuals (or use true quantile regression)
   and re-check coverage.
5. **(M-4 + M-5) Address B_ML overfitting and the leakage sentinel.** Bound tree
   depth / regularise; re-examine whether any feature carries h=5 signal; fix the
   sentinel to use real held-out data and correct the `ratio < 1.5` semantics so
   "no signal" is not reported as "leakage."
6. **(Minor) Housekeeping.** Seed `run_a_stat.py`; document/relax the E_QUANTILE
   `n_jobs` float noise; add cuDNN determinism flags for GPU C_DL runs; align B_ML
   inference NaN handling with training.

---

*Prepared for branch `audit/model-review`. Diagnosis only — no source files were
modified. All pipeline metrics were independently reconstructed in numpy and
matched to the cent; all `file:line` references were verified against the code as
it stands on this branch.*
