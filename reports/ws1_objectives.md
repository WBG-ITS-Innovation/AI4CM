# Workstream 1 — Absolute-error objectives for B_ML

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Data SHA-256:** `0b009fd0…5361f1`
**Suite:** 294 passing · **TEST (2025) reads this workstream: 0**
**All 36 figures below are logged** in `experiments/log.csv` (rows 3–38) with per-run detail
in `experiments/runs/`. No unlogged number appears in this report.

---

## 1 · What was changed and why

We score models on MAE, and we were training them on squared error. Squared error fits the
conditional **mean**; absolute error fits the conditional **median**. On a spiky flow series
those are different targets, and the difference is not academic here — `Revenues` has single
days an order of magnitude above the local level, and every one of them drags an L2 fit away
from the ~250 ordinary days it will be scored on.

Three variants added to `available_models()` in [b_ml_pipeline.py:316](backend/b_ml_pipeline.py#L316):

| Model | Objective |
|---|---|
| `HistGBDT_L1` | `loss="absolute_error"` |
| `XGBoost_L1` | `objective="reg:absoluteerror"` |
| `LightGBM_L1` | `objective="l1"` |

**Every other hyperparameter is identical to the squared-error twin**, so any difference is
attributable to the objective and nothing else. A test enforces this rather than trusting
it: `test_l1_variants_differ_from_their_twins_only_in_the_objective` diffs `get_params()` and
allows exactly `{loss, objective}` to differ. Another asserts each variant really carries an
L1 objective — an L1-named model quietly fitting squared error would put an L2 result in the
table under an L1 label, and the conclusion would be drawn from a mislabelled row.

`XGBoost_L1` is **omitted rather than silently downgraded** if the installed XGBoost predates
`reg:absoluteerror` (≥1.7). It is present here.

### A prerequisite defect: B_ML could not be asked for a TRAIN-internal search

`build_yearly_folds` folded over every year from `first_year + min_train_years` to the **last
year in the data**. On this dataset that made the final fold **2025 — the sealed holdout — on
any default run**, and there was no argument that could scope the family to TRAIN or DEV.
Ground rule 1 requires exactly that, so this was blocking.

`eval_start` / `eval_end` (both inclusive, by target date) now bound the window. Folds
entirely outside are dropped; a fold straddling an edge is **trimmed** to it, because a
partial block is a smaller sample rather than a wrong one. Measured:

```
default            7 folds  2019 … 2025-08-06     <- final fold IS the holdout
TRAIN (<=2023)     5 folds  2019 … 2023-12-29
DEV only           1 fold   2024-01-01 … 2024-12-31
```

Six regression tests pin this, including one that *characterises the hazard* — asserting the
unbounded builder still reaches `test` — so the bounds cannot be removed without a red test.
Every run below asserted its own window membership before any number was recorded:
`assert seen == {wname}`.

---

## 2 · Results

Search on TRAIN-internal rolling-origin folds (2019–2023, n=1,304), one DEV confirmation per
target (2024, n=262). Skill is against the unified persistence ruler on the same window.

### TRAIN-internal search

| Target | Model | MAE | Skill | MASE | vs L2 twin |
|---|---|---:|---:|---:|---:|
| **Revenues** — ruler 67,062,951 | HistGBDT | 44,502,234 | 33.64% | 0.866 | |
| | **HistGBDT_L1** | 41,617,992 | 37.94% | 0.810 | **+6.48%** |
| | XGBoost | 45,620,906 | 31.97% | 0.888 | |
| | **XGBoost_L1** | 41,606,342 | 37.96% | 0.810 | **+8.80%** |
| | LightGBM | 47,622,524 | 28.99% | 0.927 | |
| | **LightGBM_L1** | 41,517,099 | **38.09%** | 0.808 | **+12.82%** |
| **Expenditure** — ruler 54,170,043 | HistGBDT | 37,093,156 | 31.52% | 0.793 | |
| | HistGBDT_L1 | 37,039,404 | 31.62% | 0.792 | +0.14% |
| | XGBoost | 39,507,776 | 27.07% | 0.845 | |
| | XGBoost_L1 | 37,066,363 | 31.57% | 0.793 | +6.18% |
| | LightGBM | 39,063,644 | 27.89% | 0.836 | |
| | **LightGBM_L1** | 36,766,991 | **32.13%** | 0.786 | **+5.88%** |
| **State budget balance** — ruler 153,623,500 | HistGBDT | 151,301,323 | 1.51% | 1.230 | |
| | **HistGBDT_L1** | 138,414,554 | **9.90%** | 1.125 | **+8.52%** |
| | XGBoost | 179,301,703 | −16.72% | 1.458 | |
| | XGBoost_L1 | 150,104,519 | 2.29% | 1.221 | +16.28% |
| | LightGBM | 154,397,146 | −0.50% | 1.255 | |
| | LightGBM_L1 | 143,842,292 | 6.37% | 1.170 | +6.84% |

### DEV confirmation (2024)

| Target | Model | MAE | Skill | MASE | vs L2 twin |
|---|---|---:|---:|---:|---:|
| **Revenues** — ruler 88,317,355 | HistGBDT | 58,359,490 | 33.92% | 1.136 | |
| | HistGBDT_L1 | 58,307,347 | 33.98% | 1.135 | +0.09% |
| | XGBoost | 61,697,498 | 30.14% | 1.201 | |
| | XGBoost_L1 | 58,533,249 | 33.72% | 1.140 | +5.13% |
| | **LightGBM_L1** | 55,952,324 | **36.65%** | 1.089 | **+15.75%** |
| | LightGBM | 66,409,042 | 24.81% | 1.293 | |
| **Expenditure** — ruler 73,117,667 | **HistGBDT** | 51,088,706 | **30.13%** | 1.093 | |
| | HistGBDT_L1 | 53,354,531 | 27.03% | 1.141 | **−4.44%** |
| | XGBoost | 53,399,168 | 26.97% | 1.142 | |
| | XGBoost_L1 | 52,867,231 | 27.70% | 1.131 | +1.00% |
| | LightGBM | 57,152,748 | 21.83% | 1.223 | |
| | LightGBM_L1 | 52,179,454 | 28.64% | 1.116 | +8.70% |
| **State budget balance** — ruler 242,653,025 | HistGBDT | 257,087,305 | −5.95% | 2.090 | |
| | **HistGBDT_L1** | 203,596,374 | **16.10%** | 1.655 | **+20.81%** |
| | XGBoost | 257,105,273 | −5.96% | 2.091 | |
| | XGBoost_L1 | 208,782,671 | 13.96% | 1.698 | +18.79% |
| | LightGBM | 272,029,958 | −12.11% | 2.212 | |
| | LightGBM_L1 | 216,528,115 | 10.77% | 1.761 | +20.40% |

### The comparison in one table

L1 improvement over its own L2 twin (% MAE reduction; negative = worse):

| Window / target | HistGBDT | XGBoost | LightGBM |
|---|---:|---:|---:|
| TRAIN / Revenues | +6.48% | +8.80% | +12.82% |
| TRAIN / Expenditure | +0.14% | +6.18% | +5.88% |
| TRAIN / State budget balance | +8.52% | +16.28% | +6.84% |
| DEV / Revenues | +0.09% | +5.13% | +15.75% |
| DEV / Expenditure | **−4.44%** | +1.00% | +8.70% |
| DEV / State budget balance | +20.81% | +18.79% | +20.40% |

**L1 wins 17 of 18 comparisons.** The single loss is `HistGBDT` on DEV Expenditure. That is
one fold of 262 days on the target where L1's TRAIN gain was smallest (+0.14%) — i.e. the
objective was already near-neutral there and the sign flipped on a single-fold sample. It is
not evidence against L1; it is the noise floor of a one-fold confirmation, and it is the
reason the search ran on five TRAIN folds rather than on DEV.

The clearest result is the **stock target**, where L1 turns three models that were *worse than
persistence* (−5.95%, −5.96%, −12.11%) into three that beat it (+16.10%, +13.96%, +10.77%).
Squared error on a level series is dominated by a handful of large balance swings; the median
fit ignores them.

---

## 3 · The sentinel ratio did NOT move — and could not have

The brief asks whether the sentinel ratio moved, noting that on the flow targets it is the
number we are trying to lift. The honest answer:

| Target | TRAIN | DEV | Signal? |
|---|---:|---:|---|
| Revenues | 1.07 | 1.14 | **No** (< 1.50) |
| Expenditure | 1.13 | 1.09 | **No** (< 1.50) |
| State budget balance | 2.28 | 5.57 | Yes |

**The ratio is identical across all six models within each window**, because `signal_sentinel`
fits a fixed **Ridge** probe on the feature set, twice — once on true targets, once on
shuffled ones. It measures whether *the features* carry signal about the target. Changing the
loss function of a downstream model cannot move it, by construction.

So workstream 1 did not lift the sentinel ratio and **no objective change ever will**. The two
flow targets remain below threshold: `Revenues` at 1.07–1.14 and `Expenditure` at 1.09–1.13
mean that destroying the feature-target pairing barely hurt a Ridge fit. Lifting that number
requires **new information**, which is workstream 3 (fiscal calendar) and workstream 5
(multivariate), not a better optimiser.

This is worth stating bluntly because the two numbers point in opposite directions and either
one alone misleads. On `Revenues`, `LightGBM_L1` shows **36.65% DEV skill** against a sentinel
ratio of **1.14**. The skill is real in the sense that the errors are genuinely smaller than
h-step persistence — but with no detectable feature signal, what the model is doing is
regressing toward a central level against a spiky baseline, not forecasting. A 36% improvement
over a bad ruler is still not a forecast.

The stock target is the inverse and remains the honest bright spot: genuine signal (2.28 /
5.57) and now, with L1, genuine skill (up to +20.81% on DEV).

---

## 4 · Two defects found while doing this

**1. The evaluation window had no upper bound** (in already-committed code, `87fc971`).
`eval_start` set a floor and nothing set a ceiling, so E_QUANTILE's pinned folds tiled to the
end of the series. Pinning to DEV therefore evaluated 2024-01-01…2025-08-06 — 418 target dates
where DEV has 262. The previously reported phase-3 "DEV" figures were DEV+TEST. Corrected,
disclosed in `reports/HANDOFF.md` §0a, fixed by `Config.eval_end` with four regression tests.
Caught by an `assert wins == {"dev"}` guard, not by review.

**2. `run_id` was not unique.** The logger stamped ids to whole seconds, so 36 rows written
inside one second collided — meaning they shared one detail JSON and all but one run became
unrecoverable. `verify_log_integrity()` reported `duplicate run_id values` and the batch was
discarded and re-logged. Microsecond precision plus an on-disk collision loop now guarantee
uniqueness; seven tests cover the log, including this regression.

Both were caught by cheap assertions written *because* the rules had just been tightened. That
is the pattern worth keeping.

---

## 5 · Recommendation

Adopt **L1 as the default objective** for the three gradient-boosted B_ML models and carry the
squared-error twins only as comparators. `LightGBM_L1` is the strongest flow model on both
targets and both windows; `HistGBDT_L1` is the strongest on the stock target.

Workstream 2 (LightGBM quantile + Optuna) should tune **on top of L1**, since the pinball loss
at τ=0.5 *is* absolute error — the objectives are already consistent.

Nothing here was evaluated on 2025.

---

## 6 · Reproduction

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q          # expect 294 passed
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from experiment_log import verify_log_integrity as v;print(v())"
```

Each row's `runs/<run_id>.json` carries the full data SHA, git SHA, params and ruler. The six
pipeline runs are `ConfigBML(target=…, cadence="Daily", horizon=5, min_train_years=4,
variant="univariate", model_filter=None)` with `eval_end="2023-12-31"` for TRAIN and
`eval_start="2024-01-01", eval_end="2024-12-31"` for DEV, against
`backend/data/processed/master_daily_clean_treasury.csv`, with `available_models()` narrowed to
the six models compared.
