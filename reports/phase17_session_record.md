# Phase 17 — Item 2: the Models detail view

**Date:** 2026-08-10
**Branch:** `model/excellence` @ `b9aeee2` (34 ahead of `origin/main` @ `863f967`)
**Root suite:** 498 → **516 passed, 3 skipped**, `EXIT=0` · **Frontend:** 106 passed, `EXIT=0`
**TEST (2025) gated reads: 0** · **PR #26** open, reused · Log 153 rows, integrity ok

---

## Completed — item 2

`backend/model_reference.py` assembles the reference and emits JSON; `frontend/pages/03_Models.py`
renders it. Assembled in the **backend** interpreter, because the modelling libraries live there
and not in the one running Streamlit — the dispatch pattern item 0 established.

### Where each piece comes from

| Piece | Source | Mechanism |
|---|---|---|
| Descriptions | `model_reference.DESCRIPTIONS` | Prose — the only non-derived content. Served with `description_kind` so the page labels it |
| Hyperparameters | `b_ml_pipeline.available_models()` → `get_params(deep=True)` | Read **live**, diffed against a library-default baseline |
| Availability | `HAVE_XGB` / `HAVE_LGBM` / `HAVE_CATBOOST` | Unavailable models are listed with the missing library named |
| Measured performance | `experiment_log.read_log()` + `experiments/runs/<run_id>.json` | Per target, `run_id` on every row; nothing recomputed |
| Coverage | `coverage_low/mid/high` in those rows | `None` for point models → "not reported" |
| Gate + recipe | `registry.load_registry()` | Rendered as stored, approver included |

### All 16 pool models described

13 B_ML + 3 E_QUANTILE, plus `Theta` and `ETS` described but flagged as not in the pool. A test
greps every description for a percentage and fails if one appears — a description says what a model
*is* and must not read as evidence.

### The hyperparameter bug my own test caught

For a `Pipeline`, the library-default baseline cannot be `type(est)()`: **`Pipeline()` is empty**, so
every prefixed parameter differs from it and the whole configuration reads as deliberately chosen.
Measured before the fix: **`Ridge` reported 17 "set" parameters**, including `copy_X`,
`fit_intercept` and `steps`.

Fixed by comparing each step against a fresh instance of *its own* class, adding the pipeline's own
unprefixed defaults (without which `verbose=False` read as a choice), and excluding plumbing.
Verified against the source:

| Model | Set | Values |
|---|---:|---|
| `HistGBDT_L1` | 3 of 21 | `loss=absolute_error`, `l2_regularization=1.0`, `random_state=0` |
| `Ridge` | 2 | `imp__strategy=median`, `est__random_state=0` — correctly **omits** the `StandardScaler` arguments, because those are the library defaults |
| `Lasso` | 3 | adds `est__max_iter=20000` but **not** `tol`, which *is* the sklearn default |
| `LightGBM_L1` | 9 | `objective=l1`, `n_estimators=800`, `learning_rate=0.05`, … |

A test mutates the estimator (`l2_regularization` 1.0 → 7.5) and asserts the reported value follows.
That is the property that matters: a transcribed table would drift from the pipeline in silence.

### The coverage constraint, measured

Counted across the 153 logged runs: **0** of the point-model rows carry a coverage figure; only the
**6** quantile/CQR rows do. Coverage returns `None` for point models and renders "not reported",
with a note explaining the absence. Two tests hold it — one asserting no point model reports
coverage, and its complement asserting the quantile runs **do**, without which the first would pass
on an empty payload.

### Traceability

Every measured row carries its `run_id`. Tests assert each one resolves to a real
`experiments/runs/<run_id>.json`, and cross-check 50+ values against `experiments/log.csv` so
nothing is recomputed on the page. A model with no logged run gets an empty state stating it is
**unmeasured** rather than bad, and that nothing is inferred from a sibling.

---

## Hard-stop checks

| Required | Result |
|---|---|
| Committed | ✅ `b9aeee2` |
| Suite green | ✅ 516 passed / 3 skipped `EXIT=0`; frontend 106 `EXIT=0` |
| Re-issue if a recipe changed | **Not required** — no recipe changed; `registry/recipes.json` untouched |
| Refresh progress record if a published number moved | **Not required** — `forecasts/` untouched |

No displayed forecast number changed: this item added a reference view over existing artifacts.

---

## Resume point

Two items remain from the earlier priority list:

**Show the Treasury baseline** — plot the h-step persistence baseline on the Forecast chart as its
own labelled series, and state the baseline MAE beside the model's on the Forecast header and the
Dashboard. `mae_persistence` is in every integrity report, so it needs no recomputation. *(The
Dashboard half already landed in item 1 as the "Benchmark MAE" KPI; the Forecast chart series and
header figure are outstanding.)*

**Persist the fitted estimator with each published forecast** — save the estimators into
`forecasts/published/<issue_date>/` with library names and versions, a loader, a test that a saved
model reproduces its published predictions to tolerance, and the storage cost per issue stated.
Official runs must still refit on current data; this is for reproducing a published issue, not for
serving a stale model.

### Artifact fields still missing — unchanged

Seven, listed in `reports/phase14_session_record.md`. The worst remains `c_dl_pipeline.py:958`
writing `alignment_ok: True` as a literal with no check behind it. Item 2 was blocked by one of
them — point-model runs logging no coverage — and handles it by reporting "not reported" rather
than backfilling.
