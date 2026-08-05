# Phase 16 — Timeboxed session (10 min): the Forecast crash

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `56e90b4` (33 ahead of `origin/main` @ `863f967`)
**Root suite:** 495 → **498 passed, 3 skipped**, `EXIT=0` · **Frontend suite:** 99 passed
**TEST (2025) gated reads: 0** · **PR #26** open, reused · Log 153 rows, integrity ok

---

## ITEM 0 — COMPLETE. This was the session's completion criterion.

Both Forecast buttons raised `ModuleNotFoundError: matplotlib` from `b_ml_pipeline.py:32` when the
Streamlit frontend imported the pipeline to run a forecast.

### Two layers, and installing matplotlib would have fixed neither properly

**Layer 1 — the stated bug.** `import matplotlib.pyplot as plt` at module level made pyplot a hard
*import-time* dependency of the **forecasting** path, when it is only needed by the chart-writing
functions. Deferred via `backend/lazy_plot.py`: a proxy that imports pyplot on first attribute
access, selects the `Agg` backend (these are file-writing charts on a machine with no display), and
raises a message naming the cause if matplotlib is genuinely absent. All eight existing `plt.` call
sites work unchanged.

**Layer 2 — found by verifying, not assuming.** With pyplot deferred the import then failed on
**sklearn**. The frontend venv has none of sklearn, lightgbm, xgboost, catboost or matplotlib — and
should not: it renders with Plotly and the models belong to the backend. Installing the whole
modelling stack into the frontend to satisfy an import would have been treating the symptom.

So the page now **dispatches to the backend interpreter** and reads JSON — the pattern the Lab page
and the model-pool lookup already use. One interpreter owns the models.
`backend/forecast_modes.py` gained a CLI (`--mode/--target/--horizon/--model/--publish`) returning
either a result or an explained refusal.

### Verified end to end from the page, both modes

**Official** — Revenues, recipe `revenues-lgbm-l1-ws3-v1`, model `LightGBM_L1`, approved by
**none**:

| Target date | h | P50 (M GEL) | Band |
|---|---:|---:|---|
| 2025-08-07 | 1 | 99.6 | 42.4 – 147.8 |
| 2025-08-08 | 2 | 76.4 | 46.3 – 108.8 |
| 2025-08-11 | 3 | 77.6 | 38.8 – 104.2 |
| 2025-08-12 | 4 | 61.6 | 46.1 – 102.7 |
| 2025-08-13 | 5 | 58.3 | 44.1 – 141.9 |

**Exploratory** — `Taxes` (which has no champion recipe), `Ridge`, h=3, banner-marked ungated:
37.1 / 52.0 / 39.5 M GEL on 2025-08-07, 08-08, 08-11.

**Refusal** — official mode on `Taxes` returns `ok=false, refused=true` with the plain explanation
that substituting another target's recipe would attach five folds of evidence to a model it was
never measured on.

Rendering both modes through `AppTest`: **0 exceptions**, 41 targets offered in official mode, 13
models in exploratory.

Three regression tests added: pyplot is not imported at module level, the proxy defers, and the page
does not import `forecast_modes` directly.

---

## ITEMS 1–4 — NOT STARTED

The box was 10 minutes and item 0 consumed it. Not begun rather than half-built.

---

## Hard-stop checks

| Required | Result |
|---|---|
| Committed | ✅ tree clean |
| Suite green | ✅ 498 passed, 3 skipped, `EXIT=0`; frontend 99 |
| Re-issue if a recipe changed | **Not required** — no recipe changed |
| Refresh progress record if a published number moved | **Not required** — `forecasts/` untouched; still 1 issue, 0 scored / 15 pending |

No displayed number changed: item 0 repaired an import path and moved execution to another
interpreter. The figures above are newly *reachable*, not newly *different* — they match the
published `2025-08-06` issue exactly.

---

## Resume point, in the operator's priority order

**ITEM 1 — Dashboard KPI strip.** Delete the letter-grade card (`F / POOR`) and emoji from metric
labels. Replace with MAE (GEL millions), skill vs the unified ruler, MASE, sentinel **with its 1.50
threshold shown**, interval coverage against its nominal level **read from the artifact**, and gate
status as passed / failed / never verified. Delete any invented composite — the grade and "monthly
accuracy within 10% tolerance" — unless it traces to a logged definition, and state the formula
wherever one is kept. Console tokens, `help=` on every metric, "not reported" where unlogged.

*Known blocker for that item:* B_ML publishes no nominal interval level (`ConfigBML.nominal_pi` is
captured at the point of use but not written to the artifact), so coverage for B_ML runs must render
"not reported" rather than being scored against an assumed 90%. `frontend/intervals.py` already
handles this correctly and is wired into the Dashboard.

**ITEM 2** — plot the h-step persistence baseline on the Forecast chart as its own labelled series,
and state the baseline MAE next to the model's on both the Forecast header and the Dashboard.
`mae_persistence` is in every integrity report, so it is readable without recomputation.

**ITEM 3** — persist fitted estimators alongside each published issue, with library versions, a
loader, a reproduction test, and the storage cost per issue stated.

**ITEM 4** — Models tab detail view. Its blocker is unchanged: point-model runs log no per-tercile
coverage, so that field will legitimately read "not reported" for all three champions.
