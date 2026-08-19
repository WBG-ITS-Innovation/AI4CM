# Session — regenerate stale artifacts, and complete the family-vs-Ops picture

**Date:** 2026-08-18 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`); the agent
repo was not touched.
**Suite:** 911 → **925 passed, 4 skipped**.

Three tasks. All delivered, and two produced findings larger than the tasks themselves.

| | Task | Outcome |
|---|---|---|
| 1 | Regenerate the stale zero-baseline artifacts | done — plus **two more live copies of the bug** found and fixed |
| 2 | Sealed-window predictions for the champions | done — required a **new harness**; the champion turned out **not reproducible** |
| 3 | Final measured table, champion vs naive and vs Ops | done — supersedes the July deck |

> ## ⚠️ The two findings that matter most
>
> **1. `ws2_tune`'s DEV fold is scored against 4 holdout rows.** It calls `assert_selection_free`
> on evaluation *origins* (all DEV) and then reads truth at `origin + H` — 4 of which land in the
> sealed window. So every champion credential in `registry/recipes.json` includes 4 holdout
> observations. This is **selection on holdout data**, the one thing the four-window split exists
> to prevent. Measured impact on DEV MAE: Revenues 0.82%, Expenditure 1.16%, stock 0.51%.
>
> **2. The champion recipes are not reproducible.** The script that produced their credentials is
> absent from the repository. Reconstruction differs by **5.1% / 8.1% / 19.5%**, with n=250 against
> a logged n=262. Sealed-window figures therefore **supersede** those credentials rather than
> extend them.

---

## Task 1 — the stale artifacts, and two more copies of the bug

### The stale surface was narrower than the brief assumed, and not committed

| Named in the brief | Actually on disk |
|---|---|
| **C_DL** | 2 files, 5 rows, Revenues h=5 — the only non-null stale values |
| **A_STAT** | no ops column was ever written (`target,horizon,cadence,model,MAE,RMSE`) |
| **ensemble** | no artifacts exist |
| **weekly-stat** | no artifacts exist |
| B_ML (unasked) | column present, **all NaN** |

Plus 40 values across 8 files in `private_vault/`. And a correction to the premise:
**`backend/forecast_runs/**` is gitignored** — only 4 `SUMMARY` files are tracked, none of which
reference the metric. So the stale figures were on disk and citable, but never *committed*.

### What was done

* **Fresh C_DL run** at `backend/forecast_runs/2026-08-18/c_dl/` — 146s, 5 models, corrected ops.
* **50 stale values cleared** across 10 files (repo + vault), each with a sibling
  `OPS_SKILL_CLEARED.md` recording what was removed, the values, and why. Cleared rather than
  recomputed: those runs are the record of what they computed, and what they computed was wrong. A
  blank says "this run did not measure it", which is true.
* Verified afterwards: no stale non-null value remains anywhere outside the fresh run.

### Two more live copies of the bug

The zero-baseline defect had **four independent implementations**, and fixing one last session left
three wrong:

| File | State before | Now |
|---|---|---|
| `c_dl_pipeline.py` | fixed last session | also switched `profile` → canonical **flat** |
| **`run_a_stat.py`** | **2000/2000 values exactly zero** | delegates; 0 zeros, mean 56.8M |
| **`a_stat_models_pipeline.py`** | **raised `IndexError`** on the real series | delegates and runs |
| `b_ml_pipeline.py` | no zero bug, but a *different* method (`shift(12).rolling(36)` in month-groups = a 12-**year** shift) | left; recorded as an open item |

All now delegate to one construction. That duplication is precisely why one fix missed three
copies, so delegation was the point rather than a fourth patch.

**A consistency fix made mid-task:** `c_dl_pipeline` still called `method="profile"` while the
scorecard used flat, so the leaderboard and the scorecard were reporting against two different
comparators. Aligned to flat and re-run (146s). Profile also emitted 22 negative revenue baselines
over those rows.

### The corrected C_DL figures

`MAE_skill_vs_Ops`, as a fraction (fresh 2026-08-18 run, flat spread):

| Model | stale (vs zero) | corrected |
|---|---:|---:|
| LSTM | +0.2403 | **−0.5662** |
| GRU | +0.2310 | **−0.5768** |
| DCNN | +0.2795 | **−0.5366** |
| TRANSFORMER | +0.3052 | **−0.4800** |
| MLP | +0.3569 | **−0.5359** |

Every C_DL model is **48–58% worse** than the Treasury's current method, where the artifact claimed
23–36% better.

**Method note, since it was against my recommendation.** I proposed correcting the column in place —
the bug was in the comparator, not the predictions, and the metrics were exactly reproducible from
the stored predictions (5/5 MAEs matching to 1e-6). Full retraining was chosen instead, so the fresh
run's MAEs differ from 2026-08-04's (e.g. MLP 6.88e7 against 4.72e7): those are different models,
not corrected ones. The old run was left intact as the record of its own date, with only its wrong
column cleared.

---

## Task 2 — no harness existed, so one was built

### Why the obvious routes did not work

* **`b_ml_pipeline`** has a sealed-window path but **no `target_transform`**. The Revenues champion
  is *defined* by `target_transform: "ratio"` — ratio-to-trailing-level, 63-day causal median
  divisor, the WS4 winner. Running `LightGBM_L1` there produces a different model wearing the
  champion's name. (An early 4.1s probe did exactly that; it was not the champion.)
* **`ws2_tune.design`** implements the recipe, but its `make_folds` accepts only train/dev and calls
  `assert_selection_free` on evaluation rows. Its own comment: *"this is a SELECTION path, the one
  place a stale window does the most damage."*

### `backend/sealed_window_report.py`

Borrows the recipe from `design()`; rewrites fold construction as **reporting**.

* **Embargo the borrowed geometry lacks.** `build_yearly_folds` puts `train_end` on the last
  business day before the block while `design()` shifts targets over the whole series, so the last
  *H* training origins carry answers from inside the block — measured, **5 rows**. A training origin
  is kept only if its target predates the first evaluation origin, and causality is **asserted**,
  not trusted.
* **Ledger.** `require_test_access(..., PURPOSE_REPORT)`, naming the dates.
* **Never selects.** Asserted on the AST with docstrings stripped — the first version scanned raw
  text and tripped on its own prose ("never crowns anything"), which is how a source-text assertion
  becomes noise.
* **Parameter fidelity asserted.** The estimator comes from `available_models()` — the pipeline's own
  definitions — and every recipe-declared parameter is checked against it, so a drift fails loudly
  instead of measuring a different model as the champion. Currently identical for all three.

### The champion is not reproducible, and the record says so

| Target | reconstructed DEV MAE | logged credential | delta | n |
|---|---:|---:|---:|---:|
| Revenues | 36,936,198 | 38,931,956 | **5.13%** | 250 vs 262 |
| Expenditure | 47,414,587 | 51,602,951 | **8.12%** | 250 vs 262 |
| State budget balance | 156,349,023 | 194,104,922 | **19.45%** | 250 vs 262 |

The credentials are stamped `feature_names=['ws4:ratio']` / `['ws4:raw']`,
`fold_scheme="DEV confirmation, single 2024 fold"` — a string **no current script emits** —
`per_fold: []`, one recorded feature name. There is **no `scripts/ws4_*.py`**; only the two WS4
reports survive, and their Reproduction section re-runs tests rather than regenerating figures.

Parameter mismatch is ruled out (asserted identical), so the gap is fold geometry plus this
harness's embargo. `dev_reconstruction()` reports the gap beside any sealed-window figure so it
cannot be read as continuous with the credentials.

---

## Task 3 — the final measured table

**Sealed window (TEST 2025-01-01..2025-08-06), h=5, embargoed, `PURPOSE_REPORT` logged.**
This supersedes the July deck figures. Also at `reports/sealed_window_champion_vs_ops.csv`.

| Target | Role | Model | n | MAE | skill vs naive | **skill vs Ops** |
|---|---|---|---:|---:|---:|---:|
| **Revenues** | **CHAMPION** | LightGBM_L1 | 146 | 37,228,525 | **55.96%** | **+31.99%** |
| Revenues | best non-champion (E_QUANTILE) | ResidualRF | 204 | 33,259,969 | 49.55% | +20.86% |
| **Expenditure** | **CHAMPION** | LightGBM_L1 | 146 | 54,513,302 | **29.19%** | **+2.79%** |
| Expenditure | best non-champion | — none on disk — | 0 | — | — | — |
| **State budget balance** | **CHAMPION** | HistGBDT_L1 | 146 | 121,820,978 | **33.70%** | **n/a** |
| State budget balance | best non-champion (E_QUANTILE) | ResidualRF | 150 | 131,931,341 | 28.98% | n/a |

Ops MAE: Revenues 54,741,553 · Expenditure 56,076,535. `n/a` on the stock target because the method
aggregates a flow to an annual total and a balance level has none — not invented.

**How to read this honestly.**

* **Revenues is the strong case**: +32% over the current method, and the champion beats the best
  non-champion on the ops comparison (+32.0% vs +20.9%) despite a *higher* MAE — because ResidualRF
  is scored over 204 rows against the champion's 146, so the two MAEs are not directly comparable.
* **Expenditure is marginal**: **+2.79%**. Better than the current method, but by a margin that a
  different window could erase, and its verdict is `withheld` on accuracy anyway.
* **The stock target has no ops comparison at all**, and is `withheld`.
* `n=146` rather than 156: 4 rows lost to the embargo, the rest to incomplete features at the
  window edge.

---

## Open items

1. **`ws2_tune`'s DEV fold reads 4 holdout rows** — the leakage finding above. Fixing it changes
   what the tuner optimises and what the registry credentials mean, so it needs its own session.
   `test_sealed_window_report.py` pins it and will fail once fixed.
2. **The champion credentials are unreproducible** — the WS4 harness is gone. Either reconstruct it
   or formally supersede the credentials with figures from this harness.
3. **`b_ml_pipeline` keeps a divergent ops method** (12-year shift, flat spread). No zero bug, but
   it is not the Treasury method; it is unused for reporting and should be deleted or delegated.
4. **Expenditure still has no non-champion sealed predictions**, so its "best alternative" row is
   empty. One `b_ml` run over the sealed window would fill it.
5. **`backend/forecast_runs/**` is gitignored**, so artifact corrections do not appear in any diff
   and cannot be reviewed as one. Worth deciding whether reporting artifacts should be tracked.

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_sealed_window_report.py -q  # 14 passed
./backend/.venv/bin/python -m pytest backend/tests/test_ops_baseline.py -q          # 16 passed
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q                # 925 passed, 4 skipped
```
