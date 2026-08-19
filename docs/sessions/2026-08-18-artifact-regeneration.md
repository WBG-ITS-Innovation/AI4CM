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

> ## ⚠️ Read this before citing anything below
>
> **1. `ws2_tune`'s DEV fold is scored against 4 holdout rows — TOP-PRIORITY, blocks client
> sharing.** It calls `assert_selection_free` on evaluation *origins* (all DEV) and then reads truth
> at `origin + H` — 4 of which land in the sealed window. So every champion credential in
> `registry/recipes.json` includes 4 holdout observations: **selection on holdout data**, the one
> thing the four-window split exists to prevent. Impact on DEV MAE is small (Revenues 0.82%,
> Expenditure 1.16%, stock 0.51%) and that **does not change what it is**. Fix in its own scoped
> session before the Task 3 numbers go to a client. Pinned by an inverted test (open item 1).
>
> **2. The champion recipes are not reproducible.** The script that produced their credentials is
> absent from the repository. Reconstruction differs by **5.1% / 8.1% / 19.5%** at n=250 against a
> logged n=262. Sealed-window figures **supersede** those credentials rather than extend them, and
> `dev_reconstruction()` reports the gap beside every figure (open item 2).
>
> **3. The Task 3 table is the client-facing set and supersedes the July deck.** Every earlier
> "N% better than the Treasury's current method" figure is withdrawn — those were measured against a
> zero baseline.

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

### Decision recorded: retraining, and what the fresh run is not

I proposed correcting the column in place — the bug was in the comparator, not the predictions, and
the metrics were exactly reproducible from the stored predictions (5/5 MAEs matching to 1e-6). Full
retraining was chosen instead, **reviewed and accepted**. The consequences are recorded here rather
than left to be inferred:

* **The 2026-08-18 C_DL run is a NEW run, not a corrected 2026-08-04.** Retraining produced
  different models, so its MAEs are not comparable to the older run's — e.g. MLP **6.88e7** against
  the old **4.72e7**. Neither number is wrong; they are different fits.
* **Only the ops column may be compared across the two.** The corrected ops figures replace the
  zero-baseline ones as *the* Ops comparison for C_DL; the MAE, RMSE and interval columns of the two
  runs are separate measurements.
* **2026-08-04 stays intact** as the record of what was computed on its own date. Only its wrong
  column was cleared, with an `OPS_SKILL_CLEARED.md` beside it stating the removed values and why.

So a reader comparing the two runs should compare *ops skill against the current method*, and not
read the MAE difference as an improvement or a regression.

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

## Task 3 — THE CLIENT-FACING NUMBERS

**These supersede the July deck.** Any earlier figure of the form "N% better than the Treasury's
current method" is withdrawn — those were computed against a baseline of zero (see the ops-baseline
session), and the corrected margins are roughly a quarter of what they implied.

**Sealed window (TEST 2025-01-01..2025-08-06), h=5, embargoed, `PURPOSE_REPORT` logged.**
Machine-readable copy: `reports/sealed_window_champion_vs_ops.csv`.

> **Sharing gate.** Open item 1 — `ws2_tune`'s DEV fold reads 4 holdout rows — **must be fixed
> before these numbers go to a client.** The sealed-window figures below are themselves clean
> (embargoed, origin-bounded, logged); what is not clean is the DEV credential that selected the
> champions in the first place. A client asking "how was this model chosen?" deserves an answer that
> does not include holdout data.

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

**How to read this honestly** — and these caveats travel with the table, not in a footnote:

* **Revenues is the strong case**: +32% over the current method, and the champion beats the best
  non-champion on the ops comparison (+32.0% vs +20.9%) despite a *higher* MAE — because ResidualRF
  is scored over 204 rows against the champion's 146, so the two MAEs are not directly comparable.
* **Expenditure is marginal**: **+2.79%**. Better than the current method, but by a margin that a
  different window could erase, and its verdict is `withheld` on accuracy anyway.
* **The stock target has no ops comparison at all**, and is `withheld`.
* `n=146` rather than 156: 4 rows lost to the embargo, the rest to incomplete features at the
  window edge.
* **These are a measurement of the champion *recipe*, not a continuation of its logged
  credentials** — which are not reproducible (open item 2). The reconstruction gap on DEV is 5.13%
  (Revenues), 8.12% (Expenditure), 19.45% (stock), and `dev_reconstruction()` reports it beside any
  sealed-window figure so the two are never conflated.

---

## Open items

### 1. TOP PRIORITY — `ws2_tune`'s DEV fold is scored against holdout rows

**Must be fixed in its own scoped session, before anything from these credentials is shared with a
client.** Magnitude does not soften what it is: **selection on holdout data.**

`ws2_tune.make_folds` calls `assert_selection_free` on the evaluation **origins** — all DEV — and
then reads truth at `origin + H`, 4 of which land in the sealed window (2025-01-01, 01-02, 01-03,
01-06). The guard passes because it is checking the wrong dates. Every champion credential in
`registry/recipes.json` therefore includes 4 holdout observations.

Measured impact on DEV MAE: Revenues **0.82%**, Expenditure **1.16%**, stock target **0.51%**.

**Pinned by** `backend/tests/test_sealed_window_report.py::test_the_tuners_dev_fold_is_scored_against_holdout_rows`.
The pin is inverted on purpose: it **passes while the leak exists** and **fails once it is fixed**,
with an assertion message instructing the fixer to delete it. A companion,
`::test_the_borrowed_selection_path_still_lacks_the_embargo`, pins the related 5-row training
embargo gap the same way. So neither can be quietly lost, and neither can be "fixed" without
someone noticing.

Why it was not fixed here: correcting a selection path changes what the tuner optimises and what
every registry credential means. That is not an artifact-regeneration change.

### 2. Known limitation — the champion credentials are not reproducible

The script that produced them is absent from the repository. Runs are stamped
`feature_names=['ws4:ratio']` / `['ws4:raw']`, `fold_scheme="DEV confirmation, single 2024 fold"`
(a string no current script emits), `per_fold: []`, one recorded feature name; there is no
`scripts/ws4_*.py`, and the surviving WS4 reports re-run tests rather than regenerating figures.

Reconstruction gap: Revenues **5.13%**, Expenditure **8.12%**, stock **19.45%**, at n=250 against a
logged n=262. Parameter mismatch is ruled out — `champion_estimator` asserts every recipe-declared
parameter against `available_models()` — so the gap is fold geometry plus this harness's embargo.

**Mitigation in place:** `sealed_window_report.dev_reconstruction()` recomputes the DEV figure and
returns it alongside the logged credential with the delta and an explanatory note, so no
sealed-window number can be presented as continuous with credentials it cannot reproduce.
`::test_it_states_the_gap_to_the_logged_credential` fails if that stops being true.

Either reconstruct the WS4 harness, or formally supersede the credentials with figures from this
harness. Recorded as a limitation, not a defect to be worked around.

### 3. DECIDED — reporting artifacts stay untracked

`backend/forecast_runs/**` **remains gitignored**. The tracked surface is deliberate and stays as
it is:

* the **scorecard** (`forecasts/scorecard.csv`) stays **tracked** — it is the durable record of
  scored claims, and its header is pinned to `SCORECARD_COLUMNS` by test;
* **vault retention** covers everything else, which is what `private_vault/` exists for;
* run artifacts stay out of version control, so artifact corrections do not appear in a diff.

The practical consequence, stated so it is not rediscovered: **this session's artifact work is not
reviewable as a diff.** The commit is source-only. Verification is by re-running the reproduction
commands, and by the `OPS_SKILL_CLEARED.md` notes left beside each corrected file.

### 4. Expenditure has no non-champion sealed predictions

Its "best alternative" row in the Task 3 table is empty because no other model has sealed-window
predictions for that target. **Deliberately not added this session.** One `b_ml` run over the sealed
window would fill it.

### 5. `b_ml_pipeline` keeps a divergent ops method

`shift(12).rolling(36)` within month-groups — a 12-**year** shift — with a flat spread. No zero bug,
but it is not the Treasury method. Unused for reporting; should be deleted or delegated like the
other three.

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_sealed_window_report.py -q  # 14 passed
./backend/.venv/bin/python -m pytest backend/tests/test_ops_baseline.py -q          # 16 passed
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q                # 925 passed, 4 skipped
```
