# Phase 15 — Timeboxed session (30 min): forecast modes

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `37a085f` (32 ahead of `origin/main` @ `863f967`)
**Root suite:** 485 → **495 passed, 3 skipped**, `EXIT=0` · **Frontend suite:** 99 passed
**Log:** 153 rows, integrity ok · **TEST (2025) gated reads: 0** · **PR #26** open, reused

---

## Step 0, as reported

| Check | Result |
|---|---|
| Working tree | clean |
| Branch tip | `85a1bde`, 31 ahead, 0 unpushed |
| Root suite | 485 passed, 3 skipped, `EXIT=0` |
| Log integrity | `{'n_rows': 153, 'ok': True, 'problems': []}` |
| TEST gated reads | 0 |
| PR | #26 open → reused |
| Registry | 3 of 41 targets have a champion recipe |

---

## ITEM 1 — COMPLETE (`37a085f`)

The stated success condition for the session: *"if only this lands, the session succeeded."*

### The separation is at the boundary, not in the UI

`backend/forecast_modes.py`. An `ExploratoryResult` is a **different type with no publish path**,
and `publish_official()` refuses on the *value* of `is_official`, which is a property rather than a
mutable flag. A page cannot leak an exploratory run into the published record by forgetting to set
something.

`backend/tests/test_forecast_modes.py` — **10 tests, and the item's completion criterion.** Three
are on the boundary itself:

* passing an `ExploratoryResult` to `publish_official` raises **and** no directory is created
  before it refuses;
* `is_official` cannot be assigned on an exploratory result (`AttributeError`);
* a duck-typed impostor declaring `mode="official"` while `is_official` is `False` is also refused —
  the check is on the value, not the declared type.

Why the boundary matters concretely: `forecasts/published/` is the input to
`forecasts/scorecard.csv`, which is the evidence base for every accuracy claim the project makes
about its own published output. An ungated hand-picked model entering there would contaminate it.

### OFFICIAL mode

All 41 targets selectable; **the model is not.** `official_run()` takes only
`(target, data_path, horizon)` — a test asserts there is no `model` parameter, because a
hand-picked model under the official label would make the label meaningless when the champion was
chosen on recorded evidence.

Refuses rather than degrading, on two grounds:

| Refusal | Reason given to the analyst |
|---|---|
| No registry recipe (`NoRecipe`) | Substituting another target's recipe would attach five folds of evidence to a model it was never measured on. Exploratory mode is offered instead |
| Horizon ≠ 5 (`NotOfficial`) | The ruler, recipe selection and every gate were measured at h=5. An official forecast at h=1 would carry credentials never earned at h=1 |

Publishing goes through the existing path, so every safeguard is inherited unchanged:
`assert_forward_only()`, no truth column, one model per horizon, Georgian holidays skipped, gates
carried by `recipe_id` rather than recomputed on dates that have no truth. `next_issue_date()`
suffixes (`-r2`, `-r3`) rather than overwriting — retention is immutable. Approver renders as
**none** on every published line.

### EXPLORATORY mode

Any model, any target, any horizon. The banner states *"not gated, not published"* and names why:
the target has no champion, the chosen model is not its champion, or the horizon is unvalidated.

**One real gap found while verifying:** the model pool came back **empty**, because it lives behind
the ML libraries, which are installed in the *backend* interpreter and not the one running
Streamlit. Rather than hard-code a list that would go stale the moment a model is added or a
library removed, the page asks the backend interpreter and reports honestly if it cannot reach it.
Verified: **13 models offered**.

### Monthly — recorded, not implemented

`reports/ui_content_backlog.md` gains the design question. Monthly is not simply a longer horizon:
it is a **sum over a variable number of business days** rather than a value on one day, so it needs
its own target definition, benchmark and gate. Worth investigating on its own terms — sums cancel
daily noise, so a monthly aggregate may well be *more* predictable than the daily flows — but
bolting it onto a harness measured at h=5 would produce a number carrying no credentials.

---

## ITEM 2 — NOT STARTED

The Models reference needs plain-language descriptions for 13 models, **every hyperparameter read
from the code rather than memory**, per-target measured performance from `experiments/log.csv` with
a `run_id` on every figure, and the recipe record including approver. That is more than the
remaining box, and the brief is explicit that finishing two items cleanly beats starting five. Not
begun rather than half-built.

**Items 3 and 4 — not started**, same reason.

---

## Hard-stop checks

| Required | Result |
|---|---|
| Everything committed | ✅ working tree clean |
| Suite green | ✅ 495 passed, 3 skipped, `EXIT=0` |
| Re-issue forward forecast if any recipe changed | **Not required** — `git diff` on `registry/recipes.json` is empty for this session; no recipe changed |
| Refresh `PROGRESS_SINCE_LAST_REVIEW.md` if any published number moved | **Not required** — `git diff` on `forecasts/` is empty; the single published issue `2025-08-06` is unchanged, still 0 scored / 15 pending |
| Session record | this file |
| Push, update PR | ✅ |

No number displayed anywhere changed this session: item 1 added a way to *generate* forecasts and
did not alter any existing figure, recipe or artifact.

---

## Resume point

**ITEM 2 — the Models tab as a real reference.** Four parts, in the brief's order:

1. Plain-language description per model, marked as a general description rather than a measured
   claim.
2. Every hyperparameter the pipeline actually sets — **read from the code**, e.g. by introspecting
   `available_models()` with `get_params()`, not transcribed. Name, what it controls, current
   value, sensible tuning range.
3. Measured performance from `experiments/log.csv` **only**: per target, TRAIN and DEV MAE, skill,
   MASE, sentinel, per-tercile coverage where logged, each with its `run_id`, and *not reported*
   where a metric was never logged. Note the known gap: **point-model runs log no coverage**, so
   per-tercile will be *not reported* for the champions.
4. Gate status, and for champions the recipe record exactly as stored **including approver
   (currently `none` on all three)**.

Then items 3 (ranking view on `recommender.py`) and 4 (persist the fitted estimator).

### Artifact fields still needed — unchanged from phase 14

Seven, listed in `reports/phase14_session_record.md`. The worst remains `c_dl_pipeline.py:958`
writing `alignment_ok: True` as a **literal with no check behind it** — not a missing field but a
false one. For item 2 specifically, the blocking gap is **per-tercile coverage on point-model
runs**, which is why item 2 part 3 will legitimately show *not reported* for the champions.
