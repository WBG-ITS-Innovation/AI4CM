# P1: the LIVE window, and champions stated as fixed

**Date:** 2026-08-13 · **Branch:** `model/excellence` · parent `33b582c`
**Root suite:** 761 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**New tests:** 35 (`test_live_window.py`) · **TEST reads this session: 0** — the holdout was not
unlocked (`experiments/test_access.log` holds 2 historical entries from 2026-08-05, recorded
retrospectively in phase 7; none from today)

---

## 1. The exact prompt given

> Continue on model/excellence in the AI4CM repo. Implement P1 from the diagnostic — the two things
> blocking retrain-and-compare on client data. Decisions are made; implement them, don't re-litigate.
>
> 1. FOURTH WINDOW. New client rows after 2025-08-06 currently land inside the sealed TEST window
> (evaluation_windows.py:295), so scoring or comparing over them needs AI4CM_ALLOW_TEST_READ=1, which
> would end the clean final read. Do NOT unlock TEST. Add a fourth "LIVE" window: scored against
> arrived actuals, never selected on, never used to choose a model, recipe or hyperparameter. Make
> that constraint structural rather than documented — a selection path that reaches LIVE data should
> raise, and a test should prove it does. State in the artifact which window each number came from.
>
> 2. CHAMPION RESELECTION. There is none: recipes.json is hand-maintained, nothing writes recipes,
> new data changes fitted parameters but never the model, features or transform. Do not build
> reselection. Instead make the current behaviour explicit and visible: every run states that
> champions are fixed and only refitted, and surfaces the registry's own drift caveat — Revenues'
> ratio advantage is +1.30% at 13.5% drift vs +24.14% at 84.1%, so the champion may be wrong for a
> new period with nothing flagging it. Write that as a field the Agent can read, not only prose. If a
> cheap drift indicator on the new data is possible without a selection run, propose it but do not
> build it.
>
> Then exercise it end to end: extend the canonical file past 2025-08-06 with synthetic rows, refit,
> publish, and score against those rows through the LIVE window. Paste the real output. Confirm
> nothing selected on LIVE data.
>
> Session logging: docs/sessions/2026-08-13-live-window-and-fixed-champions.md, full narrative
> verbatim plus prompt, plan, real output, verdict, outstanding. Both suites green, commit the md with
> the code, walk me through the diff.

---

## 2. Plan

1. Close TEST at the extent it actually holds and make everything after it LIVE. The root cause is
   `end=None`, not the boundary date.
2. Enforce "never selected on" with a raising guard wired into the path that crowns a champion, plus
   a refusal at `rolling_origin_folds`. Prove the guard is not overridable by the TEST flag.
3. Write the split and the champion policy into `SUMMARY.json`, and the window into every scorecard
   row.
4. Derive the champion policy's caveats from `recipes.json` rather than retyping them.
5. Exercise end to end on a file extended into LIVE, and confirm nothing selected on it.

---

## 3. Commands run and their real output

### The split, after the change

```
windows: ['train', 'dev', 'test', 'live']
TEST : 2025-01-01 .. 2025-08-06
LIVE : 2025-08-07 .. None
   2023-06-01 -> train
   2024-06-01 -> dev
   2025-06-01 -> test
   2025-08-06 -> test
   2025-08-07 -> live
   2026-01-01 -> live
eval_start_for(score): 2025-08-07

assert_selection_free raises:
   pretend_selector: refusing to select on report-only data. Rows fall in live (n=1, from
   2025-09-01). Selection is permitted only on ['dev', 'train']: TEST is a sealed holdout with one
   logged final read, and LIVE is data that arrived after sealing — it is scored against actuals and
   never used to choose a model, recipe, hyperparameter or threshold.
```

### The champion policy, read from the registry

```
reselection : none | on_new_data: refit_only
statement   : Champions are fixed. New data changes the fitted parameters of the recipe already
              chosen for each target; it never changes which model, which features
caveats     : 1
  target    : Revenues | applies_to: ratio
  finding   : The ratio transform's advantage is DRIFT-DEPENDENT, not constant.
  evidence  : Across five TRAIN-internal rolling-origin windows the advantage over raw tracks the
              level drift of the evaluation window almost perfectly (correlation +0.987): +1.30% at
              13.5% drif...
recipes_fixed: [('Revenues', 'LightGBM_L1', 'ratio'), ('Expenditure', 'LightGBM_L1', 'raw'),
                ('State budget balance', 'HistGBDT_L1', 'raw')]
```

### End to end: extend into LIVE, refit, publish, score

```
canonical edge 2025-08-06 -> window test
extended edge  2025-08-20  -> file now spans ['dev', 'live', 'test', 'train']
rows in LIVE: 10

--- refit + publish from the moved edge ---
issue 2025-08-20  origin 2025-08-20  origin_value 23,484,271.39
target_dates ['2025-08-21', '2025-08-22', '2025-08-25', '2025-08-26', '2025-08-27']
windows of target dates: ['live']

--- score the 2025-08-06 issue against LIVE actuals ---
scored 15 / pending 0
target                scored_in_window
Expenditure           live                5
Revenues              live                5
State budget balance  live                5
  Expenditure            realized_mae      59,472,556  skill   47.25%  hit 0.8
  Revenues               realized_mae      30,950,704  skill  -13.56%  hit 0.6
  State budget balance   realized_mae     172,900,589  skill  -13.93%  hit 0.4
AI4CM_ALLOW_TEST_READ set? False

--- would a selection on this data be refused? ---
  REFUSED: end_to_end_check: refusing to select on report-only data. Rows fall in live (n=10, from
  2025-08-07), test (n=218, from 2025-01-01)
```

### Suites

```
$ ./backend/.venv/bin/python -m pytest -q          # repo root
ROOT EXIT=0
761 passed, 3 skipped in 278.98s (0:04:38)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
110 passed in 8.48s
```

---

## 4. Full narrative response, verbatim

Both P1 items are in, and the holdout was not touched: **no TEST read occurred in this session** and
`AI4CM_ALLOW_TEST_READ` was never set during any run above. (`experiments/test_access.log` carries 2
entries, both from 2026-08-05 and logged retrospectively in phase 7 — not from this work.) Root **761 passed, 3 skipped**, frontend
**110 passed**.

### 1. The fourth window

The root cause was not the boundary date, it was `end=None`. `TEST` was open-ended, so "everything
from 2025-01-01 onward" swallowed every row a client would ever add. Closing TEST at the extent it
actually holds fixes it in one line, and the rest follows:

```
TEST  2025-01-01 .. 2025-08-06   sealed holdout, one logged final read, never used to choose
LIVE  2025-08-07 ..              arrived after sealing: scored, never used to choose
```

The boundary is a constant, not something derived from whatever file is on disk. That was deliberate:
a derived boundary would move every time a client loaded data, and a boundary that moves is not a
seal. `2025-08-06` is a fact about what was sealed.

**The change reclassifies nothing.** The canonical file ends 2025-08-06, so LIVE is empty until a
client loads rows, and every existing date keeps the window it already had. That is the property most
worth guarding, because a split change that silently moved an existing number would be far worse than
the problem it solved — so `test_the_change_reclassifies_nothing_that_exists_today` asserts LIVE has
zero rows in the canonical file and that the file's last date *is* `TEST_END`. If the file grows, that
test fails and forces someone to notice that the LIVE figures are now real.

**The structural half.** "Never selected on" had to be enforceable, because the TEST discipline was a
docstring before Phase 2 and the module's own comment block admits it: *"it stated the discipline and
nothing imported it, so nothing was enforced"*. So:

`assert_selection_free(dates, context)` raises `SelectionOnReportOnlyDataError` if any date falls
outside `SELECTABLE_WINDOWS` — `{train, dev}`. It names the window, the count and the earliest
offending date, because a refusal a reader cannot act on is only half a guard.

The guard is wired into **the path that crowns a champion**: `b_ml_pipeline`, immediately before
`select_best_model`, against `pred_long["target_date"]`. That is where selection actually happens with
dates in hand. A test asserts the guard appears *before* the selection call in the source, not merely
somewhere in the file. It raises rather than trimming — a quiet trim would change what was measured
without saying so.

`rolling_origin_folds(window="live")` refuses outright, because folds are a search structure and
there is no legitimate reason to carve one out of LIVE.

Three properties I made sure of, because they are what distinguish this from decoration:

- **The LIVE guard is not overridable by the TEST flag.** `test_the_live_guard_is_not_overridable_by_the_test_access_flag`
  sets `AI4CM_ALLOW_TEST_READ=1` and still requires the raise. TEST has a release procedure because a
  holdout exists to be spent once; LIVE has none, because there is nothing to release. Sharing the
  flag would have quietly made LIVE spendable too.
- **The two refusals are different exceptions**, asserted non-subclassing in both directions. A LIVE
  violation reported as a holdout-access problem would send someone to set the wrong flag.
- **Reading LIVE is ungated.** Scoring arrived actuals is the purpose of the window, not a holdout
  consultation, so `restrict(s, "live")` just works while `restrict(s, "test")` still raises.

**Which window each number came from.** `SUMMARY.json` gains a `windows` block — the four definitions
with boundaries and purposes, `selectable` vs `report_only`, and `data_spans_windows`, which is the
practically useful one: it says which windows the input file actually covers. `forecasts/scorecard.csv`
gains `scored_in_window` per row, because a realized figure from LIVE and one from the sealed holdout
are different claims and the row should say which it is rather than leaving a consumer to infer it
from a date. Both degrade with a stated reason rather than a bare omission, per the contract's §0
pattern.

### 2. Champions are fixed, and now say so

I did not build reselection. `registry.champion_policy()` states the current behaviour as data:
`reselection: "none"`, `on_new_data: "refit_only"`, plus a `statement` meant to be rendered verbatim —
*"Champions are fixed. New data changes the fitted parameters of the recipe already chosen for each
target; it never changes which model, which features or which target transform is used."*

The reason this needed writing at all is that the absence was invisible. A client loads data, a run
completes, numbers move — and the reasonable inference is that the model was re-chosen for the new
period. It was not.

`why` explains that reselecting on client data would mean selecting on LIVE, which the guard above now
forbids; the two halves of P1 are the same decision seen from two directions. `reselection_requires`
names what it would actually take: a TRAIN/DEV pass, and a hand-edited `recipes.json`, because nothing
writes it.

**The caveats are read from `recipes.json`, not retyped.** `champion_policy()` walks the recipes and
lifts each `scaling_caveat` — `finding`, `evidence`, `how_to_quote`, `study` — so the Revenues drift
figures live in exactly one place. A test asserts field-by-field equality with the registry, and
another asserts that **no Python file anywhere contains the string `24.14`**, so the evidence cannot be
duplicated into code where the two copies would drift apart. That felt worth a test rather than a
convention, since duplicating a caveat is precisely how a caveat ends up disagreeing with itself.

So the risk is now legible: Revenues' `ratio` advantage is `+1.30%` at 13.5% level drift and `+24.14%`
at 84.1%, DEV had 81.7%, and the champion was chosen under that. For a flatter period the transform's
edge is small and the champion may be the wrong choice — with nothing in a completed run flagging it.
The Agent can read `caveats[].how_to_quote` and say so.

### A cheap drift indicator — proposed, not built

You asked for a proposal only. The registry's own evidence gives the shape: the `ratio` advantage
tracks *level drift of the evaluation window* at correlation +0.987, so drift is already the known
predictor and it needs no model.

Compute, on LIVE rows only, the same quantity the WS4 study used:
`drift = (median(level, last 63 business days) / median(level, first 63 business days of the window) - 1)`,
per target. Compare it to the drift of the window the champion was selected on (DEV, 81.7%). Emit
three fields — `drift_pct`, `selection_window_drift_pct`, and a band (`similar` / `flatter` /
`steeper`) — and have the Agent quote the caveat when the band is not `similar`.

Two properties make it safe: it is a **descriptive statistic of the target series**, not a comparison
of models, so it selects nothing and does not need `assert_selection_free`; and it needs no refit, so
it is free. What it cannot do is tell you the champion is wrong — only that the condition under which
its advantage was measured no longer holds, which is exactly the caveat's own claim.

### The end-to-end run

Extended the canonical file to 2025-08-20 with synthetic rows — 10 of them in LIVE — then refit,
published and scored.

Publishing from the moved edge behaved: origin `2025-08-20`, `origin_value` 23,484,271.39, and all
five target dates in LIVE (`2025-08-21` … `2025-08-27`, correctly skipping the weekend). Scoring the
existing `2025-08-06` issue against the new actuals scored **15 of 15**, every row labelled
`scored_in_window: live`, with **`AI4CM_ALLOW_TEST_READ` unset** — which is the whole point: a client
can now see forecast-vs-actual without spending the holdout. And `assert_selection_free` on the
extended file refuses, naming both offending windows (`live n=10`, `test n=218`).

One honest note on those realized figures. Revenues reads −13.56% and the stock target −13.93% against
the ruler, on **synthetic** actuals drawn from a trailing mean and standard deviation. That draw has no
day-of-month structure, so it is close to the worst case for a model whose entire value is calendar
timing — the numbers say nothing about real accuracy and should not be quoted as if they did. What they
demonstrate is that the path runs, labels its window, and needs no gate.

---

## 5. Verdict

**Both P1 items implemented as decided, and the holdout is intact — no TEST read this session.**

The fourth window exists, TEST is closed at its real extent, and the change is provably inert on
current data. "Never selected on" is structural: a raising guard on the champion-selection path, a
refusal at the fold builder, a distinct exception, and — the property that matters most — not
overridable by the flag that releases TEST. Reading LIVE for scoring stays ungated, because that is
what the window is for.

Champions are unchanged and now stated: `champion_policy()` is machine-readable, its caveats are read
from `recipes.json` rather than retyped, and a test forbids the drift evidence from being duplicated
into code. A cheap drift indicator is proposed with its formula and its safety argument, and not built.

The artifact says where numbers came from: `windows` in `SUMMARY.json`, `scored_in_window` on every
scorecard row.

End to end on a file extended into LIVE: publish from the moved edge, 15 of 15 scored as `live`, no
gate, and selection refused on the same data.

---

## 6. Outstanding

* **The proposed drift indicator is not built** (§4). It is the one piece that would turn the champion
  caveat from a warning a reader must apply into a per-run signal.
* **Only B_ML's selection path is guarded.** `assert_selection_free` is wired into the champion
  crowning in `b_ml_pipeline`; E_QUANTILE's best-model choice and A_STAT's do not call it. They select
  within a single run on the configured eval window rather than across it, so the exposure is smaller —
  but "smaller" is not "guarded", and the guard is one line.
* **`scripts/ws2_tune.py` and `ws7_cqr.py` hardcode `2024-01-01`/`2023-12-31` bounds** rather than
  importing the windows. They are selection paths, so they should use `restrict` and the guard.
* **The realized figures above are synthetic** and near a worst case for a calendar-driven model. No
  real post-2025-08-06 data exists, so `forecasts/scorecard.csv` remains header-only.
* **`window_for` is now four-valued**, and any consumer branching on three names will read `live` as
  unknown. The contract documents it; nothing enforces that consumers handle it.
* Unchanged from earlier records: MASE is logged and not gated (Revenues 0.758 withheld, the published
  stock target 1.578); `vs_ruler`'s threshold is `> 0%`; the 1.50 sentinel threshold is uncalibrated;
  a data gap still becomes a training observation of `0.0` at the modelling layer;
  `conditional_coverage_gate` is not wired into `quantile_quality_gate`; `check_feature_leakage` is
  weak; the four `is_stock` implementations diverge (latent); `a_stat_models_pipeline.py` is a second
  unreferenced A_STAT implementation; the committed 2026-08-04 a_stat leaderboard fails the contract
  (pre-existing); `⚡ Persistence (baseline)` uses decoration in a join key; `skill_pct` and `horizon`
  are strings; ops P0, Phase-1 cleanup, registry approval workflow, single TEST read.
