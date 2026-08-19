# The holdout read, recorded; and the verdict history, rendered

**Date:** 2026-08-14 · **Branch:** `model/excellence` · parent `a08a045`
**Root suite:** 809 passed, 4 skipped, `EXIT=0` · **Frontend suite:** 119 passed, `EXIT=0`
**New tests:** 6 holdout-read + 9 render

---

## 1. The exact prompt given

> Continue on model/excellence. Two things from the p2-followups outstanding list.
>
> 1. A_STAT reads 2025 rows without going through require_test_access at all — a pre-existing hole in
> the holdout discipline that the guard work surfaced. Close it properly: reading TEST rows must go
> through the sanctioned, logged release path like every other family, without adding a selection
> guard where A_STAT makes no selection. Confirm test_access.log records the read.
>
> 2. The Forecast page doesn't display the reconciliation, so a reader of the 2025-08-06 issue sees
> only the issue-time verdict. Render verdict-at-issue vs verdict-today with the one actionable
> sentence naming only the gates that drove the change. Same for any published issue.
>
> Same session logging. Both suites green.

---

## 2. Plan

1. Establish how A_STAT actually reaches TEST, and check the premise that other families go through
   the logged path — before copying a pattern that may not exist.
2. Render the reconciliation on the Forecast page, and smoke-test that it actually appears rather
   than trusting a green suite.

---

## 3. Commands run and their real output

### The premise, checked first

```
=== B_ML runner window ===
(no eval_start / eval_end)
=== C_DL runner ===
84:        # with TG_PARAM_OVERRIDES {"eval_start": null} to fold over all years.
85:        eval_start=ov.get("eval_start", _TEST_START),
=== does C_DL gate the read when eval_start is TEST_START? ===
backend/run_c_dl_univariate.py:11:from evaluation_windows import TEST_START as _TEST_START
backend/run_c_dl_univariate.py:85:        eval_start=ov.get("eval_start", _TEST_START),
```

No `require_test_access` anywhere in C_DL. So "like every other family" did not hold.

### The two purposes

```
selection (holdout closed): REFUSED
report (holdout closed): ALLOWED and logged
log lines: 1
```

### A_STAT, run ordinarily

```
[2026-08-13 19:25:49] Persistence MAE=81475575.75, Model MAE=81475575.75, Skill=0.00%
[2026-08-13 19:25:49] DONE.
```

`experiments/test_access.log`, all three entries:

```
entries: 3
  [1] 2026-08-05T02:42:28  purpose=(none - pre-split)
       Item 1f one-ruler verification evaluated the INCUMBENT models on the 2025 window ...
  [2] 2026-08-05T02:56:10  purpose=(none - pre-split)
       ['kind', 'timestamp', 'what', 'cause']
  [3] 2026-08-13T23:25:48  purpose=report
       A_STAT reporting evaluation for 'Revenues' at h=5 covers 156 holdout target date(s)
       from 2025-01-01 to 2025-08-06
```

### C_DL, the identical hole

```
guard inside build_yearly_folds: True
folds built: 1
=== did it log? ===
  purpose: report | caller: c_dl_pipeline.yearly_folds
  C_DL reporting evaluation covers 156 holdout target date(s) from 2025-01-01 to 2025-08-06
```

### The rendered page

```
Verdict history section rendered: True
changed-count banner: True
  withheld_as_forecast                               True
  publishable                                        True
  signal was re-thresholded from 1.5 to 1.15         True
  accuracy_vs_naive                                  True
  immutable                                          True

expander labels: ['Issue 2026-08-13 — unchanged', 'Issue 2025-08-06 — 3 of 3 verdict(s) changed']
```

### Suites

```
$ ./backend/.venv/bin/python -m pytest -q          # repo root
809 passed, 4 skipped in 261.48s
  the 4 skips are the frontend AppTest modules under the backend interpreter:
  SKIPPED frontend/tests/test_verdict_history_render.py:26: streamlit is installed in frontend/.venv only
  (+ test_pages_smoke, test_tier1_correctness, test_visual_tokens)

$ cd frontend && ./.venv/bin/python -m pytest -q
119 passed in 8.91s
119 passed in 7.27s     # re-run, for the reason in §4
```

---

## 4. Full narrative response, verbatim

Both done. Root **809 passed, 4 skipped**; frontend **119 passed**.

### 1. The holdout read, recorded rather than refused

I checked the premise before copying a pattern, and it did not hold. The brief said A_STAT should go
through the logged path "like every other family" — but **no family does.** `run_c_dl_univariate`
defaults `eval_start` to `TEST_START`, i.e. it reports on the holdout by design, and calls
`require_test_access` nowhere. B_ML's runner sets no window at all. The hole was not A_STAT-specific;
it was general, and A_STAT is simply where the guard work happened to surface it.

So the fix had to come from the discipline rather than from a sibling. The module's own opening
paragraph draws exactly the line that was missing:

> TEST is run at the end of a milestone to report what would have happened, and is never used to
> choose anything. Each time TEST is consulted **to make a choice**, it stops being a clean holdout,
> so the number of such consultations should stay at zero.

Two different acts, and the code recognised only one. `require_test_access` treated every read as a
potential choice and raised — which is why reporting reads had simply routed around it and gone
unrecorded. It now takes a `purpose`:

- `PURPOSE_SELECTION` (default) — could inform a choice. Raises unless the holdout is deliberately
  released, announces itself on stderr, and logs. Behaviour unchanged; the count should stay at zero.
- `PURPOSE_REPORT` — evaluating over the holdout to state what would have happened. **Never raises,
  always logs.** This is what the holdout is for; what was wrong was that it happened silently.

Both write to the same log with the purpose recorded, so "how many times did we look at the holdout,
and why" now has a factual answer with the two acts distinguishable.

A_STAT computes the target dates its folds will actually evaluate, filters to the TEST window, and
records the read naming the count and the range: *"A_STAT reporting evaluation for 'Revenues' at h=5
covers 156 holdout target date(s) from 2025-01-01 to 2025-08-06."* Confirmed in
`experiments/test_access.log` as entry 3, `purpose=report`.

**No selection guard was added**, as the brief required, and the source says why: A_STAT runs one
model per invocation via `TG_MODEL_FILTER` and never chooses between models — its leaderboard ranks
that model against the persistence baseline, which is a report. A test asserts
`assert_selection_free` does *not* appear in either module, so a future well-meaning addition fails.

I closed C_DL's identical hole in the same way rather than leaving a known-identical gap open one line
from the fix. Its `build_yearly_folds` now records the same way, identifying itself separately in the
log.

One honest limitation: the two historical log entries predate the purpose field, so they read as
`(none - pre-split)`. I left them as they are — they are a record of what happened, and back-filling a
field onto them would be inventing detail.

### 2. The verdict history, rendered

The Forecast page rendered only the *current* registry verdict and never showed published issues at
all, so a reader of the 2025-08-06 issue saw the verdict that issue was published under with nothing
saying it no longer holds.

`reconcile_verdicts()` turned out to be importable directly from the frontend venv — it needs only
json, pandas and the registry, not the modelling stack — so no backend dispatch was necessary. It
loads alongside everything else, wrapped so a failure there degrades to an empty list rather than
taking the page down; a test asserts that wrapping exists, because a comparison should never be a
dependency.

The new **Verdict history** section leads with the count (*"3 published verdict(s) would differ
today"*) and the sentence that matters most: the forecast numbers in those issues have not changed,
only the verdict attached to them, and the published files are left exactly as issued. Then one
expander per published issue — expanded by default when something changed, collapsed when not —
showing `withheld_as_forecast → publishable` per target with the one actionable sentence beneath it.
That sentence names only the driving gate, so Revenues reads *"signal was re-thresholded from 1.5 to
1.15, which flipped its outcome from False to True on an unchanged measurement of 1.2255"* rather than
a list of the three added gates that pass. Every expander closes with the note that the published
issue is immutable and this is a comparison, not a correction.

"Same for any published issue" is covered: it iterates every issue with a `gates.json`, newest first,
and the re-issue appears as *"Issue 2026-08-13 — unchanged"* — which is itself informative, because it
shows the reconciliation is not just a list of problems.

While there I fixed something P2 left behind. The page rendered **both** withheld verdicts as
*"WITHHELD as a forecast — shown as a guide to the typical level"*. That is right for
`withheld_as_forecast`, where the numbers remain the best central-tendency estimate and only the event
claim is withheld. It is wrong for `withheld`, which now means a documented trivial benchmark is *more
accurate* — those numbers are not a guide to anything, and offering them as one would invite a worse
decision than showing nothing. The hard branch now reads **"WITHHELD — do not use these numbers"**,
and a test asserts it is checked *before* the softer branch, since otherwise it would never fire.

I smoke-tested the rendered page rather than trusting a green suite: a reconciliation that exists in a
module and appears on no page is invisible to everyone, and the suite passing does not distinguish the
two. Nine tests now pin the rendering — both verdicts present, the driving-gate sentence, the
"numbers have not changed" reassurance, the immutability note, an expander per issue, and the
verdict-branch ordering.

### One flake, reported rather than smoothed over

The first full frontend run after adding the tests reported
`FAILED tests/test_pages_smoke.py::test_page_renders_with_no_artifacts[03_Models.py]` — a
**pre-existing** test, not one of mine. It did not reproduce: passing in isolation, passing when paired
with my new module, and passing on two consecutive full runs (119 passed each). Streamlit `AppTest`
state leaking between page renders has bitten this suite before and was previously fixed with an
autouse cache-clearing fixture, so a residual ordering sensitivity is the likely cause. I am recording
it as an unreproduced flake rather than claiming the suite was clean throughout, and it is in the
outstanding list.

---

## 5. Verdict

**Both items done, with one premise corrected.**

The holdout hole was general, not A_STAT-specific: C_DL reports on TEST by default and gated nothing
either. Rather than copy a pattern that did not exist, the fix implements the distinction the module's
own discipline already stated — a *selection* read raises unless released, a *reporting* read is
logged and permitted. A_STAT and C_DL both record their reads; `test_access.log` confirms A_STAT's 156
holdout dates. No selection guard was added to either, and a test forbids one.

The Forecast page now shows verdict-at-issue against verdict-today for every published issue, with
the driving-gate sentence and the immutability note, verified by rendering the page rather than by
inference. The page also stops describing a hard-`withheld` model as a guide to the typical level.

---

## 6. Outstanding

* **`test_page_renders_with_no_artifacts[03_Models.py]` failed once and did not reproduce** in four
  subsequent attempts. Likely residual `AppTest` state leakage. Worth a deliberate look before it is
  seen again and mistaken for a real regression.
* **The two pre-split log entries carry no `purpose`.** Left as-is; a consumer reading the log must
  treat a missing purpose as "unknown", not as either value.
* **B_ML's runner still sets no evaluation window**, so what window it reports on depends entirely on
  its caller. It did not surface here because nothing in the suite exercises an unbounded B_ML report
  run, but it is the same class of ambiguity.
* **Reporting reads now accumulate in the log on every run.** That is the intent — the count is
  supposed to be answerable — but nothing yet distinguishes "156 dates read for the daily report" from
  a genuine milestone read at a glance beyond the `purpose` field.
* Unchanged from earlier records: an unbounded E_QUANTILE run refuses rather than selecting on TEST
  (the default itself was never bounded); MASE's `season=5` denominator is gate-binding and has not had
  the scrutiny the sentinel threshold received; `forecast_integrity.MIN_SIGNAL_RATIO` is still 1.5
  while publication decides at 1.15 — two numbers for one concept; a data gap still becomes a training
  observation of `0.0` at the modelling layer; `conditional_coverage_gate` is not wired into
  `quantile_quality_gate`; `check_feature_leakage` is weak; `a_stat_models_pipeline.py` is a second
  unreferenced A_STAT implementation; the committed 2026-08-04 a_stat leaderboard fails the contract
  (pre-existing); `⚡ Persistence (baseline)` uses decoration in a join key; ops P0, Phase-1 cleanup,
  registry approval workflow, single TEST read.
