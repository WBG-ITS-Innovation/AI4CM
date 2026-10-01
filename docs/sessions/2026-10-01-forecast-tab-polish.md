# Forecast tab polish: three fixes applied, two decisions presented

**Date:** 2026-10-01
**Branch:** `fix/forecast-tab-polish`, cut from `main` at `236198a` (the merge of PR #29).
**Source:** `2026-09-30-inference-horizon-map.md`, which lives on `fix/scoring-trust-batch` and is
not on this branch; its findings are quoted here by section where they are relied on.
**Rules kept:** code and tests only in this working tree; no pipeline, scorer or artifact
regeneration ran here. The one forward run for evidence ran in the disposable clone on
synthetic data. Every fix was written failing-first and walked through before the next began.

Five items were asked for. Items 3, 4 and 5 are done. Items 1 and 2 are decisions the brief
reserved for the user; both are presented at the end of this record with their trade-offs and
a recommendation, and nothing was written for either.

---

## 3. `run_a_stat.py` falls back to the validated horizon; every runner default is pinned

**What was wrong.** Each family runner reads `TG_HORIZON` with a literal fallback. Seven fell
back to 5, the horizon every ruler, recipe and gate is measured at; `run_a_stat.py` fell back
to 6 (map §2.5, "the odd one out"). Harmless only because every caller sets the variable: a
runner started by hand without it would have evaluated at a horizon nothing is measured at,
under a leaderboard that looks like every other.

**The change.** `backend/run_a_stat.py`: `int(env["TG_HORIZON"] or 5)`, with a comment.

**The test**, `backend/tests/test_runner_horizon_defaults.py` (10), written first and failing on
`run_a_stat.py` alone. It reads every `backend/run_*.py` that mentions `TG_HORIZON`, and the
daily script, as text, extracts each fallback literal with a regex, and pins it to
`VALIDATED_HORIZON`. Reading source rather than importing is deliberate: the fallback lives
inside each runner's `main()`, so importing the module shows nothing. A regex that finds nothing
would pass by accident, so a coverage test requires all eight runners and the script to have
been found. The fallbacks pinned: `run_b_ml_univariate.py:19`, `run_b_ml_multivariate.py:19`,
`run_c_dl_univariate.py:51`, `run_c_dl_multivariate.py:51`,
`run_e_quantile_daily_univariate.py:16`, `run_e_quantile_daily_multivariate.py:17`,
`run_foundation.py:67`, `run_a_stat.py` (the corrected line), and
`scripts/run_daily_forecast.sh:53`.

## 4. The Documentation page's C_DL defaults table states the code's defaults

**What was wrong.** `frontend/pages/09_Documentation.py`, `defaults_c()`, rendered as "Defaults
by cadence" in the C tab, said:

| cadence | lookback | horizon | epochs | batch_size |
|---|---:|---:|---:|---:|
| daily | 90 | **14** | 50 | 64 |
| weekly | 104 | 8 | 80 | 32 |
| monthly | 60 | 12 | 100 | 16 |

The brief named the daily horizon of 14 (map §2.5). Measured while writing the test: not one of
those twelve numbers appears in the code. The true values, per the pipeline's own defaults
(`backend/c_dl_pipeline.py:63-65`, `:76-77`), which the runner's `ov.get(...)` fallbacks repeat
exactly (`backend/run_c_dl_univariate.py:67-71`):

| cadence | lookback | horizon | epochs | batch_size |
|---|---:|---:|---:|---:|
| daily | 64 | 5 | 30 | 128 |
| weekly | 52 | 5 | 30 | 128 |
| monthly | 36 | 5 | 30 | 128 |

The horizon is the runner's. It always replaces the pipeline's list with `[TG_HORIZON]` for
the active cadence (`run_c_dl_univariate.py:96-100`), and its fallback is the validated horizon
of five. The pipeline's own lists, `[1, 5, 20]` daily, `[1, 4, 12]` weekly and `[1, 3, 6]`
monthly (`c_dl_pipeline.py:99-104`), are reached only when no horizon is given at all; they are
now stated in a caption under the table rather than pretending to be a single default.

**Scope, stated plainly.** The brief asked for the daily horizon row. All four columns were
corrected, in all three rows, because the test that pins the horizon compares every cell of
the table to the source, and a table titled "defaults" that was wrong in every cell is one
defect, not twelve. The walkthrough quoted the numbers above before the edit was made.

**The test**, `backend/tests/test_documentation_c_dl_defaults.py` (6), failing-first on every
cell and on the missing caption. It reads both the page and `ConfigDL` as source with `ast`,
so it needs neither Streamlit nor torch and runs under either interpreter. A first draft
imported `c_dl_pipeline` for the defaults; that pulls in torch, and loading torch beside
LightGBM in one test process is where this machine's intermittent segmentation fault showed up
(see "Suites" below), so the parser replaced the import.

## 5. `test_window_touched` is measured from the holdout ledger

**What was wrong.** `forward_forecast.build_provenance` wrote `"test_window_touched": False`
as a literal (map §1.3, "One thing to read as it is"). The claim was true by construction,
since a forward run reads no truth, but nothing computed it, so a holdout read introduced by
any later change would have been published under a record still saying no, and the Forecast
page and the treasury report print "No, still sealed" from exactly that field
(`frontend/pages/07_Forecast.py:910`; `scripts/build_treasury_report.py:302`).

**The change.** `backend/forward_forecast.py`:

* `holdout_ledger_length()` counts the lines of `evaluation_windows.TEST_ACCESS_LOG`, the
  ledger every `require_test_access` call appends to, whatever its purpose
  (`backend/evaluation_windows.py:136-192`). It looks the path up at call time so a test can
  point the ledger at a file of its own; a missing ledger counts as zero.
* `build_provenance(data_path, champions, *, ledger_before)` takes the caller's reading, reads
  the ledger again, and sets `test_window_touched` to whether it grew. The two readings and
  their difference travel in a new `holdout_ledger` block, so the flag can be audited rather
  than believed. When the ledger grew, the note "no truth was read" is replaced by one that
  says how many entries appeared and that this needs explaining before the artifact is used.
* Both official callers take the "before" reading before any model is fit:
  `backend/run_forward_forecast.py` in `main`, and `backend/forecast_modes.py` in
  `official_run`, immediately before `pd.read_csv`.

**The tests**, `backend/tests/test_provenance_holdout_flag.py` (6), failing-first. They point the
ledger at a temporary file with two prior lines, so the real ledger is neither read nor
appended to. The planted read is one `require_test_access(..., purpose="report")`; the flag
flips to `True`, `reads_during_run` is 1, and the "no truth was read" note is gone. A source
scan asserts the literal is not back, and that both callers snapshot before `run_forward(`.
`test_forward_forecast.py`'s provenance test now supplies the reading, which is the one
pre-existing test this change touched.

**Clone evidence.** `backend/run_forward_forecast.py` without `--publish`, in the clone, on the
synthetic file through 2025-08-20:

```
ledger lines before: 2
[forward] data through 2025-08-20, 3881 rows
[forward] test_window_touched = False
ledger lines after : 2
holdout_ledger: {'path': '<clone>/experiments/test_access.log',
                 'lines_before': 2, 'lines_after': 2, 'reads_during_run': 0}
first note: Forward forecast: every target date is strictly beyond the last date in the data, so no truth was read ...
```

The two prior lines are the report reads A_STAT and C_DL made during the earlier daily run in
the clone; the forward run added none, and the flag says so from the count rather than from a
constant.

---

## 1. The stale reading tab: two candidate fixes, decision reserved

**The defect** (map §1.3, first of "Three facts"). The Forecast page's reading tab shows
`backend/forecast_runs/forward/latest` (`frontend/pages/07_Forecast.py:85-113` →
`backend/insights.py:347-363` → `forward_forecast.DEFAULT_OUT`). Only
`backend/run_forward_forecast.py` writes there. A page-launched official run writes nothing
without `--publish`, and with it stages per target under `forward/staging/<issue>--<target>`
and removes the staging directory on success (`backend/forecast_modes.py`, `publish_official`).
So after a page-launched run the reading tab still shows the artifact the runner last wrote,
dated 2025-08-06 on this machine.

Two facts shape both options. First, `forward/latest` holds every registry target from one
runner invocation, under one `generated_at_utc`, one data SHA and one `recipes` list. Second,
the page path publishes one target per click, into one issue directory per click, and a
published issue carries the same three files the forward directory does, renamed
(`backend/published_forecasts.py:444-448`): `forecast.csv`, `provenance.json`, `gates.json`.

**(a) `official_run` also refreshes `forward/latest`.** After a page-launched run, replace that
target's rows in `latest`'s `forward_forecast.csv`, its entry in the provenance `recipes` list
and its `gates.json` entry, and leave the other targets as they were.

* For: one loader, one artifact, no change to `insights.load_forward_artifacts`, the treasury
  report or the reading tab; the tab is current the moment the run finishes.
* Against: a merged `latest` has one `generated_at_utc`, one data SHA and one
  `holdout_ledger` block for rows produced at different times, possibly from different data if
  an upload happened between runs, so the provenance stops describing the whole file. It
  reintroduces writes to the shared forward directory from the page path, which is the exact
  shape of the 2026-08-15 defect (`forecast_modes.py`, the staging comment: "publishing one
  target overwrote the run holding all three"). The exploratory path must be kept from ever
  touching it, which is one more place to get wrong. And `latest` is git-ignored and
  machine-local, so the thing being kept current is not the record.
* Size: about half a day including a merge-by-target test and a provenance test.

**(b) The reading tab shows the newest of `forward/latest` and the published store, labelled
with its source and date.** A loader picks, per target, the newest artifact by
`generated_at_utc` across `forward/latest` and every `forecasts/published/<issue>/`, and the
tab prints where each target's numbers came from and when.

* For: nothing new is written anywhere; the published store is the immutable record and is
  exactly what a page-launched official run produces; the reader is told the source rather
  than left to infer it; `latest` stays what the runner made.
* Against: "newest" has to be defined per target, because a page issue holds one target
  while `latest` holds three, and a per-target pick is a merge done in memory, with the
  provenance shown per target rather than once. `insights.build_narrative_text` takes one
  forecasts frame and one provenance; it would receive a merged frame and a provenance per
  target, so it and the four-column header (`07_Forecast.py:498-518`, "Data through") need a
  per-target reading. The two paths' `gates.json` differ by one key (`approved_by`), which the
  reader must tolerate. Selection by `generated_at_utc` must not fall back to the directory
  name, since the runner's issue is dated by origin and the page's by wall clock.
* Size: about one day including the loader, the labels, and tests for the pick rule and the
  mixed-source render.

**Recommendation:** (b). It keeps the page from writing to a shared artifact, which the
project has already been burned by once, and it puts the source of every number on screen,
which is the honest form of "current". If (a) is chosen, the merge must also rewrite the
per-target provenance, or the artifact will claim a single vintage it does not have.

## 2. One source of truth for the validated horizon: two options, decision reserved

**The defect** (map §2.5 and §2.7). `VALIDATED_HORIZON = 5` is two independent literals,
`backend/forecast_modes.py:33` and `frontend/pages/07_Forecast.py:161`, and nothing compares
them. Changing one leaves the page's caption, slider default and warning at the other value.

One fact matters for both options: the page already puts `backend/` on its import path
(`07_Forecast.py:21-23`) and already imports a backend module at module level,
`forward_forecast` (`:27-29`), whose own imports are pandas, numpy and the calendar modules.
The venv split forbids the modelling stack, not backend modules as such.

**(a) A tiny shared constants module with no imports.** `backend/validated_horizon.py`
containing `VALIDATED_HORIZON = 5` and a docstring, nothing else. `forecast_modes.py` imports
it and re-exports the name so every existing import keeps working; the page imports it and
drops its literal.

* For: one literal, one edit to change it, and the guard added on the other branch
  (`official_run` comparing a recipe's `params.horizon` to the constant) keeps pointing at the
  one place. Zero risk to the venv split, since the module imports nothing.
* Against: a third module for one number. The project's stated preference is to read facts
  from the registry rather than type them (`frontend/backend_consts.py:18-24`), but the
  horizon is not a per-recipe fact: it is what the recipes are measured against, so it has to
  exist independently of them.
* Size: an hour including a test that the page has no literal left and that the two names
  resolve to one object.

**(b) A test that fails when the literals differ.** `frontend/tests/test_validated_horizon_agrees.py`
reading both files with a regex and asserting the two numbers match.

* For: no runtime change, no new module, and the page keeps importing nothing it does not
  already import.
* Against: still two literals, so a change is still two edits, and the test is a regex over
  source rather than a property of the code. It catches the divergence after the fact; it
  does not remove the way it happens.
* Size: half an hour.

A variant of (a) is to import the constant from `forecast_modes` directly, with no new
module. It works today, since that module's top-level imports are pandas and dataclasses
(`forecast_modes.py:24-30`), but the page's own comment says it deliberately avoids importing
the forecast machinery and dispatches by subprocess, and a future heavy import at the top of
`forecast_modes` would then take the page down with it. The no-import module in (a) cannot.

**Recommendation:** (a).

---

## Suites

| Suite | Baseline | After |
|---|---|---|
| Frontend, `frontend/.venv`, `frontend/tests` | 727 passed, 16 skipped | **727 passed, 16 skipped** |
| Backend, `backend/.venv`, both test paths | 1345 passed, 20 skipped | **1367 passed, 20 skipped** (1345 + 22), 9 minutes 35 seconds, no segmentation fault |

The frontend tally is unchanged because the three new test files live under `backend/tests`;
the `frontend/tests` invocation does not collect them. The Documentation test was also run
under the frontend interpreter directly and passes there (6 passed), since it needs no torch.

New tests, 22 in all:

* `backend/tests/test_runner_horizon_defaults.py` (10):
  `test_the_runners_with_a_default_are_the_ones_expected`,
  `test_every_horizon_default_is_the_validated_horizon` (parametrised over the eight runners
  and the daily script)
* `backend/tests/test_documentation_c_dl_defaults.py` (6):
  `test_the_runner_repeats_the_pipelines_defaults`,
  `test_the_daily_horizon_row_is_the_validated_horizon`,
  `test_every_cell_of_the_defaults_table_matches_the_code` (daily, weekly, monthly),
  `test_the_pipelines_own_horizon_lists_are_stated_beside_the_table`
* `backend/tests/test_provenance_holdout_flag.py` (6):
  `test_the_flag_is_false_when_the_ledger_did_not_grow`,
  `test_a_planted_holdout_read_flips_the_flag_to_true`,
  `test_the_delta_is_recorded_so_the_flag_can_be_audited`,
  `test_a_missing_ledger_counts_as_zero_lines`,
  `test_the_flag_is_not_a_literal_in_the_writer`,
  `test_both_official_callers_snapshot_the_ledger_before_the_fit`

One pre-existing test was updated to supply the ledger reading
(`test_forward_forecast.py::test_provenance_records_the_sealed_window_and_the_scaling_decision`).

**The intermittent segmentation fault, again.** While running a seven-file subset, the suite
crashed inside LightGBM's dataset construction immediately after a file that imported torch.
The same subset passed once the Documentation test stopped importing `c_dl_pipeline`. The
crash was first seen on the previous branch in a subset that imported no torch at all, so the
two-OpenMP-runtimes explanation fits one occurrence and not the other; it is recorded as
intermittent and environmental, and the full-suite result below is what counts.

## Commits on this branch

1. `run_a_stat.py falls back to the validated horizon; every runner default pinned`
2. `Documentation page: the C_DL defaults table states the code's defaults`
3. `test_window_touched is measured from the holdout ledger, not written as a literal`
4. this record

Items 1 and 2 await the user's choice; nothing was written for either.
