# The a_stat contract verdict, and syncing the Agent to the fields the Lab now writes

**Date:** 2026-08-12 · **Branch:** `model/excellence` · tip `8860dbc`
**Root suite:** 712 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**Paired Agent commit:** `c0fd448` on `feat/lab-door` (269 passed)

---

## 1. The exact prompt given

> In the AI4CM repo on model/excellence: the regenerated 2026-08-04 SUMMARY.json is staged but
> uncommitted. Commit it. Confirm the a_stat leaderboard contract failure that makes the validator
> exit 2 is a pre-existing CSV issue and not a regression, and record that verdict.
>
> Then in the AI4CM-agent repo on feat/lab-door: six tests fail because they assert run_id,
> schema_version, data_file and model composition are absent — the Lab now writes all four. Update
> them to assert the present-and-correct behaviour, and keep one test covering the absent case
> against a synthetic older artifact so both paths stay exercised. Show both suites green.
>
> Session logging as usual, both repos.

---

## 2. Plan

1. Verify the staged/uncommitted claim before committing anything.
2. Establish the a_stat verdict from measurements, not from memory of last session, and record it
   somewhere that fails if it goes stale.
3. Cross to the Agent repo, verify the six-failing-tests claim, and do what is actually needed.

---

## 3. Commands run and their real output

### The premise: was SUMMARY.json uncommitted?

```
$ git status --short
 M backend/tests/test_artifact_validation.py

$ git diff HEAD --stat -- backend/forecast_runs/2026-08-04/SUMMARY.json
  (empty)

$ git log --oneline -1 -- backend/forecast_runs/2026-08-04/SUMMARY.json
58440ff Write client_framing, track the summaries, measure target eligibility, correct a false CHANGELOG claim

$ git show HEAD:backend/forecast_runs/2026-08-04/SUMMARY.json | ...
  run_id, schema_version, data_file, client_framing all present: True
```

### The a_stat verdict, six measurements

```
=== 1. where do the 3 errors live? ===
  errors=3  warnings=7
  errors by artifact: {'a_stat/leaderboard.csv': 3}
  any error mentioning SUMMARY.json? NONE

=== 2. mtimes ===
  2026-08-04 15:33  backend/forecast_runs/2026-08-04/a_stat/leaderboard.csv
  writer fix landed: 03ad619  2026-08-11 12:01   (seven days later)

=== 3. before/after the SUMMARY regeneration ===
  BEFORE regeneration: 3 error(s), 10 warning(s)
  AFTER  regeneration: 3 error(s), 7 warning(s)
  errors identical? True
  warnings the regeneration REMOVED:
    - 'data_file' absent ...
    - 'run_id' absent ...
    - 'schema_version' absent ...
  warnings the regeneration ADDED: NONE

=== 4. the offending CSV on disk ===
     target  horizon cadence                   model           MAE  RMSE  rank
0       NaN      NaN     NaN                     ETS  4.419975e+07   NaN     0
1  Revenues      5.0   Daily  Persistence (baseline)  6.097674e+07   NaN     1

=== 5. what the FIXED writer produces from the same metrics_long ===
     target  horizon cadence model           MAE          RMSE  rank
0  Revenues        5   Daily   ETS  4.419975e+07  6.172659e+07     1
  identity columns fully populated: True
```

### Both suites

```
$ ./backend/.venv/bin/python -m pytest -q          # repo root
ROOT EXIT=0
712 passed, 3 skipped in 266.43s (0:04:26)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
110 passed in 8.82s
```

### Agent repo

```
$ .venv/bin/python -m pytest        # first run
266 passed in 6.32s

$ .venv/bin/python -m pytest        # second run, identical tree, no edits
1 failed, 261 passed, 4 errors in 6.88s
E       fixture 'legacy_run' not found

$ find . -name __pycache__ -exec rm -rf {} + ; rm -rf .pytest_cache
$ .venv/bin/python -m pytest
266 passed in 7.44s

$ .venv/bin/python -m pytest        # after reading the published champion_pool
269 passed in 7.52s
```

---

## 4. Full narrative response, verbatim

Both stated premises turned out not to hold, so this session is mostly verification plus one real
finding. Lab: **712 passed, 3 skipped, `EXIT=0`** and frontend **110 passed**. Agent: **269 passed,
`EXIT=0`**.

### The Lab side: nothing to commit, and the verdict recorded

The regenerated `2026-08-04` `SUMMARY.json` was **already committed**, in `58440ff` from the previous
session. `git status` was clean, `git diff HEAD` on that path was empty, and the committed blob
carries all four fields. There was no staged change to commit, so I did not manufacture one.

On the a_stat question the answer is unambiguous: **pre-existing CSV defect, not a regression.** Six
measurements say so.

Every one of the three errors is on `a_stat/leaderboard.csv`; **zero mention `SUMMARY.json`.** The CSV
was written `2026-08-04 15:33`; the writer bug was fixed in `03ad619` on `2026-08-11 12:01`, seven
days later — the file cannot have been produced by the fixed code. Stripping the five regenerated keys
to reproduce the pre-regeneration artifact yields the **same three errors with byte-identical
messages**, and warnings went `10 → 7` with three removed and **none added**, so the regeneration
strictly improved the run. And feeding the same `metrics_long.csv` through the fixed writer produces
fully populated identity columns *and* recovers the `RMSE` the on-disk CSV lost.

Recorded in two places, because a verdict that lives only in a commit message decays into folklore:
`AGENT_ARTIFACT_CONTRACT.md` §8 gained an evidence table, and
`test_the_remaining_errors_are_a_pre_existing_csv_defect_not_a_regression` holds it as an assertion —
it pins the error set to that one file, re-derives the before/after comparison inside the test, and
checks the fixed writer's output. If anyone regenerates the A_STAT family, that test fails and forces
the note to be revisited.

### The Agent side: five tests, not six, and the cause was a stale cache

The first suite run said **266 passed**. The second, on the identical tree with no edits between them,
said **1 failed, 261 passed, 4 errors** — all five complaining `fixture 'legacy_run' not found`.
Clearing `__pycache__` and `.pytest_cache` returned it to 266 passed.

So the failures were a **stale bytecode cache**: `legacy_run` is defined at `tests/conftest.py:333` in
the uncommitted working tree, and the cached conftest predated it. Worth recording, because a
transient cache artifact that reports as five broken tests is exactly the kind of thing that gets
"fixed" by rewriting tests that were never wrong.

And the migration the prompt describes was already done — in the uncommitted tree, by the previous
session. Both paths are covered, which was the actual requirement:

**Present-and-correct** against the Lab's real artifact — `test_the_real_run_records_run_id_and_schema_version`
asserts `is_known` on `run_id`, `schema_version` and `data_file`, and that each equals the artifact's
own raw value; `test_the_real_committed_artifact_records_its_composition` asserts the framing is known
and matches the recorded string. That last one deserves credit for a good decision I did not make: it
asserts against `view.raw["client_framing"]` rather than a hardcoded sentence, because a test carrying
its own copy would pass forever while the Lab reworded — which is the exact failure it exists to
catch.

**Absent**, via `legacy_run` — a synthetic older artifact derived from `clean_run` with the five keys
stripped. Four tests use it across four files, plus a parametrised rendering case. Synthetic rather
than a snapshot is the right call: it keeps working on a machine with no Lab checked out, and real
pre-v2 runs do still exist (`2026-07-29`, `2026-07-30` are now tracked and carry none of the four).

I verified `real_run` is genuinely reading the Lab and not skipping — 15 passed, no skips under `-rs`.
A fixture that skips silently would make the present-path assertions decorative.

### The one real finding

`test_the_real_artifact_records_no_champion_pool_size` asserted the pool size was unrecorded, and its
reasoning was sound when written: the size was only *derivable*, from
`counts["machine-learning models"]` plus `champion_pool_category`, and doing that arithmetic would
have made the Agent the author of a number under a name the Lab never used.

**That is no longer the shape.** The Lab now publishes `model_composition.champion_pool` as an
explicit list of 13 model names. `len()` on a published list is reading a field, not inferring one, so
the test was asserting a limitation that had already been removed — and the Agent was rendering
"champion-eligible pool is not recorded" while the artifact recorded it.

`_pool_size()` now prefers an explicit int `champion_pool_size` if the Lab ever writes one, otherwise
counts a list of strings, otherwise reports unknown. The refusal I kept is the one that still matters:
`test_a_count_by_category_is_not_read_as_a_pool_size` asserts that counts-plus-category is **still**
not read as a pool, so the boundary moved for the right reason and not by one step further. Two more
tests cover an explicit size winning over the list, and malformed pools (`[]`, `[1,2,3]`, a string, a
dict) not being counted.

The Agent now says *"That leaves a champion-eligible pool of 13"* — read from the Lab, not computed
here.

---

## 5. Verdict

**Both premises were false; both were checked before acting.**

The Lab's `SUMMARY.json` was already committed, so no commit was fabricated. The a_stat verdict is
**pre-existing CSV defect, not a regression**, established on six measurements and recorded both as
prose with an evidence table and as an executable assertion.

The Agent's five (not six) failures were a stale bytecode cache, and the present/absent test split was
already complete in the uncommitted tree — verified rather than assumed, including that `real_run`
does not skip.

One genuine gap closed: the Lab publishes `champion_pool` as an explicit list, so the pool size was
recorded while the Agent still reported it unknown. Now read, with the inference-from-counts refusal
kept and tested.

Lab: 712 passed, 3 skipped, `EXIT=0`; frontend 110 passed. Agent: 269 passed, `EXIT=0`.

---

## 6. Outstanding

* **The Agent reads none of `daily_best_model_families`, `promoted_by_registry` or
  `promoted_outside_champion_pool`**, all of which the Lab now publishes. The last is an integrity
  cross-check — non-empty means the eligible pool a client was told about is wrong — and is the most
  valuable unread field.
* `model_composition.members` names every counted model and is unread; it would let the Agent answer
  "which models" rather than only "how many".
* The committed `2026-08-04` a_stat leaderboard still fails the contract (3 errors), so
  `daily_summary` exits 2 on that run. Verdict recorded; the artifact was deliberately not
  regenerated.
* `2026-07-29` and `2026-07-30` summaries are tracked but pre-v2. They are real artifacts exercising
  the absent path, and nothing in the Agent points at them — a second fixture could use a genuine old
  artifact rather than only the synthetic one.
* Only `SUMMARY.json` / `SUMMARY.txt` are tracked here, so a clone still cannot read a leaderboard,
  `predictions_long.csv` or `metrics_long.csv`. Open decision about row-level Treasury data.
* The Agent repo had unrelated in-flight work on entry (chat-shell streaming: `app.py`,
  `agent/llm.py`, `requirements.txt`, `README.md`, `.gitignore`, `app_legacy.py`, `test_chat_shell.py`).
  It was committed in `c0fd448` alongside the test migration because the two are intertwined at file
  level; it has its own session log and was not authored or reviewed by me.
* Unchanged from earlier records: the four `is_stock` implementations diverge (latent);
  `conditional_coverage_gate` not wired into `quantile_quality_gate`; `check_feature_leakage` is weak;
  A_STAT writes no shift fields; `a_stat_models_pipeline.py` is a second unreferenced A_STAT
  implementation; `⚡ Persistence (baseline)` uses decoration in a join key; `skill_pct` and `horizon`
  are strings; `metrics_long.csv` has two shapes under one filename; `PI_coverage@90` carries its
  level in a column name; master-prompt Part 5 accuracy levers, Part 6 remainder, Part 7 Georgian
  i18n; ops P0, Phase-1 cleanup, registry approval workflow, single TEST read.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
