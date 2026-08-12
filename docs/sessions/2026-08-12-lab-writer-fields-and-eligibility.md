# Lab writer fields, target eligibility, and a CHANGELOG correction

**Date:** 2026-08-12 · **Branch:** `model/excellence` · parent `e8c26a2`
**Root suite:** 711 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**New tests:** 33 (24 eligibility/capability + 9 composition round-trip)

---

## 1. The exact prompt given

> Continue on model/excellence in the AI4CM repo. Three writer gaps the Agent audit surfaced, plus
> one new function.
>
> 1. client_framing() exists in backend/model_reference.py but daily_summary.py never writes it, so
> no committed SUMMARY.json carries it and the Agent now correctly reports "composition not
> recorded" on every run. Write it. Add it to AGENT_ARTIFACT_CONTRACT.md §1.
>
> 2. data_file: confirm the writer now emits it, then produce and commit a fresh run that actually
> carries both data_file and client_framing. The writer being fixed while no committed artifact
> shows it is the gap — the Agent is blocked on the artifact, not the code.
>
> 3. Extend forecast_modes.targets_available into a real eligibility function: numeric after
> dropping {date, is_weekend, is_holiday}, parseable/unique/sortable date index, and sufficient
> non-null history — at least DEFAULT_MIN_TRAIN (1008) plus the horizon plus one complete test fold.
> It currently returns all 41 columns with no length check. Return the reason a column is ineligible
> alongside the verdict, so a consumer can quote it rather than invent one.
>
> 4. Confirm and document: E_QUANTILE has no stock-target path (per the CHANGELOG), so State budget
> balance cannot be forecast by that family. Make that machine-readable, not a note.
>
> Session logging: docs/sessions/2026-08-12-lab-writer-fields-and-eligibility.md, full narrative
> verbatim plus prompt, plan, real output, verdict, outstanding. Run both suites, show passing
> output, commit the md with the code.

---

## 2. Plan

1. **Item 1** — write `client_framing()` plus the `composition()` counts it came from into the
   `SUMMARY.json` payload, with a companion `client_framing_unavailable_reason` so absence is
   explained rather than bare (the contract §0 pattern). Document in §1.
2. **Item 2** — confirm `data_file` is emitted, then get it into a real artifact. Investigate why no
   artifact carries it before assuming the fix is just "run something".
3. **Item 3** — `target_eligibility()` measuring numeric / date-index / history, with a
   machine-readable `code` **and** quotable prose per rejection; threshold imported from
   `evaluation_windows` so it cannot drift. Exercise the rejection paths on synthetic input.
4. **Item 4** — verify the claim before encoding it.

---

## 3. Commands run and their real output

### Both suites

```
$ ./backend/.venv/bin/python -m pytest -q        # repo root
ROOT EXIT=0
711 passed, 3 skipped in 264.21s (0:04:24)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
110 passed in 7.46s
```

### Item 1 — the writer now emits the composition

```
data_file      : master_daily_clean_treasury.csv
client_framing : 13 machine-learning models, 5 deep-learning models and 4 statistical models
                 compete on each target; prediction intervals come from 3 quantile methods;
                 3 further entries are reference baselines, not competitors.
composition    : {"machine-learning models": 13, "deep-learning models": 5,
                  "statistical models": 4, "quantile methods": 3, "reference baselines": 3}
champion_pool  : 13 models
promoted_outside_champion_pool: []
unavailable_reason present: False
```

### Item 2 — why no artifact carried it

```
=== do committed artifacts carry data_file / client_framing? ===
2026-07-29   data_file=None  client_framing=False schema_version=None
2026-07-30   data_file=None  client_framing=False schema_version=None
2026-08-04   data_file=None  client_framing=False schema_version=None

=== is backend/forecast_runs tracked? ===
.gitignore:21:backend/forecast_runs/   backend/forecast_runs/2026-08-04/SUMMARY.json
tracked run files:        0
```

First attempt at the `.gitignore` negation, which silently did nothing:

```
  IGNORED : backend/forecast_runs/2026-08-04/SUMMARY.json
  IGNORED : backend/forecast_runs/2026-08-04/SUMMARY.txt
  IGNORED : backend/forecast_runs/2026-08-04/b_ml/predictions_long.csv
```

After switching to excluding the directory's **contents**:

```
  TRACKED : backend/forecast_runs/2026-08-04/SUMMARY.json
  TRACKED : backend/forecast_runs/2026-08-04/SUMMARY.txt
  IGNORED : backend/forecast_runs/2026-08-04/BACKTEST_REPORT.md
  IGNORED : backend/forecast_runs/2026-08-04/b_ml/predictions_long.csv
  IGNORED : backend/forecast_runs/2026-08-04/b_ml/leaderboard.csv
  IGNORED : backend/forecast_runs/2026-08-04/e_quantile/run.json
```

Regenerating the real run's summary (exit 2 = the a_stat leaderboard still fails the contract; the
writer is fixed, the artifact predates it):

```
=== what the regenerated summary gained ===
  run_id           before='None'  after='2026-08-04'
  schema_version   before='None'  after='2'
  data_file        before='None'  after='master_daily_clean_treasury.csv'
  client_framing   before='None'  after='13 machine-learning models, 5 deep-learning models and 4 sta'
  model_composition present: before=False after=True

=== the family figures must be UNCHANGED (same artifacts, re-summarised) ===
  all family verdicts identical: True
  overall identical: True

=== new top-level keys ===
   ['client_framing', 'data_file', 'model_composition', 'run_id', 'schema_version']
```

### Item 3 — the threshold, and proof the check is not vacuous

```
min_history_rows(5) = 1139  (1008 + 5 + 126)
date index: {'ok': True, 'reason': None, 'n': 3867, 'n_unique': 3867,
             'n_unparseable': 0, 'monotonic': True}

candidates=41  eligible=41  ineligible=0

the five SHORTEST columns in the real file:
   3,867 rows  Compensation of employees
   3,867 rows  Decrease in financial assets
   ...
margin above the threshold for the shortest column: +2,728 rows

=== so the check must be exercised on synthetic input ===
  ShortSeries    eligible=False code=insufficient_history
      'ShortSeries' has 300 usable observations; 1,139 are needed at horizon 5 — ...
  TextColumn     eligible=False code=not_numeric
      'TextColumn' is str and none of its values parse as numbers, ...
  EmptyColumn    eligible=False code=all_null
      'EmptyColumn' has no values at all in this file, ...
```

### Item 4 — the CHANGELOG claim, measured

Design construction, showing the delta-modelling markers:

```
State budget balance  (is_stock=True)
  design rows=2738  features=12
  mean|target| =        138,573,911   <- delta if stock
  mean|origin| =      1,167,451,928   <- the level
  y_lag_0 present (delta-modelling marker): True

Revenues  (is_stock=False)
  design rows=2738  features=11
  mean|target| =         65,377,513
  mean|origin| =         65,276,488
  y_lag_0 present (delta-modelling marker): False
```

End to end on the stock target, DEV 2024 pinned so TEST is untouched:

```
[quantile] GBQuantile: n=262, P50 MAE=168,565,455.33, Persistence MAE=242,653,025.46,
           Skill=30.53%, Coverage(P10–P90)=65.6%, gate=FAIL
  best_model               GBQuantile
  skill_pct                30.532308423932825
  coverage_p10_p90         0.6564885496183206
  coverage_nominal         0.8
  quality_gate_passed      False
  run_status               FAILED_QUALITY
```

Logged selection runs on the stock target for that family:

```
E_QUANTILE runs on the stock target: 0
```

The four `is_stock` implementations:

```
name                         B_ML   C_DL    E_Q  A_STAT  agree?
State budget balance         True   True   True    True  True
balance                      True   True   True    True  True
t0                           True   True   True   False  False
net                         False  False  False    True  False
stock                       False  False  False    True  False
Revenues                    False  False  False   False  True

41-col file; names matching any stock alias:
    'State budget balance'
```

### The pinned-findings test firing as designed

```
>       assert "run_id" in msgs, "the committed SUMMARY.json predates schema_version 2"
E       AssertionError: the committed SUMMARY.json predates schema_version 2
```

### Diff stat

```
 .gitignore                                      |  21 +-
 CHANGELOG.md                                    |   9 +-
 backend/family_capabilities.py                  | 198 +++++++++++++++
 backend/forecast_modes.py                       | 187 ++++++++++++++-
 backend/forecast_runs/*/SUMMARY.{json,txt}      | 591 +++++++++++++++++++++++++
 backend/tests/test_artifact_contract.py         |  15 ++
 backend/tests/test_artifact_validation.py       |  14 +-
 backend/tests/test_daily_summary_composition.py | 139 +++++++++++
 backend/tests/test_target_eligibility.py        | 305 ++++++++++++++++++++++++
 docs/AGENT_ARTIFACT_CONTRACT.md                 | 128 +++++++++-
 scripts/daily_summary.py                        |  51 ++++
 17 files changed, 1635 insertions(+), 23 deletions(-)
```

---

## 4. Full narrative response, verbatim

All four done, but **item 4's premise turned out to be false**, and item 2's real blocker was
bigger than the field it was about. Both suites green: root **711 passed, 3 skipped, `EXIT=0`**;
frontend **110 passed, `EXIT=0`**.

### Item 1 — `client_framing` written, with the counts behind it

`_composition_fields()` in `daily_summary.py` now writes `client_framing` plus a `model_composition`
block carrying the numbers the sentence was derived from. Both, deliberately: a consumer that only
has the sentence cannot recompute anything, and one that only has counts will assemble its own
sentence and get the framing wrong.

`model_composition` carries the two different things "champion" means here, because conflating them
is how a true sentence becomes a wrong one. `champion_pool` is the 13 machine-learning models a
registry recipe may promote — what an official published forecast is selected from.
`daily_best_model_families` is all four families, because `daily_summary.py` writes a per-family
`best_model` for each and **the Agent ranks across those**, not across `champion_pool`. It also
carries `promoted_outside_champion_pool` as an integrity cross-check: non-empty would mean the
eligible pool a client was told about is wrong. Currently empty.

One thing I got wrong and the test caught: I asserted `promoted_by_registry` had three entries, one
per recipe. It has **two** — Revenues and Expenditure both promote `LightGBM_L1`. It is a set of
distinct models, not a recipe count, and the corrected test says so.

Absence follows the pattern I documented in the contract §0 rather than being bare. The composition
is derived from `model_pool()`, which imports every family's registry, and that import can fail on a
machine without a modelling library. A summary must not fail because the *catalogue* could not be
read — the run itself is unaffected — so the helper degrades to
`client_framing_unavailable_reason` naming the exception. Two tests cover it, including one that
forces the import to fail for real.

### Item 2 — the artifact gap, and the bigger one underneath it

`data_file` was already emitted, at `daily_summary.py:536`. Confirmed.

Then I checked why no artifact carried it and found something larger: **`backend/forecast_runs/` is
gitignored, and zero run files were tracked.** Not "the artifacts are stale" — there were no run
artifacts in the repository at all. Every artifact I have been validating across items 5 and 6 is an
untracked local file. A cloned repo has nothing for the Agent to read, so `data_file` and
`client_framing` had nowhere to appear even once written. That is the actual blocker, and it is not
about either field.

`SUMMARY.json` and `SUMMARY.txt` are now tracked; every row-level artifact stays ignored. The
distinction is what the files contain: the summaries are **aggregate only** — model names, MAEs,
skill percentages, gate verdicts, and the data file's *name*. `predictions_long.csv` is row-level
Treasury values, and `.gitignore` has said "never commit" about those since it was written. That is
the same line the published-forecast directory already sits on, and it is tracked for the same
reason. I checked the tracked files for absolute paths and home directories before committing: none.

A detail worth recording because it fails silently. My first `.gitignore` attempt was

```
backend/forecast_runs/
!backend/forecast_runs/*/SUMMARY.json
```

and it did nothing — all three paths still reported `IGNORED`. Git never descends into an excluded
**directory**, so a negation for a file inside one cannot take effect. The fix is to exclude the
contents (`backend/forecast_runs/**`) and re-include the directory levels, and I put that reasoning
in the file so the next person does not retry the version that looks right.

For the artifact itself I regenerated `2026-08-04`'s `SUMMARY.json` with the current writer over its
own existing family artifacts. That is a re-derivation, not a new run, and I verified it as one:
**every family verdict and the entire `overall` block are identical**, and the only change is five
new top-level keys. Nothing was re-fitted and no number moved. The command exits 2, because the
a_stat leaderboard in that run still fails the contract — the writer was fixed last session and the
CSV predates it — and I left that as it is rather than re-running a family to tidy a committed
artifact.

The pinned-findings test from item 6 failed at this point, which is exactly what I built it for: it
asserted `run_id` was *missing* from the real run, and the regeneration fixed that. Updated to
assert the opposite, with a note recording that it has already done its job once.

### Item 3 — eligibility, with the reason attached

`target_eligibility()` returns a verdict per column with a machine-readable `code`
(`not_numeric`, `all_null`, `insufficient_history`, `unusable_date_index`) **and** prose a consumer
may quote verbatim. That pairing is the point: a consumer told only "no" has to invent an
explanation, and an invented explanation shown to a treasury is worse than a blank. The prose carries
the numbers too, so nobody has to assemble them — *"'ShortSeries' has 300 usable observations; 1,139
are needed at horizon 5 — about four years to train on, a 5-day gap so no training answer falls
inside the evaluation, and one complete six-month block left to score against."*

The threshold is imported, not chosen: `DEFAULT_MIN_TRAIN (1008) + horizon + DEFAULT_EVAL_BLOCK
(126)` = 1,139 at h=5. If the fold sizing changes this moves with it, and a test asserts the identity
rather than the literal.

The date-index check is **file-level** and reported separately, because an unparseable or duplicated
date makes every column ineligible — a horizon counted in index positions is ambiguous when one date
appears twice. A consumer should say "the file is unusable", not list 41 identical column failures.
Sortable is the requirement, not already-sorted: a reversed index passes, and `monotonic` is reported
as information rather than as a rejection.

`targets_available()` now returns eligible targets by default instead of all 41, with
`include_ineligible=True` for a page that greys out the rejects and shows the reason beside each —
more useful than one that hides them.

The honest caveat: **all 41 columns are eligible today.** The canonical file is fully dense — every
column has 3,867 usable rows, +2,728 above the threshold — so the length check rejects nothing. A
check that never fires looks like it works, so every rejection path is exercised against synthetic
fixtures, and one test asserts the file *is* dense so that if it stops being dense that becomes a
deliberate discovery rather than a surprise.

### Item 4 — the premise is false, so I did not encode it

You asked me to confirm and document that E_QUANTILE has no stock-target path. **It has one.** I
checked before encoding it, and I am not making a false constraint machine-readable.

The evidence, in order. `is_stock("State budget balance")` is `True` in that family.
`_build_features` adds `y_lag_0`, which is stock-only and is the delta-modelling marker. The target
it builds is a **change**: mean |138,573,911| against a level of mean |1,167,451,928|, an order of
magnitude apart. And `run_pipeline` reconstructs the level from `origin_value + delta`. Then I ran
the family end to end on the stock target against DEV 2024 with `eval_start`/`eval_end` pinned so
TEST stayed sealed: **262 predictions, P50 MAE 168,565,455 against a persistence MAE of 242,653,025 —
skill 30.53%, coverage 65.6%, gate FAILED on coverage.** It produced a complete artifact set.

The CHANGELOG entry sits under a Phase-0/1 **"Known gaps"** heading, and its sibling bullet says
*"E_QUANTILE is still on a calendar-day index"* — fixed long ago by `to_business_index`. The whole
block describes a state the code has left. Nothing failed when it went stale, which is precisely why
it was still there.

This matters beyond pedantry, because the two statements point at different work. *"No stock path"*
invites building one. *"The path exists, has no recipe, has never been logged as a selection run,
and its one DEV run fails the coverage gate at 65.6% against a required [70%, 90%]"* invites fixing
the calibration. The second is true.

So `backend/family_capabilities.py` records what is actually the case, machine-readable as asked —
just not the claim in the prompt. `family_supports_target(family, target)` returns `supported`,
`stock_method` (`delta` or `level`), and separately `publishable`, because supported is not approved:
E_QUANTILE comes back `supported=True, publishable=False, code="supported_but_not_published"`. Only
B_ML is publishable for the stock target, and a test asserts that it is the only one. The CHANGELOG
entry is struck through with the measurement and a pointer to the module, because a capability
recorded only in prose drifts.

Two findings fell out of writing it.

**A_STAT forecasts the level directly and never models a delta** — its `_is_stock` drives only
resampling (`ffill` vs `fillna(0)`, `last` vs `sum`). Legitimate for a statistical model on a level
series, but it means a stock forecast from A_STAT is not comparable to one from B_ML without saying
which was modelled how. Recorded as `stock_method="level"` against `"delta"` for the other three.

**There are four `is_stock` implementations and they disagree.** E_QUANTILE's docstring claims
byte-identity with B_ML and C_DL, which holds for those three: `{"state budget balance", "balance",
"t0"}`. `run_a_stat._is_stock` uses `{"state budget balance", "balance", "net", "stock"}`. For a
column named `t0`, three families would model a delta and A_STAT a level. **None of the disputed
names is a column in the canonical file**, so it is latent — and latent is exactly when nobody
thinks to check. `stock_alias_divergence()` returns it as data, and a test asserts both that they
still disagree and that no disputed name has become a real column.

---

## 5. Verdict

**All four items delivered; item 4 delivered as a correction rather than as specified.**

1. `client_framing` and `model_composition` are written, with a companion reason for absence, and
   documented in contract §1. Nine round-trip tests run the writer as a subprocess and open the file
   it wrote — the only kind of test that could have caught this class of gap.
2. `data_file` confirmed emitted, and the real blocker found: `backend/forecast_runs/` was entirely
   gitignored, so no run artifact existed in a clone. The two summary files are now tracked
   (aggregate-only) and `2026-08-04`'s summary is regenerated and committed carrying both fields,
   verified to be a re-derivation with no verdict changed.
3. `target_eligibility()` measures numeric / date-index / history with a derived threshold and pairs
   every rejection with a quotable reason. All 41 columns pass today, so the rejection paths are
   proved on synthetic input.
4. **The CHANGELOG claim is false.** E_QUANTILE has a full stock path, measured end to end at
   skill 30.53% with a failed coverage gate. Encoded what is true — capability, method, and
   publishability as three separate facts — plus two findings the check surfaced.

711 passed / 3 skipped `EXIT=0`; frontend 110 passed `EXIT=0`.

---

## 6. Outstanding

* **`2026-07-29` and `2026-07-30` summaries are now tracked but predate schema v2** — no `run_id`,
  `data_file` or `client_framing`. Readable with warnings; regenerating them would need their family
  artifacts, which are ignored and may not survive.
* **The committed `2026-08-04` a_stat leaderboard still fails the contract** (3 errors), so
  `daily_summary` exits 2 on that run. Writer fixed; artifact predates it; deliberately not
  regenerated.
* **Only `SUMMARY.json` / `SUMMARY.txt` are tracked**, so a clone still cannot read a leaderboard,
  `predictions_long.csv` or `metrics_long.csv`. If the Agent needs those from a clone, that is a
  separate decision about row-level Treasury data — I did not make it.
* **The `is_stock` divergence is latent, not fixed.** Four implementations, three alias sets. It
  should become one shared function; that touches all four families.
* **E_QUANTILE's stock run fails the coverage gate (65.6% vs [70%, 90%])** and has no logged
  selection run. If a stock forecast from that family is wanted, calibration is the work — CQR is
  the obvious lever and already exists.
* `targets_available()` changed meaning (eligible-only by default). Only one caller existed, but any
  future consumer expecting all 41 must pass `include_ineligible=True`.
* Unchanged from earlier records: `conditional_coverage_gate` not wired into
  `quantile_quality_gate`; `check_feature_leakage` is weak; A_STAT writes no shift fields so it has
  no effective persistence-mimicry check at h=5; `a_stat_models_pipeline.py` is a second unreferenced
  A_STAT implementation; `⚡ Persistence (baseline)` uses decoration in a join key; `skill_pct` and
  `horizon` are strings and `best_model` embeds a number in prose; `metrics_long.csv` has two shapes
  under one filename; `PI_coverage@90` carries its level in a column name; master-prompt Part 5
  accuracy levers, Part 6 remainder, Part 7 Georgian i18n; ops P0, Phase-1 cleanup, registry
  approval workflow, single TEST read.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
