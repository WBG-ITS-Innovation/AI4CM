# Item 6 — the Agent artifact contract

**Date:** 2026-08-11 · **Branch:** `model/excellence` · parent `610e4da`
**Root suite:** 653 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**New tests:** 48 validator tests + 4 pool/registry tests
**Contract:** [`docs/AGENT_ARTIFACT_CONTRACT.md`](../AGENT_ARTIFACT_CONTRACT.md) ·
**Validator:** [`backend/artifact_validation.py`](../../backend/artifact_validation.py)

---

## 1. The exact prompt given

> Continue the session on model/excellence. Item 6: the Agent artifact contract.
>
> The AI4CM Agent (separate repo, feat/lab-door) reads SUMMARY.json, the leaderboard artifacts,
> predictions_long.csv, metrics_long.csv and the published forecast directory. Treat these as a
> published interface, not incidental output.
>
> Part 1 — document the contract as the pipeline actually produces it today: every field, its type,
> whether it is always present or conditionally present, and what a consumer should do when it is
> absent. Include the fields whose presence is now conditional by design:
> legacy_coverage_key_omitted, coverage_nominal, alignment_ok vs alignment_checked,
> EstimatorMissing, and the "not reported" cases. A consumer must be able to tell "absent because
> not applicable" from "absent because something failed" — if that distinction isn't expressible in
> the current artifacts, say so.
>
> Part 2 — add a validation step at the end of the pipeline that fails loudly if any artifact is
> malformed, incomplete, or internally inconsistent, before it is published. Inconsistent means e.g.
> a coverage value with no nominal, a skill figure that doesn't reconcile with its logged MAEs, a
> champion naming a model absent from the pool. Test it against a deliberately corrupted artifact,
> not just a clean one.
>
> Part 3 — close the small gap from item 5: DESCRIPTIONS has 18 entries but model_pool() never
> enumerates ETS and Theta, so two descriptions are unreachable from the page. Either make them
> reachable or remove them, and add a test pinning descriptions to the enumerable set so they can't
> drift apart again.
>
> Session logging: docs/sessions/2026-08-11-item-6-agent-contract.md, full narrative verbatim plus
> prompt, plan, real output, verdict, outstanding. Run both suites, show passing output, commit the
> md with the code, walk me through the diff.

---

## 2. Plan

1. **Part 1** — read the artifacts on disk, not the writers. Enumerate every field, type, presence
   and the consumer's fallback. Answer the absent-vs-failed question honestly, per field.
2. **Part 2** — `backend/artifact_validation.py`, three defect classes kept apart (malformed /
   incomplete / inconsistent), ERROR vs WARNING with `--strict`. Wire into the end of
   `daily_summary.py` returning a distinct exit code. Test every check twice: clean and corrupted.
3. **Part 3** — decide reachable-or-remove on evidence, then pin DESCRIPTIONS ≡ pool in both
   directions.

---

## 3. Commands run and their real output

### Both suites

```
$ ./backend/.venv/bin/python -m pytest -q        # repo root
ROOT EXIT=0
653 passed, 3 skipped in 231.00s (0:03:50)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
110 passed in 6.63s
```

### The validator alone

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_artifact_validation.py -q
................................................                         [100%]
48 passed in 10.35s
```

### The validator against the real committed run — first pass

```
4 error(s), 10 warning(s) across 14 artifact(s):
  [WARNING] SUMMARY.json (incomplete): 'run_id' absent -- a consumer cannot identify which run this is.
  [WARNING] SUMMARY.json (incomplete): 'schema_version' absent -- ...
  [ERROR] SUMMARY.json:C_DL (inconsistent): champion 'MLP' is not in the model pool (23 models).
  [ERROR] a_stat/leaderboard.csv (inconsistent): 'target' is populated on 1 of 2 rows ... rows: ['ETS']
  [ERROR] a_stat/leaderboard.csv (inconsistent): 'horizon' is populated on 1 of 2 rows ...
  [ERROR] a_stat/leaderboard.csv (inconsistent): 'cadence' is populated on 1 of 2 rows ...
  [WARNING] a_stat/leaderboard.csv (incomplete): column 'RMSE' is entirely empty ...
  [WARNING] a_stat/...+predictions_long.csv (inconsistent): model(s) scored in the leaderboard with
            no rows in predictions_long: ['Persistence (baseline)'] ...
  [WARNING] b_ml/...: ['⚡ Persistence (baseline)'] ...
  [WARNING] b_ml/...: decorated model name(s) used as a join key: ['⚡ Persistence (baseline)'] ...
  [WARNING] b_ml/metrics_long.csv (incomplete): metric column 'MAE_skill_vs_Ops' is entirely empty ...
  [WARNING] b_ml/metrics_long.csv (inconsistent): coverage column(s) ['PI_coverage@90'] carry their
            nominal level in the column NAME with no 'coverage_nominal' field beside them ...
  [WARNING] e_quantile/metrics_long.csv (inconsistent): 82 coverage row(s) carry no nominal level ...
  [WARNING] published/2025-08-06 (incomplete): no estimators/manifest.json ...
```

### Schema drift, measured

```
=== does any on-disk SUMMARY.json carry schema_version? ===
2026-07-29 -> schema_version: None | run_id: None | n_top: 7
2026-07-30 -> schema_version: None | run_id: None | n_top: 8
2026-08-04 -> schema_version: None | run_id: None | n_top: 8
```

### The three leaderboard schemas, and metrics_long's two shapes

```
=== a_stat/leaderboard.csv  (2 rows) ===
  cols: ['target', 'horizon', 'cadence', 'model', 'MAE', 'RMSE', 'rank']
  PARTIALLY populated: {'target': 1, 'horizon': 1, 'cadence': 1}
  ALL-NULL columns: ['RMSE']
=== b_ml/leaderboard.csv  (9 rows) ===
  cols: ['target', 'horizon', 'model', 'MAE', 'rank']
=== e_quantile/leaderboard.csv  (2 rows) ===
  cols: ['model', 'pinball_q10', 'pinball_q50', 'pinball_q90', 'coverage_p10_p90', 'MAE']

=== metrics_long shapes ===
  a_stat       WIDE (one col per metric)
  b_ml         WIDE (one col per metric)
  e_quantile   LONG (metric/value rows)
```

```
=== can leaderboard join predictions_long on `model`? ===
  in leaderboard only : ['⚡ Persistence (baseline)']
  emoji in a key?     : ['⚡ Persistence (baseline)']
```

### Part 3 — the pool before and after

```
before: pool 16 | DESCRIPTIONS 18 | unreachable ['ETS', 'Theta']
after : pool 28 | DESCRIPTIONS 28 | unreachable [] | undescribed []
        by pipeline: {'B_ML': 13, 'E_QUANTILE': 3, 'A_STAT': 7, 'C_DL': 5}
        unavailable: []
```

### The A_STAT leaderboard writer, before and after the fix

```
     target  horizon cadence                   model         MAE        RMSE  rank
0  Revenues        5   Daily                     ETS  44000000.0  51000000.0     0
1  Revenues        5   Daily  Persistence (baseline)  61000000.0         NaN     1

identity columns fully populated: True
RMSE populated for the model row: True
```

### The committed run, after the item-6 fixes

```
3 error(s), 10 warning(s) across 14 artifact(s)
```

All three remaining errors are the A_STAT identity defect; the writer is fixed and the artifact
predates it.

### Diff stat

```
 backend/artifact_validation.py             | 630 +++++++++++++++++++++++++++++
 backend/c_dl_pipeline.py                   |   7 +
 backend/c_dl_registry.py                   |  35 ++
 backend/model_reference.py                 |  86 +++-
 backend/run_a_stat.py                      |  88 +++-
 backend/tests/test_artifact_validation.py  | 562 +++++++++++++++++++++++++
 backend/tests/test_daily_summary_gating.py |  12 +-
 backend/tests/test_regression_mutations.py |  76 +++-
 docs/AGENT_ARTIFACT_CONTRACT.md            | 345 ++++++++++++++++
 reports/gate_audit.md                      |  15 +
 scripts/daily_summary.py                   |  26 ++
 11 files changed, 1862 insertions(+), 20 deletions(-)
```

---

## 4. Full narrative response, verbatim

Item 6 done. The validator found four real defects in the shipped artifacts on its first run, one of
which I then fixed at the writer, and one of which was a gap the same shape as the one Part 3 was
about. Both suites green: root **653 passed, 3 skipped, `EXIT=0`**; frontend **110 passed,
`EXIT=0`**.

### Part 1 — the contract, and the honest answer on absence

[`docs/AGENT_ARTIFACT_CONTRACT.md`](../AGENT_ARTIFACT_CONTRACT.md) documents every field of
`SUMMARY.json`, the three per-family tables, the coverage and integrity blocks, and the published
forecast directory — with type, presence, and what a consumer should do when a field is absent. It is
written from the artifacts on disk, not from the writers, and that distinction turned out to matter
more than I expected.

**On the absent-vs-failed question: it is expressible in some places and not in others, and the
difference is whether a companion field exists.** That is the finding, and I put it at the top of the
document rather than in a footnote.

Where it *is* expressible, it is because something explicitly says why:

- `gate_passed` is genuinely tri-state — `true` / `false` / **`null` = never verified**.
- `alignment_ok` is absent unless `alignment_checked` is true, and `alignment_check_error` carries
  the reason when the check could not run.
- `coverage_p10_p90` absent is explained by `legacy_coverage_key_omitted`, which states that the
  fitted alphas no longer describe an 80% interval.
- Estimators distinguish four states cleanly: no manifest (never retained), manifest with
  `retention.pruned` + `pruned_at` (deliberately pruned), manifest with blobs missing and **no**
  prune marker (**this one is an error** — unexplained absence), and a clone (blobs never present,
  digests still prove what was published).

Where it is **not** expressible:

- **`skill_pct` = `"n/a (not produced)"` is one marker for several causes** — the family produced no
  output, its integrity report had no `skill_pct`, or the value was not numeric. A consumer cannot
  separate "this family does not compute skill" from "the computation failed".
- **Any all-null CSV column asserts nothing.** `y_lo`/`y_hi` are legitimately empty for models with
  no native intervals; `MAE_skill_vs_Ops` is empty because the ops baseline does not apply to stock
  targets. Both are indistinguishable from a failure.
- **`coverage_nominal` absent** means the run predates the field — which a consumer could tell from
  `schema_version`, except that no committed `SUMMARY.json` has one.

So the recommendation in the document is the pattern the codebase already stumbled into twice and
should adopt deliberately: **where a field may legitimately be absent, write a companion field naming
the reason.** A bare null is not a contract. Until then a consumer must treat missing and `n/a` as
UNKNOWN — never zero, never a pass, never a failure.

Three contract hazards worth calling out separately, because they are the kind of thing that breaks a
consumer quietly:

**Numbers are published as strings.** `horizon` is `"5"`, `skill_pct` is `"27.51%"`, and
`best_model` is `"ETS (MAE 44,199,748)"` — a model name and a thousands-separated number in one prose
field. Recovering the name means splitting on `" ("`.

**`metrics_long.csv` has two incompatible shapes under one filename.** `e_quantile` is genuinely long
(`model, fold, metric, quantile, value`); `a_stat` and `b_ml` are wide, one column per metric. The
file called "long" is wide in two of three families. A consumer has to detect which it has.

**The three leaderboards have three different schemas.** Only `model` is guaranteed. `e_quantile` has
no `target` and no `horizon` at all, so its leaderboard cannot be keyed by target — that has to come
from `SUMMARY.json` or the directory name.

### The thing that justified reading files instead of source

`test_artifact_contract.py` asserts contract clauses by grepping the *writer*: for instance that
`'"schema_version"'` appears in `daily_summary.py`. Every one of those tests passes. And **no
`SUMMARY.json` on disk carries `schema_version` or `run_id`** — measured on all three committed runs.
The writer emits them; the artifacts predate the writer; a source test cannot see the difference.

That is the whole argument for Part 2 in one example: a contract test that never opens an artifact is
testing the intention, not the interface.

### Part 2 — the validator

`backend/artifact_validation.py`. Three defect classes, deliberately separate because the response
differs:

**malformed** — unparseable or wrong-typed: missing file, bad JSON, absent required column,
`gate_passed` that is not tri-state, `skill_pct` that is neither a number nor a known marker,
coverage outside [0, 1].

**incomplete** — parseable but a value is missing with nothing saying why. Narrower than it sounds,
because absence is legitimate here in several places, so this class is reserved for absence that no
explicit marker accounts for.

**inconsistent** — every field parses and they contradict each other. These are the dangerous ones,
because no per-field check finds them: `gate_passed: false` with an empty `gate_reasons`;
`gate_passed: true` *with* reasons; `overall` disagreeing with the families it is derived from;
duplicate family names; a coverage value with no nominal; a nominal contradicting its own quantiles;
`origin_date >= target_date`; a published `y_true`; crossed quantiles; a partially-identified
leaderboard.

Two of the inconsistency checks are worth describing because they encode bugs this project actually
had:

**Skill must reconcile with its own MAEs**, within 0.75 pp of
`(mae_persistence - mae_model) / mae_persistence`. This is the check that would have caught the WS2
harness bug, where a non-canonical ruler inflated every logged skill while both MAEs sat in the same
record disagreeing with it. The test feeds it the real numbers from that incident — a logged 32.62%
against MAEs implying 29.33% — and requires the failure message to name both.

**A champion must exist in the model pool.** If the leaderboard crowns something a consumer cannot
look up, the result cannot be explained or refitted.

`ERROR` blocks publication; `WARNING` does not and covers defects that are real but historical — an
old artifact missing a field added later is a fact about that artifact, not a reason to refuse to
read it. `--strict` promotes warnings to errors, which is what a *new* run is held to. Wired into the
end of `daily_summary.py`, returning exit code **2** so a contract failure is distinguishable from
"a family produced no output" (1). `--no-validate` exists for diagnosing a broken run, and the code
says so.

On testing it against corruption: **48 tests, and the corrupted cases are the point.** Every check is
exercised twice — once on an artifact that should pass, once on the same artifact with exactly one
thing broken. There are also two tests that assert the clean fixture passes, including under
`--strict`, because otherwise a validator that errors on everything would look like it worked.

### What it found, including one I had to fix at the writer

First run against `backend/forecast_runs/2026-08-04`: **4 errors, 10 warnings.**

The most interesting error was `champion 'MLP' is not in the model pool` — **C_DL was not enumerable
either.** Exactly the same defect Part 3 was about, in a different family, and I only found it because
the validator cross-checks a published champion against the pool. I closed it the same way rather than
weakening the check, which is the only honest option: a check that gets relaxed to make a real defect
go away is not a check.

The other three errors were one real malformation: **A_STAT's leaderboard had `target`, `horizon` and
`cadence` populated on the baseline row and blank on the winning model.** The cause is a one-line
aggregation: `lb` was built with `metr.groupby("model")["MAE"].mean()`, which drops those columns,
while the persistence row was concatenated *with* them. So "which model won for target X" was
unanswerable from the file. `RMSE` was dropped by the same expression, which is why it read as an
all-null column despite being computed in `metrics_long.csv`. Both fixed; verified that the corrected
construction populates identity on every row and carries RMSE onto the model row.

I did **not** regenerate the committed artifact, so it still has those three errors. That is
deliberate and stated in the document: re-running a family to rewrite a committed artifact for
cosmetic gain is a worse trade than recording honestly that the artifact predates the fix and the next
run will pass. There is a test pinning exactly three errors of exactly that class, so if a new error
class appears in the committed artifacts it fails, and if someone regenerates the run it also fails
and forces the document to be updated.

Two warnings I looked at and deliberately left:

**The persistence baseline row appears in the leaderboard but has no rows in `predictions_long.csv`**
— it is derived from `origin_value`, not predicted. A consumer joining on `model` silently loses it.
Real, worth knowing, documented, not a malformation.

**`⚡ Persistence (baseline)` uses an emoji in a join key.** Indefensible as a design, but changing a
model label touches artifacts, leaderboards and any frontend logic matching that literal. That is a
rename, not a fix, and not something to fold into this session.

One more thing the validator changed: three tests in `test_daily_summary_gating.py` started failing,
because their synthetic fixture crowned a model called `m1`. The validator was right — a real run
naming a champion outside the pool is the defect. So I fixed the *fixture* to use `Ridge` rather than
exempting it, and left a docstring explaining why it uses a real model name, so nobody simplifies it
back.

### Part 3 — reachable, not removed

The evidence decided this. `DESCRIPTIONS` had `ETS` and `Theta`, and I first assumed they were dead
text to delete. They are not: **`run_a_stat.py` implements seven models** — `NAIVE`, `WEEKDAY_MEAN`,
`MOVAVG`, `ETS`, `SARIMAX`, `STL_ARIMA`, `THETA` — and it is the production A_STAT, invoked by
`scripts/run_daily_forecast.sh` and by every A_STAT test. (`a_stat_models_pipeline.py` is a second,
unreferenced implementation with lowercase names; that duplication is a separate finding, listed as
outstanding.) So the descriptions were right and the enumeration was missing.

Also: the old keys were `"ETS"` and `"Theta"`, and the family dispatches on `"THETA"`. The key
matched nothing in either direction.

So both families are now enumerable through the same contract E_QUANTILE already used —
`registry_models()` — and `model_pool()` returns **28**: 13 B_ML, 3 E_QUANTILE, 7 A_STAT, 5 C_DL.
`DESCRIPTIONS` is exactly that set, pinned in both directions, so a model added without a description
fails and a description for a nonexistent model fails.

Three details worth naming:

`A_STAT_MODELS` carries a `role` of `"forecast"` or `"baseline"`, because three of the seven are
reference baselines. Summing them into a headline count would present the ruler as a rival to the
models measured against it.

`_fc` now **refuses an unknown model name.** It used to fall through to a carried-forward last value,
so `TG_MODEL_FILTER=XGBoost` would have produced a naive forecast published under the label
"XGBoost". A test asserts the refusal, and another asserts every name in the registry has a branch in
the dispatch — so an advertised model that cannot run is caught.

C_DL's catalogue went into a separate `c_dl_registry.py` with **no torch import**, so the Models page
can enumerate the family on a machine without torch. `model_pool()` probes whether `c_dl_pipeline`
imports and marks the models unavailable if not — the same rule already applied to xgboost, lightgbm
and catboost, where a model must *say* it is unavailable rather than vanish from the page.

That changes the client-facing number I pinned yesterday from 16 to 28, so I updated the pinned test
and added a note to `reports/gate_audit.md` §4. The recommendation is unchanged in substance and
sharper in detail: never a single headline number. **"13 machine-learning models and 4 statistical
models compete on each target; prediction intervals come from 3 quantile methods; 3 further entries
are reference baselines, not competitors."** The champion-eligible pool for the three live recipes is
still the 13.

---

## 5. Verdict

**Item 6 complete, all three parts.**

Part 1 delivered as `docs/AGENT_ARTIFACT_CONTRACT.md`, written from the artifacts rather than the
writers. It answers the absent-vs-failed question per field rather than in general: expressible where
a companion field states the reason (`gate_passed` tri-state, `alignment_checked`,
`legacy_coverage_key_omitted`, the four estimator states), not expressible where absence is bare
(`skill_pct`'s single marker, every all-null column, a missing `coverage_nominal`). The recommended
pattern is stated.

Part 2 delivered as `backend/artifact_validation.py` with 48 tests, every check exercised on both a
clean and a deliberately corrupted artifact, wired into the end of `daily_summary.py` with a distinct
exit code. It found 4 real errors on its first run against the shipped artifacts. One (C_DL's
unenumerable champion) I fixed by closing the gap rather than relaxing the check; three (A_STAT's
partially-identified leaderboard) I fixed at the writer and verified, without regenerating the
committed artifact.

Part 3 delivered: reachable, not removed, because the evidence said the descriptions were right and
the enumeration was missing. `model_pool()` and `DESCRIPTIONS` are now the same 28-model set, pinned
both ways. A_STAT also stopped silently forecasting naively under a requested model's name.

653 passed / 3 skipped `EXIT=0`; frontend 110 passed `EXIT=0`.

---

## 6. Outstanding

* **`backend/a_stat_models_pipeline.py` is a second, unreferenced A_STAT implementation** with
  lowercase model names and a different default set. Nothing imports it but its own docstring and a
  README. It should be retired the way `preprocessing.integrity` was, or wired up — having two is how
  the duplicated-baseline bug happened.
* **The committed `2026-08-04` A_STAT leaderboard still fails the contract** (3 errors). The writer
  is fixed; the artifact was deliberately not regenerated.
* **`⚡ Persistence (baseline)` uses decoration in a join key.** Renaming touches artifacts,
  leaderboards and frontend matching — a deliberate rename, not a fix to fold in here.
* **The derived persistence row is absent from `predictions_long.csv`**, so a leaderboard-to-
  predictions join loses it. Documented as a warning.
* **`skill_pct` and `horizon` are strings, and `best_model` embeds a number in prose.** A v3 schema
  should publish numbers as numbers and keep display strings in separate fields.
* **`metrics_long.csv` has two shapes under one filename.** Worth unifying, but it changes what every
  existing consumer parses.
* **`PI_coverage@90` (b_ml) still carries its level in a column name** with no `coverage_nominal`
  beside it — audit field #2, unfixed for B_ML.
* No E_QUANTILE or A_STAT run has been re-executed since items 5–6, so no artifact yet carries
  `coverage_nominal` or the fixed leaderboard.
* Unchanged from earlier records: `conditional_coverage_gate` not wired into
  `quantile_quality_gate`; `check_feature_leakage` is weak; a family writing no shift fields has no
  effective persistence-mimicry check at h=5; six artifact fields still unfixed; master-prompt Part 5
  accuracy levers, Part 6 remainder, Part 7 Georgian i18n; ops P0, Phase-1 cleanup, registry
  approval workflow, single TEST read.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
