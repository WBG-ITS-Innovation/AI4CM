# Session log — 2026-08-10 — C_DL alignment integrity, and the artifact-field audit

## 1 · The prompt, exactly as given

> Continue the session on model/excellence. Before item 4, fix the integrity-signal gap flagged in
> the item 3 record: c_dl_pipeline.py:958 writes alignment_ok: True as a literal with no check
> behind it. Replace it with a real check that verifies whatever alignment property the field is
> supposed to attest, and make it write False (or an explicit "not checked") when the property does
> not hold — never an unconditional True. Add a test that a deliberately misaligned input produces
> alignment_ok: False, so the field can never silently lie again.
>
> Then audit the other six missing artifact fields listed in reports/phase14_session_record.md the
> same way: for each, tell me whether it is (a) absent, (b) present but unverified like this one, or
> (c) present and honest. I want that classification before we decide which to fix.
>
> Session logging: docs/sessions/2026-08-10-alignment-integrity.md with prompt, plan, real output,
> verdict, outstanding. Run the full suite, show passing output, commit, walk me through the diff.

## 2 · Plan, as established before editing

**What the field is supposed to attest.** Read from the code rather than assumed. The canonical
implementation is `forecast_integrity.validate_alignment_step_based()`, which documents and checks:

    idx(target_date) - idx(origin_date) == h_steps

**Which index.** `build_sequences()` constructs the two dates as `idx[end_i]` (origin) and
`idx[end_i + horizon]` (target) where `idx = F.index`. So `F.index` is the index those positions
refer to, and `F` is bound at `c_dl_pipeline.py:735` — in scope at the integrity block. Verified
before editing; the real check is a drop-in.

**Behaviour on failure.** Write the checker's verdict, so a misalignment yields `False`. If the
check itself cannot run, **omit the key** and set `alignment_checked: False` — the same discipline as
`read_gate()`: an absent verdict must read as "not checked", never as a pass. A favourable default in
the exception handler would be the original bug in a new place.

## 3 · Commands run, and their real output

### The property, confirmed from source before editing

```
$ sed -n '480,510p' backend/c_dl_pipeline.py
    idx = F.index
    ...
        label_dates.append(pd.Timestamp(idx[end_i + horizon]))
        origin_dates_list.append(pd.Timestamp(idx[end_i]))

$ grep -n "F, y =" backend/c_dl_pipeline.py
735:  F, y = make_feature_frame(df, target, cadence.lower(), multivariate,
```

### The literal is gone

```
$ grep -n '"alignment_ok": True' backend/c_dl_pipeline.py
no unconditional True remains
```

### New tests

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_dl_alignment_integrity.py -v
test_deliberately_misaligned_predictions_yield_alignment_ok_false PASSED [  7%]
test_any_wrong_offset_is_caught[-3] PASSED [ 14%]
test_any_wrong_offset_is_caught[-1] PASSED [ 21%]
test_any_wrong_offset_is_caught[1] PASSED [ 28%]
test_any_wrong_offset_is_caught[2] PASSED [ 35%]
test_any_wrong_offset_is_caught[7] PASSED [ 42%]
test_calendar_gap_does_not_masquerade_as_misalignment PASSED [ 50%]
test_missing_date_columns_yield_false_not_a_pass PASSED [ 57%]
test_origin_absent_from_the_modelling_index_is_a_failure PASSED [ 64%]
test_c_dl_no_longer_writes_an_unconditional_true PASSED [ 71%]
test_c_dl_writes_alignment_from_the_shared_checker PASSED [ 78%]
test_c_dl_omits_the_verdict_when_the_check_cannot_run PASSED [ 85%]
test_the_checker_is_called_against_the_index_sequences_were_built_from PASSED [ 92%]
test_c_dl_module_still_parses_and_the_check_is_inside_the_integrity_block PASSED [100%]
```

### Mutation test — the literal restored, then reverted

```
$ MUTATION: literal True restored
$ pytest -q backend/tests/test_dl_alignment_integrity.py
FAILED backend/tests/test_dl_alignment_integrity.py::test_c_dl_no_longer_writes_an_unconditional_true
1 failed, 13 passed
$ restore -> grep count of the literal: 0
```

### Full suites

```
$ ./backend/.venv/bin/python -m pytest -q
ROOT EXIT=0
541 passed, 3 skipped in 91.70s (0:01:31)

$ PYTHONPATH=frontend:backend ./frontend/.venv/bin/python -m pytest -q frontend/tests
FRONTEND EXIT=0
110 passed in 5.39s
```

### Diff summary

```
backend/c_dl_pipeline.py | 41 ++++++++++++++++++++++++++++++++++++++++-
 1 file changed, 40 insertions(+), 1 deletion(-)
```

## 4 · Verdict

**The C_DL alignment gap is closed.** `alignment_ok` is now written from
`validate_alignment_step_based()` against `F.index`, alongside `n_misaligned`,
`misaligned_examples` (capped at 5) and `alignment_checked`. A failed check prints a warning naming
how many of how many predictions are misaligned. A check that cannot run pops the key and records
`alignment_checked: False` plus the error.

14 tests. The required one — a deliberately misaligned input yields `False` — is
`test_deliberately_misaligned_predictions_yield_alignment_ok_false`, which shifts a single
prediction by **one index position**, the smallest defect an off-by-one or resampling bug produces.
It also asserts the aligned fixture passes, so the test cannot succeed vacuously. A parametrised
test covers offsets −3, −1, +1, +2, +7.

Root suite 527 → **541 passed, 3 skipped**, `EXIT=0`. Frontend **110 passed**, `EXIT=0`.

### One fixture of mine that was wrong

`test_calendar_gap_does_not_masquerade_as_misalignment` asserted the fixture contained varying
calendar spans. It did not: on a plain business-day index, 5 business days is *always* exactly 7
calendar days, so `spans.nunique()` was 1 and the test failed. The point it was making needed a
**gap** in the index — which is exactly what a holiday is, and Georgian holidays are removed from
this project's modelling index. Rebuilt by dropping a day, giving spans of both 7 and 8 days, and the
check still passes because it counts positions rather than days.

## 5 · The audit of the other six fields

Classified against a real B_ML run
(`frontend/runs/run_B_uni_Ridge_Revenues_Daily_h5_20260731_1125`) and `experiments/log.csv`
(153 rows), not from the earlier record.

| # | Field | Class | Evidence |
|---|---|---|---|
| 1 | `nominal_pi` — B_ML's advertised interval level | **(a) absent** from the integrity report — but the level *is* discoverable elsewhere | `nominal_pi` not in `integrity_report.json`. `metrics_long.csv` carries `PI_coverage@90` / `PI_width@90`, so the level is in the **column name**. `ConfigBML.nominal_pi = 0.90` is captured at the point of use but still not written |
| 2 | Explicit quantile level for E_QUANTILE | **(b) present but unverified** | The report writes `coverage_p10_p90`; the level lives in the **key name**, not a field. Inferable and correct today, breaks silently if the quantiles change without the key changing |
| 3 | C_DL `alignment_ok` | **(c) present and honest** — as of this session | Was the literal `True`. Now from the shared checker, with `alignment_checked` distinguishing a real verdict from an absent one |
| 4 | `study` in `experiments/log.csv` | **(a) absent** | Not in `COLUMNS`. Present in `params` JSON for newer runs. The UI derives it by substring-matching the free-text `note`; buckets found that way: `other, workstream 1, ws2, ws3, ws4, ws4 robust, ws5`. A reworded note drops out of its filter |
| 5 | `window` in `experiments/log.csv` | **(a) absent** | Not in `COLUMNS`. Derived from `fold_scheme` by substring: **DEV 53 / TRAIN 100, 0 unmatched** — so the derivation is currently complete, but it is a string match, not a field |
| 6 | `nominal_coverage` per log row | **(a) absent** | Not in `COLUMNS`. **20 of 153** rows carry `coverage_low/mid/high` with no level to compare them against. The 80% nominal exists only in prose |
| 7 | Per-tercile coverage on point-model runs | **(a) absent — and the absence is handled honestly** | **0 of 133** point-model rows carry coverage. The point path never computes intervals, so there is nothing to write. The UI reports "not reported" rather than backfilling |

### Reading of that classification

Only **one** field was actively lying — number 3, now fixed. That distinction matters: an absent
field can be detected by a consumer, whereas `alignment_ok: True` with nothing behind it cannot.

**Number 2 is the one to fix next.** It is the only remaining *(b)*: a value that a consumer will
read as authoritative while nothing guarantees the key name still matches the quantiles produced.

**Numbers 4, 5 and 6 are cheap additive columns** and would remove three string-matching
derivations. Number 6 is the most consequential of the three, because 20 rows currently publish
coverage with no stated nominal.

**Numbers 1 and 7 are data gaps, not honesty gaps.** Number 7 needs the point path to compute
intervals — a modelling decision, not a logging one — and both are already surfaced as "not
reported".

## 6 · Outstanding

**Item 4, unstarted** — persist the fitted estimator with each published forecast: library names and
versions, a loader, a test that a saved model reproduces its published predictions to tolerance, and
the storage cost per issue stated. Official runs must still refit on current data.

**Six artifact fields remain**, classified above. No decision taken on which to fix — that was
explicitly reserved.

**Not re-run:** no recipe changed and no published number moved, so no re-issue and no refresh of
`reports/PROGRESS_SINCE_LAST_REVIEW.md` were due. Verified by `git diff` on `registry/recipes.json`
and `forecasts/`.
