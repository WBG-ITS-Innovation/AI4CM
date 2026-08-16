# Session 2 — regenerate a clean published issue after the P2 gate rewrite

**Filename note.** The brief named this file `2026-08-14-session2-reissue.md` and it is kept at that
path. The work actually ran on **2026-08-15 local (2026-08-16 UTC)**, which is why the issue this
session publishes is dated `2026-08-16`: `next_issue_date()` uses `pd.Timestamp.now(tz="UTC")`, and
the run crossed midnight UTC.

**Branch** `model/excellence`  **Start** `aabb27e`  **End** `21326ab`

---

## 1. The brief's hypothesis was wrong, and the difference mattered

> HYPOTHESIS to verify first, do not assume: all three need a regenerated registry/issue that
> postdates the gate rewrite, because the pipeline has not been re-run since 539ae61.

**The pipeline had already been re-run after 539ae61.** The failures had a different cause, and had
I acted on the hypothesis I would have regenerated a registry that was already correct and never
looked at the directory that was actually empty.

### What the tests actually said

The brief reported "2 failed … both KeyError: 'Revenues'". The real count is six, and the first one
names the cause outright rather than dying on a missing key:

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_publication_gates.py \
    backend/tests/test_verdict_reconciliation.py \
    backend/tests/test_failure_mode_distinctness.py \
    backend/tests/test_published_forecasts.py -q

    def test_reconciling_does_not_modify_any_published_issue():
        before = {p: p.read_bytes() for p in sorted(PUBLISHED.rglob("gates.json"))}
        reconcile_verdicts()
        after = {p: p.read_bytes() for p in sorted(PUBLISHED.rglob("gates.json"))}
        assert before == after, "reconciliation must read, never write"
>       assert before, "there should be at least one published issue to reconcile"
E       AssertionError: there should be at least one published issue to reconcile
E       assert {}

FAILED test_verdict_reconciliation.py::test_reconciling_does_not_modify_any_published_issue
FAILED test_verdict_reconciliation.py::test_every_published_target_reports_both_verdicts[Revenues-withheld_as_forecast-publishable]
FAILED test_verdict_reconciliation.py::test_every_published_target_reports_both_verdicts[Expenditure-withheld_as_forecast-withheld]
FAILED test_verdict_reconciliation.py::test_every_published_target_reports_both_verdicts[State budget balance-publishable-withheld]
FAILED test_verdict_reconciliation.py::test_the_reason_names_the_gate_that_actually_changed_it
FAILED test_verdict_reconciliation.py::test_gate_changes_distinguish_added_from_rethresholded
6 failed, 63 passed, 6 skipped in 2.94s
```

`forecasts/published/` was empty. The working tree held only `forecasts/scorecard.csv`:

```
$ git ls-tree -r --name-only 576ea5f^ -- forecasts/     # before the data-removal commit
forecasts/published/2025-08-06/{forecast.csv,gates.json,manifest.json,provenance.json}
forecasts/published/2026-08-13/{estimators/manifest.json,forecast.csv,gates.json,manifest.json,provenance.json}
forecasts/scorecard.csv

$ git ls-files forecasts/                                # now
forecasts/scorecard.csv
```

`576ea5f` moved the published issues into `private_vault/published/`, and the restore covered
`backend/data`, `frontend/data`, `experiments/` and `registry/` — but not `forecasts/published/`.

### Evidence that the pipeline had already been re-run

`private_vault/published/2026-08-13/gates.json` already carried the post-`539ae61` gate set:
`accuracy_vs_naive` present, `signal.threshold == 1.15`, `vs_ruler` absent, both passing. Pointing
the reconciler at the vault — read-only, no changes — reproduced exactly what the six failing tests
assert:

```
$ ./backend/.venv/bin/python -c "... reconcile_verdicts(published_root=Path('private_vault/published'))"
2025-08-06  Revenues               at_issue=withheld_as_forecast   today=publishable            changed=True
2025-08-06  Expenditure            at_issue=withheld_as_forecast   today=withheld               changed=True
2025-08-06  State budget balance   at_issue=publishable            today=withheld               changed=True
2026-08-13  Revenues               at_issue=publishable            today=publishable            changed=False
```

The registry was already correct too — tracked, clean against `HEAD`, carrying the real DEV
credentials (`mase` 0.757959 / 1.103854 / 1.57832).

### But a regeneration was still needed — for a different reason

The existing `2026-08-13` issue fails the brief's own Task 3 on its own terms:

```
"code": { "git_sha": "16876c14a80242bb7512c4f51d08e6d0507572dd",
          "git_dirty": true, "git_dirty_files": 15 }

reachable from HEAD: NO (dangling in the local object DB; absent from a clone)
```

So the conclusion the brief reached was right and the reason it gave was not. That distinction is
the whole finding: a regenerated issue was required for provenance, not for gates.

### A second stale artifact

The only forward run on disk was written at 16:08 UTC on 2026-08-14 against `sha e4d02b19…` with
`"is_synthetic": true`, while `backend/data/processed/master_daily_clean_treasury.csv` was by then
the real file (`sha 0b009fd0…`). `backend/synthetic_data.py` no longer exists and `is_synthetic`
appears nowhere in `provenance.py` — that artifact was an orphan of a code state that is gone. It
would have been published if anything had trusted it.

---

## 2. Verdict mismatch — reported before publishing, as instructed

The brief expected **Expenditure → `withheld_as_forecast`**, citing "sentinel 1.0882 inside the
pooled null". The code produces **`withheld`**:

```
$ ./backend/.venv/bin/python -c "... decide(Measured(...)) for each registry recipe"
Revenues                 verdict=publishable            decided_by=None                failing=[]
Expenditure              verdict=withheld               decided_by=accuracy_vs_naive   failing=['accuracy_vs_naive', 'signal']
State budget balance     verdict=withheld               decided_by=accuracy_vs_naive   failing=['accuracy_vs_naive']
```

Expenditure fails **both** gates. The sentinel reading cited is real, is recorded, and is reported
in `failing_gates` — but `_SEVERITY` in `backend/publication_gates.py` orders
`accuracy_vs_naive → WITHHELD` ahead of `signal → WITHHELD_AS_FORECAST`, so MASE 1.103854 sets the
verdict. That matches `539ae61`'s own commit message and the assertion already standing in
`test_verdict_reconciliation.py:57`.

Raised and confirmed with the user before publishing: the brief's expectation was mis-stated, the
code is right. Nothing was changed in the gate logic.

---

## 3. Verdicts, before and after

### Published verdicts: at issue vs under the current gates

| Target | verdict at issue (2025-08-06) | verdict today | deciding gate | measurement |
|---|---|---|---|---|
| Revenues | `withheld_as_forecast` | **`publishable`** | — every gate passed | MASE 0.757959, sentinel 1.2255 |
| Expenditure | `withheld_as_forecast` | **`withheld`** | `accuracy_vs_naive` | MASE 1.103854 (also fails signal, 1.0882) |
| State budget balance | `publishable` | **`withheld`** | `accuracy_vs_naive` | MASE 1.57832 (signal passes, 7.0058) |

The published record was **not** rewritten. `gates.json` for `2025-08-06` still records what was
decided on the issue date; the two verdicts are reconciled, not merged.

### The brief's expectation vs what was produced

| Target | brief expected | produced | agrees |
|---|---|---|---|
| Revenues | publishable | `publishable` | yes |
| Expenditure | `withheld_as_forecast` | `withheld` | **no** — see §2 |
| State budget balance | "re-verdicted honestly" | `withheld` | yes |

### Published issues, before and after this session

| | before | after |
|---|---|---|
| `forecasts/published/` | *empty* | `2025-08-06`, `2026-08-13`, `2026-08-16` |
| newest issue provenance | — | `git_dirty: false`, sha `4a0ff50…`, reachable from `HEAD` |
| `private_vault/published/` | `2025-08-06`, `2026-08-13` | + `2026-08-16` |

---

## 4. Changes made

Four decisions were put to the user before any change; all four were approved.

### `e3074bc` — stop tracking published issues and the experiments log

`.gitignore` asserted that `forecasts/published/` and the experiments log **are** tracked, while the
sanitization had moved both to the vault, and the restore left them untracked-but-not-ignored — so
`git add -A` would have recommitted client data.

`forecast.csv` carries `origin_value` and the full P10/P50/P90 path per target-date. That is
row-level Treasury data, not the aggregate class `backend/forecast_runs/*/SUMMARY.json` belongs to.
Both prior rules are quoted in place rather than deleted, including the cost that is now
unavoidable: `verify_log_integrity()` needs a JSON per row, so a clone can no longer verify the log.
The vault copy is what it runs against.

### `4a0ff50` — a clean tree records `git_dirty: false`, not `None`

**Found while trying to satisfy Task 3, and it blocked it.** `git_dirty` is documented as tri-state,
but `False` was unreachable:

```
$ ./backend/.venv/bin/python -c "..."
git status --porcelain  -> returncode: 0 | stdout repr: ''
_git('status','--porcelain') -> None
describe_code() on this CLEAN tree:
   git_sha          'e3074bc38f8fd2c3f5c9f6ba00869d6bc98d6459'
   git_branch       'model/excellence'
   git_dirty        None
   git_dirty_files  0
```

`_git` ended with `out.stdout.strip() or None if out.returncode == 0 else None`, folding "succeeded
with empty output" into the same `None` as "command failed". For every other caller that is right;
for `git status --porcelain` empty output *is* the answer. Every clean-tree run recorded "unknown" —
so a published artifact could never assert that its SHA fully identified what ran, which is the
entire point of the flag. It survived because the dirty path is the one exercised during
development: `2026-08-13` records `true`, correctly.

Fixed with an `allow_empty` flag on `_git`, opted into by that one caller, plus an optional `repo`
argument on `describe_code()` so the states can be driven against real repositories.

The test that should have caught it asserted only that the keys existed:

```python
code = describe_code()
assert "git_dirty" in code and "git_dirty_files" in code
```

That passes against the bug and against any value the function could return. Replaced with three
tests over throwaway git repos — clean, dirty, not-a-repo. **Verified by mutation**, restoring
`allow_empty=False`:

```
    def test_a_clean_tree_records_dirty_false_not_unknown(tmp_path):
        code = describe_code(repo=_throwaway_repo(tmp_path / "clean"))
>       assert code["git_dirty"] is False, (
            "a clean tree must record False, not None -- None means 'could not be answered'")
E       AssertionError: a clean tree must record False, not None -- None means 'could not be answered'
E       assert None is False

FAILED backend/tests/test_provenance.py::test_a_clean_tree_records_dirty_false_not_unknown
1 failed, 19 passed in 1.70s
```

The clean case, and only the clean case. Fix restored: `20 passed`.

### `21326ab` — move the retention guarantee to the vault

Three tests encoded the tracking policy `e3074bc` reversed. Each had a real concern, so each was
re-pointed rather than dropped.

`test_the_real_published_run_is_tracked_by_git` asserted the published directory was **not**
gitignored — *"retention is pointless if the directory is gitignored, which it was."* Still the right
concern; only the location changed. It now checks both halves together: the repo copy **is** ignored
and a byte-identical copy **exists** in the vault.

**Written that way, it immediately failed** — and the failure was real, not cosmetic:

```
E       AssertionError: issue 2026-08-16 has no copy in private_vault/published/ -- publishing it
        retained nothing durable. publish() writes only to forecasts/published/, so the vault copy
        is currently a manual step.
```

The issue this session had just published had retained nothing durable. Copied by hand; the gap in
`publish()` is left as an open item rather than widened into a test commit.

The estimator carve-out was **absorbed, not repealed**. It diverged from the old CSV rule for the
strongest available reason — a fitted estimator embeds training data that cannot be scrubbed at all —
and was drawn too narrow, since the numbers it protects sat in `forecast.csv` beside it. The blob
assertion is unchanged, the reasoning is quoted into the section that now covers it, and a
documentation test pins the sentence saying the blobs stay ignored on their own merits if the CSV
rule is ever revisited. Also added the inverse assertion the old test lacked: `scorecard.csv` and
`registry/recipes.json` must **not** be ignored — the failure mode of a broadening rule is that it
keeps going.

### Not committed — five unrelated documents moved out of the repo

Five `.docx`/`.md`/`.pdf` files belonging to an unrelated engagement appeared in `backend/`
mid-session, making the tree dirty and blocking `git_dirty: false`. On the user's instruction they
were moved intact to a location outside the repository. Nothing was deleted and nothing was
committed. Worth recording as a pattern rather than an incident: this repo was sanitized this week
to stop client material sitting in a tracked tree, and unrelated material landed in it four days
later. The engagement is deliberately not named here — a session record is a tracked file, and
naming a third party in it recreates in miniature exactly the exposure the sanitization removed.

---

## 5. The re-issue

Forward run regenerated on the real data from the clean tree. It reproduces the published numbers
exactly — h=1 Revenues P50 = 99,608,226, matching the `2025-08-06` artifact:

```
$ ./backend/.venv/bin/python backend/run_forward_forecast.py
[forward] data through 2025-08-06, 3867 rows
[forward] Revenues: LightGBM_L1 + GBQuantile, groups=['A_deadline','B_holiday','C_month','D_aligned_lags','E_rolling'], exog=none
           2025-08-07  h=1  P50=        99,608,226  [        42,438,365 ..        147,795,384]
           2025-08-08  h=2  P50=        76,354,154  [        46,301,458 ..        108,784,926]
           2025-08-11  h=3  P50=        77,637,466  [        38,825,007 ..        104,202,634]
           2025-08-12  h=4  P50=        61,557,272  [        46,123,434 ..        102,689,428]
           2025-08-13  h=5  P50=        58,319,423  [        44,110,906 ..        141,892,729]
[forward] test_window_touched = False
```

Every target went through `official_run` + `publish_official`. The refusals are produced by the same
code path that publishes, not by filtering the list beforehand:

```
re-issuing under the current gates as issue_date=2026-08-16

  Revenues               PUBLISHED -> 2026-08-16   (registry verdict: publishable)

  Expenditure            REFUSED: Refusing to publish 'Expenditure': its current verdict is
                         'withheld', which means a documented trivial benchmark is more accurate
                         than this model. Publishing the numbers would invite a worse decision than
                         publishing nothing. ('withheld_as_forecast' still publishes -- there the
                         numbers are the best estimate available and only the event claim is withheld.)

  State budget balance   REFUSED: Refusing to publish 'State budget balance': its current verdict is
                         'withheld', ...
```

### Task 3 — provenance of the new issue

```
=== provenance.code ===
{
  "git_sha": "4a0ff5087c690d7a229f6284b5349652e9b303fe",
  "git_branch": "model/excellence",
  "git_dirty": false,
  "git_dirty_files": 0
}
data sha  : 0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1
generated : 2026-08-16T00:44:56.839505+00:00
test_window_touched: False

=== SHA validity in current history ===
  reachable from HEAD: YES
4a0ff50 provenance: a clean tree records git_dirty false, not None

=== contrast: the old 2026-08-13 issue ===
  {"git_sha": "16876c14a80242bb7512c4f51d08e6d0507572dd", "git_dirty": true, "git_dirty_files": 15}
  reachable from HEAD: NO (dangling; absent from a clone)
```

Gates recorded on the new issue, and the forecast it carries:

```
 recipe: revenues-lgbm-l1-ws3-v1 | target: Revenues
    accuracy_vs_naive      passed=True  measured=0.757959 threshold=1.0
    coverage               passed=None  measured=None threshold=None
    leakage                passed=True  measured=False
    overfitting            passed=True  measured=< 3.0 threshold=3.0
    persistence_mimicry    passed=True  measured=False
    signal                 passed=True  measured=1.2255 threshold=1.15
    vs_ruler present: False

 targets: ['Revenues'] | rows: 5 | y_true col: False
```

Reconciliation across all three issues:

```
2025-08-06  Revenues               at_issue=withheld_as_forecast   today=publishable            changed=True
2025-08-06  Expenditure            at_issue=withheld_as_forecast   today=withheld               changed=True
2025-08-06  State budget balance   at_issue=publishable            today=withheld               changed=True
2026-08-13  Revenues               at_issue=publishable            today=publishable            changed=False
2026-08-16  Revenues               at_issue=publishable            today=publishable            changed=False
```

---

## 6. Test output

### Gate suite — before and after

Before (four files, 75 tests), as quoted in §1: **6 failed, 63 passed, 6 skipped**. The six skips
were `no re-issue committed` ×3 and `nothing published yet` ×3.

After, with `test_forecast_modes.py` and `test_provenance.py` added to the set:

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_publication_gates.py \
    backend/tests/test_verdict_reconciliation.py \
    backend/tests/test_failure_mode_distinctness.py \
    backend/tests/test_published_forecasts.py \
    backend/tests/test_forecast_modes.py \
    backend/tests/test_provenance.py -q -rs
........................................................................ [ 66%]
....................................                                     [100%]
108 passed in 4.85s
```

**No skips remain in the gate suite.** Every test is addressed by passing.

> On the brief's "64 tests": no combination of the gate-related files collects 64 under the project
> venv, and the reported 61/2/1 split does not reproduce. The counts above are what the named files
> actually collect and run. Note the earlier reported numbers were likely produced under the system
> interpreter, where four gate-related modules fail to import at all — `sklearn` is installed in
> `backend/.venv` only, so `pytest` from the system Python silently collects a different, smaller set.

### Full root suite

First full run of the session, taken after publishing:

```
$ ./backend/.venv/bin/python -m pytest -q
FAILED backend/tests/test_estimator_store.py::test_gitignore_excludes_the_blobs_but_tracks_the_manifest
FAILED backend/tests/test_estimator_store.py::test_the_divergence_from_the_csv_rule_is_documented_where_it_is_made
FAILED backend/tests/test_insights.py::test_withheld_targets_are_explained_not_hidden
FAILED backend/tests/test_treasury_report.py::test_charts_are_inline_svg
FAILED backend/tests/test_treasury_report.py::test_withheld_models_are_shown_not_omitted
5 failed, 802 passed, 8 skipped in 263.36s (0:04:23)
```

Two were the tracking-policy tests, handled in `21326ab`. The other three had a different cause
worth recording: `publish_official` calls `write_artifacts` on the **shared** working directory, so
publishing Revenues alone truncated `backend/forecast_runs/forward/latest` to Revenues, and the
insights and treasury-report tests read that artifact.

```
    def test_withheld_targets_are_explained_not_hidden(narrative):
        held = [s for s in narrative["sections"] if s["verdict"] != "publishable"]
>       assert len(held) == 2, "both flow targets should be withheld as forecasts"
E       assert 0 == 2
```

Not a gate defect. Regenerating the forward run restored all three targets and all three passed. See
open items — the truncation is a real wart.

Final:

```
$ ./backend/.venv/bin/python -m pytest -q
807 passed, 8 skipped in 274.19s (0:04:34)

$ cd frontend && ./.venv/bin/python -m pytest -q
119 passed in 9.59s
```

The eight skips:

```
SKIPPED [1] frontend/tests/test_pages_smoke.py:34: streamlit is installed in frontend/.venv only
SKIPPED [1] frontend/tests/test_tier1_correctness.py:24: streamlit is installed in frontend/.venv only
SKIPPED [1] frontend/tests/test_verdict_history_render.py:26: streamlit is installed in frontend/.venv only
SKIPPED [1] frontend/tests/test_visual_tokens.py:27: streamlit is installed in frontend/.venv only
SKIPPED [1] backend/tests/test_artifact_validation.py:544: no committed run to validate
SKIPPED [1] backend/tests/test_artifact_validation.py:579: no committed run to validate
SKIPPED [1] backend/tests/test_artifact_validation.py:633: no reference run committed
SKIPPED [1] backend/tests/test_artifact_validation.py:660: row-level artifacts are gitignored; present only where the run was made
```

The four frontend skips are the known interpreter split and pass in the frontend venv (119 passed).
The four `test_artifact_validation` skips are **not** legitimate — see open items.

### Final repo state

```
$ git log --oneline -4
21326ab Move the retention guarantee to the vault, and absorb the estimator carve-out
4a0ff50 provenance: a clean tree records git_dirty false, not None
e3074bc Stop tracking published issues and the experiments log; they are client data
aabb27e Ignore .env files, outputs, and logs

$ git status --porcelain
                      # empty

$ ls forecasts/published/     2025-08-06  2026-08-13  2026-08-16
$ ls private_vault/published/ 2025-08-06  2026-08-13  2026-08-16
```

Nothing under `experiments/`, `backend/data/`, `frontend/data/`, `forecasts/published/` or any
`.env` was committed. Each of the three commits was guarded before it was made:

```
$ git diff --cached --name-only | grep -E '^(experiments/|backend/data/|frontend/data/|forecasts/published/)|\.env'
none — clean
```

---

## 7. Open items

1. **`publish()` does not retain to the vault.** Now that `forecasts/published/` is gitignored, the
   vault is the durable record — but publishing writes only to the repo path, so retention is a
   manual `cp`. `2026-08-16` was retained by hand after a test caught it. `publish()` (or
   `publish_official`) should write both, or the next issue will be durable only if someone
   remembers. **This is the highest-value follow-up.**

2. **Four `test_artifact_validation` tests skip because their fixtures were deleted.** The same
   inconsistency as `forecasts/published/`, third instance: `.gitignore:28-38` still asserts
   `backend/forecast_runs/*/SUMMARY.json` and `SUMMARY.txt` must be tracked — *"with the whole
   directory ignored NO run artifact existed in a clone at all"* — but `576ea5f` deleted them, and
   the restore missed `backend/forecast_runs/`. The vault holds `2026-08-04` and `2026-08-12`.
   These files are documented as aggregate-only (model names, MAEs, skill percentages, gate
   verdicts, the data file's *name*), so restoring and tracking them is defensible — but it is the
   same policy call already made twice this session and it was **not** made here. Left untouched
   deliberately; decide and either restore or update the `.gitignore` block so it stops asserting
   something untrue.

3. **`publish_official` truncates the shared working artifact.** It calls `write_artifacts` on
   `backend/forecast_runs/forward/latest`, so publishing one target destroys the other targets'
   forward numbers — which the treasury report and insights read. Cost three test failures this
   session. Publishing should write to a per-issue staging directory, or not touch the shared one.

4. **`forecasts/scorecard.csv` is tracked and its schema is client data.** Header-only today, so
   harmless; its columns include `p10`, `p50`, `p90` and `y_true`. The first scored row makes it
   client data in a tracked file. Flagged in `.gitignore`; revisit before the data file moves past
   2025-08-07, the earliest published target date.

5. **`private_vault/MANIFEST.json` is stale.** It records `n_files: 700` and predates the two issues
   added to the vault this session. Nothing in the codebase reads or verifies it, so it is
   documentation that has quietly drifted — either regenerate it on vault writes or stop presenting
   it as an inventory.

6. **The synthetic-data path is half-removed.** `backend/synthetic_data.py` is gone and
   `provenance.py` has no `is_synthetic` support, but an artifact stamped `"is_synthetic": true`
   was sitting in the working tree and would have been published by anything that trusted it. Either
   restore the generator deliberately or ensure no orphaned synthetic artifact can survive a restore.

7. **`test_verdict_reconciliation.py` still hardcodes `2026-08-13`.** The three re-issue tests pass
   against the restored issue, which is correct today. If `2026-08-13` is ever pruned they revert to
   skipping silently. Consider resolving the newest published issue instead.
