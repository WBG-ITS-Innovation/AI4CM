# Item 4 — persisted estimators: implementation

**Date:** 2026-08-11 · **Branch:** `model/excellence` · parent `4cd7dd0`
**Root suite:** 569 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**Measured per-issue footprint:** 10.45 MB, 60 estimators · **Retention:** `keep_last=8`, unscored
issues protected

The plan stage is recorded separately in
[`2026-08-10-item-4-persisted-estimators.md`](2026-08-10-item-4-persisted-estimators.md).

---

## 1. The exact prompt given

> Item 4 decisions: (1) Retention — take A + C: blobs gitignored, SHA-256 digests tracked in the
> manifest, retention window with declared pruning. The deciding factor is not disk, it's that
> estimators embed Treasury data (y_train_ref_, the level series, and tree split thresholds that
> cannot be scrubbed at all), so tracked blobs would put raw client data permanently in every clone
> of a World Bank repo. Document in the code and the session log that this deliberately diverges
> from the CSV retention rule, and why. (2) Identity — accept fit_id + selection_run_id; your
> objection to borrowing a DEV run_id is correct, and do not fake a forward row in
> experiments/log.csv. (3) Also do the scrubbing you identified as behaviour-neutral: y_train_ref_
> to a one-element median array, level to level.loc[[origin]] — and prove behaviour-neutrality with
> a test, don't assert it.
>
> Then implement: joblib compress=3, the manifest with digests verified before unpickling, the
> loader refusing on digest mismatch and raising EstimatorVersionMismatch on version mismatch,
> describe_environment() extended to include joblib, and a reproduce test showing a loaded
> estimator matches its published predictions to tolerance. Note your own finding that
> _fit_predict_point currently returns a float and discards the fitted estimator — changing that
> touches the point path, so include a test proving published point predictions are unchanged by
> the capture.
>
> State the measured per-issue footprint and the pruning policy in the session log.
>
> Session logging (this and every future session): docs/sessions/2026-08-11-item-4-persisted-estimators.md
> must contain your full narrative response verbatim — the same prose you write in the chat, not a
> condensed summary — plus the exact prompt, the plan, real command output, the verdict, and
> outstanding items. The md file is the record; the chat is just where I read it.
>
> Run both suites, show passing output, commit the md with the code, walk me through the diff.

---

## 2. The plan, as approved

Carried forward from the plan record, with the three decisions applied:

1. **Serialization** — joblib, pickle protocol 5, `compress=3`. The only format that round-trips
   all 13 pool models *plus* `ScaledRegressor` exactly and without changing the numbers. ONNX has
   no converter for the wrapper and computes in float32; LightGBM native covers the booster only,
   cannot represent `HistGBDT_L1`, and measured is not even smaller (109.3 KB gzipped vs 115.6 KB);
   `skops` has no lightgbm/catboost support. Pickle executes code, so the digest is verified
   **before** the bytes reach joblib.
2. **Identity** — `fit_id = <issue_date>/<target-slug>/h<h>/<kind>`, with the DEV run carried as
   `selection_run_id` plus a note that it is not this fit. No row faked in `experiments/log.csv`.
3. **Version pinning** — `PINNED_PACKAGES = (scikit-learn, lightgbm, numpy, joblib)` plus python at
   major.minor. The loader compares against the manifest itself, because scikit-learn only *warns*.
   Digest mismatch: refuse always. Version mismatch: raise, overridable only explicitly and marked
   unverified.
4. **Retention (A + C)** — blobs gitignored, manifest tracked, `keep_last=8` with unscored issues
   protected and pruning declared in the manifest.
5. **Scrubbing** — `y_train_ref_` → one-element `median(|y|)` array; `level` → `level.loc[[origin]]`.
   Behaviour-neutrality proved by test.

---

## 3. Commands run and their real output

### Both suites

```
$ ./backend/.venv/bin/python -m pytest -q          # from repo root
ROOT EXIT=0
........................................................................ [ 88%]
.................................................................        [100%]
569 passed, 3 skipped in 194.96s (0:03:14)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
......................................                                   [100%]
110 passed in 5.42s
```

### The new test file on its own

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_estimator_store.py -q
..........................                                               [100%]
26 passed in 113.59s (0:01:53)
```

### Measured per-issue footprint, written by the real path with all three champions

```
==================================================================================================
MEASURED per-issue footprint -- written by the real path, all three champions
==================================================================================================
  Expenditure               20 estimators     4.49 MB
  Revenues                  20 estimators     4.64 MB
  State budget balance      20 estimators     1.33 MB
  TOTAL                     60 estimators    10.45 MB
  du -sh: 11M
  manifest.json (the only tracked file): 128.9 KB

  weekly issues              0.53 GB/yr unbounded
  daily issues (252 bd)      2.57 GB/yr unbounded
  WITH keep_last=8 steady state     84 MB, any cadence

  Revenues h5 point entry:
     fit_id               2025-08-06/revenues/h5/point
     selection_run_id     20260805T090357.040566_Revenues_LightGBM_L1_ratio
     library              lightgbm
     inner_class          lightgbm.sklearn.LGBMRegressor
     target_transform     ratio
     bytes                875456
     sha256               e5d65b91081272fc2fb35fab2c774d4ff35d34b0...
     scrubbed[y_train_ref_]  full training target (2406 values) replaced by a single number,
                             median(|y|) = 4.93472e+07, which is all the scale check consumes
     scrubbed[level]  trailing-level series reduced from 2763 points to 1 (only the rows this
                      artifact must predict)
==================================================================================================
```

### The bug found during implementation, in its own words

First run of the end-to-end reproduce test:

```
>       assert out["rel_diff"] < 1e-9, out
E       AssertionError: {'fit_id': '2025-08-06/revenues/h5/point',
E                        'reproduced': 106407563.90441726,
E                        'published': 58319423.373441845,
E                        'abs_diff': 48088140.53097542, ...}
E       assert 0.8245647461746739 < 1e-09
```

Diagnosis — the design matrix was rebuilt with no feature groups, then reindexed to the stored
names:

```
recipe groups -> 47 columns
empty  groups -> 17 columns
count missing: 30
all-NaN at origin? (empty-group design, reindexed to full names)
  origin: 2025-08-06 n NaN cols: 30 ['is_deadline_monthly', 'is_deadline_annual', ...]
```

### git tracking, verified on real paths

```
  IGNORED : forecasts/published/2025-08-06/estimators/revenues/h5_point.joblib
  tracked : forecasts/published/2025-08-06/estimators/manifest.json
  tracked : forecasts/published/2025-08-06/forecast.csv

$ git ls-files | grep -c '\.joblib$'
0
```

### The existing published issue

```
$ load_estimator('forecasts/published/2025-08-06', 'Revenues', 5, 'point')
EstimatorMissing: no estimator manifest at forecasts/published/2025-08-06/estimators/manifest.json.
This issue was published without retaining its estimators, so it cannot be re-derived.
```

### Diff stat

```
 .gitignore                                         |  14 +
 backend/estimator_store.py                         | 552 +++++++++++++++++++++
 backend/forecast_modes.py                          |  22 +-
 backend/forward_forecast.py                        |  76 ++-
 backend/provenance.py                              |   6 +-
 backend/run_forward_forecast.py                    |  35 +-
 backend/tests/test_estimator_store.py              | 511 +++++++++++++++++++
 docs/sessions/2026-08-10-item-4-persisted-estimators.md | 320 ++++++++++++
```

---

## 4. Full narrative response, verbatim

Item 4 is in, and one real bug surfaced while building it. Both suites green: root **569 passed, 3
skipped, `EXIT=0`**; frontend **110 passed, `EXIT=0`**.

### The diff, file by file

**`backend/estimator_store.py`** (new, 552 lines) is the whole mechanism. Its module docstring
carries the retention argument you made, because that is where someone will look when they wonder
why one published artifact is untracked while everything beside it is tracked: the blobs embed the
training target, the trailing-level series, and tree split thresholds that are functions of the
data and cannot be scrubbed at all, so committing them would put raw Treasury figures permanently
into every clone of a World Bank repository. The manifest is tracked and carries a SHA-256 per
blob, so a clone can still prove which estimator produced a published number and detect a
substituted one — what it cannot do is run it. `forecast.csv` *is* the claim; an estimator only
re-derives it.

`scrub_for_persistence` does the two reductions and returns a **disclosure dict** alongside the
scrubbed object, which goes into the manifest. That matters more than it looks: the manifest says
in plain text `"full training target (2406 values) replaced by a single number, median(|y|) =
4.93472e+07, which is all the scale check consumes"`. The blob is not data-free and the record does
not pretend otherwise.

`load_estimator` runs three checks in a deliberate order — **existence, digest, then version** —
and only then hands bytes to joblib. Unpickling executes code, so the digest cannot be checked
afterwards. There is a test that monkeypatches `joblib.load` and asserts it is never reached when
the digest is wrong, because "we check the digest" is the kind of claim that quietly stops being
true.

**`backend/forward_forecast.py`** is where the point path changed, which is the part you flagged.
`_fit_predict_point` returned a bare float and threw the fit away; it now returns
`(prediction, estimator)`. `_fit_predict_quantiles` does the same for its three. `run_forward`
gained an opt-in `estimator_sink` list. Two tests hold the line: one asserts the estimator is
actually returned and predicts the same number, and one runs `run_forward` twice on the real data —
once with capture, once without — and requires `assert_frame_equal` on `p10`, `p50`, `p90`,
`p50_quantile_model` and `origin_value`. Nothing about a published number moves.

One subtlety I wrote into the docstring rather than smoothing over: the returned quantile *values*
are sorted to repair crossing, but the estimators are keyed by the alpha each was fitted for. After
a crossing repair the p90 value may not be the one the p90 model produced. The manifest labels each
model by what it was fitted for, which is the only honest option.

`_selection_run_id` looks up the recipe's DEV `run_id` so it can travel with the blob under a name
that cannot be misread. Every manifest entry also carries `selection_run_id_note` spelling out that
it identifies the run that *chose* the recipe, not the fit in the blob — and a test asserts the
`selection_run_id` does not appear anywhere in the `fit_id`.

**`backend/forecast_modes.py`** — `OfficialResult` gained an `estimators` list and
`publish_official` saves them after publishing. `ExploratoryResult` deliberately gained nothing,
and there is a test asserting it has no `estimators` field at all: nothing unpublished should be
leaving Treasury-derived blobs on disk.

**`backend/run_forward_forecast.py`** — the production runner now captures, and gained an opt-in
`--publish` that publishes, retains and prunes. Opt-in because publishing is not reversible.

**`backend/provenance.py`** — `joblib` added to `describe_environment()`, with a one-line note that
it is there as the *reader* of the blobs rather than because it affects a fit.

**`.gitignore`** — the two rules, with the reasoning inline. A test reads the section after
`!forecasts/scorecard.csv` and asserts it contains the word "diverges", so the divergence cannot be
silently normalised later.

### The bug, because it is the interesting part

The first end-to-end reproduce test failed with `reproduced 106,407,563.90` against
`published 58,319,423.37` — 82% wrong. My `reproduce_prediction` built the design matrix with
`fiscal_groups=()` and then reindexed to the stored feature names. Measured: the recipe's design has
47 columns, the empty-group design has 17, so **30 columns were created as all-NaN and predicted
on**. LightGBM handles NaN natively, so nothing complained; it simply returned a different number
that looked exactly as plausible as the right one.

That is the failure mode this whole item exists to prevent, so it got fixed twice over. The feature
groups now travel in the manifest, so the design is rebuilt the way it was fitted. And a shortfall
now raises `ReproductionUnavailable` rather than being reindexed away — both a missing feature and
a NaN in the rebuilt origin row. The regression test edits a manifest to remove the groups and
requires the refusal, with the two real numbers in its docstring.

Worth naming what caught it: a tolerance of `1e-9` on a real published number. A looser tolerance,
or a synthetic fixture, would have passed.

### Measured footprint and the pruning policy

**10.45 MB per issue**, 60 estimators, written by the real path with all three champions. That is
0.91 MB *below* the 11.36 MB in the plan, and the difference is the scrubbing: Revenues fell from
5.58 MB to 4.64 MB once the 2406-value training target and 2763-point level series came out of its
20 blobs. The scrub pays for itself in disk as well as in exposure.

Unbounded that would be 0.53 GB/year weekly or 2.57 GB/year daily. With `keep_last=8` it is a
**steady state of about 84 MB at any cadence**. Pruning deletes blobs, never manifests, and stamps
`pruned: true` with a timestamp and `bytes_freed`, so a pruned issue loads as an explicit
`EstimatorMissing` naming the prune date rather than looking like an issue that never had
estimators. `issues_with_unscored_horizons()` protects any issue with a horizon still lacking a
realized score, however old — those are exactly the predictions still to be argued about. A missing
or unreadable scorecard protects everything, which is the right way for that to fail.

The one cost I will flag rather than bury: `manifest.json` is **128.9 KB per issue** and it *is*
tracked, mostly because 60 entries each repeat their feature-name list. At a daily cadence that is
roughly 32 MB/year of JSON in git. Git deltifies near-identical text well and I judged the
self-contained audit record worth it, but it is a real number and it is the tracked one.

### Two things you should know

**The existing published issue cannot be re-derived.** `2025-08-06` was published before any of
this existed, so it loads as `EstimatorMissing: ... published without retaining its estimators`.
I did not backfill it, and I do not think it should be: its provenance records
`git_dirty: true, git_dirty_files: 20`, so a fit generated today would be a *different* fit wearing
that issue's label — which is exactly the substitution the digest and version checks exist to
prevent. It reads as honestly unavailable. Every issue published from here on retains.

**Nothing was published to the real tree this session.** All the end-to-end runs went to temp
roots; the measurement above is from a `tempfile.mkdtemp()`. `forecasts/published/2025-08-06` is
untouched, still 0 scored / 15 pending, and no `.joblib` is tracked anywhere in the index
(`git ls-files | grep -c '\.joblib$'` → 0).

---

## 5. Verdict

**Item 4 complete.** All requested pieces present and measured: joblib `compress=3`; manifest with
per-blob SHA-256 verified before unpickling; loader refusing on digest mismatch unconditionally and
raising `EstimatorVersionMismatch` on version mismatch; `describe_environment()` extended with
joblib; a reproduce test matching a real published prediction at `rel_diff < 1e-9`; a test proving
published point predictions are unchanged by the capture; behaviour-neutrality of both scrubs proved
on a batch large enough to exercise the scrubbed attribute.

26 new tests, root suite 569 passed / 3 skipped `EXIT=0`, frontend 110 passed `EXIT=0`.

One real bug was found and fixed during implementation: `reproduce_prediction` initially rebuilt the
design with the wrong feature set and predicted on 30 all-NaN columns, returning a number 82% wrong
and entirely plausible. Both the cause and the class of failure are now guarded.

The retention divergence from the CSV rule is documented in three places — the `estimator_store`
module docstring, the `.gitignore` section, and this record — and a test asserts the `.gitignore`
explanation stays there.

---

## 6. Outstanding

* **`2025-08-06` cannot be re-derived** and was deliberately not backfilled (see §4). Every issue
  published from now on retains its estimators.
* `manifest.json` is 128.9 KB per issue and tracked (~32 MB/yr of JSON at a daily cadence). Could
  be reduced by hoisting `feature_names` to one block per target; not done, because a self-contained
  entry is easier to audit.
* Pruning is wired into `run_forward_forecast.py --publish` only. A scheduled job that publishes by
  another route would need to call `prune_estimators()` itself.
* Retention is `keep_last=8` by assumption, not by a stated Treasury requirement. Worth confirming
  against how far back an issue is realistically challenged.
* Unchanged from earlier records: six artifact fields still unfixed (classification delivered,
  decision reserved); master-prompt Part 5 accuracy levers, Part 6 remainder, Part 7 Georgian i18n;
  ops P0, artifact validator, Phase-1 cleanup, registry approval workflow, single TEST read.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
