# Item 4 — persist the fitted estimator with each published forecast

**Status: PLAN ONLY — awaiting approval. Nothing implemented.**
**Date:** 2026-08-10 · **Branch:** `model/excellence` @ `4cd7dd0` · working tree clean

---

## 1. The exact prompt given

> Continue the session on model/excellence. Item 4: persist the fitted estimator with each
> published forecast, per the spec in the item 3 record — save estimators into
> `forecasts/published/<issue_date>/` with library names and versions, a loader, a test that a
> saved model reproduces its published predictions to tolerance, and the storage cost per issue
> stated. Official runs must still refit on current data; this is for reproducing a published
> issue, not serving a stale model. **Give me the plan first and stop — do not implement until I
> approve.** The plan must cover: serialization format and why, how estimators are tied to a
> run_id, sklearn/lightgbm version-pinning and what happens on a version mismatch at load time,
> disk footprint per issue and projected growth, and whether anything in the estimator could carry
> data that shouldn't be persisted. Session logging:
> `docs/sessions/2026-08-10-item-4-persisted-estimators.md`, same five sections.

---

## 2. The plan

Every number below was measured in this session against the real champions, not estimated. The
measurement commands and their raw output are in §3.

### 2.0 Two premises in the brief that measurement contradicts

Stated before the design, because both change what the deliverable can honestly claim.

**(a) "tied to a run_id" cannot mean a logged `run_id`.** `experiments/log.csv` has 153 rows and
none of them is the published forward run — the forward path never writes a log row. The `run_id`
each recipe carries (`revenues-lgbm-l1-ws3-v1` →
`20260805T090357.040566_Revenues_LightGBM_L1_ratio`) identifies the **DEV selection fit**: trained
on TRAIN ≤2023, scored on DEV 2024. The published estimator is a *different fit* — same recipe,
all history through 2025-08-06. Binding the blob to that `run_id` would assert an identity that
does not hold. §2.2 proposes what to use instead.

**(b) Persisting all 60 estimators into git costs 2.79 GB/year at a daily cadence, permanently.**
`forecasts/published/` is deliberately git-tracked (`.gitignore:30` un-ignores it). Measured
footprint is 11.36 MB per issue against a current issue size of 16 KB — a ~700× increase, in
binary blobs that git cannot deltify and that no later commit can remove. §2.4 asks for a decision
rather than assuming one.

### 2.1 Serialization format: joblib (pickle protocol 5), compress=3

**Why not the alternatives:**

| Format | Why rejected |
|---|---|
| ONNX | No converter covers `ScaledRegressor` (a project-local wrapper), and ONNX runs in float32 — it changes the numbers, which defeats the whole point of a reproduce-to-tolerance test |
| LightGBM native `model_to_string()` | Covers the booster only, loses the sklearn wrapper's feature-name ordering, and cannot represent `HistGBDT_L1` — the balance champion — at all. So it covers 2 of 3 targets. **And it is not even a size win:** measured on a comparable fit, 273.7 KB raw / 109.3 KB gzipped vs 115.6 KB for the joblib pickle |
| `skops` | Avoids code execution on load, but has no support for lightgbm or catboost |

joblib is the only option that round-trips **every** estimator in the 13-model pool plus the
project's own wrapper, exactly, with no numeric change.

**Pickle executes code on load. The mitigation is integrity, not trust:** each blob's SHA-256 goes
in a tracked manifest, and the loader verifies the digest *before* unpickling and refuses on
mismatch, with no override. That reduces the exposure from "arbitrary code from any file on disk"
to "arbitrary code only if the attacker can already write to our git history".

**Layout:**

```
forecasts/published/<issue_date>/
  estimators/
    manifest.json                      # tracked: digests, versions, fit identity
    revenues/h5_point.joblib           # blob
    revenues/h5_q10.joblib  ...
```

### 2.2 How an estimator is tied to its fit

A **`fit_id`** minted at publish time, not a borrowed `run_id`:

```
fit_id = "<issue_date>/<target-slug>/h<h>/<point|q10|q50|q90>"
```

Each manifest entry records: `issue_date`, `recipe_id`, `selection_run_id` — **named that way so
it cannot be misread as this fit's id** — plus `data_sha256`, `git_sha`, `calendar_version`,
`n_train_rows`, `n_features`, ordered `feature_names`, `target_transform`, library and version,
and the blob SHA-256. That is the identity set already in `provenance.json`, so the manifest
*points at* the issue provenance rather than duplicating and risking divergence.

I recommend **not** writing a forward row into `experiments/log.csv` to manufacture a run_id: every
row there means "an evaluated experiment with a DEV metric", and a forward fit has no metric, so it
would be the first row with an empty `dev_mae` — degrading the log's meaning to gain a label.

### 2.3 Version pinning and mismatch behaviour

Measured environment: python 3.13.11, scikit-learn 1.8.0, lightgbm 4.6.0, joblib 1.5.3,
numpy 2.4.4, pandas 3.0.2.

**Measured: sklearn does not refuse a wrong-version load.** A pickle stamped 1.3.2 loaded under
1.8.0 returned a working estimator and predicted `[3.33333333]`. The only signal is an
`InconsistentVersionWarning` — and a warning is invisible in a CLI or Streamlit run. Relying on it
would let a wrong-version load silently produce different numbers under a published label.
LightGBM's text dump carries `version=v4`, but a pickled `LGBMRegressor` offers no cross-major
guarantee.

So **the loader does the check itself**, in three tiers:

| Condition | Behaviour |
|---|---|
| Blob digest ≠ manifest | Refuse always. No override. This is integrity, not compatibility |
| Versions differ from manifest | Raise `EstimatorVersionMismatch` naming each differing package and both versions. Loadable only via explicit `allow_version_mismatch=True`, which marks the result unverified — and the reproduce test refuses that path |
| Exact match | Load silently |

One small addition needed: `provenance.describe_environment()` currently records numpy, pandas,
scikit-learn, statsmodels, xgboost, lightgbm, torch — **not joblib**, which is the format's own
reader. Add it.

**Reproduce test:** load, rebuild the design matrix from the canonical data file whose SHA-256 the
provenance already records, predict the stored origin row, compare to the published `p10`/`p50`/`p90`
in `forecast.csv`. On an exact version match, same platform, this should be bitwise identical — I
will **measure the actual round-trip residual and set the tolerance from it** rather than asserting
a number now.

### 2.4 Disk footprint — measured

60 estimators per issue: 3 targets × 5 horizons × (1 point + 3 quantile). `run_forward` fits one
model per horizon, and `_fit_predict_point` currently returns a `float` — **the fitted estimator is
discarded today**, so capturing it is a real change to that function, not just an extra write.

| | joblib compress=3 |
|---|---|
| Revenues (LightGBM_L1, ratio) | 5.58 MB |
| Expenditure (LightGBM_L1, raw) | 4.47 MB |
| State budget balance (HistGBDT_L1, raw) | 1.31 MB |
| **Per issue** | **11.36 MB** |
| Weekly issues | 0.58 GB / year |
| Daily issues (252 business days) | 2.79 GB / year |

Current issue directory: 16 KB. Current `.git`: 13 MB.

**Dropping the quantile models saves almost nothing** — the point models dominate (~9.2 MB of the
11.36 MB), so "persist fewer estimators" is not the lever it looks like. Nor is the file format
(§2.1). The only real levers are where the blobs live and how long they are kept:

| Option | Consequence |
|---|---|
| **A — blobs untracked, manifest tracked** | `.gitignore` the `estimators/` subdirectory; the SHA-256 of every blob stays in git forever, so the *claim* remains auditable from a clone and a swapped blob is still detectable. Reproduction depends on the blob surviving locally, and a missing blob reads as an honest "not available" rather than silence |
| **B — blobs tracked in git** | Full reproduction from a bare clone. 0.58 GB/yr weekly, 2.79 GB/yr daily, unremovable without a history rewrite. Defensible only at weekly-or-rarer cadence |
| **C — retention window** | Keep blobs for the last N issues plus any issue still unscored; prune older and record `pruned: true` with the date. Bounded disk, gaps that are declared |

**My recommendation: A + C.** The distinction that makes it consistent with the tracked-CSV
principle: `forecast.csv` **is** the claim and must survive in git; the estimator is only a *means
of re-deriving* it, and a digest in git is enough to prove the blob was not swapped. This does cut
against the retention rule set for the CSVs, so it is your call, not mine.

### 2.5 Data the estimator would carry — yes, and it is Treasury data

**Measured on the fitted Revenues champion.** Its recipe uses `transform=ratio`, so all 20 of its
estimators are `ScaledRegressor` instances, and `ScaledRegressor` stores:

| Attribute | Measured content | Needed at predict time? |
|---|---|---|
| `y_train_ref_` (`target_scaling.py:224`) | the **full training target**: ndarray, len 2406, ~18.8 KB, first value `19622648.27` — actual GEL revenue figures | **No.** Only used as a median-magnitude reference by `sanity_check_prediction_scale` |
| `level` (`target_scaling.py:174`) | the **full trailing-level series**: len 2763, ~107.7 KB (longer than the training set — it spans the whole index) | **Yes.** `ratio` divides by the trailing level at the origin |

Expenditure and balance are `transform=raw`, so they get no wrapper and embed no target series.
The exposure is Revenues-specific.

**Proposed scrubbing, both behaviour-preserving:**

* `y_train_ref_` → replace the 2406-value array with a **1-element array holding `median(|y|)`**.
  `sanity_check_prediction_scale` computes exactly `nanmedian(abs(y_train))`, and the median of a
  one-element array is that element, so the check behaves identically. Deleting the attribute
  outright would `AttributeError` in `predict` (`target_scaling.py:265`). One aggregate statistic
  survives; that is a real if much smaller disclosure, and I would state it in the manifest rather
  than imply the blob is data-free.
* `level` → reduce to `level.loc[[origin]]`, a one-element Series. `_rows_level` only does
  `self.level.reindex(idx)`, and reproduction predicts exactly one origin date. `strict` defaults to
  **True**, so a request for any other origin raises instead of silently `ffill`-ing — which is the
  correct behaviour for a replay artifact.

Plus a test asserting a loaded estimator carries no full-length training array, so this cannot
regress.

**What cannot be scrubbed, and must be said plainly:** tree ensembles store split thresholds and
leaf values that are *functions of* Treasury figures. A persisted model is not a data-free object.
It must not be treated as third-party-shareable merely because the published CSV is.

**Two further notes:** the reproduce test should rebuild the design matrix from the canonical file
by SHA-256 rather than storing an origin feature row — no extra Treasury data at rest, and
reproduction is tied to the exact input. Separately, the existing published issue records
`git_dirty: true, git_dirty_files: 20`, which already limits how exactly it can be reproduced,
independently of estimators.

### 2.6 Work items, in order

1. `describe_environment()` — add joblib.
2. `_fit_predict_point` / `_fit_predict_quantiles` — return the fitted estimator alongside the
   prediction (currently discarded).
3. `backend/estimator_store.py` — `save_estimators()` (scrub → write → digest → manifest) and
   `load_estimator()` (digest check → version check → unpickle).
4. Wire into `publish_official` / `write_artifacts`.
5. `.gitignore` for the blob directory (pending the §2.4 decision) + retention pruning.
6. Tests: reproduce-to-tolerance; digest-mismatch refusal; version-mismatch refusal; no
   full-length training array in a loaded estimator; pruned issue reads as "not available".
7. Measured storage cost restated in the session record from the real written directory.

### 2.7 The two decisions I need from you

1. **§2.4** — option A, B, or C (I recommend A + C).
2. **§2.0(a)** — confirm `fit_id` + `selection_run_id` labelling, rather than binding the blob to a
   DEV `run_id` that describes a different fit.

---

## 3. Commands run and their real output

Measurement only — no source file was modified.

**Per-issue footprint and fit topology** (`backend/.venv/bin/python`, real champions, real data):

```
Revenues                 transform=ratio  5 horizons x (1 point + 3 quantile) =   5.58 MB
Expenditure              transform=raw    5 horizons x (1 point + 3 quantile) =   4.47 MB
State budget balance     transform=raw    5 horizons x (1 point + 3 quantile) =   1.31 MB

TOTAL PER ISSUE, 60 estimators, joblib compress=3 : 11.36 MB
  weekly issues  -> 0.58 GB / year
  daily issues   -> 2.79 GB / year  (252 business days)
```

**What a single fitted champion contains and costs:**

```
Revenues  (LightGBM_L1, transform=ratio)  n_train=2406 rows, 47 features
   joblib raw=  2327.5 KB   joblib compress=3=   904.4 KB   pickle=  2305.8 KB
   *** ScaledRegressor carries TRAINING DATA:
       y_train_ref_ : ndarray len=2406  ~18.8 KB  first=19622648.27
       level        : Series len=2763  ~107.7 KB

Expenditure  (LightGBM_L1, transform=raw)  n_train=2406 rows, 36 features
   joblib raw=  2126.1 KB   joblib compress=3=   817.3 KB   pickle=  2126.1 KB
   no wrapper; inner estimator only

State budget balance  (HistGBDT_L1, transform=raw)  n_train=2406 rows, 45 features
   joblib raw=   398.8 KB   joblib compress=3=   174.9 KB   pickle=   391.3 KB
   no wrapper; inner estimator only
```

**Version-mismatch behaviour** — a state stamped 1.3.2 loaded under 1.8.0:

```
state stamp at dump time: 1.8.0
__setstate__ raised?  NO -> predict [3.33333333]
warnings caught: 1
   InconsistentVersionWarning: Trying to unpickle estimator Ridge from version 1.3.2 when
   using version 1.8.0. This might lead to breaking code or invalid results.
   .original_sklearn_version = 1.3.2
```

**Format size comparison** (comparable LightGBM fit, 2406×47, 100 trees):

```
   joblib compress=3 pickle :    115.6 KB
   native model_to_string() :    273.7 KB
   native, gzipped          :    109.3 KB
```

**Git tracking and current sizes:**

```
$ git check-ignore -v forecasts/published/2025-08-06/forecast.csv
NOT ignored -> tracked
   tracked: forecasts/published/2025-08-06/forecast.csv
   tracked: forecasts/published/2025-08-06/gates.json
   tracked: forecasts/published/2025-08-06/manifest.json
   tracked: forecasts/published/2025-08-06/provenance.json
.gitignore:30  !forecasts/published/
16K   forecasts/published/2025-08-06
13M   .git
```

**run_id linkage:**

```
log rows: 153        rows mentioning "forward" anywhere: 2
Revenues     run_id=20260805T090357.040566_Revenues_LightGBM_L1_ratio     approved_by=None
Expenditure  run_id=20260805T090411.471720_Expenditure_LightGBM_L1_raw    approved_by=None
State budget balance run_id=20260805T090416.281434_State_budget_balance_HistGBDT_L1_raw approved_by=None
```

**Installed versions (backend interpreter):** python 3.13.11 · scikit-learn 1.8.0 · lightgbm 4.6.0
· joblib 1.5.3 · numpy 2.4.4 · pandas 3.0.2

---

## 4. Verdict

**Plan delivered; implementation not started, as instructed.** All five required points are covered
and every figure is measured rather than estimated.

Two premises in the brief did not survive measurement and are reported rather than worked around:
the published fit has no logged `run_id` (§2.0a), and persisting into a git-tracked directory costs
2.79 GB/year at a daily cadence, permanently (§2.0b). One concrete data-exposure finding: the
Revenues champion's estimators embed 2406 raw Treasury target values and a 2763-point level series,
both scrubbable without changing behaviour (§2.5) — and tree thresholds, which are not.

No source file was modified. Working tree clean at `4cd7dd0`.

---

## 5. Outstanding

* **Blocking:** the two decisions in §2.7 — retention option (A/B/C) and the `fit_id` naming.
* Round-trip tolerance to be set from measurement during implementation, not asserted in advance.
* `level` reduction to a one-element Series to be verified against `_rows_level` on a real loaded
  estimator before it is relied on.
* Unchanged from earlier records: six artifact fields still unfixed (classification delivered,
  decision reserved); master-prompt Part 5 accuracy levers, Part 6 remainder, Part 7 Georgian i18n.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
