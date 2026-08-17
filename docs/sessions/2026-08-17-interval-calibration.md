# Session — interval calibration for the 80% prediction bands

**Date:** 2026-08-17 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`); the agent
repo was not touched.
**Data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
(`master_daily_clean_treasury.csv`, 3,867 rows, 2015-01-05 → 2025-08-06)
**Holdout reads spent on selection: 0.** Every sealed-window figure below is a *report* over rows
that earlier logged runs already materialised to disk. No model was refitted on the holdout, and
nothing was chosen from a holdout number.

---

## The headline, stated first

**The defect this session was called to fix is mostly a measurement artifact.** The documented
caveat — *"nominal 80% captures ~50% on the largest third of days"* — comes from a metric that
groups days by **how they turned out**. Conditioning coverage on the outcome depresses it
mechanically, for a correct band as much as for a broken one.

What is genuinely broken is narrower, and three separate things:

| | Defect | Status |
|---|---|---|
| **A** | The conditional-coverage metric buckets on the realised actual | **fixed** |
| **B** | Real marginal under-coverage, concentrated on the stock target | **instrument built, measured, left off by default** |
| **C** | The coverage gate was never fed on the publication path, and its absence was hidden | **fixed** |
| **D** | E_QUANTILE read the sealed window without going through the ledger | **fixed** |
| **E** | *(found while testing)* the conformal calibration slice was not embargoed from the evaluation origin | **fixed** |

---

## 1. Diagnosis

### 1.1 The control that settles it

I did not argue this from theory. I built a band that is correct by construction — actuals drawn
from a known distribution, band set to that distribution's exact 10th and 90th percentiles, so its
true coverage is **80% by definition** — and scored it the way the Lab does.

| Perfect 80% band | Overall | Top decile by \|actual\| | Top decile by \|forecast\| |
|---|---:|---:|---:|
| constant noise | 79.9% | **69.0%** | 79.7% |
| noise varies day to day | 80.0% | **37.8%** | 79.5% |

A flawless band reads **37.8%**. Grouping by the actual selects the days whose outcome landed in
its own upper tail — days *defined* by having exceeded the forecast — so no band can score 1−α
there. Pinned in `test_intervals.py::test_tercile_split_never_buckets_on_the_outcome` and
`test_coverage_calibration.py::test_large_day_coverage_is_bucketed_on_the_forecast_not_the_outcome`
so it cannot be reintroduced.

### 1.2 The same reversal on real bands

Sealed window (TEST, 2025-01-01 → 2025-08-06), 31 model×target cells, four conditioning variables:

| Basis for "a big day" | Median top-decile coverage |
|---|---:|
| \|actual\| — what the metric used, **not knowable at forecast time** | **43.8%** |
| \|forecast\| | 87.5% |
| \|predicted change\| — the right notion for a stock | 87.5% |
| trailing volatility | 92.4% |

Three independent knowable variables agree; the outlier is the one in use.

**Corroborating evidence — the misses are one-sided.** Across all 31 cells, big-day misses through
the *lower* edge total **one**. Everything else bursts through the top. A band that is genuinely
too narrow misses both ways; a band scored on outcome-selected days misses only upward.

### 1.3 The 80% bands, measured

| Target | Model | n | Overall | by \|actual\| | by \|forecast\| | 95% CI |
|---|---|---:|---:|---:|---:|---|
| Revenues | **GBQuantile** | 205 | **78.0%** | 28.6% | **81.0%** | [62, 95] |
| Revenues | ResidualRF | 205 | 69.8% | 23.8% | **38.1%** | [19, 57] |
| State budget balance | **GBQuantile** | 156 | **73.7%** | 56.2% | 93.8% | [81, 100] |
| State budget balance | LGBMQuantile | 156 | **66.7%** | 43.8% | 50.0% | [25, 75] |
| State budget balance | ResidualRF | 156 | **53.8%** | 37.5% | 50.0% | [25, 75] |

**GBQuantile on Revenues — the band the client actually receives — is well calibrated**: 78.0%
overall and 81.0% on big days against an 80% nominal.

Two corrections to the brief. **Expenditure has no sealed-window interval artifact**, so it cannot
be measured without a fresh fit, which the agreed scope excluded. And **64.0% does not appear
anywhere in the repo**; the nearest real figures are LGBMQuantile at 66.7% overall and `0.65` in
the 2026-08-12 leaderboard. I reported what I measured rather than repeating the number.

### 1.4 Why I did not do what was asked

Two of the three candidate fixes — magnitude-conditional variance widening and quantile-level
adjustment — would have been tuned against the broken metric. To reach 80% on outcome-selected days
I would have had to inflate bands past correctness with no stopping point, because the achievable
ceiling there is ~38%. Chasing the number would have damaged the product. This was raised before
writing code and the scope was re-approved on that basis.

---

## 2. What changed

### `backend/conformal.py`
* `conditional_coverage_gate`'s `magnitude` parameter is renamed **`magnitude_at_origin`** — a
  rename, not a doc note, so every existing call site had to be looked at. It surfaced six.
* New `OutcomeConditionedCoverageError` and `_refuse_outcome_basis`: the gate now *refuses* an
  actual-derived basis on either axis rather than trusting its caller. Deliberately not
  overridable — there is no legitimate reading of "the largest days" that means "largest as it
  turned out".
* Output carries `bucketing_basis`, so a corrected figure is distinguishable on sight from a
  pre-fix one. A reader finding an old 9.6% beside a new 87.5% must see that they answer different
  questions, not that the band changed.

### `frontend/intervals.py`, `frontend/pages/01_Dashboard.py`
* `coverage_by_tercile` buckets on `yhat_p50` → `y_pred` → `origin_value`, never `y_true`. New
  `magnitude_basis()` reports which was used, or returns `None` so the page can say *not
  measurable* instead of falling back to the actual.
* The dashboard heading no longer calls this "the project's biggest known weakness"; it names the
  basis instead.

### `backend/coverage_report.py` *(new)*
The single reader for per-family coverage. Each family is scored against **its own** advertised
level (E_QUANTILE 80%; B_ML/C_DL 90% from `nominal_pi`) — scoring an 80% band against 90%
manufactures a ten-point defect. A_STAT emits `y_lo`/`y_hi` and never fills them (measured: 0 of
156 non-null in both logged runs), so it reports *no intervals* rather than 0% coverage.

### `backend/publication_gates.py`
* New `Measured.has_intervals`. Coverage was excluded from `unmeasured_gates` unconditionally, so
  "publishes a band nobody scored" was reported as **nothing at all** — which is exactly the state
  all three recipes were in. The exclusion now applies only when a model genuinely has no
  intervals.

### `backend/e_quantile_daily_pipeline.py`
* `Config.cqr` (default **off**) applies split-conformal correction to the outer quantiles,
  calibrated on a causal slice of each fold's own training data.
* `_predict_quantiles` extracted as the single definition of "how this model predicts", used by
  both the fold loop and the calibration step — a correction calibrated from one code path and
  applied to another's band is not a calibration.
* Refuses rather than silently skipping: no finite width, impossible geometry, or a correction that
  would **invert** the band all return the band untouched with the reason. A negative width is
  *not* clamped — the band being wider than it needs to be is a real finding.
* `require_test_access(..., PURPOSE_REPORT)` now called on the reporting path (defect D).

---

## 3. Defect E — a leak my own test caught

Writing the test for "calibration never reads test-period data" exposed a genuine leak in the code
I had just written.

`causal_calibration_split` puts an *h*-row gap between the **fit** and **calibration** slices, so
the model has not seen the calibration answers. It does not — and cannot — know what the
calibration is being used to correct. The calibration slice ran to the last row of `X_tr`, one row
before the fold's first origin, and that row carries a target *h* rows further on:

```
X_tr rows 0..699    first eval origin = row 700
calibration rows  525..699
last calibration row 699 -> its target is at row 704
is that target known at origin 700 ?  False
```

So the correction was measured partly against answers unavailable when the band was issued — the
precise defect this session exists to remove. Fixed by dropping the last *h* rows of the training
block before splitting. Cost of the fix, measured: essentially nil (ResidualRF 86.3% → 85.9%,
widths marginally tighter), because it drops 5 of ~640 calibration rows.

**A note on what the test asserts.** `evaluation_windows` states that "TRAIN is a floor, not a cap
… a fold predicting 2025-06-01 legitimately trains on everything up to 2025-05-31. What never
happens is training on data at or after the origin it is predicting from." My first version
asserted "no test-dated row ever" and failed — correctly, because rolling origins legitimately
expand training into later windows. The invariant is **causality**, not window purity, and the test
now asserts that: every calibration row's target must have happened by the origin whose band it
corrects. A second test covers the window-purity case (scoring DEV) where the two coincide.

---

## 4. CQR, measured honestly

DEV (2024, n=262) — a selectable window, so no holdout was spent. Nominal 80%.

**State budget balance**

| Model | Overall before → after | Top decile (forecast) before → after | Mean width before → after |
|---|---|---|---|
| GBQuantile | 65.6% → **80.5%** | 66.7% → 77.8% | 396M → 614M (**+55%**) |
| LGBMQuantile | 42.4% → 68.7% | 40.7% → 74.1% | 360M → 600M (+67%) |
| ResidualRF | 49.6% → 85.9% | 40.7% → 70.4% | 195M → 558M (+186%) |

**Revenues**

| Model | Overall before → after | Top decile (forecast) before → after | Mean width before → after |
|---|---|---|---|
| GBQuantile | 74.0% → 84.0% | 37.0% → 44.4% | 108M → 133M (+23%) |
| LGBMQuantile | 67.2% → 78.6% | 40.7% → 44.4% | 101M → 129M (+28%) |
| ResidualRF | 69.8% → 85.1% | 14.8% → 33.3% | 87M → 152M (+74%) |

**Reading this honestly.** The marginal guarantee works exactly as advertised — GBQuantile on the
stock target lands at 80.5% against an 80% nominal. It is not free: bands widen 23–186%. And it is
not a cure-all — LGBMQuantile still undershoots at 68.7%, ResidualRF overshoots to 85.9%, and on
Revenues the top-decile-by-forecast figure stays low (37.0% → 44.4%, n≈27), which is a residual
*conditional* shortfall that a marginal correction is not designed to fix.

### Why it ships switched off

* The band the client receives is **Revenues via `forward_forecast.py`**, and it measures 78.0%
  overall / 81.0% on big days against an 80% nominal. It does not need correcting; CQR would widen
  a band that is already right.
* The targets that *do* need it — the stock target especially — are **withheld for accuracy**
  (`accuracy_vs_naive`), so correcting their bands changes no published output.
* Turning it on would change published band widths materially, which is a decision to take
  deliberately and visibly rather than as a side effect of a calibration session.

This lands in the same place WS7 did — CQR available, not adopted — but now for a **measured**
reason rather than an assumed one, and with the instrument wired into the pipeline instead of
living only in a study script.

---

## 5. Publication gates re-run

`./backend/.venv/bin/python scripts/rerun_publication_gates.py` → `reports/coverage_gate_rerun.json`

| Target | Registry | Before | After | Changed | Coverage measured | Coverage gate |
|---|---|---|---|:---:|---:|---|
| Revenues | publishable | publishable | **publishable** | no | 78.0% | **pass** |
| Expenditure | withheld | withheld | **withheld** | no | — | not measured |
| State budget balance | withheld | withheld | **withheld** | no | 73.7% | **pass** |

**No verdict changed.** What changed is that coverage is now *measured* rather than absent:
`coverage` left the `unmeasured_gates` list for Revenues and the stock target, and for Expenditure
it is reported as a stated gap instead of silence.

A bug worth recording: my first version of this script picked the coverage row with the most
scored rows, which gated **Revenues against C_DL's `DCNN` band** and the stock target against
B_ML's `CatBoost_L1` — both healthy, neither the band the recipe ships. It would have reported a
pass for exactly the two targets whose own quantile bands measure worst. `coverage_for_publication`
now requires the recipe's declared `interval_model` and returns `None` rather than substituting a
different model; a test pins it.

**Two things a reader should not be comforted by.** The stock target passes at 73.7% against an 80%
nominal only because the gate's band is ±10pp — a genuine 6.3pp shortfall the tolerance absorbs,
and the one CQR would fix. And the brief described the stock target as *withheld as forecast*; the
registry says **`withheld`**, decided by `accuracy_vs_naive`, which sits above coverage in severity.
Nothing in this session moved it.

---

## 6. Tests

`852 passed, 4 skipped` across `backend/tests` and `frontend/tests`.

New — `backend/tests/test_coverage_calibration.py` (18 tests):

* per-family coverage is reported, with family/target/model on every row, each against its own
  advertised level;
* a family with empty interval columns reports *no intervals*, never 0%;
* large-day coverage is bucketed on the forecast — a by-construction-correct band must read as
  correct;
* publication coverage matches the recipe's own interval model, and prefers a single-window
  measurement over a pooled one;
* reported coverage may not be used to choose (`purpose="selection"` raises);
* **calibration never uses a row whose target postdates the origin it serves** — end to end on the
  real pipeline, evaluating into the sealed window;
* **calibration stays out of the holdout when scoring DEV** — the literal window-purity check,
  asserted where it is the right requirement;
* a correction that would invert the band is refused;
* the pipeline logs its holdout read as a *report*.

Modified: six `test_conformal.py` gate tests and one `test_intervals.py` test were passing the
actual as the bucketing basis. Their fixtures now build a **genuine** conditional defect keyed to
the forecast — dispersion that climbs with the forecast against a constant band width — so they
still fail a band that deserves to fail, for a reason a forecaster could have seen coming.
`test_publication_gates.py` gained three tests around the `has_intervals` distinction.

---

## 7. What is still open

1. **Expenditure has no interval artifact.** Nothing measured its band because none exists on
   disk. Needs a fresh fit evaluated as a `PURPOSE_REPORT` read — deliberately out of scope here.
2. **The residual conditional shortfall on Revenues** (top decile by forecast, 37.0% on DEV) is
   real and survives the metric fix. CQR is marginal by design and only partly helps. This is the
   fit-time problem WS7 named — a band that does not widen when the forecast says the day is large
   — and it is the honest remaining case for trailing-volatility features. Note the sealed-window
   figure for the same model is 81.0%, so DEV and TEST disagree; that gap is worth understanding
   before acting.
3. **`coverage_for_publication` still picks a model per target rather than being told.** It matches
   on `interval_model`, which is right, but the recipe does not record *which run* produced the
   figure. A recipe pointing at its own run_id would close the last inference.
4. **The ±10pp coverage band absorbs a 6.3pp shortfall.** Worth deciding whether that tolerance is
   the one you want now that coverage is actually measured.
5. **`B_ML` still reads the sealed window without the ledger.** Defect D was fixed for E_QUANTILE;
   `b_ml_pipeline.py` has the same gap.

---

## Reproduction

```bash
./backend/.venv/bin/python scripts/rerun_publication_gates.py
./backend/.venv/bin/python -m pytest backend/tests/test_coverage_calibration.py -q   # 18 passed
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q                 # 852 passed
```
