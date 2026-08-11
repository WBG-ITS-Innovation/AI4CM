# Quality gate audit

**Date:** 2026-08-11 · **Branch:** `model/excellence` · item 5 part 1

Every gate in the project, what failure mode it catches, what would slip past it, and whether it
fails for the accurate reason. Numbers in this document were measured in the audit session, not
recalled — the measurements are reproduced in
[`docs/sessions/2026-08-11-item-5-quality-gates.md`](../docs/sessions/2026-08-11-item-5-quality-gates.md) §3.

---

## 1. The four verdicts

Leakage, no-signal, persistence-mimicry and coverage must remain four distinct verdicts. A family
withheld for the wrong reason is nearly as bad as one not withheld, because the reason is what
determines the fix: broken features, a different feature set, a different target, or a
recalibration.

| Verdict | Decided by | Reaches the reader as |
|---|---|---|
| **Leakage** | `origin_date >= target_date`, `check_feature_leakage` | `leakage flag raised` |
| **No signal** | `signal_sentinel`, `MIN_SIGNAL_RATIO = 1.5` | `no signal beyond shuffled targets (ratio 0.83)` |
| **Persistence-mimicry** | `shift_diagnostic_horizon_aware` OR `detect_lagged_copy` | `forecast is persistence-like (shift diagnostic)` |
| **Coverage** | `quantile_quality_gate`, `conditional_coverage_gate` | `intervals miscalibrated: coverage 43.2% outside [70%, 90%] (nominal 80%)` |

**The fourth verdict did not exist before this audit.** A coverage failure reached
`daily_summary.gate_reasons()` only as `run_status=FAILED_QUALITY` or the generic
`quality gate failed (per the family's integrity report)`. An E_QUANTILE family withheld because
its 80% band covered 43% of outcomes was therefore **indistinguishable from one withheld for poor
skill** — despite needing a completely different response. Fixed by
`_coverage_failure_reasons()`, which reads the nominal level as data and names the measured
coverage. `test_failure_mode_distinctness.py` now pins all four, including a case where all four
fire at once and must produce four separate lines.

---

## 2. Gate-by-gate

### 2.1 Forward-only assertion — `forward_forecast.assert_forward_only`

* **Catches:** a forward artifact containing any date at or before the data end — on this project
  specifically, an accidental evaluation on the sealed 2025 holdout dressed up as a production run.
* **Slips past:** nothing about *feature* recency; a forecast for a legitimate future date built
  from a feature that peeked is a leakage question, not this one.
* **Accurate reason:** yes. Raises naming the offending date and the data end.

### 2.2 Alignment — `validate_alignment_step_based`

* **Catches:** `target_date != origin_date + h` positions in the modelling index. Distinguishes a
  genuine misalignment from a Georgian holiday gap, because it counts index positions rather than
  calendar days.
* **Slips past:** correct alignment *within the wrong index*. Mitigated by
  `test_dl_alignment_integrity.test_the_checker_is_called_against_the_index_sequences_were_built_from`.
* **Accurate reason:** yes, and it is tri-state — a run that could not perform the check writes
  `alignment_checked: false` with the error rather than a verdict. This was the audit field fixed
  last session: it used to be a literal `True`.

### 2.3 Feature leakage — `check_feature_leakage`

* **Catches:** feature *names* containing a negative shift, and any feature whose correlation with
  `y(t+h)` exceeds 0.99.
* **Slips past:** **this is the weakest instrument in the set.** A leaky feature at r = 0.98 is
  invisible. A leak assembled from a *combination* of features is invisible, since the test is
  univariate. And on a level target the check is prone to the opposite error: a random-walk level
  correlates ~0.98 with itself at almost any lag, so an entirely legitimate lag feature sits just
  under the threshold — the margin between "innocent lag" and "flagged leak" is about 0.01.
* **Accurate reason:** yes *when it fires* — it names the feature and the correlation. Its problem
  is miss rate, not phrasing.

### 2.4 Signal sentinel — `signal_sentinel`, `MIN_SIGNAL_RATIO = 1.5`

* **Catches:** a feature set that does not predict the target. Fits the same probe twice, once on
  true targets and once on shuffled ones, and requires the shuffled error to be ≥1.5× worse.
* **Slips past:** a feature set that carries signal about the *level* through autocorrelation while
  adding nothing beyond persistence. This is not hypothetical — it is the project's central open
  finding: the two flow targets sit at 1.00–1.23 across 71 logged runs under three probes.
* **Accurate reason:** yes, since M-5. A low ratio reports **no signal**, never leakage; the
  docstring states that leakage drives the ratio *up*. Below 1.0 it says so explicitly ("shuffling
  the targets improved held-out error"). Insufficient data returns `signal_detected: None` —
  "not measurable" is never recorded as a pass.

### 2.5 Leakage sentinel — `leakage_sentinel`

* **Catches:** nothing of its own. It is a deprecated alias that delegates to `signal_sentinel`.
* **Slips past:** everything its name implies. The name is the hazard.
* **Accurate reason:** yes — it never asserts leakage, and the deprecated shim re-exports the same
  object rather than copying it. Now mutation-guarded in both directions.

### 2.6 Persistence-mimicry — `detect_lagged_copy` (default window)

* **Catches:** a model whose predictions correlate better with a shifted target than the true one
  **and** which fails to beat the lag-1 baseline. Both conditions must hold:
  `(corr_best - corr_at_0) > 0.05` and `mae_pred >= 0.99 * mae_lag1`.
* **Slips past — two measured blind spots, one per condition:**

  | Input | Flagged? | Why |
  |---|---|---|
  | lag-1 copy of a **level** series | **no** | correlation margin only **+0.018**, under the 0.05 requirement — a random walk correlates 0.982 with itself at shift 0 |
  | h=5 copy of a **flow** series | **no** | a flow has no autocorrelation, so the shifted copy is genuinely *worse* than lag-1 (**0.979** < 0.99) and condition 2 fails |
  | lag-1 copy of a **flow** | yes | margin +1.081 — what the detector was built for |
  | h=5 copy of a **level** | yes | margin **+0.0601** against a 0.05 threshold — caught, but by 0.01 |

  So on the project's actual target shapes this detector is close to the edge of its power in three
  of four cases. It is **not** currently a live hole for B_ML, C_DL or E_QUANTILE, because
  `daily_summary.family_shift_flag` ORs it with the horizon-aware diagnostic below, which catches
  the h-step case cleanly (`best_shift = -5`, "Persistence-like"). It **is** a live hole for any
  family that writes no shift fields — see §3.
* **Accurate reason:** yes when it fires, naming the shift, both correlations and both MAEs.

### 2.7 Persistence-mimicry — `shift_diagnostic_horizon_aware`

* **Catches:** the shift that minimises MAE, over `±max(h+5, 10)`, always including `-h` and
  `-(h+1)`. Separates `-h` ("persistence-like") from `-(h+1)` ("missing lag_0", flagged strongly)
  from `0` (clean).
* **Slips past:** a model that is a *weighted blend* of persistence and signal — the best single
  shift stays 0 while much of the forecast is still carried.
* **Accurate reason:** yes, and this is the one that distinguishes two failure modes a single
  "shifted" flag would merge.

### 2.8 Overfit gate — `OVERFIT_GATE_RATIO = 3.0`, `select_best_model`

* **Catches:** a train/DEV MAE ratio above 3 — including `inf` from a train MAE of exactly 0, which
  is what ExtraTrees at unlimited depth produces by memorising. Such a model is **excluded from
  being crowned**, not deleted from the leaderboard.
* **Slips past:** a model that overfits within ratio 3. And, deliberately, a model with **no**
  recorded ratio: a missing ratio means "not measured", not "failed", so it is not blamed. The
  original defect lived exactly there — a train MAE of 0 was recorded as a *missing* ratio, turning
  the loudest possible overfit signal into a free pass. Now mutation-guarded.
* **Accurate reason:** yes, and it is the right *severity* — excluding from selection rather than
  failing the run, since an overfitting model in the leaderboard is evidence, not a defect.

### 2.9 Skill gate — `_QUALITY_GATE_SKILL_PCT = 5.0`

* **Catches:** skill below 5% against the h-step persistence ruler.
* **Slips past:** skill obtained by mimicry (§2.6/2.7 own that), and skill measured on the wrong
  window — which was a real bug, fixed by `Config.eval_start`/`eval_end` after DEV figures turned
  out to include 2025.
* **Accurate reason:** yes, quoting measured and threshold as numbers.

### 2.10 Quantile gate — `quantile_quality_gate`

* **Catches:** skill below threshold, coverage outside the band, **or** coverage not measurable.
  Three separately phrased reasons, so a coverage failure is never read as a skill failure.
* **Slips past:** **conditional** miscalibration. An interval covering 80% overall while covering
  55% on the largest-magnitude tercile passes this gate — marginal coverage is all it tests. That
  is what `conditional_coverage_gate` exists for, and it is **not wired into this gate** (§3).
* **Accurate reason:** yes, and materially better after this session. The band used to be a
  hardcoded `(0.70, 0.90)` built around an assumed nominal 80%, unconnected to the fitted alphas.
  Measured consequences of that, both directions:

  | Interval | Coverage | Judged at its own nominal | Judged at an assumed 80% |
  |---|---|---|---|
  | 90% (alphas 0.05/0.95) | 91.0% | **pass** | **fail** — "outside [70%, 90%]" |
  | 90% (alphas 0.05/0.95) | 72.0% | **fail** | **pass** |

  So the assumed band would have withheld a correctly calibrated interval and published a broken
  one. The level now travels with the number as `coverage_nominal`.

### 2.11 Conditional coverage — `conformal.conditional_coverage_gate`

* **Catches:** coverage that holds on average but fails within magnitude or volatility terciles —
  the failure that matters most to a treasury, since the days worth forecasting are the large ones.
  Currently fails 2 of 3 targets, which is why CQR is not claimed as solved.
* **Slips past:** terciles are marginal, not joint — an interval could pass on magnitude terciles
  and volatility terciles separately while failing on high-magnitude *and* high-volatility days
  together. Small tercile counts also make the estimate noisy.
* **Accurate reason:** yes, reporting per-tercile numbers rather than a single verdict.

### 2.12 Gate verdict contract — `read_gate` / `write_gate`

* **Catches:** the contradiction that produced X9 — B_ML wrote the inverted `quality_gate_failed`
  while `gate_reasons` read only `quality_gate_passed`, so a run that **failed** its gate was
  reported as passing. One reader, one writer, legacy key derived rather than set independently.
* **Slips past:** a family that writes neither key — correctly resolved to `None`, "never
  verified".
* **Accurate reason:** yes. Tri-state throughout: passed / failed / never verified. `None` must
  never render as a pass, and does not.

### 2.13 Transform scale — `sanity_check_prediction_scale`

* **Catches:** an un-inverted or doubly-inverted target transform, by comparing predicted magnitude
  against training magnitude. Runs at fit time on the in-sample batch, where a units error is
  systematic.
* **Slips past:** batches under `MIN_SANITY_BATCH = 30`, deliberately — a median over a handful of
  rows says nothing about units, and this check firing on a one-row holiday-zeroed prediction was a
  real false positive.
* **Accurate reason:** yes, naming both magnitudes and the factor.

### 2.14 Registry and log integrity

* `registry.py` refuses a withheld recipe with no plain-language reason, and nothing may imply
  approval while `approved_by` is null — currently null on all three recipes.
* `verify_log_integrity` requires a per-run JSON for every `experiments/log.csv` row: 153 rows, ok.
* `estimator_store` (item 4) verifies a SHA-256 before unpickling and refuses on a version
  mismatch, because scikit-learn only warns.

---

## 3. Findings, in priority order

1. **Coverage was not a distinct verdict.** Fixed this session. Was the highest-severity finding
   because it silently violated the four-verdict requirement.
2. **The nominal coverage level was assumed, not read.** Fixed this session (audit field #2). A
   reconfiguration of `Config.quantiles` would have produced wrong verdicts in both directions.
3. **`conditional_coverage_gate` is not wired into `quantile_quality_gate`.** Marginal coverage can
   pass while conditional coverage fails, and only the marginal one gates. **Not fixed** — wiring
   it changes published verdicts on real runs, which is a decision, not a cleanup.
4. **`check_feature_leakage` is a weak instrument.** Name-matching plus a single 0.99 univariate
   correlation. Misses r = 0.98, misses multivariate leaks, and on level targets sits ~0.01 from
   flagging innocent lags. **Not fixed** — a stronger test needs a design decision, not a threshold
   tweak.
5. **`detect_lagged_copy` has two measured blind spots** (§2.6), currently masked by the OR with the
   horizon-aware diagnostic. **A family that writes no shift fields has no effective
   persistence-mimicry check at h=5** — `pipeline_shift` returns no flag when the fields are absent,
   leaving only the detector that misses those cases. A_STAT is that family.
6. **`DESCRIPTIONS` contains two models no pool exposes** (`ETS`, `Theta`), so two descriptions are
   unreachable from the Models page. Cosmetic, but it is why `len(DESCRIPTIONS)` must never be
   presented as a model count.

---

## 4. Model counts — what to present to a client

`model_pool()` returns **16**; `available_models()` returns **13**. Both are correct and they count
different things:

* **13** — the B_ML point-model pool. This is the set a champion is selected from, and every
  measured claim in the project (skill, MASE, sentinel, gate verdicts) comes from a member of it.
* **+3** — `GBQuantile`, `ResidualRF`, `LGBMQuantile` from E_QUANTILE. These produce *intervals*,
  not point forecasts. They are not champion-eligible and are never crowned "best model".
* **= 16** in the page's pool.
* **18** in `DESCRIPTIONS`, which additionally describes `ETS` and `Theta` from A_STAT — models the
  pool never enumerates.

**Recommendation: present 13, with the 3 named separately, and never a single headline "16".**
"13 point models compete to be the champion for each target; prediction intervals come from 3
quantile methods" is both true and the composition a reader needs. Summing them implies sixteen
candidates for the same job, when three of them do a different job.

"16" is also not a project total: A_STAT and C_DL models sit outside it (A_STAT's most recent
artifact evaluated ETS against the persistence baseline; C_DL offers five architectures — LSTM,
GRU, DCNN, Transformer, MLP). Any single number is therefore a claim about *scope*, and the scope
has to be stated with it.

The composition is now pinned by
`test_regression_mutations.test_the_model_counts_are_pinned_so_a_headline_number_cannot_drift`, so
adding a model forces a deliberate update rather than a silent change to a number shown to a client.
