# Item 5 — quality gates and regression test hardening

**Date:** 2026-08-11 · **Branch:** `model/excellence` · parent `c314ed8`
**Root suite:** 602 passed, 3 skipped, `EXIT=0` · **Frontend suite:** 110 passed, `EXIT=0`
**New tests:** 26 mutation/gap tests + 8 four-verdict tests · **Full audit:** [`reports/gate_audit.md`](../../reports/gate_audit.md)

---

## 1. The exact prompt given

> Continue the session on model/excellence. Item 5: quality gates and regression test hardening.
>
> Part 1 — audit every quality gate. For each one, tell me: what failure mode it catches, what would
> slip past it, and whether it fails for the accurate reason. Leakage, no-signal,
> persistence-mimicry, and coverage failure must remain four distinct verdicts — a family withheld
> for the wrong reason is nearly as bad as one not withheld.
>
> Part 2 — regression tests for the bugs already fixed, so they cannot return silently:
> shuffled-target sentinel inversion, zero train MAE from ExtraTrees memorization, collapsed
> E_QUANTILE intervals, inverted leakage-sentinel semantics, the duplicated persistence baseline,
> and the alignment_ok literal. Where a mutation test is possible (reintroduce the bug, assert a
> test fails), do that rather than only testing the fixed behaviour.
>
> Part 3 — fix audit field #2: the E_QUANTILE quantile level lives only in the key name
> coverage_p10_p90, so a consumer reads it as authoritative while nothing guarantees the key still
> matches the quantiles actually produced. Write the level as data and assert it matches the fitted
> alphas.
>
> Also answer one open question: the Models page offers 16 models while the pipeline pool is 13.
> Tell me what the extra three are and which number is the honest one to present to a client.
>
> Session logging: docs/sessions/2026-08-11-item-5-quality-gates.md with your full narrative
> verbatim, the prompt, plan, real output, verdict, outstanding. Run both suites, show passing
> output, commit the md with the code, walk me through the diff.

---

## 2. Plan

1. **Part 1** — inventory every gate by reading the code, not the docs. For each: failure mode
   caught, what slips, whether the stated reason matches the measured cause. Verify the sharp claims
   with measurements. Write to `reports/gate_audit.md`. Check specifically whether all four verdicts
   are reachable and distinct.
2. **Part 2** — a mutation harness. For each of the six bugs, reintroduce the defect via
   `monkeypatch` and require the *existing* guard to fail. Where the guard is a test function,
   import and call it under the mutation and assert `AssertionError`. Where the guard is production
   logic, compare mutated against real on the same input and require disagreement.
3. **Part 3** — derive the interval from the fitted alphas (`coverage_spec`), write
   `coverage_nominal` / `coverage_lower_quantile` / `coverage_upper_quantile` / `coverage_band` as
   data next to every coverage number, make the gate band follow the nominal level instead of an
   assumed 80%, and refuse to emit the legacy key when it no longer names the alphas.
4. **Open question** — measure the pools rather than reasoning about them, and recommend a framing.

---

## 3. Commands run and their real output

### Both suites

```
$ ./backend/.venv/bin/python -m pytest -q        # repo root
ROOT EXIT=0
........................................................................ [ 95%]
..........................                                               [100%]
602 passed, 3 skipped in 197.74s (0:03:17)

$ cd frontend && ./.venv/bin/python -m pytest -q
FRONTEND EXIT=0
......................................                                   [100%]
110 passed in 5.39s
```

### The mutation suite alone

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_regression_mutations.py -q
..........................                                               [100%]
26 passed in 2.97s
```

### Part 3 — the nominal level, measured in both directions

```
band(0.80): (0.7, 0.9)   band(0.90): (0.8, 1.0)

A well-calibrated 90% interval measuring 91.0% coverage:
  judged against its OWN nominal 90% : (True, [])
  judged against an ASSUMED 80%      : (False, ['coverage 91.0% outside [70%, 90%] (nominal 80%)'])

A miscalibrated 90% interval measuring 72.0% coverage:
  judged against its OWN nominal 90% : (False, ['coverage 72.0% outside [80%, 100%] (nominal 90%)'])
  judged against an ASSUMED 80%      : (True, [])
```

```
spec (0.10,0.50,0.90): {'measurable': True, 'coverage_lower_quantile': 0.1,
                        'coverage_upper_quantile': 0.9, 'coverage_nominal': 0.8,
                        'coverage_key': 'coverage_p10_p90'}
spec (0.05,0.50,0.95): {'measurable': True, 'coverage_lower_quantile': 0.05,
                        'coverage_upper_quantile': 0.95, 'coverage_nominal': 0.9,
                        'coverage_key': 'coverage_p5_p95'}

emit @5/95 : {'coverage_key': 'coverage_p5_p95', 'coverage_nominal': 0.9,
              'coverage_p5_p95': 0.88, 'coverage_band': [0.8, 1.0],
              'legacy_coverage_key_omitted': "coverage key 'coverage_p10_p90' does not match the
              fitted quantiles [0.05, 0.5, 0.95], which describe a 90% interval..."}

assert raises: coverage key 'coverage_p10_p90' does not match the fitted quantiles [0.05, 0.5, 0.95]
```

### Part 1 — `detect_lagged_copy` power, measured on both target shapes

```
A perfect lag-1 copy (inside the detector's window):
  random-walk LEVEL   risk=low   flagged=False best_shift=+1 corr@0=0.9818 corr_best=1.0000 margin=+0.0182 mae_pred/mae_lag1=0.997
  noisy FLOW          risk=high  flagged=True  best_shift=+1 corr@0=-0.0806 corr_best=1.0000 margin=+1.0806 mae_pred/mae_lag1=0.998

A perfect h=5 copy (the production horizon):
  random-walk LEVEL   risk=high  flagged=True  best_shift=+3 corr@0=0.8994 corr_best=0.9595 margin=+0.0601 mae_pred/mae_lag1=2.261
  noisy FLOW          risk=low   flagged=False best_shift=-2 corr@0=-0.0515 corr_best=0.1162 margin=+0.1676 mae_pred/mae_lag1=0.979

required to flag: (corr_best - corr@0) > 0.05  AND  mae_pred >= 0.99*mae_lag1
```

```
=== shift_diagnostic_horizon_aware on a perfect h=5 copy of a level series ===
  best_shift     : -5
  interpretation : Persistence-like (best_shift≈-h=-5, compare vs persistence baseline)
```

### The open question — pools measured

```
B_ML point pool      : 13
model_pool()         : 16
DESCRIPTIONS         : 18

in pool, not B_ML (the extra 3): ['GBQuantile', 'LGBMQuantile', 'ResidualRF']
described but in NO pool       : ['ETS', 'Theta']
in pool but undescribed        : []
```

### Diff stat

```
 backend/e_quantile_daily_pipeline.py               | 191 +++++++-
 backend/tests/test_failure_mode_distinctness.py    |  75 +++
 backend/tests/test_published_baseline_is_shared.py |   6 +-
 backend/tests/test_regression_mutations.py         | 522 +++++++++++++++++++++
 reports/gate_audit.md                              | 247 ++++++++++
 scripts/daily_summary.py                           |  70 ++-
 6 files changed, 1085 insertions(+), 26 deletions(-)
```

---

## 4. Full narrative response, verbatim

Item 5 done, and the audit turned up one genuine violation of the four-verdict rule plus two
measured blind spots I had wrong on my first attempt. Both suites green: root **602 passed, 3
skipped, `EXIT=0`**; frontend **110 passed, `EXIT=0`**.

### Part 1 — the audit, and the one thing that was actually broken

The full gate-by-gate audit is in [`reports/gate_audit.md`](../../reports/gate_audit.md): fourteen
gates, each with what it catches, what slips past it, and whether the reason it gives matches the
cause. The headline is that **coverage was not a distinct verdict at all.**

`daily_summary.gate_reasons()` produced three named reasons — leakage, no signal, persistence-like —
and a coverage failure arrived only as `run_status=FAILED_QUALITY` or the generic "quality gate
failed (per the family's integrity report)". So an E_QUANTILE family withheld because its 80% band
covered 43% of outcomes was **indistinguishable from one withheld for poor skill**, even though the
first needs recalibration and the second needs a different model. That is exactly the failure you
named, so I fixed it rather than only reporting it: `_coverage_failure_reasons()` reads the nominal
level as data, quotes the measured coverage, and says nothing about skill, leakage or persistence.
The generic line now gives way when the specific cause is named — but only then, so a failure with
no named cause is still reported rather than swallowed.

The other three verdicts hold up. The sentinel has been honest since M-5: a low ratio reports **no
signal** and the docstring states that leakage drives the ratio *up*. `read_gate`/`write_gate` are
genuinely tri-state, so "never verified" cannot render as a pass.

Two things I want to flag because they are *not* fixed, and both are decisions rather than cleanups:

**`conditional_coverage_gate` is not wired into `quantile_quality_gate`.** Only marginal coverage
gates. An interval covering 80% overall while covering 55% on the largest-magnitude tercile passes —
and the large days are the ones worth forecasting. Wiring it would change published verdicts on real
runs, so I left it and wrote it up.

**`check_feature_leakage` is the weakest instrument in the set.** It matches feature *names* for
negative shifts and flags a single univariate correlation above 0.99. It misses r = 0.98, misses any
leak assembled from a combination of features, and on a level target it sits about 0.01 away from
flagging entirely innocent lag features — a random-walk level correlates ~0.98 with itself at almost
any lag. A stronger test is a design question, not a threshold tweak.

### The measurement that corrected me

I first claimed `detect_lagged_copy` misses a perfect h=5 persistence copy because its default
`max_shift=3` window excludes shift 5. I read `result["flagged_models"]` to check — a key that does
not exist, so it returned `None` and confirmed whatever I expected. **My own positive control caught
it**: I asserted the detector *does* flag a lag-1 copy, that assertion failed, and the failure
output showed the real keys (`risk`, `details`, `per_model`).

Re-measured properly, the truth is more interesting than my theory. Flagging needs **both**
`(corr_best − corr_at_0) > 0.05` and `mae_pred >= 0.99 × mae_lag1`, and each condition fails on a
different real target shape:

* **A lag-1 copy of a level series is missed** — correlation margin only **+0.018**, under the 0.05
  requirement, because a random walk correlates 0.982 with itself at shift 0.
* **An h=5 copy of a flow is missed** — a flow has no autocorrelation, so the shifted copy is
  genuinely *worse* than lag-1 (ratio **0.979** < 0.99) and the second condition fails.
* The h=5 copy of a level series *is* caught, but with a margin of **+0.0601** against a 0.05
  threshold — by 0.01.

So on this project's two target shapes the detector is near the edge of its power in three of four
cases. It is not a live hole for B_ML, C_DL or E_QUANTILE, because `family_shift_flag` ORs it with
the horizon-aware diagnostic, which catches the h-step case cleanly (`best_shift = -5`,
"Persistence-like"). It **is** a live hole for a family that writes no shift fields, since then the
narrow detector is the only one running — and `pipeline_shift` correctly returns no flag when the
fields are absent. A_STAT is that family. All four cases are now pinned in a test that tells whoever
closes the gap to update the audit.

The general lesson: reading a key that does not exist is indistinguishable from reading a `False`.
The only reason I caught it is that I wrote the positive control.

### Part 2 — mutation tests

`backend/tests/test_regression_mutations.py`, 26 tests. The pattern throughout: reintroduce the
defect with `monkeypatch`, then call the **existing** guard and require it to fail.

```python
def _assert_test_fails(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except AssertionError:
        return
    pytest.fail(f"{fn.__name__} PASSED with the bug reintroduced -- "
                f"it does not actually guard against this regression")
```

Because the existing tests also run unmutated in the same suite, the pair is a proof: the test
passes on correct code and fails on the bug, so it discriminates. A test that cannot fail is worse
than no test, because it reads as coverage.

All six, plus two more the audit surfaced:

| Bug | Mutation | Guard required to fail |
|---|---|---|
| Sentinel inversion | pre-M-5 wrapper flags leakage when ratio < 1.5 | `test_no_signal_drives_the_ratio_DOWN...`, `test_no_signal_when_features_are_noise` |
| Sentinel claims leakage at any ratio | `leakage_warning = True` always | `test_the_sentinel_never_asserts_leakage_at_any_ratio` |
| Threshold relaxed to 1.0 | `signal_detected = ratio >= 1.0` | `test_no_signal_drives_the_ratio_DOWN...` |
| ExtraTrees memorisation | selector ignores the overfit ratio | `test_memorising_model_is_excluded`, `test_overfitting_model_is_not_crowned_best` |
| `inf` ratio treated as missing | filter non-finite ratios before selecting | `test_memorising_model_is_excluded` |
| Collapsed intervals | skill-only gate | differential: mutated passes a zero-width interval, real fails **for a coverage reason and not a skill one** |
| Coverage band widened to [0,1] | permissive band | differential against the real gate |
| Inverted alias semantics | `leakage_sentinel` claims leakage | `test_deprecated_name_still_works` |
| Duplicated baseline | shim gets a divergent lag-1 baseline | `test_the_deprecated_module_still_re_exports_the_sentinel` |
| Retired duplicate resurrected | re-add `compute_baselines` to the shim | `test_the_duplicate_persistence_implementation_is_gone` |
| `alignment_ok` literal | checker always returns `True` | `test_deliberately_misaligned_predictions_yield_alignment_ok_false` |
| Literal back in the source | `BACKEND` pointed at a mutated copy | `test_c_dl_no_longer_writes_an_unconditional_true` |
| Coverage folded into the generic reason | `_coverage_failure_reasons` returns `[]` | the two new four-verdict tests |
| Nominal level assumed | strip `coverage_nominal`/`coverage_band` before judging | `test_the_nominal_level_is_read_as_data_not_assumed` |

One of these found a real gap in an existing test rather than confirming it. The duplicated-baseline
mutation had nothing to fail against: `test_the_deprecated_module_still_re_exports_the_sentinel`
checked identity for `signal_sentinel`, `leakage_sentinel` and `MIN_SIGNAL_RATIO` — but **not
`compute_persistence_baseline`**, which is the symbol that was actually duplicated. The guard was
watching the wrong objects. Adding it to that list is a one-line change and it is the whole point of
mutation testing.

Two source-level guards generalise beyond their original bug: exactly one
`def compute_persistence_baseline` may exist in the backend (asserted with its file and line), and
**no** integrity verdict — `alignment_ok`, `leakage_detected`, `signal_detected`,
`quality_gate_passed`, `gate_passed` — may be written as a bare `True` anywhere in `backend/*.py`.
The `alignment_ok` literal was one instance of a class, and the class is now closed.

### Part 3 — the nominal level as data

The defect was subtler than a wrong name. The dangerous case is not a *renamed* key, it is a
*reconfigured* one: `Config.quantiles` is ordinary configuration, and setting it to
`(0.05, 0.50, 0.95)` makes the family produce a **90%** interval. The gate still tested that
coverage against `(0.70, 0.90)` — a band hardcoded around an assumed nominal 80%.

Measured, that assumption is wrong in **both** directions:

| Interval | Coverage | At its own nominal | At an assumed 80% |
|---|---|---|---|
| 90% | 91.0% | **pass** | **fail** — "outside [70%, 90%]" |
| 90% | 72.0% | **fail** | **pass** |

So the assumption would have withheld a correctly calibrated interval *and* published a broken one —
a wrong verdict with no artifact recording why.

The fix has three parts. `coverage_spec()` derives the interval from the fitted alphas: the widest
pair is the interval, so nominal is `max − min`. `emit_coverage()` is a single emitter used by all
three artifacts — leaderboard, `metrics_long`, integrity report — so the number and its level cannot
drift apart in one of them; it writes `coverage_key`, `coverage_nominal`,
`coverage_lower_quantile`, `coverage_upper_quantile` and `coverage_band` alongside the value. And
`quantile_quality_gate` takes `nominal` and derives the band from it, defaulting to 0.80 so **every
existing verdict is unchanged** — `(0.70, 0.90)` reproduced exactly.

On the assertion you asked for: `assert_coverage_key_matches_alphas` raises `CoverageLevelMismatch`
when the key does not name the fitted alphas. But I did not put a raise in the hot path, and that is
a deliberate call worth stating. The derived key is always correct by construction, so crashing a
pipeline over a legitimate reconfiguration would be the wrong behaviour. Instead the **legacy**
`coverage_p10_p90` is emitted only when it is true, and its omission is recorded as
`legacy_coverage_key_omitted` with the full explanation. A reconfigured pipeline therefore degrades
to a consumer finding no `coverage_p10_p90` and rendering "not reported" — rather than reading a 90%
coverage figure under a name that says 80%. Silence beats a mislabelled number. The assertion itself
is exercised directly by test.

One small thing I fixed while there: `metrics_long`'s `quantile` column was always blank on a
coverage row. That is precisely the field that should have carried the level, and it now does.

### The open question — 13 or 16

Measured: **16 = 13 + 3.** The extra three are `GBQuantile`, `ResidualRF` and `LGBMQuantile`, the
E_QUANTILE interval methods. `model_pool()` unions the B_ML point models with them.

There is also a third number: `DESCRIPTIONS` has **18**, because it describes `ETS` and `Theta` from
A_STAT, which `model_pool()` never enumerates. Two descriptions are unreachable from the page.

**Present 13, name the 3 separately, and never a bare "16".** The honest sentence is: *"13 point
models compete to be the champion for each target; prediction intervals come from 3 quantile
methods."* The 13 is the set a champion is chosen from and the set every measured claim in the
project belongs to. The 3 do a different job — they produce intervals, are not champion-eligible,
and are never crowned best model. Summing them implies sixteen candidates for one job when three of
them are doing another.

And "16" is not a project total either: A_STAT and C_DL sit outside it — A_STAT's most recent
artifact evaluated ETS against the persistence baseline, and C_DL offers five architectures. So any
single headline number is a claim about scope, and the scope has to travel with it. The 13/3/16
composition is now pinned by a test, so adding a model forces a deliberate update rather than
silently changing a number shown to a client.

---

## 5. Verdict

**Item 5 complete, all three parts plus the open question.**

Part 1 delivered as `reports/gate_audit.md`: fourteen gates audited, each with failure mode caught,
what slips past, and whether the reason matches the cause. It found one genuine violation of the
four-verdict requirement — coverage was not a distinct verdict — which is now fixed and pinned by
eight new tests including a case where all four fire at once.

Part 2 delivered as 26 mutation tests. Every one of the six named bugs is now guarded by a test
proved to fail when the bug returns, plus two source-level guards that close the *class* rather than
the instance. The exercise found a real defect in an existing guard: the shim identity check was
watching the sentinels and not `compute_persistence_baseline`, the symbol that had actually been
duplicated.

Part 3 delivered: the nominal level is derived from the fitted alphas, written as data beside every
coverage number by a single emitter, and the gate band follows it. Verified in both directions that
the previous assumption produced wrong verdicts. No existing verdict moved — the 0.80 default
reproduces `(0.70, 0.90)` exactly.

One correction to my own work, caught by a positive control I wrote: my first account of
`detect_lagged_copy`'s blind spot was wrong, because I read a dict key that does not exist. The
re-measured finding is in §4 and in the audit.

602 passed / 3 skipped `EXIT=0`; frontend 110 passed `EXIT=0`.

---

## 6. Outstanding

* **`conditional_coverage_gate` is not wired into `quantile_quality_gate`.** Marginal coverage can
  pass while conditional coverage fails, and only marginal gates. Wiring it changes published
  verdicts on real runs — your decision.
* **`check_feature_leakage` is weak** (name-matching + one univariate r > 0.99). Misses r = 0.98 and
  all multivariate leaks; on level targets it is ~0.01 from flagging innocent lags. Needs a design
  decision, not a threshold change.
* **A family writing no shift fields has no effective persistence-mimicry check at h=5.** A_STAT is
  that family. Either it writes the horizon-aware diagnostic, or `detect_lagged_copy` is called with
  `max_shift >= h` and a lag-h baseline.
* `DESCRIPTIONS` carries two models (`ETS`, `Theta`) no pool exposes, so two descriptions are
  unreachable from the Models page.
* No E_QUANTILE run has been re-executed since Part 3, so no *existing* artifact carries the new
  `coverage_nominal` fields. Readers fall back to the key name for historical runs, which is correct
  for those runs — they were 10/90.
* Unchanged from earlier records: six artifact fields still unfixed (classification delivered,
  decision reserved); master-prompt Part 5 accuracy levers, Part 6 remainder, Part 7 Georgian i18n;
  ops P0, artifact validator, Phase-1 cleanup, registry approval workflow, single TEST read.
* Flow targets remain `withheld_as_forecast` — the sentinel has not cleared 1.50 under any of the
  three probes.
