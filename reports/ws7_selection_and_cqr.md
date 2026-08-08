# Workstream 7 — conformalised intervals, a conditional-coverage gate, and the selection rule

**Date:** 2026-08-05 · **Data SHA-256:** `0b009fd0…5361f1` · **Calendar version:** `4b480eae9c8f`
**TEST (2025) reads: 0** — everything below is TRAIN-calibrated and DEV-scored.
All figures traced to logged runs (`study: ws7_cqr`); reproduce with `scripts/ws7_cqr.py`.

---

## The headline

**CQR delivers exactly the guarantee it promises — marginal coverage — and does not fix the
defect, which is conditional.** Overall coverage improves substantially where the band was too
narrow. Coverage on the largest third of days improves in relative terms and remains far below any
usable floor.

**The new gate fails 2 of 3 targets that the old marginal gate passed.** That is the point of
replacing it.

---

## Before and after, per tercile, per target

Nominal 80%. Conditional floor **60%** — deliberately below nominal, because requiring 80% in
*every* bucket on a 262-row window would gate on noise; 60% is where a band stops being usable at
all, failing more often than two days in five. Calibration used 502 causal TRAIN rows.

### Revenues — `LGBMQuantile` + ratio

| | Overall | Smallest | Middle | **Largest** | Mean width | Gate |
|---|---:|---:|---:|---:|---:|---|
| before | 83.2% | 79.8% | 100.0% | **69.9%** | 127,916,748 | ✅ pass |
| CQR global | 83.2% | 79.8% | 100.0% | 69.9% | 127,916,748 | ✅ pass |
| CQR grouped | 83.2% | 79.8% | 100.0% | 69.9% | 127,916,748 | ✅ pass |

**CQR correctly does nothing here.** The conformal width came out at **−0** — the band already
covers 83.2% against a nominal 80%, so there is no correction to make. A method that widened it
anyway would be broken; this is the expected null result, and it is worth having as evidence that
the implementation is not simply inflating every band it touches.

### Expenditure — `LGBMQuantile` + raw

| | Overall | Smallest | Middle | **Largest** | Mean width | Gate |
|---|---:|---:|---:|---:|---:|---|
| before | 57.2% | 71.4% | 90.4% | **9.6%** | 83,689,405 | ❌ fail |
| CQR global | 76.8% | 98.8% | 100.0% | **31.3%** | 131,800,378 | ❌ fail |
| **CQR grouped** | **78.0%** | 100.0% | 100.0% | **33.7%** | 137,325,767 | ❌ fail |

Overall coverage rises from 57.2% to 78.0% — close to nominal, which is the marginal guarantee
working. The largest third goes from 9.6% to 33.7%: a **3.5× relative improvement** and still
barely half the floor. The band is 64% wider and still misses two of every three big days.

### State budget balance — `LGBMQuantile` + delta

| | Overall | Smallest | Middle | **Largest** | Mean width | Gate |
|---|---:|---:|---:|---:|---:|---|
| before | 58.8% | 57.1% | 67.5% | **51.8%** | 409,917,268 | ❌ fail |
| **CQR global** | **72.0%** | 82.1% | 79.5% | **54.2%** | 567,629,655 | ❌ fail |
| CQR grouped | 68.4% | 73.8% | 77.1% | 54.2% | 542,197,751 | ❌ fail |

Global beats grouped here: with 502 calibration rows split three ways, each bucket's correction is
estimated from ~167 scores, and the variance costs more than the targeting gains. **Grouping is
not uniformly better** and the choice belongs per target.

---

## Why CQR cannot fix this, stated plainly

Split-conformal quantile regression guarantees **marginal** coverage: at least 1−α *averaged over
the calibration distribution*. It says nothing about any subgroup. A global correction adds the
same absolute width everywhere, so it lifts the average by over-covering the days that were
already fine — Expenditure's smallest third goes to **100.0%** — while the largest days stay
under-covered.

Grouped calibration attacks that directly and helps where the data supports it (Expenditure
+2.4pp on the largest third over global) but is limited by how many calibration rows each bucket
gets.

The remaining gap is not a calibration problem. It is that the underlying quantile model's band
does not widen when the day is large. Fixing that needs the band to respond to a
volatility/magnitude signal at fit time — Part 5d's trailing-volatility features (group H), which
target the documented inverted width response directly.

---

## A bug of mine that produced a passing gate

My first implementation bucketed calibration scores by the **actual** `y`, then assigned
corrections at prediction time by the **predicted** midpoint — the actual is not known when a band
is issued. Fitting on one quantity and applying by another put corrections in the wrong buckets.

Measured consequences of the bug, both now fixed:

* On Expenditure it made grouped CQR **worse than no calibration at all** — overall 57.2% → 43.2%,
  largest third 9.6% → 10.8%.
* On the stock target it produced overall **85.6%** with a largest third of **71.1%** and a
  **PASSING** gate. The correct implementation gives 68.4% / 54.2% and **fails**.

So the buggy version manufactured the one clean pass in the whole exercise. Both sides now bucket
by the same observable (the band midpoint), and the docstring records why.

---

## The conditional-coverage gate

Replaces the marginal gate. `backend/conformal.conditional_coverage_gate`:

* Scores **two independent axes** — magnitude terciles and **trailing-volatility** terciles. A band
  can be well calibrated across magnitudes and still fail on the days the series is moving most;
  GBQuantile's documented inverted response (widest at *low* volatility, review §3 P2.2) is exactly
  that failure, and a magnitude-only gate cannot see it.
* Gates on the **worst** bucket, not the average.
* **Reports every bucket**, pass or fail, with n and mean width.
* Buckets thinner than 20 rows are reported and **excluded from the verdict** rather than allowed
  to fail it on a handful of rows. If that leaves nothing to judge the verdict is `None` — never
  verified, never a pass.
* The failure message names the worst bucket and states the overall figure, so a reader can see
  precisely what a marginal gate would have missed.

| Target | Old marginal gate | New conditional gate |
|---|---|---|
| Revenues | pass | **pass** (weakest bucket 69.9%) |
| Expenditure | **pass** (57.2% overall) | **FAIL** — largest third 33.7% |
| State budget balance | **pass** (58.8% overall) | **FAIL** — largest third 54.2% |

---

## The per-target selection rule

`backend/conformal.select_per_target`, applied in order:

1. **Never select a candidate whose conditional-coverage gate failed while another passes.** An
   accurate point forecast with an unusable band is not the better product.
2. Otherwise rank by DEV MAE.
3. **Tie-break within 1% of the leader** — treat those as tied, break on TRAIN-fold evidence, then
   on the sentinel.

Rule 3 is the one that **keeps L2 candidates in the pool**, and it is there because of a measured
fact: on **Expenditure the DEV-best model is a squared-error `HistGBDT`** at 51,088,706 against the
promoted L1 recipe's 51,602,951 — **1.0% ahead**. A strict argmin on one DEV fold would have
swapped the recipe on that margin, and WS2 measured that a sub-1% TRAIN-to-DEV margin predicts
nothing (the objective got the direction wrong in all three cases). So a 1% DEV gap is treated as
a tie and decided on the five-fold evidence instead.

If no candidate passes the gate, selection falls back to the full pool and the winner **ships with
its band defect stated** rather than being silently withheld — the numbers are still useful as a
central-tendency guide, which is what the flows already publish as.

---

## What was and was not adopted

| | |
|---|---|
| **Adopted** | The conditional-coverage gate, replacing the marginal one. It is stricter and it fails two targets that previously passed |
| **Adopted** | CQR as an available correction, per target, with the grouped/global choice made on measured DEV coverage rather than assumed |
| **Not adopted** | No registry recipe changed. CQR does not lift either failing target over the floor, so promoting a CQR variant would add width and complexity without earning a passing gate |
| **Deferred** | Part 5d (trailing-volatility features). The residual defect is a fit-time problem, not a calibration one |

---

## Reproduction

```bash
SC=/tmp/ws2 OUT=/tmp/ws7.csv ./backend/.venv/bin/python scripts/ws7_cqr.py
./backend/.venv/bin/python -m pytest -q backend/tests/test_conformal.py   # 20 passed
```

Band producers are each target's WS2 winning configuration read back from its logged run — the
search was not re-run. Calibration is the last 502 rows of the TRAIN window with a 5-row gap, so
every conformity score comes from a target already known when the band is issued; a test asserts
the gap is exactly *h* for h in {1, 5, 10, 21}.
