# Calibrating the signal sentinel: 1.50 → 1.15

**Date:** 2026-08-13 · Reproduce with
`./backend/.venv/bin/python scripts/calibrate_sentinel.py --draws 120 --json out.json`

---

## Why

`MIN_SIGNAL_RATIO = 1.5` was a bare constant with a one-line comment, and it was the **sole
publication gate** for two of three targets. Its justification was prose:

> The threshold is **1.50** — the error must get at least half again worse. A reading of 1.00 means
> shuffling changed nothing at all, so the margin above 1.00 exists so that noise cannot pass.
> — `docs/SIGNAL_FINDING.md`

"So that noise cannot pass" is a claim about a false-positive rate, and no false-positive rate had
ever been measured. Nobody knew what a signal-free feature set actually produces on this data, so
nobody knew whether 1.50 was tight, loose, or wildly off. `reports/sentinel_probe_study.md`
pre-registered that it would not move the threshold, which was correct discipline for that study and
left the number unvalidated.

That matters because the best flow reading anywhere — 1.421, the tree probe on Revenues — sat
**0.08 below** the line. A 5% margin on an unvalidated number was deciding what a client is told.

---

## The null

**H0: the features carry no information about the target.**

Only the *features* are nulled. The target is built exactly as the pipeline builds it (h-step ahead,
delta for a stock target) and the split is the real TRAIN → DEV, so anything the probe finds is an
artifact of the instrument rather than of the data. Two constructions, because they fail differently:

| Null | Construction | Why |
|---|---|---|
| **permuted** | the real feature matrix with its **rows permuted** | destroys the feature-target pairing while preserving every feature's marginal distribution, scale and cross-correlation. The realistic null, and the wider one |
| **noise** | standard normal draws of the same shape | cleaner but optimistic: real features have heavier tails and mutual correlation a probe can latch onto |

The threshold is set from the **permuted** null, because it is the wider of the two and therefore the
conservative choice.

---

## Result

120 draws per target per null (n = 360 pooled per null):

```
Revenues              real 1.1670    permuted: median 1.0000  p95 1.0780  p99 1.1041  max 1.1164
Expenditure           real 1.0710    permuted: median 1.0054  p95 1.0373  p99 1.0516  max 1.0542
State budget balance  real 1.2700    permuted: median 1.0020  p95 1.0238  p99 1.0325  max 1.0329

POOLED permuted (n=360):  median 1.0023  p95 1.0532  p99 1.0943  p99.9 1.1124  max 1.1164
POOLED noise    (n=360):  median 0.9989  p95 1.0131  p99 1.0230  max 1.0291
```

**False-positive rate by threshold, pooled permuted null:**

| threshold | FPR | null draws passing |
|---|---|---|
| 1.05 | **6.94%** | 25 of 360 |
| 1.10 | **0.56%** | 2 of 360 |
| 1.12 | 0.00% | 0 of 360 |
| **1.15** | **0.00%** | 0 of 360 |
| 1.50 | 0.00% | 0 of 360 |

A 60-draw re-run reproduces the same maximum (1.1164) and the same 0.00% at 1.15, with a slightly
lower p99 (1.0796) as expected from the smaller sample.

---

## What this says

**The null is tight around 1.00.** Median 1.0023, and it never exceeds 1.1164 in 360 draws. So the
sentinel is a low-variance instrument: a reading meaningfully above ~1.10 is not something
signal-free features produce.

**1.50 was never a noise margin.** Its false-positive rate is 0.00% — and so is 1.12's. A threshold
sitting far beyond the point where the FPR reaches zero buys no additional protection against noise;
it only rejects real readings. The prose rationale ("so that noise cannot pass") described a margin
of roughly 0.10; the constant implemented one of roughly 0.50.

**Two readings change meaning once the null is known.**

* **Revenues, 1.2255** (logged, ridge probe) is far outside the null — above its 99th percentile of
  1.0943 and above the observed maximum of 1.1164. Under 1.50 this was "no signal". Measured against
  the null it is signal, at p < 1/360.
* **Expenditure, 1.0882** sits **below** the pooled null's 99th percentile (1.0943). This is the
  stronger and more useful finding: Expenditure's reading is not merely under a threshold, it is
  *inside the distribution that signal-free features produce*. "Indistinguishable from no signal" is
  now a measurement rather than an inference.

---

## Decision

**`SENTINEL_MIN = 1.15`** (`backend/publication_gates.py`).

Chosen as the smallest round value whose measured false-positive rate is 0.00% over 360 draws, above
the observed null maximum of 1.1164. The next lower round value, 1.10, has a measured FPR of 0.56%.

The old constant is kept as `SENTINEL_MIN_UNCALIBRATED = 1.50` so the change is greppable and the
previous value is not quietly forgotten.

**And the sentinel is no longer the sole publication gate.** Even a calibrated single threshold
deciding publication alone is fragile. P2 adds a binding accuracy gate (MASE < 1.0), so signal is now
one input among several — which was the other half of the remit.

---

## Limitations, stated

* **The null has 360 draws; the real reading has one.** The sentinel's internal target shuffle uses a
  fixed seed (42), so each real reading is a point estimate with no variance attached. A confidence
  interval on the real side would need the seed varied too. The FPR figures are therefore about the
  threshold, not about how precisely any single reading is known.
* **The permuted null destroys temporal ordering**, which may itself alter difficulty in ways a
  merely-uninformative feature set would not.
* **Calibrated on this data, at h=5, on these three recipes' feature sets.** A different horizon or
  feature set needs a re-run; the script exists so that is cheap.
* The threshold is calibrated against noise, not against *usefulness*. Clearing 1.15 means the
  features carry detectable information, not that the forecast is good enough to act on. That is what
  the MASE gate is for.
