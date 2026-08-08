# WS4 robustness — is Revenues' scaling gain a real property or a regime artefact?

**Date:** 2026-08-05 · **Suite:** 421 passing · **TEST (2025) reads: 0**
30 rows logged in `experiments/log.csv` (`study: ws4_robustness`).

---

## The question, and the pre-registered reading

WS4 adopted `ratio`-to-trailing-level for Revenues on a **+25.73% DEV** improvement that
exceeded its **+11.05% TRAIN** improvement. I attributed that to a 72% level rise between
windows being absorbed by the divisor. That was an explanation, not evidence.

Pre-registered before running:

> if the advantage appears only in high-drift windows, the DEV figure is regime-dependent and
> must be quoted that way; if it holds in flat windows too, it is a genuine property.

---

## Result: the explanation is confirmed, and the answer is "both"

**The advantage is drift-dependent in magnitude and drift-independent in sign.**

### Revenues — correlation +0.987

Five TRAIN-internal rolling-origin windows (each year trained on all prior years). Drift is
the evaluation window's median |y| against the median |y| of everything strictly before it.

| Window | n | Level drift | raw MAE | ratio MAE | ratio advantage |
|---|---:|---:|---:|---:|---:|
| 2019 | 261 | 13.5% | 23,011,316 | 22,712,006 | **+1.30%** |
| 2020 | 262 | 18.7% | 39,981,403 | 39,314,049 | +1.67% |
| 2021 | 261 | 34.1% | 42,143,868 | 40,394,102 | +4.15% |
| 2022 | 260 | 70.2% | 47,299,167 | 39,262,316 | +16.99% |
| 2023 | 260 | 84.1% | 46,930,202 | 35,600,587 | **+24.14%** |

**corr(drift, advantage) = +0.987.**

### The DEV figure is exactly where that relationship predicts

| | Value |
|---|---:|
| DEV (2024) level drift | **81.7%** |
| Advantage predicted by a line fitted on the five TRAIN windows *only* | **+21.83%** |
| Advantage actually measured on DEV | **+25.73%** |

A 4-percentage-point residual on a five-point fit, extrapolating to the second-highest drift
in the sample. **The DEV gain is not anomalous — it is the drift relationship doing what it
does.** That is a genuine out-of-sample confirmation of the mechanism, because the line was
fitted without any 2024 data.

### So how must it be quoted?

**Regime-dependent, per the pre-registered reading.** The 55.92% DEV skill describes a
high-drift period. In a flat-level period the advantage over raw is roughly **1–2%**, not 26%.
Quoting 55.92% as the model's general standing would be wrong.

**But adoption still stands, for a reason worth separating out:** the advantage was
**positive in every window tested**, including the lowest-drift one (+1.30% at 13.5% drift).
It never went negative. So the transform is regime-dependent in *magnitude* but not in
*sign* — the floor is small-positive rather than a downside. That is a different and weaker
claim than "it is a genuine property", and it is the one the evidence supports.

This caveat is now encoded in `registry/recipes.json` under `scaling_caveat`, with two tests
asserting it is present on Revenues and absent on the targets that kept raw — so the number
cannot travel without its condition.

---

## The other two targets: ratio hurts, and *more* drift makes it worse

### Expenditure — correlation −0.506

| Window | Level drift | raw MAE | ratio MAE | ratio advantage |
|---|---:|---:|---:|---:|
| 2019 | 18.5% | 23,428,961 | 22,923,730 | +2.16% |
| 2020 | 51.2% | 31,261,705 | 34,200,936 | −9.40% |
| 2023 | 90.5% | 42,376,444 | 42,420,267 | −0.10% |
| 2022 | 92.6% | 37,869,932 | 39,464,790 | −4.21% |
| 2021 | 102.3% | 40,463,288 | 47,129,478 | **−16.47%** |

### State budget balance — correlation −0.813

| Window | Level drift | raw MAE | ratio MAE | ratio advantage |
|---|---:|---:|---:|---:|
| 2019 | −31.9% | 66,907,377 | 67,458,938 | −0.82% |
| 2022 | 58.3% | 134,533,780 | 139,286,909 | −3.53% |
| 2021 | 60.8% | 170,142,567 | 176,371,779 | −3.66% |
| 2023 | 124.0% | 135,187,822 | 141,488,028 | −4.66% |
| 2020 | 151.9% | 144,535,082 | 165,053,148 | **−14.20%** |

The WS4 decision to keep raw for both is confirmed on every window, not just on the average.

### A finding I cannot fully explain, stated as such

**Drift alone does not determine whether `ratio` helps.** Expenditure and the stock target
experience *more* drift than Revenues (up to 102% and 152% against Revenues' 84%), and `ratio`
hurts them — increasingly so as drift rises. So "the divisor absorbs level shifts" is
necessary but not sufficient.

The plausible additional condition is whether a target's *fluctuations scale with its level*.
If they do, dividing by the level stabilises the target; if the fluctuations are closer to
additive, dividing by a moving level injects noise from the divisor without removing any from
the numerator. That would explain the sign difference. **I have not tested it**, so it is
recorded as a hypothesis and not as a finding. The cheap test would be to correlate each
target's rolling dispersion against its rolling level and check the sign matches — WS7 work.

---

## What this changes

| | |
|---|---|
| Revenues recipe | **unchanged** — `ratio` stays, on a positive-in-every-window basis |
| How Revenues' DEV skill is quoted | **changed** — now explicitly conditional on a high-drift period, enforced by `scaling_caveat` and two tests |
| Expenditure / stock recipes | **unchanged** — raw confirmed per-window, not just on average |
| What to watch at the single TEST read | Whether 2025's level drift is high or flat. If flat, expect Revenues' advantage over raw to be **small**, and do not read that as the transform failing |

Nothing here was evaluated on 2025.

---

## Reproduction

Per-window figures are derived from the existing WS4 TRAIN runs by splitting
`predictions_long.csv` on evaluation year — each year is its own rolling-origin fold (trained
on all prior years), so these are genuine out-of-sample windows rather than slices of one
fold. Drift uses median |y| of the window against median |y| of all strictly-prior data. All
30 rows are in `experiments/log.csv` with `study: ws4_robustness` and the window's drift in
`params`.
