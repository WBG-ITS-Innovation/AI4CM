# Workstream 4 — Target scaling

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Data SHA-256:** `0b009fd0…5361f1`
**Suite:** 405 passing · **TEST (2025) reads: 0** · all 14 figures logged in `experiments/log.csv`

---

## The headline

**Ratio-to-trailing-level is a large win on Revenues and a loss on everything else.**

| Target | Winner | TRAIN vs raw | DEV vs raw | DEV skill |
|---|---|---:|---:|---:|
| **Revenues** | **ratio** | **+11.05%** | **+25.73%** | 40.65% → **55.92%** |
| Expenditure | raw | asinh −1.46%, ratio −6.13% | — | 29.42% |
| State budget balance | raw | asinh −1.59%, ratio −5.90% | — | 20.01% |

Revenues' DEV error falls from 52,417,152 to **38,931,956**. That is the single largest
improvement in the project so far, and it is larger on DEV than on TRAIN — see §4, which is
where the scrutiny belongs.

---

## 1 · How this was built, and why that matters more than the result

The obvious implementation transforms the target inside the pipeline, fits, and inverts the
predictions on the way out. That route passes straight through the code emitting
`origin_value` and `y_true` — the two columns the **unified persistence ruler** is computed
from. A mistake there does not raise. It silently moves the benchmark that every model in
this project is measured against.

So `backend/target_scaling.py` takes the other route. `ScaledRegressor` is an ordinary
scikit-learn regressor that transforms `y` inside `fit` and inverts inside `predict`. The
pipeline hands it a target and receives a prediction **in original units**; it never sees a
transformed value. **No line of `b_ml_pipeline.py` changed.**

The ruler is therefore identical *by construction*. It is asserted anyway:

```
Revenues              raw / asinh / ratio  ->  ruler 67,062,951.07  (identical)
Expenditure           raw / asinh / ratio  ->  ruler 54,170,043.11  (identical)
State budget balance  raw / asinh / ratio  ->  ruler 153,623,500.17 (identical)
```

And against the published 2025-definition constants:

| Target | Published | Recomputed | Difference |
|---|---:|---:|---:|
| Revenues | 83,534,152.85 | 83,534,152.854167 | 0.004 |
| Expenditure | 83,839,124.43 | 83,839,124.426218 | 0.004 |
| State budget balance | 189,930,653.98 | 189,930,653.975705 | 0.004 |

Differences are the rounding of the published two-decimal figures. **This recomputes a
model-free data statistic** — persistence against truth, no model fitted, nothing selected —
so it is not an evaluation of the sealed window and the ledger stays at zero.

A further test asserts `b_ml_pipeline.py` contains no transform code at all
(`arcsinh`, `inverse_transform`, `ScaledRegressor`, …), so if a future change threads a
transform through the pipeline instead of wrapping the estimator, the ruler guarantee weakens
from *impossible* to *checked* — and that is where it will show up.

---

## 2 · The three candidates

| | Definition | Notes |
|---|---|---|
| `raw` | identity | the incumbent, and what the others must beat |
| `asinh` | `z = asinh(y/s)`, `y = s·sinh(z)` | defined on the whole real line, so it survives the 72 negative `Revenues` days; log-like for large magnitudes. `s` is a robust scale **fitted per fold** |
| `ratio` | `z = y/L(t)`, `y = z·L(t)` | `L(t)` = causal trailing median of \|y\| over 63 business days, shifted. Scale-free and adaptive |

`log1p` remains **unavailable** while Treasury question **T1** is open: it is undefined below
−1 and 72 business days of `Revenues` print negative.

For the stock target the pipeline models the **change**, so `L` is derived from the change,
not the level — a level-scale divisor (~1e9) against a delta-scale target (~1e8) would shrink
the target by an order of magnitude. A test pins that reasoning.

---

## 3 · Full results (TRAIN-internal folds 2019–2023, n=1,304)

Vehicles: `LightGBM_L1` for the flows, `HistGBDT_L1` for the stock, on each target's WS3/WS5
recipe.

### Revenues — ruler 67,062,951.07

| Transform | MAE | Skill | vs raw |
|---|---:|---:|---:|
| **ratio** | **35,456,541** | **47.13%** | **+11.05%** |
| raw | 39,862,167 | 40.56% | — |
| asinh | 40,803,077 | 39.16% | −2.36% |

### Expenditure — ruler 54,170,043.11

| Transform | MAE | Skill | vs raw |
|---|---:|---:|---:|
| **raw** | **35,069,403** | **35.26%** | — |
| asinh | 35,579,811 | 34.32% | −1.46% |
| ratio | 37,219,822 | 31.29% | −6.13% |

### State budget balance — ruler 153,623,500.17

| Transform | MAE | Skill | vs raw |
|---|---:|---:|---:|
| **raw** | **130,265,217** | **15.20%** | — |
| asinh | 132,333,869 | 13.86% | −1.59% |
| ratio | 137,948,793 | 10.20% | −5.90% |

### DEV confirmation

| Target | Transform | DEV MAE | Skill | vs raw |
|---|---|---:|---:|---:|
| Revenues | **ratio** | **38,931,956** | **55.92%** | **+25.73%** |
| Revenues | raw | 52,417,152 | 40.65% | — |
| Revenues | asinh | 52,629,749 | 40.41% | −0.41% |
| Expenditure | raw | 51,602,951 | 29.42% | — |
| State budget balance | raw | 194,104,922 | 20.01% | — |

The Expenditure and stock `raw` rows **reproduce their WS5 DEV figures exactly**, confirming
the WS4 harness and the WS5 harness agree.

**asinh loses on all three targets.** Its compression helps the spikes and hurts the ~250
ordinary days, and since we score on absolute error over all days, that is a net loss.

---

## 4 · Why Revenues gains more on DEV than on TRAIN — and why that is not leakage

A DEV gain (+25.73%) more than twice the TRAIN gain (+11.05%) is the pattern that should
prompt a leakage check, so here is the check.

**The alignment.** The target is `y(t+5)`; the divisor is `L(t)`, the trailing level at the
**origin**, computed from `|y|` over the 63 business days *before* `t`. Dividing by `L(t+5)`
would use data after the origin and would improve every metric spuriously. The wrapper aligns
its divisor to the **feature frame's** index, which is the origin date. Two tests pin this:
one asserts the divisor equals the origin-dated level and explicitly asserts it does **not**
equal the target-dated level; another mutates the future and asserts no past divisor changes.

**The mechanism.** Revenues' level shifted between windows — the persistence ruler is
51,364,210 on TRAIN and 88,317,355 on DEV, a 72% rise. A model trained on 2015–2023 raw
magnitudes systematically under-predicts 2024. The `ratio` target is *scale-free*: it asks
"how large is this day relative to recent days", so the level shift is absorbed by the
divisor rather than having to be learned. A transform designed to be robust to level changes
should show its largest advantage precisely where the level changes most — which is what
happened.

That is a mechanistically coherent explanation with a leak check behind it, not an
unexplained jump. It should still be re-examined at the single TEST read, and the ratio
recipe is the one to watch there.

---

## 5 · A guard of mine that was wrong, and how it announced itself

`sanity_check_prediction_scale` compares a prediction batch's magnitude against the training
targets' magnitude and raises on a large mismatch — the signature of an un-inverted or
doubly-inverted prediction, which is the failure mode that produces plausible numbers in the
wrong units.

It aborted the Expenditure/asinh run:

```
ValueError: transform 'asinh': predicted magnitude 307414 is implausible against a
training magnitude of 2.7901e+07 (factor 50).
```

I checked the transform before assuming the guard was right. It was fine —
`sinh(median z)·scale` recovers the median exactly (38,379,251), and Expenditure's
distribution is near-identical to Revenues', which had passed. So I instrumented the
pipeline instead:

```
predict() batch sizes: [(1, 1304), (149, 1), (597, 1), (199, 1), ...]
calls with <30 rows: 1304 of 1314
```

**The pipeline predicts one origin at a time.** The median magnitude of a one-row batch is
just that row, so the guard fired on a legitimate holiday-zeroed day. A units error is
*systematic* — it affects every prediction — so it is detectable on the in-sample batch at
fit time, which is thousands of rows; one row cannot distinguish it from an unusual day.

Fixed by moving the load-bearing check to fit time on in-sample predictions, and skipping
batches below 30 rows at predict time. Three regression tests, including one that asserts the
exact value which triggered it (307,414) no longer raises on one row but **still raises** on a
30-row batch.

---

## 6 · A bug worth naming: `d.transform` is a pandas method

The registry-update script silently updated nothing. The filter was
`d[(d.target==t) & (d.transform==tf) & (d.window=="dev")]` — and `DataFrame.transform` is a
**pandas method**, so `d.transform == tf` compared a bound method to a string, produced
`False`, and the mask matched zero rows. No exception; the script reported success.

Caught because I checked the written file rather than trusting the script's own output. Fixed
with bracket access throughout. Worth recording because attribute access on a DataFrame column
named `transform`, `index`, `size`, `count`, `min` or `max` fails exactly this quietly.

---

## 7 · What was folded in

| Target | Scaling now | Recipe change |
|---|---|---|
| Revenues | `ratio`-to-trailing-level, 63-business-day causal median, divisor at the **origin** | `params.target_transform = "ratio"`, `trailing_window = 63` |
| Expenditure | raw (compared; raw won) | none |
| State budget balance | raw, level modelled as a change | none |

The registry's DEV credentials were re-pointed to the winning runs, and the forward forecast
now **applies the recipe's transform** — publishing a raw fit under a recipe that won on
`ratio` would make the quoted accuracy belong to a different model. A test asserts the forward
run's transform matches the registry for every target.

The published Revenues forecast changed accordingly (day 1: 94.8 → 99.6 million lari; bands
tightened from 35.1–147.1 to 42.4–147.8).

**Unchanged:** the signal verdicts. Revenues still fails the signal gate at 1.2255 and remains
`withheld_as_forecast`. A 56% improvement over the benchmark is a better central-tendency
estimate, not an event forecast — the two numbers still have to be quoted together.

---

## 8 · Reproduction

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q                        # 405 passed
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from registry import verify_against_log as v;print(v())"       # ok: True
```

Runs are `ConfigBML(...)` with `available_models()` narrowed to the target's vehicle and
wrapped in `ScaledRegressor(base=…, transform=…, level=…)`; `eval_end="2023-12-31"` for TRAIN
and `eval_start="2024-01-01", eval_end="2024-12-31"` for DEV. Every run asserted its own
window membership before any number was recorded.
