# Workstream 2 — hyperparameter tuning

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Data SHA-256:** `0b009fd0…5361f1`
**Calendar version:** `4b480eae9c8f` · **TEST (2025) reads: 0**
Every figure below traces to a row in `experiments/log.csv`. The search itself was **not
re-run**; its winning configuration was read back from `experiments/runs/<run_id>.json` and
refitted once per target by `scripts/ws2_recompute.py`.

---

## The finding

**The TRAIN-fold objective predicted the DEV outcome in none of the three cases.**

| Target | TRAIN objective gain | DEV MAE change | Direction predicted? |
|---|---:|---:|---|
| Revenues | +0.96% | **−3.13%** | no — reversed |
| Expenditure | +0.76% | **−0.17%** | no — reversed |
| State budget balance | +1.42% | **+6.09%** | no — 4× larger than TRAIN implied |

Two sub-1% TRAIN gains reversed on DEV; the largest TRAIN gain grew fourfold. A hundred trials
optimised against five fixed fold splits is enough to find configurations that suit those splits
specifically, and the TRAIN objective carries almost no information about which will generalise.
**A TRAIN-fold objective is not a reliable selection signal on this data** — which matters more
than any individual number here, because every future search would otherwise be trusted on it.

Adoption: **the stock target's tuned configuration is a candidate; neither flow target's is.**

---

## Two harness defects, and what they invalidated

Both were diagnosed in `reports/phase11_session_record.md` §4 and are fixed here. Neither
required re-running the search.

### 1 · Non-canonical ruler — every logged `skill_vs_ruler` was unusable

The harness built target dates as `origin + BDay(h)` — calendar business days, which ignore
Georgian public holidays and can land off the series — and filtered rows with `X.notna()`. Both
changed which `(y_true, origin_value)` pairs entered the baseline:

| Target | Harness ruler | Canonical ruler | Inflation |
|---|---:|---:|---:|
| Revenues | 90,800,654 | **88,317,355** | +2.8% |
| Expenditure | 76,722,514 | **73,117,667** | +4.9% |
| State budget balance | 248,793,303 | **242,653,025** | +2.5% |

An inflated ruler flatters skill, so every published WS2 skill figure was too high:

| Target | Skill as first logged | Skill on the canonical ruler |
|---|---:|---:|
| Revenues | 55.78% | **54.54%** |
| Expenditure | 32.62% | **29.30%** |
| State budget balance | 26.73% | **24.87%** |

Expenditure's was the worst case: it read as **higher** than its own untuned incumbent (32.62%
vs 29.42%) while its MAE was **worse**, which is arithmetically impossible against one ruler and
is what exposed the defect.

### 2 · Sentinel units mismatch — two of three readings measured nothing

The probe was handed the **ratio-transformed** training target alongside **original-scale** test
truth, so both the real and shuffled errors were dominated by the same units gap and the ratio
collapsed. Corrected by putting both sides in original units:

| Target | Transform | Sentinel as first logged | Corrected (ridge) | Was it valid? |
|---|---|---:|---:|---|
| Revenues | `ratio` | 1.0000 | **1.1670** | **No** — exactly 1.0000 was the tell |
| Expenditure | `raw` | 1.0710 | **1.0710** | **Yes** — identical, as predicted |
| State budget balance | `raw`, delta | 0.9835 | **3.9919** | **No** — delta target vs level truth |

Expenditure's raw-flow reading was sound all along and is reproduced to four decimal places,
which is a useful check that the correction did not simply move everything.

`dev_mae` was unaffected by either defect and is reproduced **exactly** for all three targets.

---

## Tuned versus untuned, on the same ruler

Incumbents are the registry's current recipes, traced to their logged `run_id`s.

| Target | | DEV MAE | Skill | MASE | Sentinel (ridge) |
|---|---|---:|---:|---:|---:|
| **Revenues** | incumbent `LightGBM_L1` + ratio | **38,931,956** | **55.92%** | 0.758 | 1.2255 |
| | tuned `LGBMQuantile`, 100 trials | 40,148,659 | 54.54% | 0.782 | 1.1670 |
| | change | **−3.13%** | −1.38pp | worse | lower |
| **Expenditure** | incumbent `LightGBM_L1` + raw | **51,602,951** | **29.42%** | 1.104 | 1.0882 |
| | tuned `LGBMQuantile` | 51,692,664 | 29.30% | 1.106 | 1.0710 |
| | change | **−0.17%** | −0.12pp | ~flat | ~flat |
| **State budget balance** | incumbent `HistGBDT_L1` + raw | 194,104,922 | 20.01% | 1.578 | 7.0058 |
| | tuned `LGBMQuantile` | **182,294,096** | **24.87%** | 1.482 | 3.9919 |
| | change | **+6.09%** | +4.86pp | better | **lower** |

The stock result is a **model swap as well as a retune** — the incumbent is `HistGBDT_L1` and the
search preferred `LGBMQuantile`.

### The stock gain comes with a caveat that must travel with it

Its sentinel **falls** from 7.0058 to 3.9919 while its accuracy improves. Both readings are far
above the 1.50 threshold, so the target still has clear signal either way, but a tuned model
that is more accurate *and* less reliant on its inputs is worth noting rather than presenting as
an unambiguous win. It is not grounds to reject the candidate; it is grounds not to overstate it.

---

## Sentinel under both probes

Reported per the standing decision to show both, with the tree reading **reported only** — its
threshold has never been derived from a null distribution, so it does not gate.

| Target | ridge (gates, threshold 1.50) | tree | forest | Verdict |
|---|---:|---:|---:|---|
| Revenues | 1.1670 | 1.4212 | 1.2602 | **no signal** |
| Expenditure | 1.0710 | 1.3965 | 1.1547 | **no signal** |
| State budget balance | 3.9919 | 3.1821 | 3.5801 | signal |

The tree probe again reads ~0.25–0.33 higher on the flows and still finds nothing: the highest
flow reading under any of three instruments is **1.4212**, short of 1.50. Tuning is the fifth
lever to improve accuracy without moving the flow sentinel.

---

## Intervals

The tuned models are quantile models, so they publish bands. Coverage on DEV, nominal 80%:

| Target | Overall | Smallest third | Middle third | **Largest third** | Crossings repaired |
|---|---:|---:|---:|---:|---:|
| Revenues | 83.2% | 79.8% | 100.0% | **69.9%** | 0 |
| Expenditure | 57.2% | 71.4% | 90.4% | **9.6%** | 0 |
| State budget balance | 58.8% | 57.1% | 67.5% | **51.8%** | **6** |

Expenditure's largest-third coverage is **9.6%** against a nominal 80%. On the biggest one third
of days its band contains the actual figure roughly one time in ten. That is the project's known
interval defect at its most extreme, and it is Part 4's target.

Six quantile crossings were repaired on the stock target — the first non-zero count anywhere, and
exactly what the crossing counter exists to surface. Both flows were 0.

**Not published:** per-tercile coverage for the untuned incumbents. Those runs went through the
point-model path, which does not log coverage, so there is no traceable figure to compare
against. A before/after interval comparison therefore belongs to Part 4, not here.

---

## Verdict

| Target | Adopt the tuned config? | Why |
|---|---|---|
| Revenues | **No** | 3.13% worse on DEV; sentinel lower too |
| Expenditure | **No** | 0.17% worse — indistinguishable from noise, and no reason to swap |
| State budget balance | **Candidate** | 6.09% better on DEV, held from TRAIN, but a cross-family swap on one DEV fold and a falling sentinel both argue for confirming it before promotion |

Nothing has been promoted into `registry/recipes.json` by this report. The three recomputed runs
are logged as `study: ws2_tuning_recomputed`, each recording `supersedes_run_id` so the
superseded rows stay readable without being mistaken for current.

---

## Reproduction

```bash
SC=/tmp/ws2 OUT=/tmp/ws2_recomputed.csv \
  ./backend/.venv/bin/python scripts/ws2_recompute.py
```

Reads `params_full.best_params` from each original run's JSON, refits once, and recomputes the
ruler from the target series on the DEV window with a plain h-step shift — the same computation
the registry and the WS4/WS5 rows use.
