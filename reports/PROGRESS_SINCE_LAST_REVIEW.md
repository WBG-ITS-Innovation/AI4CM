# Progress since the last review

**Date:** 2026-08-05 · **Data SHA-256:** `0b009fd0…5361f1` · **Calendar version:** `4b480eae9c8f`
**TEST (2025) reads: 0** · **Logged runs: 153** · Suite: 485 passing

Every figure below is a row in `experiments/log.csv` or a field in `registry/recipes.json` that
links to one by `run_id`. Nothing is quoted from a session-record summary table — several of those
had drifted, which is why the phase-13 record now carries a SUPERSEDED banner.

---

## 1 · Then versus now, per target

**Baseline** = the best squared-error model on the pre-programme feature set, on DEV (2024).
**Current** = the registry champion. Same window, same shared h-step persistence ruler.

| Target | | Model | DEV MAE | Skill vs ruler | MASE | Sentinel |
|---|---|---|---:|---:|---:|---:|
| **Revenues** | baseline | `HistGBDT` | 58,359,490 | 33.92% | 1.136 | 1.1382 |
| | **current** | `LightGBM_L1` + ratio | **38,931,956** | **55.92%** | **0.758** | 1.2255 |
| | change | | **−33.3%** | **+22.0pp** | 1.136 → 0.758 | +0.087 |
| **Expenditure** | baseline | `HistGBDT` | 51,088,706 | 30.13% | 1.093 | 1.0877 |
| | **current** | `LightGBM_L1` | 51,602,951 | 29.42% | 1.104 | 1.0882 |
| | change | | **+1.0% (worse)** | −0.71pp | ~flat | ~flat |
| **State budget balance** | baseline | `HistGBDT` | 257,087,305 | −5.95% | 2.090 | 5.5662 |
| | **current** | `HistGBDT_L1` + cross | **194,104,922** | **20.01%** | **1.578** | 7.0058 |
| | change | | **−24.5%** | **+25.96pp** | 2.090 → 1.578 | +1.44 |

Two of three targets improved substantially. **Expenditure did not** — its promoted recipe is 1.0%
*worse* on DEV than the baseline `HistGBDT`, and that is disclosed in the registry under
`not_the_dev_best` with the reason: selection ran on five TRAIN folds where the promoted recipe
leads clearly, and one DEV fold cannot separate a 1% gap. WS2 later measured that a sub-1% margin
predicts nothing in either direction, which supports the choice without making it a win.

The stock target crossed from **worse than the benchmark** (−5.95%) to **+20.01%** — the clearest
result in the programme.

MASE below 1.00 means better than a seasonal-naive repeat. Only Revenues achieves that (0.758).

---

## 2 · What each lever contributed, and what it did not

| Lever | What it contributed | What it did not |
|---|---|---|
| **WS1** absolute-error objectives | L1 beat its squared-error twin in **17 of 18** paired comparisons; turned three worse-than-benchmark stock models into three that beat it | Move any sentinel. Did **not** supply the best model on every target — Expenditure's DEV-best is L2 |
| **WS3** fiscal calendar | DEV error **−6.32%** (Revenues), −1.10% (Expenditure), −4.26% (stock). The shift rule (Tax Code Art. 3(6)) fires on **27.5%** of business days | Move the flow sentinel (1.138 → 1.226). Group E *hurt* two of three targets |
| **WS4** target scaling | **−25.73%** DEV error on Revenues, the single largest gain | Help Expenditure or the stock target — raw won both. The gain is **drift-dependent**: +1.30% in a flat window, +24.14% in a high-drift one (corr **+0.987**) |
| **WS5** multivariate | Nothing on the flows. `cross` gave the stock **+0.42%** | The debt-ops hypothesis **failed** — every block made both flows worse. The 0.971 correlation is contemporaneous, so lagged values cannot anticipate an auction |
| **WS2** tuning (100 trials × 3) | Stock **+6.09%** on DEV, a candidate | Both flows got **worse** (−3.13%, −0.17%). The TRAIN objective predicted the DEV direction in **none of three** cases |
| **WS7** CQR + conditional gate | Expenditure overall coverage **57.2% → 78.0%**; a gate that scores the worst bucket on two axes | Fix the defect. Largest-third coverage 9.6% → 33.7%, still far below the 60% floor. **No recipe changed** |
| **Probe study** | Established ridge misses pure interactions (1.295 vs 5.137) | Rescue the flows: highest flow reading under any of three probes is **1.4212** |

---

## 3 · The model pool

| | Before | Now |
|---|---:|---:|
| B_ML | 7 | **13** |
| E_QUANTILE | 2 | **3** |
| **Total** | **9** | **16** |

Added: `HistGBDT_L1`, `XGBoost_L1`, `LightGBM_L1` (WS1), `CatBoost_L1`, `CatBoost_Quantile`
(installed, **promoted nowhere** — unablated), `LGBMQuantile` (WS2 port, crossing-safe with
h-gapped early stopping).

---

## 4 · Correctness work that changed no number but changed what we can trust

| Fix | What it had been doing |
|---|---|
| One canonical gate key (X9) | b_ml wrote `quality_gate_failed`, the others `quality_gate_passed`; the summary read only the positive key, so **a B_ML run that failed its gate was reported as passing** |
| One publisher (X10) | `SUMMARY.json` and `integrity_report.json` each computed the verdict, and contradicted each other on the 2026-08-04 C_DL run |
| Tri-state alignment | `alignment_ok` defaulted to `True`, and C_DL writes it as a literal without performing the check, so **"never checked" rendered as "passed"** |
| Overfit exclusions surfaced | `overfit_excluded_models` was in every report and used **zero** times — an excluded model could top the leaderboard |
| Interval detection | The dashboard looked only for `y_lo`/`y_hi` with a hard-coded 90% target, so **E_QUANTILE showed nothing** and a correct 80% band was scored as broken |
| WS2 harness | A non-canonical ruler inflated every logged skill figure (Expenditure read 32.62%, actually 29.30%); a units mismatch made two of three sentinel readings meaningless |
| `run_id` in `SUMMARY.json` | The file could not identify its own run |
| X11 | A persistence baseline computed over zero prediction rows looked like an ordinary result |

---

## 5 · What we still cannot claim

**1 · The flow targets do not forecast events.** `Revenues` and `Expenditure` show no detectable
feature signal — sentinel 1.0710–1.2255 against a threshold of 1.50, across 71 logged runs, five
levers and three independent probes. Their errors genuinely are 29–56% below the benchmark, and
that improvement is regression toward a central level against a spiky benchmark, **not**
anticipation of individual days. Both publish as `withheld_as_forecast`. See
`docs/SIGNAL_FINDING.md`.

**2 · The 2025 holdout has never been evaluated against.** Zero gated reads. Every accuracy figure
here is DEV (2024) and provisional. The final check is one-shot and has deliberately not been
spent. Two retrospective disclosures are on record (`reports/HANDOFF.md` §0a) where 2025 figures
were computed outside the gate for new-versus-old comparability; **no selection was made from
them**.

**3 · The data file is stale.** The canonical dataset ends **2025-08-06**. Every forward forecast
therefore predicts dates beyond the data that are already in the past, and **no published forecast
can be scored until the input moves forward** — the track record is empty by construction, not by
oversight.

**4 · Prediction ranges are not trustworthy on the largest days.** Nominal-80% bands cover 33.7%
(Expenditure), 54.2% (stock) and 69.9% (Revenues) of the largest third. CQR improved the averages
and did not fix this; the residual is a fit-time problem.

**5 · Nothing is approved.** Every recipe's status is *candidate*, `approved_by` is null on all
three, and no approval workflow exists. No hyperparameter configuration has been promoted.

---

## 6 · What would move the needle, in order

1. **Treasury's forward auction and redemption calendar.** The unpredictable days are
   debt-operation days (negative `Revenues` coincides with negative `Increase in liabilities` on 64
   of 72, corr 0.971). We hold the historical figures and tested them — they make things worse,
   because a contemporaneous correlation is not a predictive one. Not a modelling task.
2. **A current data file.** Makes the forward output operational and starts the track record.
3. **Trailing-volatility features** (Part 5d) — the residual interval defect is that the band does
   not widen when the day is large.
4. The remaining untried accuracy levers: recency weighting, Fourier terms, interactions, a pooled
   global model across all 41 targets, and a spike-day two-part model.
