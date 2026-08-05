# Phase 2 — Target Scaling & Published-Forecast Scoring Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `4b4a515` (7 ahead of `origin/main` @ `863f967`)
**Test suite:** 377 → **417** passing
**Data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**Calendar version:** `4b480eae9c8f` · **Experiments log:** 105 rows, integrity ok
**TEST (2025) gated reads: 0** — two retrospective disclosures on record (HANDOFF §0a)
**PR #26** open, not merged — updated, not duplicated

---

## 1 · Premise verification — all held

| Premise | Measured |
|---|---|
| `model/excellence` @ `3be4ea0` / record child | ✅ `de828a0` |
| suite = 377 · log integrity ok · `calendar_version` `4b480eae9c8f` | ✅ |
| TEST gated reads = 0 · data SHA `0b009fd0…` | ✅ |
| PR #26 state | ✅ **open, not merged** → noted; no second PR; pushes update #26 |

**New standing rule adopted:** never state a test count in a commit message that was not
produced by a `pytest` run from the repo root **in that same command**. Every commit this
session complies, and each says so.

---

## 2 · Decisions recorded where they take effect · `90213a2`

| # | Decision | Where it landed |
|---|---|---|
| 1 | Dual-probe sentinel: **adopt both**; 1.50 stays with ridge; tree reading reported only until its own null distribution exists (WS7) | `docs/ROADMAP.md` §3 |
| 2 | Model-agreement disagreement: **report, do not gate** — needs a null distribution first | `docs/ROADMAP.md` §3 |
| 3 | Expenditure champion stays `LightGBM_L1` on TRAIN-fold evidence with `not_the_dev_best` disclosed; WS7 owns the tie-break and keeps L2 in the pool | `docs/ROADMAP.md` §3, registry |
| 4 | Install CatBoost | `backend/requirements.txt`; hooks now active |

CatBoost 1.2.10 installed: `HAVE_CATBOOST` is True, **13 models register**, `CatBoost_L1` on
MAE loss and `CatBoost_Quantile` on `Quantile:alpha=0.5`. Two tests pin it in both
directions — that the hooks are live, and that **CatBoost is promoted nowhere**, because it
remains unablated and enters the pool in WS2.

---

## 3 · Workstream 4 — target scaling · `250f501`

Full detail in [reports/ws4_target_scaling.md](reports/ws4_target_scaling.md).

### Result

| Target | Winner | TRAIN vs raw | DEV vs raw | DEV skill |
|---|---|---:|---:|---:|
| **Revenues** | **ratio** | **+11.05%** | **+25.73%** | 40.65% → **55.92%** |
| Expenditure | raw | asinh −1.46%, ratio −6.13% | — | 29.42% |
| State budget balance | raw | asinh −1.59%, ratio −5.90% | — | 20.01% |

Revenues DEV error **52,417,152 → 38,931,956** — the largest single improvement in the
project. `asinh` loses on all three: its compression helps the spikes and hurts the ~250
ordinary days, and we score absolute error over all days.

### The ruler could not move, and it did not

Transforming the target inside the pipeline would run through the code emitting
`origin_value` and `y_true` — the columns the unified persistence ruler is computed from,
where a mistake does not raise but silently moves the benchmark every model is measured
against.

So `ScaledRegressor` is an ordinary sklearn regressor that transforms `y` in `fit` and
inverts in `predict`. The pipeline receives predictions in **original units** and never sees
a transformed value. **No line of `b_ml_pipeline.py` changed.**

Ruler bit-identical across all three transforms within each target
(67,062,951.07 / 54,170,043.11 / 153,623,500.17), and matching the published
2025-definition constants to their quoted precision (0.004 on all three). That
recomputation is a **model-free statistic** — persistence against truth, nothing fitted,
nothing selected — so the ledger stays at zero. A further test asserts `b_ml_pipeline.py`
contains no transform code, so a future change that threads one through the pipeline
weakens the guarantee from *impossible* to *checked* right where it will be noticed.

### Why Revenues gains more on DEV than TRAIN — checked, not assumed

A DEV gain twice the TRAIN gain is the pattern that should prompt a leakage check.

**Alignment:** the target is `y(t+5)`; the divisor is `L(t)`, the trailing level at the
**origin**. Dividing by `L(t+5)` would be lookahead and would improve everything spuriously.
Two tests pin it — one asserts the divisor equals the origin-dated level and explicitly
asserts it does **not** equal the target-dated level; another mutates the future and asserts
no past divisor moves.

**Mechanism:** Revenues' level rose 72% between windows (ruler 51,364,210 TRAIN vs
88,317,355 DEV). A raw-target model trained on 2015–2023 magnitudes under-predicts 2024; a
scale-free target absorbs the shift in its divisor. A transform built to be robust to level
changes should gain most where the level changes most. Still the recipe to watch at the
single TEST read.

### Two of my own mistakes

**A guard that was wrong.** `sanity_check_prediction_scale` aborted Expenditure/asinh
("predicted magnitude 307414 … implausible against 2.7901e+07"). I checked the transform
before assuming the guard was right — it was fine. Instrumenting the pipeline showed why:

```
predict() batch sizes: [(1, 1304), (149, 1), (597, 1), ...]
calls with <30 rows: 1304 of 1314
```

**The pipeline predicts one origin at a time.** A one-row median is just that row, so the
guard fired on a legitimate holiday-zeroed day. A units error is systematic and therefore
visible on the in-sample batch at fit time; one row cannot distinguish it from an unusual
day. Load-bearing check moved to fit time; batches under 30 rows skipped. Three regression
tests, one asserting the exact triggering value no longer raises on one row but **still
raises** on a 30-row batch.

**`d.transform` is a pandas method.** The registry-update script silently updated nothing:
`d[(d.target==t) & (d.transform==tf) & …]` compared a *bound method* to a string, yielded
`False`, matched zero rows, and reported success. Caught by reading the written file rather
than trusting the script's own output. Bracket access throughout now. Attribute access on a
column named `transform`, `index`, `size`, `count`, `min` or `max` fails exactly this
quietly.

### Folded in

Revenues → `ratio`; others stay raw. Registry credentials re-pointed to the winning runs
(`verify_against_log` 12/12). The forward forecast now **applies the recipe's transform** —
publishing a raw fit under a recipe that won on `ratio` would attribute one model's accuracy
to another — with a test asserting the two match per target. Published Revenues day 1 moved
94.8 → 99.6 million lari, bands tightened.

**Signal verdicts unchanged.** Revenues still fails at 1.2255 and stays
`withheld_as_forecast`. A 56% improvement over the benchmark is a better central-tendency
estimate, not an event forecast.

One existing test was rewritten rather than deleted:
`test_provenance_records_the_sealed_window_and_pending_scaling` asserted the literal string
"WS4 pending", which became false the moment WS4 landed. Correct behaviour — the artifact
must never describe a scaling decision that does not match the fitted model.

---

## 4 · Published-forecast retention + realized scoring · `4b4a515`

Forward artifacts were gitignored, so **nothing recorded what we told anyone**.

**Retention.** `forecasts/published/<issue_date>/` — tracked — with predictions, intervals,
the gate verdicts in force at issue time, `recipe_id`, provenance and a manifest. Immutable:
re-publishing an issue date raises unless `overwrite=True`. One test runs `git check-ignore`
on the real published file, because retention is pointless if the directory is ignored —
which it was.

**Why scoring is not a holdout read.** The sealed window is sealed against evaluation
*before* commitment: you must not look, then choose. This is the opposite ordering — the
prediction was committed in writing, with a data fingerprint and a git SHA, before the truth
existed.

The enforced rule: **a published date is scored only once its truth is in the canonical
dataset.** `score_one()` raises `TruthNotAvailable`; `score_published()` records such dates
as pending. The scorer cannot reach into data we do not have, so it cannot be pointed at the
holdout to manufacture a number.

Measured on the real run: **15 pending, 0 scored**, because every published date is beyond
the data end. A test asserts exactly that — any scored row would mean the scorer had found
truth that should not exist.

The machinery is proven separately from its refusal: tests score a backdated issue end to
end — realized error, the persistence comparator (`y` at target_date − 5 business days, the
same unified ruler), an interval hit, an interval miss, and negative skill when persistence
wins.

**Surfaces.** Forecast page "Track record" section and a report section of the same name.
Both currently state plainly that nothing is scoreable yet and why, rather than showing an
empty table or a zero. `summarize_scorecard()` on an empty scorecard returns `{}` — pinned
by test so it cannot fabricate a summary.

---

## 5 · Item 3 (WS2 — Optuna) — NOT STARTED

Conditioned on context comfortably allowing. It does not. Nothing is half-applied: both
completed items are committed, the suite is green at 417, the registry reconciles, and the
app smoke-tests clean.

---

## 6 · What remains

| # | Task | Notes |
|---|---|---|
| **1** | **Workstream 2 — tuning** | **Resume point.** LightGBM quantile port (P10/P50/P90, crossing-safe) + Optuna ≈100 trials with h-gapped early stopping, on each target's final recipe. **CatBoost is installed and in the pool.** Revenues and Expenditure first, stock via delta |
| 2 | WS6 ensembling; WS7 selection + conformal intervals + dual-probe + null distributions for the tree probe and for model-agreement | WS7 now owns three null-distribution studies |
| 3 | Send **T2** — forward auction/redemption calendar is the top ask | `docs/FISCAL_CALENDAR_SOURCES.md` ready |
| 4 | Ops P0, artifact validator, Phase-1 cleanup (untrack `.venv`/`.env` **without printing contents**) | Deferred since Phase 1 |
| 5 | Registry approval workflow | So `approved_by` can stop being null |
| 6 | Single TEST read + trust pack | Once. **Watch the Revenues `ratio` recipe** — its DEV gain exceeded its TRAIN gain |

### Open, carried forward

- **Treasury T1 (negative revenues) — OUTSTANDING.** Still blocks `log1p` for scaling.
- **Both flow targets still show no event signal** (Revenues 1.2255, Expenditure 1.0882).
  WS4 improved accuracy substantially and moved the signal not at all — consistent with
  every prior workstream.
- **Interval calibration**: nominal 80% captures ~50% on the largest third of days. WS7/CQR.
- **The track record is empty by design** and fills in as truth arrives. First scoreable
  date: 2025-08-07, once the data file moves past it.
- **C_DL stock-target collapse** — parked (Q6).

---

## 7 · Reproduction

```bash
git checkout model/excellence            # 4b4a515
./backend/.venv/bin/python -m pytest -q  # expect 417 passed
./backend/.venv/bin/python backend/run_forward_forecast.py
./backend/.venv/bin/python backend/run_publish_and_score.py
./backend/.venv/bin/python scripts/build_treasury_report.py
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from registry import verify_against_log as v;print(v())"    # ok: True, 12 metrics
```
