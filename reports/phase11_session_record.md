# Phase 2 — WS4 Robustness & WS2 Tuning Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `20d4469` (13 ahead of `origin/main` @ `863f967`)
**Test suite:** 417 → **441** passing
**Data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**Experiments log:** 135 rows, integrity ok · **TEST (2025) gated reads: 0**
**PR #26** open, not merged — reused, not duplicated

---

## 1 · Premise verification — all held

| Premise | Measured |
|---|---|
| branch tip | ✅ `98c93e1` |
| suite = 417 · log integrity ok (105 rows) · registry 12/12 | ✅ |
| TEST gated reads = 0 · data SHA `0b009fd0…` | ✅ |
| PR state | ✅ #26 **open** → reused for this head→base |

---

## 2 · Item 1 — no re-issue was made, and creating one would have been harmful

Re-ran the forward forecast and diffed it against published issue `2025-08-06`: **worst
relative difference 0.000e+00** across all 15 predictions, with identical transforms and
target dates.

The published issue already carries the WS4 champions (Revenues on `ratio`) because it was
retained *after* WS4 landed, and the forward run is deterministic. Issuing a new date with
byte-identical content would put duplicate rows in the scorecard, which would then be
double-counted when truth arrives — actively harmful rather than neutral. **So nothing was
published and retention was not touched.**

Two tests added instead: one locks the three-way agreement (registry ↔ published manifest ↔
forward run) on `recipe_id` **and** `target_transform` per target, because those surfaces are
written at different times by different code paths and a mismatch would attach the wrong DEV
accuracy to published numbers; the other forbids duplicate issue dates.

---

## 3 · Item 2 — WS4 robustness · `9cd7e9b`

Full detail in [reports/ws4_robustness.md](reports/ws4_robustness.md).

### The explanation was confirmed, quantitatively and out of sample

Revenues: **corr(level drift, ratio advantage) = +0.987** across five TRAIN-internal
rolling-origin windows.

| Window | Drift | ratio advantage |
|---|---:|---:|
| 2019 | 13.5% | **+1.30%** |
| 2020 | 18.7% | +1.67% |
| 2021 | 34.1% | +4.15% |
| 2022 | 70.2% | +16.99% |
| 2023 | 84.1% | **+24.14%** |

DEV drift is **81.7%**. A line fitted on the five TRAIN windows *alone* predicts **+21.83%**
against a measured **+25.73%** — a 4pp residual on a five-point extrapolation with no 2024
data in the fit. **The DEV gain is not anomalous; it is the drift relationship doing what it
does.**

### Verdict under the pre-registered reading

**Regime-dependent, so quoted that way.** The 55.92% DEV skill describes a high-drift period;
in a flat period expect ~1–2% over raw. Encoded as `scaling_caveat` in the registry with two
tests (present on Revenues, absent on the raw targets), so the number cannot travel without
its condition.

**Adoption stands on a separate, weaker claim:** the advantage was **positive in every window
tested**, including the lowest-drift one. Regime-dependent in *magnitude*, not in *sign* — a
small-positive floor rather than a downside. The report states this as weaker than "genuine
property", because that is what the evidence supports.

### Something I could not explain, recorded as such

Expenditure (corr −0.506) and the stock target (corr −0.813) are **hurt** by `ratio`, and hurt
*more* as drift rises — despite drifting more than Revenues (up to 102% and 152%). So "the
divisor absorbs level shifts" is necessary but not sufficient. The plausible extra condition
is whether fluctuations scale with level; **I did not test it**, so it is a hypothesis with
the cheap test named (correlate rolling dispersion against rolling level, check the sign),
assigned to WS7.

**What to watch at the single TEST read:** whether 2025's drift is high or flat. If flat,
Revenues' advantage over raw should be small — and that must not be read as the transform
failing.

---

## 4 · Item 3 — WS2

### Infrastructure: COMPLETE · `9ba1144`

`backend/tuning.py` plus the E_QUANTILE port, with the two correctness properties tested.

**Crossing safety.** `crossing_safe()` sorts each row's quantiles — a p90 below a p50 is an
invalid interval, not a pessimistic one, and every coverage statistic from it is meaningless.
`count_crossings()` **reports** how often the repair fired rather than swallowing it, because
a model that crosses constantly is misconfigured and silent sorting would hide that.

**H-gapped early stopping.** Row *t* carries target *y(t+h)*, so the last *h* fit rows have
answers inside the validation block and the naive stopping decision is made against
partly-seen data. `gapped_split()` drops those rows. Tests assert the gap is exactly *h* for
h ∈ {1,5,10,21}, that the slices are disjoint, and that validation is the most recent
contiguous block.

**Units.** `FoldData` separates the training target (possibly transformed) from truth in
original units and carries an `inverse`. Load-bearing: on Revenues the ratio transform divides
by a ~5e7 level, so 0.1 in ratio space and 0.1 in lari differ by seven orders of magnitude. A
test asserts the objective inverts before measuring and that omitting the inverse makes the
objective absurd.

`LGBMQuantile` registered in E_QUANTILE's own registry, run end to end by a test that asserts
p10 ≤ p50 ≤ p90 on real pipeline output. Adding it broke a test that hard-coded
`{GBQuantile, ResidualRF}`; rather than widen the literal, the family now exposes
`registry_models()` as a single source of truth so a model cannot be added to the run loop
without appearing in the reports and tests that enumerate the family.

### Search: Revenues COMPLETE (and it did not help); other two targets in flight

**Revenues, 100/100 trials, 0 failed, 1,592s.** Best model `LGBMQuantile`.

| | TRAIN objective | DEV MAE | DEV skill |
|---|---:|---:|---:|
| Untuned incumbent (WS4 `ratio`) | 35,456,541 | **38,931,956** | **55.92%** |
| Tuned (100 Optuna trials) | **35,114,562** | 40,148,659 | 55.78% |
| Change | **+0.96%** | **−3.13%** | −0.14pp |

**Tuning gained ~1% on TRAIN and lost ~3% on DEV.** That is the signature of the
hyperparameter search mildly overfitting the TRAIN folds it was optimised against — 100
trials is enough to find configurations that suit five specific fold splits. **On this
evidence the tuned Revenues configuration must not be promoted.** The untuned incumbent
stays.

Interval coverage came out at 83.2% against a nominal 80%, and **0 quantile crossings** were
repaired — so the crossing-safe port is behaving, on this target at least.

> ### ⚠️ The sentinel figure from this harness is INVALID — do not quote it
>
> The run reported `sentinel=1.0000`, exactly. That is an artefact, not a measurement. My
> harness passed the **ratio-transformed** training target and the **original-scale** test
> truth into `signal_sentinel()`, so both the real-target and shuffled-target errors are
> dominated by the same units mismatch and their ratio collapses to 1. It measures nothing.
>
> Whether tuning moved the sentinel is therefore **still unmeasured**. Fixing it means
> passing consistently-scaled targets (or inverting the training target before the probe) and
> re-running the probe only — the search itself does not need repeating. The trustworthy
> sentinel readings remain the WS3/WS5 ones: Revenues 1.2255, Expenditure 1.0882.

### Remaining targets: in flight, results not yet incorporated

The Optuna study is running in the background: 100 trials per target across
`{LGBMQuantile, CatBoostQuantile}` on TRAIN-internal folds, then one DEV confirmation each.
Expenditure and the stock target had not completed when this record closed.

**Nothing from the search has been reported, promoted, or written into the registry.** No
`reports/ws2_tuning.md` exists, because writing one now would mean either waiting past the
point where I can finish cleanly or publishing partial results as if they were the study.

**Durability:** `scripts/ws2_tune.py` is version-controlled (`20d4469`) precisely so this is
resumable — a 45-minute search whose driver lives in a session-scoped scratch directory cannot
be reproduced. Each target's DEV confirmation is appended to `experiments/log.csv` (tracked)
as that target completes, with `study: ws2_tuning` in `params`, so an interrupted run keeps
whatever it finished.

**To resume:**

```bash
TARGETS="['Revenues','Expenditure','State budget balance']" TRIALS=100 \
  SC=/tmp/ws2 ./backend/.venv/bin/python scripts/ws2_tune.py
```

It skips targets already present in `$SC/ws2_results.csv`; to check what already landed,
filter `experiments/log.csv` on `study: ws2_tuning`.

---

## 5 · Two process notes

**The standing rule was defeated by a pipe.** `9ba1144` first claimed 441 passing while the
suite was **440 passed + 1 failed**: piping `pytest` to `tail` makes the pipeline's exit
status that of `tail`, so the `&&` guard did not short-circuit. The failure was real — adding
`LGBMQuantile` broke a test asserting exactly two registry models. Fixed the test properly,
re-ran with a bare `pytest` and `EXIT=0` checked explicitly, and amended the commit. **Rule
sharpened: check the exit code, not the tail.**

**`ast.parse()` proves syntax, not completeness.** The first assembly of
`scripts/ws2_tune.py` passed `ast.parse()` while **missing its `import` line** — a BSD `sed`
incompatibility had dropped it. It would have failed at runtime. Rebuilt and verified by
grepping for the import, `full_study` and the main guard rather than trusting the parse.

---

## 6 · What remains

| # | Task | Notes |
|---|---|---|
| **1** | **Finish the WS2 search and report it** | **Resume point.** Driver committed; results land in `experiments/log.csv`. **Revenues is done and LOST on DEV (−3.13%) — do not promote it.** Two targets remain. **Fix the sentinel harness bug first** (§4) or the "did the sentinel move" question stays unanswerable |
| 2 | WS6 ensembling; WS7 selection + conformal intervals | WS7 now owns **four** null-distribution / diagnostic studies: tree probe, model agreement, the ratio-transform "fluctuations scale with level" hypothesis, and CQR |
| 3 | Send **T2** — forward auction/redemption calendar is the top ask | Ready |
| 4 | Ops P0, artifact validator, Phase-1 cleanup (untrack `.venv`/`.env` **without printing contents**) | Deferred since Phase 1 |
| 5 | Registry approval workflow | So `approved_by` can stop being null |
| 6 | Single TEST read + trust pack | Once. Watch 2025's level drift — see §3 |

### Open, carried forward

- **Treasury T1 — OUTSTANDING.** Still blocks `log1p`.
- **Both flow targets still show no event signal** (Revenues 1.2255, Expenditure 1.0882). WS4
  improved Revenues' accuracy substantially and moved the signal not at all.
- **Interval calibration**: nominal 80% captures ~50% on the largest third of days.
- **Track record empty by design**; first scoreable date 2025-08-07 once the data moves past it.
- **C_DL stock-target collapse** — parked (Q6).

---

## 7 · Reproduction

```bash
git checkout model/excellence            # 20d4469
./backend/.venv/bin/python -m pytest -q; echo "EXIT=$?"   # 441 passed, EXIT=0
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from registry import verify_against_log as v;print(v())"  # ok: True
```
