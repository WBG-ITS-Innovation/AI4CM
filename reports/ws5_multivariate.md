# Workstream 5 — Multivariate (exogenous Treasury lines)

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Data SHA-256:** `0b009fd0…5361f1`
**Calendar version:** `4b480eae9c8f` · **Suite:** 331 passing · **TEST (2025) reads: 0**
All 22 figures below are logged in `experiments/log.csv`.

---

## The headline

**No exogenous block helps either flow target. The debt-operations hypothesis — the specific
mechanism this workstream was built to test — fails.**

| Target | Best exog block | TRAIN vs WS3 base | Sentinel |
|---|---|---:|---|
| Revenues | *none* (all blocks worse) | best was `cross` at **−0.25%** | 1.181 → 1.173 (down) |
| Expenditure | *none* (all blocks worse) | best was `cross` at **−0.49%** | 1.152 → 1.140 (down) |
| State budget balance | **`cross`** | **+0.69%** | 2.685 → 2.819 |

The flow sentinel did not move up. On the broad pool it moved **down**, to 1.043 on Revenues.
Three workstreams (WS1 objectives, WS3 calendar, WS5 multivariate) have now failed to lift it.

---

## 1 · The specific hypothesis, and why it failed

`docs/DATA_SEMANTICS.md` §1 measured that the 72 business days on which `Revenues` prints
negative are driven by netting in the debt-operation lines: `Increase in liabilities` negative
on 64/72 with correlation **0.971**, `Domestic` on 65/72 with **0.969**. Those are the days the
flow targets cannot predict. So a `debt_ops` block was tested **alone, first**, before any
broad pool — if the mechanism were real, four columns should show it.

**It made every target worse:**

| Target | WS3 base | + `debt_ops` | Δ |
|---|---:|---:|---:|
| Revenues | 39,862,167 | 40,161,027 | **−0.75%** |
| Expenditure | 35,069,403 | 35,585,195 | **−1.47%** |
| State budget balance | 131,168,230 | 131,239,950 | −0.05% |

### The reason is instructive, and it changes what to ask for

**That 0.971 correlation is contemporaneous — same-day.** It says that on a day when debt
operations net negative, `Revenues` also nets negative. It does *not* say that yesterday's
debt operations predict next week's. Every feature here is lagged by at least one step, as it
must be, and a lagged realised value cannot anticipate a future auction.

No amount of lagging converts a same-day accounting identity into a forecast. The only thing
that can anticipate an auction is a **schedule known in advance**.

So WS5's failure sharpens rather than weakens the WS3 recommendation: the ask of Treasury is
the **forward auction and redemption calendar**, not the realised debt lines — which we already
have, and which do not help. This is the difference between knowing what happened and knowing
what is scheduled, and only the second is a forecasting input.

---

## 2 · All blocks, TRAIN-internal folds (2019–2023, n=1,304)

Base is each target's WS3-winning fiscal-calendar set. Vehicles: `LightGBM_L1` for the flows,
`HistGBDT_L1` for the stock. Δ is versus that base; negative means the block **hurt**.

### Revenues — base 39,862,167 (skill 40.56%, sentinel 1.181)

| Block | MAE | Skill | Δ | Sentinel |
|---|---:|---:|---:|---:|
| `cross` | 39,962,084 | 40.41% | −0.25% | 1.173 |
| `debt_ops` | 40,161,027 | 40.11% | −0.75% | 1.166 |
| `broad` (160 features) | 40,858,157 | 39.07% | −2.50% | **1.043** |
| `tax` | 41,214,810 | 38.54% | −3.39% | 1.172 |

### Expenditure — base 35,069,403 (skill 35.26%, sentinel 1.152)

| Block | MAE | Skill | Δ | Sentinel |
|---|---:|---:|---:|---:|
| `cross` | 35,239,615 | 34.95% | −0.49% | 1.140 |
| `spend` | 35,497,291 | 34.47% | −1.22% | 1.172 |
| `debt_ops` | 35,585,195 | 34.31% | −1.47% | 1.154 |
| `broad` | 35,718,780 | 34.06% | −1.85% | 1.105 |

### State budget balance — base 131,168,230 (skill 14.62%, sentinel 2.685)

| Block | MAE | Skill | Δ | Sentinel |
|---|---:|---:|---:|---:|
| **`cross`** | **130,265,217** | **15.20%** | **+0.69%** | 2.819 |
| `debt_ops` | 131,239,950 | 14.57% | −0.05% | 2.704 |
| `tax` | 131,799,347 | 14.21% | −0.48% | 2.889 |
| `broad` | 133,422,399 | 13.15% | −1.72% | 3.290 |

**The `cross` result on the stock target is the one positive finding**, and it is mechanically
sensible: `State budget balance` is (cumulatively) revenues minus expenditure, so lagged values
of its own two components carry real information about its change. It is also small.

### The broad pool illustrates what the sentinel measures

160 exogenous features made MAE worse on all three targets while moving the sentinel in
**opposite directions**: down on Revenues (1.181 → 1.043) and up on the stock (2.685 → 3.290).
The sentinel is a Ridge probe, so it reports the signal-to-noise ratio *available to a linear
model* on the feature set. Diluting an already weak signal with 160 mostly irrelevant columns
lowers it; on a target that has genuine linear structure, more columns give the probe more to
work with even as the tree overfits. Neither movement is a statement about forecast quality —
which is exactly why both numbers get reported together, and why §3's probe study matters.

---

## 3 · DEV confirmation

| Target | Config | MAE | Skill | Sentinel |
|---|---|---:|---:|---:|
| Revenues | WS3 base (no exog) | 52,417,152 | 40.65% | 1.226 |
| Expenditure | WS3 base (no exog) | 51,602,951 | 29.42% | 1.088 |
| State budget balance | WS3 base | 194,926,464 | 19.67% | 6.926 |
| State budget balance | **WS3 base + `cross`** | **194,104,922** | **20.01%** | 7.006 |

The three no-exog rows **reproduce the WS3 DEV figures exactly** — a useful check that the WS5
harness and the WS3 harness agree.

`cross` on the stock target holds up on DEV: **+0.42%**, skill 19.67% → 20.01%. Consistent in
sign with TRAIN (+0.69%), smaller in size. It is adopted for the stock target's recipe and
nothing is adopted for the flows.

---

## 4 · Leak safety

Three structural properties, each with a test rather than an assertion:

1. **Every feature is lagged ≥ 1 step.** `lags=(1, 5, 21)` plus a calendar-aligned
   previous-month variant built from strictly earlier rows. A lag of 0 raises `ValueError` —
   a same-day flow value is not reliably known at the origin, and that is the classic
   multivariate leak.
2. **No statistic is fit.** Plain lags only: no scaling, no encoding, no imputation fit across
   rows. Nothing is fit, so there is nothing that could be fit on the wrong window. The leak
   argument is structural, not procedural.
3. **No feature selection.** No top-K, no correlation screen. The brief permitted per-fold
   top-K; none was used, so the "never on the whole series" requirement is satisfied
   trivially. **If the broad pool is ever pruned, per-fold selection is the follow-up and is
   not implemented here** — it would have to live inside the fold loop.

The load-bearing test mutates every exogenous column's future and asserts past features are
byte-identical, then asserts the mutation *is* visible after the cut so it cannot pass
vacuously. A further test confirms the target never appears in its own exogenous set — for the
`cross` block that would hand the model its own column.

13 tests. Also pinned: an empty block raises rather than silently producing zero features,
because a zero-feature run would read as "the block did not help" when the truth is "the block
was never applied".

---

## 5 · Information-set justification per feature family

| Feature | Justification |
|---|---|
| `x_<line>_lag{1,5,21}` | Exogenous line value dated 1, 5 or 21 business days before the origin; all reported and known by then |
| `x_<line>_aligned_prev_month` | Same business-day-of-month in the previous month, taken from a strictly earlier row; index map asserted to reference only past positions |

---

## 6 · Verdict

**Nothing is adopted for the flow targets.** Their recipe remains the WS3-winning fiscal
calendar. The exogenous Treasury lines, including the ones the negative-`Revenues` analysis
pointed at, do not predict them five business days ahead.

**`cross` is adopted for `State budget balance`** (+0.42% DEV).

The flow sentinel now stands at **1.226** (Revenues) and **1.088** (Expenditure) against a
threshold of 1.50, unmoved by three consecutive workstreams. On the evidence, the remaining
candidates are not modelling changes:

1. **The forward auction/redemption calendar from Treasury** (T2) — now better motivated than
   before, because WS5 showed the realised debt lines do not substitute for it.
2. **Accepting that these targets may not be predictable at h=5 beyond the level of central
   tendency**, and reporting them that way — with intervals honest about it — rather than
   continuing to search.

That second option deserves stating plainly: it is a legitimate finding, not a failure. A
five-business-day-ahead forecast of daily Treasury flows may simply not exist in this data, and
three workstreams of negative results are evidence for that rather than against it.

---

## 7 · Reproduction

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q          # expect 331 passed
```

Runs are `ConfigBML(target=…, horizon=5, min_train_years=4, variant="univariate",
model_filter=<vehicle>, fiscal_groups=<WS3 winner>, exog_blocks=<block>, exog_lags=(1,5,21))`
with `eval_end="2023-12-31"` for TRAIN and `eval_start="2024-01-01", eval_end="2024-12-31"` for
DEV. Each run asserted its own window membership before any number was recorded.
