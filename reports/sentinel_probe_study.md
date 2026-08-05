# Sentinel probe study — is the Ridge probe hiding tree-exploitable signal?

**Date:** 2026-08-05 · **Branch:** `model/excellence` · **Suite:** 342 passing
**Pre-registered reading:** this is a study of the **instrument**. The 1.50 threshold is
**not revisited this session** regardless of outcome, and any threshold decision is deferred.

---

## Why this study exists

`signal_sentinel` fits **Ridge** — a linear model. Three workstreams have now improved MAE on
the flow targets while the sentinel ratio barely moved. In `reports/ws3_fiscal_calendar.md` §4
I raised the hypothesis that the ratio might be *understating* signal a tree can use, and named
a specific mechanism: group A's content is a **rare small-cardinality event** (a deadline moved
by 1–3 days), which I claimed a linear probe could barely exploit.

This study tests that. It reaches two conclusions, and one of them **falsifies my own
hypothesis**.

---

## Design

Three fixed, untuned instruments — a tuned probe would make the ratio a function of the tuning:

| Probe | Instrument |
|---|---|
| `ridge` | `Ridge(alpha=1.0)` — the historical default |
| `tree` | `DecisionTreeRegressor(max_depth=4)` — the minimal nonlinear instrument |
| `forest` | 100 × `max_depth=4` trees — same hypothesis class, far lower variance, so a null result is more trustworthy |

Four synthetic controls (n=1,400, 12 features, 80/20 split) then the three real targets on both
their pre-WS3 and final feature sets, TRAIN (≤2023) → DEV (2024).

---

## Part 1 — Synthetic controls

| Control | ridge | tree | forest | What should happen | Happened? |
|---|---:|---:|---:|---|---|
| **null** (pure noise) | 1.020 | 1.022 | 1.016 | nothing detects | ✅ all False |
| **linear** | **5.380** | 2.379 | 3.130 | all detect | ✅ all True |
| **interaction-only** | **1.295** | **5.137** | **4.933** | ridge misses, trees catch | ✅ **ridge False, trees True** |
| **rare categorical** (8% event) | 2.281 | 2.247 | 2.287 | — | all True, **near-identical** |

Two findings.

### The Ridge probe has a real blind spot — pure interactions

A two-way interaction with **zero linear signal** and a strong effect gives ridge **1.295**
(below threshold, missed) while both tree probes read ~5.0. The blind spot is real, large, and
now pinned by a regression test
(`test_ridge_misses_interaction_only_signal_and_trees_do_not`).

### But it is *not* the blind spot I hypothesised

On the **rare categorical** control — the shape I claimed in WS3 was the problem — all three
probes agree to within 0.04 (2.281 / 2.247 / 2.287). A linear model handles a rare binary
indicator perfectly well: it is a single coefficient. **My WS3 explanation was wrong in its
specific mechanism.**

Worth noting the reverse case too: on genuinely linear signal, ridge reads *higher* than the
trees (5.380 vs 2.379). Neither instrument dominates; they measure different hypothesis classes.

---

## Part 2 — The real targets

TRAIN (≤2023) → DEV (2024). Threshold 1.50.

| Target | Feature set | ridge | tree | forest | Verdict |
|---|---|---:|---:|---:|---|
| Revenues | pre-WS3 | 1.042 | 1.157 | 1.171 | neither detects |
| Revenues | **final** | 1.167 | **1.421** | 1.260 | **neither detects** |
| Expenditure | pre-WS3 | 1.078 | 1.163 | 1.105 | neither detects |
| Expenditure | **final** | 1.071 | **1.396** | 1.155 | **neither detects** |
| State budget balance | pre-WS3 | 3.562 | 3.319 | 3.440 | both detect |
| State budget balance | **final** | 3.992 | 3.182 | 3.580 | both detect |

### The negative result survives a better instrument

**No probe finds signal in either flow target.** The tree probe reads consistently higher on
the flows — Revenues 1.167 → 1.421, Expenditure 1.071 → 1.396, a gain of ~0.25–0.33 — so ridge
*does* modestly understate, exactly as hypothesised in direction. It is not close to enough to
change the verdict: the highest reading across three instruments and two feature sets is
**1.421**, still short of 1.50.

So the conclusion from WS1, WS3 and WS5 stands, and now stands on firmer ground: **the flow
targets have no detectable signal under any of the three instruments.** That was previously a
finding from one linear probe; it is now a finding from three.

### WS3's features did add nonlinear structure that Ridge largely missed

The tree probe rises much more than ridge when the WS3 features are added:

| Target | ridge Δ | tree Δ |
|---|---:|---:|
| Revenues | +0.125 | **+0.264** |
| Expenditure | −0.007 | **+0.233** |

On Expenditure the ridge reading actually *fell* while the tree reading rose by 0.233. So the
fiscal calendar contributed real tree-exploitable structure that the default sentinel scored as
nothing — which partially vindicates WS3 (its 4–6% MAE gain was not an accident) while
confirming the gain was too small to matter for the signal question.

### The stock target is where ridge reads high

On `State budget balance` ridge reads **above** the trees (3.992 vs 3.182). The stock target's
structure is substantially linear — consistent with it being a cumulative sum of two flows —
which is also why it was always the target with genuine signal.

---

## Part 3 — What this does and does not license

**Established:**
1. The Ridge probe misses pure-interaction signal. Real, and now regression-tested.
2. It does **not** miss rare-categorical signal. My WS3 hypothesis was wrong.
3. On the actual flow targets, a nonlinear probe reads ~0.25–0.33 higher and still finds
   nothing. The negative result is robust to the instrument.
4. Neither probe dominates: ridge is stronger on linear signal, trees on interactions.

**Not established, and deliberately not acted on:**

- **The 1.50 threshold is unchanged**, and this study is not an argument for moving it. A
  threshold calibrated for one instrument does not transfer to another: the same feature set
  reads 1.167 under ridge and 1.421 under a tree, and on linear signal the ordering reverses.
  Adopting a tree probe would require re-deriving the threshold from its own null distribution,
  which this study does not do.
- **No default was changed.** `DEFAULT_PROBE` is still `ridge`, pinned by a test, because
  changing it would silently alter the meaning of every ratio in `experiments/log.csv`.
- Every sentinel result now records which `probe` produced it, so a ratio can never be compared
  against a threshold calibrated for a different instrument by accident.

### The decision that is yours

Whether to make the sentinel report **both** probes routinely — a linear reading and a
nonlinear one, each against its own threshold — rather than picking one. On the evidence here
that would be strictly more informative than either alone, since the two disagree in both
directions depending on the signal's shape.

My recommendation, for when you want it: report both, keep 1.50 for ridge, and derive a
separate tree threshold from a null-distribution study before gating on it. Nothing in this
session's results changes if you decline — no model was selected or rejected using a tree probe.

---

## Reproduction

```bash
git checkout model/excellence
./backend/.venv/bin/python -m pytest -q backend/tests/test_sentinel_probe.py   # 11 passed
```

Controls are generated with `numpy.random.default_rng(7)` at n=1,400 / 12 features; the
regression tests use seeds 3 and 11 at n=900 / 8 features. Real-target matrices are built with
the pipeline's own `lag_window_features` + `calendar_exog` (+ `build_exog_features` for the
stock target's `cross` block), split at 2024-01-01 and 2025-01-01. `signal_sentinel(...,
probe=…)` is the single entry point; all three probes are deterministic, asserted by test.
