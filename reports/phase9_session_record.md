# Phase 2 — Multivariate & Sentinel Probe Study Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `68f2cda` (11 ahead of `origin/main` @ `6c22009`)
**Test suite:** 318 → **342** passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**Calendar version:** `4b480eae9c8f`
**TEST (2025) gated reads: 0** — two retrospective disclosures on record (phase-7 §3)
**PR #25** open, ready for review, not merged — **it now carries Phase 2b *and* 2c**; a separate
Phase 2c PR could not be opened (see §6a)

---

## 1 · The instruction

> Resume from reports/phase8_session_record.md §6. Verify premises first (model/excellence @ 2ab1c3e or
> its record child, suite = 318, log integrity {'n_rows': 72, 'ok': True}, calendar_version 4b480eae9c8f,
> TEST gated reads = 0 with two retrospective disclosures, data SHA 0b009fd0…, PR #25 state); if reality
> disagrees, say so and stop.
>
> Step 0: update reports/HANDOFF.md (phase-8 commits; resume → WS5; record the second reorder: WS5 and
> WS4 precede WS2 for the same tune-once reason as before). If PR #25 has been merged, rebase onto
> origin/main, re-run the suite, push with --force-with-lease; otherwise note it and continue.
>
> Then, one commit per coherent step, stopping cleanly with a session record if context runs low — never
> start an item you cannot finish:
>
> 1. Workstream 5 — multivariate, leak-safe. [debt-ops block tested alone first; lags ≥ 1; transforms
>    TRAIN-only; per-fold top-K; ablate vs the WS3-winning base; report MAE/skill AND the sentinel]
> 2. Workstream 4 — target scaling: raw vs asinh vs ratio-to-trailing-level …
> 3. The sentinel-probe measurement study … pre-registered reading …
> 4. Only if context comfortably allows: begin workstream 2 …
>
> At session end: push, open a ready-for-review PR to main titled "Phase 2c — multivariate, target
> scaling, sentinel probe study"; record the number; do not merge. TEST stays sealed. End with
> reports/phase9_session_record.md.

---

## 2 · Premise verification — all held

| Premise | Measured |
|---|---|
| `model/excellence` @ `2ab1c3e` **or its record child** | ✅ `136ae76`, the record child |
| suite = 318 · log integrity `{'n_rows': 72, 'ok': True}` · `calendar_version` `4b480eae9c8f` | ✅ |
| TEST gated reads = 0, two retrospective disclosures · data SHA `0b009fd0…` | ✅ |
| PR #25 state | ✅ open, **not merged** — so no rebase, noted and continued |

First session in this sequence with no premise drift.

---

## 3 · Workstream 5 — multivariate (`0c5d664`)

Full detail in [reports/ws5_multivariate.md](reports/ws5_multivariate.md).

### The debt-operations hypothesis fails — and the reason changes what to ask Treasury

`docs/DATA_SEMANTICS.md` §1 measured the 72 negative-`Revenues` days as netting in `Increase in
liabilities` (64/72, corr **0.971**) and `Domestic` (65/72, **0.969**). A `debt_ops` block was
therefore tested **alone, before** any broad pool. It made every target worse: Revenues
**−0.75%**, Expenditure **−1.47%**, stock −0.05%.

**That 0.971 is contemporaneous.** It says that on a day when debt operations net negative,
`Revenues` does too. It does not say yesterday's debt operations predict next week's. Every
feature is lagged ≥ 1 step, as it must be, and a lagged realised value cannot anticipate a
future auction. No amount of lagging converts a same-day accounting identity into a forecast.

So the ask of Treasury sharpens: the **forward auction and redemption calendar**, not the
realised debt lines — which we already have and which do not help.

### Every block, TRAIN folds

```
block          Revenues   Expenditure    Stock
cross            -0.25%       -0.49%    +0.69%   <- only positive result
debt_ops         -0.75%       -1.47%    -0.05%
tax              -3.39%            -    -0.48%
spend                 -       -1.22%         -
broad (160)      -2.50%       -1.85%    -1.72%
```

**Nothing is adopted for the flows.** `cross` is adopted for `State budget balance`: +0.69%
TRAIN, **+0.42% DEV** (194,926,464 → 194,104,922; skill 19.67% → 20.01%). Mechanically
sensible — the stock is cumulative revenues minus expenditure.

The three no-exog DEV rows **reproduced the WS3 DEV figures exactly**, confirming the two
harnesses agree.

### Leak safety is structural, not procedural

Lags ≥ 1 enforced (lag 0 raises). **No statistic is fit** — plain lags only, so there is
nothing that could be fit on the wrong window. **No feature selection** — the brief permitted
per-fold top-K; none was used, so "never on the whole series" holds trivially. If the broad
pool is ever pruned, per-fold selection is the follow-up and is **not** implemented here.

13 tests; the load-bearing one mutates every exogenous column's future, asserts past features
byte-identical, then asserts the mutation *is* visible after the cut.

---

## 4 · Sentinel probe study (`68f2cda`)

Full detail in [reports/sentinel_probe_study.md](reports/sentinel_probe_study.md).
**Pre-registered reading honoured:** threshold not revisited, no default changed, decision
deferred.

### My WS3 hypothesis was wrong, and the negative result survives anyway

| Control | ridge | tree | forest |
|---|---:|---:|---:|
| null | 1.020 | 1.022 | 1.016 |
| linear | **5.380** | 2.379 | 3.130 |
| interaction-only | **1.295** ✗ | **5.137** ✓ | **4.933** ✓ |
| rare categorical | 2.281 | 2.247 | 2.287 |

**The blind spot is real:** a pure interaction with zero linear signal reads 1.295 under ridge
(missed) and ~5.0 under both trees. Now regression-tested.

**But it is not the blind spot I claimed.** `ws3_fiscal_calendar.md` §4 hypothesised that group
A's *rare small-cardinality event* was what ridge could not exploit. On that exact shape all
three probes agree to within 0.04 — a rare binary indicator is one coefficient. That
explanation was wrong.

Also: neither instrument dominates. On linear signal ridge reads *higher* than the trees.

### The real targets

| Target | Feature set | ridge | tree | forest | Verdict |
|---|---|---:|---:|---:|---|
| Revenues | final | 1.167 | **1.421** | 1.260 | neither detects |
| Expenditure | final | 1.071 | **1.396** | 1.155 | neither detects |
| State budget balance | final | 3.992 | 3.182 | 3.580 | both detect |

The tree probe reads ~0.25–0.33 higher on the flows — ridge *does* modestly understate, so the
hypothesis was right in direction — but the highest reading across three instruments and two
feature sets is **1.421**, short of 1.50. **What was a finding from one linear probe is now a
finding from three.**

**WS3 is partially vindicated:** its features moved the tree probe far more than ridge
(Revenues +0.264 vs +0.125; Expenditure +0.233 while ridge *fell* 0.007). The fiscal calendar
did add tree-exploitable structure the default sentinel scored as nothing — consistent with its
real MAE gain, still too small to matter.

### Deferred to you

Whether to report **both** probes routinely, each against its own threshold. On this evidence
that is strictly more informative than either alone. Recommendation when wanted: report both,
keep 1.50 for ridge, derive a tree threshold from its own null distribution before gating on
it. Nothing this session changes if you decline — no model was selected using a tree probe.

---

## 5 · Item 2 (WS4, target scaling) — NOT STARTED, and I reordered

**I deviated from the instruction's order**, and the reason should be on the record.

WS4 requires threading a target transform (asinh, ratio-to-trailing-level) through the
**prediction path**: transform the target, fit, then invert every prediction before
`predictions_long.csv` is written. That path is where `origin_value` and `y_true` are emitted,
and those two columns are what the unified persistence ruler is computed from. An error there
would not fail loudly — it would silently corrupt the ruler that four families and nine
sessions of work are calibrated against.

With the context remaining I could finish **one** of items 2 and 3 properly. I chose item 3
because it is self-contained (no pipeline changes), and because its result changes how every
number already produced should be read. Starting WS4 and stopping mid-implementation would have
left the ruler path half-modified, which is the outcome the standing rule exists to prevent.

WS4 remains fully specified and is the top of the queue.

---

## 6 · Item 4 (WS2) — NOT STARTED

Conditioned on context comfortably allowing. It does not.

---

## 6a · The Phase 2c PR could not be opened

The instruction asked for a ready-for-review PR titled *"Phase 2c — multivariate, target
scaling, sentinel probe study"*. `gh pr create` refused:

```
a pull request for branch "model/excellence" into branch "main" already exists: .../pull/25
```

**GitHub permits one open pull request per head→base pair.** PR #25's head is
`model/excellence`, and that branch has advanced, so **PR #25 has silently grown to contain
Phase 2c as well** — 11 commits rather than the 4 its description covers. That is a real
reviewability problem, not just a bookkeeping one: a reviewer reading #25's description would
not know WS3, WS5 and the probe study are in the diff.

What I did instead of inventing a PR number: added a **non-destructive scope-disclosure
comment** to PR #25 listing the extra commits and pointing at the three workstream reports. I
did not overwrite #25's description, since it may be mid-review, and I did not rewrite the
pushed branch.

Two ways to get separate tranches, both requiring a decision:

1. **Merge PR #25**, after which the next push can open a clean Phase 2c PR.
2. **Split**: reset `model/excellence` to `136ae76` (the Phase 2b tip) and move the Phase 2c
   commits to a new branch. This rewrites an already-pushed branch, so it needs explicit
   approval.

Recorded here because the same thing will recur every session that pushes to
`model/excellence` while an earlier PR is unmerged.

---

## 7 · What remains

| # | Task | Notes |
|---|---|---|
| **1** | **Workstream 4 — target scaling** | **Resume point.** raw vs asinh vs ratio-to-trailing-level, statistics per fold; stock keeps delta. `log1p` inapplicable while T1 is open. **Touches the prediction path — verify the unified ruler is unchanged before and after** |
| 2 | Workstream 2 — LightGBM quantile + Optuna ~100 trials | After WS4, on the final recipe. Build on `LightGBM_L1` |
| 3 | Send **T2** — now better motivated | Ask for the **forward** auction/redemption calendar. WS5 proved the realised debt lines do not substitute for it |
| 4 | Your decision on dual-probe sentinel reporting | §4. Nothing blocks on it |
| 5 | Workstreams 6, 7 | 7 owns the top-tercile coverage fix (CQR) |
| 6 | Phase-1 Steps 4/5/7 — ops P0, Agent contract + validator, cleanup (untrack `.venv`/`.env` **without printing contents**) | Deferred since Phase 1 |

### The finding that should shape what comes next

**Four workstreams (WS1, WS3, WS5) and a three-instrument probe study have now failed to find
signal in the flow targets.** Revenues 1.167–1.421, Expenditure 1.071–1.396, threshold 1.50.

`ws5_multivariate.md` §6 states the option that deserves stating plainly: **a
five-business-day-ahead forecast of daily Treasury flows may not exist in this data beyond the
level of central tendency.** Three negative results and a robustness study are evidence *for*
that proposition, not against it. Reporting the flows honestly — with intervals that admit it —
is a legitimate deliverable, and possibly the correct one.

The one untried input that could overturn it is Treasury's forward auction calendar, because it
is the only candidate that is *knowable in advance* and targets the specific days that are
unpredictable. That is now the highest-value item in the whole backlog, and it is not a
modelling task.

### Open, carried forward

- **Treasury T1 (negative `Revenues`) — answer OUTSTANDING.**
- **Top-tercile interval coverage 49–54%** against nominal 80%. WS7's CQR.
- **C_DL stock-target collapse** — parked (Q6).
- **PR #25 open, not merged, and now carrying two tranches** (§6a). No PR #26 exists.

---

## 8 · Reproduction

```bash
git checkout model/excellence            # 68f2cda
./backend/.venv/bin/python -m pytest -q  # expect 342 passed
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from experiment_log import verify_log_integrity as v;print(v())"
```

Every WS5 figure is a logged row. The probe-study figures are generated by the script in
`reports/sentinel_probe_study.md` §Reproduction and are pinned by
`backend/tests/test_sentinel_probe.py`; they are instrument diagnostics rather than model
results, so they are reported there rather than in `experiments/log.csv`.
