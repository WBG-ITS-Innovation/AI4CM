# Phase 2 — Experiments Log & Workstream 1 Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `db27cd4` (3 ahead of `origin/main` @ `6c22009`)
**Test suite:** 273 → **294** passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**TEST (2025) gated reads: 0** — two retrospective disclosures on record, see §3
**Experiments log:** 38 rows, integrity `{'n_rows': 38, 'ok': True, 'problems': []}`

---

## 1 · The instruction

> Resume from reports/phase6_session_record.md §5. Verify premises first (origin/model/excellence @
> e9c775b — the phase-6 record commit — suite = 273, TEST reads = 0, data SHA 0b009fd0…, origin/main @
> 4a925ac, PR #24 state); if reality disagrees, say so and stop.
>
> Step 0: update reports/HANDOFF.md with the phase-6 commits (resume point → item 2). Sync: if PR #24 has
> been merged, rebase model/excellence onto origin/main, re-run the suite, and push with
> --force-with-lease; if not merged, note it and continue. Add an honest note to the TEST-access ledger
> and HANDOFF: the item-1f verification evaluated incumbent models on the 2025 window outside the gated
> harness (required for new-vs-old comparability; no selection was made from it); from this point, no
> evaluation of anything on 2025 until the single final TEST read.
>
> Then, one commit per item, stopping cleanly with a session record if context runs low — never start an
> item you cannot finish:
>
> 1. Item 2 (ground rule 2): experiments/log.csv + one JSON per run with exactly the Phase-2 brief
>    columns (timestamp, git SHA, data SHA, target, feature-set hash, params, seed, fold scheme, DEV MAE,
>    MASE, skill vs unified ruler, sentinel ratio, per-tercile coverage). Re-run the two 15fb6ee DEV
>    configurations under the logger, marked as reproductions. From here on, never report an unlogged
>    number.
> 2. Item 3 (workstream 1): L1 objectives for B_ML (LightGBM objective='l1', XGBoost reg:absoluteerror,
>    HistGBDT absolute_error). TRAIN-internal rolling-origin folds, one DEV confirmation per target
>    (Revenues, Expenditure, State budget balance where the family supports it). Per-model deltas vs the
>    squared-error incumbents against the unified ruler, all logged; summarize in
>    reports/ws1_objectives.md, including whether the sentinel ratio moved — on the flow targets that is
>    the number we are trying to lift, not just MAE.
> 3. Only if context comfortably allows: begin workstream 2 (LightGBM quantile port, crossing-safe,
>    Optuna ~100 trials with h-gapped early stopping) for Revenues and Expenditure; State budget balance
>    joins via the delta path.
>
> TEST stays sealed — strictly now: no 2025 evaluations of any model until the final read. End with
> reports/phase7_session_record.md in the established format.

---

## 2 · Premise verification

| Premise | Measured |
|---|---|
| `model/excellence` @ `e9c775b` | ✅ |
| suite = 273 · TEST gated reads = 0 · data SHA `0b009fd0…` | ✅ |
| `origin/main` @ `4a925ac` | ❌ **`6c22009`** — PR #24 was **merged** 2026-08-05T02:39:11Z |

Reported and continued per the instruction's own branch. Rebased `model/excellence` onto
`origin/main`; the rebase collapsed it to `6c22009` with **0 commits ahead** (everything had
landed via the merge). Suite green at 273.

---

## 3 · Step 0 — and the second disclosure

The instructed disclosure was written in two places (`40d391b`): a `RETROSPECTIVE_DISCLOSURE`
line in `experiments/test_access.log` (gitignored) and a durable **§0a "TEST-window accounting
— read this before quoting any 2025 number"** in `reports/HANDOFF.md`.

**Then a second, unforeseen disclosure became necessary.** Populating the logger tripped a
guard I had written into the reproduction script:

```
AssertionError: expected DEV only, got {'test', 'dev'}
```

Item 1f (`87fc971`) made E_QUANTILE's pinned folds tile **forward** from `eval_start`. Correct,
but it left the **upper edge unbounded** — tiling ran to the end of the series. Harmless for the
2025 benchmark, whose window genuinely ends there; wrong for anything else. Pinning to
`DEV_START` evaluated **2024-01-01 … 2025-08-06**: 418 target dates where DEV has 262.

So the phase-3 "DEV" figures were **DEV plus the whole available holdout**:

| Figure, as reported | Reported | Actually measured | Honest DEV-only |
|---|---:|---|---:|
| GBQuantile Revenues, DEV skill | 47.98% | 2024-01-01…2025-08-06, n=410 | **46.37%** |
| GBQuantile State budget balance, DEV skill | 32.20% | same | **33.81%** |

The correction is small and that is not the point — they were not DEV numbers. **No selection
was made from them**; they were a fold-scheme sanity check. Recorded in HANDOFF §0a and the
ledger. Fixed by `Config.eval_end` with four regression tests, including one asserting that a
*contradictory* window evaluates nothing rather than falling back to the series end, since
silently ignoring an impossible cap is how an unbounded run returns.

`PR #24` was already merged, so this fix ships on the next PR, not as an amendment to it.

---

## 4 · Item 2 — the experiments log (`5f2990c`)

`backend/experiment_log.py`. `experiments/log.csv` carries exactly the brief's columns plus
`run_id` and `schema_version`; one JSON per run in `experiments/runs/` holds full SHAs, feature
names, full params and the ruler value.

Two design points: rows are **append-only** — a wrong run is superseded by a later row, never
edited away, so the record shows what was believed at the time. The **feature-set hash is over
NAMES, not values**, so the same recipe on different data hashes identically and `data_sha` is
what separates those runs.

`experiments/runs/` is now **tracked**. It was gitignored, which would have made a fresh clone
fail its own `verify_log_integrity()` check and left "reproducible from a logged run" with
nothing behind it. `test_access.log` stays untracked on purpose: a ledger of holdout reads must
not be rewritable by a checkout, which is why HANDOFF §0a is the durable record.

**The two reproductions, now DEV-only:**

```
target                  n      DEV MAE      DEV ruler  skill    MASE  cov l/m/h
Revenues              262   47,368,478   88,317,355   46.37%  0.922  82/92/49
State budget balance  262  160,614,642  242,653,025   33.81%  1.306  73/70/54
```

Top-tercile coverage is **49%** and **54%** against a nominal 80% — on DEV as on 2025. The
miscalibration is therefore not a property of one window. Workstream 7's CQR is the fix.

---

## 5 · Item 3 — workstream 1, L1 objectives (`db27cd4`)

Full detail in [reports/ws1_objectives.md](reports/ws1_objectives.md). Headline: **L1 beats its
own squared-error twin in 17 of 18 comparisons.**

```
window/target                    HistGBDT   XGBoost  LightGBM
TRAIN/Revenues                     +6.48%    +8.80%   +12.82%
TRAIN/Expenditure                  +0.14%    +6.18%    +5.88%
TRAIN/State budget balance         +8.52%   +16.28%    +6.84%
DEV/Revenues                       +0.09%    +5.13%   +15.75%
DEV/Expenditure                    -4.44%    +1.00%    +8.70%
DEV/State budget balance          +20.81%   +18.79%   +20.40%
```

Hyperparameters are identical to the twins, enforced by a test that diffs `get_params()`.
The one loss is a sign flip at the noise floor of a single-fold confirmation, on the target
where L1's TRAIN gain was smallest.

Best per target on DEV: **`LightGBM_L1`** on Revenues (36.65% skill) and Expenditure (28.64%);
**`HistGBDT_L1`** on State budget balance (16.10%). On the stock target L1 turns three
worse-than-persistence models into three that beat it.

**A prerequisite defect:** `build_yearly_folds` folded to the last year in the data, so a
default B_ML run's final fold **was the 2025 holdout**, and nothing could scope the family to
TRAIN or DEV. `eval_start`/`eval_end` added with six regression tests, one of which
deliberately characterises the hazard so the bounds cannot be quietly removed.

### The finding that matters most

**The sentinel ratio did not move, and no objective change can move it.**

| Target | TRAIN | DEV | Signal? |
|---|---:|---:|---|
| Revenues | 1.07 | 1.14 | **No** (< 1.50) |
| Expenditure | 1.13 | 1.09 | **No** |
| State budget balance | 2.28 | 5.57 | Yes |

`signal_sentinel` fits a fixed **Ridge** probe on the feature set, so it measures whether the
*features* carry signal. It is identical across all six models within a window by construction.
The brief named this as the number to lift on the flow targets; workstream 1 cannot lift it and
neither can workstreams 2 or 6. It needs **new information** — workstream 3 (fiscal calendar)
and workstream 5 (multivariate).

Stated bluntly because the two numbers disagree: `LightGBM_L1` shows 36.65% DEV skill on
Revenues against a sentinel ratio of 1.14. The errors really are smaller than h-step
persistence, but with no detectable feature signal the model is regressing toward a central
level against a spiky baseline. **A 36% improvement over a bad ruler is still not a forecast.**

### A second logger defect, caught by its own integrity check

`run_id` stamped timestamps to whole seconds, so 36 rows written inside one second collided —
sharing one detail JSON, making all but one run unrecoverable. `verify_log_integrity()` reported
`duplicate run_id values`; the batch was discarded and re-logged with microsecond precision plus
an on-disk collision loop. Seven tests now cover the log.

---

## 6 · Item 4 — workstream 2: NOT STARTED

The instruction conditioned it on context comfortably allowing, and it does not. Optuna at ~100
trials with h-gapped early stopping across two targets is a long run whose results must be
logged and interpreted; starting it here would have meant abandoning it mid-flight, against the
standing rule *never start an item you cannot finish*.

Nothing was left half-applied: both completed items are committed, the suite is green, and the
log's integrity check passes.

---

## 7 · What remains

| # | Task | Notes |
|---|---|---|
| **4** | **Workstream 2** — LightGBM quantile port, crossing-safe, Optuna ~100 trials with h-gapped early stopping; Revenues + Expenditure, stock via the delta path | **Resume point.** Tune on top of **L1** — pinball at τ=0.5 *is* absolute error, so the objectives are already consistent |
| 5 | Workstream 3 — fiscal calendar. `docs/FISCAL_CALENDAR_SOURCES.md` must be drafted before T2 can be sent | **Now the highest-value workstream**, because it is one of only two that can move the sentinel ratio on the flows |
| 6 | Workstreams 4–7 — target scaling, multivariate, ensemble, per-target selection + CQR | 7 owns the top-tercile coverage fix |
| 7 | Phase-1 Steps 4/5/7 — ops P0, Agent contract + validator, cleanup (untrack `.venv`/`.env` **without printing contents**) | Deferred since Phase 1 |

### Open, carried forward

- **Treasury T1 (negative `Revenues`) — answer OUTSTANDING.** All figures stand under Q1's
  interim policy (legitimate signed flows). If overturned, `log1p` becomes available to
  workstream 4 and the flow figures need revisiting.
- **Two flow targets have no detectable feature signal** (1.07–1.14). This is the central
  modelling problem and no amount of objective or hyperparameter work addresses it.
- **Top-tercile interval coverage 49–54%** against a nominal 80%, now confirmed on DEV as well
  as 2025. Workstream 7's CQR.
- **C_DL stock-target collapse** — parked (Q6); a scaling bug, not a capability limit.
- **PR #24 is MERGED.** `model/excellence` is 3 commits ahead of `origin/main` and needs a new
  PR.
- `reports/HANDOFF.md` needs this session's three commits before the next handoff.

---

## 8 · Reproduction

```bash
git checkout model/excellence            # db27cd4
./backend/.venv/bin/python -m pytest -q  # expect 294 passed
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from experiment_log import verify_log_integrity as v;print(v())"
# expect {'n_rows': 38, 'ok': True, 'problems': []}
```

Every number in §4 and §5 is a row in `experiments/log.csv`, with full provenance in
`experiments/runs/<run_id>.json`. The rule "never report an unlogged number" is in force from
this session onward; the figures in phase-6 §4 predate it and remain recorded only in
`87fc971`'s commit message.
