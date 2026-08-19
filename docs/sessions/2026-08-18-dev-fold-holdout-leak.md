# Session — fix the ws2_tune DEV-fold holdout leak

**Date:** 2026-08-18 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`).
**Closes** the top-priority open item from `2026-08-18-artifact-regeneration.md`.

> ## The sharing gate is lifted
>
> That record said the client-facing table must not go to a client until this was fixed. It is
> fixed, **no verdict or champion moved, and no number in the client-facing table changed.** The
> gate is lifted on the evidence below, not on assertion.
>
> **What is still open, so it is not discovered late:**
>
> * **The training embargo is the new top item** (open item 1) — 5 training rows still carry targets
>   inside the evaluation block. Not a holdout breach; its effect is model variance rather than bias
>   (Revenues −2.98%, Expenditure **+0.65%**), so it needs its own scoped session and must not be
>   waved through as an obvious correctness win. Its inverted pin holds.
> * **The sentinel could not be recomputed** (open item 2) — so this fix's effect on the `signal`
>   gate is **unmeasured, not zero**.
> * **Expenditure's MASE is 0.9961 reconstructed against 1.1039 logged** (open item 3) — an 8.6%
>   harness disagreement, not this fix. Which side of the threshold it falls on depends on which
>   harness is asked.
>
> None of the three affects the client-facing table, which is measured on the holdout fold rather
> than the DEV fold these concern.

---

## The finding, restated

A fold says "evaluate 2024". It selects rows by **origin** — the day the forecast is made — and
scores against truth at **origin + 5 business days**. So an origin on 2024-12-24 was scored against
2025-01-02: inside the sealed holdout.

`assert_selection_free` was called on the **origins**. All 250 sat in DEV, so it passed while 4 of
the rows it had just approved were scored against holdout truth.

**A second spill nobody had looked at.** The same arithmetic applies one boundary earlier:

| Window | fold | n | origins | target dates | spill |
|---|---|---:|---|---|---|
| TRAIN | 1–4 | 249–254 | train | train | — |
| **TRAIN** | **5** | 250 | train | 245 train + **5 dev** | **5 DEV rows** |
| **DEV** | 1 | 250 | dev | 246 dev + **4 test** | **4 TEST rows** |

Identical on all three targets. The DEV→TEST spill is the holdout violation. The TRAIN→DEV spill
meant the Optuna search read 5 rows of its own confirmation set — not a holdout breach, but it
erodes what DEV is for.

---

## The fix, and why not the design the record proposed

The record proposed "bound by target date". Tested literally, **it introduces a worse defect**:

| Design | DEV n | Result |
|---|---:|---|
| CURRENT — origin-bounded | 250 | **4 holdout target dates** |
| A — target-bounded only | 251 | **5 rows evaluated at origins the model trained on** |
| B — origin-bounded, drop last *h* | 245 | clean, but drops one more row than spills |
| **C — intersection (chosen)** | **246** | **clean, and exact** |

Pure target-bounding pulls late-December 2023 origins into the DEV fold, and those origins are
`≤ train_end` — the model trained on them. Trading a 4-row holdout read for a 5-row train/eval
overlap is not a fix.

**C:** an evaluation row survives only if its **origin is in the fold's block** *and* its **target
date is in a window this search may read**. Stated as an invariant rather than an arithmetic trim,
which is why it drops exactly the 4 rows that spill where B would drop 5.

`ALLOWED_TARGET_WINDOWS = {"train": {"train"}, "dev": {"dev"}}` — a TRAIN search may not read DEV
truth, because a search that has already seen part of its confirmation set is not confirmed by it.

**The guard moved to where the violation lives.** `assert_selection_free` is now called on the
target dates as well as the origins.

**Cost, measured and identical on all three targets:**

```
TRAIN   1259 -> 1254   (5 rows, fold 5 only)
DEV      250 ->  246   (4 rows, every one with a test-window target)
```

The same invariant was applied to `sealed_window_report.sealed_folds`, so a DEV-scoped
reconstruction is measured on the same row set the corrected credentials use, and a sealed-window
call cannot reach past `TEST_END` into LIVE.

---

## Re-derived credentials, and the verdict answer

The fix removes **evaluation** rows only. Training is `origins ≤ train_end`, untouched, so the
fitted model is identical — the same model scored on 246 rows instead of 250. That makes the change
**exactly computable with no refit**, and the 4 dropped rows are confirmed to be exactly the ones
with `test`-window targets.

| Target | DEV MAE 250 → 246 | delta | MASE 250 → 246 |
|---|---|---:|---|
| Revenues | 38,044,471 → 37,761,040 | **−0.745%** | 0.7407 → 0.7352 |
| Expenditure | 47,166,000 → 46,566,923 | **−1.270%** | 1.0089 → 0.9961 |
| State budget balance | 158,815,900 → 157,490,000 | **−0.835%** | 1.2914 → 1.2806 |

### Does any verdict or champion move? No.

Applying the measured delta to the logged credentials, with the sentinel held fixed:

| Target | verdict before | verdict after | MASE |
|---|---|---|---|
| Revenues | publishable | **publishable** | 0.7580 → 0.7523 |
| Expenditure | withheld | **withheld** | 1.1039 → 1.0898 |
| State budget balance | withheld | **withheld** | 1.5783 → 1.5651 |

`scripts/rerun_publication_gates.py` confirms: **no verdict changes.** No champion moves either —
champions are fixed by policy and nothing in this change reselects.

### Two honest limits on that answer

**The sentinel could not be recomputed.** Reconstructing it gives degenerate values — exactly 1.0000
on Revenues against a logged 1.2255 — which is the known unreproducibility (the WS4 harness is
absent). So the fix's effect on the `signal` gate is **unmeasured**, not zero. It is held fixed above
rather than guessed. Since `signal` sits *below* `accuracy_vs_naive` in severity and no MASE crosses
its threshold, the verdicts stand on the gate that actually decided them.

**Expenditure sits on the threshold under recomputation.** Its reconstructed MASE is 0.9961 — under
1.0 — where the logged credential is 1.1039. The two disagree by 8.6% because the credential's
harness is gone, not because of this fix: the fix's own delta is −1.27%, which takes 1.1039 to
1.0898 and leaves it failing. **The registry was not touched.** Whether Expenditure should publish is
a question about which harness owns the credentials, and it belongs to the reproducibility item.

---

## The client-facing table did not move

Recomputed after the change, every figure identical:

```
Revenues              n 146->146  MAE 37,228,525->37,228,525  naive 55.96->55.96  ops 31.99->31.99
Expenditure           n 146->146  MAE 54,513,302->54,513,302  naive 29.19->29.19  ops  2.79-> 2.79
State budget balance  n 146->146  MAE 121,820,978->121,820,978 naive 33.70->33.70  ops   n/a
```

Unchanged because sealed-window evaluation targets were **already** all inside TEST, so the new
target-window filter drops nothing there. The leak was at the DEV boundary, and the client-facing
numbers are measured on the holdout — a different fold. So the table in
`2026-08-18-artifact-regeneration.md` stands as written, now without a gate on it.

---

## Tests

**Deleted**, per its own assertion message: `test_the_tuners_dev_fold_is_scored_against_holdout_rows`
— it documented the leak's existence and failed once fixed, exactly as designed. A comment stands
where it was, pointing at the replacement.

**Rewritten:** `test_a_dev_scoped_call_logs_the_year_boundary_rows_it_really_reads` →
`test_a_dev_scoped_call_no_longer_reads_holdout_targets`. Its history is the point: it first asserted
"no holdout read" and **failing was the finding**; it was then rewritten to assert the leak; it now
asserts the fixed state.

**New — `backend/tests/test_no_fold_reads_holdout_truth.py`, 18 tests.** The permanent replacement
asserts the *property* per fold builder rather than documenting one instance of its absence:

* every ws2_tune fold, all three targets, both windows, reads truth only from selectable windows;
* a TRAIN search reads no DEV truth (its own confirmation set);
* a DEV fold reads no holdout truth;
* the reporting harness reads only the window it was asked for, on both scopes;
* no fold anywhere reads LIVE truth;
* the guard is applied to target dates, not only origins;
* the row cost is pinned at DEV 246 / TRAIN 1254, so a future change cannot quietly widen it.

**Kept:** `test_the_borrowed_selection_path_still_lacks_the_embargo`. See open item 1 — that gap is
real and separate, and its pin still holds.

---

## Open items

### 1. NEW TOP PRIORITY — the training embargo, for its own scoped session

**Reviewed and accepted as the next item. Its pin holds:**
`backend/tests/test_sealed_window_report.py::test_the_borrowed_selection_path_still_lacks_the_embargo`
passes while the gap exists and fails once it is fixed, with an assertion message telling the fixer
to delete it. Same inverted-pin discipline that carried the DEV-boundary leak to resolution.

**What it is.** Training rows are `origins ≤ train_end`, and 5 of them carry targets **inside the
evaluation block**. For a DEV fold those targets are DEV, so it is **not** a holdout breach — but the
model is fitted on answers from the very block it is then scored on.

**Measured cost of removing them, and it is not a systematic inflation:**

| Target | DEV MAE with the 5 rows | without | delta |
|---|---|---|---|
| Revenues | 37,761,043 | 36,636,788 | **−2.98%** |
| Expenditure | 46,566,923 | 46,870,622 | **+0.65%** |
| State budget balance | 157,489,983 | 155,557,963 | −1.23% |

Revenues improves, Expenditure gets **worse**. With 5 rows out of ~2,400 that is **model variance
from a 0.2% change in training data, not a bias being removed** — which is exactly why it should not
be waved through as an obvious correctness win.

**Why it was not done here**, recorded so the scoping decision is not re-litigated:

* it is outside "the DEV-fold holdout leak" — it is a different property;
* it changes **training** rather than **scoring**, so unlike this session's fix it cannot be computed
  exactly without refitting, and the model itself changes;
* it moves credentials by up to **3%** — three times this session's effect — so it interacts with
  items 2 and 3 below rather than being independent of them.

**Head start for that session:** `backend/sealed_window_report.sealed_folds` already implements this
embargo (a training origin is kept only if its target predates the first evaluation origin, asserted
rather than trusted), so the pattern exists to copy rather than design.

### 2. Known limit — the sentinel cannot be recomputed

The `signal` gate's input could not be reconstructed. The attempt returns **exactly 1.0000** against
a logged **1.2255** on Revenues, which is the known missing-harness problem (the WS4 script that
produced the credentials is absent from the repository).

So **the fix's effect on the `signal` gate is unmeasured, not zero.** It was held fixed at the logged
value when re-deriving verdicts, and the verdicts stand on the gate that actually decided them:
`signal` sits *below* `accuracy_vs_naive` in severity, and no MASE crosses its 1.0 threshold. That is
a real gap in the answer, not a formality — any future work on the `signal` gate must reconstruct the
harness first.

### 3. Known limit — Expenditure's harness disagreement

Its **reconstructed MASE is 0.9961 — under the 1.0 threshold — against a logged 1.1039.**

That is an **8.6% harness disagreement, not this fix**, whose own −1.27% leaves the logged value
failing at 1.0898. The registry was not touched, and Expenditure's verdict did not move.

Resolving it means deciding **which harness owns the credentials** — reconstruct WS4 so the logged
values are reproducible, or formally supersede them with figures from a harness that is. Until then,
Expenditure sits on the accuracy threshold and which side it falls on depends on which harness is
asked. See item 2; the two are the same underlying problem.

---

## Suite

```
before this session   925 passed, 4 skipped
  - 1 deleted pin     924
  + 18 new invariant  942 passed, 4 skipped
```

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_no_fold_reads_holdout_truth.py -q  # 18 passed
./backend/.venv/bin/python -m pytest backend/tests/test_sealed_window_report.py -q         # 13 passed
./backend/.venv/bin/python scripts/rerun_publication_gates.py                              # no verdict changes
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q                       # 942 passed, 4 skipped
```

Per-target credential deltas: `reports/dev_fold_fix_credentials.csv`.
