# Session — close the B_ML sealed-window ledger gap

**Date:** 2026-08-17 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`); the agent
repo was not touched.
**Scope:** deliberately small. No model refitted, no published artifact touched, no other open
item picked up.
**Follows:** `docs/sessions/2026-08-17-interval-calibration.md` §7 item 5.

---

## What was wrong

B_ML was the last of the four families whose holdout read did not reach the ledger. A_STAT and
C_DL were wired up in P1; E_QUANTILE earlier the same day. This was not a theoretical gap:

| Evidence | Measured |
|---|---|
| `experiments/test_access.log`, 148 entries at session start | A_STAT 16 · C_DL 121 · E_QUANTILE 9 · **B_ML 0** |
| Both logged B_ML runs (`2026-08-04`, `2026-08-12`) | evaluated `target_date` **2025-01-01 → 2025-08-06** |
| Daily runner, `scripts/run_daily_forecast.sh:133` | `{"folds":1,"min_train_years":4}` — **no `eval_start`** |

`folds_override` keeps the **last** fold, and on this index that fold is the holdout and nothing
else:

```
folds_override = None -> 7 folds; last is train<=2024-12-31, test 2025-01-01..2025-08-06
folds_override = 1    -> exactly that fold, windows = ['test']
```

So every B_ML daily run evaluated the sealed window end to end while the ledger recorded nothing.
"How many times was the holdout consulted?" had no factual answer for this family.

---

## Every sealed-window touch point, classified

Task 1 was to enumerate and classify before writing anything. Ten sites; the classification was
approved before any code changed.

### Report — sealed-window reads

| # | Site | Why report |
|---|---|---|
| 1 | `b_ml_pipeline.py:732` `build_yearly_folds` | Builds the fold whose test block *is* the holdout. Fold construction chooses nothing. **The ledger call goes here.** |
| 2 | `:775-975` fold loop | Predicts at test origins, reads `y_true` from test actuals. Downstream of #1; one entry per run covers it. |
| 3 | `:1029-1034` `predictions_long` / `metrics_long` | Writes metrics computed over test rows. |
| 4 | `:1038-1059` leaderboard | Per-model MAE over test rows plus the shared persistence ruler. Artifact = report; becomes selection input at #6. |
| 5 | `:1391-1414` quality gate | Applies `_QUALITY_GATE_SKILL_PCT = 5.0` to a holdout metric and sets `run_status`. **Classified report**, and the classification was put to the client explicitly rather than assumed: the threshold is a constant in source, so this *applies* a decision already made. Choosing a threshold from the data would be selection; reading a pass/fail off a fixed one is not. |

### Selection — already guarded, unchanged

| # | Site | Status |
|---|---|---|
| 6 | `:1078` `select_best_model` | Ranks models by MAE **computed on test rows** — genuinely selection. Guarded by `assert_selection_free` at `:1073-1077`, which raises. |

### Not sealed-window reads — each verified to read only up to `train_end`

| # | Site | Data it sees |
|---|---|---|
| 7 | `:796` `multivariate_exog` (top-K by correlation) | `s_train_full` = rows ≤ `train_end` |
| 8 | `:868-876` conformal PI radius | last 20% of the training block; targets built by `shift` *inside* that block, so the final *h* rows go NaN and are dropped — targets stay ≤ `train_end` |
| 9 | `:891-896` `overfit_ratios` | train/val split inside the training block |
| 10 | `:1220-1278` `signal_sentinel` | `s.index <= train_end` |

Worth recording about **#8**: this is precisely the embargo E_QUANTILE was *missing* until earlier
today. B_ML gets it right, and for a structural reason — it builds targets by shifting within the
training slice rather than indexing across the full series, so the rows whose targets would fall
past `train_end` become NaN and drop out on their own.

---

## No leakage finding — with one fragility recorded

The brief said to stop and report if any selection-context read of the sealed window turned up.
One exists (#6, ranking by holdout MAE), but it is already guarded and the guard genuinely fires.

I checked that rather than assuming it, because the guard sits inside a `try:` spanning lines
1062–1495:

```
Try block 1062..1495 ENCLOSES the guard at 1077
   handler 1477: except RuntimeError  ->  raise
   handler 1480: except Exception     ->  print("[WARN] ..."); traceback.print_exc()
```

`SelectionOnReportOnlyDataError` → `RuntimeError` → `Exception`, and Python matches handlers **in
declaration order**, so the `raise` at 1477 wins. The guard is effective.

**The fragility, recorded here by decision rather than fixed:** its effectiveness depends on the
ordering of two handlers roughly 400 lines from the guard itself. Reverse them — or give the
exception a non-`RuntimeError` base — and the guard degrades into a printed warning while a
champion is crowned on holdout data. Nothing in the code says those two handlers are
load-bearing. Deliberately out of scope for this session; flagged as an open item.

---

## The change

**`backend/b_ml_pipeline.py`** — one insertion, immediately after `build_yearly_folds`:

```python
from evaluation_windows import PURPOSE_REPORT, require_test_access, window_for
_holdout = [t for (_tr_end, _ts, _te) in folds
            for t in s.index[(s.index >= _ts) & (s.index <= _te)]
            if window_for(t) == "test"]
if _holdout:
    require_test_access(
        f"B_ML reporting evaluation for {cfg.target!r} at h={cfg.horizon} covers "
        f"{len(_holdout)} holdout target date(s) from {min(_holdout).date()} to "
        f"{max(_holdout).date()}",
        caller="b_ml_pipeline.run_pipeline_ml", purpose=PURPOSE_REPORT)
```

Three things about it:

* **`PURPOSE_REPORT` records and returns.** Reporting on the holdout is what the holdout is for, so
  this must not refuse. The selection door is a separate, still-shut door.
* **`test_start`/`test_end` are already TARGET dates** in B_ML (`positions` are target positions
  and the origin is `pos - h`), so unlike the E_QUANTILE fix no origin→target conversion is
  needed. The dates gated are the dates scored.
* **Guarded by `if _holdout`.** A call on every run regardless of window would make the ledger
  count meaningless again — noisily this time. A DEV-pinned run stays silent, and a test asserts
  that.

The entry a real run produces:

```
purpose = report
caller  = b_ml_pipeline.run_pipeline_ml
reason  = B_ML reporting evaluation for 'Revenues' at h=5 covers 156 holdout
          target date(s) from 2025-01-01 to 2025-08-06
```

156 matches A_STAT's and C_DL's format and the scored row count in the artifacts.

**A number I got wrong mid-session and corrected:** I first reported 218 holdout dates. That came
from feeding the raw calendar-day CSV index straight to `build_yearly_folds` in an ad-hoc check,
bypassing the pipeline's business-day reindexing. The pipeline's own figure is 156.

**Expected side effect, stated before the change was made:** a default daily B_ML run now writes a
ledger line and then still raises `SelectionOnReportOnlyDataError` at crowning, exactly as
E_QUANTILE does. That refusal is pre-existing; only the ledger line is new.

---

## Tests

**`backend/tests/test_live_window.py`** — four added, alongside the existing B_ML selection-guard
tests that already live there:

| Test | What it holds |
|---|---|
| `test_the_default_b_ml_fold_geometry_lands_squarely_on_the_holdout` | The *justification* for the call, asserted on the fold builder rather than described in a comment: the daily config yields one fold whose every target date is `"test"`. If that geometry changes, this fails and the rationale gets revisited. |
| `test_b_ml_logs_its_holdout_read_as_a_report` | End to end on the real pipeline: the read is announced, announced as a **report**, names the family, names the caller, and states the date range. Expects `SelectionOnReportOnlyDataError` — the ledger call happens at fold construction, well before crowning, and that ordering is the point: the read is announced when it *starts*. |
| `test_a_dev_pinned_b_ml_run_does_not_claim_a_holdout_read` | No false positives. A DEV-bounded run completes (crowning on DEV is legitimate) and records nothing. |
| `test_a_selection_purpose_read_of_the_holdout_still_raises` | A report records and returns; a selection read raises. The fix opened no new door. |

Both pipeline tests monkeypatch `require_test_access`, so they assert on the call without writing
to the real ledger. Verified: `experiments/test_access.log` gained 13 entries over the suite run,
all C_DL's own pre-existing logging, and **still zero naming B_ML**.

### Suite

```
before:  852 passed, 4 skipped
after:   856 passed, 4 skipped        (+4, the tests above)
```

---

## Open items

1. **The handler-order fragility at `b_ml_pipeline.py:1477-1480`** — recorded above. The selection
   guard works only because `except RuntimeError: raise` precedes `except Exception`. Worth either
   narrowing that `try:` or pinning the order in a test; both are larger changes than this session
   allowed.
2. **The quality gate consumes a holdout metric.** Classified report and I stand by it, but it is
   the one site where the report/selection line is a judgement rather than a fact about the code.
   If the 5.0% threshold ever becomes data-derived, it becomes selection.
3. **All four families now log; none of the four is bounded to DEV by default.** A_STAT, B_ML and
   C_DL take no `eval_start` from the daily runner, so each default run consults the holdout and
   now says so. Whether the daily run *should* be touching the holdout at all is a separate
   question this session did not open.

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_live_window.py -q     # 39 passed
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q         # 856 passed, 4 skipped
```
