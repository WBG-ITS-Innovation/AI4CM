# Phase 2 — Yardstick Completion Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `94db732` (14 ahead of `origin/main`)
**Test suite:** 246 → **272** passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**TEST (2025) reads: 0** — `experiments/test_access.log` empty

---

## 1 · The instruction

> Resume from reports/phase4_session_record.md §8. Verify premises first (model/excellence @ 904a520 or
> its session-record child, suite = 246, TEST reads = 0, data SHA 0b009fd0…, PR #23 state on origin); if
> reality disagrees, say so and stop.
>
> Step 0: update reports/HANDOFF.md with the phase-4 commits and PR #23, per that record's own note.
> Then sync: if PR #23 has been merged into origin/main, rebase model/excellence onto origin/main and
> re-run the suite before proceeding; if not merged, note it and continue.
>
> Then, one commit per item, stopping cleanly with a session record if context runs low — never start an
> item you cannot finish:
>
> 1. Item 1b: add an eval_start concept to c_dl_pipeline and pin it to TEST_START from
>    evaluation_windows. C_DL stays parked per Q6 — the pin exists so the one-ruler check can include it.
> 2. Item 1e: input selection by explicit name + recorded SHA-256 (never mtime) in run_daily_forecast.sh,
>    with the file name, sha256, latest_data_date and n_rows recorded in provenance by all four runners.
> 3. Item 1f: re-run the §4 backtest on the canonical data. Verification: all four families report
>    literally one persistence number (83,534,152.85 on the 2025 window definition — confirm against the
>    shared function); new-vs-old tier table in the commit message. Include Expenditure — first honest
>    numbers for a priority target — and report per target: skill vs the unified ruler, sentinel ratio,
>    and per-magnitude-tercile coverage. Also add docs/DATA_SEMANTICS.md recording the
>    negative-Revenues characterization (72 business days, driven by Increase in liabilities/Domestic
>    netting; identical in raw) and the two questions sent to Treasury, so the audit trail carries it.
> 4. Item 2 (ground rule 2): experiments/log.csv + one JSON per run with exactly the Phase-2 brief
>    columns (timestamp, git SHA, data SHA, target, feature-set hash, params, seed, fold scheme, DEV MAE,
>    MASE, skill vs unified ruler, sentinel ratio, per-tercile coverage). Re-run the two 15fb6ee DEV
>    configurations under the logger, marked as reproductions. From here on, never report an unlogged
>    number.
> 5. Item 3 (workstream 1): L1 objectives for B_ML (LightGBM objective='l1', XGBoost
>    reg:absoluteerror, HistGBDT absolute_error). TRAIN-internal rolling-origin folds, one DEV
>    confirmation. Per-model deltas vs the squared-error incumbents against the unified ruler, all
>    logged; summarize in reports/ws1_objectives.md.
> 6. Only if context comfortably allows: begin workstream 2 (LightGBM quantile port, crossing-safe,
>    Optuna ~100 trials with h-gapped early stopping) for Revenues and Expenditure; State budget balance
>    joins via the delta path.
>
> TEST stays sealed. End with reports/phase5_session_record.md in the established format.

---

## 2 · Premise verification

All stated premises held. Two things to record.

| Premise | Measured | Verdict |
|---|---|---|
| `model/excellence` @ `904a520` or its child | `98a1da4` (the phase-4 record) | ✅ as anticipated |
| suite = 246 | 246 passed | ✅ |
| TEST reads = 0 | file absent | ✅ |
| data SHA `0b009fd0…` | matches | ✅ |
| PR #23 state | **MERGED** 2026-08-05T00:48:05Z; `origin/main` now `4a925ac` | ✅ |

### The negative-Revenues claim needed checking, and one part was wrong

The instruction asked me to record "72 business days, driven by Increase in
liabilities/Domestic netting; **identical in raw**". I had never measured this, so I did:

| Claim | Measured | Verdict |
|---|---|---|
| 72 business days | **72** of 2,763 | ✅ exact |
| Driven by `Increase in liabilities` / `Domestic` netting | negative on 64/72 and 65/72; correlation with `Revenues` **0.971** and **0.969** | ✅ strongly supported |
| Identical in raw | raw has **73**; the extra is `2021-01-19` | ❌ **wrong by one day** |

`2021-01-19` is a Georgian public holiday carrying a reported −783,716.11 that
`business_days_zero_flows` correctly zeroed. On genuine business days the two sets *are*
identical, so the substance holds — but the document records 73-vs-72 rather than
repeating "identical". Also noted for its own sake: the source reports a non-zero value
on a public holiday.

---

## 3 · Step 0 — HANDOFF update and sync

PR #23 **merged**, so `model/excellence` was rebased onto `origin/main` (`4a925ac`), 11
commits replayed cleanly, suite green at 246, data SHA unchanged.

`reports/HANDOFF.md` updated (`7c89e5d` pre-rebase): phase-4 commits added, PR #23
recorded, resume point moved from 1c to 1b, and the four now-fixed issues struck through
in the §6 register.

---

## 4 · Item 1b — C_DL pinned to TEST_START (`b7bccd7`)

C_DL folded over every available year, so its published skill was a 2019–2025 average
while the other three families reported on the 2025 holdout:

```
C_DL reported  +10.84%   (2019-2025 average, n=1,722)
C_DL actually   -5.19%   (shared 2025 window,  n=156)
```

The gap is the ruler, not the model: persistence over 2019–2025 is 52,957,744, far easier
than over 2025 alone.

`build_yearly_folds()` now takes `eval_start`. Folds ending before it are dropped; a
block **straddling** it is trimmed rather than discarded, so no in-window rows are
wasted. `ConfigDL.eval_start` defaults to `None` — a direct caller is not silently
re-scoped — while both shipped runners default to `evaluation_windows.TEST_START`,
imported rather than hardcoded.

Effect on the real label index: **7 folds → 1**, test block `2025-01-01 … 2025-08-06`.

C_DL remains parked (Q6). The pin exists so item 1f's one-ruler check can *include* it.

**Mutations:** `cutoff` forced to `None` → 4 failed; runner stops passing `eval_start` →
1 failed (the wiring test). A first attempt — disabling the `test_end < cutoff` guard —
was **not** caught, correctly: it is behaviourally neutral, since such folds fall through
to the trimming branch where the slice is empty and they are dropped anyway. That guard
is a fast path, not load-bearing.

---

## 5 · Item 1e — input by name + SHA-256, provenance everywhere (`db06426`)

Before: provenance for **one of four** families, no git SHA, no input hash, and
`ls -t | head -1` selection. `touch` on any file in `data/processed` silently changed
what the pipeline forecast.

**Input selection.** Explicit `DATA_FILE_NAME` (default
`master_daily_clean_treasury.csv`), SHA-256 computed and logged up front so it survives a
later family failure. `AI6CM`-style pinning via `AI4CM_EXPECTED_DATA_SHA256` makes every
runner refuse to start on a mismatch — selecting by *name* is necessary but not
sufficient, because a file can be regenerated in place.

**Provenance.** New `backend/provenance.py`, one implementation for all four families,
recording `data_file{name, path, sha256, n_rows, latest_data_date, size_bytes}`,
`code{git_sha, git_branch, git_dirty, git_dirty_files}`, `environment{python, platform,
package versions}`, `config`, `seed`, `stale_override`, and the `TG_*` env vars.

Deliberate details: `git_dirty` is recorded because a SHA alone does not identify what
ran on a modified tree; package versions include `xgboost`/`lightgbm`, whose availability
silently changes B_ML's model set; `stale_override` captures Q5's explicit override so it
can never be untraceable; `record_run()` never raises, because provenance is a record and
not a gate. b_ml's hand-rolled dict was **replaced**, not left alongside.

Real run:

```
[provenance] B_ML: data=master_daily_clean_treasury.csv sha256=0b009fd031ad3fa0...
             rows=3867 latest=2025-08-06 git=b7bccd76+dirty -> .../provenance.json
```

Pinning fails closed: `AI4CM_EXPECTED_DATA_SHA256=000…` → *"Refusing to run: the input is
not the file this run was pinned to."*

**Mutations:** restore `ls -t | head -1` → 1 failed; a runner stops recording provenance
→ 1 failed.

### Two of my own assertions were wrong, not the code

- The mtime check first matched the explanatory **comment** documenting the old
  behaviour — flagging the fix as the bug. It now inspects executable lines only.
- The pre-existing unbound-variable guard from PR #18 correctly flagged
  `AI4CM_EXPECTED_DATA_SHA256` as read-but-never-assigned. It is a documented env input,
  so it went into that test's `EXTERNAL` allowlist exactly as the test instructs. That
  guard doing its job is the system working.

**This is now the fourth and fifth time a fixture, not the code, was at fault.** On this
codebase, treat a failing *new* assertion as "check the fixture first".

---

## 6 · Item 1f — partially done (`94db732`)

`docs/DATA_SEMANTICS.md` landed, since it is self-contained and the measurements existed:
the negative-Revenues characterization with the one-day correction, both Treasury
questions with **honest status**, blank-cell semantics, weekend/holiday handling, the
1,095 removed fabricated values, and reproduction commands for every figure.

On the Treasury questions: the brief referred to "the two questions sent". Only **T1**
(negatives) has gone. **T2** (fiscal calendar) cannot sensibly be sent until
`docs/FISCAL_CALENDAR_SOURCES.md` exists, so Treasury has a concrete list to confirm
rather than an open question. Recorded as *not yet sent*.

**The backtest itself was not run.** It needs 3 targets × 4 families, the one-ruler
verification, a new-vs-old tier table, and per-target sentinel ratio and per-tercile
coverage — more than remained. Not started rather than half-started.

---

## 7 · What remains

| # | Task | Notes |
|---|---|---|
| **1f** | Backtest re-run; confirm all four families report **one** persistence number (expect 83,534,152.85 on the 2025 definition, via the shared function); new-vs-old tier table; **Expenditure's first honest numbers**; per-target skill / sentinel ratio / per-tercile coverage | Item 1's verification step. All five prerequisites (1a–1e) are now done |
| 2 | Ground rule 2 — `experiments/log.csv` + per-run JSON; re-run the two `15fb6ee` DEV configs as marked reproductions | "Never report an unlogged number" takes effect after this |
| 3 | Workstream 1 — L1 objectives, `reports/ws1_objectives.md` | |
| 4 | Workstream 2 — LightGBM quantile + Optuna | |

**Item 1 is now 5/6.** Everything the one-ruler check depends on is in place: E_QUANTILE
on a business-day index with a stock path, C_DL pinned, the duplicate integrity module
retired so the published baseline comes from the shared function, `mae_seasonal_naive` no
longer impersonating persistence, and the input identified by hash.

### Still open from earlier sessions

- **Expenditure has no honest numbers yet.** Priority target; 1f is where it first gets one.
- E_QUANTILE stock-target coverage 70.2%, 0.2pp inside the gate band — the bootstrap-CI
  gate (review §3 P3.2) in workstream 7 is what would judge it properly.
- `reports/HANDOFF.md` needs this session's four commits added before the next handoff.

---

## 8 · Reproduction

```bash
git checkout model/excellence            # 94db732
./backend/.venv/bin/python -m pytest -q  # expect 272 passed
cat experiments/test_access.log 2>/dev/null || echo "TEST reads: 0"

./backend/.venv/bin/python -m pytest \
  backend/tests/test_c_dl_eval_window_pin.py \
  backend/tests/test_provenance.py -v     # 8 + 18

# provenance end to end
cd backend && TG_MODEL_FILTER=Ridge TG_TARGET=Revenues TG_CADENCE=Daily TG_HORIZON=5 \
  TG_DATE_COL=date TG_DATA_PATH=data/processed/master_daily_clean_treasury.csv \
  TG_OUT_ROOT=/tmp/prov TG_PARAM_OVERRIDES='{"folds":1,"min_train_years":4}' \
  ../backend/.venv/bin/python run_b_ml_univariate.py | grep provenance
```

`docs/DATA_SEMANTICS.md` §4 carries the commands for every figure in it. No number in
this record is in `experiments/log.csv` yet — ground rule 2 is item 2, and the
"never report an unlogged number" rule takes effect from workstream 1 onward.
