# Verification

Exact commands for a third party to re-run every check that currently exists, with expected outputs and
tolerances. Nothing here requires reading the locked TEST window.

**Scope, stated plainly.** Sections 1–4 are runnable today. Sections 5–7 describe checks whose
infrastructure has not been built yet; each says what is missing. See `CHANGELOG.md` for what is and is
not done.

**Environment.** All commands run from the repository root. The backend interpreter is
`./backend/.venv/bin/python`; create it with `scripts/setup_unix.sh` if absent.

---

## 1 · Test suite

```bash
./backend/.venv/bin/python -m pytest -q
```

**Expected:** `225 passed`. Runtime ~30s.

A non-zero exit means a regression. The suite includes the mutation-validated regression tests listed in
`CHANGELOG.md`; if one of those fails, the corresponding fix has been reverted.

---

## 2 · Preprocessing regeneration and data integrity

Regenerates the canonical dataset from the source workbook and confirms it byte-matches what is
committed.

```bash
cd backend
PP_INPUT_PATH=data/Balance_by_Day_2015-2025.xlsx \
PP_VARIANT=clean_treasury \
PP_OUT_ROOT=/tmp/ai4cm_regen \
PP_RUN_OUTPUTS=/tmp/ai4cm_regen/run \
  ../backend/.venv/bin/python run_preprocess.py
cd ..

shasum -a 256 /tmp/ai4cm_regen/clean_treasury/Balance_by_Day_2015-2025__clean_treasury.csv
shasum -a 256 backend/data/processed/master_daily_clean_treasury.csv
```

**Expected:** both hashes are
`0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`.

**Tolerance:** none. The hashes must match exactly. Preprocessing is deterministic — there is no seed
and no parallelism in this path.

**Expected log line** during the run:

```
Business-day coverage OK: 2645/2763 reported, 117 absent as public holidays, 1 allow-listed.
```

If instead it raises `N business day(s) missing from the source and not explained by a Georgian public
holiday or the allow-list`, the source workbook has changed. That is the check working: supply the
missing data or add the dates to `KNOWN_ABSENT_BUSINESS_DAYS` with a reason. Do not silence it.

---

## 3 · The causality property

The single assertion that would have caught the original cleaning bug: perturbing a later row must
leave earlier rows bit-identical.

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/test_causal_cleaning.py -v
```

**Expected:** `14 passed`, including
`test_a_later_value_cannot_change_an_earlier_row`.

**Tolerance:** exact zero. The test asserts `max|Δ| == 0.0` on all rows before the perturbation, not a
small number.

To confirm the test has teeth rather than passing vacuously, revert the fix and watch it fail — **from a
backup copy, never `git checkout` on a file with uncommitted work**:

```bash
cp backend/preprocessing/preprocess.py /tmp/preprocess_SAFE.py
# edit _clean_flow_column_causally to compute the imputation reference from the
# whole series, e.g. recent_observed = all_observed[-weekday_weeks:]
./backend/.venv/bin/python -m pytest backend/tests/test_causal_cleaning.py -q
# expected: 2 failed
cp /tmp/preprocess_SAFE.py backend/preprocessing/preprocess.py
```

---

## 4 · Positive controls: do the detectors fire on injected faults?

Passing tests only prove the *negative* controls behave. These inject leakage, no-signal and
persistence-mimicry and check what the detectors actually say.

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/test_failure_mode_distinctness.py \
  backend/tests/test_sentinel_holdout_split.py \
  backend/tests/test_signal_sentinel_semantics.py -v
```

**Expected:** `27 passed`.

**What these establish, and their known limits** (review §2.5):

| Control | Expected verdict |
|---|---|
| Pure-noise features | `signal_detected=False`, ratio ≈ 1, and **never** reported as leakage |
| Legitimate backward-only features | `signal_detected=True`, ratio ≈ 5 |
| An oracle feature equal to `y(t+h)` | ratio ≈ 447 — reported as **"signal present"**, i.e. a clean pass |

That last row is a documented gap, not a passing check: `MIN_SIGNAL_RATIO = 1.5` has no upper
counterpart, so leakage is indistinguishable from very strong signal. `check_feature_leakage` — the
only real feature-level detector — has zero production callers. Do not read a green suite as "no
leakage detected".

**TRAIN/DEV/TEST enforcement:**

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/test_evaluation_windows_enforced.py -v
```

**Expected:** `18 passed`. Confirm the holdout is still sealed:

```bash
cat experiments/test_access.log 2>/dev/null || echo "no TEST reads recorded"
```

**Expected:** the file is absent or empty. Any line in it is a TEST consultation, with its reason and
timestamp.

---

## 5 · Independent metric recomputation — *not yet runnable*

The review recomputed every published figure from `predictions_long.csv` with plain numpy and matched to
the cent (§4.1). That procedure still applies, but **the numbers to recompute no longer exist**: the
Phase-1 data change superseded them, and the re-run has not been done.

What is missing: a backtest executed against the regenerated data. Once
`backend/forecast_runs/<date>/` exists on the current data, recomputation is:

```bash
# for each family: mean|y_true - y_pred| must match the leaderboard MAE,
# and mean|y_true - origin_value| must match integrity_report.json's mae_persistence
```

**Expected tolerance when it is runnable:** exact for A_STAT and C_DL; relative 1e-15 to 1e-13 for
B_ML and E_QUANTILE, which are not byte-reproducible because `n_jobs=-1` reduces across threads in
nondeterministic order. Use numeric tolerance, not `cmp`.

---

## 6 · Backtest reproducibility — *not yet runnable*

The procedure (review §4.1) is to run the pipeline twice and diff:

```bash
MODE=backtest ./scripts/run_daily_forecast.sh
cp -R backend/forecast_runs/$(date +%F) /tmp/run1
MODE=backtest ./scripts/run_daily_forecast.sh
diff -rq /tmp/run1 backend/forecast_runs/$(date +%F)
```

**Expected when runnable:** `SUMMARY.json` and `BACKTEST_REPORT.md` byte-identical; A_STAT and C_DL
predictions byte-identical; B_ML and E_QUANTILE differing at relative ≤1e-13 with every MAE identical to
four decimals.

**Why it is not runnable as a trust check today:** the run would use the current pipeline, whose
"one shared baseline" claim is still false — E_QUANTILE is on a calendar-day index, so its `h=5` is 5
calendar days against every other family's 5 business days. A reproducible run of a
non-comparable comparison is not a verification. This is Phase-1 Step 2, not started.

---

## 7 · Artifact validator — *not built*

Phase-1 Step 5. `git ls-files '*.py' | xargs grep -l validate_artifact` returns nothing: there is no
artifact validation anywhere in the repository. `reports/VALIDATION.json` is a stub recording that.

The contract the validator must enforce is documented in `docs/reviews/2026-08-04_review.md` §6,
including the twelve known defects (C1–C12) — among them the missing `data_file` key, the missing
per-family `notes`, A_STAT's NaN leaderboard join keys, and three incompatible meanings of `rank`.

---

## Summary

| # | Check | Status |
|---|---|---|
| 1 | Test suite (225) | ✅ runnable |
| 2 | Preprocessing regeneration + SHA-256 | ✅ runnable |
| 3 | Causality property | ✅ runnable |
| 4 | Positive controls + window enforcement | ✅ runnable, with a documented leakage-detector gap |
| 5 | Independent metric recomputation | ⛔ no current backtest to recompute |
| 6 | Backtest reproducibility | ⛔ blocked on the unified yardstick |
| 7 | Artifact validator | ⛔ not built |
