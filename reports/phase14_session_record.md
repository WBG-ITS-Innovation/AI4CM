# Phase 14 — Master programme, session 1 of 2–3

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `5dcd763` (30 ahead of `origin/main` @ `863f967`)
**Root suite:** 453 → **485 passed, 3 skipped**, `EXIT=0` · **Frontend suite:** 59 → **99 passed**
**Log:** 153 rows, integrity ok · **TEST (2025) gated reads: 0** · **PR #26** open, reused
**Data SHA-256:** `0b009fd0…5361f1` · **Calendar version:** `4b480eae9c8f`

---

## STEP 0 — reconciliation, as reported at the start

| Check | Result |
|---|---|
| Working tree | clean, nothing uncommitted |
| Branch tip | `319b1de`, 23 ahead, 0 unpushed |
| Root / frontend suites | 453 (+1 skip) / 59, both `EXIT=0` |
| Log integrity · registry | `{'n_rows': 138, 'ok': True}` · 12/12 metrics |
| TEST gated reads | 0 |
| PR | #26 open → reused, no second PR opened |

**`frontend/assets/` contained exactly one file: `logo.svg`** (63,868 bytes,
`viewBox="0 0 77.199 78.92"`). It did not exist when I checked in the previous session and I had
reported 2c blocked; it was added between sessions, so 2c went ahead.

---

## Completed this session

### Part 1 — Tier-1 correctness (`bc2ce9d`)

All four Tier-1 items plus the item-7 crash, each with a regression test. Recovered from `0e321b7`
rather than rewritten, then stripped of what that commit also carried and is *not* authorised: the
MAPE→MASE swap (Tier 3) and the staleness warning (Tier 2) stay reverted, enforced by a test.

| Item | Was | Now |
|---|---|---|
| 1 · intervals | Only `y_lo`/`y_hi`, 90% hard-coded → E_QUANTILE rendered **empty**, a correct 80% band scored as broken | Both schemas; level read from the artifact; "not reported" when absent |
| 2 · alignment | `get("alignment_ok", True)` → **"never checked" read as "passed"** | Tri-state |
| 3 · overfit | `overfit_excluded_models` used **zero** times | Excluded models marked, with the gate ratio |
| 4 · best model | Two answers, one shown silently | Reconciled; the gated choice stated as authoritative |
| 7 · crash | `df[visible]` on an empty frame raised | Empty state naming file, location, command |

Mutation-tested rather than assumed: removing only the empty-state guard failed 1 test and did
**not** crash, which revealed two independent guards; removing both failed 2 including the crash
test.

### Part 2 — the console tokens, logo, chart chrome (`f262f6d`)

Palette **replaced** with the console tokens — the operator's override of `DESIGN_TOKENS.md`'s own
"not a request to restyle" line. Contrast recomputed from the hex values rather than quoted, and it
reproduces the document exactly: 16.53 / 15.43 / 7.82 / 7.30 / 8.09 on the six slots; pass 7.03,
warn 6.44, stop 7.14, accent 9.89, muted 5.93, faint 4.58, control 3.13. **All pass, none needed
adjusting.** Greyscale separation 182/255. Asserted in tests so a future edit fails in CI.

`logo.svg` inlined **verbatim**, sized by CSS only. A test compares path-by-path against the file:
every `d=`, every fill/stroke and the viewBox must survive. Another confirms a missing file
degrades to name + subtitle.

Chrome applied to 16 existing charts with a token-matched Plotly template and **non-colour
encodings** (dash + marker per role) so charts survive greyscale printing. A test asserts chrome
leaves x/y data byte-identical.

### Part 3 — artifact and metric correctness (`c1b795c`, `72af9ef`, `683a185`)

**3a, six fixes, 12 contract tests.** X9 was the serious one: b_ml wrote `quality_gate_failed`
while the others wrote `quality_gate_passed`, and the summary read only the positive key — **a
B_ML run that failed its gate was reported as passing.** One canonical key, one reader, and the
legacy key now *derived* rather than set independently. X10: one publisher, the summary derives and
records `gate_source`. F1: numeric measured/threshold. `run_id` + `schema_version` in
`SUMMARY.json`. C8: the interval's advertised level captured. X11: a baseline over zero prediction
rows raises a gate reason.

**3b, `reports/ws2_tuning.md`.** Both harness defects fixed and metrics recomputed from the
existing runs — the search was **not** re-run; `params_full.best_params` was read back and refitted
once per target. The non-canonical ruler had inflated every logged skill figure (Expenditure read
32.62%, actually 29.30%); the sentinel units mismatch had invalidated two of three readings
(Revenues 1.0000 → 1.1670; stock 0.9835 → 3.9919; Expenditure's 1.0710 reproduced **identically**,
confirming it was sound). `dev_mae` reproduced exactly for all three.

**3c, `docs/SIGNAL_FINDING.md`.** Five levers, no flow sentinel above 1.50 across 71 logged runs
and three probes; highest reading ever **1.2255** (ridge) / **1.4212** (tree).

### Part 4 — CQR, conditional gate, selection rule (`c9702e5`)

**CQR delivers the marginal guarantee and does not fix the defect, which is conditional.**

| Target | Overall | Largest third | Gate |
|---|---|---|---|
| Revenues | 83.2% → 83.2% | 69.9% → 69.9% | pass → pass |
| Expenditure | 57.2% → **78.0%** | 9.6% → **33.7%** | pass → **FAIL** |
| Stock | 58.8% → **72.0%** | 51.8% → 54.2% | pass → **FAIL** |

Revenues is the expected null — conformal width `−0`, because the band already over-covers.
Grouped calibration is **not** uniformly better: it wins on Expenditure and loses on the stock
target, where splitting 502 rows three ways costs more variance than it gains.

The new gate scores **two** axes (magnitude and trailing volatility), gates on the worst bucket,
reports every bucket, and returns *never verified* rather than a pass when nothing is judgeable.
The selection rule keeps L2 in the pool via a 1% tie-break, because Expenditure's DEV-best **is**
an L2 model.

**No registry recipe changed** — CQR lifts neither failing target over the floor.

### Part 8 — the progress record (`5dcd763`)

`reports/PROGRESS_SINCE_LAST_REVIEW.md`, every figure traced to a logged run, surfaced as a
**section** on Overview (no new page, no nav item) rendering the file directly so the two cannot
drift. Reports Expenditure being **1.0% worse than its baseline** in the same table as the wins.

---

## What changed per lab page — visual and Part-1 correctness only

| Page | Visual | Correctness |
|---|---|---|
| **Overview** | tokens, logo header, tabular numerals, 72ch measure | + progress-record section (Part 8) |
| **00_Data_Preprocessing** | tokens, logo header | — |
| **00_Lab** | tokens, logo header, chrome on 1 chart | — |
| **01_Dashboard** | tokens, logo header, chrome on 13 charts, 6 metric tooltips | intervals, alignment tri-state, overfit marking, best-model reconciliation |
| **02_History** | tokens, logo header | empty-state crash fix |
| **03_Models** | tokens, logo header | — |
| **04_Compare** | tokens, logo header, chrome on 2 charts | — |
| **05_Forecast** | tokens, logo header, chrome on the band chart, 4 metric tooltips | — |

---

## Not started — Parts 5, 6, 7

Stopped rather than begun, per the standing rule.

| Part | Scope | Note |
|---|---|---|
| **5** | Eight accuracy levers (5a–5h) | A session on its own. **5d is the priority**: Part 4 established that the residual interval defect is a fit-time problem, and trailing-volatility features target it directly |
| **6** | Forecasting first-class (6a–6f) | The forward engine, `assert_forward_only()`, retention and the scorecard all exist; what is missing is the *lab* entry point, any-target selection, re-issue, and export |
| **7** | Georgian localisation | Deliberately last per the brief — unverified translations cannot be shown to the client anyway |

---

## Artifact fields I needed that do not exist

| # | Field | Where | Status |
|---|---|---|---|
| 1 | `nominal_pi` published | `integrity_report.json` | **Captured this session** at the point of use (C8); not yet threaded into the written report, so the lab still renders "not reported" for B_ML |
| 2 | Explicit quantile level for E_QUANTILE | same | **Still absent.** Inferred from column names (`p10`…`p90` → 80%); correct today, breaks silently if naming changes |
| 3 | `alignment_verified`, or not writing `alignment_ok` | C_DL | **Still absent — the worst of these.** `c_dl_pipeline.py:958` writes a literal `True` with no check behind it. Not a missing field but a **false** one; the UI works around it by pipeline name |
| 4 | `study` | `experiments/log.csv` | **Still absent.** Any study filter must substring-match the free-text `note` |
| 5 | `window` | `experiments/log.csv` | **Still absent.** Same, against `fold_scheme` |
| 6 | `nominal_coverage` per row | `experiments/log.csv` | **Still absent.** The log stores `coverage_low/mid/high` with no level to compare them to |
| 7 | Per-tercile coverage for point-model runs | `experiments/log.csv` | **Blocked a deliverable.** The incumbents ran through the point path, which logs no coverage, so `ws2_tuning.md` could not publish a before/after interval comparison — stated there rather than estimated |

---

## Mistakes I made, and what caught them

* **A test matched its own explanatory comment.** The Tier-1 regression test asserted the removed
  code was absent, and the comment quoting that code satisfied the match. Assertions now run on
  executable lines only.
* **A CQR bug manufactured a passing gate.** Bucketing calibration scores by the *actual* `y` while
  applying corrections by the *predicted* midpoint put them in the wrong buckets: grouped CQR came
  out **worse than no calibration** on Expenditure (57.2% → 43.2%) and produced a **passing** gate
  on the stock target (85.6% / 71.1%) that the correct version fails (68.4% / 54.2%). Caught
  because the numbers were implausibly good in one place and implausibly bad in another.
* **Tightening `read_gate` broke two tests, and the fixture was only half the story.** A stale
  fixture omitted the gate key, but `run_status == "SUCCESS"` is a genuine positive assertion, so
  reading it as unverified would have misrepresented historical artifacts. Added as a fallback
  strictly after both keys, with six resolution cases asserted.
* **An import landed inside a nested `try` block**, where its scope 300 lines later was not
  guaranteed. Moved to module level.
* **`scripts/ws2_tune.py` read `os.environ["SC"]` at import**, so it could not be imported as a
  library. Defaulted.

---

## Reproduction

```bash
git checkout model/excellence            # 5dcd763
./backend/.venv/bin/python -m pytest -q; echo "EXIT=$?"        # 485 passed, 3 skipped
PYTHONPATH=frontend:backend ./frontend/.venv/bin/python -m pytest -q frontend/tests   # 99
SC=/tmp/ws2 OUT=/tmp/ws2r.csv ./backend/.venv/bin/python scripts/ws2_recompute.py
SC=/tmp/ws2 OUT=/tmp/ws7.csv  ./backend/.venv/bin/python scripts/ws7_cqr.py
cd frontend && ./.venv/bin/streamlit run Overview.py
```
