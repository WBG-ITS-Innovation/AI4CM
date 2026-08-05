# UI content backlog — proposed, not implemented

Written during the phase-13 **visual** pass. Everything here changes what is computed or
displayed, so none of it was implemented. Ordered by how misleading the current behaviour is,
because that is the ordering that matters for a lab whose selling point is honesty.

Several items were *briefly implemented earlier in the session and then reverted* when the
scope was corrected to visual-only. The working code is recoverable from git history
(`0e321b7`, `9f8d27f`, `6c763a0`) if any of these is later approved.

---

## Tier 1 — the page currently states something untrue

### 1. The interval-diagnostics tab shows nothing for E_QUANTILE, and mis-scores 80% bands

`01_Dashboard` detects intervals with `{"y_lo","y_hi"}.issubset(...)` and compares coverage
against a hard-coded 90% target. Two consequences, both measured:

* **E_QUANTILE renders an empty panel.** It writes `yhat_p10`/`yhat_p50`/`yhat_p90`, so the
  column check fails — for the one model family whose entire purpose is intervals.
* **A correctly calibrated 80% band is scored as broken**, being 10 points short of a target
  it never advertised.

*Proposal:* detect both schemas and read the advertised level from the data (quantile column
names carry it: `p10`…`p90` → 80%). Where no artifact records a level, report coverage as a
measurement with no pass/fail verdict rather than assuming one.

*Blocked on:* nothing in the UI. See artifact gap A1 below for the B_ML half.

### 2. `alignment_ok` renders "never checked" as "passed"

The page reads `integrity.get("alignment_ok", True)`. Two problems: a **missing** key defaults
to a pass, and `c_dl_pipeline.py:958` writes `"alignment_ok": True` as a **literal** without
performing the check. So a DL run displays a green alignment tick that nothing verified.

*Proposal:* tri-state display — passed / failed / never verified — driven by whether the check
actually ran. **The real fix is in the backend** (artifact gap A3): the UI can only work around
a field that asserts something untrue.

### 3. Overfit-excluded models are ranked as if eligible

`overfit_excluded_models` is present in the integrity report and used **zero times** in the UI.
A model the capacity gate excluded from selection can therefore sit at the top of the
leaderboard looking like the winner.

*Proposal:* mark excluded models distinctly and state why, with the gate ratio.

### 4. Two different "best model" answers, one shown silently

The page ranks by lowest MAE via `recommender.py`; the integrity report's `best_model`
additionally applies the overfit gate. These can disagree, and the page shows one without
saying so. The gated answer is the authoritative one.

*Proposal:* reconcile and surface any disagreement.

---

## Tier 2 — a real defect is invisible

### 5. Per-magnitude-tercile coverage is not shown anywhere

The project's largest known product defect is that nominal-80% ranges capture roughly **half**
of outcomes on the largest third of days — and the largest days are the ones a cash buffer
exists for. Overall coverage hides this completely.

*Proposal:* a coverage-by-day-size breakdown wherever coverage is reported.

### 6. No reliability view, so wrong-shape and wrong-width bands look identical

Coverage alone cannot distinguish a band that is too narrow from one centred in the wrong
place. Plotting where inside its own band each actual landed separates them.

### 7. `02_History` crashes when there are no runs

With an empty runs directory the frame has no columns and `df[visible]` raises *"None of
[Index([...])] are in the [columns]"*. A fresh clone gets a traceback where it should be told
there is nothing yet. **This is a live crash, not a cosmetic gap.**

### 8. No staleness signal

If the canonical data file was updated after the newest run finished, every figure in the lab
describes an earlier version of the data and nothing says so.

---

## Tier 3 — useful, not misleading by omission

### 9. MAPE/sMAPE on flow targets

Both are offered as metrics on targets with near-zero and negative days, where a percentage
error is undefined or explodes. MASE is already computed in `metrics_long.csv`.
*Proposal:* prefer MASE for flow targets; keep MAPE available with a caveat.

### 10. The experiments log is unreachable from the lab

`experiments/log.csv` is the audit trail — every reported number with the data and code
fingerprints that produced it — and it can only be read from a terminal. A filterable view
with click-through to `experiments/runs/<run_id>.json` would make the audit trail usable by
the people the audit is for.

### 11. No cross-family view on the shared ruler

`04_Compare` compares runs but not skill against the one shared persistence benchmark, which
is the only reason skill numbers from different families are comparable at all. Any such view
must annotate each bar with its sentinel reading: a large skill number beside a failing signal
test is regression to the mean, not a forecast, and the two numbers only mean something
together.

### 12. `05_Forecast` does not say the dates are beyond the data

The forecast covers dates after the end of the data, so no actual values exist for them. Read
as "upcoming", a viewer will look for an accuracy figure that cannot exist.

### 13. Overview has no scope statement

A short "what this lab is / what it does not claim" panel would front-load the four facts that
make every other page readable: nothing is approved, the 2025 holdout has never been evaluated
against, two of three targets report a typical level rather than an event forecast, and
coverage is 3 of 41 budget lines for one week.

### 14. Pages cannot be pointed at a different runs directory

`RUNS_DIR` is hard-coded to `frontend/runs`. `frontend/paths.py` exists (added this session,
now unwired) and would let a deployment mount artifacts elsewhere. It also blocks the page
smoke tests from exercising the empty-artifact and single-run states — they currently all read
the same real artifacts, which is noted in that test's docstring.

---

## Artifact gaps — these need a backend change, not a UI change

| # | Field | Where | Why it matters |
|---|---|---|---|
| **A1** | `nominal_pi` — the advertised coverage level of B_ML's conformal intervals | `artifacts/integrity_report.json` | `ConfigBML.nominal_pi = 0.90` configures the intervals and is **never written anywhere**, so the UI has nothing to score `y_lo`/`y_hi` against |
| **A2** | Explicit quantile/nominal level for E_QUANTILE | same | Currently inferable from column names only; breaks silently if naming changes |
| **A3** | `alignment_verified`, or simply not writing `alignment_ok` | C_DL integrity report | The field is not missing, it is **false confidence** — a literal `True` with no check behind it. The only item here where an artifact actively asserts something untrue |
| **A4** | `study` | `experiments/log.csv` | Any study filter must otherwise substring-match the free-text `note`, which a reworded note silently breaks |
| **A5** | `window` | `experiments/log.csv` | Same, against `fold_scheme` |
| **A6** | `nominal_coverage` per row | `experiments/log.csv` | The log stores `coverage_low/mid/high` with no level to compare them to |

---

## Files present but deliberately unwired

Added during this session before the scope correction, left in place (nothing was deleted) and
**not imported by any page**:

* `frontend/intervals.py` — artifact-driven interval detection and calibration (items 1, 5, 6)
* `frontend/paths.py` — runs-directory resolver (item 14)
* `ui_styles.gate_badge_tri`, `ui_styles.empty_state`, `ui_styles.ds_metric`,
  `ui_styles.reading_this_chart`, `ui_styles.HELP` — components for items 2 and 7

They are inert. Wiring any of them in is a content change and belongs to whichever item above
is approved.
