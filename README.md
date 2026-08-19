# AI4CM Forecast Lab

Daily cash forecasting for the Georgian Treasury, built so that every number it
publishes can be traced to the evidence that earned it — and so that a number
which has not earned its evidence is **withheld rather than shown**.

The Lab forecasts Treasury cash lines five business days ahead, puts each
forecast through publication gates that decide whether it may be published at
all, and records what it said before the answer existed so it can later be
scored against what actually happened.

Two ways into this document:

- **If you work at the Treasury** and want to know what this does and what you
  can trust it for → [What it is](#what-it-is),
  [What it will not do](#what-it-will-not-do),
  [What has been measured](#what-has-been-measured), then
  [Current state](#current-state) and [Known limits](#known-limits).
- **If you are reviewing the code** → [How trust is enforced](#how-trust-is-enforced),
  [Setup](#setup), [Running a forecast](#running-a-forecast-and-reading-the-gates),
  and [Repository layout](#repository-layout).

---

## What it is

Three Treasury cash lines are forecast today, at a horizon of **five business
days**:

| Target | Kind | Champion recipe | Publication verdict |
|---|---|---|---|
| **Revenues** | flow | `LightGBM_L1` + `GBQuantile` intervals | **publishable** |
| **Expenditure** | flow | `LightGBM_L1` + `GBQuantile` intervals | withheld — fails accuracy and signal |
| **State budget balance** | stock | `HistGBDT_L1` + `GBQuantile` intervals | withheld — fails accuracy |

Three of 41 lines in the dataset. The rest have not been modelled.

Each official forecast is published as an immutable issue under
`forecasts/published/<issue_date>/`, holding the forecast itself, the gate
verdicts that let it through, a manifest recording the data and git SHA it was
built from, and provenance. A published issue is a record of what was said
*before its truth existed* — which is what makes scoring it afterwards
meaningful rather than circular.

## What it will not do

These are design decisions, not gaps:

- **It will not publish a line that fails a gate.** When the gates withhold a
  line, you get the verdict and the plain-language reason. Two of the three
  targets are withheld today, and the Lab does not relax a gate to make a model
  pass.
- **It will not forecast further than five business days.** That is the only
  horizon at which the shared benchmark, recipe selection and every gate were
  measured. An official run at any other horizon is **refused**, not quietly
  relabelled — an official-looking forecast at h=1 would carry credentials
  never earned at h=1.
- **It will not let an analyst swap the model on an official run.** The
  champion is fixed by recorded evidence; a selectable "official" model would
  make the label meaningless.
- **It will not treat "not measured" as "passed".** Every gate is tri-state —
  pass, fail, or not measured — and an unmeasured gate is reported as such.
- **It will not read its sealed holdout to make a choice.** Selection reads
  raise; reporting reads are permitted and logged. See
  [the ledger](#the-sealed-holdout-and-its-ledger).

---

## What has been measured

Measured on the **sealed holdout** (2025-01-01 … 2025-08-06), at h=5, with a
training embargo applied and the read logged to the holdout ledger. n=146 rows
per target. Recorded 2026-08-18.

| Target | Champion | vs naive baseline | vs the Treasury's current method |
|---|---|---:|---:|
| **Revenues** | `LightGBM_L1` | **+55.96%** | **+31.99%** |
| **Expenditure** | `LightGBM_L1` | +29.19% | **+2.79%** |
| **State budget balance** | `HistGBDT_L1` | +33.70% | **n/a** |

Percentages are reductions in mean absolute error. Absolute figures are
deliberately not reproduced here; they are in
`reports/sealed_window_champion_vs_ops.csv` and the session records.

**These caveats travel with the table, not in a footnote:**

- **Revenues is the strong case.** A third better than the method the Treasury
  uses today, on data the models had never seen.
- **Expenditure's margin is thin.** +2.79% is better than the current method,
  but by a margin a different window could erase — and its publication verdict
  is `withheld` on accuracy regardless.
- **The stock target has no comparison to the current method at all.** That
  method aggregates a flow to an annual total, and a balance level has none. It
  is recorded as `n/a` rather than invented.
- **n=146, not 156.** Four rows were dropped by the training embargo; the rest
  had incomplete features at the window edge.
- **This measures the champion *recipe*, not a continuation of its logged
  credentials**, which are not reproducible — see
  [Known limits](#known-limits). The Lab reports that gap beside every
  sealed-window figure so the two can never be read as continuous.
- **Every "N% better than the Treasury's current method" figure produced before
  2026-08-18 is withdrawn**, including the July deck. Those were computed
  against a baseline that returned identically zero, overstating the margin
  roughly fourfold. The defect and its correction are recorded in
  `docs/sessions/2026-08-18-session6-prep.md`.

---

## How trust is enforced

The forecasting is ordinary supervised learning. What makes the output
trustworthy is the machinery around it, and that machinery is code with tests,
not convention.

### Four windows, and rolling origins

```
TRAIN   2015-01-05 .. 2023-12-31    model fitting
DEV     2024-01-01 .. 2024-12-31    all tuning, feature selection, thresholds, model comparison
TEST    2025-01-01 .. 2025-08-06    SEALED holdout — final reporting only
LIVE    2025-08-07 .. (data end)    arrived after sealing — scored, never used to choose
```

Defined once in [`backend/evaluation_windows.py`](backend/evaluation_windows.py)
and used by every family and every tuner, because the audit's most expensive
finding was that each family had been measuring itself against its own baseline
on its own window.

Selection is permitted **only** on TRAIN and DEV, and `assert_selection_free`
enforces it. TRAIN is a floor rather than a cap: when evaluating on DEV or TEST
the pipelines roll their origins forward, so a fold predicting 2025-06-01
legitimately trains on everything up to 2025-05-31. What never happens is
training on data at or after the origin being predicted from.

A fold is also required to have its **truth** inside an allowed window, not just
its origin. That distinction is not academic — it was a real defect, found and
fixed on 2026-08-18: folds selected rows by origin and scored them at
`origin + 5 business days`, so four DEV rows were being scored against holdout
truth while the guard passed, because it was checking the wrong dates.

### The sealed holdout and its ledger

`require_test_access(reason, caller, purpose)` gates every read of the holdout,
and distinguishes two different acts:

- **`PURPOSE_SELECTION`** — a read that could inform a choice. **Raises** unless
  the holdout has been deliberately released via `AI4CM_ALLOW_TEST_READ=1`, and
  prints an unmissable stderr banner when it is. This count should stay at zero.
- **`PURPOSE_REPORT`** — evaluating over the holdout to state what would have
  happened. This is what a holdout is *for*, so it never raises — but it is
  logged, because "how many times did we look at it" must have a factual answer
  rather than a recollection.

Both append to `experiments/test_access.log`, naming the reason, the caller and
the dates covered.

### Two baselines, not one

Every leaderboard row and every scorecard row carries **two** comparators:

- **The naive ruler** — h=5 business-day persistence. The single shared ruler
  across all four families, so that "skill" means the same thing in every
  report.
- **The Ops baseline** — the Treasury's current planning method: a three-year
  average annual total, split across months by each month's historical share,
  then spread flat across working days. Reconstructed
  **vintage-correctly**: each row is scored against the version of the baseline
  that was actually in force at its own origin, so the comparison is literally
  "against what the Treasury had in hand". It is `NaN` for the stock target,
  with the reason recorded.

Selection and the gates use MASE against a TRAIN-only seasonal-naive
denominator. `skill_vs_ruler` is reported but no longer gates: measured, "beat
persistence by any margin" is close to vacuous on a spiky flow — a featureless
constant scores 38.58% on Revenues against the same ruler.

### The sentinels

- **Signal (shuffled-target control).** Fit the same features twice, once on
  true targets and once on shuffled ones, and compare held-out error. If the
  features carry real information, destroying the pairing should hurt. The
  threshold is **1.15**, calibrated against a measured null distribution — 360
  permutation draws, false-positive rate 0.00%, null maximum 1.1164. It replaced
  a bare constant of 1.50 whose null distribution had never been estimated and
  which was rejecting real readings.
- **Leakage.** Feature-level checks, plus step-based alignment validation
  (`idx(target) − idx(origin) == h`) and origin-before-target enforcement.
- **Persistence mimicry.** Is the forecast just a lagged copy of the target?
  Detected by comparing correlation at every shift: an honest model peaks at
  shift 0, a lagged copy does not.
- **Shift diagnostics** and an **overfitting ratio** (train/DEV MAE), reported
  alongside.

### The gates, and the three verdicts

[`backend/publication_gates.py`](backend/publication_gates.py) is the
publication decision. The registry is its *output*, and a test asserts the two
agree — so the reasoning cannot live in a commit message while the numbers live
in a file.

| Gate | Threshold | Failure gives |
|---|---|---|
| **accuracy vs naive** | MASE < 1.0 | `withheld` |
| **leakage** | no future information in features | `withheld` |
| **signal** | sentinel ≥ 1.15 | `withheld_as_forecast` |
| **persistence mimicry** | predictions must not be a lagged copy | `withheld_as_forecast` |
| **interval coverage** | outcomes fall inside the band at its nominal rate, ±10 points | `withheld_as_forecast` |

MASE's threshold is definitional rather than chosen: it is a ratio against a
benchmark, so 1.0 is break-even. There is deliberately no safety margin —
adding one would reintroduce the uncalibrated-constant problem the module
exists to fix. The margin is *reported* instead, so a model at 0.99 is visibly
marginal.

A recipe that fails nothing is `publishable`. The other two verdicts differ in
what they withhold:

- **`withheld`** — a documented alternative is strictly better, so showing the
  numbers at all would invite a worse decision than not showing them.
- **`withheld_as_forecast`** — the numbers remain the best central-tendency
  estimate available, but a specific claim cannot be made. The numbers are
  shown; the claim is withheld.

Every failing gate is reported, not only the one that set the verdict: a model
that both loses to the naive benchmark and shows no signal has two problems,
and whoever fixes one should know about the other.

### OFFICIAL and EXPLORATORY

Separated at the boundary in
[`backend/forecast_modes.py`](backend/forecast_modes.py), not by convention in
the UI — a page can forget a flag.

- **OFFICIAL** — the target's registry champion recipe, refit on all data
  through the data end, published immutably. The model is not selectable.
- **EXPLORATORY** — any model, any target, any horizon. It runs and displays,
  carries its own caveat banner, and `ExploratoryResult` **has no publish method
  at all**. It can never reach `forecasts/published/` or the scorecard.

---

## The four model families

| Family | Models | Status |
|---|---|---|
| **A · Statistical** (`A_STAT`) | ETS, SARIMAX, STL-ARIMA, Theta, simple baselines | Runs; no champion promoted |
| **B · Machine Learning** (`B_ML`) | Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, HistGBDT, XGBoost, LightGBM (CatBoost registered but unablated) | **All three champions are here** |
| **C · Deep Learning** (`C_DL`) | LSTM, GRU, DCNN, Transformer, MLP | Parked; no recipe promotes a C_DL model |
| **E · Quantile** (`E_QUANTILE`) | GBQuantile, ResidualRF — P10/P50/P90 | Supplies the interval model for all three champions |

Stated plainly because it is the kind of result that tends to go unmentioned:
on the corrected comparison, **every C_DL model is 48–58% worse than the
Treasury's current method** on Revenues. The family is kept because the
comparison is worth having, not because it is close.

---

## Setup

Python **3.11–3.13**. The suite is currently green on 3.13.11.

**macOS / Linux**
```bash
chmod +x scripts/setup_unix.sh scripts/run_app_unix.sh
./scripts/setup_unix.sh          # builds both venvs and wires the frontend to the backend
./scripts/run_app_unix.sh        # http://localhost:8501
```

**Windows (PowerShell)**
```powershell
scripts\setup_windows.bat
scripts\run_app_windows.bat
```

The backend and frontend have **separate virtual environments** — the modelling
stack lives in `backend/.venv` and belongs there. The UI locates the backend
through `frontend/.tg_paths.json`, written by the setup script.

**The Treasury dataset is not in this repository.** It was removed on
2026-08-14 and is loaded from secure storage; `backend/data/` is untracked.
Without it the app still starts and the suite still runs — data-dependent tests
**skip** rather than fail. `frontend/sample_data/` holds synthetic CSVs for
exercising the UI.

### Running the tests

```bash
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q
```

Expected with the dataset in place: **942 passed, 4 skipped**, about nine and a
half minutes. The four skips are frontend tests needing streamlit, which is
installed in `frontend/.venv` only.

---

## Running a forecast, and reading the gates

From the repository root:

```bash
# An official forecast for one target, at the validated horizon
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode official --target Revenues \
    --data backend/data/processed/master_daily_clean_treasury.csv

# The same, published as a new immutable issue
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode official --target Revenues \
    --data backend/data/processed/master_daily_clean_treasury.csv \
    --publish

# Anything else — any model, any horizon — must be exploratory
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode exploratory --target Expenditure --model ETS --horizon 10 \
    --data backend/data/processed/master_daily_clean_treasury.csv
```

`--issue-date` defaults to the next free issue date. It exists because deriving
the issue date from the data instead produced the same date on every run from an
unchanged file, and therefore collided on the second publish.

Scoring published issues against actuals once truth arrives:

```bash
./backend/.venv/bin/python backend/run_publish_and_score.py
```

### Reading `gates.json`

Each published issue carries a `gates.json` keyed by recipe id. Each gate reads:

```json
"accuracy_vs_naive": {
  "passed": true,
  "name": "accuracy vs repeating the same weekday last week",
  "metric": "MASE",
  "measured": 0.757959,
  "threshold": 1.0,
  "margin_pct": 24.2,
  "reason_plain": "This model is 24.2% more accurate than simply repeating what happened on the same weekday last week, so it is the better of the two estimates."
}
```

Three things to read carefully:

1. **`passed` is tri-state.** `true`, `false`, or `null` — and `null` means *not
   measured*, never *passed*. Unmeasured gates are listed separately in
   `unmeasured_gates`.
2. **`reason_plain` is written for a reader with no notes.** It carries no
   acronyms and no model names; tests enforce that.
3. **`decided_by` names the gate that set the verdict**, while `reasons` lists
   every failure. A verdict has one cause; a model can have several problems.

The UI presents the same material: `05_Forecast` runs official and exploratory
forecasts, and the Dashboard's *Forecast Integrity* tab shows alignment,
leakage, shift diagnostics and the gate outcome for an individual run.

---

## Current state

Honest status as of **2026-08-19**:

| | |
|---|---|
| Test suite | **942 passed, 4 skipped** |
| Coverage | 3 of 41 Treasury lines, 5 business days ahead |
| Publishable targets | **1 of 3** (Revenues) |
| Sealed-holdout evaluation | Done, logged, reported — supersedes the July deck |
| Published issues | 3 |
| **Forecasts scored against a real outcome** | **Zero.** 25 rows pending |
| Approvals | **None.** Every recipe is `candidate — pre-tuning`; `approved_by` is null |
| Tuning | None. All principled defaults |

**No forecast has yet been scored against a real outcome.** Every published
forecast targets a date whose actuals have not arrived, so the scorecard is
correctly empty — schema-complete, zero rows. Two things follow that a reader
should not have to infer:

- **There is no realized track record.** The accuracy figures above come from
  evaluation on held-out historical windows, not from watching these forecasts
  come true.
- **Nothing here is approved.** No approval workflow exists yet, so no model can
  honestly be called production-ready.

The forecasting has been **honestly evaluated** on held-out historical data. It
has not been proven in production, and this document will not say it has until
forecasts have been scored against real outcomes.

## Known limits

Open, known, and written down rather than discovered late. Each traces to a
session record.

1. **The training embargo is incomplete.** Five training rows carry targets
   inside the evaluation block, so the model is fitted on answers from the block
   it is then scored on. It is **not** a holdout breach. Removing them moves DEV
   error by −2.98% on Revenues (better) and **+0.65% on Expenditure (worse)** —
   model variance from a 0.2% change in training data rather than a bias being
   removed, which is exactly why it is not being waved through as an obvious
   correctness win. Pinned by an inverted test that fails once it is
   fixed. *(Top priority; `docs/sessions/2026-08-18-dev-fold-holdout-leak.md`)*
2. **The champion credentials are not reproducible.** The script that produced
   them is absent from the repository. Reconstruction differs by **5.13%**
   (Revenues), **8.12%** (Expenditure) and **19.45%** (stock target). The Lab
   reports that gap beside every sealed-window figure so nothing can be
   presented as continuous with credentials it cannot reproduce. Either
   reconstruct the original harness or formally supersede the credentials.
   *(`docs/sessions/2026-08-18-artifact-regeneration.md`)*
3. **The signal sentinel cannot be recomputed**, for the same missing-harness
   reason — reconstruction returns exactly 1.0000 against a logged 1.2255. So
   the effect of the 2026-08-18 fold fix on the `signal` gate is **unmeasured,
   not zero**. The verdicts stand on `accuracy_vs_naive`, which is the gate that
   actually decided them.
4. **Expenditure's two harnesses disagree by 8.6%.** Its reconstructed MASE is
   0.9961 — *under* the 1.0 threshold — against a logged 1.1039. Which side of
   the gate it falls on depends on which harness is asked. The registry was not
   touched; its verdict remains `withheld`.
5. **Nothing has been scored yet**, and the schema's first contact with reality
   is still ahead of it. The fixtures behind it are synthetic by necessity.
6. **`b_ml_pipeline` keeps a divergent Ops method** — a 12-year shift rather
   than the Treasury's method. Unused for reporting; should be deleted or
   delegated like the other three.

Several older documents in this repository predate the work above and carry a
dated notice saying what supersedes them: `CHANGELOG.md`, `VERIFICATION.md`,
`docs/ROADMAP.md`, `docs/SIGNAL_FINDING.md` and `docs/EXECUTION_PLAN.md`. Their
reasoning is still worth reading; their status lines are not current.

---

## Repository layout

```
AI4CM/
├── backend/                    Modelling, gates, publishing, scoring
│   ├── evaluation_windows.py   The four windows, the ledger, MASE
│   ├── publication_gates.py    The publication decision and its thresholds
│   ├── forecast_modes.py       OFFICIAL vs EXPLORATORY, and the CLI
│   ├── published_forecasts.py  Immutable issues, the vault, the scorecard
│   ├── ops_baseline.py         The Treasury's current method, vintage-correct
│   ├── sealed_window_report.py Reporting over the holdout, embargoed and logged
│   ├── forecast_integrity.py   Sentinels: signal, leakage, persistence mimicry
│   ├── artifact_validation.py  Fails loudly before anything is published
│   ├── {a_stat,b_ml,c_dl,e_quantile}_*.py    The four families
│   └── tests/                  The larger half of the suite
├── frontend/                   Streamlit UI (Overview + 7 pages)
├── registry/recipes.json       Champion recipes, credentials, gate verdicts
├── forecasts/
│   ├── published/<date>/       Immutable issues: forecast, gates, manifest, provenance
│   └── scorecard.csv           Scored claims — schema-complete, zero rows
├── experiments/                Append-only run log and the holdout ledger
├── reports/                    Workstream analyses and session records
├── scripts/                    Setup, daily run, gate reruns, tuning
└── docs/                       Durable documentation and session records
```

Generated and untracked: `backend/data/`, `backend/forecast_runs/`,
`frontend/runs/`, `frontend/runs_uploads/`, both `.venv/` directories,
`frontend/.tg_paths.json`.

## Documentation

- [`docs/sessions/`](docs/sessions/README.md) — one record per working session:
  what was attempted, what was found (including where the brief's premise turned
  out to be wrong), what was verified with real output, and what was
  deliberately left undone. The detailed history behind everything summarised
  here.
- [`docs/AGENT_ARTIFACT_CONTRACT.md`](docs/AGENT_ARTIFACT_CONTRACT.md) — the
  published interface consumed by the AI4CM Agent, and what absence means in it.
- [`docs/DATA_SEMANTICS.md`](docs/DATA_SEMANTICS.md) — what the dataset's
  columns mean, and the questions still open with Treasury.
- [`docs/FISCAL_CALENDAR_SOURCES.md`](docs/FISCAL_CALENDAR_SOURCES.md) — the
  fiscal calendar's sources, and the sign-off requested from Treasury.
- [`reports/`](reports/) — workstream analyses, including the sentinel
  calibration study and the sealed-window comparison. **Read with the date in
  mind:** several of these predate 2026-08-18 and quote comparisons against the
  Treasury's current method that have since been withdrawn. Unlike the documents
  listed above, they do not yet carry notices saying so.

### A related repository

The **AI4CM Agent** is a conversational interface to this Lab: it answers
questions about the published forecast, runs official forecasts, takes in new
actuals and scores past forecasts — always by calling the Lab's own audited
entry points, never by reimplementing them. It cannot disagree with this
pipeline about what is trustworthy, because it has no code that could form a
second opinion.
