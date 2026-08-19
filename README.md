# AI4CM Forecast Lab

Daily cash forecasting for the Georgian Treasury, built so that every number it
publishes can be traced to the evidence that earned it — and so that a number
which has not earned its evidence is **withheld rather than shown**.

The Lab forecasts Treasury cash lines five business days ahead, puts each
forecast through publication gates that decide whether it may be published at
all, and records what it said before the answer existed so it can later be
scored against what actually happened.

Two ways into this document:

- **If you want to use it** → [What it is](#what-it-is),
  [What it will not do](#what-it-will-not-do),
  [What has been measured](#what-has-been-measured), then
  [Getting started](#getting-started-no-technical-background-needed) and
  [Your first 10 minutes](#your-first-10-minutes).
- **If you are reviewing the code** → [How trust is enforced](#how-trust-is-enforced),
  [For developers](#for-developers), [Current state](#current-state),
  [Known limits](#known-limits) and [Repository layout](#repository-layout).

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
  relabelled.
- **It will not let an analyst swap the model on an official run.** The
  champion is fixed by recorded evidence; a selectable "official" model would
  make the label meaningless.
- **It will not treat "not measured" as "passed".** Every gate is tri-state —
  pass, fail, or not measured — and an unmeasured gate is reported as such.
- **It will not overwrite what it previously said.** Publishing twice in one day
  produces a second, suffixed issue rather than replacing the first.
- **It will not read its sealed holdout to make a choice.** Selection reads
  raise; reporting reads are permitted and logged.

---

## What has been measured

Sealed holdout (2025-01-01 … 2025-08-06), h=5, n=146 per target, recorded
2026-08-18. Percentages are reductions in mean absolute error.

| Target | Champion | MASE | vs naive baseline | vs the Treasury's current method |
|---|---|---:|---:|---:|
| **Revenues** | `LightGBM_L1` | 0.758 | **+55.96%** | **+31.99%** |
| **Expenditure** | `LightGBM_L1` | 1.104 | +29.19% | **+2.79%** |
| **State budget balance** | `HistGBDT_L1` | 1.578 | +33.70% | **n/a** |

> **Read these with [Known limits](#known-limits).** Only Revenues is
> publishable; Expenditure's margin is thin and the stock target has no
> comparison to the current method at all. What each figure does and does not
> cover is set out there, and in full in
> [`docs/sessions/`](docs/sessions/README.md).

Every "N% better than the Treasury's current method" figure produced **before
2026-08-18 is withdrawn**, the July deck included — those were computed against
a baseline that returned identically zero.

---

## Getting started (no technical background needed)

This section assumes you have never opened a terminal. It takes about fifteen
minutes, most of which is waiting for downloads.

### Before you begin

**1. You need Python.** Version 3.11, 3.12 or 3.13. Download it from
[python.org/downloads](https://www.python.org/downloads/) and run the installer.

> **On Windows, tick the box that says "Add Python to PATH"** on the first
> installer screen. It is easy to miss and everything below fails without it.

**2. You need a terminal.** This is a window where you type commands.

- **Windows:** press the Start button, type `PowerShell`, and open it.
- **Mac:** press `Cmd + Space`, type `Terminal`, and press Enter.

**3. Check Python is installed.** Type this and press Enter:

```
python3 --version
```

On Windows, type `python --version` instead.

You should see something like `Python 3.13.11`. If you see an error, see
[If something goes wrong](#if-something-goes-wrong) below.

### The steps

Type each command, press Enter, and wait for it to finish before the next one.
Copy and paste rather than retyping.

**Step 1 — go to the folder where you keep documents.**

```
cd Documents
```

*Nothing visible happens. That is correct.*

**Step 2 — download the code.**

```
git clone https://github.com/WBG-ITS-Innovation/AI4CM.git
```

*You will see `Cloning into 'AI4CM'...` and a progress counter. It creates a new
folder called `AI4CM`.*

> No `git`? Download the code as a ZIP from the repository page in your browser
> (green **Code** button → **Download ZIP**), unzip it into `Documents`, and
> rename the folder to `AI4CM`. Then continue from Step 3.

**Step 3 — go into the folder you just downloaded.**

```
cd AI4CM
```

*Nothing visible happens. Every command from here runs inside this folder.*

**Step 4 — create a private space for the app's software.**

On **Mac**:

```
python3 -m venv frontend/.venv
```

On **Windows (PowerShell)**:

```
python -m venv frontend\.venv
```

*Takes a few seconds and prints nothing. It makes a folder that keeps this
project's software separate from the rest of your computer.*

**Step 5 — switch into that space.**

On **Mac**:

```
source frontend/.venv/bin/activate
```

On **Windows (PowerShell)**:

```
frontend\.venv\Scripts\Activate.ps1
```

*Your prompt changes — a `(.venv)` appears at the start of the line. That is how
you know it worked.*

**Step 6 — install what the app needs.**

```
python -m pip install -r frontend/requirements.txt
```

*This downloads four packages and their dependencies; it takes a minute or two
and prints a lot of text. The last line should begin `Successfully installed`
and include `streamlit-1.40.1`.*

**Step 7 — start the app.**

```
python -m streamlit run frontend/Overview.py
```

*The terminal prints a short block ending in a web address.*

### You know it worked when…

The terminal shows:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

and your browser opens by itself at that address. If it does not open, copy
`http://localhost:8501` into your browser's address bar.

The page you land on is titled **Georgia Treasury Forecast Lab**, with a row of
links across the top — *Open Lab*, *Open Dashboard*, *Compare Runs*, *See
History*, *Read about Models*, *Data Pre-processing* — and a **Recent runs**
section at the bottom which, on a fresh install, says *"No runs yet"*.

**Leave the terminal window open.** Closing it stops the app. To stop it
deliberately, click the terminal and press `Ctrl + C`.

### To run models as well

Steps 1–7 give you the app and let you read everything already recorded. Running
a model needs a second, larger environment, because the modelling libraries live
apart from the interface on purpose.

On **Mac**, one command does it and also tells the app where to find it:

```
./scripts/setup_unix.sh
```

On **Windows**:

```
scripts\setup_windows.bat
```

*This one takes several minutes — it downloads PyTorch and several other large
libraries. It ends with `✅ Setup complete.`*

Check it worked:

```
./backend/.venv/bin/python scripts/verify_backend_env.py
```

*Prints a short list where every entry says `OK`:*

```
{
  "numpy": "OK",
  "pandas": "OK",
  ...
  "torch": "OK",
  "openpyxl": "OK"
}
```

**One more thing you will need: the data.** The Treasury dataset is not in this
repository and never will be — it was removed on 2026-08-14 and is held in
secure storage. Without it the app runs and every page loads, but the pages that
forecast have nothing to forecast from. Put the dataset at
`backend/data/processed/master_daily_clean_treasury.csv`.

### If something goes wrong

**"python3: command not found" / "Python was not found"**

Python is not installed, or the installer did not add it to your PATH. On
Windows, re-run the Python installer, choose **Modify**, and make sure **"Add
Python to PATH"** is ticked. On Mac, try `python3` rather than `python` — macOS
does not ship a command called `python`.

**Windows: "running scripts is disabled on this system"**

PowerShell blocks scripts by default, which stops Step 5. Run this once, then
repeat Step 5:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

It applies to your user account only and does not need an administrator.

**`pip install` fails with a certificate or connection error**

This is normal on a corporate network that inspects traffic. Ask your IT
department for the organisation's certificate bundle, then point Python at it
before retrying Step 6:

```
python -m pip install --cert /path/to/bundle.pem -r frontend/requirements.txt
```

If your organisation runs its own package mirror, use that instead:

```
python -m pip install --index-url https://your-mirror/simple -r frontend/requirements.txt
```

**"Port 8501 is already in use"**

The app is already running in another terminal window — look for it before
starting a second copy. If you want two, give the new one its own port:

```
python -m streamlit run frontend/Overview.py --server.port 8502
```

and open `http://localhost:8502` instead.

**The app starts but a page says "Forecast generation needs the backend
interpreter"**

You have completed Steps 1–7 but not the [modelling
environment](#to-run-models-as-well). Everything already recorded still
displays; only generating something new is blocked.

### For developers

Both environments, wired, from a clean checkout:

```bash
git clone https://github.com/WBG-ITS-Innovation/AI4CM.git && cd AI4CM
./scripts/setup_unix.sh        # both venvs + frontend/.tg_paths.json
./scripts/run_app_unix.sh      # http://localhost:8501
```

The backend and frontend keep **separate virtual environments** deliberately:
the modelling stack lives in `backend/.venv`, and the Streamlit process never
imports it. Pages that need a model dispatch to that interpreter as a subprocess
and read JSON back. `frontend/.tg_paths.json` is how the UI locates it.

Running a forecast from the command line, from the repository root:

```bash
# Official — uses the target's registry champion, at the validated horizon
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode official --target Revenues \
    --data backend/data/processed/master_daily_clean_treasury.csv

# ... and publish it as a new immutable issue
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode official --target Revenues \
    --data backend/data/processed/master_daily_clean_treasury.csv --publish

# Anything else — any model, any horizon — is exploratory and cannot be published
./backend/.venv/bin/python backend/forecast_modes.py \
    --mode exploratory --target Expenditure --model ETS --horizon 10 \
    --data backend/data/processed/master_daily_clean_treasury.csv

# The forward forecast the UI's Forecast page reads
./backend/.venv/bin/python backend/run_forward_forecast.py

# Score published issues against actuals, once truth has arrived
./backend/.venv/bin/python backend/run_publish_and_score.py
```

The test suite:

```bash
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q
```

Expected with the dataset in place: **942 passed, 4 skipped**, about nine and a
half minutes. Without the dataset the data-dependent tests **skip** rather than
fail. The four standing skips are frontend tests needing streamlit, which is
installed in `frontend/.venv` only.

---

## Your first 10 minutes

A guided tour that doubles as a smoke test: if all five steps behave as
described, your installation is sound.

**What you need first.** Steps 2–4 need the modelling environment and the
dataset. Step 1 and step 5 work without either.

### 1 · Confirm the app found its backend — 1 minute

On the **Overview** page, look at *Backend paths (auto-detected)*. Two boxes
should be filled in: a path ending in `backend/.venv/bin/python`, and one ending
in `backend`.

**Blank boxes** mean the setup script has not run or the app was started from
the wrong folder. Fix that before going further — every later step depends on
it.

### 2 · Run a model and read the leaderboard — 4 minutes

Open **Lab** from the top row. Pick a target, leave the cadence and horizon at
their defaults, choose family **B · Machine Learning**, and press run.

Before it launches, the page runs a **data quality pre-flight** and shows what
it found under three headings — *blockers*, *warnings*, *info*. A blocker stops
the run. **This is the system working**: it is refusing to spend ten minutes
producing a number from input it has already identified as unusable. Read what
it says rather than looking for a way past it.

When the run finishes, open **Dashboard** and select the **Leaderboard** tab.
There is a *Metric* selector above the chart. The two entries that answer
different questions:

| Metric | The question it answers |
|---|---|
| **MAE** | How large is this model's typical error, in lari? Lower is better. |
| **MAE_skill_vs_Ops** | How much better is it than the Treasury's current planning method? Positive is better; **negative means the current method wins.** |

Two things to notice, both deliberate:

- **A row called `⚡ Persistence (baseline)`.** That is the naive ruler — "assume
  today repeats in five business days" — carried as a competitor rather than
  hidden in a footnote. A model that cannot beat it has not earned anything.
- **Red bars.** Those are models the overfitting gate **excluded from
  selection**. They are drawn for comparison only, and a low bar there is not a
  good model — it memorised the history.

### 3 · Find the gate verdicts and their reasons — 2 minutes

Open **Forecast**. On a fresh install this page will say **"No forward run
found."** — it shows predictions for dates beyond the end of the data, and those
are generated on demand rather than committed. It prints the command; run it in
a second terminal, from the repository folder:

```
./backend/.venv/bin/python backend/run_forward_forecast.py
```

Then reload the page. Each target now gets a coloured banner:

- **Green — "Usable as a forecast."**
- **Red — "WITHHELD — do not use these numbers."** A trivial benchmark beats
  this model, so the numbers are not a guide to anything.
- **Red — "WITHHELD as a forecast — shown as a guide to the typical level."**
  The numbers are the best central estimate available, but a specific claim
  cannot be made from them.

Under each banner is a row of **Checks**, one per gate, each with a tick or a
cross and a plain-language sentence — no acronyms, no model names. That sentence
is the answer to "why is this line not published?", and it is generated from the
same code that made the decision, not written by hand.

Expect two of the three targets to be red. That is the current honest state, not
a fault in your installation.

### 4 · Run an official forecast, and watch it refuse to overwrite — 2 minutes

Still on **Forecast**, in *Generate a forecast*, choose **Official**, pick
**Revenues**, tick *Publish to forecasts/published/ under a new issue date*, and
run.

Now run it a second time. The second issue is **not** written over the first: it
lands at today's date with `-r2` appended. Run it again and you get `-r3`.

**This is the system working, not a bug.** A published forecast is the only
record of what was said on a given day. Overwriting it would destroy the
evidence that scoring later depends on, so the Lab suffixes instead. The same
principle explains the *Verdict history* panel further down the page: when the
gates changed, published issues were left exactly as issued and the panel shows
what they would say today alongside what they said then. It never rewrites the
original.

### 5 · Find the files on disk — 1 minute

Nothing in the interface is the source of truth; every panel is reading a file
you can open yourself.

| What | Where |
|---|---|
| Runs you launched from the **Lab** page | `frontend/runs/<run_id>/outputs/` |
| Published official forecasts | `forecasts/published/<issue_date>/` |
| Gate verdicts for a published issue | `forecasts/published/<issue_date>/gates.json` |
| Scored claims, once truth arrives | `forecasts/scorecard.csv` |
| Daily pipeline output | `backend/forecast_runs/<date>/<family>/` |
| Every read of the sealed holdout | `experiments/test_access.log` |

Open a `leaderboard.csv` next to the chart you were just looking at and confirm
the numbers match. If they do, the tour has done its job.

> **A known rough edge:** the Overview page carries its own *Quick Start* block
> with out-of-date paths from an earlier layout of this project. Follow
> [Getting started](#getting-started-no-technical-background-needed) above
> instead. It is left alone here because this repository's documentation and its
> source are changed in separate passes.

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
  then spread flat across working days. Reconstructed **vintage-correctly**:
  each row is scored against the version of the baseline that was actually in
  force at its own origin, so the comparison is literally "against what the
  Treasury had in hand". It is `NaN` for the stock target, with the reason
  recorded.

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

In a published issue this is all in `gates.json`, keyed by recipe id. Three
fields repay attention: **`passed` is tri-state** (`true` / `false` / `null`,
where `null` means *not measured*, never *passed*, and unmeasured gates are
listed again under `unmeasured_gates`); **`reason_plain`** is the sentence the
UI shows, written to survive a reader with no notes; and **`decided_by`** names
the single gate that set the verdict, while `reasons` lists every failure.

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

### What the measured table does and does not cover

1. **Revenues is the strong case; Expenditure is marginal.** +2.79% over the
   current method is better, but by a margin a different window could erase —
   and Expenditure's publication verdict is `withheld` on accuracy regardless.
   **The stock target has no comparison to the current method at all**: that
   method aggregates a flow to an annual total and a balance level has none, so
   it is recorded as `n/a` rather than invented. Its verdict is `withheld` too.
2. **n=146 rather than 156.** Four rows are dropped by the training embargo
   (item 3); the rest had incomplete features at the window edge.
3. **The training embargo is incomplete.** Five training rows carry targets
   inside the evaluation block, so the model is fitted on answers from the block
   it is then scored on. It is **not** a holdout breach. Removing them moves DEV
   error by −2.98% on Revenues (better) and **+0.65% on Expenditure (worse)** —
   model variance from a 0.2% change in training data rather than a bias being
   removed, which is exactly why it is not being waved through as an obvious
   correctness win. Pinned by an inverted test that fails once it is fixed.
   *(Top priority; `docs/sessions/2026-08-18-dev-fold-holdout-leak.md`)*

### What cannot currently be reproduced

4. **The champion credentials are not reproducible.** The script that produced
   them is absent from the repository. Reconstruction differs by **5.13%**
   (Revenues), **8.12%** (Expenditure) and **19.45%** (stock target). So the
   measured table above is a measurement of the champion *recipe*, not a
   continuation of its logged credentials; the Lab reports that gap beside every
   sealed-window figure so the two can never be read as continuous. Either
   reconstruct the original harness or formally supersede the credentials.
   *(`docs/sessions/2026-08-18-artifact-regeneration.md`)*
5. **The signal sentinel cannot be recomputed**, for the same missing-harness
   reason — reconstruction returns exactly 1.0000 against a logged 1.2255. So
   the effect of the 2026-08-18 fold fix on the `signal` gate is **unmeasured,
   not zero**. The verdicts stand on `accuracy_vs_naive`, which is the gate that
   actually decided them.
6. **Expenditure's two harnesses disagree by 8.6%.** Its reconstructed MASE is
   0.9961 — *under* the 1.0 threshold — against a logged 1.1039. Which side of
   the gate it falls on depends on which harness is asked. The registry was not
   touched; its verdict remains `withheld`.

### Everything else

7. **Nothing has been scored yet**, and the schema's first contact with reality
   is still ahead of it. The fixtures behind it are synthetic by necessity.
8. **`b_ml_pipeline` keeps a divergent Ops method** — a 12-year shift rather
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

Generated and untracked — absent from a fresh clone, created as you use the
Lab: `backend/data/`, `backend/forecast_runs/`, `forecasts/published/`,
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
