# AI4CM Forecast Lab

## What this is

AI4CM forecasts daily cash lines for the Georgian State Treasury, five working days
ahead. Every forecast it publishes is written down before the day it describes, and is
later scored against the figure the Treasury actually reports. Where a forecast fails one
of its checks, the Lab says so and withholds it rather than showing it anyway.

## What it is not

**Nothing here publishes to the Treasury.** Publishing writes a dated folder to disk in
this repository and nothing else. No figure leaves this machine, no system is notified,
and no report is filed. A person decides what to do with the numbers.

It is also not a finished product. It is a research and evaluation workbench. Three of the
41 lines in the data are modelled; the rest have not been. Two of those three are withheld
today, because they failed a check.

---

## The words used here

The app uses these words in exactly this sense, and so does this document.

**Champion.** The one model an official forecast for a given Treasury line uses. It was chosen once,
on recorded evidence from data it had never been fitted on, and loading new data refits
it without ever re-choosing it.

**Exploratory.** A run you launched yourself to see what a model would do. It is measured on the training
and development data only, it is never published, it is never written to the official
forecast, and it never enters the scorecard. Every page that produces one says so while
it is showing it.

**Train and dev.** The two earliest stretches of the history. Train is what a model learns from. Dev is the
next stretch, used to compare models and pick between them. Everything you launch from
the Lab is measured on these two and nothing later.

**Sealed window.** The most recent stretch of history the models never saw while being chosen. We keep it
untouched so the final score is honest. It can only be spent once, so no experiment may
be measured on it, and any that tries is refused.

**Horizon h.** How many working days ahead a forecast reaches. At h=1 it predicts the next working day;
at h=5 it predicts five working days out. The further ahead it reaches, the wider its
honest range has to be. Official forecasts use h=5 and only h=5, because that is the one
horizon everything here was measured at.

**Baseline.** A deliberately simple rule that every model is measured against, such as assuming the
value from five working days ago simply repeats. Beating it is the floor, not an
achievement.

**Skill.** How much smaller a model's typical error is than the baseline's, as a percentage. The
baseline holds the last known figure flat, so at this project's horizon of five working
days it is the figure from five working days earlier. 40% means the model's errors are
40% smaller than that.

**Gate.** A check a forecast must pass before it may be published, such as being more accurate
than the simple rule of thumb. Every gate has a plain-language reason attached to its
verdict, and none can be switched off from this interface.

**MASE.** A model's error divided by the error of repeating the same weekday from the previous
week. Below 1.00 means better than that simple rule and above 1.00 means worse, so 1.00
is the break-even point rather than a threshold anybody chose.

**Holdout.** A block of history deliberately kept away from the models while they were being chosen,
so that measuring them on it says something about days they had never seen.

**Withheld.** Withheld means we do not offer the numbers as a forecast. It happens for one of two
reasons. Either a simple rule of thumb was more accurate, so the numbers should not be
used at all. Or the model could not show that it anticipates individual days, so the
numbers are a guide to the typical level and nothing more. The page always says which of
the two applies.

**P10.** The low end of the published range. The actual figure should fall below it about one day in ten.

**P50.** The central estimate. The actual figure should fall above it about as often as below it.

**P90.** The high end of the published range. The actual figure should fall above it about one day in ten.

**Run folder.** The folder one run writes everything into, named for what it ran and when. It holds the
predictions, the metrics, the leaderboard, the plots, and a record of the exact
configuration it ran with. Nothing is overwritten: a second run gets its own folder.

**Pending.** A published forecast whose day has not been reported yet, so there is no actual figure
to score it against. It is listed rather than hidden.

---

## Setting it up from nothing

### Before you start

You need **Python 3.11, 3.12 or 3.13**, and **git**. Check what you have:

```
python3 --version
git --version
```

On Windows type `python --version` instead. If Python is missing, install it from
[python.org/downloads](https://www.python.org/downloads/). On Windows, tick the box that
says "Add Python to PATH" on the first installer screen; it is easy to miss and everything
below fails without it.

You need about 3 GB of disk space and 15 minutes, most of it waiting for downloads.

### Why there are two virtual environments

This is the single thing most likely to confuse you, so it comes before the commands.

A virtual environment is a private folder of Python packages, so one project cannot break
another. This repository uses **two** of them, on purpose:

| Environment | What lives in it | What it runs |
|---|---|---|
| `backend/.venv` | the modelling stack: scikit-learn, LightGBM, XGBoost, CatBoost, statsmodels, PyTorch | the forecasting pipelines, and the tests |
| `frontend/.venv` | Streamlit, pandas, numpy, plotly | the web app, and only the web app |

They are separate because the app is a thin interface. It reads files the backend wrote
and launches the backend as a subprocess; it never imports a model. Keeping the modelling
stack out of it means the app starts in a second and cannot be broken by a library upgrade
made for a model.

**The consequence you will meet.** Streamlit is installed in `frontend/.venv` and nowhere
else. The modelling libraries are installed in `backend/.venv` and nowhere else. Neither
environment can run the other's job:

```
./backend/.venv/bin/python -m streamlit run frontend/Overview.py
```

```
No module named streamlit
```

That error means you used the wrong environment, not that anything is broken.

### The steps

Everything below is run from the repository root, and every command names the interpreter
it wants explicitly. That is deliberate: it means you never have to remember which
environment is active, and none of these commands needs you to activate anything.

**1. Get the code.**

```
git clone https://github.com/WBG-ITS-Innovation/AI4CM.git
cd AI4CM
```

**2. Build the backend environment.** This is the long one, a few minutes.

```
python3 -m venv backend/.venv
./backend/.venv/bin/python -m pip install --upgrade pip
./backend/.venv/bin/python -m pip install -r backend/requirements.txt
./backend/.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

The last line installs the processor-only build of PyTorch. It is much smaller than the
default, and nothing here needs a graphics card.

**3. Build the frontend environment.** This one is quick.

```
python3 -m venv frontend/.venv
./frontend/.venv/bin/python -m pip install --upgrade pip
./frontend/.venv/bin/python -m pip install -r frontend/requirements.txt
```

**4. Tell the app where the backend is.**

```
./backend/.venv/bin/python -c "import json, pathlib; pathlib.Path('frontend/.tg_paths.json').write_text(json.dumps({'backend_python': str(pathlib.Path('backend/.venv/bin/python').absolute()), 'backend_dir': str(pathlib.Path('backend').absolute())}, indent=2))"
```

You can also do this in the app, on the Overview page, if you would rather see it.

`absolute()` and not `resolve()`, which matters. Inside a virtual environment
`bin/python` is a symlink to the interpreter it was built from, and `resolve()` follows
it, which would record the system Python instead of the environment that has the packages.

On Windows the interpreter is at `backend\.venv\Scripts\python.exe` rather than
`backend/.venv/bin/python`, in this and every command below.

There is a script that does steps 2 to 4 in one go, `scripts/setup_unix.sh` or
`scripts/setup_windows.bat`. It is worth doing them by hand once, so that when something
fails you know which step failed.

### The optional extra: foundation models

Two pretrained forecasters, Chronos and TimesFM, can be added. They are **optional** and
deliberately outside the core install, because they pull roughly 1 GB of model weights and
a fresh clone should not wait for that.

```
./backend/.venv/bin/python -m pip install -r backend/requirements-foundation.txt
```

Without them, the Lab's Foundation family says they are not installed and everything else
works normally. Nothing in the core setup needs them.

---

## Starting the app

```
./frontend/.venv/bin/python -m streamlit run frontend/Overview.py
```

You should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

A browser tab opens on the Overview page. In the sidebar you should see the Treasury seal
and the wordmark at the top, then ten pages, then a language selector offering English and
ქართული.

If the Overview page reports that it cannot find the backend interpreter, step 4 above did
not take. Fix the paths on that page and press Save.

---

## Running the tests

Two commands, one per environment.

```
./backend/.venv/bin/python -m pytest -q
```

**Expect 1329 passed, 19 skipped.** Takes about 10 minutes.

```
./frontend/.venv/bin/python -m pytest frontend/tests -q
```

**Expect 707 passed, 17 skipped.** Takes about 30 seconds.

The first command collects **both** suites, because `pytest.ini` lists
`backend/tests` and `frontend/tests` as its test paths. Under the backend interpreter the
frontend tests that need Streamlit skip themselves cleanly, which is most of the 19
skips.
The second command runs the frontend suite under the interpreter that has Streamlit, where
they all execute. So the two numbers are not comparable and neither is a subset of the
other.

**The backend suite needs the Treasury data file** at
`backend/data/processed/master_daily_clean_treasury.csv`. That file is not in the
repository, and without it 9 tests fail, every one of them naming the missing file. See
"Data" below.

---

## The pages, in the order the sidebar shows them

| Page | What it is for |
|---|---|
| **Overview** | The landing page. Confirms the app can find the backend, lists recent runs, states what the project does and does not claim. |
| **Start here** | The guide. What every page is for, what you can do there, one thing to try, and the difference between an official forecast and an experiment. |
| **Data Preprocessing** | Turn a raw Treasury export into the clean daily series the models read, and see what the cleaning changed. |
| **Lab** | The workbench. Try any model on any Treasury line at any horizon. Nothing here is published. |
| **Dashboard** | The detail behind one experimental run: predictions over actuals, where the error fell, the integrity checks. |
| **Compare** | Two to six experimental runs on the same axes. |
| **History** | Every run this Lab has produced, including the ones that failed their checks, with their files. |
| **Forecast** | The forecast itself, in two tabs: one for producing a forecast, one for reading and evidencing the published ones. |
| **Scorecard** | Published forecasts against what actually happened, and the one place new actual figures are uploaded. |
| **Documentation** | The reference: every model, its settings, the promoted recipes and their evidence, and how to add a model. |

---

## Data

### What the app expects

One CSV, at `backend/data/processed/master_daily_clean_treasury.csv`. One row per
business day, a `date` column, and one numeric column per Treasury line. The file in use
holds 3,867 rows covering 2015-01-05 to 2025-08-06, across 41 lines.

**It is not in the repository, and that is deliberate.** It holds real Treasury figures.
`.gitignore` ignores any directory named `data`, so it cannot be committed by accident.
You supply your own copy.

The Lab does not need it. The Lab reads whatever file you upload to it, so you can run
experiments before you have the canonical file in place.

### Where things are written

| What | Where | In git? |
|---|---|---|
| Files you upload to the Lab | `frontend/runs_uploads/` | no |
| One folder per Lab run | `frontend/runs/<run_id>/` | no |
| Backend pipeline runs | `backend/forecast_runs/<date>/` | no, except the summaries |
| Published forecasts, one folder per issue date | `forecasts/published/<issue_date>/` | no |
| The scorecard | `forecasts/scorecard.csv` | yes, and it is header-only until real figures arrive |
| The audit trail of every logged run | `experiments/log.csv` | yes |

A run folder holds `predictions_long.csv`, `metrics_long.csv`, `leaderboard.csv`,
`plots/`, and `artifacts/` with the configuration and the integrity report. There is no
`predictions.csv` and no `metrics.csv`; the long form is what gets written.

### Adding newly reported figures

Upload them on the Scorecard page. The file is checked before anything is written, you see
what would change, and the file being replaced is kept with a timestamp. What happens to
the model afterwards is in [docs/REFRESH_AND_RETRAIN.md](docs/REFRESH_AND_RETRAIN.md), and
the short version is: it is refitted automatically on the next official forecast, and
which model is champion never changes on its own.

---

## What has actually been measured

Worth being precise about, because the count of models and the count of measurements are
very different numbers:

> 21 machine-learning models, 5 deep-learning models and 7 statistical models compete on
> each target; prediction intervals come from 6 quantile methods; 3 further entries are
> reference baselines, not competitors. Of those, 8 have a recorded result on at least one
> target and 33 are registered candidates with no recorded result yet.

That sentence is generated from the registry, not typed. The number to read is the second
one: 44 models are available and 8 have been measured.

Of the three modelled Treasury lines, one is publishable today and two are withheld.

---

## If something goes wrong

**`No module named streamlit`**

You used the backend environment to run the app. Streamlit lives only in
`frontend/.venv`. Use `./frontend/.venv/bin/python -m streamlit run frontend/Overview.py`.

**`No module named sklearn`, or `lightgbm`, or `torch`**

The reverse: you used the frontend environment for something that needs the modelling
stack. Use `./backend/.venv/bin/python`.

**This is the same mistake twice, and it is the commonest one here.** It is also why every
command in this document names its interpreter explicitly instead of assuming an activated
environment. If you do activate one, the shell prompt does not tell you which of the two it
is, and `python` then silently means something different from what you intended.

**The app says it cannot find the backend interpreter**

`frontend/.tg_paths.json` is missing or points somewhere that no longer exists. Redo step 4
of the setup, or set the two paths on the Overview page and press Save.

**9 backend tests fail, all mentioning `master_daily_clean_treasury.csv`**

The Treasury data file is not in place. See "Data" above. The rest of the suite passes
without it.

**The Lab's Foundation family says no model is installed**

The optional extras are not installed. Either install them, or use another family. Nothing
else is affected.

**A foundation model install fails on `lag-llama`**

It is not offered, deliberately. It pins an old version of gluonts, which forces pandas to
be built from source, and that build does not compile on Python 3.13. Chronos and TimesFM
are the two that work.

**The first foundation forecast is slow, later ones are not**

Expected. Model weights download once and are then cached in `~/.cache/huggingface/hub`.
Each checkpoint is pinned to an exact revision, so a later run uses the same weights and
downloads nothing.

**An SSL or certificate error while installing**

Usually a corporate network intercepting HTTPS. Either run the install off that network, or
ask whoever administers it for the certificate bundle and point pip at it with
`--cert`. Do not disable certificate checking to get past it.

**A page shows an error**

Every page is written to render rather than crash, so an error box with a reason in it is
usually the page telling you something is missing rather than the app failing. The message
says what it looked for. If a page raises instead, that is a bug worth reporting.

---

## Repository layout

```
backend/          the pipelines, the models, the gates, the publication logic
  data/           the Treasury data. Never committed
  requirements.txt              the core install
  requirements-foundation.txt   the optional pretrained models
frontend/         the Streamlit app
  Overview.py     the entry point
  pages/          the nine other pages, numbered in nav order
                  (frontend/pages, one file per sidebar entry)
  tests/          the frontend suite
docs/             how to add a model, how a data refresh works, data semantics
forecasts/        published issues, and the scorecard
registry/         registry/recipes.json: which model is champion for each line, and why
experiments/      the audit trail. Every measured figure traces to a row here
reports/          written records and the demo runbook
scripts/          setup and run helpers, and one-off analyses
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) | Putting a new model on the shelf, and how to check it is reachable |
| [docs/REFRESH_AND_RETRAIN.md](docs/REFRESH_AND_RETRAIN.md) | What happens to the model when new figures arrive, and what does not |
| [docs/DATA_SEMANTICS.md](docs/DATA_SEMANTICS.md) | What each column means, and the traps in the source data |
| [docs/AGENT_ARTIFACT_CONTRACT.md](docs/AGENT_ARTIFACT_CONTRACT.md) | The artifacts another program may rely on |
| [docs/SIGNAL_FINDING.md](docs/SIGNAL_FINDING.md) | The signal check, what it measures, and how its threshold was calibrated |
| [docs/sessions/](docs/sessions/) | A record of every working session, with the evidence for each claim |
