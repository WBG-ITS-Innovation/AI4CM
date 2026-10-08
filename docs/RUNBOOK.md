# Weekly runbook

This page is for the Treasury officer who runs the weekly forecast in the app. You do not need
to be a developer to follow it. It covers one file, two steps, what the numbers mean, and what
not to do.

This runbook is in English. A Georgian version, written and reviewed by a Georgian speaker, does
not exist yet.

Sources are named in brackets after each figure. A file path in brackets means the figure is
written in that file in this repository.

## The one file

Each week you upload one file. It is the complete master data file: all history from the first
day through the latest day the Treasury has reported, with every column, as an Excel workbook
(.xlsx) or a CSV file.

* The file has 44 columns: the date, 41 Treasury lines and two calendar markers for weekends
  and holidays [the header of the data file in use; README.md, section "Data", counts the 41
  lines].
* In an Excel workbook only the first sheet is read, and its first row must hold the column
  names [backend/ingest_actuals.py, function `land`]. The older .xls format is not accepted.
  Save it as .xlsx or as CSV.
* Whether the export from the Treasury system already has this form is UNKNOWN. The data file
  in use was produced from a Treasury workbook by the Data Preprocessing page of this app
  [docs/DATA_SEMANTICS.md, line "Source"]. If your export does not have the 44 columns, it will
  be refused, and the person who maintains the app needs to say whether it must go through
  Data Preprocessing first.

**Never upload a fragment**, such as a file holding only the new days. Every day the system
already holds must be in the file you upload, or the check refuses it and nothing is installed
[backend/ingest_actuals.py, function `validate`]. Installing replaces the whole history, so the
file must be the complete export plus the new days.

One upload does three jobs.

1. **It grades last week's forecasts.** Every published forecast day that the file now covers
   is scored against the real figure when you confirm the upload.
2. **It is the history the next forecast learns from.** The next official run fits the model
   again on the whole file, the new days included.
3. **Its last date is where the next forecast starts.** An official forecast covers the 5
   business days after the last date in the file [frontend/pages/07_Forecast.py, "Horizon is
   fixed at 5 business days"; backend/published_forecasts.py, the forecast origin is the last
   date in the data].

## Weekly step 1: the Scorecard page

1. Open the **Scorecard** page and go to the section **Upload actuals**.
2. Choose your file under **Updated data file (CSV or Excel)**. Nothing is written yet.
3. Read the six figures under **What this file would change**: Rows now, Rows in the upload,
   Last day now, Last day after, New days added, Existing values revised
   [frontend/pages/08_Scorecard.py].
4. Read the result below them.
   * Green, **This file passed every check.** Check that New days added matches the days
     reported since your last upload. Then tick the box that begins **I understand this
     replaces the data**, and press **Install this file and score the forecasts**.
   * Red, **This file was not installed, and nothing on disk has changed.** Each reason is
     listed below it. Fix the file and upload it again. The common causes are listed under
     "When the check refuses the file".
5. After you confirm, the page says **Installed.** and then **Scored.**, with how many
   forecast days were scored and how many are still waiting. Reload the page to see the new
   grades.

### Scored and pending

* **Scored** means the file now holds the real figure for a day that a published forecast
  covered. The gap between forecast and real figure is measured and added to the record.
* **Pending** means the file does not reach that day yet. Nothing is guessed. The day stays in
  the list, with its forecast, until the real figure arrives in a later upload.

### Reading the grades

For each Treasury line the page shows three figures [frontend/pages/08_Scorecard.py].

* **Typical error**: the average size of the gap between the central estimate and the real
  figure, ignoring direction.
* **Actual fell inside the range**: the share of scored days on which the real figure fell
  between Low and High.
* **Better than the naive rule by**: how much smaller the model's error is than the error of
  assuming the figure from 5 working days earlier repeats [same file, help text].

Under them is a verdict. **Holding up** means the model still beats the naive rule and its
range is covering about as many days as it claims. **Degrading** means one of those has stopped
being true. With fewer than 5 scored days the page says it is too early to call [same file,
function `_health_verdict`]. If you see Degrading, tell the person responsible for the model.
Nothing in the app changes the model in response.

### When the check refuses the file

The words you will see are written in [backend/ingest_actuals.py].

* **A missing column.** The message says the file "is missing" a number of columns and names
  them. A column was renamed, removed, or left out of the export. Export again with every
  column and do not rename any.
* **Days already held are missing.** The message says the file "holds" a number of rows from a
  start date, "but the data already held has" more, and names the first missing day. The file
  is a fragment, or a stretch of history was left out of the export. Upload the complete export.
* **A repeated date.** The message says the file "has" a number of "repeated date(s)". The
  same day appears twice, often because two exports were pasted together. Keep one row for
  each date.
* **A last date that is not newer than the data held.** The message says the file "ends on" a
  date "and the data already held ends on" the same or a later date. The file is old, or this
  week's file is already installed. Check the export date. If it is already installed, there
  is nothing to do.
* **The same file again.** The message says it is "byte for byte the same file". It is already
  installed.
* **The first sheet of a workbook is not the data.** The message begins "Only the first sheet
  of" and names every sheet and the columns found on the first one. Move the data sheet to the
  front and save the workbook again, or save the data as CSV.
* **Excel cannot be read on this computer.** The message begins "This installation cannot read
  Excel files yet". Upload the CSV instead and tell the person who maintains the app.

Two messages are warnings, not refusals. You may still confirm, but read them first.

* **Values on dates already held have changed.** The file revises past figures. See "Do not"
  below.
* **The file adds columns.** They are stored, and no forecast uses them.

## Weekly step 2: the Forecast page, Official mode

Do this after step 1, so the forecast starts from this week's data.

1. Open the **Forecast** page and go to **Generate a forecast**.
2. Set **Mode** to **Official**.
3. Choose the Treasury line or lines under **Target(s)**.
4. Decide on **Publish** (below), then press **Run the champion recipe**.

The forecast covers the 5 business days after the last date in the file [frontend/pages/07_Forecast.py].
For each day it gives three numbers, in the units the page shows.

* **Central (P50)** is the central estimate: the system judges the real figure as likely to
  land above it as below it.
* **Low (P10)** is the lower edge of a range the system judges 8 in 10 outcomes should fall
  inside.
* **High (P90)** is the upper edge of that same range.

The 8 in 10 is the coverage the range is built for [backend/published_forecasts.py, column
`interval_nominal`, recorded as 0.8]. Whether the range truly catches 8 in 10 has not yet been
measured [registry/recipes.json records the coverage check as not tested for every line].

### When to tick Publish

Tick **Publish to forecasts/published/ under a new issue date** when the file installed in step 1
is this week's file and this is the forecast the Treasury will use. A run without Publish shows
the numbers and adds nothing to the published record.

The app refuses to publish a line whose current verdict is withheld, and says why
[backend/forecast_modes.py, function `refuse_withheld`]. On 7 October 2026 the verdict is
publishable for Revenues and withheld for Expenditure and State budget balance
[registry/recipes.json].

**A published forecast is permanent.** It is written once, into a folder named for the day it
was issued, and the app has no way to edit or delete it. Publishing again on the same day adds
a second issue beside the first, under the same date with a suffix [backend/forecast_modes.py,
function `next_issue_date`]. Both stay in the record and both are scored.

## Where the numbers' credentials come from

* **The champion was chosen once, on data before 2025, and checked.** For each line, the model,
  its inputs and its settings were chosen on data up to 31 December 2024 and recorded in
  registry/recipes.json [backend/registry.py, function `champion_policy`; backend/evaluation_windows.py,
  the held back test period starts on 1 January 2025]. Each champion carries six recorded checks
  [registry/recipes.json]. One of the six, interval coverage, has not been measured.
* **Every official run fits the champion again, from the start, on the whole file.** The newest
  days are included. There is no button for this; it happens on every run
  [docs/REFRESH_AND_RETRAIN.md].
* **The weekly refit never changes which model is champion.** New data moves the numbers inside
  the chosen model. Nothing in the app writes registry/recipes.json [docs/REFRESH_AND_RETRAIN.md].
* **Choosing a different champion is a separate, deliberate event.** It means running the
  tournament again on training and development data, recording it, and editing
  registry/recipes.json by hand. It is never a side effect of uploading a file
  [docs/REFRESH_AND_RETRAIN.md, the section on what choosing again would take].

## Do not

* **Do not edit history in the file without noting it.** The check counts changed past values
  and reports them. It does not refuse them, and every forecast already scored against an old
  value is scored again against the new one [backend/ingest_actuals.py]. If a past figure was
  corrected, write down the date, the line and the reason before you upload, and keep that note
  with your weekly records.
* **Do not use Exploratory output for decisions.** Exploratory runs are not checked, not
  published and never scored [frontend/pages/07_Forecast.py]. They exist for comparison only.
* **Do not expect a published forecast to be editable.** If a published forecast rested on
  wrong data, correct the file, upload it, and publish a new issue. The earlier issue stays as
  it was, and the record shows both.

## Operator appendix

This part is for the person who maintains the app.

**Starting the app.** One command, from the repository folder [README.md, "Starting the app"]:

```
./frontend/.venv/bin/python -m streamlit run frontend/Overview.py
```

On Windows the interpreter is `frontend\.venv\Scripts\python.exe`.

**Two environments.** `frontend/.venv` runs the app. `backend/.venv` runs the models, and the app
calls it for every forecast. Its path is set on the Overview page [README.md, "Why there are two
virtual environments"]. Excel uploads need the openpyxl package in `frontend/.venv`. After
updating the app, install the frontend requirements again:

```
./frontend/.venv/bin/python -m pip install -r frontend/requirements.txt
```

Without it, an Excel upload is refused with a message asking for the CSV.

**Where the files live.**

* The data file in use: `backend/data/processed/master_daily_clean_treasury.csv`. It is always
  a CSV, also after an Excel upload.
* Its earlier versions: `backend/data/processed/backups/`. Every upload keeps the file it
  replaced there, named with the UTC time of the upload. Nothing deletes them automatically. To
  undo an upload, copy the newest backup over the data file. There is no restore button.
* Uploaded files: `frontend/runs_uploads/actuals/`. An Excel upload also leaves there the CSV
  made from it, named after the workbook with `.csv` added.
* Published forecasts: `forecasts/published/`, one folder per issue date. The grades:
  `forecasts/scorecard.csv`.

**Step 1 from the command line**, if the app is unavailable. Without `--install` it only checks:

```
./backend/.venv/bin/python backend/ingest_actuals.py --file <your file> --install --score
```
