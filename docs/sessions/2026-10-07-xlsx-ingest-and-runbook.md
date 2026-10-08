# Excel ingest, the weekly runbook, and the parked work filed as issues

**Date:** 2026-10-07
**Branch:** `feat/xlsx-ingest-and-runbook`, cut from `main` at `4b34dac`.
**Scope:** code, tests and documentation in the working tree. Everything that published,
ingested or scored ran in a fresh disposable clone under `/tmp`, on synthetic data, by the
method of `2026-09-30-scoring-loop-audit.md` Part 2.

---

## Working tree safety, stated first

* No pipeline, scorer, ingest or artifact regeneration was started in the working tree by
  this session's commands.
* The real canonical CSV was compared before and after the session and is unchanged. The
  comparison value is deliberately not recorded here.
* What was read from real files, all structure and no values: the canonical file's header
  (column names), its dtype counts and date text pattern, and the sheet count and first sheet
  header of the captured Treasury workbook in `frontend/runs_uploads/`. No value was printed.
* The two test suites append to `experiments/test_access.log` by design, as recorded in
  `2026-09-29-lab-audit.md`. Counts are under "Verification".
* The existing Scorecard page tests rewrite the tracked `forecasts/scorecard.csv` on every
  render (finding F3). The rewrite is idempotent and git shows the file unchanged. The new
  tests in this session stub the scorer and do not.
* One environment change: `openpyxl==3.1.5` was installed into `frontend/.venv`.

---

## Part 1. Excel ingest

### 1.1 The recommended shape, confirmed against the code

The brief recommended converting an `.xlsx` to CSV at the landing step and running the existing
`validate` and `install` unchanged. The code supports that shape and argues for it.
`install` ends in `shutil.copy2(candidate, canon)`, and its comment states why: the SHA in every
downstream provenance record must be the SHA of a file that exists. Teaching `validate` and
`install` to read a workbook would let the canonical file become a workbook, or make `install`
write something other than what it checked. So the conversion sits in front of both, and the Excel work
changed neither. (`validate` later gained the fragment check, approved separately on 2026-10-08.)

The brief's line references have moved. The upload block is now `frontend/pages/08_Scorecard.py:545-553`
and the install call `:593` (after this change: `:545-562` and `:598`). The CLI is
`backend/ingest_actuals.py` `_cli`, and the module line numbers shift by the new function.

### 1.2 What `land` does

`land(upload, canonical=None) -> (candidate, refusal)` in `backend/ingest_actuals.py`.

* **A CSV** comes back as the same path and nothing is written. It is validated and byte copied
  exactly as before.
* **An `.xlsx`** has its first worksheet read and written beside it as `<name>.xlsx.csv`. The
  full workbook name is kept, so a CSV the officer keeps under the workbook's stem, such as
  `master.csv` beside `master.xlsx`, is never overwritten. The CLI writes next to the officer's
  file, which is why this mattered.
* **A refusal** is an `IngestCheck`, the type `validate` returns, so both callers render a
  landing refusal and a validation refusal the same way.

Both callers changed by a few lines. The page: `candidate, refused = land(candidate)` then
`check = refused if refused is not None else validate(candidate)`; the install call is unchanged
and receives the landed CSV because `candidate` was rebound. The CLI: the same two lines, and
`install(candidate)`.

### 1.3 Number types: the one subtle part

Excel stores every number the same way, and pandas returns a column whose values are all whole
numbers as `int64`. The CSV export writes those amounts with a decimal point, so they read back
as `float64`. Measured in both environments with a scratch probe: a whole number float column
read back as `int64` from a workbook and `float64` from CSV.

The canonical file's structure decided the rule. It holds 41 float columns, every one written
with a decimal point, and two genuine integer flags, `is_weekend` and `is_holiday`. Making every
workbook number a float would break the flags. Trusting pandas would break any amount column
that happens to hold only whole numbers. The rule applied: **the type comes from the file being
replaced.** Where the canonical holds a column as float and the workbook gave integers, the
column is restored to float. No value, row or column order changes.

`validate` would not have caught the difference. With the rule removed, a mistyped workbook
still passed with zero revisions, because the revision check compares numbers within a
tolerance. Only the new type test failed.

### 1.4 Refusals, worded for an officer

The wrong first sheet message names the sheets in order and the columns found. Verbatim, from the
clone run below:

> Only the first sheet of notes_first.xlsx is read, and that sheet, 'Notes', is missing 44 of
> the 44 columns the current data has: date, Revenues, Income, Taxes, Income tax, Profit tax and
> 38 others. The columns found on 'Notes' are: Note. The sheets in this workbook, in order, are:
> 'Notes', 'Master export'. If the data is on another sheet, move that sheet to the front and save
> the workbook again, or upload the CSV export instead.

Two further refusals: a file that is not a workbook ("could not be opened as an Excel workbook",
with the password hint), and an environment without openpyxl ("This installation cannot read
Excel files yet ... Upload the CSV export instead"). The second exists because a client machine
that pulls this change without reinstalling `frontend/requirements.txt` would otherwise crash the
page on the first Excel upload.

Before this change, an `.xlsx` given to the CLI produced: "master.xlsx could not be read as a CSV
file. The reader reported: 'utf-8' codec can't decode byte ...". The page did not offer `.xlsx`.

### 1.5 openpyxl, per environment

| Environment | Needed by | Before | Action |
| --- | --- | --- | --- |
| `backend/.venv` | the CLI | openpyxl 3.1.5 installed; declared `openpyxl>=3.1` in `backend/requirements.txt` | none |
| `frontend/.venv` | the page, which imports `backend/ingest_actuals` | absent | pinned `openpyxl==3.1.5` in `frontend/requirements.txt`, matching the backend version and that file's `==` style; installed; it brought `et_xmlfile 2.0.0`, the version the backend has |

### 1.6 Tests, failing first

**Backend, `backend/tests/test_ingest_workbook.py` (12).** Written before any code.

| Run | Result |
| --- | --- |
| Against `main` | collection error: `cannot import name 'land'` |
| Against a temporary `land` that returns the upload unchanged, which is what the system did | 9 failed, 3 passed. The 3 are guards on behaviour that must not change (CSV byte copy, no overwrite beside the workbook, a workbook given straight to `install` is refused) |
| Full `land` without the number type rule | 4 failed, every one on `Expenditure` `int64` against `float64` |
| With the rule | 12 passed; with `test_ingest_actuals.py`, 39 passed |

**Frontend, `frontend/tests/test_scorecard_xlsx_upload.py` (6).** AppTest in Streamlit 1.40.1
exposes `file_uploader` as an unknown element: its type list and help are readable, a file cannot
be set. The tests replace `st.file_uploader` with a wrapper that renders the real widget and then
returns a synthetic upload; everything after that call is the page's own code, through the
confirmation checkbox and the Install button. Redirected: `ingest_actuals.CANONICAL` and
`BACKUP_DIR` to a synthetic stand in, and `published_forecasts.score_published` to an empty result.
The landing directory is a page constant, so each test lands a uniquely named synthetic file and
removes exactly what it created.

| Run | Result |
| --- | --- |
| Against `main`, `frontend/.venv` without openpyxl | 5 failed, 1 passed (the CSV regression). The workbook fixtures could not be built: `No module named 'openpyxl'` |
| openpyxl pinned and installed, page unchanged | 4 failed, 2 passed. Uploader accepts only `.csv`; an `.xlsx` is read as CSV and refused with the decode error |
| Page changed | 6 passed |

### 1.7 Clone evidence: the CLI path end to end

**Setup.** `git clone` of the working tree into `<clone>` under `/tmp`, at `4b34dac`, with the five
changed files copied in and both environments symlinked. The only `.pth` in either environment is
`distutils-precedence.pth`. The clone arrived with no `backend/data/`, no `forecasts/published/`,
no `private_vault/`, no `experiments/log.csv`.

**Synthetic data.** A scratchpad generator, seed 20261007, takes only the 44 column names from the
real header and builds one series, written twice: through 2025-08-06 as the clone's canonical CSV,
and through 2025-08-20 as an `.xlsx` with Excel typed dates, data on the first sheet and a "Notes"
sheet second, plus a CSV twin of the same frame. Checked: 3,867 and 3,881 rows, 44 columns, the
first 3,867 rows of the longer frame equal the shorter, no NaNs, 1,104 and 1,108 weekend rows,
dtype counts 41 float, 2 integer, 1 text. Flows post on Georgian working days (holidays from
`backend/preprocessing/holidays.py`) with annual seasonality, a month end spike and 4.5% growth;
the budget balance is the year resetting cumulative net. One line, `Valuables`, is whole lari
throughout, so the clone exercises the number type rule.

**Publish.** `backend/run_forward_forecast.py --publish`: exit 0, `test_window_touched = False`.
Revenues published to `forecasts/published/2025-08-06`, five target dates 2025-08-07 to 2025-08-13.
Expenditure and State budget balance were refused: "its current verdict is 'withheld'" (F4).

**Before new actuals.** `backend/run_publish_and_score.py --score-only`: 1 issue, scored 0,
pending 5. The clone's holdout ledger did not exist.

**Ingest, as `.xlsx`.**
`backend/ingest_actuals.py --file <inbox>/synthetic_through_2025-08-20.xlsx --install --score`,
exit 0. Fingerprints and paths removed:

```json
{
 "check": {"candidate": "<inbox>/synthetic_through_2025-08-20.xlsx.csv", "ok": true,
  "blockers": [], "warnings": [],
  "summary": {"rows_now": 3867, "rows_new": 3881, "rows_added": 14,
   "last_date_now": "2025-08-06", "last_date_new": "2025-08-20",
   "columns_now": 44, "columns_new": 44, "columns_missing": [], "columns_added": [],
   "revisions": 0}},
 "install": {"installed": true, "rows_before": 3867, "rows_after": 3881, "rows_added": 14,
  "last_date_before": "2025-08-06", "last_date_after": "2025-08-20", "revisions": 0},
 "score": {"scored": 5, "pending": 0}
}
```

**Every pending row scored: 5 of 5, none pending.** Checked afterwards:

| Check | Result |
| --- | --- |
| Installed canonical is a workbook (zip header) | no |
| Parsed equal to the CSV twin, exact, dtype for dtype | yes |
| Byte identical to the CSV twin | yes. Both were written by pandas from the same values; a Treasury CSV export formatted differently would only match parsed |
| `Valuables` dtype after install | `float64` |
| Backups | 1, the previous file: 3,867 rows ending 2025-08-06 |
| Scorecard rows | 5, Revenues, all `scored_in_window = live` |
| `y_true` equals the installed Revenues on each target date | yes |
| `abs_error` recomputes from `y_true` and `p50` | yes |
| Clone holdout ledger | never created |

**Refusal path.** The same workbook with "Notes" moved to the front, `--install`: exit 1, no
`install` key, the message quoted in 1.4, canonical unchanged, still one backup, no landed CSV.

**The new tests in the clone,** where no real data exists: 12 passed and 6 passed.

---

## Part 2. The runbook

`docs/RUNBOOK.md`, for a Treasury officer, in the brief's order, with one line added to the
`README.md` documentation table. Style rules were checked by a script that strips code and looks
for any dash or hyphen in the remaining prose and for a list of marketing adjectives: none found.
Dates in prose are written out ("31 December 2024") because ISO dates contain hyphens.

It was drafted departing from the brief in two places, both because the brief's sentence was not
true of the code: fragments (F1) and the form of the Treasury export (F2). F1 was then fixed, and
the runbook's fragment paragraph now states exactly what the validator enforces ("Fix applied
2026-10-08" below). F2 stays UNKNOWN in the runbook and is an open question. The officer sections do not mention the CLI's
`--overwrite` flag on `run_publish_and_score.py`, which can replace a published issue: true,
but writing it there would invite its use.

---

## Findings

F1 was fixed on 2026-10-08. F2 to F5 are reported and not fixed.

**F1. The validator does not refuse a fragment.** The brief's runbook sentence was "the validator
refuses fragments". `validate` checks columns, dates, duplicates, a later last date, identical
bytes and revisions, and nothing compares the upload's history with the history held. Measured
with synthetic data: a file holding only the two newest weeks passed with 0 blockers and 0
warnings; history starting a month late, and history with a month missing in the middle, each
passed with only a revisions warning. Installing a fragment replaces the whole history; the backup
survives, but every later refit trains on the fragment. The runbook states this and gives the
officer a guard: Rows in the upload must be larger than Rows now. *Proposed fix,* failing first:
a blocker when any date already held is absent from the upload, naming how many and the first.

> **Fixed 2026-10-08**, as proposed, after approval. See "Fix applied 2026-10-08" below.

**F2. Whether the Treasury system's export is the 44 column file is unconfirmed.**
`docs/DATA_SEMANTICS.md` records the canonical file's source as a Treasury workbook, produced by
the `clean_treasury` preprocessing, which reindexes to calendar days, zeroes flows on non business
days, forward fills levels and adds the calendar flags. The captured workbook has several sheets
and its first sheet has no header row the reader can use, so uploaded as it is, `land` refuses it.
The runbook writes this as UNKNOWN and names the Data Preprocessing page. If the weekly export
must pass through that page, the runbook needs a step before step 1.

**F3. The Scorecard page runs the scorer on every render.** `load_scoring` calls
`score_published(DATA)`, which rewrites the tracked `forecasts/scorecard.csv`. Every existing
AppTest render of the page therefore runs the scorer in the working tree. The output is identical
for unchanged inputs, which is why git stays clean. Not changed; recorded because a brief that
says "no scorer run here" is contradicted by the existing frontend suite.

**F4. Correction to `2026-09-30-scoring-loop-audit.md` §2.2.** That record says the runner path
`run_forward_forecast.py --publish` applies no verdict check. In this clone it refused
Expenditure and State budget balance as withheld. The behaviour changed after that record; the
record's sentence is historical.

**F5. `README.md`, section "Data", says one row per business day.** The canonical file holds one
row per calendar day with flags marking weekends and holidays, as `docs/DATA_SEMANTICS.md` states.
Not changed.

---

## Part 3. The parked work, filed as issues

**Status: drafted 2026-10-07, approved as drafted 2026-10-08.** Each body points at its record
section and carries no real value. On 2026-10-07 `gh` was not on this session's shell path, so
nothing was sent then. Filing status is under "Filing" at the end of this part.

**1. GBQuantile quantiles skip _enforce_monotone; crossings uncounted on the official forward path** [#32]
Labels: `parked-until-agent-phase2`. Rank first.
GBQuantile fits each quantile independently in `backend/e_quantile_daily_pipeline.py` and returns
a hardcoded crossed count of 0, so a P10 above the P50, or a P50 above the P90, is neither repaired
by `_enforce_monotone` nor reported. GBQuantile was the interval model the forward run chose for
all three targets in the 2026-09-29 clone, so this sits on the client path, and crossing was
observed in a newer quantile model that does route through the repair. See
`docs/sessions/2026-09-29-lab-audit.md`, "Phase A3", finding 2, and "What the live runs add", point 4.

**2. ops_monthly_baseline shifts 12 years inside a per-month groupby, all NaN, plus the latent twin fallback in c_dl_pipeline** [#33]
Labels: `parked-until-agent-phase2`.
`ops_monthly_baseline` in `backend/b_ml_pipeline.py` applies `shift(12)` inside a groupby by
calendar month, where each group holds one row per year, so it shifts twelve years rather than
twelve months and returns all NaN on a ten year series. `backend/c_dl_pipeline.py` falls back to
the same construct when fewer than three complete years exist; no current series reaches that
branch. See `docs/sessions/2026-09-29-lab-audit.md`, "Phase A3", finding 1, and
`docs/sessions/2026-09-30-scoring-loop-audit.md`, §2.6 F6.

**3. ResidualRF repairs crossings and reports zero** [#34]
Labels: `parked-until-agent-phase2`.
The ResidualRF path in `backend/e_quantile_daily_pipeline.py` makes its quantiles monotone with an
elementwise maximum and then returns a hardcoded crossed count of 0. The module's own docstring
names the cost: constant crossing is how a misconfigured quantile model looks from outside, and a
zero count hides it. See `docs/sessions/2026-09-29-lab-audit.md`, "Phase A3", finding 3.

**4. Sealed-window champion-vs-ops table has no regenerating script and its ops read left no ledger entry; add a logged writer** [#35]
Labels: `parked-until-agent-phase2`.
`reports/sealed_window_champion_vs_ops.csv` cannot be regenerated from code in the repository, and
the sealed window read behind its ops column has no ledger entry. Add a small writer, in
`sealed_window_report` or under `scripts/`, that rebuilds the table from `evaluate_champion` and
the ops baseline and calls `log_sealed_window_read`, then cite it from the 2026-08-18 record. See
`docs/sessions/2026-09-30-scoring-loop-audit.md`, §1.4 and §2.6 F4.

**5. A_STAT leaderboard and predictions_long disagree on the persistence baseline row** [#36]
Labels: `parked-until-agent-phase2`.
A_STAT scores "Persistence (baseline)" in `leaderboard.csv` but writes no rows for it to
`predictions_long.csv`, so a consumer joining the two loses the baseline without notice; the
artifact contract check printed this as a warning in the 2026-09-29 clone run. Either write the baseline's
predictions or drop it from the leaderboard with a stated reason. See
`docs/sessions/2026-09-29-lab-audit.md`, "Phase B", "A_STAT's warning".

**6. Interval coverage has never been measured; obtain the TEST-window figure through the logged report path so the gate stops reading not tested** [#37]
Labels: `parked-until-agent-phase2`.
Every recipe in `registry/recipes.json` records the coverage gate as not tested, so the pages say
"not tested" and the runbook has to say the 8 in 10 range is unmeasured. The 2026-10-01 decision
(Option A) bounded E_QUANTILE selection to DEV and deferred the TEST window coverage figure to the
logged report path, `require_test_access`; that read has not been made. See
`docs/sessions/2026-09-29-lab-audit.md`, "Phase A2" (c) and "OPEN DECISION", and `docs/RUNBOOK.md`.

**7. A_STAT and C_DL evaluate on post-seal rows with no selection guard and no window label; set one evaluation-window policy for all four families** [#38]
Labels: `parked-until-agent-phase2`.
In the daily script, B_ML and E_QUANTILE are bounded to DEV, while A_STAT and C_DL evaluate on rows
after the seal with no selection guard; only some of those reads reach the ledger, and neither
family's artifacts name the window they were computed on. Decide one policy for all four families:
where each may evaluate, which reads are report reads, and how each artifact labels its window.
See `docs/sessions/2026-09-30-inference-horizon-map.md`, §1.4, Flow B.

**8. backend/tests has no conftest** [#39]
Labels: `test-infra`. Reworded 2026-10-08 to stay inside its item (see "Filing"); approved as reworded.
There is no `conftest.py` anywhere in the repository, so 72 of the 79 backend test files (counted
2026-10-07) repeat their own `sys.path` setup, and the two backend ingest test files each define
their own synthetic frame helper. A conftest in `backend/tests` would hold the path setup and a
synthetic stand-in for the canonical file. See `docs/sessions/2026-09-29-lab-audit.md`, "Phase A1".

**9. Adopt ui_styles.gate_badge_tri for gate verdicts or retire it** [#40]
Labels: `design`.
The three state gate is shown two ways. The Forecast page and one Documentation section render
it as translated text through `format_gel.gate_verdict()`; the Dashboard and another Documentation
section use the design system badge `ui_styles.gate_badge_tri()`. Choose one: adopting the badge
everywhere changes the Forecast page's visual form and must keep the translation, and retiring it
means moving the Dashboard to the text form. See `docs/sessions/2026-09-29-lab-audit.md`, "Fixes
applied (2026-09-30)", fix 1.

**10. Georgian translation of docs/RUNBOOK.md, produced and reviewed by a Georgian speaker** [#41]
Labels: `documentation`. The brief named `docs`; the repository's existing `documentation` label
was used instead, by decision of 2026-10-08.
The weekly runbook exists in English only, and it was deliberately not machine translated. A
Georgian version should be written and reviewed by a Georgian speaker, keeping the runbook's style
rules and its UNKNOWN markers. See `docs/sessions/2026-10-07-xlsx-ingest-and-runbook.md`, "Part 2".

**11. AGENT_ARTIFACT_CONTRACT.md section 7 no longer matches what the Lab writes** [#42]
Labels: `documentation`. Added 2026-10-08 by the user, body verbatim, filed last. Its citations
were checked read only before filing: `ec0accf` is the contract's last change (2026-08-19), its
section 7 is `forecasts/published/<issue_date>/`, `backend/forward_forecast.py:334-393` carries
`holdout_ledger` and the measured `test_window_touched`, and `backend/published_forecasts.py:125`
and `:221` hold schema 3 and `recipe_status` beside `publication_verdict`.
The contract was last changed on 2026-08-19 (ec0accf). Since then forward provenance gained a
holdout_ledger block and test_window_touched became a measured value
(backend/forward_forecast.py:334-393), and the scorecard moved to schema 3 with recipe_status
beside publication_verdict, whose meaning changed from registry status to the verdict
reconstructed from the issue's gate flags (backend/published_forecasts.py:125, 221). Update
section 7 to document all three. The agent repo's 2026-10-07 diagnostic found the gap.

### Filing

**The two conditions, checked 2026-10-08 before anything was sent.** Each draft was extracted
from this part of the record exactly as written. Checked mechanically: its title and labels equal
the brief's, character for character; it points at a session record; it holds no hex string of
seven or more characters, no local path and no username; and the only numbers in it are dates,
section numbers, the twelve years in title 2 and a count of test files in body 8. All ten passed.

**One draft held as a deviation, per the brief's rule.** Draft 8's item is `backend/tests`; its
body proposes "a shared conftest per suite" and names "both suites", which extends it to
`frontend/tests`. It is not filed until approved or reworded.

**The repository.** `gh` 2.92.0 was found on this machine outside this session's shell path,
logged in, with admin permission on the repository. Issues are enabled and the repository is
public, so every issue is world readable. None of the four labels exists: the repository has only
GitHub's default set, which includes `documentation` and not `docs`. Creating labels publishes to
the repository and was not part of the approval, so nothing was filed pending that decision.

**Decisions of 2026-10-08.** Create `parked-until-agent-phase2`, `test-infra` and `design`; do
not create `docs`, and label issue 10 with the existing `documentation`. Issue 8 approved as
reworded. Issue 11 added, verbatim, label `documentation`, filed last. All eleven filed in rank
order with `gh` invoked at its absolute path.

**Filed 2026-10-08.** The three labels were created, then all eleven issues were filed in rank
order, and each was read back from GitHub: title equal to the approved one, label as decided. No
`docs` label exists.

| Rank | Issue | Label |
| --- | --- | --- |
| 1 | #32 | `parked-until-agent-phase2` |
| 2 | #33 | `parked-until-agent-phase2` |
| 3 | #34 | `parked-until-agent-phase2` |
| 4 | #35 | `parked-until-agent-phase2` |
| 5 | #36 | `parked-until-agent-phase2` |
| 6 | #37 | `parked-until-agent-phase2` |
| 7 | #38 | `parked-until-agent-phase2` |
| 8 | #39 | `test-infra` |
| 9 | #40 | `design` |
| 10 | #41 | `documentation` |
| 11 | #42 | `documentation` |

---

## Fix applied 2026-10-08: an upload must hold every date already held

Approved at the gate as F1's proposal. `validate` gains one blocker, placed after the check that
the record is extended: every date in the file currently held must be present in the upload. The
revision counter is unchanged and still counts changed values on the dates both files hold.

The refusal states both sides and the instruction. Verbatim, on a synthetic stand in of 65 rows
from 2024-01-01 and an upload of only the next two weeks:

> master.csv holds 10 rows starting on 2024-04-01, but the data already held has 65 rows starting
> on 2024-01-01. 65 of the days already held are not in this file, the first of them 2024-01-01.
> Installing replaces the whole history, so an upload must be the complete export: every day
> already held plus the new days. Upload the complete export rather than a part of it.

**Failing first.** Four tests in `backend/tests/test_ingest_actuals.py`, run against the
validator as it stood: 3 failed, 1 passed. The fragment with every column and a newer last date
came back `ok`, the full file missing one historical date came back `ok`, and no refusal message
existed to inspect. The complete export plus new days passed, with its one revision counted, as it
must. After the change, all four pass, and the three F1 cases replayed above are each refused with
one blocker.

**Copy changed with it.** The module docstring's list of what must hold went from four items to
five. One clause was added to the two officer facing lists of the checks, the Scorecard page's
"What happens when a file is uploaded" and step 1 of `docs/REFRESH_AND_RETRAIN.md`, which the page
renders. The runbook's fragment paragraph now states the rule and its source, its interim guard
("Rows in the upload must be larger than Rows now") was removed, and its list of refusals gained
"Days already held are missing".

---

## Open questions

* **Whether the Treasury's standard weekly export contains exactly the 44 columns the current
  file holds is UNKNOWN. This is a question for the client.** The evidence that it may not is F2.
  Until it is answered, `docs/RUNBOOK.md` says UNKNOWN, and an export without them is refused with
  the missing columns named.

---

## Verification

**Both suites green, twice.** First after the Excel work (2026-10-07), against the session's
starting baselines; then after the fragment fix (2026-10-08), against the first run.

| Suite | Command | Baseline | After Excel work | After fragment fix |
| --- | --- | --- | --- | --- |
| Frontend | `./frontend/.venv/bin/python -m pytest frontend/tests -q` | 734 passed, 16 skipped | 740 passed, 16 skipped (+6) | **740 passed, 16 skipped** (+0) |
| Backend | `./backend/.venv/bin/python -m pytest -q` | 1401 passed, 21 skipped | 1413 passed, 22 skipped (+12, +1 skipped) | **1417 passed, 22 skipped** (+4) |

The backend's extra skip is `frontend/tests/test_scorecard_xlsx_upload.py`, which gates on
Streamlit at module level, like the other page tests, and so is skipped as one module in the
backend environment. No existing test was changed; four new ones were added to an existing file.

**The 22 new tests,** each written before the code that satisfies it.

* `backend/tests/test_ingest_workbook.py`, 12:
  `test_a_workbook_passes_the_same_checks_a_csv_does`,
  `test_installing_a_workbook_gives_the_same_canonical_file_as_the_csv`,
  `test_a_csv_upload_is_still_installed_byte_for_byte`,
  `test_excel_typed_dates_and_numbers_survive_to_the_same_values`,
  `test_only_the_first_sheet_is_read`,
  `test_a_workbook_whose_first_sheet_is_not_the_data_is_refused_naming_what_it_found`,
  `test_a_file_that_is_not_really_a_workbook_is_refused_in_plain_words`,
  `test_a_missing_excel_reader_is_a_refusal_not_a_crash`,
  `test_landing_never_overwrites_a_csv_kept_beside_the_workbook`,
  `test_a_workbook_handed_straight_to_install_is_refused_never_copied`,
  `test_the_cli_installs_a_workbook_as_a_csv`,
  `test_the_cli_refuses_a_wrong_first_sheet_and_writes_nothing`
* `backend/tests/test_ingest_actuals.py`, 4 added to the existing file (2026-10-08):
  `test_a_fragment_with_every_column_and_a_newer_last_date_is_refused`,
  `test_the_complete_export_plus_new_days_still_passes`,
  `test_a_full_file_missing_one_historical_date_is_refused`,
  `test_the_fragment_refusal_states_both_sides`
* `frontend/tests/test_scorecard_xlsx_upload.py`, 6:
  `test_the_frontend_environment_can_read_a_workbook`,
  `test_the_uploader_accepts_a_workbook_and_says_only_the_first_sheet_is_read`,
  `test_a_workbook_upload_is_checked_and_passes`,
  `test_installing_a_workbook_from_the_page_puts_a_csv_in_place`,
  `test_a_csv_upload_on_the_page_is_still_installed_byte_for_byte`,
  `test_a_wrong_first_sheet_is_refused_on_the_page_in_plain_words`

**Working tree after both suites.** On each of the two runs, `git status` was identical before
and after: the suites changed no tracked file, `forecasts/scorecard.csv` included.
`experiments/test_access.log` grew by 13 entries on each run, the same count the 2026-09-29 audit
recorded for a full run. The canonical
CSV is unchanged. The landing directory `frontend/runs_uploads/actuals/` was left as found
(absent).

---

## What this session did not do

No pipeline, scorer or ingest ran in the working tree. `install` is unchanged; `validate` changed
only by the fragment check. F2 to F5 are reported, not fixed. The clone under `/tmp` and the
scratchpad generator are disposable; the fragment fix was not re-run in the clone, because its
tests exercise the same function on synthetic files and the clone adds nothing to that.
