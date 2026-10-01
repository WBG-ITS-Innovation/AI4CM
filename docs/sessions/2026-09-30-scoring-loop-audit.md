# Scoring-loop audit: the published store, two client-facing numbers, and one full pending-to-scored cycle

**Date:** 2026-09-30
**Working tree:** `12d3555` on `fix/ui-rendering-and-script`. Read-only, except this record and a
dated correction to `2026-09-30-inference-horizon-map.md`.
**Clone:** a fresh `git clone` of the same commit under this session's scratchpad directory, with
the two virtualenvs symlinked in. Everything that published, ingested or scored ran there, on
synthetic data. No real value was read into the clone.
**Artifact integrity:** 1,236 files under `forecasts/`, `backend/forecast_runs/`, `backend/data/`,
`private_vault/`, `registry/` and `experiments/` were fingerprinted (md5) before the first command
and after the last. **0 changed.** The holdout ledger `experiments/test_access.log` was 754 lines
before and 754 after. `git status` shows only the two session records.

This record is written to teach: each step says what was done, what the code does at that point,
and where to read it.

---

## Why this session exists

The map written earlier today claimed that `forecasts/published/` was empty on this machine and
that `list_published` returned nothing. The brief for this session says it is not empty: 54 files
across three issues, and the 10 September facts said 25 rows were pending. Part 1 resolves that in
the working tree, read-only. Part 2 exercises the whole loop, publish → new actuals → score, in the
clone, so the scorer's arithmetic can be checked by hand.

---

## Part 1 — working tree, read-only

### 1.1 `list_published` returns three issues. The map asserted without checking.

The call, made in the working tree's backend interpreter with nothing else imported:

```
PUBLISHED_ROOT = <repository root>/forecasts/published
2025-08-06
2026-08-13
2026-08-16
```

`list_published` is four lines (`backend/published_forecasts.py:481-485`): if the root exists,
return every subdirectory holding a `forecast.csv`, sorted. There is no discovery bug. The
directory listing the map relied on had been piped through `head`, which cut it after the
parent-directory line, and the truncated output was read as an empty directory. **The map stated
a fact it had not verified.** The wrong sentences are now struck through in that record with the
correction beside them, and a dated note at its top says so (§1.3 below).

### 1.2 Inventory of the store

54 files: 4 per issue for the three issues, plus 20 estimator blobs and one manifest for each of
the two later issues (4 × 3 + 21 × 2 = 54). Counts and dates only; no amount from any of these
files appears in this record.

| | 2025-08-06 | 2026-08-13 | 2026-08-16 |
|---|---|---|---|
| Targets | Revenues, Expenditure, State budget balance | Revenues | Revenues |
| Rows in `forecast.csv` | 15 | 5 | 5 |
| Horizons | 1..5 | 1..5 | 1..5 |
| Origin date | 2025-08-06 | 2025-08-06 | 2025-08-06 |
| Target dates | 08-07, 08-08, 08-11, 08-12, 08-13 (2025) | same five | same five |
| Data at issue | SHA `0b009fd0…`, 3,867 rows, ends 2025-08-06 | same | same |
| Generated (UTC, from `provenance.json`) | 2026-08-05 09:10 | 2026-08-13 22:38 | 2026-08-16 00:44 |
| Branch at issue | model/excellence | model/excellence | model/excellence |
| Gate flags recorded in `gates.json` | pre-P2 set: `signal`, `overfitting`, `vs_ruler` | post-P2 set: `accuracy_vs_naive`, `signal`, `leakage`, `persistence_mimicry`, `coverage`, `overfitting` | same as 2026-08-13 |
| Verdict at issue, derived | Revenues **withheld_as_forecast**; Expenditure **withheld_as_forecast**; State budget balance **publishable** | Revenues **publishable** | Revenues **publishable** |
| Verdict today (registry) | publishable / withheld / withheld | publishable | publishable |
| Estimator blobs | none | 20 on disk, 20 in manifest, not pruned | 20 on disk, 20 in manifest, not pruned |
| `approved_by` key in `gates.json` | absent | present, `null` | present, `null` |

**How each issue got its date.** A published issue is named by `issue_date`. When the caller
passes none, `publish` derives it from the origin date in the forecast, which is the last date in
the data (`backend/published_forecasts.py:429-430`). That is what happened to the first issue:
generated 2026-08-05 UTC, named `2025-08-06` after its data. The other two are wall-clock: the
Forecast page's publish path uses `next_issue_date()`, which is today's UTC date with an `-r2`
suffix on a same-day re-issue (`backend/forecast_modes.py:464-482`, `:523-530`). The
`--issue-date` flag did not exist until 2026-08-18 (`docs/sessions/2026-08-18-session6-prep.md:12`,
`:30-31`), so neither used it. Note the 2026-08-16 issue was generated at 00:44 UTC, which is the
evening of the 15th locally; the retention record's filename note says exactly this
(`2026-08-15-session2p5-retention.md:3-4`).

**Which publish path each came through, from the artifact alone.** The runner
`backend/run_forward_forecast.py` writes a `gates.json` entry with `target`, `gates` and `status`
(`:68-71`). The page path `forecast_modes.official_run` writes the same plus `approved_by`
(`:351-354`). The first issue has no `approved_by`; the other two do. So the first issue came
through the runner, all three targets at once, and the later two through the page path, Revenues
only.

**Verdict at publish time.** `gates.json` records each gate's `passed` flag but never recorded a
publication verdict. `_verdict_from_recorded_gates` (`backend/published_forecasts.py:838-852`)
reconstructs it under the pre-P2 policy: a failed leakage gate means withheld, a failed signal
gate means withheld-as-forecast, any other failure withheld-as-forecast, else publishable.
`reconcile_verdicts` (`:855-929`) applies it to every issue and pairs it with the registry's
current verdict. Its output, read-only:

```
2025-08-06 | Revenues             | at issue: withheld_as_forecast | today: publishable | changed: True
2025-08-06 | Expenditure          | at issue: withheld_as_forecast | today: withheld    | changed: True
2025-08-06 | State budget balance | at issue: publishable          | today: withheld    | changed: True
2026-08-13 | Revenues             | at issue: publishable          | today: publishable | changed: False
2026-08-16 | Revenues             | at issue: publishable          | today: publishable | changed: False
```

**The 25 pending rows, reconciled.** 15 + 5 + 5 = 25. The figure appears in two records as the
scorer's own output: "0 scored, 25 pending" (`2026-08-18-session6-prep.md:153`) and "0 scored, 25
pending, 3 issues" (`2026-08-19-ux-final-polish.md:402`). Every one of the 25 rows has a target
date between 2025-08-07 and 2025-08-13, and the canonical file ends 2025-08-06, so every row is
pending. Two of the three issues are re-issues of the same forecast origin, so the store holds
three separate published predictions for each Revenues target date. The Scorecard page's caption
says so (`frontend/pages/08_Scorecard.py:298-302`).

**Which actuals end-date scores every pending row.** A row is scored when its target date is in
the data with a finite value for its target (`backend/published_forecasts.py:573-582`); the truth
series is reindexed to business days and never filled (`:490-516`). The latest target date is
**2025-08-13**. A data file carrying Revenues, Expenditure and State budget balance through
2025-08-13 makes all 25 rows scoreable. Part 2 confirms it with a file through 2025-08-20.

**Estimators.** The first issue was generated on 2026-08-05; estimator retention was implemented
on 2026-08-11 (`docs/sessions/README.md`, the two "Persisted estimators" rows), so it has none and
`reproduce_prediction` would raise `EstimatorMissing` for it (`backend/estimator_store.py:298-304`).
The other two hold 4 kinds × 5 horizons = 20 blobs each, all present, none pruned.

### 1.3 The map corrected, not silently

`2026-09-30-inference-horizon-map.md` now carries a dated correction block after its header and
the false sentence in §1.4 step A3 is struck through with the corrected statement beside it. What
was wrong: "`forecasts/published/` is empty on this machine … `list_published` returns nothing …
0 scored, 0 pending". What is true: three issues, 25 rows; with a file ending 2026-09-30 the
scorer scores all 25 and labels each `scored_in_window: live`
(`backend/published_forecasts.py:735-736`, `:757-761`). The claim about a fresh clone (no
`forecasts/published/`, since `.gitignore:72` ignores it) stands.

### 1.4 Provenance of the two client-facing numbers

The numbers: **Revenues, sealed window, 55.96% skill vs the naive rule, +31.99% vs the Treasury
method**, in the table at `docs/sessions/2026-08-18-artifact-regeneration.md:182`, with a
machine-readable copy at `reports/sealed_window_champion_vs_ops.csv` (`:172`). The CSV's columns
are `target, role, model, n, MAE, skill_vs_naive_%, skill_vs_ops_%, ops_MAE`; the Revenues
champion row has `n = 146`.

**55.96% — `backend/sealed_window_report.py`, function `evaluate_champion` (`:246-271`).** It
builds folds with `sealed_folds` (`:122-206`), which is the only fold builder in the repository
that covers the sealed window with an embargo and logs the read as a report (`:153-162`,
`:198-205`). Each fold carries `origin_values`, documented as "h-step persistence, i.e. the shared
ruler" (`:112`, `:166`), and the prediction frame carries `origin_value`, `y_true` and `y_pred`
per row (`:265-270`). Skill against the naive rule is then the persistence error against the
model error over those rows. The holdout ledger confirms the run: three `sealed_window_report.sealed_folds`
entries at 2026-08-19 00:11–00:12 UTC, one per target, each "covers 146 holdout target date(s)
from 2025-01-08", which is the table's `n = 146` (the embargo drops the first four target dates;
the record says so at `:200-201`). The table is dated 2026-08-18 local.

**+31.99% — `backend/ops_baseline.py`.** The comparator is the Treasury planning method at the
vintage in force at each row's origin: `vintage_cache` (`:271-286`) → `ops_series_for_vintage`
(`:250-268`) → `ops_figure_for_month` (`:189-247`), and the percentage is `skill_vs` (`:306-315`).
This is the only implementation in the repository that produces a vintage-correct daily figure;
the leaderboard (`backend/b_ml_pipeline.py:1049-1067`) and the scorer
(`backend/published_forecasts.py:652-662`, `:719-734`) both use it, and the Dashboard view says
the same pair is used so chart and table cannot disagree (`frontend/ops_baseline_view.py:38-42`).

**What cannot be shown, stated plainly.** The script that wrote the CSV is not in the repository.
The record's reproduction section lists three test commands and nothing that regenerates the
table (`2026-08-18-artifact-regeneration.md:285-289`). The ledger holds no "Ops-baseline
comparison" entry from that session's snippet: the only such entries, one per day from
2026-08-18 to 2026-08-21, come from `b_ml_pipeline.leaderboard`, which is a different path
(156 dates, no embargo, no target transform). So the ops half of the table was computed by a call
into `ops_baseline` that did not go through `log_sealed_window_read` (`:318-334`). The
attribution above is by definition and by elimination, not by re-running a script. Recorded as a
gap in §2.6, finding F4.

**Does the parked all-NaN `ops_monthly_baseline` touch either number? No.** The call graph:

* `b_ml_pipeline.ops_monthly_baseline` (`backend/b_ml_pipeline.py:278-281`) is called in exactly
  one place, `run_pipeline_ml` (`:672`). Its output goes to two CSV files (`:674-677`), to
  `evaluate_block` (`:1017`), and to three plot functions (`:1520-1525`).
* `evaluate_block` accepts `ops_series` and never reads it: the body (`:451-503`) references only
  `y_true`, `y_pred`, `y_lo`, `y_hi`, and writes `MAE_skill_vs_Ops` as `np.nan` (`:501`).
* `sealed_window_report` imports two things from `b_ml_pipeline`: `build_yearly_folds` (`:130`)
  and `available_models` (`:215`). Not the baseline.
* `ops_baseline` never imports it either. Its whole-series form uses
  `c_dl_pipeline.ops_monthly_baseline_treasury` (`backend/ops_baseline.py:166`), and its scoring
  form does its own month arithmetic in `ops_figure_for_month`.
* `frontend/ops_baseline_view.py:32-34` records the same conclusion from the other direction:
  "Scoring never read those files, so no published figure was affected."

One related note, not on this path: the same twelve-year-shift construct
(`shift(12).rolling(36)` inside a per-month group) survives as a fallback in
`c_dl_pipeline.ops_monthly_baseline_treasury` for a series with fewer than three complete years
(`backend/c_dl_pipeline.py:209-210`). A ten-year series never reaches it.

---

## Part 2 — the disposable clone, synthetic data

### 2.1 Setup

`git clone` of the working tree at `12d3555` into the scratchpad. `backend/.venv` and
`frontend/.venv` symlinked from the working tree; the only `.pth` file in either is
`distutils-precedence.pth`, so no package resolves back to the working tree's source. Every path
the loop writes is derived from `__file__`: the ledger (`backend/evaluation_windows.py:105`), the
published root, scorecard and vault (`backend/published_forecasts.py:67-68`, `:75-76`), the
canonical file and its backups (`backend/ingest_actuals.py:50-57`). The clone arrived with no
`backend/data/`, no `forecasts/published/`, no `private_vault/`, no `experiments/log.csv`,
exactly as a client clone does.

**Synthetic data, one generator, two truncations.** A script in the scratchpad (seed 20260930)
builds one long series and writes two files from it: through **2025-08-06** and through
**2025-08-20**. The second is therefore the first plus later rows and nothing else, which the
brief calls the same generator lineage. Checked: 3,867 and 3,881 rows, 44 columns each, the first
3,867 rows of the longer file equal the shorter file row for row (`DataFrame.equals` → True),
no NaNs, 1,104 and 1,108 weekend rows. Column names are the canonical file's 44 names; only
names were taken. Flows post on Georgian working days (holidays from
`backend/preprocessing/holidays.py`), with annual seasonality, a month-end spike and 4.5% growth;
the stock is a year-resetting cumulative net position. The shorter file was installed as the
clone's canonical file.

Every number below is synthetic. It says nothing about real accuracy; it exercises the loop.

### 2.2 Publish (step 5, first half)

**Page path, one target at a time**, the command the Forecast page dispatches
(`frontend/pages/07_Forecast.py:416-418`):

```
backend/forecast_modes.py --mode official --target <T> --data <clone canonical> --publish --issue-date 2026-09-30
```

Revenues: `"ok": true … "published_to": …/forecasts/published/2026-09-30`. Five rows, origin
2025-08-06, target dates 2025-08-07, 08-08, 08-11, 08-12, 08-13.

Expenditure and State budget balance, verbatim:

```
"ok": false, "refused": true,
"reason": "Refusing to publish 'Expenditure': its current verdict is 'withheld', which means a
documented trivial benchmark is more accurate than this model. Publishing the numbers would invite a
worse decision than publishing nothing. ('withheld_as_forecast' still publishes -- there the numbers
are the best estimate available and only the event claim is withheld.)"
```

That is `publish_official` (`backend/forecast_modes.py:400-412`) reading the registry verdict
(`registry/recipes.json:229`, `:332`). The forecasts were produced; only the publish was refused.

**Runner path**, `backend/run_forward_forecast.py --publish`: all three targets, 60 estimators
retained, published to `forecasts/published/2025-08-06` (issue date = origin date, the default at
`published_forecasts.py:429-430`), mirrored to the clone's `private_vault/published/`. It calls
`published_forecasts.publish` directly (`run_forward_forecast.py:95`) and applies no verdict
check, so the two targets the page path refused are published here.

**Score before any actuals arrive**, `backend/run_publish_and_score.py --score-only`:

```
[score] published issues: 2
[score] scored=0  pending=20  -> …/clone/forecasts/scorecard.csv
[score] nothing scoreable yet -- every published date is still in the future.
[score] awaiting truth for 15 target-dates, earliest 2025-08-07
```

20 rows: 15 from the runner issue, 5 from the page issue. 15 target-dates: 5 dates × 3 targets.

### 2.3 Ingest through the CLI and score (step 5, second half)

```
backend/ingest_actuals.py --file synthetic_through_2025-08-20.csv --install --score
```

The JSON it printed, paths shortened:

```json
{
 "check": {
  "ok": true, "blockers": [], "warnings": [],
  "summary": {
   "rows_now": 3867, "rows_new": 3881, "rows_added": 14,
   "last_date_now": "2025-08-06", "last_date_new": "2025-08-20",
   "columns_now": 44, "columns_new": 44, "columns_missing": [], "columns_added": [],
   "revisions": 0,
   "sha_now": "5229e5a7…", "sha_new": "7a55e68e…"
  }
 },
 "install": {
  "installed": true,
  "backup": "…/backend/data/processed/backups/master_daily_clean_treasury.20260930T081653Z.csv",
  "rows_before": 3867, "rows_after": 3881, "rows_added": 14,
  "last_date_before": "2025-08-06", "last_date_after": "2025-08-20",
  "sha_before": "5229e5a7…", "sha_after": "7a55e68e…", "revisions": 0
 },
 "score": { "scored": 20, "pending": 0, "scorecard": "…/clone/forecasts/scorecard.csv" }
}
```

Reading it against the code: the six checks are `validate` (`backend/ingest_actuals.py:189-263`);
`rows_added` counts rows dated after the held last date (`:265-267`), fourteen calendar days here;
`revisions` is zero because the overlapping rows are identical (`:129-163`); the backup name is
UTC (`:288-292`); `--score` calls `score_published` on the canonical path (`:373-378`). No blocker,
no warning, nothing on stderr beyond library warnings.

### 2.4 The scorecard, checked by hand (step 6)

Twenty rows, 31 columns, all `persistence_source = "artifact: origin_value"`, all
`scored_in_window = "live"`. Two rows were recomputed from the installed file and the published
`forecast.csv`, with no scorer code involved beyond reading the two files.

**Row 1: issue 2026-09-30, Revenues, h=4, target date 2025-08-12** (an interval miss).

```
published : p10 52,278,882.86   p50 89,817,949.24   p90 122,870,447.18   origin_value 83,312,264.56
truth     : y(2025-08-12) = 178,119,864.79                           (from the installed file)

point error        |178,119,864.79 − 89,817,949.24|  =  88,301,915.55      scorecard 88,301,915.55  ✓
persistence pred   origin_value                     =  83,312,264.56
                   value in file at 2025-08-06      =  83,312,264.56
                   2025-08-12 minus 4 business days = 2025-08-06 → 83,312,264.56  (same number) ✓
persistence error  |178,119,864.79 − 83,312,264.56|  =  94,807,600.23      scorecard 94,807,600.23  ✓
skill vs ruler     (94,807,600.23 − 88,301,915.55) / 94,807,600.23 × 100 = 6.862%   scorecard 6.862% ✓
interval           52,278,882.86 ≤ 178,119,864.79 ≤ 122,870,447.18 → False        scorecard False ✓
window             window_for(2025-08-12) = live                                 scorecard live  ✓
```

What each line is in the code: the point error is `abs(y - p50)` (`backend/published_forecasts.py:587`);
the ruler is read from the artifact's `origin_value` and recomputed from the actuals at
`target_date − h` business days as a cross-check (`:519-562`, `:541-548`); skill is
`(pae − ae) / pae × 100` (`:589`); the hit flag is `p10 ≤ y ≤ p90` (`:600`); the window label is
`window_for(target_date)` (`:735-736`). `baseline_disagreements` was empty: the published ruler
and the recomputed one agreed on all 20 rows (`:743-752`).

**Row 2: issue 2025-08-06, State budget balance, h=3, target date 2025-08-11** (a stock target,
modelled as a change and rebuilt as a level).

```
published : p10 5,823,147,934.19   p50 5,981,478,268.50   p90 6,027,772,326.71   origin_value 6,001,176,895.43
truth     : y(2025-08-11) = 6,086,545,964.17

point error        |6,086,545,964.17 − 5,981,478,268.50| = 105,067,695.67     scorecard 105,067,695.67 ✓
persistence        2025-08-11 minus 3 business days = 2025-08-06 → 6,001,176,895.43 = origin_value ✓
persistence error  |6,086,545,964.17 − 6,001,176,895.43| =  85,369,068.74     scorecard  85,369,068.74 ✓
skill vs ruler     (85,369,068.74 − 105,067,695.67) / 85,369,068.74 × 100 = −23.075%   scorecard −23.075% ✓
interval           5,823,147,934.19 ≤ 6,086,545,964.17 ≤ 6,027,772,326.71 → False   scorecard False ✓
ops comparator     NaN, ops_source "not defined: the Treasury method aggregates a flow to an annual
                   total, and a balance level has no annual total"          (ops_baseline.py:110-111)
```

The scorer compares the published level to the actual level; the delta-versus-level
reconstruction happened at publish time (`backend/forward_forecast.py:290-299`) and the
scorecard never sees a delta.

**The Treasury-method comparator, by hand, for row 1.** The origin is 2025-08-06, so the latest
complete year a planner had was 2024 and the window is 2022–2024 (`_vintage_year`,
`backend/ops_baseline.py:179-186`). From the installed file, Revenues:

```
2024: annual 27,368,807,594   August 2,070,287,464   share 0.075644
2023: annual 26,893,381,650   August 1,924,143,378   share 0.071547
2022: annual 24,971,428,164   August 2,291,487,727   share 0.091764
mean annual 26,411,205,802 × mean August share 0.079652 = August 2025 total 2,103,701,239
spread flat over the 21 weekdays of August 2025 = 100,176,249.48 per day
ops_figure_for_month(…, 2025, 8, 2024) → total 2,103,701,239, weight on 2025-08-12 = 1/21 → 100,176,249.48
scorecard ops_pred = 100,176,249.48 ✓
ops error |178,119,864.79 − 100,176,249.48| = 77,943,615.31
skill vs ops = (1 − 88,301,915.55 / 77,943,615.31) × 100 = −13.289%     scorecard −13.289% ✓
```

That is `ops_figure_for_month` (`:206-247`) with the flat spread (`:246`) and `skill_vs`
(`:306-315`), attached per row at `published_forecasts.py:719-734`.

### 2.5 Pending to scored, every row, and what the page shows (step 7)

**Before** was measured by running `score_published` against the pre-ingest backup file with a
scratch scorecard path (the function takes one, `backend/published_forecasts.py:619`), so the real
clone scorecard was not touched by the check: 0 scored, 20 pending, 15 pending target-dates.
**After** is the real scorecard: 20 scored, 0 pending, 0 baseline disagreements.

| Issue | Target | h | Target date | Before | After | Window |
|---|---|---:|---|---|---|---|
| 2025-08-06 | Expenditure | 1–5 | 08-07, 08-08, 08-11, 08-12, 08-13 | pending ×5 | scored ×5 | live |
| 2025-08-06 | Revenues | 1–5 | same five | pending ×5 | scored ×5 | live |
| 2025-08-06 | State budget balance | 1–5 | same five | pending ×5 | scored ×5 | live |
| 2026-09-30 | Revenues | 1–5 | same five | pending ×5 | scored ×5 | live |

Every row moved; none was dropped; none was scored twice. The two Revenues issues carry identical
numbers, because the same recipe was refitted on the same file from the same origin.

**The Scorecard page, rendered in the clone** (Streamlit `AppTest` against the clone's page,
frontend interpreter, no exception):

```
Predictions scored: 20      Still pending: 0      Forecast issues retained: 2      Data reported through: 2025-08-20
[success] Every published forecast has been scored. Nothing is waiting.
Published forecasts → "Already scored (20)"  (one 20 × 7 table)
Results →
  Expenditure           5 scored   Typical error 32.4 M   Inside the range 40%   Better than the naive rule by −40.0%   Degrading.
  Revenues             10 scored   Typical error 42.4 M   Inside the range 80%   Better than the naive rule by −14.6%   Degrading.
  State budget balance  5 scored   Typical error 96.3 M   Inside the range 80%   Better than the naive rule by −24.3%   Degrading.
  three 8-column results tables (5, 10, 5 rows); one "previous data file(s) kept" table (1 row)
```

The state sentence is chosen at `frontend/pages/08_Scorecard.py:250-266`; the metrics at
`:227-246`; the per-target block at `:454-481`; the health verdict at `:367-410`. On synthetic
noise the models do not beat persistence, so "Degrading" is the page doing its job on data that
has no calendar structure to learn; it is not a statement about the real champions.

**What the page would show on the real store** if actuals through 2025-08-13 were installed,
inferred from the same code and the inventory in §1.2, not run: 25 scored, 0 pending, 3 issues;
the success line; Revenues with n = 15 (three issues of one origin), Expenditure and State budget
balance with n = 5 each; and every row labelled `live`. The two definitional points in F1 and F2
below would apply to it exactly as they applied here.

### 2.6 Findings — described and proposed, not fixed

**F1. The scorecard's `publication_verdict` column holds the recipe's status, not its verdict.**
Every one of the 20 synthetic rows carries `publication_verdict = "candidate -- pre-tuning"`.
The column is documented as "the gate verdict in force at issue"
(`backend/published_forecasts.py:177`). The scorer fills it from the `status` field of the
issue's `gates.json` (`:685-688`), and both writers put the recipe's registry `status` there
(`backend/run_forward_forecast.py:70`; `backend/forecast_modes.py:353`), never the publication
verdict. The real store would produce the same value on all 25 rows, because all three recipes
carry that status (`registry/recipes.json:45`). The verdict-at-issue is derivable from the same
file by `_verdict_from_recorded_gates` (`:838-852`), which `reconcile_verdicts` already uses.
Nothing in the app reads the column today (grep: no reader outside the writer), and the page
test's synthetic scorecard fills it with `"publishable"`, which shows what the column was meant
to hold (`frontend/tests/test_scorecard_page.py:92`). *Proposal:* have `score_published` write
`_verdict_from_recorded_gates(gates)` into the column, and carry the status under its own name
if it is wanted. A one-row test on a real issue's `gates.json` would pin it.

> **Fixed 2026-10-01** (`fix/scoring-trust-batch`, commit "Scorecard: write the verdict at
> issue, carry the recipe status under its own name"). `score_published` writes the verdict the
> issue's recorded gate flags imply into `publication_verdict` and the registry status into a new
> `recipe_status` column; schema version 3. An issue with no recorded flags gets a blank verdict,
> never a pass. `_verdict_from_recorded_gates` now reads a failed `accuracy_vs_naive` as
> `withheld`, the P2 severity, for the post-P2 issues that carry that gate. Seven tests in
> `backend/tests/test_scorecard_verdict_column.py`, on temporary fixtures only. Clone evidence in
> "Fixes applied 2026-10-01" below.

**F2. Two definitions of "better than the naive rule" sit under one label.** The page's
per-target metric is the mean of the per-row skill percentages
(`frontend/pages/08_Scorecard.py:471-475`), and `_health_verdict` decides "degrading" on the same
mean (`:381-382`, `:395-398`). The scorer's own per-target summary computes skill from the two
mean errors, `(persistence MAE − MAE) / persistence MAE` (`backend/published_forecasts.py:798-806`),
which is how the registry, the ruler note and the sealed-window table define skill. On the same
20 rows:

| Target | Page (mean of row skills) | Scorer summary (from mean errors) |
|---|---:|---:|
| Expenditure | −40.0% | +0.37% |
| Revenues | −14.6% | −5.45% |
| State budget balance | −24.3% | −18.76% |

The gap on Expenditure is one row whose persistence error was tiny, giving a row skill of −251.7%
that dominates the mean; the aggregate definition is insensitive to that. The health verdict
flipped on it: Expenditure reads "no longer more accurate than the simple rule" while its
aggregate skill is positive. *Proposal:* the page should read the aggregate from
`summarize_scorecard` (already returned in `load_scoring()`'s `summary`) and the verdict should
turn on that, or the page should label its figure as an average of daily skills. Either way, one
definition per label.

> **Fixed 2026-10-01** (commit "Scorecard page: one definition of skill, and say when an origin
> is re-issued"). The per-target metric and `_health_verdict` both read the aggregate from
> `summarize_scorecard`, computed on the rows on screen so a substituted scorecard is summarised
> the same way; no mean-of-row-skills figure survives on the page. The page tests' synthetic
> scorecard, whose `skill` parameter had contradicted its own errors, now scales the persistence
> error so the aggregate equals the parameter. Four tests added to
> `frontend/tests/test_scorecard_page.py`, the first of them
> `test_one_near_zero_persistence_row_does_not_flip_the_health_verdict`. Rendered in the clone
> after the fix: Expenditure 0.4%, Revenues −5.4%, State budget balance −18.8%, which are the
> scorer's own figures, and Expenditure's verdict no longer cites skill.

**F3. The runner's publish path does not apply the verdict refusal.** Demonstrated in §2.2: the
page path refused Expenditure and State budget balance as `withheld`; `run_forward_forecast.py
--publish` published both (`:95` calls `publish`, not `publish_official`). The map noted this
from the code; this session saw it happen. *Proposal:* route the runner through
`publish_official`, or state in its docstring that it is a bulk path that bypasses the verdict.

> **Fixed 2026-10-01** (commit "Route the runner's --publish through the page's verdict
> guard"). One guard, `forecast_modes.refuse_withheld`, with the wording moved out of
> `publish_official` rather than copied; the runner applies it per target, publishes a filtered
> copy from a staging directory and keeps `forward/latest` complete for the surfaces that read
> it. `withheld_as_forecast` still publishes. Seven tests in
> `backend/tests/test_runner_publish_guard.py`, one of which asserts the refusal sentence
> appears in exactly one module. Clone evidence in "Fixes applied 2026-10-01" below.

**F4. The +31.99% table's producing code is not in the repository, and its ops half left no
ledger entry.** §1.4. The naive-skill half is attributable to a logged `sealed_folds` run at
2026-08-19 00:11 UTC with n = 146; the ops half is attributable to `ops_baseline` by
elimination, and no `log_sealed_window_read` entry records it. *Proposal:* add a small writer in
`sealed_window_report` (or a `scripts/` file) that regenerates
`reports/sealed_window_champion_vs_ops.csv` from `evaluate_champion` plus `vintage_cache` /
`ops_prediction_for`, calling `log_sealed_window_read` so the read is on the ledger, and cite it
from the 2026-08-18 record's reproduction section.

**F5. Same-origin re-issues are counted as separate predictions in per-target summaries.**
Observation, not a defect: the page's caption says a line forecast in two issues from the same
origin is two published predictions (`frontend/pages/08_Scorecard.py:298-302`). It means the real
store's Revenues summary would be three copies of one forecast (n = 15), which weights that origin
three times against Expenditure's once. Worth a sentence beside the per-target block.

> **Fixed 2026-10-01**, in the same commit as F2. A caption under each target's heading states
> how many issues its rows come from and that a re-issue from the same origin counts as a
> separate prediction. Test:
> `frontend/tests/test_scorecard_page.py::test_the_per_target_block_says_re_issues_count_separately`.

**F6. A latent twelve-year shift remains in `c_dl_pipeline`.** `ops_monthly_baseline_treasury`
falls back to the parked construct when fewer than three complete years exist
(`backend/c_dl_pipeline.py:209-210`). Not reached by any ten-year series; recorded so it is not
rediscovered.

---

## What this session did not do

No fix was applied. The clone and its scratch files are disposable. The working tree's artifacts
are byte-identical to the start of the session, the ledger did not grow, and the only files
changed are this record and the dated correction to the map. The E_QUANTILE window decision and
the findings of the 2026-09-29 audit are untouched.

---

# Fixes applied 2026-10-01

Branch `fix/scoring-trust-batch`, cut from `main` at `236198a` (the merge of PR #29). Code and
tests only in the working tree; no pipeline, scorer or artifact regeneration ran there. Every
fix was written failing-first, and each was walked through before the next began. Everything
that publishes, ingests or scores ran in the disposable clone on the synthetic pair from Part 2,
with the clone's canonical file being the longer one (through 2025-08-20, so LIVE rows exist).
F1, F2, F3 and F5 above carry their own dated "Fixed" notes; this section holds the clone
evidence and the E_QUANTILE resolution. F4 and F6 remain as proposed.

### E_QUANTILE: Option A, decided and applied

The open decision in `2026-09-29-lab-audit.md` is decided as **Option A**: selection never reads
the holdout. The family's overrides in `scripts/run_daily_forecast.sh` are now
`{"eval_start": "2024-01-01", "eval_end": "2024-12-31", "min_train_years": 4}`, which is
`evaluation_windows.DEV`, both bounds inclusive by target date
(`backend/e_quantile_daily_pipeline.py:781-809` map them to origin indices). The comment beside
the override states the window, the reason, and that the TEST-window coverage figure comes
later through the logged report path (`require_test_access(..., purpose="report")`) in the
scoring session. The script's header comment no longer says the family aborts; the order
`A_STAT B_ML C_DL E_QUANTILE` is kept.

**The count, from the calendar alone.** On the business-day index the pipeline builds
(`to_business_index` reindexes to `B`), `_time_folds` with these bounds gives **53 five-day
blocks and 262 evaluation points**, first target 2024-01-01, last 2024-12-31, every target in
DEV. The previous window, 2025-01-01 to the end of the file, gave 151 holdout target dates and a
refusal. 262 clears the ~150 the old comment asked for, and it is the calendar's number, not
the data's: it transfers to the real file unchanged.

**Test:** `backend/tests/test_runner_script_vars.py::test_e_quantile_selects_only_inside_the_dev_window`,
beside the B_ML one, comparing both bounds to the window constants.

**Clone run, full default order, synthetic data through 2025-08-20.**

```
[daily] Using data file: …/clone/backend/data/processed/master_daily_clean_treasury.csv
[daily] === Running A_STAT ===            [runner] DONE
[daily] === Running B_ML ===
[pipeline] Evaluation window bounded: [start .. 2024-12-31] -> 1 fold(s)        [runner] DONE
[daily] === Running C_DL ===
[DL] eval_start=2025-01-01 pinned: 1 fold(s) kept, 6 dropped for starting before it   [runner] DONE
[daily] === Running E_QUANTILE ===
[runner] Evaluation window pinned: origins >= 2024-01-01 (428 rows)
[runner] Evaluation window capped at 2024-12-31 (origin index <= 2581)
[runner] Elapsed: 438.0s
[daily] === Writing summary ===
[daily] DONE. Summary: …/clone/backend/forecast_runs/2026-10-01/SUMMARY.txt
EXIT=0
```

| Check | Result |
|---|---|
| Script exit code | 0 |
| `SUMMARY.json` / `SUMMARY.txt` | both written; `families` has four entries, each `ok: true` |
| E_QUANTILE `run_status` | `SUCCESS`, best model chosen (so `assert_selection_free` was reached and passed) |
| E_QUANTILE `predictions_long.csv` | 1,572 rows, 6 models, **262 distinct target dates**, 2024-01-01 .. 2024-12-31, every one in window `dev` |
| Clone holdout ledger after the run | two report reads, from `run_a_stat.main` and `c_dl_pipeline.yearly_folds`; **none from E_QUANTILE** |
| `SUMMARY.json` `windows.data_spans_windows` | `[train, dev, test, live]`, the file's real extent |

The "428 rows" in the pin line is the count of origins from 2024-01-01 to the clone file's end;
the cap on the next line cuts that to 2024. A_STAT and C_DL still evaluate 2025 as reporting
reads, which is unchanged and logged. Two artifact-contract warnings about `b_ml/metrics_long.csv`
were printed at the end, as in the 2026-09-29 audit; they are unrelated to this change.

### Clone evidence for F3 and F1

**F3.** `backend/run_forward_forecast.py --publish` on the clone (data through 2025-08-20):

```
[forward] REFUSED Expenditure: Refusing to publish 'Expenditure': its current verdict is 'withheld',
  which means a documented trivial benchmark is more accurate than this model. … ('withheld_as_forecast'
  still publishes -- …)
[forward] REFUSED State budget balance: Refusing to publish 'State budget balance': … (same sentence)
[forward] published Revenues to …/clone/forecasts/published/2025-08-20
```

The store afterwards: `2025-08-06` (all three targets, from before the fix), `2025-08-20`
(Revenues only, from the fixed runner), `2026-09-30` (Revenues only, page path). The runner now
refuses exactly the two targets the page refuses, with the page's words.

**F1.** `backend/run_publish_and_score.py --score-only` on the clone: 3 issues, 20 scored, 5
pending (the new issue's dates are past the data). The scorecard has 32 columns and
`schema_version` 3 on every row:

| Issue | Target | `publication_verdict` | `recipe_status` |
|---|---|---|---|
| 2025-08-06 | Expenditure | withheld | candidate -- pre-tuning |
| 2025-08-06 | Revenues | publishable | candidate -- pre-tuning |
| 2025-08-06 | State budget balance | withheld | candidate -- pre-tuning |
| 2026-09-30 | Revenues | publishable | candidate -- pre-tuning |

The two withheld verdicts are derived from the issue's own recorded gate flags, where the
accuracy gate failed; before the fix every row read `candidate -- pre-tuning` under the
verdict's name.

**F2.** The Scorecard page rendered in the clone after scoring shows the scorer's aggregates
(Expenditure 0.4%, Revenues −5.4%, State budget balance −18.8%) where it showed −40.0%, −14.6%
and −24.3% before, and the F5 caption under each target.

### Horizon guard

`official_run` refuses a recipe whose `params.horizon` differs from `VALIDATED_HORIZON`, with a
reason naming the recipe, both numbers, and the way out (`backend/forecast_modes.py`, inside
`official_run`, before any data is read). Tests in `backend/tests/test_official_horizon_guard.py`
(4) doctor the real Revenues recipe to horizon 10 in memory. This closes the first bullet of
the map's §2.7 "what breaks silently".

### Suites and new tests

| Suite | Baseline | After |
|---|---|---|
| Frontend, `frontend/.venv` | 727 passed, 16 skipped | **731 passed, 16 skipped** |
| Backend, `backend/.venv` (both test paths) | 1345 passed, 20 skipped | see the closing note of this record |

New tests, 23 in all:

* `backend/tests/test_runner_publish_guard.py` (7):
  `test_the_runner_refuses_the_targets_the_page_refuses`,
  `test_the_refusal_reason_is_the_page_paths_wording`,
  `test_withheld_as_forecast_still_publishes_through_the_runner`,
  `test_nothing_is_published_when_every_target_is_refused`,
  `test_the_published_manifest_names_only_the_targets_that_were_published`,
  `test_the_refusal_wording_lives_in_one_place`,
  `test_the_runner_does_not_publish_the_shared_forward_directory_directly`
* `frontend/tests/test_scorecard_page.py` (4 added):
  `test_one_near_zero_persistence_row_does_not_flip_the_health_verdict`,
  `test_the_per_target_skill_is_the_scorers_aggregate_not_an_average_of_daily_skills`,
  `test_no_surviving_average_of_daily_skills_goes_unlabelled`,
  `test_the_per_target_block_says_re_issues_count_separately`
* `backend/tests/test_scorecard_verdict_column.py` (7):
  `test_a_failed_signal_gate_is_written_as_withheld_as_forecast_not_as_the_status`,
  `test_all_gates_passing_is_written_as_publishable`,
  `test_the_recipe_status_is_carried_under_its_own_name`,
  `test_an_issue_with_no_recorded_gate_flags_gets_no_verdict_rather_than_a_pass`,
  `test_a_failed_accuracy_gate_reads_as_withheld_under_the_gate_set_that_records_it`,
  `test_the_schema_version_was_bumped_for_the_new_column`,
  `test_recipe_status_sits_beside_the_verdict_in_the_column_order`
* `backend/tests/test_runner_script_vars.py` (1 added):
  `test_e_quantile_selects_only_inside_the_dev_window`
* `backend/tests/test_official_horizon_guard.py` (4):
  `test_a_recipe_declared_at_another_horizon_is_refused_before_any_data_is_read`,
  `test_the_reason_names_both_horizons`,
  `test_the_real_recipes_declare_the_validated_horizon`,
  `test_a_recipe_that_declares_no_horizon_is_not_refused_on_that_ground`

One pre-existing test fixture was corrected (`_synthetic_scorecard` in the page tests, see F2's
note), and one tracked schema file was rewritten by hand: the header-only
`forecasts/scorecard.csv`, from `SCORECARD_COLUMNS`, which the schema test requires to match.

**A note on the backend suite on this machine.** While running subsets during this session, a
segmentation fault inside LightGBM's dataset construction appeared twice when
`test_scorecard_schema`, `test_published_forecasts` and `test_ops_baseline` ran in one process,
with and without the new test file, and did not reproduce on the third attempt of the same
combination nor in the clone. It is intermittent and predates this branch; the full-suite
result is stated in the closing note below.

### Closing note, 2026-10-01

**Both suites green.** Frontend under `frontend/.venv`: **731 passed, 16 skipped** (727 + 4).
Backend under `backend/.venv`, both test paths: **1364 passed, 20 skipped** (1345 + 19), in
9 minutes 42 seconds, no segmentation fault. The four new frontend page tests skip under the
backend interpreter, as every page test does, which is why the two suites add different counts.

Working tree after the suites, against the fingerprint taken at the start of the 2026-09-30
audit: two files differ. `experiments/test_access.log` grew from 754 to 772 lines, the
suites' report reads, which the ledger exists to record. `forecasts/scorecard.csv` has the
new 32-column header, written by hand from the schema constant. No forecast, run, registry,
vault or data artifact changed. No pipeline, scorer or forward run was executed in this
working tree.
