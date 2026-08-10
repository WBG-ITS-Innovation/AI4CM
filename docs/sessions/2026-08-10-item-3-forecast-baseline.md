# Session log — 2026-08-10 — Item 3: the Forecast baseline series

## 1 · The prompt, exactly as given

> Continue the session on model/excellence. Item 3: the Forecast baseline series. Requirements: the
> Forecast page must plot the h-step persistence benchmark series alongside model predictions, read
> from audited run artifacts — never recomputed on the page (same principle as MASE: one
> implementation of a published number). The header figure and chart series that were left
> outstanding from item 1 are in scope. Before building, tell me in plain language where the
> baseline series will come from (which file, which function) and confirm it is identical to the
> benchmark used in evaluation.
>
> Session logging requirement (this session and all future ones): create
> docs/sessions/YYYY-MM-DD-item-3-forecast-baseline.md containing (1) the exact prompt I gave you,
> (2) your plan as stated before implementation, (3) the actual commands run and their real output
> (test results, diffs summary), (4) the final session record verdict, and (5) anything left
> outstanding. Commit the md file with the code.
>
> Build it, run the full suite, show me the passing output, commit, and walk me through the diff.

## 2 · The plan, as stated before implementing

### The chart series

`forward_forecast.csv` / `forecasts/published/<issue>/forecast.csv`, column **`origin_value`**,
written by `backend/forward_forecast.run_forward()`.

It is already the persistence prediction. Evaluation's benchmark is
`forecast_integrity.compute_persistence_baseline()`, whose docstring states its definition as
**`y_hat(t+h) = y(t)` (origin value)** and which computes it from exactly that column. For a forward
run all five horizons share one origin, so `origin_value` is constant at **46,490,793.48** — a flat
carried-forward line. Measured: `origin_value.nunique() == 1`.

**So the page reads the same column the evaluator reads — not a reimplementation, the identical
field.**

### The header figure

`experiments/runs/<dev_credentials.run_id>.json`, field **`ruler`**, reached via
`registry.load_registry()`. The registry labels it *"h=5 business-day persistence, the single shared
ruler across all four families"*.

| Target | Benchmark MAE | Model DEV MAE |
|---|---:|---:|
| Revenues | 88,317,355.17 | 38,931,955.83 |
| Expenditure | 73,117,667.21 | 51,602,950.54 |
| State budget balance | 242,653,025.46 | 194,104,921.84 |

Cross-checked before building: `(ruler − dev_mae) / ruler` reproduces each recorded skill to
**1.4e-05**; the residual is the log storing skill to four decimals, not a different benchmark.

### One limit stated up front

Forward dates have **no truth**, so the benchmark there is a rival *prediction*, not an error. Its
accuracy appears in the track record once actuals arrive.

## 3 · Commands run, and their real output

### Verification before building

```
$ ./backend/.venv/bin/python -c "...origin_value per horizon..."
  target  horizon origin_date  origin_value target_date          p50
Revenues        1  2025-08-06   46490793.48  2025-08-07 9.960823e+07
Revenues        2  2025-08-06   46490793.48  2025-08-08 7.635415e+07
Revenues        3  2025-08-06   46490793.48  2025-08-11 7.763747e+07
Revenues        4  2025-08-06   46490793.48  2025-08-12 6.155727e+07
Revenues        5  2025-08-06   46490793.48  2025-08-13 5.831942e+07
origin_value distinct across horizons: 1 (1 = a flat carried-forward line)

$ ... skill reconciliation ...
Revenues               recorded=55.9181000000  implied=55.9181139917  diff=1.40e-05
Expenditure            recorded=29.4248000000  implied=29.4247854049  diff=1.46e-05
State budget balance   recorded=20.0072000000  implied=20.0072113369  diff=1.13e-05
```

### The new backend tests

```
$ ./backend/.venv/bin/python -m pytest backend/tests/test_forecast_baseline.py -v
test_benchmark_column_is_the_one_the_evaluator_uses PASSED [  9%]
test_benchmark_series_reproduces_the_evaluators_prediction PASSED [ 18%]
test_benchmark_series_returns_none_rather_than_computing_a_substitute PASSED [ 27%]
test_published_issue_carries_the_benchmark_column PASSED [ 36%]
test_benchmark_is_flat_across_horizons_from_one_origin PASSED [ 45%]
test_benchmark_equals_the_target_value_at_the_origin PASSED [ 54%]
test_header_benchmark_mae_comes_from_the_dev_credentials_run PASSED [ 63%]
test_header_figures_reconcile_with_the_recorded_skill PASSED [ 72%]
test_a_target_with_no_recipe_reports_no_benchmark_rather_than_borrowing_one PASSED [ 81%]
test_benchmark_labels_state_it_is_a_prediction_not_an_error PASSED [ 90%]
test_forecast_page_reads_the_shared_reader_and_computes_nothing PASSED [100%]
```

### Rendered page, real values

```
$ AppTest on frontend/pages/05_Forecast.py
exceptions: []
  metric Model error on 2024 (million lari): 194.1
  metric Benchmark error (million lari): 242.7
  metric Model is better by: 20.01%
benchmark caption: True
no-truth caveat: True
run provenance caption: True

table columns: ['Date', 'Low', 'Central', 'High', 'Benchmark']
benchmark column values: ['46.5', '46.5', '46.5', '46.5', '46.5']
```

### Full suites

```
$ ./backend/.venv/bin/python -m pytest -q
ROOT EXIT=0
527 passed, 3 skipped in 85.84s (0:01:25)

$ PYTHONPATH=frontend:backend ./frontend/.venv/bin/python -m pytest -q frontend/tests
FRONTEND EXIT=0
110 passed in 5.84s
```

### Diff summary against the previous commit

```
backend/forward_forecast.py              | 80 ++++++++++++++++++++++++++++++++
 frontend/pages/05_Forecast.py            | 68 ++++++++++++++++++++++++++-
 frontend/tests/test_tier1_correctness.py | 66 ++++++++++++++++++++++++++
 3 files changed, 213 insertions(+), 1 deletion(-)
```

## 4 · Verdict

**Item 3 complete.** The Forecast page plots the persistence benchmark as its own dash-dot,
cross-marked trace read from `origin_value`; the header shows model error, benchmark error and the
skill between them, all three from the audited DEV run; the table gained a Benchmark column. Nothing
is recomputed on the page — a test asserts the page contains no `compute_persistence_baseline`, no
`.shift(`, and no `mae_persistence =`.

Root suite 516 → **527 passed, 3 skipped**, `EXIT=0`. Frontend 106 → **110 passed**, `EXIT=0`.

### Two things I got wrong along the way

**Four of my edit anchors did not match the file.** `reading_this_chart` was not in
`05_Forecast.py` at all — that caption belonged to work reverted during the visual-pass correction —
and the `ui_styles` import was single-line, not parenthesised. The whole edit was written as one
atomic script, so nothing was applied; I verified each anchor individually before retrying rather
than patching blind.

**I could not assert on the rendered figure.** Streamlit 1.40.1's `AppTest` cannot read a
`plotly_chart` element's value — the accessor raises a `session_state` KeyError. I had written two
tests claiming to inspect the figure. Rather than leave a test that asserts less than it says, both
were rewritten to check the two halves that together mean the trace is reached: the page adds it
under `if _bench is not None`, and `benchmark_series` returns a series for every published target.
The docstring now states the limitation explicitly.

## 5 · Outstanding

**Item 4 — persist the fitted estimator with each published forecast.** Save the estimators into
`forecasts/published/<issue_date>/` with library names and versions, a loader, a test that a saved
model reproduces its published predictions to tolerance, and the storage cost per issue stated.
Official runs must still refit on current data; this is for reproducing a published issue, not
serving a stale model.

**Seven artifact fields still missing**, listed in `reports/phase14_session_record.md`. The worst
remains `c_dl_pipeline.py:958` writing `alignment_ok: True` as a literal with no check behind it.

**The benchmark's realized error on forward dates cannot be shown yet**, and will not be until the
canonical data moves past 2025-08-06. The scorecard is wired for it (0 scored / 15 pending); this is
a data-freshness limit, not a gap in the code.
