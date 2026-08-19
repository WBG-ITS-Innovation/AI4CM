# Session — P0: the two defects that made published realized accuracy wrong

**Date:** 2026-08-13
**Repo:** `AI4CM` (the Lab), branch `model/excellence`, parent `776155e`
**Counterpart:** `AI4CM-agent` at `c0fd448` — unaffected by this change, run anyway.
**Scope:** P0 items 1 and 2 from `docs/sessions/2026-08-13-refresh-retrain-score-diagnostic.md`.
P1 and P2 untouched, as instructed.

---

## 1. The prompt, verbatim

> Continue on model/excellence in the AI4CM repo. Implement P0 from the diagnostic — the two
> defects that make published realized accuracy wrong.
>
> 1. score_one's persistence baseline (published_forecasts.py:146, :165-166, :239). Read
> row["origin_value"] rather than recomputing; fall back to truth.iloc[pos -
> int(row["horizon"])]. A published issue has one origin, so the h-step ruler is the same
> number for all five horizons. Assert the scored baseline equals the artifact's origin_value
> to tolerance — the same one-implementation rule already enforced on the Forecast page.
>
> 2. Missing actual scored as zero (:139). _truth_series routes actuals through
> to_business_index which zero-fills missing flows, so a genuinely absent actual is scored as
> y_true=0.00 with a fabricated error instead of reported pending. Score against the raw
> series, not the zero-filled modelling series, so TruthNotAvailable can actually fire for a
> flow inside the data range.
>
> Then re-score the 15 pending forecasts against a synthetic actuals set and paste the
> corrected realized skill per target, alongside the wrong numbers, so I can see the size of
> the correction. Add a regression test for each defect — mutation-tested where possible:
> reintroduce the fixed horizon and assert a test fails.
>
> Session logging: docs/sessions/2026-08-13-p0-scoring-correctness.md, full narrative verbatim
> plus prompt, plan, real output, verdict, outstanding. Both suites green, commit the md with
> the code, walk me through the diff.

---

## 2. Plan

1. Capture the **wrong** numbers first, with the unmodified scorer, against a deterministic
   synthetic actuals set — otherwise there is nothing to compare a correction to.
2. Fix defect 1 (read the ruler) and defect 2 (raw truth series).
3. Re-score with the identical actuals file and diff the two scorecards row by row.
4. A regression test per defect, each paired with a mutation that must turn it red.
5. Both suites; commit code, tests and this record together.

---

## 3. What was actually wrong, at the line level

`score_one(row, truth, horizon_steps: int = 5)` computed the comparator as
`truth.iloc[pos - horizon_steps]`. `horizon_steps` was a **parameter defaulted to 5** and
`score_published` passed its own identically-defaulted 5 through for every row, while
`row["horizon"]` — the row's real horizon — was read thirteen lines away and written straight
into the output. So the scorecard reported h=1…5 per issue and scored all five against
`y(target_date − 5 business days)`.

The project's ruler is `ŷ(t+h) = y(t)`: the value at the **origin**. A published issue has one
origin, so the ruler is one number for all five horizons — and it is already in the artifact,
as `origin_value`. `forward_forecast.py` documents that column as exactly this quantity and
`frontend/pages/05_Forecast.py:197` reads it rather than recomputing, with a comment saying
why. The scorer was the one place that rebuilt it, at the point where it decides published
accuracy.

The second defect is one line: `_truth_series` returned
`to_business_index(pd.read_csv(data_path), "date", target)`. That function is the *modelling*
series — `b_ml_pipeline.py:181-186` fills a missing flow day with `0.0` and forward-fills a
stock. Correct for fitting, fatal for scoring: after the fill every business day in range is
finite, so `TruthNotAvailable` could not fire for a flow inside the data range no matter what
the file actually contained.

---

## 4. Real output

### 4.1 The premise, checked before relying on it

`origin_value` is not approximately the actual at the origin; it is the actual at the origin.

```
$ backend/.venv/bin/python -c "...compare forecast.csv origin_value to the canonical file..."
Revenues               origin_value=[46490793.48]   canonical=46490793.480000   diff=0.000000
Expenditure            origin_value=[1.83904534e+08] canonical=183904534.380000 diff=0.000000
State budget balance   origin_value=[1.75698032e+09] canonical=1756980317.790000 diff=0.000000
```

That exactness is what sets the agreement tolerance below.

### 4.2 Re-scoring the 15 published predictions

Synthetic actuals: the canonical file extended over 2025-08-07 … 2025-08-13 (5 business days),
values drawn from each target's own last-130-day distribution under a fixed seed
(`np.random.default_rng(20260813)`), so the run is reproducible and the magnitudes are right
for each series. Only `date` and the three target columns are populated on the new rows —
scoring reads nothing else. The identical file is used for both runs; the pre-fix scorer is
loaded from `git show HEAD:backend/published_forecasts.py` so the worktree is never reverted.

**Per-target realized skill — the correction:**

```
                         BEFORE (wrong)         AFTER (corrected)      change
 Revenues                    -3.39%                  +35.79%          +39.18 pts, sign flip
 Expenditure                 +70.13%                 +70.52%           +0.39 pts
 State budget balance        -30.20%                 -39.12%           -8.92 pts

 realized_mae is identical in both runs -- the forecast did not change, only the ruler:
 Revenues              50,598,466   |  persistence_mae  48,940,424 -> 78,802,980
 Expenditure           26,383,854   |  persistence_mae  88,318,886 -> 89,496,798
 State budget balance 135,506,515   |  persistence_mae 104,078,783 -> 97,404,978
```

**Per row — one origin, one ruler:**

```
target                 h  persistence_before  persistence_after  skill_before  skill_after
Revenues               1          82,103,398         46,490,793        32.23%       59.07%
Revenues               2         104,560,410         46,490,793       -41.73%       23.77%
Revenues               3          73,486,529         46,490,793        26.82%       73.34%
Revenues               4          75,125,619         46,490,793       -64.72%       30.38%
Revenues               5          46,490,793         46,490,793        13.70%       13.70%
Expenditure            1          70,362,496        183,904,534        15.83%       80.54%
Expenditure            2         248,014,995        183,904,534        92.63%       86.47%
Expenditure            3          44,108,644        183,904,534        55.83%       38.61%
Expenditure            4          48,183,370        183,904,534        47.30%       28.30%
Expenditure            5         183,904,534        183,904,534        85.38%       85.38%
State budget balance   1       1,974,843,056      1,756,980,318        80.99%     -120.89%
State budget balance   2       1,838,063,266      1,756,980,318      -474.23%      -39.40%
State budget balance   3       1,867,444,342      1,756,980,318        66.86%      -41.43%
State budget balance   4       1,894,389,280      1,756,980,318      -432.81%      -35.97%
State budget balance   5       1,756,980,318      1,756,980,318       -33.33%      -33.33%

rows whose skill did NOT change: 3 of 15
rows unchanged: Revenues h=5, Expenditure h=5, State budget balance h=5
```

Exactly the three h=5 rows are unchanged, which is the diagnostic's claim reproduced
independently: **12 of 15 were wrong, and only h=5 was ever right.** The `persistence_before`
column also reproduces the diagnostic's numbers to the lari (82,103,398 / 104,560,410 /
73,486,529 / 75,125,619 / 46,490,793), even though the synthetic *actuals* differ from that
session's draw — because those baselines were read out of real history, not out of the
synthetic tail. Two independent generators, same wrong ruler.

**Read this correction carefully.** It is not "the forecast got better". `realized_mae` is
byte-identical in both runs. What changed is the yardstick: the wrong one was, per horizon, a
different day's value — sometimes an easy day to beat, sometimes an impossible one. Revenues
looked worse than persistence and was in fact 35.79% better; State budget balance looked
better than it is. Neither number was noise, and neither was safe to show a client.

### 4.3 The missing-actual defect, before and after

2025-08-11 removed from the same synthetic actuals:

```
########## GAP (2025-08-11 absent), PRE-FIX ##########
scored=15  pending=0
            Revenues        3  2025-08-11  p50 7.763747e+07  y_true 0.000000e+00  abs_error 7.763747e+07   skill    -5.65%
         Expenditure        3  2025-08-11  p50 8.950422e+07  y_true 0.000000e+00  abs_error 8.950422e+07   skill  -102.92%
State budget balance        3  2025-08-11  p50 1.748292e+09  y_true 1.864058e+09  abs_error 1.157662e+08   skill -3318.58%

########## GAP (2025-08-11 absent), POST-FIX ##########
scored=12  pending=3
pending: [('Expenditure', '2025-08-11'), ('Revenues', '2025-08-11'), ('State budget balance', '2025-08-11')]
```

Both halves of the fill are visible there. The two flows were scored against a fabricated
**zero**. The stock was scored against 1,864,058,000 — which is not an observation of
2025-08-11 at all, it is 2025-08-08's balance carried forward by `ffill`, and it produced a
skill of −3318%. Afterwards all three report pending, which is the truthful answer: the file
does not say what happened that day.

### 4.4 Mutation testing — each fix reintroduced in the source

Not monkeypatched: the actual file was edited, the suite run, and the file restored from a
copy taken beforehand.

```
=== suite under MUTATION A (fixed horizon reintroduced) ===
FAILED test_one_origin_means_one_ruler_for_every_horizon
FAILED test_the_scored_ruler_equals_the_artifacts_origin_value
FAILED test_a_row_without_origin_value_falls_back_to_its_own_horizon
FAILED test_reintroducing_the_fixed_horizon_fails_this_suite
FAILED test_a_disagreeing_origin_value_is_reported
5 failed, 7 passed

=== suite under MUTATION B (zero-fill reintroduced) ===
FAILED test_a_missing_flow_day_is_pending_not_a_real_zero
FAILED test_a_missing_stock_day_is_not_forward_filled
FAILED test_the_scoring_series_is_not_the_modelling_series
3 failed, 9 passed

=== restored ===
12 passed
file identical to pre-mutation
```

---

### 4.5 Independent reproduction on the diagnostic's own draw

§4.2 used a fresh synthetic draw, so its per-target *before* figures differ from the diagnostic's.
This run repeats the exercise with the **identical generator the diagnostic used** — seed 0, each
column drawn from its own trailing-60-row mean and standard deviation, dates 2025-08-07 … 2025-08-13
— so the before column reproduces the diagnostic to the decimal and the headline `-1449%` row is
visible directly. The pre-fix comparator is monkeypatched in rather than the worktree reverted.

```
REALIZED SKILL vs PERSISTENCE -- same 15 forecasts, same synthetic actuals
  scored 15 / pending 0   (wrong run: 15/0)

target                     WRONG (fixed 5)     CORRECTED     correction
Revenues                            -7.23%        31.27%        +38.50pp
Expenditure                         74.34%        75.08%         +0.74pp
State budget balance                -1.44%        -7.39%         -5.94pp

target                    persistence_mae WRONG        CORRECTED    realized_mae (same)
Revenues                             23,303,531       36,359,271             24,987,960
Expenditure                          74,914,777       77,153,743             19,224,604
State budget balance                184,518,075      174,303,536            187,180,179

Revenues, per row -- the -1449% row is h=1:
 horizon target_date  persistence_pred  skill_vs_ruler_pct  persistence_pred_fixed  skill_vs_ruler_pct_fixed     persistence_source
       1  2025-08-07       82103398.35        -1449.481283             46490793.48                 55.162600 artifact: origin_value
       2  2025-08-08      104560409.78           67.130516             46490793.48                 13.965310 artifact: origin_value
       3  2025-08-11       73486528.68            8.164318             46490793.48                 40.014656 artifact: origin_value
       4  2025-08-12       75125619.10         -212.901892             46490793.48                 43.037397 artifact: origin_value
       5  2025-08-13       46490793.48          -72.908653             46490793.48                -72.908653 artifact: origin_value

baseline_disagreements: []
```

The three figures the diagnostic reported — **−1449.48%, −212.90%, −72.91%** — reproduce exactly,
and the first two become **+55.16%** and **+43.04%**. `-72.91%` at h=5 is unchanged, because h=5 was
the one horizon the old comparator got right. Every row now reads `artifact: origin_value` and every
row uses the same 46,490,793.48, which is what "one origin, one ruler" means. Per target, Revenues
moves **−7.23% → +31.27%**, a 38.50-point correction that flips the sign.

Two notes on reading this. `realized_mae` is unchanged in every row — the forecast did not move, only
the yardstick. And `State budget balance` gets *worse* under the correct ruler (−1.44% → −7.39%);
the correction is not a flattering adjustment, it is simply the right denominator.

---

## 5. The diff, and one decision inside it

`backend/published_forecasts.py`, +127 −15. Four changes:

1. **`_truth_series` rewritten** to build the business-day index itself and `reindex` without
   filling. Same calendar as the modelling series, gaps left open.
2. **`_persistence_for` extracted**, with the precedence spelled out: the artifact's
   `origin_value`; else `truth.iloc[pos − int(row["horizon"])]`; else the legacy
   `horizon_steps` for a row carrying neither. It returns the recomputed value alongside the
   one it used, so the two can be cross-checked.
3. **`persistence_source` added to the scorecard** — `"artifact: origin_value"` or
   `"recomputed: …"` — so which ruler was used is in the record rather than in folklore.
4. **`score_published` reports `baseline_disagreements`**: rows where the artifact's
   `origin_value` and the actuals at `target_date − h` differ beyond a float round-trip.

**The decision worth flagging.** My first implementation *raised* on that disagreement and
kept the row out of the scorecard. I reverted it. Two reasons. It broke
`test_score_published_reports_future_dates_as_pending_not_as_zero`, whose fixture publishes an
`origin_value` of 1e8 against real canonical actuals — the guard was right and the fixture is
synthetic, but that is a signal about how easily it fires. More importantly, a legitimate
**data revision** would have silently emptied the client's whole track record: every row of
the affected issue would stop scoring, and the page would say "nothing scoreable yet". The
number that was *published* is the comparator the forecast was committed against, so the row
still scores against it, and the divergence is reported instead. The strict equality the
prompt asked for is enforced where this project already enforces its one-ruler rule — in a
test (`test_the_scored_ruler_equals_the_artifacts_origin_value`), the same way
`test_published_baseline_is_shared.py` enforces it for the DEV ruler.

A second thing my own test caught: I first set the agreement tolerance to `rtol=1e-6`, which
on a 46.5M figure accepts a **46-lari** divergence. That is a margin of acceptable
disagreement, and for a quantity that agrees to 0.000000 there is no such thing. It is now
`rtol=1e-9, atol=0.01` — a float64-through-CSV round trip and nothing more.

---

## 6. Tests

`backend/tests/test_realized_scoring_correctness.py`, 12 tests:

* **Defect 1** — one issue yields one ruler across all five horizons; the scorecard's
  `persistence_pred` equals the artifact's `origin_value` to tolerance; the live committed
  issue really does carry one `origin_value` and one `origin_date` per target (the premise the
  fix rests on); a row without `origin_value` falls back to **its own** h, not 5; the
  monkeypatched pre-fix comparator produces a demonstrably different number.
* **Defect 2** — a missing flow day is pending, not a zero; a missing stock day is not
  forward-filled; the scoring series and the modelling series differ *in the way that matters*
  (stated as a difference, so it survives either being rewritten).
* **Both** — a mutation test that reintroduces the bug and asserts the suite would catch it.
* Plus the disagreement report, the tolerance itself, and `persistence_source` being recorded.

---

## 7. Suites

```
AI4CM        backend/tests   687 passed → 699 passed   EXIT=0   (4:39)
AI4CM-agent  tests           269 passed                EXIT=0   (untouched by this change)
```

687 + 12 new = 699. The pre-existing 687 all still pass: nothing in this change altered a
number any other test asserts, which is the expected shape for a fix to a path that only
`test_published_forecasts.py` touched and only through its public behaviour.

---

## 8. Verdict

**P0.1 — fixed.** The ruler is read from the artifact, and the fallback reads the row's own
horizon. 12 of 15 realized skills and all three per-target aggregates were wrong; the Revenues
headline moves from −3.39% to +35.79% on the same forecast and the same actuals.

**P0.2 — fixed.** Scoring reads the raw actuals. `TruthNotAvailable` now fires for a flow
inside the data range, which it structurally could not do before, and the stock's forward-fill
is gone with it — that half was not in the diagnostic's write-up and is the same defect.

**What this does not fix.** The correction is demonstrated against synthetic actuals, because
real ones for 2025-08-07 onward do not exist yet. The arithmetic is exact and the ruler is now
read from the artifact, so the corrected figures will hold when real actuals arrive — but no
client-facing accuracy number has been *validated* against reality by this session, and none
should be quoted from the numbers above.

---

## 9. Outstanding

1. **P1 and P2 are untouched** — new data landing inside the sealed TEST window
   (`evaluation_windows.py:295`), no champion reselection path, MASE logged but not gated, the
   near-vacuous `vs_ruler > 0%` threshold, and the uncalibrated 1.50 sentinel line. See §5 of
   the diagnostic; the priority order there still stands.
2. **A gap in the client's upload is still indistinguishable from a public holiday** at the
   *modelling* layer. This session separated scoring from that fill, so a gap no longer
   fabricates a scored zero — but `b_ml_pipeline.to_business_index` still turns a
   data-collection gap into a training observation of 0.0, and nothing checks that a new
   file's rows are dense. Scoring is now honest about gaps; fitting is not yet.
3. **`baseline_disagreements` is reported but nothing displays it.** The Forecast page shows
   scored rows and pending rows; a row scored against a ruler that no longer matches the
   actuals looks identical to any other. It should be surfaced there.
4. **The one-ruler rule now has three enforcement points** (`test_unified_baseline`,
   `test_published_baseline_is_shared`, and the new
   `test_the_scored_ruler_equals_the_artifacts_origin_value`) and no single place that states
   it. Worth one short doc rather than a fourth test the next time this recurs.

---

## 10. Addendum: the 2026-08-12 reference artifact

Committed in the same session, separately from the scoring fix because it is evidence rather than
code.

`backend/forecast_runs/2026-08-12/` is a real `mode: production` run over `State budget balance`
(families A_STAT, B_ML, E_QUANTILE). It matters for three reasons:

* It **carries all four fields** — `run_id: 2026-08-12`, `schema_version: 2`,
  `data_file: master_daily_clean_treasury.csv`, `client_framing` present with the full
  `model_composition` block. So the writer changes reach a genuinely fresh run, not only the
  regenerated 2026-08-04 summary.
* It **passes the artifact contract with 0 errors** (7 warnings, all previously catalogued: the
  derived baseline row absent from `predictions_long`, the decorated join key, empty
  `sMAPE`/`MAPE`). `2026-08-04` still carries 3 errors from a pre-fix a_stat leaderboard, so this
  is the first contract-clean summary and the better thing to point a consumer at.
* It **settles the a_stat regression question by demonstration.** Same writer, same columns,
  different output, because this run was produced after the fix:

```
  2026-08-04: identity fully populated=False   RMSE populated=False
  2026-08-12: identity fully populated=True    RMSE populated=True
```

Only `SUMMARY.json` and `SUMMARY.txt` are tracked; the row-level artifacts stay gitignored because
they carry Treasury predictions. Two tests make the reference status enforceable rather than a claim
in a commit message:

* `test_the_reference_summary_is_contract_clean_and_carries_every_field` validates the summary
  **under `strict=True`** and requires zero findings, plus each of the four fields and an empty
  `promoted_outside_champion_pool`. It deliberately validates the summary alone, because that is
  what a clone actually has.
* `test_the_reference_a_stat_leaderboard_is_fully_identified` checks the identity columns and RMSE,
  and **skips** where the row-level CSVs are absent — so it is informative on the machine that made
  the run and silent everywhere else.
