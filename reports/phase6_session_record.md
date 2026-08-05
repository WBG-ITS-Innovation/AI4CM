# Phase 2 — One-Ruler Verification Session Record

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `87fc971` (17 ahead of `origin/main`)
**Test suite:** 272 → **273** passing
**Canonical data SHA-256:** `0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`
**TEST (2025) reads: 0**
**PR #24** — *Phase 2a — unified yardstick + provenance* — opened draft, now **ready for review** (not merged)

---

## 1 · The instruction

> Resume from reports/phase5_session_record.md §7. Verify premises first (model/excellence @ 94db732,
> suite = 272, TEST reads = 0, data SHA 0b009fd0…, origin/main @ 4a925ac with PR #23 merged); if reality
> disagrees, say so and stop.
>
> Step 0: update reports/HANDOFF.md with the phase-5 commits (resume point → 1f). Push model/excellence
> with --force-with-lease (it was rebased). Open a DRAFT pull request model/excellence → main titled
> "Phase 2a — unified yardstick + provenance", body linking the phase 3–5 session records; record the PR
> number. It stays draft until item 1f's verification passes; do not merge it yourself.
>
> Standing note: Treasury's answer on negative Revenues is outstanding. Proceed under Q1's interim policy
> (legitimate signed flows) and record that status honestly wherever the topic is cited.
>
> Then, one commit per item, stopping cleanly with a session record if context runs low — never start an
> item you cannot finish:
>
> 1. Item 1f — the backtest re-run and one-ruler verification, on the canonical data, for Revenues,
>    Expenditure, and State budget balance across all four families (C_DL included via its pin; it stays
>    parked otherwise per Q6). Verification: for each target, all four families publish literally one
>    persistence number from the shared function — for Revenues on the 2025 window definition expect
>    83,534,152.85; record the analogous single number for the other two targets. New-vs-old tier table
>    in the commit message. Report per target and per family: skill vs the unified ruler, sentinel ratio,
>    and per-magnitude-tercile coverage — Expenditure's first honest numbers included. Mark the draft PR
>    ready-for-review in the description once this verification is green (still do not merge).
> 2. Item 2 (ground rule 2): experiments/log.csv + one JSON per run with exactly the Phase-2 brief
>    columns … Re-run the two 15fb6ee DEV configurations under the logger, marked as reproductions. From
>    here on, never report an unlogged number.
> 3. Item 3 (workstream 1): L1 objectives for B_ML … summarize in reports/ws1_objectives.md.
> 4. Only if context comfortably allows: begin workstream 2 …
>
> TEST stays sealed. End with reports/phase6_session_record.md in the established format.

---

## 2 · Premise verification

All held. Two expected differences:

| Premise | Measured |
|---|---|
| `model/excellence` @ `94db732` | **`e7b3159`** — the phase-5 record commit landed after it |
| push with `--force-with-lease` needed | **already pushed** last session after the SSL retry; 0 unpushed commits |
| suite 272 · TEST reads 0 · data SHA `0b009fd0…` · `origin/main` `4a925ac` · PR #23 MERGED | all ✅ |

---

## 3 · Step 0

`reports/HANDOFF.md` updated (`e413cd1`): phase-5 commits added, resume point → 1f, three more register
entries struck through, and OQ1 reworded to state that T1 has been **sent** and the answer is
**outstanding**, with Q1's interim policy named.

**PR #24** opened as draft, linking the phase 3–5 records, then marked **ready for review** once 1f went
green. Not merged.

---

## 4 · Item 1f — the one-ruler verification (`87fc971`)

### Result: GREEN

```
target                    unified ruler (2025 window, h=5 business days)
Revenues                              83,534,152.85   <- as predicted
Expenditure                           83,839,124.43
State budget balance                 189,930,653.98

all four families, all three targets: n=156, 2025-01-01..2025-08-06
spread across families: 0.000000
```

12 pipeline runs (3 targets × 4 families), every one pinned to the data SHA via
`AI4CM_EXPECTED_DATA_SHA256`.

### Two defects only a cross-family comparison could surface

The first run reported **DISAGREE** on all three targets. That is the verification doing its job.

1. **C_DL published float32 truth and origin values.** At 1e8 magnitudes float32 rounds to ~8 units, so
   its baseline differed in the last cents (`83,534,152.28` vs `.85`). Fixed by widening `origin_value`
   to float64 *and* reading `y_true` from the float64 **source series** at the label dates — casting the
   float32 tensor back would not recover lost precision. The model still trains on float32.

2. **E_QUANTILE pinned on ORIGIN dates** while the other three define their window by **target** date,
   so its first target landed h steps late (2025-01-08, n=151). It also tiled fold blocks **backward**
   from the series end, dropping the front remainder whenever the window length was not a multiple of the
   horizon (156 → 150). Both fixed: pinned folds tile **forward** from `eval_start`, and `eval_start`
   resolves against target dates. On Expenditure the two together cost a **5.8M** baseline difference.

### Tier table — skill vs the unified ruler

**Revenues** (old ruler 60,976,736 → now 83,534,152)

| Model | Old | Now |
|---|---:|---:|
| E_QUANTILE / ResidualRF | 49.54% | **48.12%** |
| E_QUANTILE / GBQuantile | 48.45% | 47.00% |
| B_ML / RandomForest | 36.75% | 43.39% |
| B_ML / XGBoost | 36.77% | 33.08% |
| A_STAT / ETS | 27.51% | 31.25% |
| C_DL / TRANSFORMER | +10.84%\* | **−12.74%** |

\* a 2019–2025 average; on the shared window it was −5.19%.

**Expenditure — first honest numbers for a priority target**

| Model | Skill |
|---|---:|
| E_QUANTILE / GBQuantile | **33.22%** |
| B_ML / ExtraTrees | 32.21% |
| A_STAT / ETS | 31.18% |
| C_DL / TRANSFORMER | 14.35% |
| C_DL / MLP | −18.79% |

**State budget balance** (E_QUANTILE via its new delta path)

| Model | Skill |
|---|---:|
| E_QUANTILE / ResidualRF | **31.66%** |
| E_QUANTILE / GBQuantile | 26.74% |
| B_ML / Lasso | 6.46% |
| A_STAT / ETS | −0.00% |
| C_DL / all five | −759.73% |

### Three findings worth carrying forward

**Skill and signal point in opposite directions.** Sentinel ratios (threshold 1.5): Revenues **1.14**,
Expenditure **1.05**, State budget balance **4.27**. The two flows are *below* threshold, so 30–45% skill
there is regression to the mean against a spiky baseline, not forecasting. The stock target is the one
with genuine signal — and the *lowest* skill (6.46%). Reporting either number alone would mislead.

**Every interval model fails on the big days.** Per-magnitude-tercile coverage (low/mid/high):

| Model | Revenues | Expenditure | Stock |
|---|---|---|---|
| E_QUANTILE / ResidualRF | 75/79/**52** | 37/96/**40** | 58/56/**46** |
| E_QUANTILE / GBQuantile | 79/92/**58** | 67/100/**58** | 85/73/75 |
| B_ML / RandomForest | 98/96/**65** | 96/100/**56** | — |

Nominal-80% bands cover **40–58% of the largest days** — the days a cash buffer exists for. Confirms
review §3.2 on the honest data. CQR (workstream 7) is the fix.

**C_DL on the stock target is broken, not merely bad.** All five architectures report MAE
`1,632,88x,xxx` — identical to four significant figures against a 189,930,654 ruler. Identical outputs
across five architectures means collapse to a constant: the original C-1 target-scaling failure recurring
for levels, since `target_transform="auto"` resolves to `"none"` for a stock. C_DL stays parked (Q6);
recorded so it is not read as a modelling result.

### One existing test changed intentionally

`test_time_folds_eval_start_respects_min_train` expected an end-anchored grid. Forward tiling starts at
the earliest **legal** index instead (31 vs 35 in that fixture), honouring the same `min_train` floor
while covering 19 rows of the window rather than 15 — strictly more evaluation data. A new test pins the
property that actually mattered: a window whose length is not a multiple of the horizon must be covered
end to end.

---

## 5 · What remains

**Item 1 is COMPLETE.** The unified yardstick holds: one persistence number per target, from the shared
function, across all four families.

| # | Task | Notes |
|---|---|---|
| **2** | Ground rule 2 — `experiments/log.csv` + per-run JSON with the Phase-2 brief columns; re-run the two `15fb6ee` DEV configs as marked reproductions | **Not started.** "Never report an unlogged number" takes effect after this. Every figure in §4 is currently unlogged |
| 3 | Workstream 1 — L1 objectives for B_ML; `reports/ws1_objectives.md` | Not started |
| 4 | Workstream 2 — LightGBM quantile + Optuna | Not started |

### Open, carried forward

- **Treasury T1 (negative `Revenues`) — answer OUTSTANDING.** All §4 numbers stand under Q1's interim
  policy that they are legitimate signed flows. If overturned, the Revenues and Expenditure figures need
  revisiting, and `log1p` becomes available to workstream 4.
- **C_DL stock-target collapse** — new this session. Parked per Q6, but it is a scaling bug, not a
  capability limit, and would need the C-1 fix extended to levels if C_DL is ever unparked.
- **Top-tercile interval coverage** — 40–58% against a nominal 80%. Workstream 7's CQR and the
  conditional-coverage gate are the planned fix.
- **PR #24 is ready for review, not merged.**
- `reports/HANDOFF.md` needs this session's three commits before the next handoff.

---

## 6 · Reproduction

```bash
git checkout model/excellence            # 87fc971
./backend/.venv/bin/python -m pytest -q  # expect 273 passed
cat experiments/test_access.log 2>/dev/null || echo "TEST reads: 0"
```

The 12 backtest runs are reproduced by invoking each runner with
`TG_CADENCE=Daily TG_HORIZON=5 TG_DATE_COL=date`,
`TG_DATA_PATH=data/processed/master_daily_clean_treasury.csv`,
`AI4CM_EXPECTED_DATA_SHA256=0b009fd031ad3fa0dbdb35fd9a3733144b04a8e9d37fa4298499e073265361f1`, and per
family: A_STAT `{"folds":1,"min_train_years":4}` with `TG_MODEL_FILTER=ETS`; B_ML
`{"folds":1,"min_train_years":4}`; E_QUANTILE `{"eval_start":"2025-01-01","min_train_years":4}`; C_DL
`{"quick_mode":true,"max_epochs":3,"min_train_years":4}` via `run_c_dl_quick_univariate.py`.

The one-ruler check is `compute_persistence_baseline()` from `forecast_integrity` applied to each
family's `predictions_long.csv`, restricted to one model and dropping NaN `origin_value`/`y_true`.

**None of the §4 figures is in `experiments/log.csv`** — ground rule 2 is item 2, and the "never report
an unlogged number" rule takes effect from workstream 1 onward. These are recorded here and in
`87fc971`'s commit message instead.
