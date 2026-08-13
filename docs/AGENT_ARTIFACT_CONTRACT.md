# The Agent artifact contract

**As the pipeline produces it on 2026-08-12.** Branch `model/excellence`.

The AI4CM Agent (separate repo, `feat/lab-door`) reads `SUMMARY.json`, the per-family
`leaderboard.csv` / `predictions_long.csv` / `metrics_long.csv`, and
`forecasts/published/<issue_date>/`. These are a **published interface**, not incidental output: a
malformed field here is not an internal detail, it is a wrong answer given to someone.

This document describes what the code writes **today**, verified by reading the artifacts rather
than the writers. Enforced by [`backend/artifact_validation.py`](../backend/artifact_validation.py);
run it with

```
./backend/.venv/bin/python backend/artifact_validation.py <run_dir> [--strict] [--published-root DIR]
```

It also runs automatically at the end of `scripts/daily_summary.py` and returns exit code **2**
rather than publishing a failing artifact.

---

## 0. Read this first: what absence means

**The single most important thing for a consumer is telling "absent because not applicable" from
"absent because something failed". In this contract that is expressible in some places and not in
others, and the difference is whether a companion field exists.**

**Expressible — a companion field states the reason:**

| Field | How absence is explained |
|---|---|
| `gate_passed` | Tri-state. `true` / `false` / **`null` = never verified**. `null` must never render as a pass |
| `alignment_ok` | Absent ⇒ look at `alignment_checked`. `false` + `alignment_check_error` says the check could not run. `alignment_ok` is **never** written unconditionally |
| `coverage_p10_p90` | Absent ⇒ look at `legacy_coverage_key_omitted`, which carries the full reason (the fitted alphas no longer describe an 80% interval). Silence beats a mislabelled number |
| estimator blobs | `estimators/manifest.json` absent ⇒ never retained (issue predates retention). Present with `retention.pruned = true` + `pruned_at` ⇒ deliberately pruned. Either way the loader raises `EstimatorMissing` with which case it is |

**Not expressible — absence is bare, and a consumer cannot tell why:**

| Field | Why not |
|---|---|
| `skill_pct` = `"n/a (not produced)"` | One marker for several causes: the family produced no output, its integrity report had no `skill_pct`, or the value was not numeric. A consumer cannot separate "this family does not compute skill" from "the computation failed" |
| Any all-null CSV column (`RMSE`, `y_lo`/`y_hi`, `MAE_skill_vs_Ops`) | An empty cell asserts nothing. `y_lo`/`y_hi` are legitimately empty for models with no native intervals; `MAE_skill_vs_Ops` is empty because the ops baseline does not apply to stock targets — but both look identical to a failure |
| `coverage_nominal` absent | Means the run predates the field. A consumer can now tell from `schema_version` on `2026-08-04` onward; the two earlier runs still carry none (§1) |

**Recommended pattern, and the one this project should converge on:** where a field may legitimately
be absent, write a *companion field naming the reason*, as `alignment_checked` and
`legacy_coverage_key_omitted` already do. A bare null is not a contract.

**Until then, a consumer should treat a missing or `n/a` value as UNKNOWN — never as zero, never as
a pass, and never as a failure.**

---

## 1. `SUMMARY.json`

Written by `scripts/daily_summary.py`. One object.

### Top level

| Field | Type | Presence | If absent |
|---|---|---|---|
| `run_id` | string | since v2 (`2026-08-04` onward; the two earlier runs predate it) | The run cannot be identified. Fall back to the directory name |
| `schema_version` | int (`2`) | same as above | Assume version 1 and expect every field marked "since v2" to be missing |
| `run_date` | string `YYYY-MM-DD` | always | Malformed — reject |
| `target` | string | always | Malformed — reject |
| `cadence` | string | always | Malformed — reject |
| `horizon` | **string**, e.g. `"5"` | always | Malformed — reject. Note the type: it is *not* an int |
| `data_file` | string — the input file's **bare name**, e.g. `"master_daily_clean_treasury.csv"` | since v2 | The run cannot say which dataset produced it. Do **not** substitute a plausible file; render "not recorded" |
| `families` | array of objects | always, non-empty | Malformed — reject |
| `overall` | object | always | Recompute from `families`; it is derived and can contradict them |
| `mode` | `"production"` \| `"backtest"` | always | Assume production |
| `freshness` | object: `line` (string), `stale` (bool), `backtest` (bool) | always | Treat staleness as unknown |
| `client_framing` | string — the composition sentence a client may be shown verbatim | since v2; absent ⇒ see the next row | Do **not** assemble your own from `model_composition`. Report "composition not recorded" |
| `client_framing_unavailable_reason` | string | **only** when the composition could not be derived | Explains the absence above. Present means the model *catalogue* could not be read (e.g. a missing modelling library) — the run itself is unaffected |
| `model_composition` | object (below) | with `client_framing` | Counts unavailable |

`model_composition` carries the numbers the sentence was derived from, so a consumer can requote it
or recompute without re-deriving the categories itself:

| Field | Type | Meaning |
|---|---|---|
| `counts` | `{category: int}` | Five categories: machine-learning models, deep-learning models, statistical models, quantile methods, reference baselines |
| `members` | `{category: [name]}` | Every counted model named |
| `champion_pool` | `[name]` | The models a **registry recipe** may promote as its `point_model` — what an official published forecast is selected from (currently the 13 machine-learning models) |
| `champion_pool_category` | string | Which category that pool is |
| `daily_best_model_families` | `[family]` | Every family this file writes a per-family `best_model` for. **The Agent ranks across these, not across `champion_pool`** — conflating the two is how a true sentence becomes a wrong one |
| `promoted_by_registry` | `[name]` | Distinct models the live recipes actually promote. Three recipes currently name **two** models (Revenues and Expenditure share `LightGBM_L1`), so this is not a recipe count |
| `promoted_outside_champion_pool` | `[name]` | Integrity cross-check. **Non-empty means the eligible pool a client was told about is wrong** |

**Never sum `counts` into a headline number.** The entries are not one kind of thing: three are
reference baselines and three are interval methods, and summing presents the ruler as a rival to the
models measured against it. See `reports/gate_audit.md` §4.

> **`client_framing` — the same gap as `data_file`, one layer up.** `model_reference.client_framing()`
> and `composition()` existed, were tested, and were written by nothing, so no artifact carried the
> composition and the Agent correctly reported "composition not recorded" on every run. A derived
> sentence nobody publishes is not a contract field. Written 2026-08-12 together with the counts it
> came from, and absent only with a companion reason (§0's pattern).

> **`data_file` — why it is written rather than dropped (review C1).** It was absent from the
> JSON while `SUMMARY.txt` printed `Data file: <name>` on line 4, so two artifacts of the same
> run disagreed about whether the input was knowable, and a consumer reaching for it got `None`
> and rendered the word. The writer already had the value — `daily_summary.py` takes
> `--data-file` and uses it for the freshness check — so the field was one line away the whole
> time. It carries the **bare name only**: the digest, row count and date range live in
> `provenance.json` (§7), and a second copy here would be a second place for them to drift. The
> absolute path is deliberately excluded — it is machine-specific and does not belong in a
> published interface.

> **Resolved 2026-08-12.** The three runs carried **neither `run_id` nor `schema_version`**, despite
> `test_artifact_contract.py` asserting both — that test greps the *writer's source* and cannot see
> what is on disk. `2026-08-04`'s `SUMMARY.json` has since been regenerated by the current writer and
> now carries `run_id`, `schema_version`, `data_file`, `client_framing` and `model_composition`.
> Every family verdict and the whole `overall` block are identical: it is the same run re-summarised
> from the same family artifacts, not a new run. `2026-07-29` and `2026-07-30` still predate the
> writer and remain readable with warnings.
>
> **`backend/forecast_runs/` was also entirely gitignored, so no run artifact existed in a clone at
> all** — a bigger blocker than any single field, because a consumer had nothing to read. `SUMMARY.json`
> and `SUMMARY.txt` are now tracked (they are aggregate-only: model names, MAEs, skill percentages,
> gate verdicts, and the data file's name). Every row-level artifact stays ignored. Note the pattern
> in `.gitignore`: the directory's **contents** are excluded, not the directory, because git never
> descends into an excluded directory and the negations would otherwise do nothing.

### `families[]`

| Field | Type | Presence | If absent |
|---|---|---|---|
| `name` | string, `A_STAT` \| `B_ML` \| `C_DL` \| `E_QUANTILE` | always | Malformed — reject |
| `ok` | bool | always | Assume false |
| `models` | string, comma-separated **display** list (may contain emoji) | always | Read the leaderboard instead |
| `best_model` | string, `"<Model> (MAE 1,234,567)"` — the MAE is embedded in prose | always | Read the leaderboard instead |
| `best_model_display` | string; carries `WITHHELD — <reasons>` when the gate failed | always | Fall back to `best_model` **and** `gate_passed` |
| `skill_pct` | string `"27.51%"` **or** `"n/a (not produced)"` | always | Treat as UNKNOWN |
| `run_status` | `"SUCCESS"` \| `"FAILED_QUALITY"` | always | Treat as UNKNOWN |
| `integrity_verified` | bool — whether an integrity report was *found*, not whether it passed | always | Assume false |
| `gate_passed` | **bool or null** (tri-state) | always | Treat as `null` = never verified |
| `gate_reasons` | array of strings; non-empty iff `gate_passed is false` | always | If `gate_passed` is false with no reasons, the artifact is inconsistent |
| `gate_source` | string — which reader produced the verdict *(since v2)* | conditional | Verdict provenance unknown |
| `gate_published_by_family` | what the family itself published *(since v2)* | conditional | Cannot cross-check the verdict |
| `n_prediction_rows` | int *(since v2)* | conditional | Cannot tell a baseline over zero rows from a real one |
| `baseline_without_predictions` | bool *(since v2)* | conditional | Same |
| `leakage_flag` | bool | always | Assume unknown, not false |
| `shift_flag` | bool | always | Assume unknown, not false |

**Contract hazards a consumer must handle:**

* **`skill_pct` and `horizon` are strings.** Parse, do not cast blindly; `skill_pct` may be a
  sentinel.
* **`best_model` embeds a number in prose.** Splitting on `" ("` is the only way to recover the
  name, and the MAE inside it is display-formatted with thousands separators.
* **`models` and `best_model` may contain decoration.** B_ML labels its baseline
  `⚡ Persistence (baseline)`.
* **`gate_reasons` is the only place the four verdicts are distinguished** — leakage, no signal,
  persistence-like, and (since item 5) `intervals miscalibrated`. Do not infer the cause from
  `run_status`.
* **`best_model` is a per-family best, not "the champion".** The word means two different things
  in this project and they are not the same size:
  * **The registry champion** — `registry/recipes.json` promotes one `point_model` per target,
    selected from the **13 machine-learning models**. That is what an official published forward
    forecast uses, and what "the champion-eligible pool is 13" refers to.
  * **`families[].best_model`** — written here for **every** family that produced a leaderboard:
    A_STAT, B_ML, C_DL and E_QUANTILE alike. A consumer that ranks families by `skill_pct` to
    pick a winner is choosing across **four families**, not across the registry's 13.

  Both are legitimate; conflating them is not. A consumer presenting a `families[].best_model`
  must not call it the champion recipe, and must not imply it was selected from the pool the
  registry selects from. `backend/model_reference.composition()` derives both, and
  `test_model_composition.py` fails if a recipe ever promotes a model from outside the
  machine-learning pool.

### `overall`

`families_requested`, `families_ok`, `families_gate_passed`, `leakage_flags`, `shift_flags`,
`quality_gate_failures` — all ints, all always present, all **derived**. The validator recomputes
each from `families` and errors on any disagreement.

---

## 2. `<family>/leaderboard.csv`

**There is no common schema. Four families write three different ones.**

| Family | Path | Columns |
|---|---|---|
| `a_stat` | `a_stat/leaderboard.csv` | `target, horizon, cadence, model, MAE, RMSE, rank` |
| `b_ml` | `b_ml/leaderboard.csv` | `target, horizon, model, MAE, rank` |
| `c_dl` | **`c_dl/daily/leaderboard.csv`** | `target, horizon, model, MAE, rank` — **the same schema as `b_ml`** |
| `e_quantile` | `e_quantile/leaderboard.csv` | `model, pinball_q10, pinball_q50, pinball_q90, coverage_p10_p90, MAE` |

Only **`model`** is guaranteed. `rank` is absent from `e_quantile`; `target` and `horizon` are
absent from it entirely, so an `e_quantile` leaderboard **cannot be keyed by target** — the target
must come from `SUMMARY.json` or the directory.

**`c_dl` — two things a consumer must know.** Its artifacts are **one level deeper**: everything
is under `c_dl/<cadence>/`, e.g. `c_dl/daily/leaderboard.csv`, not `c_dl/leaderboard.csv`. Glob
recursively or the family reads as absent. And beside the leaderboard sits a **differently-shaped
file whose name also begins `leaderboard`**:

```
c_dl/daily/leaderboard.csv                 <- the leaderboard. b_ml's schema.
c_dl/daily/leaderboard_<Target>_h<h>.csv   <- NOT a leaderboard. See below.
```

`leaderboard_<Target>_h<h>.csv` carries `model, MAE, target, horizon, cadence, RMSE, sMAPE, MAPE,
R2, PI_coverage@90, PI_width@90, Monthly_TOL10_Accuracy, MAE_skill_vs_Ops` — that is the **wide
metrics table** of §4, written per target and horizon, under a name that begins with the word
"leaderboard". It has no `rank`. **Match on the exact filename `leaderboard.csv`; a prefix match
picks up a metrics table and reads its columns as a ranking.**

| Field | Type | Presence | If absent |
|---|---|---|---|
| `model` | string | always, non-null | Malformed — reject the file |
| `MAE` | float | `a_stat`, `b_ml`, `c_dl` always; **`e_quantile` from 2026-07-30 only** | Treat as UNKNOWN. The committed `2026-07-29` `e_quantile` leaderboard has no `MAE` column at all |
| `rank` | int — **`a_stat` 0-based, `b_ml` 0-based, `c_dl` 1-based** | conditional | Sort by MAE ascending. Never compare a `rank` across families |
| `RMSE` | float | `a_stat` only | Not computed for this family |
| `pinball_q*` | float | `e_quantile` only | Not applicable |
| `coverage_p10_p90` | float in [0, 1] | `e_quantile` only, and only when both interval quantiles were produced | See §0 |

**Not unified, deliberately.** `c_dl`'s leaderboard already *is* `b_ml`'s schema, so there is
nothing to unify in the columns. The remaining differences — the `daily/` nesting, the 1-based
`rank`, and the `leaderboard_*` sibling — cannot be changed without rewriting three committed
artifacts for cosmetic gain, which §8 declines to do for the same reason it declines to
regenerate A_STAT. They are documented here instead, which is what a consumer actually needs.

**Fixed in item 6:** A_STAT's leaderboard populated `target` / `horizon` / `cadence` on the
*baseline* row only and left them blank on the winning model, because `lb` was built with
`groupby("model")["MAE"]`, which dropped them, while the persistence row was concatenated with them.
"Which model won for target X" was unanswerable from the file. `RMSE` was dropped the same way,
which is why it read as an all-null column despite being computed in `metrics_long.csv`.

**Known, not fixed:** the persistence baseline row appears in the leaderboard but has **no rows in
`predictions_long.csv`** — it is derived from `origin_value`, not predicted. A consumer joining the
two on `model` silently loses it. Reported as a WARNING.

---

## 3. `<family>/predictions_long.csv`

The row-level artifact. Required: `origin_date`, `target_date`, `y_true`, `model`.

| Field | Type | Presence | If absent |
|---|---|---|---|
| `origin_date` | date | always | Malformed — the row cannot be placed in time |
| `target_date` | date, strictly **after** `origin_date` | always | Malformed |
| `origin_value` | float — **this IS the h-step persistence prediction** | always in practice | The benchmark cannot be derived; do not substitute another baseline |
| `y_true` | float | always (backtest); **never** in a published forward forecast | Absent legitimately for forward dates |
| `y_pred` | float | `a_stat`, `b_ml`; in `e_quantile` it is an alias of `yhat_p50` | Treat as UNKNOWN |
| `y_lo`, `y_hi` | float | present as columns, **all-null for models with no native intervals** | Cannot distinguish "no native PI" from "PI failed" — see §0 |
| `yhat_p10/p50/p90` | float | `e_quantile` only | Not applicable |
| `model` | string | always | Malformed |
| `horizon` | int | always | Take from `SUMMARY.json` |
| `split_id`, `fold`, `defn_variant`, `horizon_note`, `cadence`, `target` | mixed | family-dependent | Ignore |

`origin_date >= target_date` on any row is an **error**: the model would have been predicting a date
it could already see.

---

## 4. `<family>/metrics_long.csv`

**Two incompatible shapes share one filename.** A consumer must detect which it has.

**LONG** (`e_quantile`) — `model, fold, metric, quantile, value`. One row per metric per fold.

**WIDE** (`a_stat`, `b_ml`) — one column per metric: `target, horizon, model, MAE, RMSE, sMAPE,
MAPE, R2, PI_coverage@90, PI_width@90, Monthly_TOL10_Accuracy, MAE_skill_vs_Ops` (b_ml);
`target, horizon, cadence, model, MAE, RMSE` (a_stat).

Detect with `{"metric", "value"} <= set(columns)` → long, else wide. **The filename says "long" and
two of three families are wide.**

| Field | Type | Presence | If absent |
|---|---|---|---|
| `metric` / `value` | string / float | long form only | It is the wide form |
| `quantile` | float — **the nominal coverage level on a `coverage_*` row** since item 5; blank on older artifacts | long form | The level is only inferable from the metric *name* |
| `PI_coverage@90` | float | b_ml | The interval's level is in the **column name**, with no `coverage_nominal` beside it — the same defect audit field #2 fixed for E_QUANTILE |
| `MAE_skill_vs_Ops` | float | b_ml, **all-null** | Not applicable to stock targets; indistinguishable from a failure |

---

## 5. Coverage fields — the level is data (since item 5)

Written by `e_quantile_daily_pipeline.emit_coverage()`, one implementation for all three artifacts.

| Field | Type | Presence | Meaning |
|---|---|---|---|
| `coverage_key` | string | whenever coverage is measurable | The key the value is published under, derived from the fitted alphas |
| `coverage_p<lo>_p<hi>` | float in [0, 1] | whenever measurable | The measured coverage. The key names the actual quantiles |
| `coverage_nominal` | float | whenever measurable | **The level the interval was fitted for.** Read this; do not infer from the key name |
| `coverage_lower_quantile` / `coverage_upper_quantile` | float | whenever measurable | The alphas. `upper - lower` must equal `coverage_nominal` |
| `coverage_band` | `[lo, hi]` | whenever measurable | The accepted band, `nominal ± 0.10` |
| `coverage_p10_p90` | float | **only when the fitted alphas really are 0.10/0.90** | The legacy key. Its absence is explained by the next field |
| `legacy_coverage_key_omitted` | string | only when the legacy key was suppressed | The full reason. Its presence means: the alphas changed, so an 80%-labelled key would have been a mislabelled 90% number |
| `coverage_unavailable_reason` | string | only when fewer than two quantiles were configured | No interval exists to measure |

**A coverage value with no numeric `coverage_nominal` is an ERROR.** So is a `coverage_nominal` that
contradicts the recorded quantiles.

Point-model runs log **no coverage at all**. That is legitimate and correct: they produce no
intervals. Consumers render "not reported" — never 0, never "failed".

---

## 6. Integrity / alignment fields

Per-family `integrity_report.json` (and mirrored into `SUMMARY.json` via `gate_reasons`).

| Field | Type | Presence | If absent |
|---|---|---|---|
| `quality_gate_passed` | bool or null | canonical since X9 | Read `quality_gate_failed` (legacy, inverted) or `run_status`; **never treat absence as a pass** |
| `quality_gate_failed` | bool | legacy, kept in sync | Derived from the canonical key, never set independently |
| `run_status` | `"SUCCESS"` \| `"FAILED_QUALITY"` | always | UNKNOWN |
| `alignment_checked` | bool | C_DL | The check did not run |
| `alignment_ok` | bool | **only when `alignment_checked` is true** | Not checked — *not* "passed". Never written unconditionally (fixed 2026-08-10) |
| `n_misaligned`, `misaligned_examples` | int, list | with `alignment_ok` | — |
| `alignment_check_error` | string | only when the check raised | Why no verdict exists |
| `signal_detected` | bool or null | families that run the sentinel | Not measured — A_STAT does not run it; do not read absence as failure |
| `shuffled_to_normal_ratio` | float | with `signal_detected` | — |
| `leakage_warning` | bool, **always `false`** | always | This instrument measures signal, never leakage |
| `mae_persistence`, `mae_model` / `mae_p50` | float | usually | Skill cannot be reconciled |
| `skill_pct` | float | usually | UNKNOWN |
| `best_shift`, `shift_interpretation` | int, string | families that run the diagnostic | Not measured. A family writing neither has **no effective persistence-mimicry check at h=5** — see `reports/gate_audit.md` §2.6 |

**Consistency rule the validator enforces:** `skill_pct` must follow from `mae_persistence` and the
model MAE published beside it, within 0.75 pp. This is the check that would have caught the WS2
harness bug, where a non-canonical ruler inflated every logged skill while both MAEs sat in the same
record disagreeing with it.

---

## 7. `forecasts/published/<issue_date>/`

The immutable record of what was said before the truth existed. `<issue_date>` is the forecast
origin; a same-day re-issue takes `-r2`, `-r3`.

### `forecast.csv`

`target, horizon, origin_date, origin_value, target_date, p10, p50, p90, p50_quantile_model,
point_model, interval_model, n_train_rows, n_features, modelled_as, target_transform`

Required: `target, horizon, target_date, origin_date, p10, p50, p90`.

* **`y_true` must never be present.** A forecast issued before its truth existed cannot have one.
* `p10 <= p50 <= p90` on every row.
* `origin_value` is the persistence benchmark, carried flat across horizons from one origin.
* `modelled_as` is `"level"` or `"delta (level reconstructed)"` — required to interpret `p50`.

### `provenance.json`

`run_kind, generated_at_utc, data{name, sha256, n_rows, latest_data_date}, code{git_sha, git_branch,
git_dirty, git_dirty_files}, environment{python, platform, packages{}}, calendar_version, quantiles,
horizons, recipes, test_window_touched, notes`

`code.git_dirty` — the committed `2025-08-06` issue was generated from a **dirty tree**
(`git_dirty_files: 20`), which limits how exactly it can be reproduced regardless of estimators.

### `gates.json`

Keyed by `recipe_id`. Each: `target`, `status`, `approved_by`, and `gates{<name>: {passed, name,
measured, threshold, reason_plain, corroboration}}`.

**`approved_by` is `null` on all three recipes. Nothing may render as approved while it is null.**
`measured` and `threshold` are numeric (F1); `reason_plain` is Treasury-facing prose.

### `manifest.json`

`issue_date, targets, horizons, target_dates, recipes, data_sha_at_issue, git_sha_at_issue,
calendar_version, test_window_touched, note`

### `estimators/manifest.json` — conditional by design

Tracked; the blobs beside it are **gitignored** (they embed Treasury training data). See
[`backend/estimator_store.py`](../backend/estimator_store.py).

| State | What a consumer sees |
|---|---|
| No `estimators/` directory | `EstimatorMissing: ... published without retaining its estimators, so it cannot be re-derived`. Legitimate — the issue predates retention |
| Manifest present, blobs present | Loadable, after a SHA-256 check and a version check |
| Manifest present, `retention.pruned = true` | `EstimatorMissing` naming the prune date. Deliberate |
| Manifest present, blobs gone, **no** prune marker | **ERROR — unexplained absence.** This is the one case that means something broke |
| A cloned repo | Blobs never present; the manifest digests still prove what was published |

Each entry: `fit_id`, `target`, `horizon`, `kind`, `file`, `sha256` (required), `bytes`,
`recipe_id`, `selection_run_id` + `selection_run_id_note`, `target_transform`, `n_train_rows`,
`n_features`, `feature_names`, `fiscal_groups`, `exog_blocks`, `origin_date`, `scrubbed`.

**`selection_run_id` is not this fit's id.** It identifies the DEV run that *chose* the recipe
(TRAIN ≤2023, scored on DEV 2024). The published fit uses all history through the issue date and has
no logged `run_id` — the forward path writes no row to `experiments/log.csv`. Use `fit_id`.

---

## 7b. What can be forecast — target eligibility and family capability

Not part of a written artifact: two functions a consumer calls directly, added because the Agent was
otherwise left inventing its own answers to "can this be forecast?".

### `forecast_modes.target_eligibility(data_path, horizon=5)`

`{column: Eligibility}` for all 41 candidate columns. `targets_available()` previously read one row
and returned every column, so a length check was impossible by construction; it now returns the
**eligible** ones by default, with `include_ineligible=True` for a page that greys out the rejects.

| Field | Type | Meaning |
|---|---|---|
| `eligible` | bool | Whether the column can be forecast **on the data** |
| `code` | string or `null` | `null` when eligible; otherwise `not_numeric`, `all_null`, `insufficient_history`, `unusable_date_index`. **Branch on this, not on the prose** |
| `reason` | string or `null` | Plain language a consumer may quote **verbatim**. Never assemble your own — an invented explanation shown to a treasury is worse than a blank |
| `n_usable` / `n_required` | int | Non-null observations against the threshold |
| `dtype`, `first_date`, `last_date` | string | What was measured |

The threshold is derived, not chosen: `DEFAULT_MIN_TRAIN (1008) + horizon + DEFAULT_EVAL_BLOCK (126)`
= **1,139** at h=5 — four years to train on, a horizon-sized embargo, and one complete six-month block
left to score against. Below it a number can be produced but not evaluated, which here is the same as
not having one.

`unusable_date_index` is **file-level**: an unparseable or duplicated date makes every column in the
file ineligible, because a horizon counted in index positions is ambiguous. `date_index_status()`
reports it separately so a consumer says "the file is unusable" rather than listing 41 identical
column failures.

**Measured today: all 41 columns are eligible** — the canonical file is fully dense (3,867 usable rows
everywhere, +2,728 above the threshold). The rejection paths are therefore exercised only by
synthetic fixtures in `test_target_eligibility.py`; a check that never fires looks like it works.

Eligibility is about the **data**. A target can be eligible and still have no recipe
(`recipe_status`) and no family that supports its kind (below).

### `family_capabilities.family_supports_target(family, target)`

| Field | Meaning |
|---|---|
| `supported` | A code path exists for this **kind** of series |
| `target_kind` | `"stock"` or `"flow"` |
| `stock_method` | `"delta"` (models the change, reconstructs the level) or `"level"` (forecasts the level directly) |
| `publishable` | `supported` **and** a registry recipe promotes this family for the target |
| `code` | `null`, `unknown_family`, `kind_unsupported`, or `supported_but_not_published` |
| `reason` / `evidence` / `notes` | Quotable prose, how the claim was established, and caveats |

**Supported is not approved.** Currently only **B_ML** is `publishable` for `State budget balance`.

**Correction, 2026-08-12.** `CHANGELOG.md` asserted under "Known gaps" that *"E_QUANTILE has no
stock-target path, so State budget balance cannot yet be forecast by the family"*. **That is false
and was measured to be false**: E_QUANTILE adds `y_lag_0`, models the delta and reconstructs the
level, exactly as B_ML and C_DL do. Run end to end against DEV 2024 (TEST untouched): 262
predictions, P50 MAE 168,565,455 against a persistence MAE of 242,653,025 — **skill 30.53%**,
coverage 65.6%, gate **FAILED on coverage**. Its sibling bullet in the same block ("still on a
calendar-day index") was also fixed long ago.

So the honest statement is *"the path exists, has no recipe, and its one DEV run fails the coverage
gate"* — which invites fixing the calibration, where *"unbuilt"* invited building it. The capability
is now machine-readable because a claim recorded only in prose drifts: nothing fails when it goes
stale, as that entry demonstrated.

**One latent inconsistency, recorded rather than left to a rename.** There are four `is_stock`
implementations. B_ML, C_DL and E_QUANTILE agree on `{"state budget balance", "balance", "t0"}`;
`run_a_stat._is_stock` uses `{"state budget balance", "balance", "net", "stock"}`. For a column named
`t0`, three families would model a delta and A_STAT a level. **None of the disputed names is a column
in the canonical file**, so it is latent — and latent is exactly when nobody checks.
`stock_alias_divergence()` returns the disagreement as data.

---

## 8. What the validator checks

| Class | Examples |
|---|---|
| **malformed** | missing file; unparseable JSON/CSV; required key or column absent; `gate_passed` not tri-state; `skill_pct` neither number nor marker; coverage outside [0, 1] |
| **incomplete** | `run_id` / `schema_version` absent; empty `families`; leaderboard with no rows; all-null metric column; estimator with no `sha256`; missing `provenance.json` |
| **inconsistent** | `gate_passed: false` with no reasons; `gate_passed: true` with reasons; `overall` disagreeing with `families`; duplicate family names; coverage with no nominal; nominal contradicting its quantiles; skill not reconciling with its MAEs; champion absent from the pool; `origin_date >= target_date`; partially-identified leaderboard; published `y_true`; crossed quantiles; blobs gone with no prune marker |

`ERROR` blocks publication. `WARNING` does not, and covers defects that are real but historical.
`--strict` promotes warnings to errors and is what a **new** run should meet.

### Current standing of the committed artifacts

`backend/forecast_runs/2026-08-04` — measured, not assumed: **3 errors, 7 warnings**.

`daily_summary.py` therefore exits **2** on this run.

#### Verdict: pre-existing CSV defect, not a regression (2026-08-12)

All three errors are one defect in one file: `a_stat/leaderboard.csv` has `target`, `horizon` and
`cadence` populated on 1 of 2 rows, so the winning model row is unidentified. The evidence that this
predates every recent change, rather than being caused by one:

| Check | Result |
|---|---|
| Which artifact holds the errors | **`a_stat/leaderboard.csv`, all 3.** Zero errors mention `SUMMARY.json` |
| When the CSV was written | `2026-08-04 15:33` |
| When the writer bug was fixed | `03ad619`, `2026-08-11 12:01` — **seven days later** |
| Errors before the 2026-08-12 SUMMARY regeneration | 3 |
| Errors after it | 3, with **byte-identical messages** |
| Warnings before → after | 10 → 7: three removed (`run_id`, `schema_version`, `data_file`), **none added** |
| Does the fixed writer still produce it? | **No** — the same `metrics_long.csv` now yields fully populated identity columns *and* recovers the `RMSE` the on-disk CSV lost |

So the regeneration strictly improved this run and is not implicated. **The writer is fixed (§2);
this artifact predates the fix.** It was not regenerated because re-running the family would rewrite
a committed artifact and move real numbers for cosmetic gain. The honest position: the committed
A_STAT leaderboard would not pass today's contract, and the next A_STAT run will.

Held as an assertion by
`test_artifact_validation.py::test_the_remaining_errors_are_a_pre_existing_csv_defect_not_a_regression`,
so this cannot later be mistaken for a regression, and regenerating the A_STAT family forces this
note to be revisited.

Warnings:

* `SUMMARY.json`: **clean** — `run_id`, `schema_version`, `data_file`, `client_framing` and `model_composition` all present since the 2026-08-12 regeneration
* `a_stat/leaderboard.csv`: `RMSE` all-null *(the writer is fixed; this artifact predates it)*
* `a_stat`, `b_ml`: the persistence baseline row has no rows in `predictions_long.csv`
* `b_ml`: `⚡ Persistence (baseline)` — decoration in a join key
* `b_ml/metrics_long.csv`: `MAE_skill_vs_Ops` all-null; `PI_coverage@90` carries its level in the
  column name
* `e_quantile/metrics_long.csv`: 82 coverage rows with a blank `quantile` *(the writer is fixed)*
* `published/2025-08-06`: no estimators — cannot be re-derived

Pinned by `test_artifact_validation.py::test_the_real_run_is_readable_and_its_findings_are_recorded`,
so closing any of them fails that test and forces this section to be updated.
