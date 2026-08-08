# Phase 13 — Frontend Quality Pass Session Record

> ## ⚠️ SUPERSEDED MID-SESSION — read this first
>
> The scope was corrected to a **visual pass only** after the work below had been committed.
> **Every change described in §2 that alters what is computed or displayed was reverted**
> (`frontend/Overview.py`, `frontend/pages/*`, `frontend/utils_frontend.py` were restored to
> their session-start state and are byte-identical to `706ada9`).
>
> What was kept: the theme, typography, spacing, tabular numerals, `format_gel.py` formatting,
> chart chrome on charts that already existed, `help=` tooltips on metrics already on screen,
> and the page smoke tests.
>
> What shipped visually is described in `reports/phase13_visual_pass.md`. Every reverted
> proposal is written up as a backlog item with rationale in `reports/ui_content_backlog.md`.
> §2 below is retained as the record of what was attempted and measured, **not** as a
> description of the current app.

**Date:** 2026-08-05
**Branch:** `model/excellence` @ `6c763a0` (21 ahead of `origin/main` @ `863f967`)
**Root suite:** 441 → **453 passed, 1 skipped**, `EXIT=0` · **Frontend suite:** 59 passed
**PR #26** open, not merged — reused
**Scope respected:** no modelling code, no evaluation windows, no ruler code touched. TEST sealed.

---

## 1 · Premise verification

| Premise | Measured |
|---|---|
| branch tip | ✅ `706ada9` |
| suite = 441, `EXIT=0` | ✅ |
| PR state | ✅ #26 open, not merged → reused |

---

## 2 · What changed, per page

### Shared foundation · `707c3cc`

**`format_gel.py`** — `NOT_REPORTED = "not reported"` is now the single rendering for absent
data, deliberately a *phrase*. Both `0` and an em dash read as numbers on a dashboard, and
"coverage 0" is a catastrophic model while "coverage not reported" is an unmeasured one.
`is_missing()` also catches the strings pandas writes for nulls (`"nan"`, `"None"`), which is
how an em dash becomes a number in practice. Added `pct`, `pct_points`, `ratio`, `count`,
`number`.

**`ui_styles.py`** (appended, not replaced, so pages migrate incrementally) — type scale,
spacing scale, one accent matched to `treasury_report.html`, tabular numerals forced on
metrics/dataframes/tables. Components: `ds_metric`, **`gate_badge_tri`**, `reading_this_chart`,
`empty_state`, `sample_data_badge`, one `plotly_layout`, and explicit hovertemplates including
the P10/P50/P90 triple. `HELP` carries ten plain-language tooltips written once and reused.

**`intervals.py`** (new) — interval detection and calibration read from artifacts.

**`paths.py`** (new) — one runs-dir resolver honouring `AI4CM_RUNS_DIR`.

### 01_Dashboard · `0e321b7`

The interval tab **worked for the first time**. It had looked only for `y_lo`/`y_hi` with a
hard-coded 90% target, which produced two silent failures at once: E_QUANTILE
(`yhat_p10/p50/p90`) rendered an **empty panel** — the one family whose entire purpose is
intervals — and a correctly calibrated 80% band was scored as 10 points short.

Now: detection via `detect_intervals()`, nominal level **read from the artifact**, plus two
new charts — **per-magnitude-tercile coverage** (the project's biggest known product defect,
now the most prominent thing in the tab) and a **reliability chart** showing where inside its
own band each actual landed, with explicit below/above bars, which distinguishes a
wrongly-*centred* band from a merely too-narrow one.

Three claims the page could not support, fixed:

| Claim | Was | Now |
|---|---|---|
| Alignment | `integrity.get("alignment_ok", True)` — C_DL writes that key as a **literal** `True` without performing the check, and a missing key also defaulted to `True`, so "never checked" rendered as "passed" | tri-state; DL runs and runs without `n_misaligned` render *never verified* |
| Overfit-excluded models | `overfit_excluded_models` used **zero** times; an excluded model could top the leaderboard looking like the winner | red bars, gate ratio in hover, caption naming them |
| Best model | page ranked by lowest MAE; the integrity report additionally applies the overfit gate; disagreement shown silently | reconciled, and on disagreement the page states the gated choice is authoritative |

MAPE → **MASE** on the metric selector, with the tooltip explaining why: the flow targets have
near-zero and negative days where a percentage error is undefined or explodes.

### 02_History · `0e321b7`

**Crashed with no runs** — `df[visible]` on an empty frame raised *"None of [Index([...])] are
in the [columns]"*, a traceback where a fresh clone should be told there is nothing yet. Now an
empty state naming the file, the location and the command. Added freshness metrics and a
**staleness warning**: if the canonical data file was modified after the newest run finished,
every figure in the lab describes an earlier version of the data, and nothing else said so.

### 04_Compare · `9f8d27f`

New first tab: **skill against the one shared ruler**, horizontal bars annotated with each
run's signal reading. Amber bars fail the signal test, with a caption stating that a large
skill number next to a failing signal test is not a forecast but regression to the mean
against a spiky benchmark. Comparability notes render when horizons differ, when evaluation
windows differ, and which families are present.

**The ruler is read from artifacts, not recomputed** — see §4.

### 05_Forecast · `9f8d27f`

Proper fan chart (shaded band, faint dotted edges, P50 on top, full triple in one hover).
**Out-of-data banner** stating the dates lie beyond the end of the data rather than being
upcoming days in a backtest, naming the data end and the forecast range, and that no actuals
exist yet so nothing on the page is an accuracy measurement. Gate badges switched to tri-state.
Track record from `forecasts/scorecard.csv` retains its honest "nothing scoreable yet" state.

### 03_Models · `6c763a0`

Registry already rendered as recorded. Added an **experiments-log explorer**: `experiments/log.csv`
is the audit trail and was reachable only from a terminal, so it existed but nobody using the
lab could follow it. Filterable by target/window/study, with a selector that reads
`experiments/runs/<run_id>.json` and surfaces data fingerprint, code version, benchmark value
and fold scheme. Approver now reads **"none"**, and the detail expander says "none — no
approval workflow exists yet".

### Overview · `6c763a0`

A **"what this lab is / what it does not claim"** panel, first on the page. The four
non-claims: nothing is approved; the 2025 holdout has never been evaluated against; on
Revenues and Expenditure the system reports a typical level rather than an event forecast
because its own signal test fails there; it covers 3 of 41 budget lines and one week.

### 00_Lab, 00_Data_Preprocessing — **not individually reworked**

Both pick up `.streamlit/config.toml`, the page-title convention and the runs-dir resolver, and
both are covered by the smoke tests in all four states. Neither received a chart/tooltip pass.
Stated plainly rather than implied: **these two pages are unchanged in substance.**

### Item 7 — dead files

`frontend/models/registry.py`, `app/app.py`, `core/db.py`, `make_ml_heatmaps.py`,
`make_weekly_from_daily_stat.py` are **all already absent**, removed in earlier sessions.
Nothing to delete, so nothing was deleted — recording the check rather than claiming a removal.

`frontend/sample_data/` still exists on disk but **no page references it**, so it is already
off every default path; a smoke test now guards against regression.

---

## 3 · Artifact fields I needed that do not exist

| # | Field | Where it should live | Why it matters | Current behaviour |
|---|---|---|---|---|
| 1 | **`nominal_pi`** — the advertised coverage level for B_ML's conformal intervals | `artifacts/integrity_report.json` | `ConfigBML.nominal_pi = 0.90` configures the intervals and is **never written anywhere**. Without it the lab cannot score `y_lo`/`y_hi` against anything | Renders *not reported*; coverage shown as a measurement with **no verdict** |
| 2 | An explicit **nominal/quantile level** for E_QUANTILE | same | Currently *inferred* from column names (`yhat_p10`…`yhat_p90` → 80%). Correct today, but it breaks silently if column naming changes | Inferred, with the provenance shown on screen |
| 3 | **`alignment_verified`** (or simply not writing `alignment_ok`) | C_DL's integrity report | `c_dl_pipeline.py:958` writes `"alignment_ok": True` as a **literal** without performing the check. The field is not missing — it is *false confidence*, which is worse | Page treats DL runs as *never verified* by pipeline name, which is a workaround, not a fix |
| 4 | **`study`** | `experiments/log.csv` | The explorer's study filter is derived by substring-matching the free-text `note` ("ws3", "ws4 robust", …). Fragile: a reworded note silently drops out of its filter | Derived from `note` |
| 5 | **`window`** | `experiments/log.csv` | Same — derived by substring-matching `fold_scheme` for "dev"/"train" | Derived from `fold_scheme` |
| 6 | **`nominal_coverage`** per row | `experiments/log.csv` | The log stores `coverage_low/mid/high` but not the level they should be compared against | Assumed 80% only in prose, never in a verdict |

Items 4–6 are small additive columns. Item 1 is a one-line write. **Item 3 is the one worth
fixing properly** — a hardcoded pass is the only case here where an artifact actively asserts
something untrue.

---

## 4 · Mistakes I made, and what caught them

**The smoke test tested nothing, twice.** I wrote it against an `AI4CM_RUNS_DIR` override the
pages did not read (`RUNS_DIR = APPROOT / "runs"` was hard-coded), so all four "states"
silently read the developer's real runs folder and the no-artifacts case was never exercised.
After adding `paths.py`, the tests became **order-dependent**: `utils_frontend` froze
`RUNS_ROOT` at import, so the first test's `tmp_path` leaked into every later test. Fixed with
a lazy path-like — which also stops a real deployment being pinned to whatever path was set
when the module first loaded.

**I did not trust 34 green in 3 seconds.** Mutation-tested it: a deliberate `raise` at a page's
import *and* one buried inside the Dashboard's interval tab each produced 4 failures,
confirming tab bodies really execute. Both mutations reverted from scratchpad backups.

**I nearly put a second ruler implementation in the UI.** My first Compare tab imported
`backend.forecast_integrity` to compute the persistence baseline in the page. That pulled
sklearn into a venv that does not have it — the page raised on load — but the worse problem was
architectural: a second implementation of the shared ruler living in the frontend is exactly
what made the WS2 tuning harness produce skill figures incomparable with every other
workstream. Both integrity reports already record `mae_persistence` and `skill_pct` from the
one shared function, so the page now reads those.

**A string-replace trap, three times.** Replacing `"inject_global_css()"` or
`"_render_registry()"` matched inside a same-named `def` line before the call site, producing
`def inject_global_css()\ninject_design_system(): pass`. Caught each time by an immediate
`ast.parse`, and once more by the smoke test when an Overview import landed after its own use.
Anchoring on a bare call name is unsafe when a same-named def exists in the file.

**`ast.parse` is not enough** (carried from phase 11 and hit again): it proves syntax, not that
an import exists or that names resolve in order. The smoke test is what actually caught the
Overview ordering bug.

---

## 5 · What remains

| # | Task | Notes |
|---|---|---|
| 1 | Write the six artifact fields in §3 | Item 3 (C_DL's hardcoded `alignment_ok`) is the only one that asserts something untrue |
| 2 | Chart/tooltip pass on **00_Lab** and **00_Data_Preprocessing** | Not done this session; both render and are smoke-tested, but unchanged in substance |
| 3 | The WS2 harness defects from phase 11 | Non-canonical ruler and sentinel units mismatch — still open, still blocking `reports/ws2_tuning.md` |
| 4 | Everything in `docs/ROADMAP.md` | Unchanged by this session |

---

## 6 · Reproduction

```bash
git checkout model/excellence            # 6c763a0
./backend/.venv/bin/python -m pytest -q; echo "EXIT=$?"     # 453 passed, 1 skipped
PYTHONPATH=frontend:backend ./frontend/.venv/bin/python -m pytest -q frontend/tests
cd frontend && ./.venv/bin/streamlit run Overview.py
```

The frontend suite needs `frontend/.venv` (it has streamlit); the root suite skips the page
smoke tests via `pytest.importorskip("streamlit")` so one bare `pytest` still works.

---

## 5 · Master prompt received mid-session — premise check and status

A large priority-ordered prompt (Parts A–D, "two to three sessions expected") arrived after the
visual pass was complete. Premises verified by measurement before anything else:

| Premise | Measured |
|---|---|
| Branch tip | `e8ca2f9`, 22 ahead of `origin/main` (`863f967`) |
| Root suite | **453 passed, 1 skipped**, `EXIT=0` |
| Log integrity | `{'n_rows': 138, 'ok': True, 'problems': []}` |
| Registry vs log | `{'recipes': 3, 'metrics_checked': 12, 'ok': True}` |
| TEST gated reads | **0** |
| PR state | **#26 open, not merged** → reuse, do not open a second |

### "Check what is already complete" — nothing is

The prompt cautions against redoing finished phase-12 work. **None of it exists.** Measured
by file check:

```
absent  reports/ws2_tuning.md              absent  reports/accuracy_levers.md
absent  docs/SIGNAL_FINDING.md             absent  reports/PROGRESS_SINCE_LAST_REVIEW.md
absent  reports/ws7_selection_and_cqr.md   absent  reports/model_pool_expansion.md
absent  frontend/i18n.py                   absent  docs/I18N_REVIEW.md
absent  reports/phase12_session_record.md  absent  frontend/assets/logo.svg
```

So there is no completed work to preserve and no duplication risk. Every item in Parts A–D is
unstarted except as noted below.

### A blocker on A2 that cannot be worked around

**`frontend/assets/logo.svg` does not exist.** A2 requires rendering "the client's official
Treasury logo — use as provided, do not redraw or recolour". There is nothing to render, and
redrawing it is explicitly forbidden and would be wrong regardless: an approximated official
emblem is worse than none. **A2 needs the file supplied.** The rest of the header (app name,
one-line subtitle) is implementable without it and should be built so the logo drops in.

`~/Projects/ai4cm-ui/handoff/` contains `DESIGN_TOKENS.md`, `hover-copy.md` and `README.md`,
so A1 and A4 have their sources.

### Status of every item, for the next session

| Item | Status |
|---|---|
| **A1** token adoption | **Not started.** Note the reversal: this session followed DESIGN_TOKENS.md's own "not a request to restyle" and kept the lab palette; A1 now **overrides** that and asks for the tokens to become the lab's look. Contrast ratios must be re-measured and stated |
| **A2** logo header | **Blocked** — `frontend/assets/logo.svg` absent (above) |
| **A3** chart aesthetics | **Partly done.** `plotly_chrome()` exists and is applied to 3 of 9 Dashboard figures, both Compare figures and the Forecast band chart. Remaining: the other 6 Dashboard figures, a Plotly template matching the tokens, GEL-millions via `format_gel.py` on displayed values, hovertemplates, and non-colour encodings for greyscale |
| **A4** tooltips | **Partly done.** 10 tooltips on metrics already on screen (6 Dashboard, 4 Forecast), written by hand. Not yet sourced from `handoff/hover-copy.md`, and the specific metrics A4 names (skill, sentinel/1.50, MASE, coverage, WITHHELD, never-verified) are mostly not on screen as metrics yet |
| **A5** Georgian i18n | **Not started.** Non-trivial: needs the strings table, toggle, `session_state`, `translation_status: "unverified"` on every Georgian string, `docs/I18N_REVIEW.md`, and the test that toggling leaves every displayed number byte-identical |
| **A6** bilingual smoke test | **Not started.** Monolingual version exists (`frontend/tests/test_pages_smoke.py`, 34 cases) and its docstring already states what it does not verify |
| **B1** upstream artifact fixes | **Not started.** Six items (run_id, X9, X10, F1, C8, X11). C8 overlaps the interval-nominal gap I hit and documented as A1/A2 in the backlog |
| **B2** WS2 defects + `ws2_tuning.md` | **Not started.** Both defects are diagnosed with causes in `reports/phase11_session_record.md` §4; the prompt's instruction not to re-run the search matches what that record says |
| **B3** `SIGNAL_FINDING.md` | **Not started** |
| **B4** WS7 (CQR, conditional-coverage gate, selection rule) | **Not started.** This is the biggest product defect and Tier 2 items 5–6 of the UI backlog exist only because it is unfixed |
| **C1–C8** accuracy levers | **Not started** |
| **D1–D2** progress record | **Not started** |

### One correction the next session should carry

The master prompt's sourcing rule — *"read from artifacts and `experiments/log.csv`, never from
a session record's summary table — those have drifted"* — is well founded. Phase 11 §4 documents
that the WS2 harness wrote **non-canonical rulers** (Revenues 90,800,654 vs canonical
88,317,355; Expenditure 76,722,514 vs 73,117,667), so **every `skill_vs_ruler` value on the
three `study: ws2_tuning` rows in `experiments/log.csv` is unusable** and must be recomputed
before publication. The `dev_mae` values on those rows are sound.

