# Session — MVP consolidation: seven tasks across the interface, the registry and the copy

**Date:** 2026-08-19 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`).
**Constraint honoured throughout:** gates, selection logic and champion governance untouched.
Every change is at the caller level, in the interface, in the model registry, or in documentation.
`evaluation_windows.py`, `publication_gates.py`, `registry.py` and `select_best_model` are
unchanged apart from nothing at all.

Seven tasks, all landed, in seven commits.

| | Task | Outcome |
|---|---|---|
| 1 | Exploratory runs must not default into the sealed window | Guard was right, caller was wrong. UI runs now bound to train and dev **by construction**; failures explained in plain words |
| 2 | Forecast page: champion plus top alternatives | `model_shelf.py` reads the run ledger; comparison runs through the exploratory path and cannot publish |
| 3 | Scorecard page, with upload | New page; `ingest_actuals.py` is the single validation and install path for the UI *and* the command line |
| 4 | Registry: adding a model is one entry, untested models honestly labelled | `model_catalog.py`; status **derived** from the ledger; three new candidates; `docs/ADDING_A_MODEL.md` |
| 5 | "Start here" guide page, first in the sidebar | Coverage asserted against the pages directory; two pre-existing bugs found while getting it green |
| 6 | Copy and interaction pass across the entire UI | Rules are tests, not a style note. 57 em dashes, 10 double hyphens, 9 missing intros, 6 undefined terms |
| 7 | Georgian toggle, marked for native review | Source-text keyed, one dictionary file, coverage measured at 78.6% and stated on screen |

> **Three findings in this record matter more than the tasks that surfaced them.**
>
> 1. **Seven of the thirteen models the Lab offered had no recorded result on any target**, and the
>    Lab's default selection, Ridge, was one of them. They sat in the list beside a champion with
>    five folds of evidence behind it, with nothing distinguishing them (Task 4).
> 2. **The full test suite is two commands, not one.** Run under `backend/.venv` alone, every one
>    of the eleven page-rendering test modules silently skips. Every "full suite green" in this
>    repository's history before today was measured that way (Task 5).
> 3. **The Dashboard crashed outright on any run with no integrity report**, which is the ordinary
>    state of an older run and of any run that failed before writing one (Task 5).

---

## Task 1 — the exploratory window

### What was reported

A UI-launched single-model run (Ridge, State budget balance, h=6) raised
`SelectionOnReportOnlyDataError` and showed the user a traceback.

### What was actually wrong

The guard was correct: the run really was about to rank models using rows from the sealed
holdout. The cause was two levels upstream and had two parts.

**The Lab sent no evaluation bound at all.** `TG_PARAM_OVERRIDES` carried `folds`,
`min_train_years` and `demo_clip_months` and nothing else. Every family reads an absent bound the
same way, which is to fold forward to the last year in the file, and on this dataset that makes
the final fold `train <= 2024-12-31 / evaluate 2025-01-01..2025-08-06`. That is the holdout end to
end. C_DL was worse: its runners *default* `eval_start` to `TEST_START`, so omitting the key
selects the holdout rather than leaving it open.

**The Demo profile made bounding alone insufficient.** It set `demo_clip_months: 12`, clipping the
series to the last twelve months of the file. On this dataset that is almost entirely holdout and
post-seal data, so a bounded evaluation over a clipped series has nothing left to measure, and
A_STAT's `_fallback_fold` would then quietly hand back the newest rows in the file, which are
exactly the rows the bound exists to exclude.

### What changed

`frontend/exploratory.py` is the single definition of what an exploratory run is. It pins
`eval_end` to the last dev date, sets `eval_start` to an **explicit null** rather than omitting it,
and clears `demo_clip_months`.

Three families could not honour a bound at all and now can. All three additions default to `None`,
so an unbounded reporting run is unchanged:

* **A_STAT** — `_yearly_folds` and `_fallback_fold` take optional bounds. The fallback mattered
  most, for the reason above.
* **C_DL** — `ConfigDL` gains `eval_end`; `build_yearly_folds` trims to it.
* **E_QUANTILE** — `Config` already had `eval_end` and neither runner read it, so a caller could
  set it and be no better off.

Failures now read as sentences. `backend/runner_errors.py` gives all four families one error-report
shape, where only B_ML wrote one before, and `frontend/run_errors.py` turns it into plain language,
distinguishing a guard refusal, which is the system working, from a fault, which is not. The
traceback moved behind a "Technical detail" expander.

### What was deliberately not done

The guard was not weakened, wrapped or bypassed. Tests assert it still refuses TEST and LIVE, that
it cannot be evaded through the Lab's advanced JSON box, and that the exploratory path produces no
configuration reaching report-only data on any family or profile.

### Consequence worth stating

The Demo profile is now slower. It controls fold count and training depth and no longer shortens
the data span, because the clip is what pinned the window onto the holdout. The page says so.

---

## Task 2 — what came second

The Forecast page showed one model per target and nothing else, so "this is the best model" had to
be taken on trust.

`backend/model_shelf.py` reads `experiments/log.csv` and its run sidecars and answers three
questions per target: who the champion is and on what evidence, what else was measured and whether
it would clear the gates, and how the champion compares to the Treasury's current planning method.

**It chooses nothing, and that is tested.** The champion is read from the registry, which is
hand-edited; eligibility is judged by `publication_gates`, the same code that decided the
champion's own verdict, rather than by a threshold retyped here. One test asserts the registry
bytes are unchanged after building every shelf; another asserts the module contains no write path.

Two facts the page could not previously state:

| Target | What the shelf shows |
|---|---|
| Expenditure | Two alternatives beat the champion on the naive benchmark. The registry already recorded that the champion is not the dev best and why; the page now shows it |
| State budget balance | Twelve models measured, none beats the naive benchmark. Rendered as a sentence, because an empty table would read as "nothing was tried" |

Skill against the Treasury's method is computed over the same 262 working days of 2024 the
champion's own credentials cover, so the percentage means something. On a stock target the method
is undefined and the panel says why rather than leaving a blank a reader could take for a tie.

"Compare alternatives" runs the champion and its runners-up through
`forecast_modes.exploratory_run`, which returns a type with no publish path. Bannered before the
run and again beside the results.

**Copy fixed on the way:** `withheld_as_forecast` was being printed at the reader on both sides of
an arrow in the verdict history and inside the generated explanation sentence. Both now speak
words. The substance of the reconciliation is unchanged, which is why its tests assert substance.

---

## Task 3 — the Scorecard, and the one door for actuals

The track record lived in a section at the foot of the Forecast page, and there was no way to add
actuals except by copying a file over the canonical dataset by hand. Nothing checked that file and
nothing kept the one it replaced.

**The page** renders honestly in both states, and both are tested. Today: 0 scored, 25 pending,
every pending date listed, because a track record that shows only the rows it has scored is one
that can be made to look good by scoring selectively. The scored branch is exercised against a
synthetic fixture, and the page says on screen when it is reading one.

**The ingest.** `backend/ingest_actuals.py` is one validation and one install, called by the page
and by the command line. Four refusals, each a way a load has gone wrong or plausibly could: a
missing column the recipes read, a file that does not extend the record, the identical file
uploaded twice, repeated dates. Revisions to days already held are counted and reported rather than
refused. The file being replaced is copied to a timestamped backup first.

**Found while testing:** a candidate with a repeated date raised inside the revision counter
instead of surfacing the blocker, so the page would have crashed on exactly the malformed file the
blocker exists to catch.

**Page order.** The sidebar ran Data Pre-processing, Lab, Dashboard, History, Models, Compare,
Forecast: roughly the order the pages were built in, and close to the reverse of the order a
Treasury reader needs them. Renumbered to Forecast, Scorecard, Dashboard, Compare, History, Models,
Data Pre-processing, Lab, leaving `00` free for the guide. Historical session records keep the old
filenames, because they record what was true when they were written.

---

## Task 4 — the model shelf

### The finding

Adding a model meant editing a dictionary in the middle of `b_ml_pipeline.py`, between the feature
builder and the training loop, with the conditional import blocks interleaved into it. That was the
stated problem. The unstated one was worse:

> **Seven of the thirteen models the Lab offered had no row in `experiments/log.csv` on any
> target.** Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees and both CatBoost variants. They
> appeared in the model list beside `LightGBM_L1`, which has five folds of evidence behind it, with
> nothing distinguishing them. **Ridge, the first entry and the Lab's default selection, was one of
> the seven** — and it is the model whose run opened this session.

### What changed

`backend/model_catalog.py` is the shelf: one `ModelSpec` per model with a plain-language summary,
the packages it needs and the date it was added. `available_models()` builds from it and names no
model of its own, which a test asserts. Factories import inside their bodies, so the Streamlit
interpreter can read the catalogue without the modelling stack.

**Status is derived from the ledger, never declared.** A declared status is a claim that drifts the
moment somebody adds a model and does not run it, which is precisely the state this found the
project in. Baselines are a third status: asking whether the ruler was measured is the wrong
question.

### The refactor is proved to have moved nothing

Every published number in this project came out of the old construction, so a moved hyperparameter
would have made all of them quietly unreproducible and nothing else in the suite would have
noticed. Two independent proofs:

1. A snapshot of every estimator's full parameter set, taken programmatically from the pre-catalogue
   code **before** the change, asserted entry by entry.
2. Ridge and `HistGBDT_L1` built both ways, fitted on identical data, predictions compared for
   exact equality.

### New candidates

`Huber` and `GBDT_L1` in B_ML, `ETS_DAMPED` in A_STAT. None needs a package that is not already
installed, so nothing was flagged for installation. All three show as UNTESTED, as do both CatBoost
entries.

### The client sentence

It read "13 machine-learning models compete on each target", which sounds like thirteen measured
contenders. It now adds that **8 of the 28 non-baseline entries have a recorded result and 20 do
not**. `docs/AGENT_ARTIFACT_CONTRACT.md` carries the same two counts and warns against quoting the
shelf count as a measure of work done.

`docs/ADDING_A_MODEL.md` is the fifteen-minute guide, including what the checks will do to a bad
model.

---

## Task 5 — the guide, and two bugs it uncovered

`frontend/pages/00_Start_here.py` answers the same three questions for every page and ends with the
two doors explained plainly. Coverage is asserted against the pages directory, so adding a page
without a section fails the test.

Getting the suite green in **both** interpreters surfaced two defects, neither introduced by this
session:

**The Dashboard crashed on any run with no integrity report.** `_integ` was `None` in that case,
every other read of it was guarded, and two were not, so the page died with a raw `AttributeError`
rather than rendering. That is the ordinary state of an older run and of any run that failed before
writing a report. It now loads as an empty dict.

**The Dashboard quoted a superseded threshold.** It told readers the signal check required a ratio
of 1.50. P2 calibrated that against a measured null distribution and moved it to 1.15; 1.50 survives
in `publication_gates` only as the uncalibrated value nothing applies. So the KPI beside every run
quoted a stricter bar than the one any verdict was decided by, and the shared help text said the
same. Both now read the number from `publication_gates` rather than retyping it. The test that
should have caught this asserted 1.50 and rendered against whatever happened to be in the
developer's runs folder; it now builds its own run and reads the threshold from the gate code.

### The suite is two commands

```bash
./backend/.venv/bin/python  -m pytest -q              # 1209 passed, 9 skipped
./frontend/.venv/bin/python -m pytest -q frontend/    #  482 passed, 7 skipped
```

`streamlit` lives only in `frontend/.venv` and the modelling stack only in `backend/.venv`. Running
under either alone silently skips the other's tests: under `backend/.venv`, all eleven
page-rendering modules skip. Six tests added earlier in this session were failing under
`frontend/.venv` and passing under `backend/.venv` for exactly that reason.

---

## Task 6 — the copy pass

Copy rules that live in a style note get followed for a fortnight. These are tests.

`frontend/tests/ui_copy.py` defines what "user-visible" means: it parses each page and collects the
arguments of the calls that render text, the `help=` tooltips, this project's presentation helpers,
and the module-level copy dictionaries. A comment is never a call argument, so the long explanatory
comments this codebase relies on are out of scope by construction, which is the only reason rules
this strict are liveable.

| Rule | What it found |
|---|---|
| No em dashes or double hyphens | 57 and 10, across seven pages, all rewritten rather than substituted |
| Every page opens with what it is for | Nine of ten opened on a control. Compare Runs opened with a run selector |
| Every technical term is defined where used | MASE, champion, exploratory, holdout, sealed window, withheld |
| Long explanations behind an expander | Enforced at 700 characters |
| Nothing unfinished, nothing placeholder, nothing overclaimed | A dozen strings that trailed off |

`ui_styles.GLOSSARY` is now the one definition of each term; `glossary_note` puts the terms a page
uses behind an expander beside its intro.

**The prose detector needed two corrections of its own**, both of which had been exempting exactly
the copy that most needed checking. It treated any string starting with `**` as markup, so every
bold-led paragraph escaped every rule; and it walked into non-literal operands of a concatenation,
so `"...days. " + pub["reason_plain"]` was read as ending in the words "reason_plain".

---

## Task 7 — Georgian

A selector (English / ქართული) in the sidebar of every page. Translation reaches the reader through
the shared helpers, so ten pages needed one call each rather than an edit at every string.

**The design and its one real cost.** Translations are keyed by their English source text, so
`frontend/translations_ka.py` holds English sentences on the left and Georgian on the right with no
identifiers, and the code still reads as English prose. That was the requirement: one file for a
native speaker. The cost is that editing an English string silently orphans its translation, with
no error and no visible sign. `test_i18n.py` is the guard for exactly that, and **it caught
fourteen orphans on its first run**.

**Coverage is measured, not claimed.** 55 of 70 interface phrases, **78.6%**, the remainder being
the Lab's long configuration tooltips. The sidebar note in Georgian states the figure beside the
standing sentence that the text is machine generated and unreviewed. Data is excluded from both the
dictionary and the count, and a test asserts no Treasury line name, model name or date is in it.

**Two bugs found on the way, both introduced by this task and caught before landing:**

* Routing tooltips through the translator rewrote `HELP["x"]` across every page including the Lab,
  which keeps its own `HELP` dictionary and shares none of those keys. **Every tooltip on the Lab
  page silently became empty**, and nothing failed, because an empty tooltip renders as no tooltip
  rather than as an error.
* The copy checker from Task 6 could not see `ui_styles.HELP` or `GLOSSARY` at all, because they
  are assignments rather than call arguments. **Three em dashes had been sitting in the tooltips
  the whole time the punctuation rule was passing.**

---

## What remains open

| | Item | Why it is open |
|---|---|---|
| 1 | **The Georgian needs a native speaker** | Every string is machine generated. `frontend/translations_ka.py` is the only file to edit, and the app states its unreviewed status on every page while Georgian is selected |
| 2 | Translation coverage is 78.6% | The gap is the Lab's long configuration tooltips. Measured and stated on screen rather than hidden |
| 3 | **The three new candidates are unmeasured** | `Huber`, `GBDT_L1` and `ETS_DAMPED` are registered and badged UNTESTED. Moving them to measured needs a run entered in `experiments/log.csv` through the evaluation protocol, which is a deliberate exercise |
| 4 | 20 of 28 non-baseline models have no recorded result | Includes both CatBoost entries and every linear and forest model. The shelf now says so; running them is separate work |
| 5 | The Scorecard's scored branch has never run on real data | 0 scored, 25 pending. It is exercised against a synthetic fixture, and the page labels that fixture on screen. The first real scoring happens when actuals past 2025-08-06 are loaded |
| 6 | Nothing is approved | Every recipe's status remains `candidate`, and `approved_by` is `null` on all three. Unchanged by this session and deliberately so |
| 7 | The Demo profile is slower | It no longer shortens the data span, because the clip is what pinned the evaluation onto the sealed window. Caption added; no faster honest alternative was found |

## What was deliberately not touched

`evaluation_windows.py`, `publication_gates.py`, `registry.py` and `select_best_model`. No gate
threshold moved, no window boundary moved, no champion changed, and `registry/recipes.json` is
byte-identical to where the session started. A test asserts the last of those directly.
