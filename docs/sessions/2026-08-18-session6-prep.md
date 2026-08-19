# Session — Lab prep for the scoring loop (Session 6-prep)

**Date:** 2026-08-18 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`).
The agent repo was **read** (its Session 5 record, §8, on request) and not modified.
**Constraints honoured:** no model refitted, no published artifact under `forecasts/published/`
touched, three items planned and approved one at a time.

Four scoped items, all landed:

| | Item | Outcome |
|---|---|---|
| 1 | `--issue-date` CLI fix, as proposed in the agent's Session 5 §8 | applied verbatim, plus the docstring it falsified |
| 2 | Scorecard schema, fixed before the first real row exists | 22 → 27 columns (31 after item 4), documented, versioned |
| 3 | The hardcoded test date | fixed by derivation, and the decay made self-reporting |
| 4 | The Ops baseline wired in as a second comparator | **found it returned identically zero**; fixed, then wired — 27 → 31 columns |

> **The finding that matters most in this record is in Item 4.** The Treasury-comparison metric
> the project has been reporting was computed against a baseline of **zero**, overstating the
> margin roughly fourfold. Any figure of the form "N% better than the Treasury's current method"
> predating this session — including the July deck — should be treated as withdrawn.

---

## Item 1 — the publish CLI could not publish

### The proposal, restated

From `ai4cm-agent-v2/docs/sessions/2026-08-17-session5-agent-action-layer.md` §8, marked *"Do
not apply this in the agent repo. It belongs to AI4CM."*: `_cli` called `publish_official(res)`
with no `issue_date`, so `publish()` derived one from `max(origin_date)` — the end of the data,
a date that already has an issue. Add `--issue-date` defaulting to `next_issue_date()`; keep the
flag so a deliberate re-issue can name its own date.

### Verified before implementing

| Claim | Check |
|---|---|
| `publish()` derives from `max(origin_date)` | `published_forecasts.py:238-239` |
| that date already has an issue | `max(origin_date)` = **2025-08-06**, and `forecasts/published/2025-08-06` exists |
| `_cli` passed no issue date | `forecast_modes.py:521` |
| the UI hits the same wall | `05_Forecast.py:384` offers *"Publish … under a new issue date"*; line 390 appends only `--publish` |
| nothing exercised it | the only CLI test asserts on **module source text** — it greps for the string `"--publish"` and never runs it |

That last row is why the bug shipped: a test can name a flag without proving it works.

### What changed

`--issue-date` added, defaulting to `next_issue_date()`; the publish call passes
`a.issue_date or next_issue_date()`.

**It also settles what an issue date means.** `publish()`'s docstring called the origin-date
default *"the honest label"* without noting that the origin date is a property of the *data*, so
it is the same date on every run from an unchanged file and therefore collides on the second
publish. Both conventions were already on disk — `2025-08-06` from the fallback, `2026-08-13`
and `2026-08-16` set by hand in Session 2. The CLI now standardises on wall-clock, and the
docstring says so instead of recommending the path that crashes.

### Tests — five, mutation-checked

Two publishes in a row give `<today>` then `<today>-r2`; an explicit `--issue-date` is honoured;
`published_to` is the real path; the new default is inert when `--publish` is absent; and the old
collision is pinned directly, so the default's *reason* is recorded rather than only its effect.

The pre-existing source-grep test was **kept** — its real job is proving the page does not import
the modelling stack — and these run the path alongside it.

With the fix reverted, all three CLI tests fail. Verified, not assumed.

**A trap worth recording.** `publish()` sets `vault_root = VAULT_PUBLISHED` whenever
`published_root is None`, and the CLI passes no root — so a test redirecting only
`PUBLISHED_ROOT` would have written into the real `private_vault/`. Both constants are
redirected. Confirmed afterwards that both real roots still hold exactly `2025-08-06`,
`2026-08-13`, `2026-08-16`.

---

## Item 2 — the scorecard schema

### What was already there, and why it was not enough

The schema was **not** undefined: `SCORECARD_COLUMNS` held 22 names, and three tests asserted
the written columns equalled that tuple. But a check of the form *"the output matches the
constant"* is self-consistency — it passes just as happily when a required field is missing.

One was. Checked against the required list field by field, **origin was absent**. And Item 1
made that worse in the same session: with `issue_date` now wall-clock, *nothing in a scored row
recorded the data vintage the forecast was made from*. A row could say "issued 2026-08-18,
target 2025-08-13" and a reader could not tell a five-day-ahead call from a backfill.

### The schema — 27 columns, grouped by what each part is for

Added: `schema_version`, `origin_date`, `origin_value`, `interval_nominal`, `interval_model`.

* **`origin_date`** — the data vintage. The field that was missing.
* **`origin_value`** — the level at the origin. `persistence_pred` is *derived* from it, so
  without it the shared ruler could not be audited from the scorecard alone. It now can, and the
  existing ruler test was tightened to check exactly that rather than reaching back into the
  artifact.
* **`interval_model`** — which model produced the band, distinct from `point_model`. Not
  cosmetic: measured on the sealed window at the same 80% nominal, GBQuantile covered 78.0% of
  revenues outcomes against ResidualRF's 69.8%, and 73.7% against 53.8% on the stock target. A
  band miss that cannot be attributed is of little use to a retraining decision.
* **`interval_nominal`** — the level `inside_interval` was measured against, read from the
  published quantile columns. Keeping that level only in column names is the defect the
  E_QUANTILE audit already had to fix once, where a figure keyed `coverage_p10_p90` described a
  different interval.
* **`schema_version`** — rows accumulate across issues over months and this field set will grow.
  Without it, a consumer reading a mixed file infers the schema from which columns happen to be
  present, which is guessing.

Order was set now because **zero rows existed** — the one moment it costs no migration. Nothing
reads the file positionally (`estimator_store` checks by name), so this was free.

The constant is now the documented schema: every field carries its meaning and its source, and
the pairs that are easy to conflate — `issue_date` vs `origin_date`, `data_sha_at_issue` vs
`scored_at_data_sha` — say why both exist.

### A real limitation found while testing, and deliberately refused rather than papered over

`interval_nominal_of` reads the level from whatever quantile pair an artifact carries — `p5`/`p95`
→ 0.90, correctly. **The scorer cannot process such an artifact.** It names the three columns
directly in three places, so a differently-quantiled issue died on a raw `KeyError: 'p10'` deep
inside pandas, before the schema was reached at all.

Unreachable today: `forward_forecast.QUANTILES` is fixed at `(0.10, 0.50, 0.90)`. So the field as
written was future-proofed *in the record and not in the code* — right while the scorer refused
to produce the row.

Resolved as an explicit refusal, not a generalisation. `UnsupportedIntervalShape` now fires at
the issue boundary naming what it found and what is supported, and leaves no partially written
scorecard. Making the arithmetic quantile-agnostic changes how every score is computed; that
belongs in a session that scopes it, not in a schema session as a fallback. A companion test
asserts the supported shape still equals what the publisher emits, so the refusal cannot start
catching real issues unnoticed.

### One more layer, found on review

`summarize_scorecard` printed `nominal_coverage` from the `NOMINAL_COVERAGE` constant, right
beside an `interval_hit_rate` computed from the rows. That is the same defect `interval_nominal`
exists to remove, moved one layer up into the summary a reader actually looks at. It now reads
the level from the rows, and when a file mixes two band levels it reports **both** rather than
picking one. The constant remains only as the fallback for rows written before the field existed.

### The tracked file had to be regenerated

`forecasts/scorecard.csv` is **tracked and not gitignored**, so a schema change that did not
rewrite it would leave the repository shipping a header contradicting its own definition.
Regenerated by running the real scorer: 27-column header, still **0 data rows**. A test now
asserts the committed header matches the constant, so that drift cannot recur.

### Verified against the real published issues

`score_published` over the three real issues reports **0 scored, 25 pending** — the honest
result, because every published target date is beyond the canonical data end (2025-08-06). The
schema round-trips; there is simply no truth yet.

### Tests — 34, in a new file

The required-field list is written out **as a literal**, not derived from `SCORECARD_COLUMNS`:
deriving it would reproduce exactly the weakness of the checks it replaces. Beyond presence,
they assert: every required field is *populated* on a scored row; `origin_date` precedes
`target_date` and the gap equals the recorded horizon; `issue_date` and `origin_date` are
genuinely different values; the recorded ruler equals the recorded `origin_value`;
`interval_nominal` follows the artifact; an issue predating `interval_model` scores with it
**blank rather than guessed**; the two data hashes differ when the scoring data differs; and
absence readers return blank for every absent form rather than a substitute.

---

## Item 3 — the hardcoded test date

### Where it is

Exactly one future-dated literal in the entire test suite, and **none** in non-test source
(scanned, not assumed):

```
backend/tests/test_live_window.py:79
    days = pd.date_range("2015-01-05", "2026-12-31", freq="D")
```

### What breaks when the date passes — nothing, and that is the defect

I expected a failing assertion and tested for one:

```
end=2026-12-31: names=['dev','live','test','train'] -> passes? True
end=2026-01-01: names=['dev','live','test','train'] -> passes? True
end=2025-09-01: names=['dev','live','test','train'] -> passes? True
```

The range is fixed at 2015→2026 and `window_for` maps that span onto all four windows whenever
it runs. So past 2026-12-31 the test still passes — while checking a period lying **entirely in
the past**. Every day the project actually operates in falls outside the range being verified.

This is not a bomb that fails loudly; it is a test that **decays into a tautology**, which is
worse, because breaking is a report. And it is exactly why bumping the constant is the wrong
fix: `2027-12-31` buys twelve months and keeps the mechanism.

### The fix, in two parts

1. The upper bound is derived from the clock — `max(Timestamp("2026-12-31"), today + 1 year)`,
   the literal kept as a **floor** so the span can never shrink below one that exercises all
   four windows.
2. **`today` is asserted to be inside the checked range.** This is the part that matters: it
   converts the failure mode from silent to loud. `today` was also added to the
   exactly-one-window check, so that property is verified for now and not only for five
   historical boundaries.

Mutation-checked. With the bound reverted to a passed literal, the test now fails with:

```
AssertionError: the tiling check spans 2015-01-05..2026-01-01, which excludes today
(2026-08-18) — so it is verifying only the past. This is the failure the derived bound
exists to make visible; do not fix it by widening a literal.
```

Where the old test would have passed.

---

## Item 4 — the Ops baseline, and a metric that was measuring nothing

The ask: wire the Treasury's current planning method in as a second comparator beside the naive
ruler, on every leaderboard row and in the scorecard, as `PURPOSE_REPORT` reads through the
ledger, with the gate untouched.

### The finding: `skill_vs_Ops` was skill against zero

While validating the construction I found that `ops_daily_from_monthly(method="profile")` — the
only branch any caller used — returned **identically zero for every date**:

```
ORIGINAL daily ops baseline: 1983 non-NaN values
  exactly zero : 1983
  nonzero      : 0
monthly baseline was NON-zero: 92 of 92 months   (e.g. 2025-01 = 1,720,083,190)
```

The monthly totals were right. The daily spread was zero. Cause, at `c_dl_pipeline.py:236`:

```python
p = (s/s.sum()).reindex(days, fill_value=0.0).values
```

`s/s.sum()` is indexed by the working days of the same month in a **previous year**; `days` are
the working days of the **current** month. No label can match across years, so every weight
became `fill_value=0.0` and `mval * p == 0`.

**Blast radius.** `MAE_skill_vs_Ops` is emitted by C_DL, A_STAT, the ensemble post-process and the
weekly-stat path. All of them compared against this series, so all of them reported *skill against
a zero forecast* — i.e. against `mean|y|`. Measured on revenues that read 55–62% where the truth
against the stated method is −8% to +8%: a **fourfold overstatement of the project's headline
client-facing claim**, passing silently because nothing asserted the baseline was non-zero.

Fixed at source: the shape is carried across by **position**, not by date label, with working-day
counts differing between months handled by interpolation. Two tests now pin it — the spread is not
identically zero, and each month's daily values sum to its monthly total (the property the bug
violated most visibly: every month summed to 0 against a non-zero total).

### Construction decisions

**Flat spread is canonical.** The method as stated is *"3-year average annual total, split across
months by each month's historical share, then spread across working days"* — which is
total ÷ working days. It is also the harsher comparison and emits no negative figures. The
corrected **profile** spread is available as a sensitivity but inherits the 72 negative days in
the raw revenues series into 6 negative daily baselines (min −184M); a negative *revenue* planning
figure is not defensible to a reader even when the arithmetic is faithful.

**Causality: it is honestly computable at a daily h-step origin, and this was proved not argued.**
Every figure depends only on complete prior calendar years, so the whole 2025 baseline is knowable
on 2025-01-01. Verified by truncation — recomputing from data cut at four separate origins inside
the sealed window reproduces the full-history values exactly (monthly 354/354, daily 7618/7618, on
both flow targets).

**The one place that is not automatic, and the vintage construction.** At h=5, four sealed-window
target dates (2025-01-01, 01-02, 01-03, 01-06) have origins in December 2024, before 2024 closed —
so the 2025 baseline did not exist at those origins. Each row is therefore scored against the
vintage **in force at its own origin**: those four use the window ending 2023, the rest the window
ending 2024. Two vintages, all 156 rows usable, and the comparison is literally "against what the
Treasury had in hand".

**Flows only.** The method aggregates a flow to an annual total; a balance level has none, so the
stock target records `NaN` with `ops_source` stating why. No baseline was invented for it.

### Measured numbers — sealed window, h=5, replacing the July deck figures

**Revenues.** Ops MAE = **41,794,782** (identical across models by construction).

| Family | Model | n | MAE | skill vs naive | **skill vs Ops** |
|---|---|---:|---:|---:|---:|
| E_QUANTILE | ResidualRF | 205 | 33,245,860 | 49.54% | **+20.77%** |
| E_QUANTILE | GBQuantile | 205 | 33,964,365 | 48.45% | **+19.06%** |
| B_ML | XGBoost | 156 | 38,554,577 | 36.77% | +7.75% |
| B_ML | RandomForest | 156 | 38,565,591 | 36.75% | +7.73% |
| B_ML | ExtraTrees | 156 | 39,347,999 | 35.47% | +5.85% |
| B_ML | HistGBDT | 156 | 41,978,520 | 31.16% | −0.44% |
| B_ML | LightGBM | 156 | 42,969,803 | 29.53% | −2.81% |
| B_ML | Lasso | 156 | 43,315,354 | 28.96% | −3.64% |
| B_ML | Ridge | 156 | 43,333,157 | 28.93% | −3.68% |
| A_STAT | ETS | 156 | 44,199,748 | 27.51% | −5.75% |
| B_ML | ElasticNet | 156 | 45,064,066 | 26.10% | −7.82% |

**Five of eleven models are worse than the Treasury's current method.** The champion
`LightGBM_L1` has **no sealed-window predictions in this run** — it holds `LightGBM`, its
squared-error twin, which sits at −2.81%. So the honest statement is: *the champion's margin over
the current method has not been measured on the sealed window.*

**State budget balance** — champion `HistGBDT_L1` at MAE 173,314,894, skill vs naive **+8.75%**.
`skill_vs_ops` is `NaN`: the method is undefined for a level. (Best on this target is E_QUANTILE
`ResidualRF` at +31.12%, but the target's verdict is `withheld` on accuracy.)

**Expenditure** — **no sealed-window predictions exist for this target in any run on disk.** No
ops comparison is possible without a refit, which this session excluded. Nothing is reported for
it rather than a substitute.

### What was wired

* **`backend/ops_baseline.py`** — one causal, vintage-correct reader, so ops is not computed five
  different ways. Logs its sealed-window reads via `require_test_access(..., PURPOSE_REPORT)`.
* **Scorecard 27 → 31 columns**, `schema_version` **1 → 2**: `ops_pred`, `ops_abs_error`,
  `skill_vs_ops`, `ops_source` — mirroring the existing persistence quartet, so an ops figure is
  auditable from its own row exactly as the naive one is. Tracked CSV regenerated; header test
  updated.
* **`b_ml` leaderboard** carries `ops_MAE` and `skill_vs_ops_pct`, or an `ops_note` explaining the
  absence.
* Ops figures attach to **pending** rows too — what the current method said is knowable before
  truth arrives.
* **`skill_vs_ruler_pct` kept** as the naive column (renaming would churn the agent contract, the
  frontend and three test modules) with its equivalence documented in the schema.

### The gate is untouched, and tested to be

Three tests: `publication_gates.py` contains no occurrence of "ops"; `Measured` has no ops field;
and all three verdicts are byte-identical to the registry. That last one matters because five of
eleven models are *worse* than the current method — if ops skill ever leaked into gating it would
change what publishes.

---

## Open items

1. **One issue holds one target.** §8's second observation, unapplied by design. `publish()`
   keys the destination on issue date alone, so a second `publish_official` for the same date
   raises `FileExistsError`. It cannot bite today — only Revenues is publishable; Expenditure
   and State budget balance are both `withheld`, which `publish_official` refuses outright — but
   it needs a modelling decision: should one issue hold several targets, or should several
   issues share a date? Inventing a `-r2` date for a second *target* would label it a re-issue,
   which it is not.
2. **The scorer names `p10`/`p50`/`p90` directly.** Now refused loudly instead of crashing, but
   the arithmetic is still quantile-specific while the schema can describe any pair. Generalise
   it when a second shape actually exists, and note that `abs_error` (against `p50`) and
   `inside_interval` (against the outer pair) both need it.
3. **Nothing has been scored yet.** The scorecard is schema-complete with 0 rows, and will stay
   that way until actuals arrive past 2025-08-06. Session 6's first real rows are the first test
   of this schema against reality; the fixtures are synthetic by necessity.
4. **The champion's margin over the current method is unmeasured.** `LightGBM_L1` has no
   sealed-window predictions on Revenues and Expenditure has none at all, so the single number a
   Treasury reader would ask for — "how much better is the model you would actually run?" — does
   not exist yet. Its L2 twin sits at −2.81%. Getting it needs a `PURPOSE_REPORT` refit.
5. **Existing artifacts still contain zero-baseline `MAE_skill_vs_Ops` values.** The function is
   fixed, but the committed C_DL / A_STAT / ensemble / weekly-stat metrics were written before the
   fix and are stale by roughly a factor of four. Not rewritten here — the rules put committed run
   artifacts out of bounds — so they must not be cited until regenerated.
6. **The `profile` spread remains available and remains capable of negative planning figures.**
   Flat is canonical and tested; if anyone switches to profile for a Treasury-facing number, the
   negative-day inheritance needs addressing first (clip before normalising, or state it).
7. **Carried from 2026-08-17:** `SelectionOnReportOnlyDataError` should get a base that survives
   handler reordering; and whether every default daily run should consult the holdout is
   deferred to the scheduled-daily-runs session.

---

## Suite

```
baseline at session start   856 passed,  4 skipped
after item 1 (+5 CLI)       861 passed,  4 skipped
after items 2-3 (+34)       895 passed,  4 skipped
after item 4 (+16 ops)      911 passed,  4 skipped
```

Runtime unchanged at 7:38. An early version of the Ops reader re-parsed the canonical CSV once
per vintage per target and resampled the full series once per *month*, which made the suite many
times slower; the parse is now memoised on (path, mtime, target) and the monthly frame computed
once per vintage. Worth recording because the reporting path is called from a great many tests,
so anything expensive there is paid repeatedly.

Item 3 added no tests — it changed one, so its effect is inside the totals rather than added.

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_forecast_modes.py -q      # 21 passed
./backend/.venv/bin/python -m pytest backend/tests/test_scorecard_schema.py -q    # 34 passed
./backend/.venv/bin/python -m pytest backend/tests/test_live_window.py -q         # 39 passed
./backend/.venv/bin/python -m pytest backend/tests/test_ops_baseline.py -q         # 16 passed
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q              # 911 passed, 4 skipped
```
