# Session — making the repository presentable to a Treasury reader

**Date:** 2026-08-19 · **Branch:** `model/excellence` · **Repo:** Lab only (`AI4CM`); the agent
repo was **read** (its `README.md` and `docs/sessions/README.md`, as the structural model for this
work) and not modified.
**Constraints honoured:** documentation only. No behaviour change, no source edit, no artifact
touched, no model run. Plan approved before any file was written.

Three tasks, all landed.

| | Task | Outcome |
|---|---|---|
| 1 | Audit what a first-time reader sees | README five months stale; **four durable documents actively contradict the code** |
| 2 | Rewrite `README.md` for the two audiences | done — trust architecture, measured headline with caveats attached, current state and known limits |
| 3 | Sessions index; stale docs marked rather than left silently wrong | `docs/sessions/README.md` added; five documents given dated notices |

> **The finding that matters most in this record is in Task 1.** The stale README was the
> *expected* problem. The unexpected one is that `CHANGELOG.md` and `docs/SIGNAL_FINDING.md` do not
> merely lag the code — they state the opposite of it, in the two places a reader would most
> reasonably look for the project's status and its central scientific finding.

---

## Task 1 — what a first-time reader actually sees

### The README

Last substantive edit **2026-03-31**. It opens: *"a local forecasting sandbox for Treasury
time-series … designed for hands-on exploration, model comparison, and capacity building."*

That was accurate when written. Everything that makes this repository defensible was built after
it, and none of it appears: no publication gates, no four-window split, no sealed holdout or its
ledger, no OFFICIAL/EXPLORATORY separation, no published issues, no scorecard, and no measured
result of any kind. A Treasury reader would have found a model-comparison toy; a reviewer would
have had no entry point to the trust machinery.

Two concrete errors on top of the framing:

* it instructs the reader to clone `georgia-treasury-prototype.git`, which is **not** the remote
  (`WBG-ITS-Innovation/AI4CM.git`);
* it advises avoiding Python 3.13, while **both venvs run 3.13.11** and the suite is green on it.

### `User_Guide.pdf` — the premise was out of date, and in the good direction

The brief asked whether it was stale. It is **absent**. Tracked history:

```
33ddc37  User Guide                                    (added)
f385c11  Rename Georgia_Treasury_..._User_Guide.pdf to User_Guide.pdf
576ea5f  Remove datasets from repo; load from secure storage   (deleted, 2026-08-14)
```

`2026-08-13-public-exposure-audit.md` §4 had already established what it was: **a 2-byte stub**
whose entire tracked content is `\r\n`. Not a PDF, no figures, nothing ever committed into it. It
was removed with the datasets. No tracked file references it. **Nothing to do** — recorded here so
the question is not re-opened.

### `docs/` — no index, and four documents that contradict the code

23 session records with no index over them. More seriously:

| File | Claims | Actually |
|---|---|---|
| `VERIFICATION.md` §1 | "**Expected:** `225 passed`" | **942 passed, 4 skipped** |
| `CHANGELOG.md` (Unreleased) | "no modelling has been done; no per-target candidates exist; the TEST window has not been read (`experiments/test_access.log` is empty)" | three registered champions; the ledger holds hundreds of reporting reads |
| `docs/ROADMAP.md` (status 2026-08-05) | "2025 sealed, **zero** evaluations"; "29–41% lower error" | sealed window evaluated and reported under `PURPOSE_REPORT`; figures superseded 2026-08-18 |
| `docs/SIGNAL_FINDING.md` (2026-08-05) | "**TEST (2025) reads: 0**"; the stock target forecasts genuinely while the two flows do not | **inverted** by the 1.50 → 1.15 sentinel recalibration: Revenues is `publishable`, the stock target is `withheld` on accuracy |

The last row is the one worth pausing on. `SIGNAL_FINDING.md` is the document a technical reader
would treat as the project's central scientific claim, and its headline conclusion is now backwards
relative to the gates — not because the finding was wrong when made, but because the threshold it
was read against was uncalibrated and has since been calibrated. A reader with no other context
would draw exactly the wrong conclusion about which target to trust.

### One stale string deliberately left alone

`registry/recipes.json` carries `"windows": {"test": "2025 -- SEALED, never evaluated"}`. That is
false — the holdout has been evaluated and logged. It is **not** edited here: the registry is a data
file whose contents are asserted against `publication_gates` by test, this session's scope is
documentation, and changing it belongs with whoever next re-derives the registry. Recorded as an
open item rather than fixed in passing.

---

## Task 2 — the README

Rewritten around the same two-audience structure the agent repository uses, so a reader moving
between the two repositories meets the same shape. 489 lines.

**What it now says, in order:** what the Lab is (three targets, their champions, their verdicts) →
what it will not do → **what has been measured** → how trust is enforced → the four families →
setup → running a forecast and reading `gates.json` → current state → known limits → layout.

### The measured headline, stated as the records state it

The sealed-window table is reproduced with **every caveat attached to it rather than footnoted** —
that Expenditure's +2.79% is a margin a different window could erase and its verdict is `withheld`
anyway; that the stock target has no comparison to the current method at all; that n=146 rather
than 156; that these measure the champion *recipe* and not a continuation of credentials that
cannot be reproduced; and that **every pre-2026-08-18 "N% better than the Treasury's current
method" figure is withdrawn**, the July deck included.

### Figures discipline — percentages and MASE only

**Decided and confirmed before writing.** No absolute lari amount appears in the README. The
public-exposure audit's tier (a) item 8 lists the Markdown files' lari-scale figures as attributed
client performance figures requiring redaction, and observes that ratios and percentages carry the
argument without disclosing amounts. The README is the most-read file in a repository that pushes
to a World Bank organisation, so it takes the conservative side of a judgement the data owner has
not yet made. Absolute figures stay where they already live:
`reports/sealed_window_champion_vs_ops.csv` and the session records.

### Language discipline

*"Honestly evaluated"*, never *"proven in production"* — carried verbatim from the agent
repository's README, including the sentence that says the document will not claim otherwise until
forecasts have been scored against real outcomes. **Zero** is stated as the number of forecasts
scored so far, in the Current state table, in bold.

### Claims verified rather than transcribed

Everything asserted about the code was checked against the code, and four drafting errors were
caught that way:

* the layout said "Overview + 6 pages"; there are **7** (`frontend/pages/*.py`);
* the coverage gate was described as "band must cover at its nominal rate"; the implemented
  threshold is nominal **±10 points**, so it now says so;
* the `gates.json` example wrapped a string across three lines, which is not valid JSON. A README
  that shows a reader how to read an artifact should not show them something that would not parse;
  the block is now checked by script;
* two session records were cited by bare filename and now carry their `docs/sessions/` prefix.

The CLI form in the README was taken from `05_Forecast.py`'s own dispatcher and its `--help`
checked, rather than reconstructed from the argparse source. Every relative link and every
backtick-quoted path in the six touched files resolves — verified by script, 0 broken.

---

## Task 3 — the index, and the notices

### `docs/sessions/README.md`

One row per record, 24 rows including this one, each saying what the session covers rather than
repeating its title.
Mirrors the agent repository's index so the two read alike. Adds two things a bare list cannot
carry: that the suite counts scattered through the records are a growth curve from 569 to 942, and
an explanation of the **inverted pin** convention — a test that passes while a defect exists and
fails once it is fixed — which a reader will otherwise meet without warning in three separate
records.

### Dated notices, not rewrites

Five documents were given a short notice directly under their title. **The distinction that
governed every one of them: the currency claim is removed, the reasoning is kept.**

| File | What the notice says |
|---|---|
| `CHANGELOG.md` | the three false status claims, named; figures of the withdrawn form are withdrawn |
| `VERIFICATION.md` | 225 → 942; the dataset must be restored from secure storage; **the method is still how this project verifies itself**, and each session record's Reproduction section is the current equivalent |
| `docs/ROADMAP.md` | the status table is superseded on two counts; **the sequencing and its argument are not** |
| `docs/SIGNAL_FINDING.md` | the conclusion is inverted by the recalibration, stated plainly; **the method — the three-instrument robustness study, and the event-forecast vs central-tendency-guide distinction — is not superseded** |
| `docs/EXECUTION_PLAN.md` | historical; Phase 1 is done and its branch is merged; kept for decisions D1–D8 |

Marking rather than deleting is the point. Each of these documents is the record of a decision made
with the evidence available at the time, and a history that has been tidied is harder to trust than
one that admits what it got superseded by.

---

## What was deliberately not done

* **No document body was rewritten.** Only `README.md` was replaced; the other five keep every word
  and gain a notice.
* **`registry/recipes.json` was not touched** — see Task 1.
* **No absolute lari figure was added anywhere**, and none was removed from where it already lives.
* **`docs/AGENT_ARTIFACT_CONTRACT.md`, `docs/DATA_SEMANTICS.md` and
  `docs/FISCAL_CALENDAR_SOURCES.md` were checked and left alone** — all three still describe what
  the code does and what is open with Treasury.
* **`reports/` was not audited.** It holds 43 files including the withdrawn July-era figures. The
  session records that withdraw them are authoritative, but a reader browsing `reports/` directly
  would not know that. Recorded as an open item — and, since it could not be closed here, the
  README's pointer to `reports/` carries an explicit "read with the date in mind" caution rather
  than sending a reader in unwarned.

---

## Open items

1. **`registry/recipes.json` says the holdout was never evaluated.** A one-string correction, but it
   belongs with a registry re-derivation rather than a docs pass, because the file is asserted
   against `publication_gates` by test.
2. **`reports/` carries withdrawn figures with nothing marking them.** Ten-plus workstream and phase
   records quote "N% better than the Treasury's current method" values computed against the zero
   baseline. The correction is recorded in the session records; the reports themselves say nothing.
   Same treatment as this session gave `docs/` would close it.
3. **The public-exposure audit's remediation is still unimplemented.** Its tier (a) and (b) plans —
   synthetic data substitution, prose redaction, history rewrite — remain open. This session took
   the conservative side of that judgement for the README only.

---

## Suite

```
before   942 passed, 4 skipped
after    942 passed, 4 skipped
```

Unchanged, and expected to be: no test asserts on `README.md` or on any document touched here. The
suite was run in full nonetheless, because `test_session_records_name_no_third_parties.py` greps
**every tracked text file**, and this session added several hundred lines of new prose to that surface.

---

## Reproduction

```bash
./backend/.venv/bin/python -m pytest backend/tests/test_session_records_name_no_third_parties.py -q
./backend/.venv/bin/python -m pytest backend/tests frontend/tests -q     # 942 passed, 4 skipped
```
