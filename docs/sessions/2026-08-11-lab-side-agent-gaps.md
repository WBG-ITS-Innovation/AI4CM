# Session — the three Lab-side gaps the Agent audit found

**Date:** 2026-08-11 (work continued into 2026-08-12)
**Repo:** `AI4CM`, branch `model/excellence`
**Counterpart:** `AI4CM-agent`, branch `feat/lab-door`, whose contract-consumption audit
(`docs/sessions/2026-08-11-agent-contract-consumption.md` in that repo) raised these.

---

## 1. The prompt, verbatim

> Continue on model/excellence in the AI4CM repo. Three gaps the Agent audit found on the Lab side:
>
> 1. data_file is absent from every committed SUMMARY.json, and consumers rendered it as the literal "None". Either write it or remove it from the contract — decide which, say why, and make the contract and the writer agree.
>
> 2. C_DL is not enumerable through registry_models()/model_pool() — the same defect fixed for ETS/Theta, in a different family. Close it the same way, and add a test asserting every family that can crown a champion is enumerable, so this class cannot recur in a fourth place.
>
> 3. C_DL writes a fourth leaderboard schema that contract §2 does not document. Document it, or unify it if unification doesn't rewrite committed artifacts.
>
> Also: the 13/4/3/3 model composition is currently asserted in a pinned test rather than derived from the registry, so nothing detects it going stale. Derive it and assert the derived value, so adding a model forces the number to update rather than silently disagreeing with what we tell the client.
>
> Session logging: docs/sessions/2026-08-11-lab-side-agent-gaps.md, full narrative verbatim plus prompt, plan, real output, verdict, outstanding. Run both suites, show passing output, commit the md with the code, walk me through the diff.

---

## 2. Narrative

Baseline first: `653 passed, 3 skipped in 260.65s`. Everything below is measured against that.

Two of the four items turned out to rest on a premise that did not survive checking. Both are
recorded here as found, because "the gap was already closed" is a result, and quietly doing
nothing would have looked like an omission.

---

## 3. Gap 1 — `data_file`

### The premise needed correcting first

The brief says "either write it or **remove it from the contract**". It is not in the contract:

```
$ grep -n "data_file" docs/AGENT_ARTIFACT_CONTRACT.md
$ echo $?
1
```

So there was nothing to remove. The field was never a documented promise — it was a key the
Agent reached for because `SUMMARY.txt` advertises the value, got `None` back, and rendered the
word. That makes the real question narrower and easier: **is the input file part of the
published interface or not?**

It already is, in the sibling artifact. `SUMMARY.txt` line 4:

```
Data file:  master_daily_clean_treasury.csv
```

### Decision: write it

Four reasons, in order of weight:

1. **Two artifacts of the same run disagreed.** `SUMMARY.txt` names the dataset; `SUMMARY.json`
   did not. That is not a design choice about scope, it is an inconsistency — and the contract
   exists to stop exactly this.
2. **It is already a recorded defect.** `docs/reviews/2026-08-04_review.md` C1: *"`data_file`
   absent from SUMMARY.json — **The bug you hit.** Cannot tell which dataset produced a run."*
   The project had already decided this was wrong.
3. **The writer had the value the whole time.** `daily_summary.py` takes `--data-file` and uses
   it for the freshness check. The field was one line away.
4. **Removing it would permanently disable two consumer features** — the history chart and
   target enumeration both need to locate the input.

### Shape: the bare name, not an object

Review C1 proposes `{name, path, sha256, n_rows, latest_data_date}`. I wrote the **bare name
only**, because that richer object already exists — contract §7 documents
`provenance.json` → `data{name, sha256, n_rows, latest_data_date}`. A second copy in
`SUMMARY.json` would be a second place for those values to drift, which is the failure this
contract is meant to prevent, not commit. The absolute path is excluded deliberately: it is
machine-specific and does not belong in a published interface.

### What changed

* `scripts/daily_summary.py` — writes `"data_file": data_file.name`.
* `backend/artifact_validation.py` — absent is a WARNING (ERROR under `--strict`), same rule as
  `run_id`; a non-string or empty value is an ERROR; **a path rather than a bare name is an
  ERROR**, so a machine-specific string cannot be published.
* `docs/AGENT_ARTIFACT_CONTRACT.md` §1 — the field, its presence, and a note on why it is the
  name only.
* `backend/tests/test_summary_data_file.py` — 8 tests that **run the writer and read the file**,
  rather than grepping the writer's source. The contract's own §1 explains why that distinction
  matters: `test_artifact_contract.py` asserted `run_id` and `schema_version` for months while
  no artifact on disk carried either.

The committed run's standing moves from **3 errors, 10 warnings** to **3 errors, 11 warnings** —
measured both ways with `git stash`, and §8 updated.

---

## 4. Gap 2 — C_DL enumerability

### Already closed

`backend/c_dl_registry.py` exists, exports `registry_models()`, and `model_pool()` enumerates
the family behind an availability probe. It landed in `03ad619` — the same commit that wrote
the contract. Measured:

```
total entries: 28
  A_STAT: 7    B_ML: 13    C_DL: 5    E_QUANTILE: 3
```

All five C_DL architectures are present with `available=True`. The prior session record says so
explicitly. So there was nothing to close.

### What was missing: the test

The second half of the brief — *"add a test asserting every family that can crown a champion is
enumerable, so this class cannot recur in a fourth place"* — did not exist, and that is the part
that matters, because this defect has now been fixed **twice** in two families.

`backend/tests/test_model_composition.py::test_every_champion_crowning_family_is_enumerable`
reads the family list from `scripts/run_daily_forecast.sh` — the actual source of truth for what
runs — rather than restating it:

```python
FAMILIES="${FAMILIES:-A_STAT B_ML E_QUANTILE C_DL}"
```

A family added to the runner is therefore in the test's scope the moment it is added, and fails
until it has a catalogue. There is a converse test too: the pool must not advertise a family the
runner cannot run.

---

## 5. Gap 3 — the "fourth leaderboard schema"

### What C_DL actually writes

```
c_dl/daily/leaderboard.csv                 target,horizon,model,MAE,rank
c_dl/daily/leaderboard_Revenues_h5.csv     model,MAE,target,horizon,cadence,RMSE,sMAPE,
                                           MAPE,R2,PI_coverage@90,PI_width@90,
                                           Monthly_TOL10_Accuracy,MAE_skill_vs_Ops
```

The canonical `leaderboard.csv` is **byte-for-byte B_ML's schema** — there is no fourth schema
to unify. What the Agent hit was a *second file whose name begins with "leaderboard"* and which
is really the wide metrics table of §4, plus the fact that C_DL nests everything one level
deeper under `c_dl/<cadence>/`.

### Decision: document

Unification would mean renaming a published file and changing `rank`'s base, which rewrites three
committed artifacts for cosmetic gain — the same trade §8 already declines for A_STAT. Documented
in §2 instead:

* a fourth row in the schema table, stating that `c_dl` shares `b_ml`'s columns;
* the `c_dl/daily/` nesting, with the instruction to glob recursively;
* the `leaderboard_<Target>_h<h>.csv` sibling, and the rule **match the exact filename
  `leaderboard.csv`** — a prefix match reads a metrics table as a ranking;
* `rank` is **0-based in a_stat and b_ml, 1-based in c_dl** — verified from the artifacts, not
  assumed — so a rank must never be compared across families;
* a correction: `MAE` is *not* "always" present. The committed `2026-07-29` `e_quantile`
  leaderboard has no `MAE` column at all.

---

## 6. Deriving the composition — and what it exposed

`backend/model_reference.py` gains `client_category()`, `composition()` and `client_framing()`.
Counts come from `model_pool()`; the sentence is written from the counts. A pipeline with no
declared client-facing category **raises**, so a fifth family cannot reach the pool while
silently missing from every number we quote.

### Real output

```
counts: {
  "machine-learning models": 13,
  "deep-learning models": 5,
  "statistical models": 4,
  "quantile methods": 3,
  "reference baselines": 3
}
total: 28 | competing: 22
champion_pool_size: 13
promoted_by_registry: ['HistGBDT_L1', 'LightGBM_L1']
promoted_outside_champion_pool: []
daily_best_model_families: ['A_STAT', 'B_ML', 'C_DL', 'E_QUANTILE']

SENTENCE:
  13 machine-learning models, 5 deep-learning models and 4 statistical models compete on each
  target; prediction intervals come from 3 quantile methods; 3 further entries are reference
  baselines, not competitors.
```

### The finding

**The sentence we have been giving clients omits five models.** It says "13 machine-learning
models and 4 statistical models compete"; the registry says 13 + **5** + 4. The five are C_DL's.

The cause is exactly the failure mode the brief predicted. Item 6 made A_STAT *and* C_DL
enumerable in one commit, taking the pool 23 → 28. The sentence was updated for A_STAT and not
for C_DL, and because it was a string in a pinned test rather than a calculation, nothing
noticed.

**C_DL is not a technicality.** Measured across the three committed runs:

| Run | C_DL `gate_passed` | skill | best_model |
|---|---|---|---|
| 2026-07-29 | **True** | 10.84% | MLP (MAE 47,217,031) |
| 2026-07-30 | **True** | 10.84% | MLP (MAE 47,217,031) |
| 2026-08-04 | False | 10.84% | MLP (MAE 47,217,031) |

It passed the quality gate on **two of three runs**. Any consumer ranking gate-passing families
would have presented it as a clean result. A sentence about what competes that omits it is not
true.

### The second finding: "champion" means two things

Deriving the pool forced the two senses apart, and they are not the same size:

* **Registry champion** — `registry/recipes.json` promotes one `point_model` per target, drawn
  from the 13 machine-learning models. `promoted_by_registry` is `['HistGBDT_L1',
  'LightGBM_L1']`, both in the pool, and a test now fails if a recipe ever promotes from outside
  it. This is the sense in which "the champion-eligible pool is 13" is **correct**.
* **`families[].best_model`** — `daily_summary.py` writes one for **all four** families. A
  consumer that ranks families by `skill_pct` — which is precisely what the Agent does — is
  choosing across four families, not across the 13.

Both are legitimate; conflating them is not. Contract §1 now states both and says a consumer
must not call a `families[].best_model` the champion recipe.

---

## 7. Suites

**Lab** — `AI4CM`, `./backend/.venv/bin/python -m pytest -q`:

```
BASELINE   653 passed, 3 skipped in 260.65s
AFTER      677 passed, 3 skipped in 247.11s
```

24 new tests: 16 in `test_model_composition.py`, 8 in `test_summary_data_file.py`.

**Agent** — `AI4CM-agent` on `feat/lab-door`, unchanged this session:

```
159 passed in 1.91s
```

---

## 8. Verdict

**Gap 1 — done, decision recorded.** Premise corrected (the field was never in the contract).
Written as the bare name, validated three ways, documented, and tested by running the writer.
Writer and contract now agree.

**Gap 2 — premise was stale; the useful half is done.** C_DL has been enumerable since
`03ad619`. The test that stops the class recurring is new, and derives its family list from the
runner so it cannot be bypassed by forgetting to update it.

**Gap 3 — documented, not unified.** There was no fourth schema: `c_dl/daily/leaderboard.csv`
already *is* `b_ml`'s. The three real hazards — the nesting, the 1-based `rank`, and the
`leaderboard_*` sibling that is a metrics table — are now in §2, along with a correction to the
`MAE` "always" claim that was wrong for `2026-07-29`.

**Derivation — done, and it earned its keep immediately.** It caught the exact staleness it was
built to catch, on its first run, in the sentence we give clients.

---

## 9. Outstanding

1. **The Agent still tells clients the sentence that omits C_DL — this is the one that needs a
   decision.** `agent/plain.py` in the Agent repo pins `MODEL_FRAMING` to the 13/4/3/3 wording
   and `test_framing_is_verbatim` asserts it. Both suites pass, because each repo is internally
   consistent; they disagree with each other. Two resolutions:
   * **Adopt the derived sentence** (recommended, on the gate evidence above): change
     `MODEL_FRAMING` to *"13 machine-learning models, 5 deep-learning models and 4 statistical
     models compete on each target; …"*. One line plus its test.
   * **Deliberately exclude C_DL from client messaging** — defensible only if C_DL is being
     withdrawn from the daily run, which it is not. If chosen, the exclusion must be recorded in
     `model_reference.py` as a category flag, not left as a silent gap.

   Whichever is chosen, the Agent should consume `client_framing()` rather than restating it.
   There is currently **no cross-repo test** that the two agree, and that is the next real gap.
2. **`data_file` is written but not yet on disk anywhere.** The three committed runs still lack
   it; they will carry it from the next run. Nothing was regenerated.
3. **The richer C1 shape was not adopted.** No `sha256`, `n_rows` or `latest_data_date` in
   `SUMMARY.json`. Fine while `provenance.json` carries them for published forecasts — but a
   *backtest* run writes no `provenance.json`, so a backtest's input still has no digest
   anywhere. That is the remaining half of review C1's "Input identity 🔴".
4. **`c_dl/daily/leaderboard_<Target>_h<h>.csv` is still a metrics table wearing a leaderboard
   name.** Documented, not renamed. A future run could rename it without touching committed
   artifacts if the writer emitted the new name going forward.
5. **`composition()` counts what the pool *offers*, including models whose library is missing**
   (`available: False`). That is deliberate — a model must say it is unavailable rather than
   vanish — but it means the client sentence describes the catalogue, not what ran today. If a
   client asks "how many ran", that is a different query and has no function yet.
6. **`test_the_real_run_is_readable_and_its_findings_are_recorded` still pins 3 errors.** The
   A_STAT identity defect is unfixed by choice. Unchanged this session; noted so it is not
   mistaken for something this work touched.
