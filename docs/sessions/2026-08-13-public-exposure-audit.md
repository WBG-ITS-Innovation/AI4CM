# Public exposure audit — what is currently visible in a public clone

**Audit run:** 2026-08-14 (filename per the request) · **Branch:** `model/excellence` @ `f48164a`
**Remote:** `https://github.com/WBG-ITS-Innovation/AI4CM.git` · **Nothing was changed.**
Read-only throughout: no file modified, no history touched, no secret value printed.

---

## 0. Headline

**The complete Georgian Treasury daily series is public, in four copies, plus its two raw source
Excel workbooks.** 3,867 rows × 44 columns, 2015-01-05 → 2025-08-06, values to **GEL 3,038,342,979**.

It is not merely in `HEAD`. It entered in commit **`67a1625` (2025-11-13)** and that commit is an
ancestor of **all 21 pushed branches**, so every clone made in the last nine months contains it.
Deleting the files now changes nothing about what has already been distributed.

There is **no LICENSE** and no statement of data ownership or permission anywhere in the repo.

The good news, stated as plainly as the bad: **no credential was found.** No private key, no API
token, no internal hostname, no password. The two `.env` files are benign path configuration, and
the alarming-looking `corp_bundle.pem` is a standard public CA bundle.

---

## 1. Every tracked file containing Treasury values

### A. Raw source workbooks — real client files, verbatim

| Path | Size |
|---|---|
| `backend/data/Balance_by_Day_2015-2025.xlsx` | 1,040 KB |
| `backend/data/Balance_by_Day_2022-2025.xlsx` | 237 KB |
| `frontend/data/Balance_by_Day_2015-2025.xlsx` | 1,040 KB |
| `frontend/data/Balance_by_Day_2022-2025.xlsx` | 237 KB |

These are the files as received. Duplicated across `backend/` and `frontend/`.

### B. Full processed dataset — six copies

`backend/data/processed/` and `frontend/data/processed/`, each containing:
`master_daily_clean_treasury.csv`, `master_daily_clean_conservative.csv`, `master_daily_raw.csv`.

Measured on each: **3,867 rows, 44 columns, 2015-01-05 … 2025-08-06, max |value| 3,038,342,979**.
1,058–1,666 KB per file.

### C. Real row extracts with Georgian column headers — 7 files

`backend/test_outputs/runs/*/preprocess_preview.csv` (7 files, incl. two under
`test_verification/`). Each holds **500 real rows**; the `test_clean_treasury` one spans
2022-01-03 … 2023-05-17 with max |value| **2,255,320,440** and native headers
(`შემოსულობები`, `შემოსავლები`, `გადასახადები`, …).

The 7 sibling `preprocess_report.json` files carry **no** lari values, but four of them embed a
local developer path.

### D. Derived figures — real amounts in lari, and gate verdicts

| What | Where | What it exposes |
|---|---|---|
| Run summaries | `backend/forecast_runs/{2026-07-21,2026-07-29,2026-07-30,2026-08-04,2026-08-12}/SUMMARY.{json,txt}` (9 files) | `best_model: "RandomForest (MAE 38,565,591)"`, `skill_pct: "36.75%"`, gate verdicts, `data_file` name |
| Published forecasts | `forecasts/published/2025-08-06/forecast.csv`, `forecasts/published/2026-08-13/forecast.csv` | `origin_value 46,490,793.48`, and p10/p50/p90 per horizon |
| Gates / manifests / provenance | `forecasts/published/*/{gates.json,manifest.json,provenance.json}`, `.../estimators/manifest.json` | measured gate values, data SHA-256, git SHA, feature names |
| Experiment log | `experiments/log.csv` | 153 rows; `dev_mae` from **22,712,006 to 272,029,958** lari, plus MASE, skill, sentinel |
| Per-run detail | `experiments/runs/*.json` (152 files) | per-run metrics and parameters |
| Registry | `registry/recipes.json` | champion recipes, DEV credentials, gate measurements, publication verdicts |
| Scorecard | `forecasts/scorecard.csv` | **header only — no data** |

### E. Prose quoting lari-scale figures — 37 tracked Markdown files

Highest density: `docs/reviews/2026-08-04_review.md` (148 grouped figures),
`docs/sessions/2026-08-13-p0-scoring-correctness.md` (55), `reports/model_audit_2026-07-21.md` (43),
`reports/ws1_objectives.md` (42), `reports/ws3_fiscal_calendar.md` (41), then `ws4_target_scaling`,
`ws4_robustness`, `ws5_multivariate`, `HANDOFF.md`, `CHANGELOG.md`, and most `docs/sessions/*`.

These are aggregate error figures and skill percentages, not row-level data — a materially lower
sensitivity than A–C, but they are Treasury performance figures attributed to a named client.

### F. Not Treasury data

`frontend/sample_data/cash_demo.csv` and `quick_sample.csv` are **synthetic** (generic column names,
monthly 2019, values in the tens of thousands). `backend/data/holidays_georgia_2015_2025.csv` and
`frontend/data/holidays_georgia_2015_2025.csv` are **public** holiday calendars.

---

## 2. Git history

**Everything found is in `HEAD` *and* in history. Nothing was ever deleted, so nothing is hidden.**

```
git log --all --diff-filter=A -- backend/data/* frontend/data/* *.xlsx *master_daily*
  -> 67a1625  2025-11-13  omakhlouk
     (all 12 data paths added in this single commit)

git merge-base --is-ancestor 67a1625 origin/main   -> YES
remote branches containing 67a1625                 -> 21 of 21
distinct blob versions of master_daily_clean_treasury.csv -> 2
files added-then-deleted in history (csv/xlsx/db/env/pem)  -> 0
git count-objects -vH  -> size-pack 6.48 MiB
```

So: the data has been in every clone of every branch since **2025-11-13** — roughly nine months.
Removing it from `HEAD` would leave it fully retrievable.

---

## 3. Credentials, endpoints, internal paths

**No credential was found.** Reported with the same confidence as the bad news:

| Item | Finding |
|---|---|
| `backend/.env` (tracked, 139 B) | 4 assignments: `DATA_MASTER`, `HOLIDAYS_CSV`, `OUTPUTS_ROOT`, `MPLBACKEND`. All relative paths / a matplotlib flag. Longest value 37 chars. Zero credential-shaped names, zero URLs |
| `frontend/.env` (tracked, 157 B) | 2 assignments: `TG_BACKEND_PY`, `TG_BACKEND_DIR`. Longest value 76 chars. Zero credential-shaped names |
| `frontend/certs/corp_bundle.pem` (tracked, 289 KB) | **146 `BEGIN CERTIFICATE` blocks, 0 `PRIVATE KEY`.** A standard public root-CA bundle (Entrust, DigiCert, CommScope, FNMT …) — certifi-style. Not a secret. Its presence implies a TLS-intercepting corporate proxy, which is a weak inference about environment, not a disclosure |
| `.cache/cache.db`, `data/experiments.db` (tracked) | Empty of records: `Cache` 0 rows, `datasets` 0 rows, `runs` 0 rows. Only 16 diskcache `Settings` rows. Clutter, not data |
| Internal hostnames / endpoints | **None.** The only org/gov domains are `nbg.gov.ge` and `mof.ge`, appearing as *public-source citations* in `backend/preprocessing/fiscal_calendar.py`, `docs/FISCAL_CALENDAR_SOURCES.md`, `reports/HANDOFF.md`. No `worldbank.org`, no SharePoint, no RFC1918 address, no internal FQDN |
| Local developer paths | 7 tracked files embed `/Users/<name>`: `README.md`, `backend/requirements.txt`, four `preprocess_report.json`, `backend/tests/test_summary_data_file.py`, `frontend/pages/00_Data_Preprocessing.py`, `frontend/.env`, plus one session doc. Minor: discloses a username and machine layout |

Note the standing instruction was honoured: **no `.env` value was printed at any point.** They were
classified by name pattern, value length, and shape only.

---

## 4. `User_Guide.pdf` and `reports/`

**`User_Guide.pdf` is a 2-byte stub** — its entire tracked content is `\r\n`. It is not a PDF and
contains no figures at all. Whatever was intended for it was never committed.

**`reports/` does contain client figures**, in prose: 43 grouped lari-scale figures in
`model_audit_2026-07-21.md`, 42 in `ws1_objectives.md`, 41 in `ws3_fiscal_calendar.md`, and so on
across the workstream and session records. These are model error figures, skill percentages and
sentinel ratios — aggregate, attributed, and interpretable as Treasury forecasting performance.

---

## 5. The plan, in three tiers

Nothing below is implemented.

### (a) Must be removed before this stays public

| # | What | Why |
|---|---|---|
| 1 | The 4 `.xlsx` source workbooks | The client's files verbatim |
| 2 | The 6 `master_daily_*.csv` | Complete 10.5-year daily series, values to GEL 3.04bn |
| 3 | The 7 `preprocess_preview.csv` | 500 real rows each, native Georgian headers |
| 4 | `forecasts/published/*/forecast.csv` (2) | Real `origin_value` and forecast amounts |
| 5 | The 9 `SUMMARY.{json,txt}` | MAEs in lari embedded in prose, skill percentages |
| 6 | `experiments/log.csv` + 152 `experiments/runs/*.json` | 153 rows of lari-scale error figures |
| 7 | `registry/recipes.json` gate measurements | Real DEV MAE / MASE / sentinel per champion |
| 8 | The 37 Markdown files' lari-scale figures | Attributed client performance figures. Redact the numbers; the reasoning is the asset and can stay |

**And immediately, independent of any file change: make the repository private.** See §6.

### (b) Should be replaced with synthetic equivalents so the repo still works

The repo must remain runnable and testable. Each of these has a working substitute:

| Replace | With | Why it works |
|---|---|---|
| `master_daily_clean_treasury.csv` | A synthetic generator producing the same 44-column schema, same business-day index, same magnitudes and seasonality — committed as a script plus a fixed-seed CSV | Every test that reads it needs *shape and scale*, not real values. The eligibility, window, ruler and scoring tests are all schema-driven |
| The `.xlsx` workbooks | A tiny synthetic workbook exercising the Excel→daily converter | `preprocess_data.py` needs a real `.xlsx` structure, not real numbers |
| `preprocess_preview.csv` × 7 | Regenerate from the synthetic input | They are test *outputs*; they should never have been committed as fixtures |
| `experiments/log.csv` + run JSONs | A synthetic log with the same 19 columns and plausible magnitudes | `verify_log_integrity()` checks row↔JSON correspondence, not values. **Caveat: this breaks the audit trail.** The real log is genuine provenance and should be preserved privately, not destroyed |
| Published issues | Re-issue against the synthetic dataset | Preserves the contract, the reconciliation and the scorecard machinery |
| `registry/recipes.json` measurements | Re-derive from synthetic runs via `publication_gates.decide()` | The logic is already the single source; the numbers are its output |
| The 37 Markdown files | Replace absolute lari with relative statements ("36.8% below the benchmark", "MASE 0.758") | Ratios and percentages carry the argument without disclosing amounts. **This is a judgement call for the data owner**, not mine: even a percentage attributed to a named treasury may be restricted |

Sequencing that keeps the repo green: build the generator → point tests at it → regenerate
derived artifacts → redact prose → then purge, with both suites run after each step.

### (c) Safe to keep

- All 145 `.py` source files, all tests, `scripts/`
- `frontend/sample_data/*.csv` — already synthetic
- `holidays_georgia_2015_2025.csv` (both copies) — public information
- `frontend/certs/corp_bundle.pem` — public CA bundle, no private key (better still: replace with
  `certifi` and delete 289 KB of vendored roots)
- `docs/` and `reports/` **reasoning, method and architecture** — the genuine intellectual asset,
  and the reason redaction beats deletion
- `forecasts/scorecard.csv` — header only
- The two `.db` files are safe but should be untracked as clutter
- `User_Guide.pdf` — a 2-byte stub; delete as noise

---

## 6. History: rewrite, or make it private?

**Both, in that order — and rewriting alone would be false comfort.**

The facts: the data is reachable from all 21 pushed branches via `67a1625`, dated **2025-11-13**. The
repository has been public for approximately nine months with the client's complete dataset in it.

**A history rewrite is technically easy.** The pack is 6.48 MiB; `git filter-repo --path-glob` over
the twelve data paths would take minutes, and there are no added-then-deleted blobs to chase.

**But a rewrite does not undo disclosure, for three reasons:**

1. **Anyone may already hold a clone.** Nine months of public exposure cannot be retracted. There is
   no way to know whether it was cloned, and the absence of evidence is not evidence of absence.
2. **GitHub retains unreachable objects.** After a force-push, blobs stay fetchable by SHA through
   the web UI and API until garbage collection, which is not immediate and not under your control.
   GitHub Support must be asked explicitly to purge them.
3. **Forks are permanent.** Any fork keeps the objects, and a rewrite of the origin does not touch
   a fork. Network-wide removal requires GitHub Support.

**Recommended order:**

1. **Make the repository private now.** One click, no coordination, stops further exposure. Do this
   before any file change — a purge commit on a public repo advertises exactly what to look for in
   the history.
2. **Notify whoever owns the data** (the Treasury counterpart and the WBG task lead) that the
   dataset was publicly accessible from 2025-11-13. Treat it as disclosed. This is theirs to assess,
   not ours to minimise.
3. **Then** do the tier-(b) synthetic replacement and the tier-(a) purge, on the private repo.
4. **Then** `git filter-repo`, force-push all 21 branches, and open a GitHub Support request to
   expunge unreachable objects and enumerate forks.
5. Only consider public again once the working tree contains no client data, history is clean, a
   LICENSE and a data-provenance statement exist, and the owner has agreed in writing.

**If the choice is one or the other: make it private.** A rewrite on a still-public repo protects
future clones at best, while creating a precise index of what to retrieve from the old history.

---

## 7. One open question I could not resolve

`README.md` lines 159 and 166 instruct cloning
`https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git`, a **different repository**
from this remote. If that repo exists and shares this history, **the same dataset is exposed there
too**, and remediating `AI4CM` alone would be insufficient. `gh` is not authenticated in this
environment, so I could not check its existence or visibility. **This should be checked first** — it
may change the scope of everything above.

---

## 8. Verdict

The exposure is **client data, not credentials**. No key, token, password or internal endpoint was
found, and I want that stated as clearly as the rest so effort goes where the risk actually is.

What is exposed is the whole of it: two raw workbooks, six copies of the full processed series,
seven 500-row extracts with native headers, real published forecast amounts, 153 logged error
figures in lari, and 37 documents quoting Treasury performance — public since 2025-11-13, reachable
from all 21 branches, with no licence and no permission statement.

Deleting the files is not remediation. **Make it private first**, treat the data as disclosed, tell
the owner, and only then rewrite. And check `georgia-treasury-prototype` before doing anything else.

---

## 9. Outstanding

* **Confirm whether `WBG-ITS-Innovation/georgia-treasury-prototype` exists and is public.** Blocks
  scoping.
* **Confirm the repository's actual current visibility via the API.** I relied on the statement that
  it is public; `gh` was unavailable, so I did not verify it independently.
* No synthetic generator exists yet — tier (b) is a design task, not a deletion.
* Whether aggregate percentages attributed to the Treasury are themselves restricted is the data
  owner's call and determines how deep the prose redaction goes.
* `.gitignore` lists `experiments/test_access.log` and other paths, but `backend/.env`,
  `frontend/.env`, `.cache/cache.db` and `data/experiments.db` are tracked *despite* being the kind
  of file the ignore rules anticipate — tracking always wins over ignoring, which is worth a
  one-line check in CI.
* The Phase-1 instruction to untrack `.venv` and `.env` was never completed; it has been carried in
  the outstanding list since Phase 1 and is now part of this audit's tier (a).
