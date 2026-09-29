# Why the forecasting tab looks different to the client

**Date:** 2026-09-29
**Scope:** investigation only. Read-only throughout — no commits, no pushes, no
file changes anywhere in the repository.
**Outcome:** nothing needed syncing. The code was already identical. The
difference is data that is withheld from every clone by design.

---

## The question

The forecasting tab rendered differently locally than it did for the client. The
working assumption was drift: unpushed local commits, or a stale deployment
serving an old build. The task was to find the drift and push it.

**There was no drift.** The assumption was wrong, and the real cause is a
property of how this repository is built.

## What the investigation found

### Local and remote are the same commit

| Check | Result |
| --- | --- |
| Branch | `main`, tracking `origin/main` |
| Working tree | clean |
| Ahead of `origin/main` | 0 |
| Behind `origin/main` | 0 |
| Stash entries | none |

`frontend/pages/07_Forecast.py` was diffed against the `origin/main` version
after fetching. **The diff is empty — the file is byte-identical.** No local
commits exist that are not on `origin/main`.

### No unmerged work, and nothing is deploying

- `origin/main` is the **only** remote branch. No unmerged frontend work exists
  on any branch.
- `backup-pre-strip-20260820` and `backup-local-main-20260819` are **local-only**
  backups from the history rewrite, with no upstream. Nobody else can see them.
  The first is content-identical to `main` under `frontend/`.
- **There is no deployment configuration in the repository at all** — no
  Dockerfile, no compose file, no CI workflows, no cloud manifest. No stale build
  is being served, because nothing is being served. The app is run locally from a
  clone.

### The actual cause: the data does not travel

The 2026-08-13 sanitization moved the real Georgian Treasury data into
`private_vault/` and added ignore rules so that no clone of this World Bank
repository carries client data. See `.gitignore`, which documents each rule and
its reasoning at length.

The forecast page reads `forecasts/published/`
(`frontend/pages/07_Forecast.py:73-74`). That directory is ignored. So are the
run artifacts the page depends on. The gap:

| Directory | This machine | A fresh clone |
| --- | --- | --- |
| `forecasts/published/` | 54 files | 0 |
| `frontend/runs/` | 401 files | 0 |
| `backend/forecast_runs/` | 288 files | 0 (bar `SUMMARY.json`) |
| `experiments/runs/` | 153 files | 0 |
| `private_vault/` | 731 files | 0 |
| `data_preprocessed/` | 2 files | 0 |

**The operator sees a populated app. The client sees the same code with empty
states.** This is verified, and it is expected behaviour rather than a bug — the
data is withheld deliberately.

### A second possibility, not ruled out

The client's clone may simply be old. Measured against
`backup-local-main-20260819`, current `main` **adds 923 lines** to
`07_Forecast.py`, which would be a dramatic visible difference on its own.

This cannot be settled from here, because it depends on the client's machine. To
distinguish the two causes, ask the client to run:

```bash
git rev-parse --short HEAD
```

If the answer is `1f53b36`, their code is current and the difference is purely
missing data.

## The history rewrite

`main`'s history was rewritten on 2026-08-20 (see
`2026-08-20-strip-coauthor-trailers.md`). The rewrite is visible in the fact that
one commit now carries two hashes — pull request #28 is `675bbe6` in the
pre-rewrite lineage and `b988d42` on `main` today.

The pre-rewrite tip was checked and **is not an ancestor** of `origin/main`.

**Any clone created before 2026-08-20 cannot fast-forward.** `git pull` will
either fail or produce a spurious merge. Such a clone needs to be deleted and
cloned fresh.

## Actions taken

**None beyond reading.** No push was made, because there was nothing to push —
zero commits ahead of `origin/main` with a clean tree. Any push would have been a
no-op. No artifact in the working tree was created, modified, or regenerated.

## Recommendation carried forward

Branch protection on `main` is currently off. That is a live risk independent of
this task: this repository's history has already been rewritten once, and an
unprotected `main` is what allows that to happen by accident.

Since no push was required, protection can be restored immediately. On GitHub:
**Settings → Branches**, add or edit a ruleset for `main`, and enable **Restrict
force pushes** and **Restrict deletions**. The live state can then be confirmed
with:

```bash
gh api repos/WBG-ITS-Innovation/AI4CM/branches/main/protection
```

## Facts for the message to the client

- **Current `main` SHA:** `1f53b36`
  (`1f53b369c9c75bb2980566dd85594719c34c6d29`)
- **A fresh clone is required** for any clone predating 2026-08-20. The history
  was rewritten and an older clone cannot fast-forward.
- **How to run the app:**

  ```bash
  ./frontend/.venv/bin/python -m streamlit run frontend/Overview.py
  ```

  Serving at http://localhost:8501. The two-virtualenv setup is in `README.md`
  (Streamlit lives in `frontend/.venv`, the modelling stack in `backend/.venv`;
  neither can see the other).
- **An empty forecast tab is expected.** The published forecasts and run
  artifacts are withheld from every clone by design, so a clean clone shows empty
  states rather than populated ones. This is not a fault to be fixed.
