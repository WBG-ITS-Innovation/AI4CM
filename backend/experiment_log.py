"""Append-only experiments log (Phase-2 ground rule 2).

Every number in a report must be reproducible from a logged run. The log is the index;
one JSON per run carries the detail.

Columns are fixed by the Phase-2 brief and deliberately not extensible in place -- a
column set that drifts cannot be compared across runs. Adding a column means bumping
``SCHEMA_VERSION`` so old rows stay readable and a reader can tell which shape they hold.

Two design points worth stating:

* **Append-only.** Rows are never rewritten. A run that turned out to be wrong is
  superseded by a later row, not edited away, so the record shows what was believed at
  the time.
* **The feature-set hash is over NAMES, not values.** Two runs with the same features on
  different data must hash the same, so that ``data_sha`` is what distinguishes them.
  Hashing values would conflate "different features" with "different data".
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

SCHEMA_VERSION = 1

#: Exactly the columns named in the Phase-2 brief, in order. `run_id`, `schema_version`
#: and `calendar_version` are bookkeeping: the first links a row to its JSON, the second
#: makes the shape self-describing, the third exists because Q3 requires fiscal-calendar
#: results to be re-runnable after Treasury confirms UNVERIFIED dates.
COLUMNS: Sequence[str] = (
    "run_id",
    "schema_version",
    "timestamp",
    "git_sha",
    "data_sha",
    "target",
    "feature_set_hash",
    "params",
    "seed",
    "fold_scheme",
    "dev_mae",
    "mase",
    "skill_vs_ruler",
    "sentinel_ratio",
    "coverage_low",
    "coverage_mid",
    "coverage_high",
    "calendar_version",
    "note",
)

LOG_DIR = Path(__file__).resolve().parent.parent / "experiments"
LOG_CSV = LOG_DIR / "log.csv"
RUNS_DIR = LOG_DIR / "runs"


def feature_set_hash(feature_names: Iterable[str]) -> str:
    """Stable 12-hex digest of a feature set.

    Sorted, so column order cannot change the hash; over names only, so the same recipe
    on different data hashes identically and ``data_sha`` is what separates those runs.
    """
    names = sorted(str(n) for n in feature_names)
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:12]


def params_hash(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def make_run_id(target: str, model: str, timestamp: Optional[str] = None) -> str:
    """Unique id for one run: UTC timestamp to the microsecond, then target_model.

    Second resolution is not enough. A sweep logs tens of rows inside one second, and
    ``target_model`` repeats across windows, so a whole-second stamp produced colliding
    ids -- caught by ``verify_log_integrity`` reporting ``duplicate run_id values`` after
    a 36-row batch. Microseconds separate them, and the suffix loop below guarantees
    uniqueness even if two ids still coincide: a duplicate id means two runs share one
    detail JSON, so one of them is unrecoverable.
    """
    ts = (timestamp or datetime.now(timezone.utc).isoformat()).replace(":", "").replace("-", "")
    slug = "".join(c if c.isalnum() else "_" for c in f"{target}_{model}").strip("_")
    base = f"{ts[:22]}_{slug}"[:96]
    run_id, n = base, 1
    while (RUNS_DIR / f"{run_id}.json").exists():
        run_id = f"{base}_{n}"[:96]
        n += 1
    return run_id


def log_run(
    *,
    target: str,
    model: str,
    git_sha: Optional[str],
    data_sha: Optional[str],
    feature_names: Iterable[str],
    params: Mapping[str, Any],
    seed: Optional[int],
    fold_scheme: str,
    dev_mae: Optional[float],
    mase: Optional[float],
    skill_vs_ruler: Optional[float],
    sentinel_ratio: Optional[float] = None,
    coverage: Optional[Mapping[str, float]] = None,
    ruler: Optional[float] = None,
    per_fold: Optional[Sequence[Mapping[str, Any]]] = None,
    calendar_version: Optional[str] = None,
    note: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one row to ``log.csv`` and write the matching ``runs/<run_id>.json``.

    ``coverage`` is the per-magnitude-tercile interval coverage, keyed
    ``low``/``mid``/``high``. It is optional because point models publish no intervals --
    but when a model does publish them, the terciles are the number that matters: a
    marginally calibrated band can still cover half the largest days (review §3.2).
    """
    ts = datetime.now(timezone.utc).isoformat()
    run_id = make_run_id(target, model, ts)
    cov = dict(coverage or {})

    row = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts,
        "git_sha": (git_sha or "")[:12],
        "data_sha": (data_sha or "")[:16],
        "target": target,
        "feature_set_hash": feature_set_hash(feature_names),
        "params": params_hash(params),
        "seed": "" if seed is None else seed,
        "fold_scheme": fold_scheme,
        "dev_mae": "" if dev_mae is None else f"{dev_mae:.4f}",
        "mase": "" if mase is None else f"{mase:.6f}",
        "skill_vs_ruler": "" if skill_vs_ruler is None else f"{skill_vs_ruler:.4f}",
        "sentinel_ratio": "" if sentinel_ratio is None else f"{sentinel_ratio:.4f}",
        "coverage_low": "" if cov.get("low") is None else f"{cov['low']:.4f}",
        "coverage_mid": "" if cov.get("mid") is None else f"{cov['mid']:.4f}",
        "coverage_high": "" if cov.get("high") is None else f"{cov['high']:.4f}",
        "calendar_version": calendar_version or "",
        "note": note,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    is_new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        if is_new:
            w.writeheader()
        w.writerow(row)

    detail = {
        **row,
        "model": model,
        "git_sha_full": git_sha,
        "data_sha_full": data_sha,
        "feature_names": sorted(str(n) for n in feature_names),
        "params_full": dict(params),
        "ruler": ruler,
        "coverage_terciles": cov,
        "per_fold": list(per_fold or []),
        **dict(extra or {}),
    }
    (RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(detail, indent=2, default=str), encoding="utf-8")
    return detail


def read_log() -> "list[dict]":
    if not LOG_CSV.exists():
        return []
    with LOG_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verify_log_integrity() -> Dict[str, Any]:
    """Check the log is intact: header matches, every row has a JSON, ids unique.

    A log that has silently lost its detail files, or grown a column, cannot support
    "every number is reproducible from a logged run".
    """
    problems: "list[str]" = []
    rows = read_log()
    if LOG_CSV.exists():
        with LOG_CSV.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if tuple(header) != tuple(COLUMNS):
            problems.append(f"header drift: {header} != {list(COLUMNS)}")

    ids = [r.get("run_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate run_id values")
    for r in rows:
        rid = r.get("run_id", "")
        if rid and not (RUNS_DIR / f"{rid}.json").exists():
            problems.append(f"missing detail JSON for {rid}")

    return {"n_rows": len(rows), "ok": not problems, "problems": problems}
