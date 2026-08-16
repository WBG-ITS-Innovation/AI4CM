"""Run provenance: what produced a set of forecasts.

Item 1e. Before this, provenance existed for exactly one of four families
(``b_ml``), no run recorded the git SHA or the input file's hash, and the daily
script selected its input with ``ls -t | head -1`` -- by modification time. Two runs
on the same date could therefore use different inputs, and afterwards nothing said
which (review §7.2, §7.4). Asked "what produced the number in the 2026-07-30
report", the honest answer was that we could not fully say.

One implementation, used by all four runners, so they cannot drift apart -- the same
reason the duplicate integrity module was retired in item 1c.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

#: Env var an operator sets to publish despite a stale feed (decision Q5:
#: stale blocks publication, overridable only explicitly, and the override is
#: recorded here so it can never be a silent decision).
STALE_OVERRIDE_ENV = "AI4CM_ALLOW_STALE_DATA"


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256, so a large CSV is not read into memory twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_input(data_path: str | Path, date_col: str = "date") -> Dict[str, Any]:
    """Identify the input file by content, not by position in a directory listing.

    Returns name, absolute path, sha256, row count, and the latest date it contains.
    The hash is what makes a run reproducible: a reviewer can confirm they are looking
    at the same bytes rather than the same filename.
    """
    p = Path(data_path).resolve()
    info: Dict[str, Any] = {
        "name": p.name,
        "path": str(p),
        "exists": p.exists(),
        "sha256": None,
        "n_rows": None,
        "latest_data_date": None,
        "size_bytes": None,
    }
    if not p.exists():
        return info

    info["sha256"] = sha256_of(p)
    info["size_bytes"] = p.stat().st_size
    try:
        import pandas as pd
        dates = pd.read_csv(p, usecols=[date_col])[date_col]
        parsed = pd.to_datetime(dates, errors="coerce")
        info["n_rows"] = int(len(dates))
        latest = parsed.max()
        info["latest_data_date"] = None if pd.isna(latest) else str(latest.date())
    except Exception as exc:            # noqa: BLE001 - provenance must never abort a run
        info["read_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _git(*args: str, allow_empty: bool = False,
         cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git command; ``None`` means "could not be answered".

    ``allow_empty`` exists because for one caller -- ``git status --porcelain`` -- empty
    output IS the answer, and the default conflates it with failure. Every other caller here
    asks a question whose empty answer is meaningless (a SHA, a branch name), so the default
    stays as it was.
    """
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True, timeout=10,
                             cwd=str(cwd or Path(__file__).resolve().parent))
    except Exception:                   # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text if (text or allow_empty) else None


def describe_code(repo: Optional[Path] = None) -> Dict[str, Any]:
    """Git SHA, dirty flag and branch, so a run maps to a state of the repo.

    ``git_dirty`` is tri-state: ``True`` dirty / ``False`` clean / ``None`` not answerable
    (no git, not a repo). Until 2026-08-15 the third state swallowed the second: ``_git``
    returned ``None`` for an empty stdout, so a CLEAN tree read as "unknown" and ``False``
    was unreachable. The bug survived because the only test asserted the keys existed rather
    than what they said, and because the dirty path -- the one that actually gets exercised
    during development -- was correct throughout.

    ``repo`` overrides the working directory, so the three states can be driven against real
    throwaway repositories in a test rather than mocked.
    """
    dirty = _git("status", "--porcelain", allow_empty=True, cwd=repo)
    return {
        "git_sha": _git("rev-parse", "HEAD", cwd=repo),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo),
        # A dirty tree means the SHA alone does not identify what ran.
        "git_dirty": None if dirty is None else bool(dirty),
        "git_dirty_files": len(dirty.splitlines()) if dirty else 0,
    }


def describe_environment(packages=("numpy", "pandas", "scikit-learn", "statsmodels",
                                   "xgboost", "lightgbm", "torch", "joblib")) -> Dict[str, Any]:
    """Package versions.

    These change the model set silently: ``b_ml_pipeline`` gates XGBoost and LightGBM
    behind import success, so the same config produces a different leaderboard on a
    machine where one is missing.

    ``joblib`` is here because it is the reader of the persisted estimators
    (``estimator_store``), not because it affects a fit: a pickle written by one joblib and
    read by another is exactly the mismatch that must be detectable at load time.
    """
    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception:               # noqa: BLE001
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def build_provenance(
    family: str,
    data_path: str | Path,
    config: Dict[str, Any],
    date_col: str = "date",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full provenance record for one run."""
    return {
        "family": family,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": describe_input(data_path, date_col=date_col),
        "code": describe_code(),
        "environment": describe_environment(),
        "config": config,
        "seed": seed,
        "stale_override": os.environ.get(STALE_OVERRIDE_ENV, "") .strip().lower()
                          in {"1", "true", "yes"},
        "env_vars": {k: os.environ.get(k) for k in (
            "TG_FAMILY", "TG_MODEL_FILTER", "TG_TARGET", "TG_CADENCE", "TG_HORIZON",
            "TG_DATA_PATH", "TG_DATE_COL", "TG_PARAM_OVERRIDES", "TG_OUT_ROOT",
            "AI4CM_EXPECTED_DATA_SHA256",
        )},
        **(extra or {}),
    }


def write_provenance(out_root: str | Path, record: Dict[str, Any]) -> Path:
    """Write ``artifacts/provenance.json`` under a run directory."""
    path = Path(out_root) / "artifacts" / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def record_run(
    out_root: str | Path,
    family: str,
    data_path: str | Path,
    config: Dict[str, Any],
    date_col: str = "date",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build, write and summarise provenance in one call. Never raises.

    Provenance is a record of what happened; failing to write it must not abort a run
    that has otherwise succeeded, so errors are reported and swallowed.
    """
    try:
        record = build_provenance(family, data_path, config,
                                  date_col=date_col, seed=seed, extra=extra)
        path = write_provenance(out_root, record)
        d = record["data_file"]
        sha = (d.get("sha256") or "")[:16]
        print(f"[provenance] {family}: data={d.get('name')} sha256={sha}... "
              f"rows={d.get('n_rows')} latest={d.get('latest_data_date')} "
              f"git={(record['code'].get('git_sha') or '?')[:8]}"
              f"{'+dirty' if record['code'].get('git_dirty') else ''} -> {path}")
        return record
    except Exception as exc:            # noqa: BLE001
        print(f"[provenance][WARN] could not record provenance: "
              f"{type(exc).__name__}: {exc}")
        return {}


def verify_expected_sha(data_path: str | Path,
                        expected: Optional[str] = None) -> None:
    """Fail closed when the input is not the file the caller expected.

    ``expected`` defaults to ``AI4CM_EXPECTED_DATA_SHA256``. Selecting an input by
    name is necessary but not sufficient: a file can be regenerated in place. Pinning
    the hash is what makes "the same run" mean the same bytes.
    """
    expected = expected or os.environ.get("AI4CM_EXPECTED_DATA_SHA256", "").strip()
    if not expected:
        return
    actual = sha256_of(data_path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Input file SHA-256 mismatch for {Path(data_path).name}.\n"
            f"  expected {expected}\n  actual   {actual}\n"
            "Refusing to run: the input is not the file this run was pinned to. "
            "Either update AI4CM_EXPECTED_DATA_SHA256 deliberately, or regenerate "
            "the dataset."
        )
    print(f"[provenance] input SHA-256 matches the pinned value ({actual[:16]}...)")
