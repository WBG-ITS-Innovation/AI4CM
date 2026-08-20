"""Installing new actuals: one validation, one install, one place.

Why this module exists
----------------------
Loading a new data file was a manual act. Somebody copied a CSV over
``backend/data/processed/master_daily_clean_treasury.csv`` and ran the scorer. Nothing
checked that the file had the columns the recipes need, that its dates actually extended
past what was already held, or that it differed from the file already installed. Nothing
kept the previous file either, so a bad load was unrecoverable.

The Scorecard page needs to do exactly this from a browser, and the obvious way to build
that would have been a second implementation living in the UI. Two ingest paths means two
sets of checks, and the one that is easier to reach is the one that gets used. So this is
the single path: the page calls it, the command line calls it, and there is nothing else.

What "valid" means here
-----------------------
Four things must hold before anything is written, and each corresponds to a way a load
has gone wrong or plausibly could:

* **Schema.** Every column the canonical file holds must still be present. A recipe that
  names a feature column cannot be refitted against a file that dropped it, and the
  failure would surface much later as a KeyError inside a pipeline.
* **It must extend the record.** New actuals exist to add dates. A file whose last date is
  not later than the one already installed adds no truth, so scoring it would produce the
  same scorecard and quietly suggest that nothing arrived.
* **It must differ.** Identical bytes means the same file was uploaded twice. Installing it
  would rotate a backup and change nothing.
* **Revisions are reported, never silent.** Where an overlapping date's value changed, the
  count is reported and the reader confirms it. Actuals genuinely do get revised, so this
  is a fact to state rather than a reason to refuse.

What this module deliberately does NOT do
-----------------------------------------
It does not choose anything, score anything, or touch a window. Installing data and
scoring against it are separate acts and stay separate: the caller scores afterwards, via
``published_forecasts.score_published``, which already refuses to evaluate a date whose
truth has not arrived.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

BACKEND = Path(__file__).resolve().parent

#: The file every pipeline, the scorer and the forward runner read.
CANONICAL = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"

#: Where the file being replaced is kept. One directory, timestamped names, never pruned
#: automatically: an ingest that cannot be undone is an ingest nobody should run.
BACKUP_DIR = BACKEND / "data" / "processed" / "backups"

DATE_COLUMN = "date"


class IngestRefused(RuntimeError):
    """Raised when an install is attempted on a candidate that did not pass validation."""


@dataclass
class IngestCheck:
    """The verdict on a candidate file, in a form a page can render without deciding anything.

    ``blockers`` prevent an install. ``warnings`` are facts the person confirming should
    know and then decide about. Both are complete sentences, because they are shown verbatim.
    """

    candidate: str
    ok: bool = False
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Everything a confirmation step needs to state what is about to change.
    summary: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {"candidate": self.candidate, "ok": self.ok, "blockers": list(self.blockers),
                "warnings": list(self.warnings), "summary": dict(self.summary)}


@dataclass
class IngestResult:
    """What an install actually did, in the words a result summary uses."""

    installed: bool
    canonical: str
    backup: Optional[str]
    rows_before: int
    rows_after: int
    rows_added: int
    last_date_before: Optional[str]
    last_date_after: Optional[str]
    sha_before: Optional[str]
    sha_after: Optional[str]
    revisions: int = 0

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


def _sha(path: Path) -> Optional[str]:
    from provenance import sha256_of

    try:
        return sha256_of(path)
    except OSError:
        return None


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    return df


def _last_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if DATE_COLUMN not in df.columns:
        return None
    dates = df[DATE_COLUMN].dropna()
    return None if dates.empty else pd.Timestamp(dates.max())


def _count_revisions(current: pd.DataFrame, candidate: pd.DataFrame) -> int:
    """How many overlapping (date, column) cells changed value.

    Counted rather than listed: a revision of one cell and a revision of four thousand are
    different events, and the number is what tells them apart at a glance.
    """
    shared_cols = [c for c in current.columns
                   if c in candidate.columns and c != DATE_COLUMN]
    if not shared_cols or DATE_COLUMN not in current.columns:
        return 0
    # Deduplicated first, keeping the last row for a repeated date. A repeated date is
    # separately a blocker, but this function still runs on the way to producing the
    # summary the page displays, and a duplicate would otherwise make the two frames
    # different lengths and raise here instead of surfacing the blocker.
    a = (current.dropna(subset=[DATE_COLUMN])
         .drop_duplicates(subset=[DATE_COLUMN], keep="last")
         .set_index(DATE_COLUMN)[shared_cols])
    b = (candidate.dropna(subset=[DATE_COLUMN])
         .drop_duplicates(subset=[DATE_COLUMN], keep="last")
         .set_index(DATE_COLUMN)[shared_cols])
    shared_idx = a.index.intersection(b.index)
    if len(shared_idx) == 0:
        return 0
    a, b = a.loc[shared_idx], b.loc[shared_idx]
    both_null = a.isna() & b.isna()
    differs = (a != b) & ~both_null
    numeric = [c for c in shared_cols
               if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    for col in numeric:
        # A float that survived a CSV round trip can differ in the last bit without any
        # value having been revised. Counting that as a revision would make every reload
        # look like a restatement of the whole history.
        close = (a[col] - b[col]).abs() <= 1e-6 * a[col].abs().clip(lower=1.0)
        differs[col] = differs[col] & ~close.fillna(False)
    return int(differs.to_numpy().sum())


def validate(candidate: Path, canonical: Optional[Path] = None) -> IngestCheck:
    """Check a candidate actuals file without writing anything.

    Returns an :class:`IngestCheck` in every case, including for a file that cannot be
    read at all. Raising here would mean the page had to catch exceptions in order to
    display a message, and a message is what it needs.
    """
    candidate = Path(candidate)
    canon = Path(canonical or CANONICAL)
    check = IngestCheck(candidate=str(candidate))

    if not candidate.exists():
        check.blockers.append(f"The file {candidate.name} was not found.")
        return check

    try:
        new = _read(candidate)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        check.blockers.append(
            f"{candidate.name} could not be read as a CSV file. The reader reported: {exc}."
        )
        return check

    if DATE_COLUMN not in new.columns:
        check.blockers.append(
            f"{candidate.name} has no column called '{DATE_COLUMN}', so its rows cannot be "
            f"placed in time. Every data file this system reads names its date column "
            f"'{DATE_COLUMN}'."
        )
        return check

    unparseable = int(new[DATE_COLUMN].isna().sum())
    if unparseable == len(new):
        check.blockers.append(
            f"None of the {len(new)} dates in {candidate.name} could be read as dates. "
            f"Check the date format in the file."
        )
        return check
    if unparseable:
        check.warnings.append(
            f"{unparseable} of {len(new)} rows have a date that could not be read. Those "
            f"rows will be present in the file but will not be scored against."
        )

    duplicates = int(new[DATE_COLUMN].dropna().duplicated().sum())
    if duplicates:
        check.blockers.append(
            f"{candidate.name} has {duplicates} repeated date(s). Each date must appear "
            f"once, because a date with two rows has two different actual values."
        )

    if not canon.exists():
        check.blockers.append(
            f"There is no canonical data file at {canon} to compare against, so this "
            f"upload cannot be checked. Restore the canonical file first."
        )
        return check

    current = _read(canon)
    missing = [c for c in current.columns if c not in new.columns]
    if missing:
        shown = ", ".join(missing[:6]) + (" and others" if len(missing) > 6 else "")
        check.blockers.append(
            f"{candidate.name} is missing {len(missing)} column(s) the current data has: "
            f"{shown}. The published recipes read those columns, so a file without them "
            f"cannot replace the current one."
        )

    added_cols = [c for c in new.columns if c not in current.columns]
    if added_cols:
        check.warnings.append(
            f"{candidate.name} adds {len(added_cols)} column(s) the current data does not "
            f"have: {', '.join(added_cols[:6])}. They will be stored but no published "
            f"recipe uses them."
        )

    last_now, last_new = _last_date(current), _last_date(new)
    if last_now is not None and last_new is not None and last_new <= last_now:
        check.blockers.append(
            f"{candidate.name} ends on {last_new.date()}, and the data already held ends "
            f"on {last_now.date()}. New actuals have to extend the record, otherwise there "
            f"is no new truth to score any forecast against."
        )

    sha_now, sha_new = _sha(canon), _sha(candidate)
    if sha_now and sha_new and sha_now == sha_new:
        check.blockers.append(
            "This is byte for byte the same file that is already installed, so installing "
            "it would change nothing."
        )

    revisions = _count_revisions(current, new)
    if revisions:
        check.warnings.append(
            f"{revisions} value(s) on dates already held have changed in this file. "
            f"Revised actuals are normal, and any forecast already scored against the old "
            f"value will be scored again against the new one."
        )

    new_rows = 0
    if last_now is not None:
        new_rows = int((new[DATE_COLUMN] > last_now).sum())

    check.summary = {
        "rows_now": int(len(current)),
        "rows_new": int(len(new)),
        "rows_added": new_rows,
        "last_date_now": str(last_now.date()) if last_now is not None else None,
        "last_date_new": str(last_new.date()) if last_new is not None else None,
        "columns_now": int(len(current.columns)),
        "columns_new": int(len(new.columns)),
        "columns_missing": missing,
        "columns_added": added_cols,
        "revisions": revisions,
        "sha_now": sha_now,
        "sha_new": sha_new,
        "canonical": str(canon),
    }
    check.ok = not check.blockers
    return check


def backup_name(now: Optional[datetime] = None, canonical: Optional[Path] = None) -> str:
    """The name the file being replaced is kept under. UTC, so names sort chronologically."""
    canon = Path(canonical or CANONICAL)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{canon.stem}.{stamp}{canon.suffix}"


def install(candidate: Path, canonical: Optional[Path] = None,
            backup_dir: Optional[Path] = None,
            check: Optional[IngestCheck] = None) -> IngestResult:
    """Back up the current canonical file and put ``candidate`` in its place.

    Refuses unless the candidate validates. Passing an already-computed ``check`` avoids
    reading both files twice when a page has just shown its result to a person; it is
    re-validated regardless, because the file on disk can change between the confirmation
    and the click.
    """
    candidate = Path(candidate)
    canon = Path(canonical or CANONICAL)
    backups = Path(backup_dir or BACKUP_DIR)

    verdict = validate(candidate, canon)
    if not verdict.ok:
        raise IngestRefused(
            "This file was not installed. " + " ".join(verdict.blockers)
        )

    summary = verdict.summary
    backups.mkdir(parents=True, exist_ok=True)
    backup_path = backups / backup_name(canonical=canon)
    shutil.copy2(canon, backup_path)

    # copy2 rather than a parse-and-rewrite: the installed file is then byte for byte the
    # file that was checked, so the SHA recorded in every downstream provenance record is
    # the SHA of something that exists rather than of a reformatted derivative.
    shutil.copy2(candidate, canon)

    after = _read(canon)
    last_after = _last_date(after)
    return IngestResult(
        installed=True,
        canonical=str(canon),
        backup=str(backup_path),
        rows_before=int(summary["rows_now"]),
        rows_after=int(len(after)),
        rows_added=int(summary["rows_added"]),
        last_date_before=summary["last_date_now"],
        last_date_after=str(last_after.date()) if last_after is not None else None,
        sha_before=summary["sha_now"],
        sha_after=_sha(canon),
        revisions=int(summary["revisions"]),
    )


def list_backups(backup_dir: Optional[Path] = None) -> List[Path]:
    """Every retained previous version, newest first."""
    backups = Path(backup_dir or BACKUP_DIR)
    if not backups.exists():
        return []
    return sorted(backups.glob("*.csv"), reverse=True)


def _cli() -> int:
    import argparse
    import json
    import sys

    sys.path.insert(0, str(BACKEND))

    ap = argparse.ArgumentParser(
        description="Validate, and optionally install, a new actuals file.")
    ap.add_argument("--file", required=True, help="the candidate CSV")
    ap.add_argument("--install", action="store_true",
                    help="install it after validation; without this, only checks")
    ap.add_argument("--score", action="store_true",
                    help="score published forecasts against the installed data afterwards")
    args = ap.parse_args()

    check = validate(Path(args.file))
    out: Dict = {"check": check.as_dict()}
    if args.install:
        if not check.ok:
            print(json.dumps(out, default=str))
            return 1
        out["install"] = install(Path(args.file)).as_dict()
        if args.score:
            from published_forecasts import score_published

            scored = score_published(CANONICAL)
            out["score"] = {"scored": scored["scored"], "pending": scored["pending"],
                            "scorecard": scored["scorecard"]}
    print(json.dumps(out, default=str))
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
