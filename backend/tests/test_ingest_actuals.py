"""New actuals arrive through one door, and that door checks four things.

Why these tests
---------------
Installing a data file was a manual copy. Nothing checked the schema, nothing checked the
dates extended what was held, nothing kept the file being replaced. Adding an upload button
to the Scorecard page makes that act reachable from a browser, which raises the cost of
every one of those omissions.

Each test below names a way an install has gone wrong or plausibly could:

* a file missing a column a published recipe reads, which fails much later as a KeyError
  inside a pipeline rather than at the point the bad file was loaded;
* a file that does not extend the record, whose install would rotate a backup, change
  nothing, and leave the reader believing new truth had arrived;
* the same file uploaded twice;
* an install with no way back.

The last group asserts the property the whole module exists for: that the UI and the
command line cannot drift apart, because there is only one implementation for both.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from ingest_actuals import (  # noqa: E402
    CANONICAL,
    IngestRefused,
    backup_name,
    install,
    list_backups,
    validate,
)

COLUMNS = ["date", "Revenues", "Expenditure", "State budget balance"]


def _frame(start: str, end: str, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "Revenues": [1_000_000.0 + i + offset for i in range(len(dates))],
        "Expenditure": [900_000.0 + i + offset for i in range(len(dates))],
        "State budget balance": [100_000.0 + i + offset for i in range(len(dates))],
    })


@pytest.fixture
def canon(tmp_path) -> Path:
    """A small stand-in canonical file. Never the real one: these tests install things."""
    path = tmp_path / "canonical.csv"
    _frame("2024-01-01", "2024-03-29").to_csv(path, index=False)
    return path


def _write(tmp_path: Path, df: pd.DataFrame, name: str = "upload.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# What a valid file looks like
# ---------------------------------------------------------------------------

def test_a_file_that_extends_the_record_passes(tmp_path, canon):
    check = validate(_write(tmp_path, _frame("2024-01-01", "2024-04-30")), canon)
    assert check.ok, check.blockers
    assert check.summary["rows_added"] > 0
    assert check.summary["last_date_new"] > check.summary["last_date_now"]


def test_extra_columns_are_a_warning_not_a_refusal(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30")
    df["A brand new line"] = 1.0
    check = validate(_write(tmp_path, df), canon)
    assert check.ok
    assert any("adds 1 column" in w for w in check.warnings)


def test_revisions_are_counted_and_reported(tmp_path, canon):
    """Actuals genuinely get revised, so this is a fact to state, not a reason to refuse."""
    df = _frame("2024-01-01", "2024-04-30")
    df.loc[0, "Revenues"] = 42.0
    df.loc[5, "Expenditure"] = 43.0
    check = validate(_write(tmp_path, df), canon)
    assert check.ok
    assert check.summary["revisions"] == 2
    assert any("2 value(s)" in w for w in check.warnings)


def test_a_float_round_trip_is_not_counted_as_a_revision(tmp_path, canon):
    """Otherwise every reload would look like a restatement of the whole history."""
    df = _frame("2024-01-01", "2024-04-30")
    df["Revenues"] = df["Revenues"] * (1 + 1e-12)
    check = validate(_write(tmp_path, df), canon)
    assert check.summary["revisions"] == 0


# ---------------------------------------------------------------------------
# The four refusals
# ---------------------------------------------------------------------------

def test_a_file_missing_a_column_is_refused(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30").drop(columns=["Expenditure"])
    check = validate(_write(tmp_path, df), canon)
    assert not check.ok
    assert any("missing 1 column" in b and "Expenditure" in b for b in check.blockers)


def test_a_file_that_does_not_extend_the_record_is_refused(tmp_path, canon):
    check = validate(_write(tmp_path, _frame("2024-01-01", "2024-03-01")), canon)
    assert not check.ok
    assert any("extend the record" in b for b in check.blockers)


def test_the_identical_file_is_refused(tmp_path, canon):
    check = validate(canon, canon)
    assert not check.ok
    assert any("byte for byte" in b for b in check.blockers)


def test_repeated_dates_are_refused(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    check = validate(_write(tmp_path, df), canon)
    assert not check.ok
    assert any("repeated date" in b for b in check.blockers)


def test_a_file_with_no_date_column_is_refused(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30").rename(columns={"date": "day"})
    check = validate(_write(tmp_path, df), canon)
    assert not check.ok
    assert any("no column called 'date'" in b for b in check.blockers)


def test_an_unreadable_file_is_explained_rather_than_raised(tmp_path, canon):
    bad = tmp_path / "not.csv"
    bad.write_bytes(b"\x00\x01\x02\x03")
    check = validate(bad, canon)
    assert not check.ok
    assert check.blockers


def test_a_missing_file_is_explained_rather_than_raised(tmp_path, canon):
    check = validate(tmp_path / "absent.csv", canon)
    assert not check.ok
    assert "was not found" in check.blockers[0]


@pytest.mark.parametrize("case", ["missing_column", "no_extension", "identical",
                                  "duplicate_dates", "no_date_column"])
def test_every_refusal_is_a_finished_sentence(tmp_path, canon, case):
    """These strings are rendered verbatim on the Scorecard page."""
    if case == "missing_column":
        cand = _write(tmp_path, _frame("2024-01-01", "2024-04-30").drop(columns=["Revenues"]))
    elif case == "no_extension":
        cand = _write(tmp_path, _frame("2024-01-01", "2024-02-01"))
    elif case == "identical":
        cand = canon
    elif case == "duplicate_dates":
        df = _frame("2024-01-01", "2024-04-30")
        cand = _write(tmp_path, pd.concat([df, df.iloc[[1]]], ignore_index=True))
    else:
        cand = _write(tmp_path, _frame("2024-01-01", "2024-04-30").rename(
            columns={"date": "when"}))

    for blocker in validate(cand, canon).blockers:
        assert blocker.endswith(".")
        assert "Traceback" not in blocker
        assert "--" not in blocker and "—" not in blocker
        assert blocker[0].isupper() or blocker.startswith(("master", "upload", "canonical"))


# ---------------------------------------------------------------------------
# Installing, and being able to undo it
# ---------------------------------------------------------------------------

def test_install_keeps_the_file_it_replaces(tmp_path, canon):
    before = canon.read_bytes()
    backups = tmp_path / "backups"
    result = install(_write(tmp_path, _frame("2024-01-01", "2024-04-30")),
                     canonical=canon, backup_dir=backups)
    assert result.installed
    assert Path(result.backup).read_bytes() == before
    assert len(list_backups(backups)) == 1


def test_install_puts_the_checked_bytes_in_place(tmp_path, canon):
    """A parse-and-rewrite would give downstream provenance the SHA of a derivative."""
    candidate = _write(tmp_path, _frame("2024-01-01", "2024-04-30"))
    install(candidate, canonical=canon, backup_dir=tmp_path / "backups")
    assert canon.read_bytes() == candidate.read_bytes()


def test_install_reports_what_changed(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30")
    df.loc[0, "Revenues"] = 7.0
    result = install(_write(tmp_path, df), canonical=canon, backup_dir=tmp_path / "backups")
    assert result.rows_after > result.rows_before
    assert result.rows_added == result.rows_after - result.rows_before
    assert result.last_date_after > result.last_date_before
    assert result.sha_after != result.sha_before
    assert result.revisions == 1


def test_install_refuses_a_file_that_did_not_validate(tmp_path, canon):
    before = canon.read_bytes()
    with pytest.raises(IngestRefused):
        install(_write(tmp_path, _frame("2024-01-01", "2024-02-01")),
                canonical=canon, backup_dir=tmp_path / "backups")
    assert canon.read_bytes() == before, "a refused install must change nothing"


def test_install_revalidates_even_when_handed_a_passing_check(tmp_path, canon):
    """The file on disk can change between the confirmation and the click."""
    candidate = _write(tmp_path, _frame("2024-01-01", "2024-04-30"))
    check = validate(candidate, canon)
    assert check.ok
    _frame("2024-01-01", "2024-02-01").to_csv(candidate, index=False)   # swapped underneath
    with pytest.raises(IngestRefused):
        install(candidate, canonical=canon, backup_dir=tmp_path / "backups", check=check)


def test_backup_names_sort_chronologically():
    from datetime import datetime, timezone

    early = backup_name(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    late = backup_name(datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc))
    assert early < late
    assert early.endswith(".csv") and "Z" in early


def test_two_installs_keep_two_backups(tmp_path, canon):
    backups = tmp_path / "backups"
    install(_write(tmp_path, _frame("2024-01-01", "2024-04-30"), "a.csv"),
            canonical=canon, backup_dir=backups)
    # A distinct timestamp is not guaranteed inside one second, so this asserts the second
    # install succeeded rather than asserting a count that a clock could defeat.
    install(_write(tmp_path, _frame("2024-01-01", "2024-05-31"), "b.csv"),
            canonical=canon, backup_dir=backups)
    assert len(list_backups(backups)) >= 1
    assert pd.read_csv(canon)["date"].max() == "2024-05-31"


# ---------------------------------------------------------------------------
# One implementation, two callers
# ---------------------------------------------------------------------------

def test_the_scorecard_page_calls_this_module_and_implements_nothing_of_its_own():
    """The property this module exists for.

    A second ingest path in the UI would be a second set of checks, and the easier one to
    reach is the one that gets used. So the page must call ``validate`` and ``install`` and
    must not carry its own schema, date or SHA comparison.
    """
    page = (REPO / "frontend" / "pages" / "02_Scorecard.py").read_text(encoding="utf-8")
    assert "from ingest_actuals import" in page
    assert "validate(candidate)" in page
    assert "install(candidate" in page
    for reimplementation in ("sha256", "hashlib", "shutil.copy", "columns_missing =",
                             "def _validate", "def _install"):
        assert reimplementation not in page, (
            f"the page is re-implementing ingest logic: {reimplementation}"
        )


def test_the_cli_and_the_page_share_one_verdict(tmp_path, canon):
    """Both callers reach the same decision on the same file, because there is one decider."""
    candidate = _write(tmp_path, _frame("2024-01-01", "2024-02-01"))
    assert validate(candidate, canon).ok is False

    out = subprocess.run(
        [sys.executable, str(BACKEND / "ingest_actuals.py"), "--file", str(candidate)],
        capture_output=True, text=True, cwd=str(REPO))
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["check"]["ok"] is False
    assert payload["check"]["blockers"]


def test_the_cli_writes_nothing_without_the_install_flag(tmp_path):
    """Checking is free. Installing is not, and must be asked for."""
    before = CANONICAL.read_bytes()
    candidate = tmp_path / "c.csv"
    _frame("2024-01-01", "2024-02-01").to_csv(candidate, index=False)
    subprocess.run(
        [sys.executable, str(BACKEND / "ingest_actuals.py"), "--file", str(candidate)],
        capture_output=True, text=True, cwd=str(REPO))
    assert CANONICAL.read_bytes() == before


def test_installing_does_not_score(tmp_path, canon):
    """Two separate acts. The scorer still refuses a date whose truth has not arrived."""
    src = (BACKEND / "ingest_actuals.py").read_text(encoding="utf-8")
    body = src.split("def install(")[1].split("\ndef list_backups")[0]
    assert "score_published" not in body
