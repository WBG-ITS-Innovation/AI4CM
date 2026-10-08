"""The weekly upload accepts the master export as an Excel workbook.

Why the workbook is converted at the landing step
-------------------------------------------------
Officers keep the master export in Excel. ``validate`` and ``install`` read a CSV, and
``install`` byte-copies what it checked over the canonical file, so that the SHA every
provenance record carries is the SHA of a file that exists. Teaching both of them to read a
workbook would mean the canonical file could become a workbook, or that ``install`` writes
something other than what it checked.

So the workbook is turned into a CSV once, on arrival, and that CSV goes through the same
``validate`` and ``install`` as any other upload. A CSV upload does not pass through the
conversion at all: it is checked and copied exactly as before.

What these tests pin down
-------------------------
* A workbook passes the same checks a CSV does, and installs to the same canonical file,
  row for row and type for type.
* Excel stores every number the same way, and pandas hands a column of whole numbers back
  as integers. The CSV export writes those amounts with a decimal point. The test frame
  carries a column of whole-number amounts for exactly that reason, and an integer flag
  beside it that must stay an integer.
* Only the first sheet is read, and a workbook whose first sheet is not the data is refused
  in words an officer can act on: which sheets the workbook has and what the first one holds.
* Neither path can put a workbook where every pipeline expects a CSV, and landing never
  overwrites a file the officer keeps beside the workbook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import ingest_actuals  # noqa: E402
from ingest_actuals import (  # noqa: E402
    DATE_COLUMN,
    IngestRefused,
    _read,
    install,
    land,
    validate,
)

COLUMNS = ["date", "Revenues", "Expenditure", "State budget balance", "is_holiday"]


def _frame(start: str, end: str) -> pd.DataFrame:
    """Synthetic actuals with the three number shapes the real file has."""
    dates = pd.bdate_range(start, end)
    n = len(dates)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        # Amounts with a fractional part.
        "Revenues": [1_000_000.25 + i for i in range(n)],
        # Amounts held as floating point whose values are all whole numbers. This is the
        # column Excel loses the type of.
        "Expenditure": [900_000.0 + i for i in range(n)],
        "State budget balance": [-50_000.5 - i for i in range(n)],
        # A genuine integer flag, which must stay an integer.
        "is_holiday": [i % 2 for i in range(n)],
    }, columns=COLUMNS)


def _as_read(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` as the system reads it back: the date column parsed, nothing else touched."""
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN])
    return out


def _workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    """Write ``sheets`` in order, with the date column stored as Excel dates, not text."""
    with pd.ExcelWriter(path, engine="openpyxl") as book:
        for name, df in sheets.items():
            out = df.copy()
            if DATE_COLUMN in out.columns:
                out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN])
            out.to_excel(book, sheet_name=name, index=False)
    return path


@pytest.fixture
def canon(tmp_path) -> Path:
    """A small stand-in canonical file. Never the real one: these tests install things."""
    path = tmp_path / "canonical.csv"
    _frame("2024-01-01", "2024-03-29").to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# A workbook goes through the same door as a CSV
# ---------------------------------------------------------------------------

def test_a_workbook_passes_the_same_checks_a_csv_does(tmp_path, canon):
    book = _workbook(tmp_path / "master.xlsx", {"Data": _frame("2024-01-01", "2024-04-30")})

    candidate, refused = land(book, canon)

    assert refused is None
    assert candidate.suffix == ".csv"
    check = validate(candidate, canon)
    assert check.ok, check.blockers
    assert check.summary["rows_added"] > 0
    assert check.summary["revisions"] == 0, "a converted workbook must not look like a restatement"


def test_installing_a_workbook_gives_the_same_canonical_file_as_the_csv(tmp_path):
    """Row for row and type for type, the two formats must leave the same data installed."""
    held, new = _frame("2024-01-01", "2024-03-29"), _frame("2024-01-01", "2024-04-30")
    installed = {}
    for fmt in ("csv", "xlsx"):
        folder = tmp_path / fmt
        folder.mkdir()
        canonical = folder / "canonical.csv"
        held.to_csv(canonical, index=False)
        if fmt == "csv":
            upload = folder / "master.csv"
            new.to_csv(upload, index=False)
        else:
            upload = _workbook(folder / "master.xlsx", {"Data": new})
        candidate, refused = land(upload, canonical)
        assert refused is None, refused and refused.blockers
        install(candidate, canonical=canonical, backup_dir=folder / "backups")
        installed[fmt] = canonical

    assert not installed["xlsx"].read_bytes().startswith(b"PK"), (
        "the canonical file is a workbook; every pipeline reads it as a CSV")
    via_csv, via_xlsx = _read(installed["csv"]), _read(installed["xlsx"])
    assert dict(via_xlsx.dtypes.astype(str)) == dict(via_csv.dtypes.astype(str))
    pd.testing.assert_frame_equal(via_xlsx, via_csv, check_exact=True)


def test_a_csv_upload_is_still_installed_byte_for_byte(tmp_path, canon):
    """The regression this change must not cause. A CSV is checked and copied as uploaded."""
    upload = tmp_path / "master.csv"
    _frame("2024-01-01", "2024-04-30").to_csv(upload, index=False)
    before = sorted(p.name for p in tmp_path.iterdir())

    candidate, refused = land(upload, canon)

    assert refused is None
    assert candidate == upload, "a CSV must be checked as uploaded, not as a copy of it"
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "landing a CSV wrote a file"
    install(candidate, canonical=canon, backup_dir=tmp_path / "backups")
    assert canon.read_bytes() == upload.read_bytes()


def test_excel_typed_dates_and_numbers_survive_to_the_same_values(tmp_path, canon):
    df = _frame("2024-01-01", "2024-04-30")
    book = _workbook(tmp_path / "master.xlsx", {"Data": df})

    # The fixture must be what it claims: dates stored as Excel dates and amounts as Excel
    # numbers, not text that merely looks like them.
    from openpyxl import load_workbook

    sheet = load_workbook(book).worksheets[0]
    first_row = dict(zip([c.value for c in sheet[1]], sheet[2]))
    assert first_row["date"].is_date
    assert all(first_row[c].data_type == "n" for c in COLUMNS[1:])

    candidate, refused = land(book, canon)

    assert refused is None
    as_text = pd.read_csv(candidate, dtype=str)
    assert as_text[DATE_COLUMN].tolist() == df[DATE_COLUMN].tolist(), (
        "dates must land as the YYYY-MM-DD text the canonical file holds")
    pd.testing.assert_frame_equal(_read(candidate), _as_read(df), check_exact=True)


def test_only_the_first_sheet_is_read(tmp_path, canon):
    current, last_week = _frame("2024-01-01", "2024-04-30"), _frame("2024-01-01", "2024-04-19")
    book = _workbook(tmp_path / "master.xlsx", {"Data": current, "Last week": last_week})

    candidate, refused = land(book, canon)

    assert refused is None
    pd.testing.assert_frame_equal(_read(candidate), _as_read(current), check_exact=True)


# ---------------------------------------------------------------------------
# Refusals an officer can act on
# ---------------------------------------------------------------------------

def test_a_workbook_whose_first_sheet_is_not_the_data_is_refused_naming_what_it_found(
        tmp_path, canon):
    notes = pd.DataFrame({"Prepared by": ["Synthetic"], "Comment": ["Weekly export"]})
    book = _workbook(tmp_path / "master.xlsx",
                     {"Notes": notes, "Data": _frame("2024-01-01", "2024-04-30")})

    candidate, refused = land(book, canon)

    assert refused is not None and not refused.ok
    [message] = refused.blockers
    assert "first sheet" in message
    assert "'Notes', 'Data'" in message, "every sheet, in order, so the officer can find the data"
    assert "Prepared by" in message and "Comment" in message, "the columns actually found"
    assert "date" in message, "what the sheet lacks"
    assert message.endswith(".")
    assert "--" not in message and "—" not in message
    assert "Traceback" not in message and "KeyError" not in message
    assert not (tmp_path / "master.xlsx.csv").exists(), "a refused workbook must land nothing"


def test_a_file_that_is_not_really_a_workbook_is_refused_in_plain_words(tmp_path, canon):
    fake = tmp_path / "master.xlsx"
    fake.write_bytes(b"date,Revenues\n2024-01-01,1\n")

    _, refused = land(fake, canon)

    assert refused is not None
    assert "could not be opened as an Excel workbook" in refused.blockers[0]
    assert refused.blockers[0].endswith(".")


def test_a_missing_excel_reader_is_a_refusal_not_a_crash(tmp_path, canon, monkeypatch):
    """An environment built before openpyxl was required must refuse in words, not crash."""
    book = _workbook(tmp_path / "master.xlsx", {"Data": _frame("2024-01-01", "2024-04-30")})

    def no_reader(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'.")

    monkeypatch.setattr(pd, "ExcelFile", no_reader)
    _, refused = land(book, canon)

    assert refused is not None
    assert "cannot read Excel files" in refused.blockers[0]
    assert "CSV" in refused.blockers[0]


# ---------------------------------------------------------------------------
# What landing must never do
# ---------------------------------------------------------------------------

def test_landing_never_overwrites_a_csv_kept_beside_the_workbook(tmp_path, canon):
    kept = tmp_path / "master.csv"
    kept.write_text("the officer's own file\n", encoding="utf-8")
    book = _workbook(tmp_path / "master.xlsx", {"Data": _frame("2024-01-01", "2024-04-30")})

    candidate, refused = land(book, canon)

    assert refused is None
    assert candidate != kept
    assert kept.read_text(encoding="utf-8") == "the officer's own file\n"


def test_a_workbook_handed_straight_to_install_is_refused_never_copied(tmp_path, canon):
    """``install`` byte-copies. A caller that skipped landing must not install a workbook."""
    before = canon.read_bytes()
    book = _workbook(tmp_path / "master.xlsx", {"Data": _frame("2024-01-01", "2024-04-30")})

    with pytest.raises(IngestRefused):
        install(book, canonical=canon, backup_dir=tmp_path / "backups")
    assert canon.read_bytes() == before


# ---------------------------------------------------------------------------
# The command line lands a workbook before it checks one
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated(monkeypatch, tmp_path, canon) -> Path:
    """Point the CLI's defaults at the stand-in. It takes no path for them, by design."""
    monkeypatch.setattr(ingest_actuals, "CANONICAL", canon)
    monkeypatch.setattr(ingest_actuals, "BACKUP_DIR", tmp_path / "backups")
    return canon


def _run_cli(monkeypatch, capsys, *argv: str) -> Tuple[int, Dict]:
    monkeypatch.setattr(sys, "argv", ["ingest_actuals.py", *argv])
    code = ingest_actuals._cli()
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_cli_installs_a_workbook_as_a_csv(isolated, tmp_path, monkeypatch, capsys):
    new = _frame("2024-01-01", "2024-04-30")
    book = _workbook(tmp_path / "master.xlsx", {"Data": new})

    code, out = _run_cli(monkeypatch, capsys, "--file", str(book), "--install")

    assert code == 0, out["check"]["blockers"]
    assert out["install"]["installed"] is True
    assert out["check"]["candidate"].endswith(".csv")
    assert not isolated.read_bytes().startswith(b"PK")
    pd.testing.assert_frame_equal(_read(isolated), _as_read(new), check_exact=True)


def test_the_cli_refuses_a_wrong_first_sheet_and_writes_nothing(
        isolated, tmp_path, monkeypatch, capsys):
    before = isolated.read_bytes()
    notes = pd.DataFrame({"Prepared by": ["Synthetic"]})
    book = _workbook(tmp_path / "master.xlsx",
                     {"Notes": notes, "Data": _frame("2024-01-01", "2024-04-30")})

    code, out = _run_cli(monkeypatch, capsys, "--file", str(book), "--install")

    assert code == 1
    assert "first sheet" in out["check"]["blockers"][0]
    assert isolated.read_bytes() == before
    assert not (tmp_path / "backups").exists(), "a refusal must not rotate a backup"
