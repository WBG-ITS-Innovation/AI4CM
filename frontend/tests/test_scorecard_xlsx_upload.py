"""The Scorecard upload takes the master export as an Excel workbook, through the shared ingest.

How the page path is exercised
------------------------------
Streamlit's AppTest cannot put a file on a ``file_uploader``: in the installed version it is
an unknown element, readable but not settable. So these tests replace ``st.file_uploader``
with a wrapper that still renders the real widget, so its label, help and accepted types can
be read, and then hands the page a synthetic upload. Everything after that call is the
page's own code: it writes the bytes to its landing directory, lands them, validates and,
when a person confirms, installs.

Three things are redirected, each for a stated reason:

* the canonical file and its backups point at a synthetic stand-in, because the page installs
  over whatever ``ingest_actuals.CANONICAL`` names;
* ``score_published`` returns an empty result, because the page scores on every render and
  the scorer rewrites the tracked scorecard. A test of the upload has no reason to run it;
* the landing directory is a page constant and cannot be redirected, so each test lands its
  file under a unique synthetic name and removes exactly what it created afterwards.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = FRONTEND / "pages" / "08_Scorecard.py"
LANDING = FRONTEND / "runs_uploads" / "actuals"
REQUIREMENTS = FRONTEND / "requirements.txt"
COLUMNS = ["date", "Revenues", "Expenditure", "State budget balance", "is_holiday"]


def _frame(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    n = len(dates)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "Revenues": [1_000_000.25 + i for i in range(n)],
        "Expenditure": [900_000.0 + i for i in range(n)],    # whole numbers held as floats
        "State budget balance": [-50_000.5 - i for i in range(n)],
        "is_holiday": [i % 2 for i in range(n)],
    }, columns=COLUMNS)


def _workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as book:
        for name, df in sheets.items():
            out = df.copy()
            if "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"])
            out.to_excel(book, sheet_name=name, index=False)
    return path


class _Upload:
    """The two attributes of Streamlit's UploadedFile that the page reads."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self) -> memoryview:
        return memoryview(self._data)


@pytest.fixture(autouse=True)
def _clear_caches():
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def stand_in(monkeypatch, tmp_path) -> Path:
    import ingest_actuals
    import published_forecasts

    canon = tmp_path / "canonical.csv"
    _frame("2024-01-01", "2024-03-29").to_csv(canon, index=False)
    monkeypatch.setattr(ingest_actuals, "CANONICAL", canon)
    monkeypatch.setattr(ingest_actuals, "BACKUP_DIR", tmp_path / "backups")

    def no_scoring(*args, **kwargs) -> Dict:
        return {"scored": 0, "pending": 0, "issues": 0,
                "scorecard": str(tmp_path / "scorecard.csv"), "summary": {},
                "pending_dates": [], "baseline_disagreements": []}

    monkeypatch.setattr(published_forecasts, "score_published", no_scoring)
    return canon


@pytest.fixture
def upload(monkeypatch):
    """Hand the page ``path`` as if a person had chosen it in the uploader."""
    import streamlit as st

    landing_existed = LANDING.exists()
    created: List[Path] = []
    real_uploader = st.file_uploader

    def _choose(path: Path) -> str:
        name = f"pytest-synthetic-{uuid.uuid4().hex[:8]}{path.suffix}"
        created.extend([LANDING / name, LANDING / f"{name}.csv"])
        data = path.read_bytes()

        def uploader(label, *args, **kwargs):
            real_uploader(label, *args, **kwargs)
            return _Upload(name, data)

        monkeypatch.setattr(st, "file_uploader", uploader)
        return name

    yield _choose
    for path in created:
        path.unlink(missing_ok=True)
    if not landing_existed and LANDING.exists() and not any(LANDING.iterdir()):
        LANDING.rmdir()


def _run() -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=180)
    at.run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    return at


def _confirm_and_install(at: AppTest) -> AppTest:
    [box] = [c for c in at.checkbox if c.label.startswith("I understand this replaces")]
    box.check().run()
    [button] = [b for b in at.button if b.label == "Install this file and score the forecasts"]
    button.click().run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    return at


def _values(at: AppTest, kind: str) -> str:
    return "\n".join(str(e.value) for e in getattr(at, kind))


def _as_read(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


# ---------------------------------------------------------------------------
# The environment and the widget
# ---------------------------------------------------------------------------

def test_the_frontend_environment_can_read_a_workbook():
    """The page runs in this environment, and ingest reads workbooks with openpyxl."""
    pins = [line.strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
    assert any(line.startswith("openpyxl==") for line in pins), (
        "openpyxl must be pinned in frontend/requirements.txt, like every other dependency")
    import openpyxl  # noqa: F401


def test_the_uploader_accepts_a_workbook_and_says_only_the_first_sheet_is_read(stand_in):
    [widget] = _run().get("file_uploader")
    assert {".csv", ".xlsx"} <= set(widget.proto.type)
    assert "first sheet" in widget.proto.help


# ---------------------------------------------------------------------------
# A workbook through the page, end to end
# ---------------------------------------------------------------------------

def test_a_workbook_upload_is_checked_and_passes(stand_in, upload, tmp_path):
    new = _frame("2024-01-01", "2024-04-30")
    name = upload(_workbook(tmp_path / "master.xlsx", {"Data": new}))
    before = stand_in.read_bytes()

    at = _run()

    assert "This file passed every check." in _values(at, "success"), _values(at, "error")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows in the upload"] == f"{len(new):,}"
    assert (LANDING / f"{name}.csv").exists(), "the workbook was not landed as a CSV"
    assert stand_in.read_bytes() == before, "nothing may be written before confirmation"


def test_installing_a_workbook_from_the_page_puts_a_csv_in_place(stand_in, upload, tmp_path):
    new = _frame("2024-01-01", "2024-04-30")
    upload(_workbook(tmp_path / "master.xlsx", {"Data": new}))

    at = _confirm_and_install(_run())

    assert "Installed." in _values(at, "success"), _values(at, "error")
    assert not stand_in.read_bytes().startswith(b"PK"), "a workbook was installed as the data"
    import ingest_actuals

    pd.testing.assert_frame_equal(ingest_actuals._read(stand_in), _as_read(new),
                                  check_exact=True)


def test_a_csv_upload_on_the_page_is_still_installed_byte_for_byte(stand_in, upload, tmp_path):
    csv = tmp_path / "master.csv"
    _frame("2024-01-01", "2024-04-30").to_csv(csv, index=False)
    upload(csv)

    _confirm_and_install(_run())

    assert stand_in.read_bytes() == csv.read_bytes()


def test_a_wrong_first_sheet_is_refused_on_the_page_in_plain_words(stand_in, upload, tmp_path):
    notes = pd.DataFrame({"Prepared by": ["Synthetic"], "Comment": ["Weekly export"]})
    upload(_workbook(tmp_path / "master.xlsx",
                     {"Notes": notes, "Data": _frame("2024-01-01", "2024-04-30")}))
    before = stand_in.read_bytes()

    at = _run()

    errors = _values(at, "error")
    assert "This file was not installed, and nothing on disk has changed." in errors
    assert "Only the first sheet" in errors
    assert "'Notes', 'Data'" in errors
    assert "Prepared by, Comment" in errors
    assert not [c for c in at.checkbox if c.label.startswith("I understand this replaces")], (
        "a refused file must not be offered for confirmation")
    assert stand_in.read_bytes() == before
