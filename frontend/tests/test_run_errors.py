"""A failed run must produce a sentence, never a traceback and never raw JSON.

What went wrong before
----------------------
The Lab's failure path was ``st.error(f"Run failed (exit code {rc}). Check the log below
for details.")`` followed by five thousand characters of backend log. For a guarded
refusal that is doubly wrong: the reader is shown a Python traceback, and is told the
software failed when in fact the software declined to do something it should not do.

These tests hold two lines. Every recognised failure maps to a complete sentence with no
jargon and no punctuation debris, and a guard refusal is distinguishable from a fault, so
the UI can word the two differently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from run_errors import (  # noqa: E402
    RunFailure,
    explain_failure,
    is_guard_refusal,
)

#: Every class the mapping claims to know, with a message shaped like the real one.
KNOWN = [
    ("SelectionOnReportOnlyDataError",
     "b_ml_pipeline.select_best_model('State budget balance', h=6): refusing to select "
     "on report-only data. Rows fall in test (n=148, from 2025-01-01)."),
    ("TestWindowAccessError", "Refusing to read the TEST window (2025-01-01 onward)."),
    ("HoldoutAccessError", "touched window(s) ['test']"),
    ("FileNotFoundError", "No such file or directory: 'master_daily_clean_treasury.csv'"),
    ("PermissionError", "[Errno 13] Permission denied"),
    ("MemoryError", ""),
    ("KeyError", "'State budget balance'"),
    ("ImportError", "cannot import name 'CatBoostRegressor'"),
    ("ModuleNotFoundError", "No module named 'catboost'"),
    ("ValueError", "Window 'train' has 40 rows: too few for 5 folds at horizon 6"),
    ("ValueError", "5 folds of 126 rows plus min_train_rows=1008 do not fit"),
    ("ValueError", "No dates fall in window 'dev'."),
    ("ValueError", "eval_start 2025-09-01 is beyond the last origin date"),
    ("ValueError", "Degenerate MASE scale (0.0) for 'Revenues'"),
    ("ValueError", "Need more than 5 points in 'train' to scale MASE."),
]


def _write_report(tmp_path: Path, error_type: str, message: str) -> Path:
    out = tmp_path / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "error.json").write_text(json.dumps({
        "error_type": error_type,
        "error": message,
        "traceback": f"Traceback (most recent call last):\n  ...\n{error_type}: {message}\n",
    }), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("error_type,message", KNOWN)
def test_every_known_failure_reads_as_prose(tmp_path, error_type, message):
    root = _write_report(tmp_path, error_type, message)
    failure = explain_failure(root, log_text="")

    assert failure.headline
    assert failure.headline.endswith("."), "a headline must be a finished sentence"
    assert "Traceback" not in failure.headline
    assert "{" not in failure.headline and "}" not in failure.headline
    assert "--" not in failure.headline, "no double hyphens in user-visible copy"
    assert "—" not in failure.headline, "no em dashes in user-visible copy"
    assert error_type not in failure.headline, "the class name is not an explanation"


@pytest.mark.parametrize("error_type,message", KNOWN)
def test_the_traceback_survives_but_only_as_detail(tmp_path, error_type, message):
    failure = explain_failure(_write_report(tmp_path, error_type, message), log_text="")
    assert "Traceback" in failure.detail
    assert failure.error_type == error_type


def test_the_selection_guard_reads_as_a_deliberate_stop(tmp_path):
    """The single message a Treasury reader is most likely to meet."""
    root = _write_report(tmp_path, "SelectionOnReportOnlyDataError", KNOWN[0][1])
    failure = explain_failure(root, log_text="")
    assert is_guard_refusal(failure)
    assert "sealed window" in failure.headline
    assert "train and dev" in failure.headline


def test_a_fault_is_not_reported_as_a_deliberate_stop(tmp_path):
    failure = explain_failure(_write_report(tmp_path, "FileNotFoundError", "x"), log_text="")
    assert not is_guard_refusal(failure)


def test_two_value_errors_do_not_collapse_into_one_message(tmp_path):
    """``ValueError`` alone says nothing, so the message has to be read.

    Without the fragment table every configuration problem in the backend would arrive as
    the same unhelpful sentence.
    """
    a = explain_failure(_write_report(tmp_path / "a", "ValueError",
                                      "Window 'train' has 40 rows: too few for 5 folds"))
    b = explain_failure(_write_report(tmp_path / "b", "ValueError",
                                      "Degenerate MASE scale (0.0) for 'Revenues'"))
    assert a.headline != b.headline


def test_an_unrecognised_class_still_gets_a_finished_sentence(tmp_path):
    failure = explain_failure(_write_report(tmp_path, "SomeNovelError", "unexpected"))
    assert failure.headline.endswith(".")
    assert "Traceback" not in failure.headline


def test_the_log_tail_is_used_when_no_report_was_written():
    """Old runs, and processes killed before they could write, still get explained."""
    log = (
        "[runner] START pipeline\n"
        "Traceback (most recent call last):\n"
        '  File "b_ml_pipeline.py", line 1135, in run_pipeline_ml\n'
        "evaluation_windows.SelectionOnReportOnlyDataError: refusing to select on "
        "report-only data. Rows fall in test (n=148, from 2025-01-01).\n"
    )
    failure = explain_failure("/nonexistent", log_text=log)
    assert failure.error_type == "SelectionOnReportOnlyDataError"
    assert is_guard_refusal(failure)
    assert "sealed window" in failure.headline


def test_a_silent_death_names_the_exit_code_rather_than_guessing():
    failure = explain_failure("/nonexistent", log_text="", exit_code=137)
    assert "137" in failure.headline
    assert failure.headline.endswith(".")


def test_a_silent_death_without_an_exit_code_still_says_something():
    failure = explain_failure("/nonexistent", log_text="")
    assert failure.headline.endswith(".")
    assert failure.error_type == ""


def test_a_corrupt_report_falls_back_to_the_log(tmp_path):
    out = tmp_path / "artifacts"
    out.mkdir(parents=True)
    (out / "error.json").write_text("{not json", encoding="utf-8")
    failure = explain_failure(tmp_path, log_text="MemoryError: out of memory\n")
    assert failure.error_type == "MemoryError"


def test_run_failure_is_a_three_field_record():
    """Guards the shape the Lab unpacks."""
    assert RunFailure._fields == ("headline", "error_type", "detail")
