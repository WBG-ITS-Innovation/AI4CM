"""Turn a failed run into a sentence a Treasury reader can act on.

Why this module exists
----------------------
A guarded refusal is not a crash. When ``assert_selection_free`` stops a run, the system
is working: it has just declined to choose a model using data that choices may not
touch. Showing that as ``SelectionOnReportOnlyDataError`` above a forty-line traceback
tells the reader the software is broken, which is the opposite of what happened.

So every failure the Lab knows about gets a complete sentence saying what happened and
what to do next. Failures it does not recognise get a complete sentence too, plus the
one line of technical detail that identifies them; the traceback stays available behind
an expander rather than being the headline.

The mapping keys are exception class names, taken from ``artifacts/error.json`` written
by ``backend/runner_errors.py``. When that file is missing -- an old run, or a process
killed before it could write -- the log tail is scanned for the last ``Name: message``
line Python prints at the end of a traceback, which recovers the class name in practice.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple, Optional

#: Last line of a Python traceback: "some.module.ExceptionName: the message".
_FINAL_LINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?([A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt))\s*:\s*(.*)$"
)


class RunFailure(NamedTuple):
    """What to show, and what to keep behind the expander."""

    headline: str        #: One or two complete sentences, no jargon, no traceback.
    error_type: str      #: Exception class name, or "" when it could not be recovered.
    detail: str          #: Traceback or log tail. Never shown by default.


#: Plain-language explanations, keyed by exception class name.
#:
#: Each entry is written to be read by someone who has never seen this codebase: it says
#: what the system did, why, and what the reader can change. None of them apologise for
#: the guard, because the guard doing its job is the good outcome.
_EXPLANATIONS = {
    "SelectionOnReportOnlyDataError": (
        "This run was stopped before it could compare models using data that is not "
        "allowed to influence a choice. Exploratory runs are measured on train and dev "
        "data only, so the sealed window stays unspent for the final official reading. "
        "Check that the evaluation end date in the advanced settings is not later than "
        "the last dev date."
    ),
    "TestWindowAccessError": (
        "This run tried to read the sealed window, which stays closed except for one "
        "deliberate, recorded final reading. Nothing was read. Narrow the evaluation "
        "window to train and dev data and run again."
    ),
    "HoldoutAccessError": (
        "This run tried to read data that is held back from model fitting. Nothing was "
        "read. Narrow the evaluation window to train and dev data and run again."
    ),
    "FileNotFoundError": (
        "A file this run needed was not found. Confirm that the data file is still in "
        "place and that the backend folder on the Overview page points at this project."
    ),
    "PermissionError": (
        "This run could not read or write a file it needed because the operating system "
        "refused access. Check the permissions on the project folder and run again."
    ),
    "MemoryError": (
        "This run ran out of memory. Choose a shorter horizon, fewer folds, or the Demo "
        "profile, and run again."
    ),
    "KeyError": (
        "A column this run expected was missing from the data file. Confirm that the "
        "date column and the target column are named as the Lab expects."
    ),
    "ImportError": (
        "A software package this model needs is not installed in the backend "
        "environment. Install it, or choose a model that is already available."
    ),
    "ModuleNotFoundError": (
        "A software package this model needs is not installed in the backend "
        "environment. Install it, or choose a model that is already available."
    ),
}

#: Message fragments that pin down a ``ValueError``, which on its own says nothing.
#: Ordered: the first fragment found in the message wins.
_VALUE_ERROR_HINTS = (
    (
        "too few for",
        "There is not enough history before the sealed window to build the requested "
        "number of folds at this horizon. Choose the Demo profile, a shorter horizon, "
        "or fewer folds, and run again.",
    ),
    (
        "do not fit",
        "The requested folds do not fit inside the available history at this horizon. "
        "Choose fewer folds or a shorter horizon and run again.",
    ),
    (
        "no dates fall in window",
        "No rows of this data file fall inside the window this run was asked to measure "
        "on. Load a file with earlier history, or widen the horizon settings."
    ),
    (
        "beyond the last origin",
        "The evaluation start date is later than the last date this data file can "
        "forecast from. Choose an earlier evaluation start, or load more recent data."
    ),
    (
        "degenerate mase scale",
        "This target does not vary enough in the training data to score against, so a "
        "skill number would be meaningless. Choose a different target or cadence."
    ),
    (
        "need more than",
        "There are too few observations in the training window to score this target. "
        "Choose a coarser cadence, or load a file with more history."
    ),
)

#: Shown when the class name means nothing to us. Still a complete sentence.
_UNKNOWN = (
    "This run did not finish. The technical detail below names the step that failed; "
    "the run wrote nothing to the official forecast folders."
)

#: Shown when the process died without leaving any recoverable message at all.
_SILENT = (
    "This run stopped without reporting a reason. The full log is below. If it is "
    "empty, the backend Python environment on the Overview page is the first thing "
    "to check."
)


def _first_sentence(text: str) -> str:
    """The first line of a message, trimmed, for use as one-line technical detail."""
    line = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    return line[:300]


def _explain(error_type: str, message: str) -> str:
    """Map one exception class plus its message to a complete plain sentence."""
    if error_type == "ValueError":
        low = (message or "").lower()
        for fragment, sentence in _VALUE_ERROR_HINTS:
            if fragment in low:
                return sentence
        return (
            "This run was given a setting it cannot work with. "
            + (f"The backend reported: {_first_sentence(message)}" if message else _UNKNOWN)
        )
    known = _EXPLANATIONS.get(error_type)
    if known:
        return known
    if not error_type:
        return _SILENT
    return _UNKNOWN


def _read_error_report(out_root) -> Optional[dict]:
    path = Path(out_root) / "artifacts" / "error.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _scan_log(text: str) -> tuple:
    """Recover ``(error_type, message)`` from the tail of a backend log."""
    for line in reversed((text or "").splitlines()):
        match = _FINAL_LINE.match(line.strip())
        if match:
            return match.group(1), match.group(2)
    return "", ""


def explain_failure(out_root, log_text: str = "", exit_code: Optional[int] = None) -> RunFailure:
    """Describe a failed run in plain words.

    ``out_root`` is the run's output root; ``log_text`` is whatever the Lab already has
    of the backend log. Either may be empty, and the result is still a complete sentence.
    """
    report = _read_error_report(out_root)
    if report:
        error_type = str(report.get("error_type") or "")
        message = str(report.get("error") or "")
        detail = str(report.get("traceback") or "") or log_text
    else:
        error_type, message = _scan_log(log_text)
        detail = log_text

    headline = _explain(error_type, message)
    if not report and not error_type and exit_code not in (None, 0):
        headline = (
            f"This run stopped with exit code {int(exit_code)} and did not report a "
            "reason. The full log is below."
        )
    return RunFailure(headline=headline, error_type=error_type, detail=detail or "")


def is_guard_refusal(failure: RunFailure) -> bool:
    """True when the run was stopped by a discipline guard rather than a fault.

    The Lab words these differently: a guard refusal is the system working, so it is
    shown as a warning with the exploratory note beside it, not as an error.
    """
    return failure.error_type in (
        "SelectionOnReportOnlyDataError",
        "TestWindowAccessError",
        "HoldoutAccessError",
    )
