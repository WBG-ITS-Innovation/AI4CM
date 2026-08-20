"""One error report shape, written by every runner.

Why this module exists
----------------------
When a run failed, what the UI could show depended on which family had failed.
``run_b_ml_univariate.py`` wrote ``artifacts/error.json`` with the exception type and a
full traceback; the A_STAT, C_DL and E_QUANTILE runners wrote nothing at all, so the
only evidence was the log tail. The Lab therefore printed the log tail, which for a
guarded refusal meant a wall of Python traceback in front of a Treasury reader.

The fix has two halves and this is the backend half: every runner writes the same file,
so the frontend has exactly one thing to read. ``frontend/run_errors.py`` is the other
half and turns that file into a sentence.

The traceback is still recorded. It moves behind a "Technical detail" expander in the
UI rather than being deleted, because the person debugging the Lab needs it and the
person reading a forecast does not.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

#: Path of the report, relative to a run's output root.
ERROR_REPORT_NAME = "artifacts/error.json"


def error_report_path(out_root) -> Path:
    """Where the report for ``out_root`` lives."""
    return Path(out_root) / "artifacts" / "error.json"


def write_error_report(out_root, exc: BaseException, context: Optional[str] = None) -> Path:
    """Record one failed run and return the path written.

    Never raises. A runner calls this from an ``except`` block on its way to a non-zero
    exit code, and an error while reporting an error would replace a diagnosable failure
    with a confusing one.
    """
    path = error_report_path(out_root)
    payload = {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    if context:
        payload["context"] = context
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return path
