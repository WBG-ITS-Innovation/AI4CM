"""A gate verdict has three states, and no page may flatten it to two.

The bug this holds down
-----------------------
``dev_credentials.gates.<name>.passed`` is tri-state. ``true`` and ``false`` are
verdicts; ``null`` means the check was never run. Every champion recipe in the registry
carries ``coverage.passed = null``, because those models report no prediction intervals,
so their calibration could not be measured.

Two pages rendered the field with ``"passed" if g.get("passed") else "failed"``, which
reports "failed" for all three of them -- a result the Lab never obtained. Both then
printed the reason immediately afterwards: "this model reports no prediction intervals,
so their calibration was not measured". The verdict argued with its own caption, and the
verdict is the part a reader believes.

It was fixed on the Forecast page first and missed on the Documentation page, because the
mapping was written inline in both. That is why it now lives in one place, and why the
last test here scans every page rather than the two known ones.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

PAGES = sorted((FRONTEND / "pages").glob("*.py"))

#: The shape of the bug: a truthiness test over `passed` cannot tell False from None.
COLLAPSE = re.compile(r'\(\s*["\']passed["\']\s*\)\s*if\s+\w+\.get\(\s*["\']passed["\']\s*\)')


def _helper():
    try:
        from format_gel import gate_verdict
    except ImportError:
        pytest.fail("format_gel has no `gate_verdict`: gate outcomes are still being "
                    "mapped with a truthiness test inside the pages")
    return gate_verdict


def test_a_gate_that_was_never_tested_is_not_reported_as_failed():
    """``None`` is the whole point: it must not land on either verdict."""
    verdict = _helper()
    assert verdict(None) == "not tested"
    assert verdict(None) != "failed"
    assert verdict(None) != "passed"


def test_true_and_false_still_read_as_before():
    verdict = _helper()
    assert verdict(True) == "passed"
    assert verdict(False) == "failed"


def test_an_unrecognised_value_is_not_reported_as_a_pass():
    """An artifact that grows a new value must fail safe, not claim a verdict it lacks."""
    verdict = _helper()
    assert verdict("maybe") == "not tested"


def test_the_forecast_page_no_longer_collapses_the_gate_to_a_boolean():
    source = (FRONTEND / "pages" / "07_Forecast.py").read_text(encoding="utf-8")
    assert not COLLAPSE.search(source), (
        "07_Forecast.py is collapsing a tri-state gate with a truthiness test again; "
        "`passed=None` renders as 'failed'")


def test_the_documentation_page_no_longer_collapses_the_gate_to_a_boolean():
    source = (FRONTEND / "pages" / "09_Documentation.py").read_text(encoding="utf-8")
    assert not COLLAPSE.search(source), (
        "09_Documentation.py is collapsing a tri-state gate with a truthiness test again; "
        "`passed=None` renders as 'failed'")


def test_no_page_re_implements_the_gate_mapping():
    """The generalisation of the two tests above.

    The Documentation page kept the bug for as long as it did because the mapping was
    written out inline in each page, so fixing one said nothing about the other. Scanning
    every page means a third copy cannot appear quietly.
    """
    offenders = [p.name for p in PAGES if COLLAPSE.search(p.read_text(encoding="utf-8"))]
    assert not offenders, (
        "these pages map a tri-state gate with a truthiness test instead of calling "
        f"format_gel.gate_verdict: {offenders}")
