"""A gate verdict has three states, and the page must not flatten it to two.

The bug this holds down
-----------------------
``dev_credentials.gates.<name>.passed`` is tri-state. ``true`` and ``false`` are
verdicts; ``null`` means the check was never run. Every champion recipe in the
registry carries ``coverage.passed = null``, because those models report no
prediction intervals, so their calibration could not be measured.

The page used to render the field with ``"passed" if g.get("passed") else "failed"``,
which reports "failed" for all three of them -- a result the Lab never obtained.
Worse, the reason line printed directly beneath said the opposite: "this model
reports no prediction intervals, so their calibration was not measured". The badge
argued with its own caption, and the badge is the part a reader believes.

These tests are about the mapping only. They deliberately avoid rendering the page,
so they need neither Streamlit nor a populated ``forecasts/published/``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
PAGE = FRONTEND / "pages" / "07_Forecast.py"
SOURCE = PAGE.read_text(encoding="utf-8")


def _gate_verdict_fn():
    """Pull ``_gate_verdict`` (and the table it reads) out of the page and make it callable.

    Executing the module itself would run the whole Streamlit page. Lifting the two
    definitions out keeps the test on the real source -- not a copy of it -- while staying
    independent of any data the page would otherwise need.
    """
    tree = ast.parse(SOURCE)
    wanted = {"_GATE_WORDS_EN", "_gate_verdict"}
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(SOURCE, node))
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            chunks.append(ast.get_source_segment(SOURCE, node))
    if len(chunks) < 2:
        pytest.fail(
            "the Forecast page has no tri-state gate mapping: expected module-level "
            "`_GATE_WORDS_EN` and `_gate_verdict`, found " + str(len(chunks)) + " of 2")
    ns = {"_translate": lambda s: s}       # identity: this test is about the mapping
    exec("\n\n".join(chunks), ns)          # noqa: S102 - the source under test
    return ns["_gate_verdict"]


def test_a_gate_that_was_never_tested_is_not_reported_as_failed():
    """``None`` is the whole point: it must not land on either verdict."""
    verdict = _gate_verdict_fn()
    assert verdict(None) == "not tested"
    assert verdict(None) != "failed"
    assert verdict(None) != "passed"


def test_true_and_false_still_read_as_before():
    """The fix must not disturb the two states that were already right."""
    verdict = _gate_verdict_fn()
    assert verdict(True) == "passed"
    assert verdict(False) == "failed"


def test_an_unrecognised_value_is_not_reported_as_a_pass():
    """An artifact that grows a new value must fail safe, not claim a verdict it lacks."""
    verdict = _gate_verdict_fn()
    assert verdict("maybe") == "not tested"


def test_the_page_no_longer_collapses_the_gate_to_a_boolean():
    """The original expression, named so it cannot come back unnoticed.

    A truthiness test over ``passed`` cannot distinguish ``False`` from ``None``; any
    return of this shape is the bug returning.
    """
    collapsed = '_translate("passed") if g.get("passed") else _translate("failed")'
    assert collapsed not in SOURCE, (
        "07_Forecast.py is collapsing a tri-state gate with a truthiness test again; "
        "`passed=None` renders as 'failed'")
