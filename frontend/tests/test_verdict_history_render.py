"""The Forecast page must show what a published issue said THEN as well as NOW.

A published issue is immutable: its `gates.json` records the verdict at issue time. When the
publication gates were corrected in P2, every verdict moved — so the 2025-08-06 issue holds a
stock-target forecast that was publishable then and is withheld now, and a Revenues forecast that
was withheld then and is publishable now. Both statements are true and they answer different
questions.

Before this, the page rendered only the current registry verdict, so a reader of an old issue saw
the verdict it was issued under with nothing saying it no longer holds. These tests pin the
rendering, because a reconciliation that exists in a module and appears on no page is not visible to
anyone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

FORECAST = FRONTEND / "pages" / "05_Forecast.py"


@pytest.fixture(scope="module")
def rendered():
    at = AppTest.from_file(str(FORECAST), default_timeout=240).run()
    assert not at.exception, at.exception
    return at


def _all_text(at) -> str:
    return "\n".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [w.value for w in at.warning]
        + [s.value for s in at.success]
        + [e.value for e in at.error]
    )


def test_the_page_has_a_verdict_history_section(rendered):
    assert "Verdict history" in _all_text(rendered)


def test_both_verdicts_are_shown_for_a_changed_target(rendered):
    """Verdict-at-issue AND verdict-today, not one replacing the other."""
    blob = _all_text(rendered)
    assert "withheld_as_forecast" in blob, "the issue-time verdict must appear"
    assert "publishable" in blob, "the current verdict must appear"


def test_the_actionable_sentence_names_the_gate_that_drove_the_change(rendered):
    """Only the driving gate. Listing every gate that differs buries the answer."""
    blob = _all_text(rendered)
    assert "signal was re-thresholded from 1.5 to 1.15" in blob
    assert "1.2255" in blob, "the measurement is unchanged and is quoted as such"
    assert "accuracy_vs_naive" in blob, "the added gate that withheld two targets"


def test_the_page_says_the_forecast_numbers_did_not_change(rendered):
    """The most likely misreading is that the forecast was revised. It was not."""
    blob = _all_text(rendered)
    assert "have not changed" in blob
    assert "only the verdict attached to them" in blob


def test_the_page_says_the_published_issue_is_immutable(rendered):
    assert "immutable" in _all_text(rendered)
    assert "not a correction" in _all_text(rendered)


def test_every_published_issue_gets_an_expander(rendered):
    """"Same for any published issue" -- not only the one that changed."""
    labels = [e.label for e in rendered.expander if "Issue " in e.label]
    assert len(labels) >= 2, labels
    assert any("3 of 3 verdict(s) changed" in x for x in labels), labels
    assert any("unchanged" in x for x in labels), labels


def test_the_changed_count_is_stated_up_front(rendered):
    assert "published verdict(s) would differ today" in _all_text(rendered)


def test_a_hard_withheld_verdict_is_not_offered_as_a_guide_to_the_level():
    """P2 split the two withheld verdicts, and the page used to render both the same way.

    `withheld_as_forecast` keeps the numbers usable as a central-tendency estimate.
    `withheld` means a trivial benchmark is MORE accurate, so calling those numbers "a guide to
    the typical level" would invite a worse decision than showing nothing.
    """
    src = FORECAST.read_text()
    assert 'pub["verdict"] == "withheld"' in src
    assert "do not use these numbers" in src
    i_hard = src.index("do not use these numbers")
    i_guide = src.index("shown as a guide to the typical level.**\\n\\n")
    assert i_hard < i_guide, (
        "the hard-withheld branch must be checked before the softer one, or it never fires")


def test_the_reconciliation_never_blocks_the_page():
    """It is a comparison, not a dependency: a failure there must not take the page down."""
    src = FORECAST.read_text()
    i = src.index("reconcile_verdicts()")
    window = src[i - 400:i + 400]
    assert "except Exception" in window
    assert "reconciliation = []" in window
