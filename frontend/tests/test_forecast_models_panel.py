"""The Forecast page must name the champion, show the runners-up, and publish neither.

What this holds down
--------------------
The Models panel adds two things that could go wrong quietly.

* It could stop rendering. A panel that silently disappears when the ledger is unreadable
  leaves the page looking exactly as it did before, so "there is only one model" would be
  back without anyone noticing. These tests assert the champion, the runners-up and the
  comparison against the Treasury's current method are all actually on the page.
* The comparison could stop being exploratory. That is the serious one. The separation is
  enforced in ``backend/forecast_modes.py``, and the tests below assert this page never
  reaches for the official path when comparing, and that the banner saying so is present
  both before a comparison runs and after it produces numbers.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

from registry import load_registry  # noqa: E402

PAGE = FRONTEND / "pages" / "07_Forecast.py"
SOURCE = PAGE.read_text(encoding="utf-8")
TARGETS = [r["target"] for r in load_registry()["recipes"]]


def _compare_body() -> str:
    """Just the comparison helper, with adjacent string literals rejoined.

    Copy in this file is written as implicitly concatenated literals across lines, so a
    sentence a reader sees as one string is several in the source. Without the rejoin, an
    assertion about that sentence fails on where the line happened to wrap.

    The function is located by parsing rather than by slicing between text markers. It used to
    slice from the ``def`` to the comment ``# ── Per target``, and when the page was split into
    tabs that comment became indented, so the slice ran to the end of the file and swallowed the
    official-mode dispatch four hundred lines away. The test then failed while the property it
    guards was still true, which is the worst way for a test to fail.
    """
    tree = ast.parse(SOURCE)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_render_compare_alternatives"),
              None)
    assert fn is not None, "_render_compare_alternatives has gone from the Forecast page"
    body = ast.get_source_segment(SOURCE, fn)
    return re.sub(r'"\s*\n\s*"', "", body)


@pytest.fixture(scope="module")
def rendered() -> str:
    """The whole page as text, once. Rendering it is slow and nothing here mutates it."""
    at = AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    parts = []
    for collection in ("markdown", "caption", "info", "warning", "error", "success",
                       "metric", "subheader", "header"):
        for element in getattr(at, collection, []):
            parts.append(str(getattr(element, "value", "")) or str(element))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The panel is there
# ---------------------------------------------------------------------------

def test_the_panel_renders(rendered):
    assert "Models measured on this line" in rendered


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_names_its_champion(rendered, target):
    champion = next(r["point_model"] for r in load_registry()["recipes"]
                    if r["target"] == target)
    assert champion in rendered, f"the champion for {target} is not named on the page"


def test_the_champion_is_marked_as_the_champion(rendered):
    assert "champion" in rendered.lower()


def test_the_champion_sentence_states_the_evidence(rendered):
    """The one sentence that has to survive every future copy edit."""
    assert "chosen by measured evidence" in rendered
    assert "never fitted on" in rendered


def test_the_treasury_method_comparison_is_present(rendered):
    assert "Treasury's current method" in rendered


def test_a_target_with_no_eligible_alternative_explains_itself(rendered):
    """State budget balance has twelve measured models and no gate-eligible one."""
    assert "none of them beat the naive benchmark" in rendered


# ---------------------------------------------------------------------------
# The comparison never becomes official
# ---------------------------------------------------------------------------

def test_the_comparison_dispatches_only_the_exploratory_mode():
    """Read at the source, because this is the property a future edit would break.

    ``_render_compare_alternatives`` is the only new code path that runs a model, and it
    must never pass ``--mode official`` or ``--publish``.
    """
    body = _compare_body()
    assert '"--mode", "exploratory"' in body
    assert '"--mode", "official"' not in body
    assert "--publish" not in body
    assert "publish_official" not in body


def test_the_comparison_is_bannered_before_and_after_it_runs():
    assert _compare_body().count("**EXPLORATORY.**") >= 2, (
        "the banner must be visible before the run and beside its results"
    )


def test_the_comparison_states_that_nothing_is_written():
    body = _compare_body()
    for claim in ("not published", "not written to the official forecast",
                  "not entered in the", "scorecard"):
        assert claim in body


def test_the_official_button_says_the_model_is_not_selectable(rendered):
    assert "The model is not selectable in this mode" in rendered


def test_the_official_button_names_the_champion_recipe():
    assert 'st.button("Run the champion recipe"' in SOURCE


# ---------------------------------------------------------------------------
# Every verdict opens with a sentence, not a term of art
# ---------------------------------------------------------------------------

def test_each_verdict_has_a_plain_intro_sentence():
    for opener in ("**Usable as a forecast.** Every check",
                   "**Do not use these numbers.** A simple rule of thumb",
                   "**Shown as a guide to the typical level, not as a forecast.**"):
        assert opener in SOURCE


def test_no_verdict_banner_shouts_a_bare_code(rendered):
    """``withheld_as_forecast`` is a code in the registry and must not reach the reader.

    It leaked in two places: the verdict banner at the top of each target, and the verdict
    history at the foot of the page, which printed the raw code on both sides of an arrow.
    """
    assert "withheld_as_forecast" not in rendered
    assert "WITHHELD —" not in rendered
    assert "shown as a guide only" in rendered, (
        "the verdict history must show words, not codes"
    )


def test_the_panel_copy_has_no_em_dashes_or_double_hyphens():
    """Sentence discipline, checked on the strings this task introduced."""
    panel = SOURCE.split("MODELS PANEL")[1].split("if sec:")[0]
    for line in panel.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for quoted in re.findall(r'"([^"]*)"', stripped):
            assert "—" not in quoted, f"em dash in user-visible copy: {quoted}"
            assert "--" not in quoted, f"double hyphen in user-visible copy: {quoted}"
