"""The guide must cover every page, link to it correctly, and explain the two doors.

Why each of these is a test rather than a review note
----------------------------------------------------
A guide is the first thing that goes stale. A page gets added or renamed, the guide does
not move, and the result is worse than no guide: a reader is now looking at a list that
confidently omits the page they need. So coverage is asserted against the directory
listing rather than against a list somebody maintains.

The two-doors section is asserted separately because it carries the one distinction the
whole system turns on. If it ever softens into "experiments are less formal", the page has
stopped saying the thing it exists to say.
"""
from __future__ import annotations

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

PAGE = FRONTEND / "pages" / "00_Start_here.py"
SOURCE = PAGE.read_text(encoding="utf-8")
PAGES_DIR = FRONTEND / "pages"

#: Every page file in the app, excluding the guide itself.
OTHER_PAGES = sorted(p.name for p in PAGES_DIR.glob("*.py") if p.name != PAGE.name)


@pytest.fixture(scope="module")
def rendered() -> str:
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    at.run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    parts = []
    for collection in ("markdown", "caption", "info", "warning", "error", "success",
                       "subheader", "header"):
        for element in getattr(at, collection, []):
            parts.append(str(getattr(element, "value", "")))
    # Elements inside a column are not reachable from the top-level accessors, and the two
    # doors are rendered in two columns. Without this the section that carries the most
    # important distinction in the app would be invisible to every test here.
    for column in at.get("column"):
        for collection in ("markdown", "caption", "info", "warning", "error", "success"):
            for element in getattr(column, collection, []):
                parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# It is first, and it is complete
# ---------------------------------------------------------------------------

def test_the_guide_sorts_first_in_the_sidebar():
    """Streamlit orders the sidebar by filename, so being first is a naming fact."""
    names = sorted(p.name for p in PAGES_DIR.glob("*.py"))
    assert names[0] == PAGE.name, f"the sidebar would open on {names[0]}"


@pytest.mark.parametrize("page", OTHER_PAGES)
def test_every_page_has_a_section(page):
    """Asserted against the directory, so adding a page without a section fails here.

    A guide that confidently omits the page a reader needs is worse than no guide.
    """
    assert f"pages/{page}" in SOURCE, f"{page} exists in the app and is absent from the guide"


def test_the_overview_entry_point_is_covered_too():
    assert "Overview.py" in SOURCE


@pytest.mark.parametrize("page", OTHER_PAGES)
def test_every_linked_page_actually_exists(page):
    """The other direction: a link to a page that was renamed is a dead end."""
    for match in re.findall(r'"(pages/[^"]+\.py)"', SOURCE):
        assert (FRONTEND / match).exists(), f"the guide links to {match}, which does not exist"


def test_every_section_answers_the_same_three_questions(rendered):
    """One shape per section, so the page can be scanned rather than read."""
    assert rendered.count("**What you can do there.**") == len(OTHER_PAGES) + 1
    assert rendered.count("**One thing to try.**") == len(OTHER_PAGES) + 1


def test_the_lab_is_described_in_two_sentences_at_the_top(rendered):
    assert "forecasts daily Treasury cash lines" in rendered
    assert "every number it shows can be checked" in rendered


# ---------------------------------------------------------------------------
# The two doors
# ---------------------------------------------------------------------------

def test_the_two_doors_are_explained(rendered):
    assert "The two doors" in rendered
    assert "Official" in rendered and "Exploratory" in rendered


def test_the_official_door_says_what_it_went_through_and_what_it_permits(rendered):
    assert "Once, on recorded evidence" in rendered
    assert "never been fitted on" in rendered
    assert "never re-chooses it" in rendered
    assert "Publish it." in rendered


def test_the_exploratory_door_says_plainly_that_nothing_is_published(rendered):
    assert "never published" in rendered
    assert "never entered in the" in rendered
    assert "no verdict attaches to the result" in rendered


def test_the_two_doors_do_not_soften_into_a_matter_of_degree(rendered):
    """The separation is a type boundary in the code, not a convention about care.

    If this section ever reads as "experiments are less formal", the page has stopped
    saying the thing it exists to say.
    """
    lowered = rendered.lower()
    for softening in ("less formal", "informal", "roughly", "usually not published",
                      "generally not published"):
        assert softening not in lowered


# ---------------------------------------------------------------------------
# Sentence discipline and honesty
# ---------------------------------------------------------------------------

def test_no_em_dashes_or_double_hyphens_in_visible_copy():
    for line in SOURCE.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for quoted in re.findall(r'"([^"]*)"', stripped):
            assert "—" not in quoted, f"em dash in user-visible copy: {quoted}"
            assert "--" not in quoted, f"double hyphen in user-visible copy: {quoted}"


def test_the_guide_does_not_overclaim(rendered):
    lowered = rendered.lower()
    for overclaim in ("proven in production", "guaranteed", "state of the art",
                      "always accurate", "best in class"):
        if overclaim == "proven in production":
            # Present once, as a denial. Anything else is the claim itself.
            assert "nothing here has been proven in production" in lowered
            continue
        assert overclaim not in lowered


def test_the_guide_says_what_the_evaluation_actually_is(rendered):
    assert "honestly evaluated" in rendered
    assert "held back from" in rendered


def test_no_sentence_is_left_unfinished():
    """Sweeps this page's own copy for a truncated or placeholder string."""
    for quoted in re.findall(r'"([^"]{20,})"', SOURCE):
        if quoted.startswith("pages/") or quoted.endswith(".py"):
            continue
        assert "TODO" not in quoted and "TBD" not in quoted and "..." not in quoted, quoted


def test_the_overview_page_points_at_the_guide():
    """A guide nobody can find from the landing page is a guide nobody reads."""
    overview = (FRONTEND / "Overview.py").read_text(encoding="utf-8")
    assert 'st.page_link("pages/00_Start_here.py"' in overview
    assert "New here?" in overview
