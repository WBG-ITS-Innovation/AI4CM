"""The Georgian toggle: it works, it says it is unreviewed, and its keys have not rotted.

The risk this file exists for
-----------------------------
Translations are keyed by their English source text. That keeps the code legible and gives
a native speaker one file with English sentences on the left and Georgian on the right, and
it has one specific failure mode: edit an English string and its translation silently stops
being found. Nothing raises, nothing looks wrong, and the page quietly reverts to English.

``test_every_dictionary_key_is_still_a_string_the_app_uses`` is the guard for exactly that.
It renders every page, collects every phrase the app asked to translate, and fails if a
dictionary entry no longer matches any of them. A failure means either the English moved
and the key needs updating, or the entry is for copy that no longer exists.

The rest holds the honesty properties: English is untouched by default, Georgian mode says
on every page that the text has not been reviewed by a native speaker, and no Treasury line
name, model name or date is in the dictionary.
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

pytest.importorskip("streamlit", reason="the pages need streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import i18n  # noqa: E402
from i18n import (  # noqa: E402
    DEFAULT_LANGUAGE,
    LANGUAGES,
    PENDING_REVIEW_NOTE_KA,
    SESSION_KEY,
    coverage,
    current_language,
    t,
)
from translations_ka import TRANSLATIONS  # noqa: E402

PAGES = sorted(FRONTEND.glob("pages/*.py")) + [FRONTEND / "Overview.py"]

#: Georgian's Unicode block, for asserting text actually is Georgian.
_GEORGIAN = re.compile(r"[Ⴀ-ჿ]")


def _render(path: Path, language: str = "en") -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=180)
    at.session_state[SESSION_KEY] = language
    at.run()
    if at.exception:
        pytest.fail(f"{path.name} in {language}: "
                    + "\n".join(str(e.value) for e in at.exception))
    return at


def _text(at: AppTest) -> str:
    parts = []
    for collection in ("markdown", "caption", "info", "warning", "error", "success"):
        for element in getattr(at, collection, []):
            parts.append(str(getattr(element, "value", "")))
    for column in at.get("column"):
        for collection in ("markdown", "caption", "info", "warning", "error", "success"):
            for element in getattr(column, collection, []):
                parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


@pytest.fixture(scope="module")
def asked_phrases():
    """Every English string the app can put through the translation layer.

    Rendering the pages is most of it, but not all: a verdict banner only appears for a
    target carrying that verdict, and a tooltip in an unreached branch is real copy that
    simply did not run today. So the set is the rendered phrases plus the static copy
    dictionaries, which is the honest domain of the translation file.

    Anything outside this set is genuinely orphaned: it corresponds to no string in the
    app under any state.
    """
    i18n.reset_requested()
    for page in PAGES:
        _render(page, "ka")
    universe = set(i18n.requested_phrases()) | set(i18n.requested_phrases(i18n.DOMAIN_MIXED))

    from ui_styles import GLOSSARY, HELP

    universe |= set(HELP.values()) | set(GLOSSARY) | set(GLOSSARY.values())

    # Copy passed to a literal t("...") call. Parsed rather than matched with a regular
    # expression, because these are written as adjacent literals across several lines and a
    # pattern anchored on the opening quote captures only the first of them.
    #
    # The name to look for is read from each page's OWN import rather than assumed. It used to
    # be the fixed pair ("t", "_t"), and that silently stopped seeing the Forecast page the day
    # its import was renamed to `_translate`: three verdict banners vanished from this set, and
    # the one that renders only for a `withheld_as_forecast` target was reported as an orphaned
    # translation when the copy was untouched and on screen. A hardcoded alias list is a second
    # place to remember, so there is no list.
    import ast

    for page in PAGES:
        tree = ast.parse(page.read_text(encoding="utf-8"))
        aliases = {alias.asname or alias.name
                   for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module == "i18n"
                   for alias in node.names if alias.name == "t"}
        if not aliases:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name not in aliases:
                continue
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    universe.add(arg.value)
    return universe


# ---------------------------------------------------------------------------
# 1. The mechanism
# ---------------------------------------------------------------------------

def test_english_is_the_default_and_changes_nothing():
    assert DEFAULT_LANGUAGE == "en"
    assert t("Start here", language="en") == "Start here"
    assert t("A phrase nobody has ever translated", language="en") == \
        "A phrase nobody has ever translated"


def test_georgian_returns_georgian_for_a_known_phrase():
    out = t("Start here", language="ka")
    assert out != "Start here"
    assert _GEORGIAN.search(out)


def test_a_missing_translation_falls_back_to_english_rather_than_a_blank():
    """A blank or a key would be worse than the English, which every reader here can read."""
    assert t("No such phrase exists in the dictionary", language="ka") == \
        "No such phrase exists in the dictionary"


def test_an_empty_string_stays_empty():
    assert t("", language="ka") == ""


def test_an_unknown_language_code_falls_back_rather_than_raising():
    assert current_language({SESSION_KEY: "fr"}) == "en"
    assert current_language({}) == "en"


def test_both_languages_are_offered_and_georgian_is_named_in_georgian():
    assert set(LANGUAGES) == {"en", "ka"}
    assert _GEORGIAN.search(LANGUAGES["ka"])


# ---------------------------------------------------------------------------
# 2. The dictionary has not rotted
# ---------------------------------------------------------------------------

def test_every_dictionary_key_is_still_a_string_the_app_uses(asked_phrases):
    """The failure mode of keying on source text, caught rather than described.

    Editing an English string orphans its translation with no error and no visible sign:
    the page simply reverts to English. A key here that matches nothing the app renders
    means either the English moved and this entry needs updating, or the copy is gone and
    so should the entry be.
    """
    orphans = sorted(k for k in TRANSLATIONS if k not in asked_phrases)
    assert not orphans, (
        "these dictionary entries match no string the app renders any more:\n  "
        + "\n  ".join(repr(o[:100]) for o in orphans))


def test_every_translation_is_actually_georgian():
    """A copied English value would be an untranslated entry claiming to be translated."""
    not_georgian = [k for k, v in TRANSLATIONS.items()
                    if not _GEORGIAN.search(v) and v != k]
    # MASE, P10, P50 and P90 are deliberately left as they are: they are labels a reader
    # meets on charts and in files, and translating them would break that correspondence.
    allowed = {"MASE", "P10", "P50", "P90"}
    assert set(not_georgian) <= allowed, not_georgian


def test_no_data_string_is_in_the_dictionary():
    """Translating a Treasury line name would break the link to the file it came from."""
    forbidden = ("State budget balance", "Revenues", "Expenditure", "LightGBM_L1",
                 "HistGBDT_L1", "2025-08-06", "master_daily_clean_treasury.csv")
    present = [f for f in forbidden if f in TRANSLATIONS]
    assert not present, f"data must not be translated: {present}"


def test_every_placeholder_survives_translation():
    """A figure is substituted into ``{}`` at display time and must still have a slot."""
    for english, georgian in TRANSLATIONS.items():
        assert english.count("{}") == georgian.count("{}"), repr(english[:80])


def test_no_translation_is_empty():
    assert all(v.strip() for v in TRANSLATIONS.values())


# ---------------------------------------------------------------------------
# 3. It says it is unreviewed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_shows_the_pending_review_note_in_georgian(page):
    """A machine translation that does not say it is one is one somebody will quote."""
    at = _render(page, "ka")
    notes = [str(w.value) for w in at.sidebar.warning]
    assert any(PENDING_REVIEW_NOTE_KA[:40] in n for n in notes), (
        f"{page.name} does not carry the pending-review note")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_page_shows_the_note_in_english(page):
    at = _render(page, "en")
    notes = [str(w.value) for w in at.sidebar.warning]
    assert not any(PENDING_REVIEW_NOTE_KA[:40] in n for n in notes)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_offers_the_toggle(page):
    at = _render(page, "en")
    labels = [str(getattr(r, "label", "")) for r in at.sidebar.radio]
    assert any("ენა" in lbl for lbl in labels), f"{page.name} has no language toggle"


def test_the_note_states_measured_coverage_rather_than_claiming_completeness(asked_phrases):
    stats = coverage("ka", phrases=sorted(asked_phrases))
    assert 0 < stats["percent"] < 100.0 or stats["percent"] == 100.0
    assert stats["translated"] <= stats["asked"]


# ---------------------------------------------------------------------------
# 4. Georgian actually reaches the page
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_georgian_reaches_the_body_of_every_page(page):
    """Every page has a translated intro, so every page should render some Georgian."""
    assert _GEORGIAN.search(_text(_render(page, "ka"))), (
        f"{page.name} renders no Georgian at all with Georgian selected")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_english_mode_renders_no_georgian_body_text(page):
    """A leak the other way would mean a string was translated unconditionally."""
    body = _text(_render(page, "en"))
    # The toggle names Georgian in Georgian, which is correct and lives in the sidebar.
    assert not _GEORGIAN.search(body), f"{page.name} shows Georgian while set to English"


def test_the_lab_keeps_its_own_tooltips():
    """A first pass rewrote these to the shared tooltips, which share none of these keys.

    Every tooltip on the Lab page silently became empty and nothing failed, because an
    empty tooltip renders as no tooltip rather than as an error.
    """
    at = _render(FRONTEND / "pages" / "08_Lab.py", "en")
    widgets = at.get("selectbox") + at.get("slider") + at.get("radio")
    helps = [str(getattr(w, "help", "") or "") for w in widgets]
    assert sum(1 for h in helps if h.strip()) >= len(helps) - 1, helps
