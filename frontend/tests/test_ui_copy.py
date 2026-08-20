"""House style for everything a reader sees, enforced rather than reviewed.

Why these are tests
-------------------
Copy rules that live in a style note get followed for a fortnight. These are the rules the
MVP consolidation set, applied to every page at once, and each of them corresponds to
something that was actually wrong:

* **Pages opened on a control.** Compare Runs opened with a run selector, so a reader had
  to work out from the widgets what the page was for. Every page now calls ``page_intro``.
* **Terms arrived unexplained.** MASE, skill, champion, holdout, sealed window, withheld,
  P10/P50/P90 all reached a reader with nothing beside them. ``ui_styles.GLOSSARY`` is the
  one definition of each, and a page that uses a term must carry its definition.
* **Long explanations sat in front of the reader.** Several pages opened with four
  paragraphs of methodology. Anything past a paragraph belongs behind an expander.
* **Punctuation and unfinished sentences.** Em dashes and double hyphens throughout,
  inherited from the commit messages and comments, where they are fine and where they stay.

The definition of "user-visible" lives in ``ui_copy.py``: it parses each page and collects
the arguments of the calls that render text. A comment is never a call argument, so the
long explanatory comments this project relies on are out of scope by construction, which
is the only reason a rule this strict is liveable.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_copy import (  # noqa: E402
    Copy,
    all_text_of,
    helper_modules,
    pages,
    terms_used,
    visible_copy,
)

pytest.importorskip("streamlit", reason="ui_styles imports streamlit")
from ui_styles import GLOSSARY  # noqa: E402

PAGES = pages(FRONTEND)
HELPERS = helper_modules(FRONTEND)
ALL_FILES = PAGES + HELPERS

#: Strings that are markup, identifiers or data rather than prose, so the sentence rules
#: do not apply to them. Kept narrow on purpose: a broad exemption is how a rule dies.
_NOT_PROSE = re.compile(
    r"^(?:[<>{}\[\]#*_`|\-=\s]|https?://|rgba?\(|#[0-9a-fA-F]{3,8}$)"
)


def _is_prose(text: str) -> bool:
    """Is this string a sentence a reader reads, rather than markup or a column name?

    Leading emphasis and list markers are stripped first. Copy in this app frequently opens
    with a bold lead-in, and treating "**What it is.** A forecast produced by ..." as markup
    because of its first character exempted the longest blocks on the page from every rule
    below, which is the opposite of what a prose filter is for.
    """
    stripped = text.strip().lstrip("*_-# \n")
    if len(stripped) < 25:
        return False
    if _NOT_PROSE.match(stripped):
        return False
    if stripped.count(" ") < 4:
        return False
    if "<" in stripped and ">" in stripped and "style" in stripped.lower():
        return False
    return True


def _prose(path: Path):
    return [c for c in visible_copy(path) if _is_prose(c.text)]


def _report(bad) -> str:
    return "\n".join(f"  {c.where()}: {c.text[:110]}" for c in bad)


# ---------------------------------------------------------------------------
# 1. Punctuation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_no_em_dashes_in_visible_copy(path):
    """Em and en dashes read as informal asides and the house style forbids them.

    They are fine in comments and commit messages, which this rule never sees.
    """
    bad = [c for c in visible_copy(path) if "—" in c.prose_only or "–" in c.prose_only]
    assert not bad, f"em or en dash in user-visible copy:\n{_report(bad)}"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_no_double_hyphens_in_visible_copy(path):
    bad = [c for c in visible_copy(path)
           if "--" in c.prose_only and "<!--" not in c.text and "-->" not in c.text]
    assert not bad, f"double hyphen in user-visible copy:\n{_report(bad)}"


# ---------------------------------------------------------------------------
# 2. Nothing truncated, nothing placeholder, nothing overclaimed
# ---------------------------------------------------------------------------

PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "lorem ipsum", "coming soon",
                "under construction", "placeholder text")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_no_placeholder_strings(path):
    bad = [c for c in visible_copy(path)
           if any(marker.lower() in c.text.lower() for marker in PLACEHOLDERS)]
    assert not bad, f"placeholder copy:\n{_report(bad)}"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_no_ascii_ellipsis_standing_in_for_a_finished_sentence(path):
    """``...`` is nearly always a sentence somebody meant to finish.

    A single-character ellipsis is allowed, because that is what the spinners use while
    something is running and it is a deliberate mark rather than a trailing off.
    """
    bad = [c for c in visible_copy(path) if "..." in c.prose_only]
    assert not bad, f"unfinished sentence:\n{_report(bad)}"


OVERCLAIMS = ("proven in production", "production-proven", "guaranteed", "state of the art",
              "state-of-the-art", "best in class", "best-in-class", "world class",
              "always accurate", "never wrong", "fully validated", "battle-tested")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_nothing_a_reader_sees_overclaims(path):
    """"Honestly evaluated", never "proven in production"."""
    bad = []
    for c in visible_copy(path):
        lowered = c.text.lower()
        for claim in OVERCLAIMS:
            if claim not in lowered:
                continue
            # A denial is not a claim. "Nothing here has been proven in production" is
            # exactly the sentence this project should be making.
            before = lowered.split(claim)[0][-40:]
            if any(neg in before for neg in ("not ", "nothing ", "never ", "no ")):
                continue
            bad.append(c)
    assert not bad, f"overclaim in user-visible copy:\n{_report(bad)}"


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_every_prose_sentence_is_finished(path):
    """A visible paragraph must end in a full stop, question mark or colon.

    Catches the copy that trails off mid-clause, which is what a truncated string looks
    like once it is rendered.
    """
    bad = []
    for c in _prose(path):
        # A block that ends in a bullet list ends correctly. Trailing list items are
        # dropped before the last line is examined, because "- **Horizon** (steps ahead)"
        # is an item and items do not take full stops in this app's copy.
        text = c.text.rstrip()
        # A whole string wrapped in emphasis is a heading. "**Days awaiting their actual
        # figure**" introduces a table; it is not a sentence and does not take a full stop.
        if re.fullmatch(r"\*\*[^*]{3,80}\*\*", text.strip()):
            continue
        # A block ending in a fenced code block ends with the command, correctly.
        if text.rstrip().endswith("```"):
            continue
        # Peel trailing structure until a sentence is what remains. A block that ends
        # "**Theta**" followed by two bullets ends correctly: the last thing a reader sees
        # is a list item under a sub-heading, and neither takes a full stop.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        while lines:
            last = lines[-1].strip()
            # The Lab's tooltips use "• " rather than markdown bullets, because they are
            # rendered by the browser as a tooltip and not as markdown.
            is_bullet = (last[:2] in ("- ", "* ", "• ") or last[:1] == "•"
                         or last[:3].rstrip(".").isdigit())
            is_subheading = bool(re.fullmatch(r"\*\*[^*]{2,80}\*\*", last))
            if not (is_bullet or is_subheading):
                break
            lines.pop()
        end = "\n".join(lines).rstrip().rstrip("*_`)")
        if not end:
            continue
        if end[-1] in ".!?:;" or end.endswith("…"):
            continue
        # A heading or a table row is not a sentence and is not required to end in one.
        if c.call in ("header", "subheader", "title", "metric", "page_link", "expander",
                      "button", "download_button", "form_submit_button", "radio",
                      "selectbox", "checkbox", "slider", "multiselect", "text_input",
                      "number_input", "toggle", "file_uploader", "section_header",
                      "page_header", "render_app_header", "spinner", "dataframe"):
            continue
        if end.startswith("#") or "|" in end or end.endswith(("%", ")")):
            continue
        # A string ending in an interpolated value finishes at runtime, with whatever the
        # value is. "Could not parse the integrity report: {}" is a finished sentence once
        # the reason is substituted in.
        if end.endswith("{}"):
            continue
        bad.append(c)
    assert not bad, f"sentence does not finish:\n{_report(bad)}"


# ---------------------------------------------------------------------------
# 3. Every page says what it is for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_every_page_opens_with_what_it_is_for(path):
    """One shared helper, so the intro reads the same on every page.

    Compare Runs opened with a run selector and nothing else, which is the state this
    exists to prevent.
    """
    source = path.read_text(encoding="utf-8")
    assert source.count("page_intro(") >= 1, f"{path.name} has no page_intro"


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_the_intro_is_one_or_two_finished_sentences(path):
    intros = [c for c in visible_copy(path) if c.call == "page_intro"]
    assert intros, f"{path.name} has no page_intro"
    for intro in intros:
        text = intro.text.strip()
        assert 60 <= len(text) <= 420, f"{intro.where()}: {len(text)} characters"
        assert text.endswith("."), f"{intro.where()} does not finish: {text[-60:]}"
        sentences = [s for s in re.split(r"(?<=[.!?]) ", text) if s.strip()]
        assert 1 <= len(sentences) <= 3, f"{intro.where()}: {len(sentences)} sentences"


# ---------------------------------------------------------------------------
# 4. Every technical term is explained where it is used
# ---------------------------------------------------------------------------

#: Terms that must never appear to a reader without their definition on the same page.
#:
#: Deliberately not every word in the glossary. These are the ones measured as actually
#: reaching a reader unexplained, and a term nobody uses does not need policing.
POLICED_TERMS = ("MASE", "sealed window", "holdout", "champion", "exploratory", "withheld")


@pytest.mark.parametrize("term", POLICED_TERMS)
def test_every_policed_term_has_exactly_one_definition(term):
    assert term in GLOSSARY, f"{term} is policed but has no definition"
    assert GLOSSARY[term].strip().endswith("."), f"{term} has an unfinished definition"
    assert "—" not in GLOSSARY[term] and "--" not in GLOSSARY[term]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_a_page_that_uses_a_term_also_defines_it(path):
    """A term a reader meets with nothing beside it is a term they will guess at.

    Satisfied by any of: a tooltip carrying the definition, a glossary expander listing
    the term, or the definition written inline. All three are honest answers; what is not
    an answer is the bare word.
    """
    source = path.read_text(encoding="utf-8")
    visible = all_text_of(path)
    used = terms_used(visible, set(POLICED_TERMS))

    missing = []
    for term in sorted(used):
        definition = GLOSSARY[term]
        explained = (
            definition[:60] in visible                       # written out inline
            or f'"{term}"' in source and "glossary_note(" in source   # behind the expander
            or f"'{term}'" in source and "glossary_note(" in source
            or f'term_help("{term}"' in source                # in a tooltip
            or f"term_help('{term}'" in source
        )
        if not explained:
            missing.append(term)
    assert not missing, (
        f"{path.name} uses {missing} without defining any of them. Add "
        f"glossary_note(...) or term_help(...), or write the definition inline.")


# ---------------------------------------------------------------------------
# 5. Long explanations belong behind an expander
# ---------------------------------------------------------------------------

#: Longest a single visible block may be while sitting in front of the reader.
#:
#: Roughly a substantial paragraph. Past this the page stops being scannable, and the
#: material is almost always methodology a reader wants on demand rather than by default.
MAX_INLINE_CHARS = 700


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_long_explanations_sit_behind_an_expander(path):
    bad = [c for c in _prose(path)
           if len(c.text) > MAX_INLINE_CHARS and not c.in_expander]
    assert not bad, (
        f"{len(bad)} explanatory block(s) longer than {MAX_INLINE_CHARS} characters are "
        f"rendered in front of the reader. Move them inside a st.expander:\n{_report(bad)}")


def test_the_rule_is_measured_against_something_real():
    """A guard on the guard: if nothing anywhere is near the limit, the limit is not a rule."""
    longest = max((len(c.text) for p in PAGES for c in _prose(p) if c.in_expander),
                  default=0)
    assert longest > 300, (
        "no page hides a long explanation behind an expander, so the rule above is "
        "passing vacuously")
