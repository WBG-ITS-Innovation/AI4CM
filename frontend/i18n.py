"""Two languages, one dictionary file, and an honest account of how far it goes.

The approach, and why this one
------------------------------
Translations are keyed by their **English source text**, not by an invented identifier.
``t("Start here")`` returns the Georgian if the dictionary has it and the English if it
does not. That choice has one real cost and two real benefits, and the cost is the reason
the coverage figure below exists.

The cost: editing an English string silently orphans its translation. A key-based scheme
would survive that, at the price of every call site becoming an identifier a reader of the
code cannot understand and a translator cannot see in context.

The benefits: the English stays legible at every call site, so the code still reads as
English prose; and a native speaker corrects exactly one file, ``translations_ka.py``,
where each entry is an English sentence next to its Georgian, with no key to look up. That
was the requirement, and this is the shape that meets it.

The cost is made visible rather than argued away. :func:`coverage` reports how many of the
phrases the app asks for are actually translated, the language note on every page states
it, and ``test_i18n.py`` fails if an entry in the dictionary no longer matches any string
in the app, which is exactly the orphaning the source-text scheme risks.

What is translated, and what is not
-----------------------------------
Chrome, page intros, the guide page, tooltips, glossary definitions and the verdict
sentences. **Not** data: a target name, a model name, a date, a figure or a file path
stays as it is, because translating "LightGBM_L1" or "2025-08-06" would make the app
harder to use rather than easier, and translating a Treasury line name would break the
correspondence with the source file.

Status
------
The Georgian is machine-generated and has not been reviewed by a native speaker. Every
page says so while Georgian is selected. It is a starting point for that review, not a
finished translation, and nothing in this project should be shown to a Georgian-speaking
audience as though the review had happened.
"""
from __future__ import annotations

from typing import Dict, List, Optional

#: The languages the app offers, in the order the toggle shows them.
LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ka": "ქართული",
}

DEFAULT_LANGUAGE = "en"

#: Where the selected language lives between reruns.
SESSION_KEY = "ai4cm_language"

#: Shown on every page while Georgian is selected. Deliberately not dismissible.
PENDING_REVIEW_NOTE_KA = (
    "თარგმანი ელოდება მშობლიური ენის მცოდნის შემოწმებას. ეს ტექსტი მანქანურად არის "
    "თარგმნილი და შესაძლოა შეიცავდეს შეცდომებს. რიცხვები, თარიღები და მოდელების "
    "სახელები არ ითარგმნება."
)

PENDING_REVIEW_NOTE_EN = (
    "Translation pending native review. This text was machine generated and may contain "
    "errors. Figures, dates and model names are not translated."
)

#: Translation domains.
#:
#: The distinction is between copy the dictionary has committed to covering and strings
#: that arrive at the same helper carrying data. ``section_header`` is called both with a
#: fixed heading and with a Treasury line name, and only the first is something a
#: translator should ever see. Both still go through the lookup, so a fixed heading is
#: translated wherever it appears; what differs is whether a miss counts against coverage.
#:
#: Without this the coverage figure counted "Expenditure" and
#: "HistGBDT_L1 · recipe statebudgetbalance-histgbdt-l1-ws5-v1" as untranslated phrases,
#: which both understated the translation and implied a model name ought to be translated.
DOMAIN_UI = "ui"
DOMAIN_MIXED = "mixed"

#: Every English phrase the app has asked to translate this session, whether or not the
#: dictionary had it, paired with its domain. Used by :func:`coverage` and by the test
#: that catches orphaned dictionary entries.
_REQUESTED: List[str] = []
_REQUESTED_UI: List[str] = []


def _table(language: str) -> Dict[str, str]:
    if language == "ka":
        from translations_ka import TRANSLATIONS

        return TRANSLATIONS
    return {}


def current_language(session_state=None) -> str:
    """The selected language, defaulting to English.

    ``session_state`` is injected in tests so the translation layer can be exercised
    without a Streamlit runtime, which is the only reason this is a parameter.
    """
    if session_state is None:
        try:
            import streamlit as st

            session_state = st.session_state
        except Exception:                          # noqa: BLE001 - no runtime, no choice
            return DEFAULT_LANGUAGE
    value = session_state.get(SESSION_KEY, DEFAULT_LANGUAGE)
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


def t(text: str, language: Optional[str] = None, session_state=None,
      domain: str = DOMAIN_UI) -> str:
    """The Georgian for ``text`` if the dictionary has it, otherwise ``text`` unchanged.

    Never raises and never returns an empty string. A missing translation shows the
    English, which is a worse experience than a translation and a much better one than a
    blank or a key.

    ``domain`` says whether a miss should count against coverage. See the domain constants.
    """
    if not text:
        return text
    lang = language or current_language(session_state)
    _REQUESTED.append(text)
    if domain == DOMAIN_UI:
        _REQUESTED_UI.append(text)
    if lang == DEFAULT_LANGUAGE:
        return text
    return _table(lang).get(text, text)


def coverage(language: str = "ka", phrases: Optional[List[str]] = None) -> Dict:
    """How much of the interface copy is actually translated.

    The honest counterpart to a source-text scheme: it says how far the dictionary reaches
    instead of leaving a reader to discover the gaps one page at a time. Counted over the
    ``ui`` domain only, so a Treasury line name passing through a shared header helper is
    not reported as a missing translation.
    """
    table = _table(language)
    asked = sorted(set(phrases if phrases is not None else _REQUESTED_UI))
    have = [p for p in asked if p in table]
    return {
        "language": language,
        "asked": len(asked),
        "translated": len(have),
        "missing": [p for p in asked if p not in table],
        "percent": round(100.0 * len(have) / len(asked), 1) if asked else 0.0,
    }


def requested_phrases(domain: str = DOMAIN_UI) -> List[str]:
    """Every phrase translation was requested for in ``domain``, in first-seen order."""
    source = _REQUESTED_UI if domain == DOMAIN_UI else _REQUESTED
    seen, out = set(), []
    for phrase in source:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


def reset_requested() -> None:
    """Forget what has been asked for. Used between test cases."""
    _REQUESTED.clear()
    _REQUESTED_UI.clear()


# ---------------------------------------------------------------------------
# The toggle and the note, rendered on every page
# ---------------------------------------------------------------------------

def language_toggle(location: str = "sidebar") -> str:
    """Render the language selector and return the chosen language code.

    Lives in the sidebar so it sits above the page on every page without each page having
    to leave room for it, and so it stays in the same place as a reader moves around.
    """
    import streamlit as st

    container = st.sidebar if location == "sidebar" else st
    codes = list(LANGUAGES)
    current = current_language()
    chosen = container.radio(
        "Language / ენა",
        codes,
        index=codes.index(current),
        format_func=lambda code: LANGUAGES[code],
        horizontal=True,
        key=SESSION_KEY,
        help="Switch the interface language. Figures, dates, target names and model names "
             "are shown as they are recorded and are not translated.",
    )
    return chosen


def language_note() -> None:
    """State the translation's status while Georgian is selected. Not dismissible.

    A machine translation that does not say it is one is a machine translation somebody
    will quote. The coverage figure is measured rather than claimed, so the note gets less
    apologetic as the dictionary fills up rather than staying the same sentence forever.
    """
    import streamlit as st

    if current_language() != "ka":
        return
    stats = coverage("ka", phrases=requested_phrases())
    st.sidebar.warning(
        f"{PENDING_REVIEW_NOTE_KA}\n\n"
        f"({stats['translated']} / {stats['asked']}, {stats['percent']}%)"
    )


def install(location: str = "sidebar") -> str:
    """The one call a page makes: show the toggle, show the note, return the language."""
    chosen = language_toggle(location)
    language_note()
    return chosen
