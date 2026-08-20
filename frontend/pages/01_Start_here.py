# pages/01_Start_here.py — The guide. First in the sidebar, and written to be read first.
#
# Why this page exists: the app had eight pages and no way in. A reader arriving at it had to
# infer from the names which one answered their question, and two of the names ("Lab",
# "Dashboard") do not tell you. This says what each page is for, what you can do there, and one
# thing to try, and it ends with the distinction the whole system turns on.
#
# Every section links to its page, so this is a way in rather than a description of one.
from __future__ import annotations

from pathlib import Path

import streamlit as st

from i18n import install as install_language  # language toggle + pending-review note
from i18n import t  # this page carries most of the fixed copy in the app
from ui_styles import glossary_note  # plain-language definitions, on demand
from ui_styles import inject_design_system, inject_global_css, page_header, page_intro
from ui_styles import render_app_header
from ui_styles import section_header

st.set_page_config(page_title="Start here · Treasury Forecast", page_icon="🧭", layout="wide")
inject_global_css()
inject_design_system()

# The language toggle and, in Georgian, the standing note that the translation has
# not been reviewed by a native speaker. One call per page; everything else the
# reader sees is translated inside the shared helpers.
install_language()
render_app_header("Start here", "What this Lab is, and where to go for each question")
page_intro(
    "This page is the way in. It says what each page of the Lab is for, what you can do "
    "there, and the difference between an official forecast and an experiment."
)
glossary_note("champion", "exploratory", "withheld", "gate", "sealed window", "baseline")

st.markdown(
    page_header("🧭 Start here",
                "What each page is for, what you can do there, and one thing to try"),
    unsafe_allow_html=True,
)

st.markdown(t(
    "This Lab forecasts daily Treasury cash lines, and it is built so that every number it "
    "shows can be checked. Nothing is published unless it has been measured on days the model "
    "was never shown, and where a model failed a check the Lab says so rather than quietly "
    "leaving it out."
))

st.info(
    "**New here? Read the two doors at the bottom of this page first.** Everything in this "
    "app is either an official forecast, which is checked and publishable, or an experiment, "
    "which is clearly marked and never published. Knowing which one you are looking at is the "
    "single most useful thing to understand."
)

st.divider()


# ──────────────────────────────────────────────────────────────────────
# The pages
# ──────────────────────────────────────────────────────────────────────
def _link(path: str, label: str) -> None:
    """A link to another page, resolved against whichever file is the entry point.

    Streamlit resolves a page link relative to the entry point, and this app has two. Run
    normally the entry point is ``Overview.py``, so the path is ``pages/07_Forecast.py``.
    Rendered in isolation, as the page tests do, the entry point is this file and the same
    link has to be ``07_Forecast.py``. Trying both means the guide renders in either case;
    falling back to a caption means a page that is genuinely missing is named rather than
    taking the whole guide down with it.
    """
    for candidate in (path, Path(path).name):
        try:
            st.page_link(candidate, label=label)
            return
        except Exception:                          # noqa: BLE001 - resolution, not logic
            continue
    st.caption(f"{label} (this page could not be linked from here)")


def _page(icon: str, title: str, path: str, purpose: str, do: str, try_this: str) -> None:
    """One section per page. The same three questions each time, so the page scans."""
    st.markdown(section_header(f"{icon} {title}", purpose), unsafe_allow_html=True)
    st.markdown(f"**{t('What you can do there.')}** {do}")
    st.markdown(f"**{t('One thing to try.')}** {try_this}")
    _link(path, f"Open {title}")
    st.write("")


_page(
    "🔭", "Forecast", "pages/07_Forecast.py",
    "The forecast itself: what each Treasury line is expected to do over the next working days.",
    "Read the central estimate and the range around it for each line, see which model produced "
    "it and what earned that model its place, and see the next best alternatives with their "
    "measured accuracy beside them. You can also generate a fresh forecast, and publish it as "
    "an official issue.",
    "Open the Models panel under any line and read the one sentence explaining why that model "
    "is the champion. Then open **Compare alternatives** to see how much the number would move "
    "if a different model produced it.",
)

_page(
    "🎯", "Scorecard", "pages/08_Scorecard.py",
    "Forecast against reality: how the published forecasts actually did once the day arrived.",
    "See every published prediction that has been scored, with the actual figure beside it, and "
    "every prediction still waiting for its day. You can also upload newly reported actuals, "
    "which are checked before anything is written and scored immediately afterwards.",
    "Look at the pending list. Every date on it was published before that day happened, which "
    "is what makes this a record rather than a re-run of history.",
)

_page(
    "📈", "Dashboard", "pages/04_Dashboard.py",
    "The detail behind one experimental run: predictions, errors and diagnostics.",
    "Overlay a model's predictions on the actual series, inspect where the error came from, and "
    "download the underlying files.",
    "Load a run and look at where its largest errors fall. On these lines they cluster on "
    "month ends and tax deadlines, which is the part of the problem that is genuinely hard.",
)

_page(
    "🔀", "Compare runs", "pages/05_Compare.py",
    "Two to six experimental runs side by side, on the same axes.",
    "Put different models, horizons or configurations next to each other and see which one is "
    "actually better rather than which one you expected to be.",
    "Compare the same model at two horizons. The further ahead a forecast reaches, the wider "
    "its honest range has to be, and this is where that becomes visible.",
)

_page(
    "🕒", "History", "pages/06_History.py",
    "Every experimental run this Lab has produced, oldest to newest.",
    "Browse past runs, see what each was configured with, and download its outputs.",
    "Find a run that failed a check. The Lab keeps those rather than deleting them, because a "
    "record that holds only the successes is not a record.",
)

_page(
    "🧩", "Models", "pages/09_Documentation.py",
    "The shelf: every model available here, what it does, and whether anybody has measured it.",
    "Read what each model is in plain language, see which ones have a recorded result and which "
    "are registered candidates nobody has run yet, and look up the exact settings any of them "
    "was configured with.",
    "Sort the shelf by status. Rather more of these are untested than measured, which is worth "
    "knowing before quoting how many models this project has.",
)

_page(
    "🧺", "Data pre-processing", "pages/02_Data_Preprocessing.py",
    "Turning a raw Treasury export into the clean daily series the models read.",
    "Upload a source file, see what the cleaning steps did to it, and check the result before "
    "it is used.",
    "Run a file through and read the report. It states what was changed and why, so a surprising "
    "number later can be traced back to a decision made here.",
)

_page(
    "🧪", "Lab", "pages/03_Lab.py",
    "The workbench: run any model on any line, at any horizon, as an experiment.",
    "Choose a family, a model, a target and a horizon, launch it, and watch the backend log as "
    "it runs. Every run here is exploratory and is measured on train and dev data only.",
    "Run the same model at two different horizons and watch the error grow. Nothing you launch "
    "here is published, so there is nothing to be careful about.",
)

st.markdown(section_header("📊 Overview", "The landing page: what the Lab is, and its settings."),
            unsafe_allow_html=True)
st.markdown(
    "**What you can do there.** Confirm the Lab can find its backend, see how many runs exist, "
    "and read what the project does and does not claim."
)
st.markdown(
    "**One thing to try.** Check the backend paths if anything elsewhere reports that the "
    "backend interpreter could not be found."
)
_link("Overview.py", "Open Overview")

st.divider()


# ══════════════════════════════════════════════════════════════════════
# THE TWO DOORS
#
# The distinction the whole system turns on, and the one thing a new reader most needs. It is
# enforced in backend/forecast_modes.py rather than by convention: an exploratory result is a
# different type with no publish path, and publish_official() refuses it outright.
# ══════════════════════════════════════════════════════════════════════
st.markdown(section_header("The two doors",
                           "Everything here is either official or exploratory, and they never mix"),
            unsafe_allow_html=True)

st.markdown(t(
    "Every number this Lab produces comes through one of two doors. Which door it came through "
    "determines what you may do with it."
))

left, right = st.columns(2, gap="large")

def _door(rows) -> None:
    """One labelled point per paragraph.

    Written as separate blocks rather than one long string on purpose: this is the section
    a new reader is most likely to actually read, and a single 800-character paragraph is
    the surest way to make sure they do not.
    """
    for label, body in rows:
        st.markdown(f"**{t(label)}** {body}")


with left:
    st.success("### 🔒 Official")
    _door([
        ("What it is.",
         "A forecast produced by the champion model for that Treasury line, at the one "
         "horizon everything here was measured at."),
        ("How the model was chosen.",
         "Once, on recorded evidence, from data it had never been fitted on. Loading new "
         "data refits the model but never re-chooses it, and no run can change the choice."),
        ("What it went through.",
         "Every publication check, each with a plain-language reason attached to its "
         "verdict. Where a check failed, the forecast is either withheld or shown as a "
         "guide to the typical level rather than as a forecast, and the page says which "
         "and why."),
        ("What you may do with it.",
         "Publish it. It is written once with the date it was issued, never edited "
         "afterwards, and scored against the actual figure when that day arrives."),
        ("Where.", "The Forecast page, in Official mode."),
    ])

with right:
    st.warning("### 🧪 Exploratory")
    _door([
        ("What it is.",
         "Any model, on any line, at any horizon, because somebody wanted to see what it "
         "would do."),
        ("How the model was chosen.", "By you, from a list. That is the point of it."),
        ("What it went through.",
         "Nothing. No check was measured for the combination you chose, so no verdict "
         "attaches to the result, and none is claimed."),
        ("What you may do with it.",
         "Look at it, compare it, learn from it. It is never published, never written to "
         "the official forecast folders, and never entered in the scorecard. Every page "
         "that produces one says so on screen while it is showing it."),
        ("Where.",
         "The Lab, the Forecast page in Exploratory mode, and Compare alternatives."),
    ])

with st.expander("Why the separation is worth this much trouble"):
    st.markdown(
        "The value of a forecast comes entirely from having been written down before the day "
        "it describes, using only what was known at the time. That is easy to say and easy to "
        "lose: one exploratory result promoted into the published record, once, and the whole "
        "track record becomes a claim rather than a measurement.\n\n"
        "So the separation is built into the code rather than left to care. An exploratory "
        "result is a different kind of object with no way to publish it at all, and the "
        "publishing function refuses anything else. A page cannot leak one into the published "
        "record by forgetting a setting, because there is no setting.\n\n"
        "The same reasoning protects the data. A block of recent history is sealed, and no "
        "experiment may be measured on it. That block is the one honest final reading this "
        "project has, and it can only be spent once. Anything you run from the Lab is bound to "
        "the earlier data by construction, and if a configuration would reach past that "
        "boundary the Lab stops it and says so in plain words."
    )

st.divider()
st.caption(t(
    "Everything in this Lab is honestly evaluated on data the models were held back from. "
    "Nothing here has been proven in production, and no page claims otherwise."
))
