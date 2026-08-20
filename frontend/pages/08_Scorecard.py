# pages/08_Scorecard.py — Forecast against reality, and the one place actuals arrive.
#
# This page answers the only question that finally matters about a forecast: when the day
# came, what actually happened? Everything else in this app is measured on history the models
# were held back from. This is measured on days that had not happened when the number was
# published, which is a stronger claim and a much smaller table.
#
# Two rules are load-bearing here rather than stylistic:
#
#   1. "Pending" is shown, always, with its dates. A track record that displays only the rows
#      it has scored is a track record that can be made to look good by scoring selectively.
#      Today that list is the whole story: 25 predicted days, none of them arrived yet.
#   2. Uploading actuals goes through backend/ingest_actuals.py, the same validation and
#      install the command line uses. A second ingest path in the UI would be a second set of
#      checks, and the easier one to reach is the one that gets used.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APPROOT = Path(__file__).resolve().parents[1]
REPOROOT = APPROOT.parent
sys.path.insert(0, str(REPOROOT / "backend"))

from format_gel import NOT_REPORTED, UNIT_LABEL  # noqa: E402
from format_gel import gel_millions as m  # noqa: E402
from paths import scorecard_is_overridden, scorecard_path  # noqa: E402
from ui_styles import COLORS, inject_global_css, page_header, section_header  # noqa: E402
from ui_styles import inject_design_system, plotly_chrome  # noqa: E402
from i18n import install as install_language  # language toggle + pending-review note
from i18n import t as _t  # this page's fixed headings
from ui_styles import glossary_note  # plain-language definitions, on demand
from ui_styles import page_intro  # the one-or-two-sentence intro every page opens with
from ui_styles import render_app_header  # noqa: E402
from ui_styles import render_brand  # the one brand header, in the sidebar

st.set_page_config(page_title="Scorecard · Treasury Forecast", layout="wide")
inject_global_css()
inject_design_system()

# The language toggle and, in Georgian, the standing note that the translation has
# not been reviewed by a native speaker. One call per page; everything else the
# reader sees is translated inside the shared helpers.
render_brand()
install_language()
render_app_header("Scorecard", "What was forecast, and what actually happened")
page_intro(
    "This page compares published forecasts to what actually happened, once the day "
    "arrives. It is also where newly reported actuals are uploaded."
)
glossary_note("pending", "champion", "baseline", "skill", "P10", "P50", "P90")

DATA = REPOROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
UPLOAD_DIR = APPROOT / "runs_uploads" / "actuals"


# ── What this page is, and where you are in the loop ──────────────────────────
#
# The state figures come from the scorer, not from a sentence somebody typed. "25 rows are
# waiting" was true when it was written and would be a lie the day the first actual arrives.
st.markdown(_t(
    "Published forecasts are compared to what actually happened, once the real figures "
    "arrive. This is also where you upload them."
))

st.markdown(f"**{_t('How the loop works.')}** " + _t(
    "A forecast is written down before the day it describes, and left exactly as issued. When "
    "the Treasury reports that day's real figure, you upload it here. The forecast is then "
    "scored against it. Until the figure arrives there is nothing to score against, so the "
    "forecast waits and is listed as pending."
))

with st.expander(_t("What does this mean?")):
    st.markdown(
        "- A published forecast is **immutable**. It is written once, with the date it was "
        "issued, and never edited afterwards. That is what makes this a record rather than a "
        "reconstruction.\n"
        "- **Pending** means the actual figure for that day has not arrived yet. The scorer "
        "refuses to evaluate a date whose truth is not in the data, which is why the pending "
        "list is shown in full rather than hidden.\n"
        "- **Absolute error** is the gap between the central estimate and the actual figure, "
        "ignoring whether the forecast was high or low.\n"
        "- **Inside the range** means the actual figure fell between the low and high "
        "estimates that were published with it. Those were advertised to cover eight days in "
        "ten, so a hit rate close to 80% is what a well calibrated range looks like.\n"
        "- **Better than the naive rule** compares the forecast to assuming the value from "
        "five working days earlier simply repeats. It is the shared benchmark every model in "
        "this project is measured against.\n"
        "- **Better than the current method** compares it to the Treasury's own planning "
        "construction. That comparison is reported and never used to decide whether a "
        "forecast may be published."
    )


# ──────────────────────────────────────────────────────────────────────
# Reading the record
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def load_scoring() -> Dict:
    """Score whatever truth has arrived, and report the rest as pending.

    The same call the Forecast page's track record makes. It never raises for a pending
    date, because a fresh forecast being unscoreable is the normal state and not an error.
    """
    try:
        from published_forecasts import list_published, score_published

        out = score_published(DATA)
        out["issues_list"] = [p.name for p in list_published()]
        return out
    except Exception as exc:                      # noqa: BLE001 - a page must still render
        return {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(show_spinner=False, ttl=30)
def load_scorecard_rows(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=30)
def load_published_rows(scorecard_path_str: str) -> pd.DataFrame:
    """Every published prediction, one row each, marked scored or pending.

    Read from the publication log itself, ``forecasts/published/<issue_date>/forecast.csv``,
    because that IS the record: the file written on the day, never edited since. Nothing is
    copied or cached anywhere else.

    Whether a row is scored is NOT decided here. ``score_published`` has just written
    ``forecasts/scorecard.csv``, and a row is scored if it appears in that file. Re-deciding it
    on this page would be a second implementation of the scorer's own rule about which dates
    have truth, and the easier one to reach is the one that goes wrong.

    Returns an empty frame when nothing has been published, which is the state of a fresh
    clone: ``forecasts/published/`` is not in the repository.
    """
    from published_forecasts import list_published

    frames = []
    for issue_dir in list_published():
        fc_path = issue_dir / "forecast.csv"
        try:
            fc = pd.read_csv(fc_path)
        except (OSError, ValueError):
            continue
        fc["issue_date"] = issue_dir.name
        frames.append(fc)
    if not frames:
        return pd.DataFrame()

    published = pd.concat(frames, ignore_index=True)

    scored_keys = set()
    sc = Path(scorecard_path_str)
    if sc.exists():
        try:
            done = pd.read_csv(sc)
        except (OSError, ValueError):
            done = pd.DataFrame()
        if not done.empty and {"issue_date", "target", "target_date"} <= set(done.columns):
            if "y_true" in done.columns:
                done = done[done["y_true"].notna()]
            scored_keys = set(zip(done["issue_date"].astype(str),
                                  done["target"].astype(str),
                                  done["target_date"].astype(str)))

    published["scored"] = [
        (str(i), str(t), str(d)) in scored_keys
        for i, t, d in zip(published["issue_date"], published["target"],
                           published["target_date"])
    ]
    return published.sort_values(["target_date", "target", "issue_date"])


def _published_table(g: pd.DataFrame) -> pd.DataFrame:
    """The published rows in the words and units a reader uses."""
    out = pd.DataFrame()
    out[_t("For the day")] = g["target_date"].astype(str)
    out[_t("Treasury line")] = g["target"].astype(str)
    out[_t("Issued")] = g["issue_date"].astype(str)
    out[_t("Champion model")] = g["point_model"].astype(str)
    out[_t("Low (P10)")] = g["p10"].map(m)
    out[_t("Central (P50)")] = g["p50"].map(m)
    out[_t("High (P90)")] = g["p90"].map(m)
    return out


scoring = load_scoring()
if "error" in scoring:
    st.error(
        "**The track record could not be read.** Nothing has been changed. The system "
        f"reported: {scoring['error']}."
    )
    st.stop()

sc_path = scorecard_path()
rows = load_scorecard_rows(str(sc_path))

if scorecard_is_overridden():
    st.warning(
        "**You are looking at a substituted scorecard, not the real one.** The file being "
        f"read is {sc_path}. This is a display override used to exercise the page against "
        "example rows. Nothing on this screen describes a real published forecast."
    )

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Predictions scored", f"{scoring['scored']}",
              help="Published forecasts whose day has arrived and whose actual figure is in "
                   "the data.")
with c2:
    st.metric("Still pending", f"{scoring['pending']}",
              help="Published forecasts for days that have not been reported yet. There is "
                   "nothing to score them against, so they are counted and listed rather "
                   "than evaluated.")
with c3:
    st.metric("Forecast issues retained", f"{scoring['issues']}",
              help="How many separate publication dates are held. Each is written once and "
                   "never edited.")
with c4:
    _last = pd.to_datetime(pd.read_csv(DATA, usecols=["date"])["date"],
                           errors="coerce").max() if DATA.exists() else None
    st.metric("Data reported through", str(_last.date()) if _last is not None else NOT_REPORTED,
              help="The last day the source data covers. Forecasts for later days cannot be "
                   "scored until the data reaches them.")


# ── Where you actually are, said in one sentence from the live numbers ─────────
if scoring["scored"] == 0 and scoring["pending"] > 0:
    st.info(
        f"**{scoring['pending']} " + _t("published forecast rows are waiting for actual "
        "figures. Upload the Treasury's reported values to score them.") + "**"
    )
elif scoring["pending"] > 0:
    st.info(
        f"**{scoring['scored']} " + _t("rows scored, ") + f"{scoring['pending']} "
        + _t("still waiting for actual figures.") + "**"
    )
elif scoring["scored"] > 0:
    st.success(_t("Every published forecast has been scored. Nothing is waiting."))
else:
    st.info(_t(
        "Nothing has been published yet, so there is nothing to score. A forecast is "
        "published from the Forecast page, in Official mode."
    ))

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLISHED FORECASTS
#
# The inventory: what was published, and which rows are still waiting. Read straight from
# forecasts/published/<issue_date>/forecast.csv, the file written on the issue date and never
# edited since, so this table cannot disagree with the record.
#
# Pending is shown FIRST and in full. A track record that displays only the rows it has scored
# is a track record that can be made to look good by scoring selectively.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(_t("Published forecasts"),
                           _t("Every forecast that was issued, and whether it has been scored")),
            unsafe_allow_html=True)

published = load_published_rows(str(sc_path))

if published.empty:
    st.info(_t(
        "No published forecast is held on this machine. Publishing writes a dated folder under "
        "forecasts/published/, which is not kept in the repository, so a fresh clone starts "
        "with none. Issue one from the Forecast page, in Official mode."
    ))
else:
    _pending_rows = published[~published["scored"]]
    _scored_rows = published[published["scored"]]

    st.caption(
        f"{len(published)} " + _t("published row(s) across") + f" {published['issue_date'].nunique()} "
        + _t("issue(s).") + " " + _t(
            "A day can appear more than once, because a line forecast in two issues from the "
            "same origin is two separate published predictions."
        )
    )

    if not _pending_rows.empty:
        st.markdown(f"**{_t('Waiting for actual figures')}** ({len(_pending_rows)})")
        st.caption(_t(
            "Every one of these was written down before the day it describes. As soon as that "
            "day's real figure is uploaded below, it is scored and moves to the results."
        ))
        st.dataframe(_published_table(_pending_rows), hide_index=True,
                     use_container_width=True)

    if not _scored_rows.empty:
        st.markdown(f"**{_t('Already scored')}** ({len(_scored_rows)})")
        st.caption(_t("The error and the skill for these are in the results section below."))
        st.dataframe(_published_table(_scored_rows), hide_index=True,
                     use_container_width=True)
    else:
        st.caption(_t("None scored yet, so the results section below is empty."))



# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
#
# Error and skill per scored day, plus one plain verdict on whether the model is holding up.
#
# The pending list is NOT repeated here. It is in the published forecasts section above, in
# full, which is where somebody looking for "what is waiting" will look. This section is about
# rows that have an answer.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(_t("Results"),
                           _t("How the scored forecasts actually did")),
            unsafe_allow_html=True)

_DISPLAY_COLUMNS = {
    "issue_date": "Issued",
    "target_date": "For the day",
    "p50": "Central estimate",
    "y_true": "Actual",
    "abs_error": "Absolute error",
    "inside_interval": "Inside the range",
    "skill_vs_ruler_pct": "Better than the naive rule by",
    "skill_vs_ops": "Better than the current method by",
}


def _as_table(g: pd.DataFrame) -> pd.DataFrame:
    """The scored rows in the words and units a reader uses."""
    out = pd.DataFrame()
    out["Issued"] = g["issue_date"].astype(str)
    out["For the day"] = g["target_date"].astype(str)
    out["Central estimate"] = g["p50"].map(m)
    out["Actual"] = g["y_true"].map(m)
    out["Absolute error"] = g["abs_error"].map(m)
    out["Inside the range"] = g["inside_interval"].map(
        lambda v: "yes" if bool(v) else "no")
    out["Better than the naive rule by"] = g["skill_vs_ruler_pct"].map(
        lambda v: f"{float(v):.1f}%" if pd.notna(v) else NOT_REPORTED)
    out["Better than the current method by"] = (
        g["skill_vs_ops"].map(lambda v: f"{float(v):.1f}%" if pd.notna(v) else NOT_REPORTED)
        if "skill_vs_ops" in g.columns else [NOT_REPORTED] * len(g))
    return out


def _health_verdict(g: pd.DataFrame) -> str:
    """Healthy or degrading, from what the scorer already recorded. No new measurement.

    Two questions, both answered from columns that are already in the scorecard:

      * is it still beating the shared benchmark? ``skill_vs_ruler_pct`` above zero;
      * is the published range still covering what it claims? ``inside_interval`` near the
        advertised level.

    Deliberately not a trend, a rolling window or a significance test. On a handful of scored
    days none of those would mean anything, and a confident-looking verdict computed from four
    rows is worse than no verdict. So this says what it can see and says how much it saw.
    """
    n = len(g)
    skill = g["skill_vs_ruler_pct"].dropna()
    beats = float(skill.mean()) if len(skill) else None
    hit = float(g["inside_interval"].astype(bool).mean()) if n else None
    nominal = (float(g["interval_nominal"].dropna().iloc[0])
               if "interval_nominal" in g.columns and g["interval_nominal"].notna().any()
               else 0.80)

    if n < 5:
        return (f"**{_t('Too few scored days to call it either way.')}** "
                + _t("Only") + f" {n} " + _t(
                    "day(s) have an actual figure so far. A verdict from this many rows would "
                    "be a guess with a confident face on it."))

    problems = []
    if beats is not None and beats <= 0:
        problems.append(_t(
            "it is no longer more accurate than the simple rule of thumb it is measured "
            "against"))
    if hit is not None and hit < nominal - 0.15:
        problems.append(_t("the published range is covering fewer days than it claims to"))

    if not problems:
        return (f"**{_t('Holding up.')}** " + _t("Over") + f" {n} " + _t(
            "scored day(s) it is still more accurate than the simple rule of thumb, and the "
            "published range is covering about as many days as it claims."))
    return (f"**{_t('Degrading.')}** " + _t("Over") + f" {n} " + _t("scored day(s), ")
            + _t(" and ").join(problems) + ". " + _t(
                "This is the signal that the choice of model deserves looking at again. "
                "Re-choosing it is a deliberate, recorded decision, not something this page "
                "does."))


def _chart(g: pd.DataFrame, target: str) -> go.Figure:
    """Central estimate, published range and actual, on one axis."""
    d = pd.to_datetime(g["target_date"], errors="coerce")
    fig = go.Figure()
    if {"p10", "p90"} <= set(g.columns) and g["p10"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(d) + list(d[::-1]),
            y=list(g["p90"] / 1e6) + list((g["p10"] / 1e6)[::-1]),
            fill="toself", fillcolor="rgba(99,110,250,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="Published range"))
    fig.add_trace(go.Scatter(x=d, y=g["p50"] / 1e6, mode="lines+markers",
                             line=dict(color=COLORS["info"], width=3),
                             name="Central estimate"))
    fig.add_trace(go.Scatter(x=d, y=g["y_true"] / 1e6, mode="lines+markers",
                             line=dict(color="black", width=2),
                             marker=dict(symbol="x", size=8), name="Actual"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Million lari", xaxis_title=None,
                      legend=dict(orientation="h", y=-0.2), hovermode="x unified",
                      title=f"{target}: what was forecast, and what happened")
    plotly_chrome(fig)
    return fig


if rows.empty or "y_true" not in rows.columns or rows["y_true"].notna().sum() == 0:
    st.info(
        f"**{_t('Nothing has been scored yet, and that is the honest state.')}** "
        + _t(
            "A forecast is scored only once its actual figure arrives, and the scorer refuses "
            "to evaluate a day whose real value it does not hold. That refusal is what keeps "
            "this a record of what was said in advance rather than a re-run of history."
        )
        + " " + _t("The rows that are waiting are listed above.")
    )
else:
    scored = rows[rows["y_true"].notna()].copy()
    st.caption(
        f"{len(scored)} scored prediction(s). {UNIT_LABEL.capitalize()}, except where a "
        f"percentage is shown."
    )
    for target, g in scored.groupby("target"):
        g = g.sort_values("target_date")
        st.markdown(section_header(str(target), f"{len(g)} scored prediction(s)"),
                    unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric(f"Typical error ({UNIT_LABEL})", m(g["abs_error"].mean()),
                      help="The average size of the gap between the central estimate and "
                           "the actual figure, ignoring direction.")
        with s2:
            _hit = float(g["inside_interval"].astype(bool).mean())
            st.metric("Actual fell inside the range", f"{_hit:.0%}",
                      help="The published range was advertised to cover eight days in ten, "
                           "so a figure close to 80% is what a well calibrated range looks "
                           "like. Well below it means the range is too narrow.")
        with s3:
            _skill = g["skill_vs_ruler_pct"].dropna()
            st.metric("Better than the naive rule by",
                      f"{_skill.mean():.1f}%" if len(_skill) else NOT_REPORTED,
                      help="Averaged over this target's scored days. The naive rule is "
                           "assuming the value from five working days earlier repeats.")

        st.markdown(_health_verdict(g), unsafe_allow_html=False)

        st.plotly_chart(_chart(g, str(target)), use_container_width=True,
                        config={"displaylogo": False})
        st.dataframe(_as_table(g), hide_index=True, use_container_width=True)

if scoring.get("baseline_disagreements"):
    st.warning(
        f"**{len(scoring['baseline_disagreements'])} row(s) disagree with their own recorded "
        f"benchmark.** The benchmark stored in the published file and the one recomputed from "
        f"the actuals do not match. Each row is still scored against what was published, "
        f"because that is what the forecast was committed against, but the disagreement is "
        f"reported rather than absorbed."
    )
    st.dataframe(pd.DataFrame(scoring["baseline_disagreements"]),
                 hide_index=True, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD ACTUALS
#
# The one place new truth enters the system. It calls backend/ingest_actuals.py, which is the
# same validation and install the command line uses -- there is deliberately no second
# implementation here, because two ingest paths means two sets of checks and the easier one to
# reach is the one that gets used.
#
# Nothing is written until a person has read what will change and confirmed it. The
# confirmation is not a formality: an install replaces the file every pipeline reads.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(_t("Upload actuals"),
                           _t("Add the days the Treasury has now reported, and score against them")),
            unsafe_allow_html=True)

st.markdown(_t(
    "Upload the data file with the newly reported days in it. Nothing is written until you "
    "confirm, and you will see exactly what would change before you do. The file currently in "
    "use is kept with a timestamp, so an upload can always be undone."
))
st.caption(_t(
    "Upload the whole file, not just the new rows. It must contain every column the current "
    "file has, and its last day must be later than the last day already held."
))

uploaded = st.file_uploader(
    "Updated data file (CSV)", type=["csv"],
    help="It must contain every column the current file has, and its dates must reach past "
         "the last day already held. Anything else is refused with the reason stated.")

if uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = UPLOAD_DIR / uploaded.name
    candidate.write_bytes(uploaded.getbuffer())

    from ingest_actuals import IngestRefused, install, validate

    check = validate(candidate)
    summary = check.summary or {}

    st.markdown("**What this file would change**")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Rows now", f"{summary.get('rows_now', 0):,}")
        st.metric("Rows in the upload", f"{summary.get('rows_new', 0):,}")
    with q2:
        st.metric("Last day now", str(summary.get("last_date_now") or NOT_REPORTED))
        st.metric("Last day after", str(summary.get("last_date_new") or NOT_REPORTED))
    with q3:
        st.metric("New days added", f"{summary.get('rows_added', 0):,}")
        st.metric("Existing values revised", f"{summary.get('revisions', 0):,}")

    for w in check.warnings:
        st.warning(w)

    if not check.ok:
        st.error(f"**{_t('This file was not installed, and nothing on disk has changed.')}** "
                 + _t("Each reason below has to be fixed in the file before it can be used."))
        for b in check.blockers:
            st.error(b)
    else:
        st.success(
            f"**{_t('This file passed every check.')}** " + _t(
                "Nothing has been written yet. Confirming replaces the data every forecast is "
                "produced and scored from, and then scores the published forecasts against it."
            )
        )
        confirmed = st.checkbox(
            "I understand this replaces the data every forecast is produced and scored from. "
            "The file being replaced will be kept with a timestamp.")
        if st.button("Install this file and score the forecasts",
                     type="primary", disabled=not confirmed):
            try:
                result = install(candidate, check=check)
            except IngestRefused as exc:
                st.error(f"**Nothing was installed.** {exc}")
            else:
                st.success(
                    f"**Installed.** The data now runs to {result.last_date_after}, with "
                    f"{result.rows_added:,} new day(s) added and {result.revisions:,} existing "
                    f"value(s) revised. The previous file was kept at {result.backup}."
                )
                with st.spinner("Scoring the published forecasts against the new actuals …"):
                    load_scoring.clear()
                    load_scorecard_rows.clear()
                    rescored = load_scoring()
                if "error" in rescored:
                    st.error(
                        "The data was installed, but scoring afterwards did not finish. The "
                        f"system reported: {rescored['error']}."
                    )
                else:
                    st.success(
                        f"**Scored.** {rescored['scored']} prediction(s) now have an actual "
                        f"figure to be measured against, and {rescored['pending']} are still "
                        f"waiting for theirs. The record was written to "
                        f"{rescored['scorecard']}."
                    )
                    st.caption("Reload this page to see the updated table above.")


with st.expander("What happens when a file is uploaded"):
    st.markdown(
        "1. The file is read and checked. It must contain every column the current data has, "
        "its dates must be readable and unrepeated, and its last day must be later than the "
        "last day already held. A file that is byte for byte identical to the one installed "
        "is refused, because installing it would change nothing.\n"
        "2. Values that changed on days already held are counted and reported. Revised "
        "actuals are normal, so this is stated rather than refused.\n"
        "3. Nothing is written until you confirm. On confirmation the file currently in use "
        "is copied into a timestamped backup, and the upload takes its place.\n"
        "4. Every published forecast is then scored against the new actuals. Days that are "
        "still ahead of the data stay pending, which is what the scorer does rather than "
        "guessing at them."
    )

_backups = []
try:
    from ingest_actuals import list_backups

    _backups = list_backups()
except Exception:                                # noqa: BLE001 - never block the page
    _backups = []

if _backups:
    with st.expander(f"{len(_backups)} previous data file(s) kept"):
        st.caption(
            "Each one is the file that was in use immediately before an upload replaced it. "
            "They are never removed automatically."
        )
        st.dataframe(pd.DataFrame({"File": [p.name for p in _backups],
                                   "Kept at": [str(p.parent) for p in _backups]}),
                     hide_index=True, use_container_width=True)


st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RETRAINING
#
# An explainer, not a button, and deliberately so. There is no retrain path in this UI and
# none is being added: re-choosing which model is champion is a deliberate, recorded decision
# taken on TRAIN and DEV data, not something a page does after an upload.
#
# Every sentence below comes from registry.champion_policy(), which is built from
# registry/recipes.json and documents itself as "render `statement` verbatim". Writing the
# policy out by hand here would be a second copy of it, and the copy is the one that goes
# stale.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header(_t("What happens to the model when new data arrives"),
                           _t("It is refitted. It is never re-chosen on its own")),
            unsafe_allow_html=True)

try:
    from registry import champion_policy

    _policy = champion_policy()
except Exception as _exc:                        # noqa: BLE001 - never block the page
    _policy = None
    st.caption(f"{_t('The champion policy could not be read.')} {type(_exc).__name__}.")

if _policy:
    st.markdown(f"**{_t('When you upload actuals.')}** " + _t(
        "Installing the file replaces the data the system reads, and nothing else happens at "
        "that moment. The next time an official forecast runs, it fits the champion model "
        "again on all the data, the new days included. That happens on its own, with no "
        "button to press."
    ))

    st.markdown(f"**{_t('What never changes on its own.')}** " + _policy["statement"])

    with st.expander(_t("Why re-choosing is a deliberate decision")):
        st.markdown(_policy["why"])
        st.markdown(f"**{_t('The risk this leaves.')}** " + _policy["risk"])
        st.markdown(f"**{_t('What re-choosing would require.')}**")
        for _req in _policy["reselection_requires"]:
            st.markdown(f"- {_req}")
        if _policy.get("caveats"):
            st.markdown(f"**{_t('Measured instances the registry already records.')}**")
            for _cav in _policy["caveats"]:
                st.markdown(f"- **{_cav['target']}** ({_cav['recipe_id']}). {_cav['finding']}")
                if _cav.get("how_to_quote"):
                    st.caption(_cav["how_to_quote"])

    st.markdown(f"**{_t('If the results above say degrading.')}** " + _t(
        "That is the signal to look at the choice of model again. It is not something this "
        "page acts on. The written procedure is in the repository."
    ))

_REFRESH_DOC = REPOROOT / "docs" / "REFRESH_AND_RETRAIN.md"
if _REFRESH_DOC.exists():
    with st.expander(_t("Read the refresh and retrain procedure")):
        st.markdown(_REFRESH_DOC.read_text(encoding="utf-8"))
    st.caption("Rendered from `docs/REFRESH_AND_RETRAIN.md`.")
else:
    st.caption(_t("The written procedure belongs at `docs/REFRESH_AND_RETRAIN.md`."))
