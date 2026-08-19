# pages/02_Scorecard.py — Forecast against reality, and the one place actuals arrive.
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
from ui_styles import render_app_header  # noqa: E402

st.set_page_config(page_title="Scorecard · Treasury Forecast", page_icon="🎯", layout="wide")
inject_global_css()
inject_design_system()
render_app_header("Scorecard", "What was forecast, and what actually happened")

DATA = REPOROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
UPLOAD_DIR = APPROOT / "runs_uploads" / "actuals"

st.markdown(page_header("🎯 Scorecard",
                        "Every published forecast, scored against the actual figure once it arrives"),
            unsafe_allow_html=True)

st.markdown(
    "Every forecast this system publishes is written down before the day it describes, and is "
    "then scored against the actual figure once that figure arrives in the data. A row marked "
    "**pending** is a forecast whose day has not yet been reported, so there is nothing to "
    "score it against."
)

with st.expander("What does this mean?"):
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

st.divider()


# ──────────────────────────────────────────────────────────────────────
# Forecast against reality
# ──────────────────────────────────────────────────────────────────────
st.markdown(section_header("Forecast against reality",
                           "One row per published prediction whose day has arrived"),
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
        f"**Nothing has been scored yet, and that is the honest state.** "
        f"{scoring['issues']} forecast issue(s) are retained and "
        f"{scoring['pending']} predicted day(s) are still ahead of the data. A prediction is "
        f"scored only once its actual figure arrives, and the scorer refuses to evaluate a "
        f"date whose truth is not held. That refusal is what keeps this a record of what was "
        f"said in advance rather than a re-run of history."
    )
    pending = scoring.get("pending_dates") or []
    if pending:
        st.markdown("**Days awaiting their actual figure**")
        st.caption(
            "Every one of these was published before the day it describes. As soon as the "
            "data reaches that day, it will be scored and appear in the table above."
        )
        st.dataframe(
            pd.DataFrame(pending, columns=["Target", "For the day"]).sort_values(
                ["For the day", "Target"]),
            hide_index=True, use_container_width=True)
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

        st.plotly_chart(_chart(g, str(target)), use_container_width=True,
                        config={"displaylogo": False})
        st.dataframe(_as_table(g), hide_index=True, use_container_width=True)

    if scoring.get("pending_dates"):
        with st.expander(f"{scoring['pending']} prediction(s) still awaiting their actual figure"):
            st.dataframe(
                pd.DataFrame(scoring["pending_dates"], columns=["Target", "For the day"]),
                hide_index=True, use_container_width=True)

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
st.markdown(section_header("Upload actuals",
                           "Add newly reported days, then score the forecasts against them"),
            unsafe_allow_html=True)

st.markdown(
    "Upload an updated data file to bring newly reported days into the system. The file is "
    "checked before anything is written, and you will see exactly what will change before you "
    "confirm it. The file currently in use is kept, with a timestamp, so an upload can always "
    "be undone."
)

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
        st.error("**This file was not installed, and nothing on disk has changed.**")
        for b in check.blockers:
            st.error(b)
    else:
        st.success(
            "**This file passed every check.** Nothing has been written yet. Confirm below "
            "to replace the data the system reads and score the published forecasts against "
            "it."
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
