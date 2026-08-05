# pages/05_Forecast.py — Forward forecast: the next five working days.
#
# This page shows the one thing the rest of the app does not: predictions for dates that do
# not exist in the data yet. Everything else in AI4CM backtests against known answers.
#
# Two presentation rules are load-bearing, not stylistic:
#   1. A model that fails a check is shown with its numbers AND its failure, side by side.
#      Hiding a withheld verdict would make this page a sales tool.
#   2. Money is in millions of lari everywhere. Raw 9-digit figures are not communication.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APPROOT = Path(__file__).resolve().parents[1]
REPOROOT = APPROOT.parent
sys.path.insert(0, str(REPOROOT / "backend"))

from ui_styles import COLORS, inject_global_css, page_header, section_header  # noqa: E402

st.set_page_config(page_title="Forecast · Treasury Forecast", page_icon="🔭", layout="wide")
inject_global_css()

GEN_CMD = "./backend/.venv/bin/python backend/run_forward_forecast.py"


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def load_all() -> Optional[Dict]:
    try:
        import insights as ins
        from registry import load_registry

        art = ins.load_forward_artifacts()
        reg = load_registry()
        narr = ins.build_narrative_text(art["forecasts"], reg, art["provenance"])
        return {
            "forecasts": pd.DataFrame(art["forecasts"]),
            "provenance": art["provenance"],
            "registry": reg,
            "narrative": narr,
            "dir": art["dir"],
        }
    except FileNotFoundError:
        return None


from format_gel import UNIT_LABEL, gel_millions as m  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────
st.markdown(page_header("🔭 Forward forecast",
                        "The next five working days — dates that are not yet in the data"),
            unsafe_allow_html=True)

data = load_all()
if data is None:
    st.warning(
        "**No forward run found.** This page shows predictions for future dates, which are "
        "generated on demand rather than committed to the repository.\n\n"
        f"Generate one with:\n```bash\n{GEN_CMD}\n```"
    )
    st.stop()

fc: pd.DataFrame = data["forecasts"]
fc["target_date"] = pd.to_datetime(fc["target_date"])
prov = data["provenance"] or {}
reg = data["registry"]
recipes = {r["target"]: r for r in reg["recipes"]}

# ── Verdict first ─────────────────────────────────────────────────────
st.markdown(data["narrative"]["narrative"]["headline"])

c1, c2, c3, c4 = st.columns(4)
n_pub = sum(1 for r in reg["recipes"]
            if r["publication"]["verdict"] == "publishable")
with c1:
    st.metric("Budget lines covered", f"{len(recipes)} of 41")
with c2:
    st.metric("Called a forecast", f"{n_pub} of {len(recipes)}")
with c3:
    st.metric("Working days ahead", str(int(fc["horizon"].max())))
with c4:
    st.metric("Data through", str(pd.to_datetime(
        prov.get("data", {}).get("latest_data_date", fc["origin_date"].max())).date()))

st.info(data["narrative"]["narrative"]["scope"])
st.markdown(data["narrative"]["narrative"]["signal_finding"])

st.divider()

# ── Per target ────────────────────────────────────────────────────────
sections = {s["target"]: s for s in data["narrative"]["narrative"]["sections"]}

for target in fc["target"].unique():
    rows = fc[fc["target"] == target].sort_values("horizon")
    rec = recipes.get(target)
    sec = sections.get(target)
    if rec is None:
        continue

    pub = rec["publication"]
    publishable = pub["verdict"] == "publishable"

    st.markdown(section_header(target, f"{rec['point_model']} · recipe {rec['id']}"),
                unsafe_allow_html=True)

    # Verdict banner — never hidden.
    if publishable:
        st.success(f"**Usable as a forecast.** {pub['reason_plain']}")
    else:
        st.error(
            f"**WITHHELD as a forecast — shown as a guide to the typical level.**\n\n"
            f"{pub['reason_plain']}"
        )
        if pub.get("named_fix"):
            st.warning(f"**What would change this:** {pub['named_fix']}")

    left, right = st.columns([3, 2], gap="large")

    # Band chart
    with left:
        fig = go.Figure()
        d = rows["target_date"]
        fig.add_trace(go.Scatter(
            x=list(d) + list(d[::-1]),
            y=list(rows["p90"] / 1e6) + list((rows["p10"] / 1e6)[::-1]),
            fill="toself", fillcolor="rgba(99,110,250,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="Likely range (8 days in 10)"))
        fig.add_trace(go.Scatter(
            x=d, y=rows["p50"] / 1e6, mode="lines+markers",
            line=dict(color=COLORS["info"], width=3),
            marker=dict(size=9), name="Central estimate"))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Million lari", xaxis_title=None,
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Table in millions
    with right:
        tbl = pd.DataFrame({
            "Date": rows["target_date"].dt.strftime("%a %d %b"),
            "Low": rows["p10"].map(lambda v: m(v)),
            "Central": rows["p50"].map(lambda v: m(v)),
            "High": rows["p90"].map(lambda v: m(v)),
        })
        st.caption(UNIT_LABEL.capitalize())
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    # Gate badges with plain-language reasons
    gates = rec["dev_credentials"]["gates"]
    st.markdown("**Checks** — tested on 2024")
    gcols = st.columns(len(gates))
    for col, (key, g) in zip(gcols, gates.items()):
        with col:
            icon = "✅" if g.get("passed") else "❌"
            st.markdown(f"{icon} **{g.get('name', key)}**")
            st.caption(g.get("reason_plain", ""))

    if sec:
        with st.expander("In plain language", expanded=not publishable):
            for p in sec["paragraphs"]:
                st.markdown(p)

    # The honest "not the best" disclosure, where it applies.
    nb = rec["dev_credentials"].get("not_the_dev_best")
    if nb:
        with st.expander("⚠️ This is not the single best 2024 result — why it was chosen"):
            st.markdown(
                f"A different model did better on 2024: **{nb['better_option']}**, with a "
                f"typical error of {m(nb['its_dev_mae'])} million lari versus "
                f"{m(nb['this_dev_mae'])} million for the model shown "
                f"({nb['gap_pct']:.1f}% apart).\n\n{nb['why_promoted_anyway']}"
            )

    st.divider()

# ── Provenance footer ─────────────────────────────────────────────────
st.markdown(section_header("Provenance", "Everything needed to reproduce this page"),
            unsafe_allow_html=True)
if data["narrative"]["narrative"].get("provenance_line"):
    st.caption(data["narrative"]["narrative"]["provenance_line"])

pcols = st.columns(4)
d = prov.get("data", {})
c = prov.get("code", {})
with pcols[0]:
    st.caption("Data fingerprint")
    st.code(str(d.get("sha256", "—"))[:24] + "…", language=None)
with pcols[1]:
    st.caption("Code version")
    st.code(str(c.get("git_sha", "—"))[:12] + ("  (modified)" if c.get("git_dirty") else ""),
            language=None)
with pcols[2]:
    st.caption("Fiscal calendar version")
    st.code(prov.get("calendar_version", "—"), language=None)
with pcols[3]:
    st.caption("2025 holdout used?")
    st.code("No — sealed" if prov.get("test_window_touched") is False else "CHECK",
            language=None)

with st.expander("Limitations, in plain language"):
    for l in data["narrative"]["narrative"]["limitations"]:
        st.markdown(f"- {l}")

with st.expander("Full provenance record (JSON)"):
    st.json(prov)

st.caption(
    f"Generated {prov.get('generated_at_utc', '')[:19]} UTC · artifacts in `{data['dir']}` · "
    f"regenerate with `{GEN_CMD}`"
)
