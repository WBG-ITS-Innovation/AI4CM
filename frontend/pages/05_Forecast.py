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

from ui_styles import inject_design_system, plotly_chrome  # presentation only
from ui_styles import render_app_header  # presentation only
st.set_page_config(page_title="Forecast · Treasury Forecast", page_icon="🔭", layout="wide")
inject_global_css()

inject_design_system()

render_app_header("Forward forecast", "The next working days — dates beyond the end of the data")
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
    st.metric("Budget lines covered", f"{len(recipes)} of 41",
              help="How many of the daily Treasury data's 41 budget lines this page covers. "
                   "It is not a view of the whole budget.")
with c2:
    st.metric("Called a forecast", f"{n_pub} of {len(recipes)}",
              help="How many of the covered lines produced a figure we are willing to call "
                   "a forecast. The remainder are shown as a guide to the typical level, "
                   "with the reason stated on each one.")
with c3:
    st.metric("Working days ahead", str(int(fc["horizon"].max())),
              help="How far ahead the forecast runs, counted in Georgian working days — "
                   "weekends and public holidays are skipped.")
with c4:
    st.metric("Data through", str(pd.to_datetime(
        prov.get("data", {}).get("latest_data_date", fc["origin_date"].max())).date()),
              help="The last date present in the source data. The forecast covers dates "
                   "after this, so no actual values exist for them yet.")

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
        plotly_chrome(fig)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displaylogo": False})

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



# ══════════════════════════════════════════════════════════════════════════════
# GENERATE A FORECAST  —  two clearly separated modes
#
# The separation is enforced in backend/forecast_modes.py, not here: an exploratory result is a
# different type with no publish path, and publish_official() refuses it. This page cannot leak
# one into the published record by forgetting a flag.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("Generate a forecast",
                           "Official runs use the registry champion; exploratory runs do not"),
            unsafe_allow_html=True)

_DATA = REPOROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
try:
    from forecast_modes import (EXPLORATORY_LABEL, VALIDATED_HORIZON, NoRecipe, NotOfficial,
                                exploratory_run, horizon_status, next_issue_date,
                                official_run, publish_official, recipe_status,
                                targets_available)
    _modes_ok = True
except Exception as _exc:                                    # pragma: no cover
    _modes_ok = False
    st.info(f"Forecast generation unavailable ({_exc}).")

if _modes_ok:
    _mode = st.radio(
        "Mode", ["Official", "Exploratory"], horizontal=True,
        help=("Official: the target's champion recipe, refitted on all data and published "
              "immutably. The model is not selectable — the champion was chosen on recorded "
              "evidence.  Exploratory: any model, any target, any horizon; shown but never "
              "published, never scored."))

    _all_targets = targets_available(_DATA) if _DATA.exists() else []

    if _mode == "Official":
        _sel = st.multiselect("Target(s)", _all_targets,
                              default=[t for t in _all_targets if t in recipes][:1])
        st.caption(f"Horizon is fixed at {VALIDATED_HORIZON} business days — the only horizon at "
                   f"which the benchmark, recipe selection and gates were measured.")
        _refusals = [t for t in _sel if not recipe_status(t)["has_recipe"]]
        for _t in _refusals:
            st.error(recipe_status(_t)["explanation"])
        _runnable = [t for t in _sel if t not in _refusals]
        if _runnable:
            st.caption("Will run: " + ", ".join(
                f"**{t}** → `{recipe_status(t)['recipe_id']}` ({recipe_status(t)['model']})"
                for t in _runnable))
        if st.button("Run and publish", disabled=not _runnable, type="primary"):
            _issue = next_issue_date()
            _ok, _fail = [], []
            for _t in _runnable:
                try:
                    _res = official_run(_t, _DATA)
                    _ok.append((_t, _res))
                except (NoRecipe, NotOfficial) as _e:
                    _fail.append((_t, str(_e)))
            for _t, _why in _fail:
                st.error(f"**{_t}** refused. {_why}")
            if _ok:
                st.success(
                    f"Ran {len(_ok)} official forecast(s). Issue date **{_issue}** — a new date, "
                    f"because published issues are immutable and are never overwritten.")
                for _t, _res in _ok:
                    _appr = list(_res.gates.values())[0].get("approved_by")
                    st.markdown(
                        f"- **{_t}** · recipe `{_res.recipe_id}` · model `{_res.model}` · "
                        f"approved by **{_appr if _appr else 'none'}**")
                st.caption(
                    "Publishing writes to `forecasts/published/<issue_date>/` with full "
                    "provenance. Gate verdicts are inherited by recipe_id from the DEV "
                    "credentials run — they are not recomputed on forward dates, which have no "
                    "truth. Nothing here is approved: every recipe's status is *candidate*.")
                st.info("Artifacts are written by `backend/run_forward_forecast.py`; this button "
                        "runs the same code path. Re-run that script to regenerate outside the "
                        "lab.")
    else:
        _t = st.selectbox("Target", _all_targets, index=0 if _all_targets else None)
        # The model pool lives behind the ML libraries, which are installed in the BACKEND
        # interpreter, not this one. Ask that interpreter rather than hard-coding a list here --
        # a hard-coded list silently goes stale the moment a model is added or a library removed.
        @st.cache_data(show_spinner=False, ttl=120)
        def _model_pool() -> list:
            import subprocess
            for _py in (REPOROOT / "backend" / ".venv" / "bin" / "python",
                        REPOROOT / "backend" / ".venv" / "Scripts" / "python.exe"):
                if not _py.exists():
                    continue
                try:
                    out = subprocess.run(
                        [str(_py), "-c",
                         "import sys;sys.path.insert(0,'backend');"
                         "from b_ml_pipeline import available_models;"
                         "print('\\n'.join(sorted(available_models())))"],
                        cwd=str(REPOROOT), capture_output=True, text=True, timeout=90)
                    if out.returncode == 0:
                        return [l for l in out.stdout.split("\n") if l.strip()]
                except Exception:
                    pass
            return []

        _pool = _model_pool()
        if not _pool:
            st.warning(
                "**The model pool could not be read.** It lives behind the modelling libraries, "
                "which are installed in the backend interpreter "
                "(`backend/.venv`), not the one running this page. Exploratory runs need that "
                "interpreter available. Nothing is wrong with the published forecasts above — "
                "they were produced by the backend and read from artifacts.")
        _m = st.selectbox("Model", _pool,
                          help="Any model in the pool, including ones never ablated on this "
                               "target. Read live from the backend interpreter, so it cannot "
                               "go stale.")
        _h = st.slider("Horizon (business days)", 1, 10, VALIDATED_HORIZON)
        _hs = horizon_status(_h)
        if not _hs["validated"]:
            st.warning(_hs["explanation"])
        st.error(f"**{EXPLORATORY_LABEL}.** Results below are not published, do not enter the "
                 f"track record, and carry no gate verdict.")
        if st.button("Run (exploratory)", disabled=not (_t and _m)):
            _r = exploratory_run(_t, _m, _DATA, horizon=_h)
            st.markdown(_r.banner)
            _f = _r.forecasts.copy()
            _f["target_date"] = pd.to_datetime(_f["target_date"])
            st.dataframe(pd.DataFrame({
                "Date": _f["target_date"].dt.strftime("%a %d %b"),
                "Low": _f["p10"].map(m), "Central": _f["p50"].map(m),
                "High": _f["p90"].map(m)}), hide_index=True, use_container_width=True)
            st.caption(f"{UNIT_LABEL.capitalize()}. Exploratory output — not written to "
                       f"`forecasts/published/`, not exportable as official.")

st.divider()
# ── Track record: how have PAST published forecasts actually done? ────────────
#
# This is the section that makes the accuracy claim auditable over time. It reads the
# scorecard, which only ever contains dates whose truth has arrived -- so it can grow
# without ever touching the sealed holdout.
st.markdown(section_header("Track record",
                          "How past published forecasts actually performed"),
            unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=30)
def load_track_record():
    from published_forecasts import list_published, score_published
    DATA = REPOROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
    try:
        out = score_published(DATA)
        return out, len(list_published())
    except Exception as exc:  # pragma: no cover - defensive on a demo machine
        return {"error": str(exc)}, 0


tr, n_issues = load_track_record()
if "error" in tr:
    st.info(f"No track record available yet ({tr['error']}).")
elif tr["scored"] == 0:
    st.info(
        f"**Nothing scoreable yet.** {n_issues} forecast issue(s) retained, "
        f"{tr['pending']} predicted days still in the future.\n\n"
        "A published forecast is scored only once its actual value arrives in the data — "
        "the scorer refuses to evaluate a date whose truth we do not yet hold. That is "
        "what keeps this an honest track record rather than a re-run of history."
        + (f"\n\nEarliest awaiting truth: **{tr['pending_dates'][0][1]}**."
           if tr.get("pending_dates") else "")
    )
else:
    st.caption(f"{tr['scored']} scored predictions across {n_issues} issue(s). "
               f"Millions of lari.")
    rows = []
    for target, s_ in tr["summary"].items():
        rows.append({
            "Target": target,
            "Days scored": s_["n"],
            "Realized error": m(s_["realized_mae"]),
            "Benchmark error": m(s_["persistence_mae"]),
            "Better by": f"{s_['skill_vs_ruler_pct']:.1f}%",
            "In range": f"{s_['interval_hit_rate']:.0%} (target {s_['nominal_coverage']:.0%})",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Every scored prediction"):
        sc = pd.read_csv(REPOROOT / "forecasts" / "scorecard.csv")
        st.dataframe(sc, hide_index=True, use_container_width=True)

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
