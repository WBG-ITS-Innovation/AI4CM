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

# The benchmark readers are pure pandas + json, so this interpreter can import them directly.
# One definition of the benchmark column then serves both the evaluator and this page.
from forward_forecast import BENCHMARK_LABEL as _BENCH_LABEL  # noqa: E402
from forward_forecast import benchmark_mae_for_target as _benchmark_mae  # noqa: E402
from forward_forecast import benchmark_series as _benchmark_series  # noqa: E402
from ui_styles import COLORS, inject_global_css, page_header, section_header  # noqa: E402
from ui_styles import TOK as _TOK  # noqa: E402
from ui_styles import HELP, reading_this_chart  # noqa: E402

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

        # Verdict-at-issue vs verdict-today for every published issue. A published issue is
        # immutable, so when the gates change the two legitimately differ -- and a reader of an
        # old issue would otherwise see only the verdict it was issued under.
        try:
            from published_forecasts import reconcile_verdicts
            reconciliation = reconcile_verdicts()
        except Exception:                       # noqa: BLE001 - never block the page on this
            reconciliation = []

        return {
            "reconciliation": reconciliation,
            "forecasts": pd.DataFrame(art["forecasts"]),
            "provenance": art["provenance"],
            "registry": reg,
            "narrative": narr,
            "dir": art["dir"],
        }
    except FileNotFoundError:
        return None


from format_gel import NOT_REPORTED, UNIT_LABEL, pct_points  # noqa: E402
from format_gel import gel_millions as m  # noqa: E402


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
    elif pub["verdict"] == "withheld":
        # P2 made these two mean different things, and the page used to render both as
        # "shown as a guide to the typical level". That is wrong for `withheld`: it means a
        # documented trivial benchmark is MORE accurate, so the numbers are not a guide to
        # anything and presenting them as one would invite a worse decision.
        st.error(
            f"**WITHHELD — do not use these numbers.**\n\n{pub['reason_plain']}"
        )
    else:
        st.error(
            f"**WITHHELD as a forecast — shown as a guide to the typical level.**\n\n"
            f"{pub['reason_plain']}"
        )
        if pub.get("named_fix"):
            st.warning(f"**What would change this:** {pub['named_fix']}")

    # ── the audited benchmark error beside the model's, so the comparison is visible ─────
    # Model MAE from the registry's DEV credentials; benchmark MAE from
    # experiments/runs/<run_id>.json (field `ruler`), which the registry labels
    # "h=5 business-day persistence, the single shared ruler across all four families".
    # Neither is recomputed here. Cross-checked once: (ruler - dev_mae)/ruler reproduces each
    # recorded skill to ~1e-5, the residual being the log storing skill to four decimals.
    _bm = _benchmark_mae(target)
    if _bm.get("available"):
        _h1, _h2, _h3 = st.columns(3)
        with _h1:
            st.metric(f"Model error on 2024 ({UNIT_LABEL})", m(_bm["model_mae"]),
                      help=("Average absolute error of this recipe on the 2024 window, read from "
                            "the audited run that earned the recipe its credentials."))
        with _h2:
            st.metric(f"Benchmark error ({UNIT_LABEL})", m(_bm["benchmark_mae"]),
                      help=HELP["ruler"] + "  " + str(_bm.get("ruler_note", "")))
        with _h3:
            st.metric("Model is better by",
                      pct_points(_bm["skill_pct"]) if _bm.get("skill_pct") is not None
                      else NOT_REPORTED,
                      help=HELP["skill"])
        st.caption(f"Both figures come from run `{_bm['run_id']}` on {_bm.get('window')} "
                   f"(n={_bm.get('n')}). They describe how this recipe performed on 2024 — not the "
                   f"forecast below, which has no actual values to be scored against yet.")
    else:
        st.caption(f"No audited benchmark error is recorded for {target}: "
                   f"{_bm.get('ruler_note', NOT_REPORTED)}.")

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

        # ── the persistence benchmark, READ from the artifact ───────────────────────
        # `origin_value` IS the h-step persistence prediction. Evaluation's
        # compute_persistence_baseline() documents its definition as y_hat(t+h) = y(t) and computes
        # it from this same column, so plotting it reads the evaluator's own field rather than
        # reimplementing the benchmark. Flat because all five days share one origin.
        _bench = _benchmark_series(rows)
        if _bench is not None:
            fig.add_trace(go.Scatter(
                x=d, y=_bench / 1e6, mode="lines+markers",
                line=dict(color=_TOK["stop_ink"], width=2, dash="dashdot"),
                marker=dict(size=7, symbol="x"),
                name=_BENCH_LABEL,
                hovertemplate=("<b>%{x|%a %d %b %Y}</b><br>"
                               "Benchmark (value carried forward): %{y:,.1f} M GEL<br>"
                               "<i>the yardstick every model here is scored against</i>"
                               "<extra></extra>")))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Million lari", xaxis_title=None,
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )
        plotly_chrome(fig)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displaylogo": False})
        st.markdown(reading_this_chart(
            "The solid line is the central estimate for each working day and the shaded band is "
            "the range we expect the actual figure to fall inside on eight days out of ten. A "
            "wider band means less certainty about that day, not a worse forecast.<br><br>"
            "The <b>dash-dot line with crosses is the benchmark</b>: assume the value from the "
            "origin date simply repeats. It is flat because all five days are forecast from the "
            "same origin. Every accuracy figure in this project is stated as an improvement over "
            "that line, which is what makes figures from different model families comparable. "
            "<b>These dates have no actual value yet</b>, so the benchmark here is a rival "
            "prediction to compare against — not an error."), unsafe_allow_html=True)

    # Table in millions
    with right:
        tbl = pd.DataFrame({
            "Date": rows["target_date"].dt.strftime("%a %d %b"),
            "Low": rows["p10"].map(lambda v: m(v)),
            "Central": rows["p50"].map(lambda v: m(v)),
            "High": rows["p90"].map(lambda v: m(v)),
            "Benchmark": (_benchmark_series(rows).map(lambda v: m(v))
                          if _benchmark_series(rows) is not None
                          else [NOT_REPORTED] * len(rows)),
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
# The modelling stack (sklearn, lightgbm, xgboost, catboost, matplotlib) lives in the BACKEND
# interpreter. Importing the pipeline from here crashed on matplotlib and would then have crashed
# on sklearn in turn -- so this page dispatches to that interpreter and reads JSON, the same
# pattern the Lab page and the model-pool lookup use. One interpreter owns the models.
import json as _json
import subprocess as _sp

_BACKEND_PY = next((p for p in (REPOROOT / "backend" / ".venv" / "bin" / "python",
                                REPOROOT / "backend" / ".venv" / "Scripts" / "python.exe")
                    if p.exists()), None)
_modes_ok = _BACKEND_PY is not None
if not _modes_ok:
    st.warning(
        "**Forecast generation needs the backend interpreter** (`backend/.venv`), which was not "
        "found. The models and their libraries live there, not in the interpreter running this "
        "page. Published forecasts above are unaffected — they are read from artifacts.")

VALIDATED_HORIZON = 5
EXPLORATORY_LABEL = "exploratory — not gated, not published"


def _dispatch(args: list, timeout: int = 600) -> dict:
    """Run backend/forecast_modes.py and return its JSON, or an explained failure."""
    try:
        out = _sp.run([str(_BACKEND_PY), "backend/forecast_modes.py", *args],
                      cwd=str(REPOROOT), capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "refused": False, "reason": f"could not start the backend: {exc}"}
    line = next((l for l in reversed(out.stdout.splitlines()) if l.strip().startswith("{")), "")
    if not line:
        return {"ok": False, "refused": False,
                "reason": (out.stderr.strip().splitlines() or ["no output from the backend"])[-1]}
    try:
        return _json.loads(line)
    except Exception as exc:
        return {"ok": False, "refused": False, "reason": f"unreadable backend output: {exc}"}


@st.cache_data(show_spinner=False, ttl=300)
def _targets_and_recipes() -> dict:
    # Targets come from the data file's header and recipes from the registry JSON. Neither needs
    # the modelling stack, so neither needs a dispatch.
    cols = [c for c in pd.read_csv(_DATA, nrows=1).columns
            if c not in {"date", "is_weekend", "is_holiday"}] if _DATA.exists() else []
    reg = {}
    _rp = REPOROOT / "registry" / "recipes.json"
    if _rp.exists():
        for rec in _json.loads(_rp.read_text())["recipes"]:
            reg[rec["target"]] = {"recipe_id": rec["id"], "model": rec["point_model"],
                                  "approved_by": rec["approved_by"]}
    return {"targets": cols, "recipes": reg}

if _modes_ok:
    _mode = st.radio(
        "Mode", ["Official", "Exploratory"], horizontal=True,
        help=("Official: the target's champion recipe, refitted on all data and published "
              "immutably. The model is not selectable — the champion was chosen on recorded "
              "evidence.  Exploratory: any model, any target, any horizon; shown but never "
              "published, never scored."))

    _tr = _targets_and_recipes()
    _all_targets, _reg = _tr["targets"], _tr["recipes"]

    if _mode == "Official":
        _sel = st.multiselect("Target(s)", _all_targets,
                              default=[t for t in _all_targets if t in _reg][:1])
        st.caption(f"Horizon is fixed at {VALIDATED_HORIZON} business days — the only horizon at "
                   f"which the benchmark, recipe selection and gates were measured.")
        _runnable = [t for t in _sel if t in _reg]
        for _t in [t for t in _sel if t not in _reg]:
            st.error(f"**{_t} has no champion recipe, so no official forecast can be issued for "
                     f"it.** Substituting another target's recipe would attach five folds of "
                     f"evidence to a model it was never measured on. Use exploratory mode, where "
                     f"nothing is published and no gate is claimed.")
        if _runnable:
            st.caption("Will run: " + ", ".join(
                f"**{t}** → `{_reg[t]['recipe_id']}` ({_reg[t]['model']})" for t in _runnable))
        _pub = st.checkbox("Publish to forecasts/published/ under a new issue date", value=False)
        if st.button("Run", disabled=not _runnable, type="primary"):
            for _t in _runnable:
                with st.spinner(f"Running {_t} …"):
                    _args = ["--mode", "official", "--target", _t, "--data", str(_DATA)]
                    if _pub:
                        _args.append("--publish")
                    _r = _dispatch(_args)
                if not _r.get("ok"):
                    st.error(f"**{_t}**: {_r.get('reason', 'unknown failure')}")
                    continue
                _appr = _r.get("approved_by")
                st.success(f"**{_t}** · recipe `{_r['recipe_id']}` · model `{_r['model']}` · "
                           f"approved by **{_appr if _appr else 'none'}**"
                           + (f" · published to `{_r['published_to']}`"
                              if _r.get("published_to") else ""))
                _f = pd.DataFrame(_r["forecasts"])
                _f["target_date"] = pd.to_datetime(_f["target_date"])
                st.dataframe(pd.DataFrame({
                    "Date": _f["target_date"].dt.strftime("%a %d %b"),
                    "Low": _f["p10"].map(m), "Central": _f["p50"].map(m),
                    "High": _f["p90"].map(m)}), hide_index=True, use_container_width=True)
                st.caption(
                    f"{UNIT_LABEL.capitalize()}. Gate verdicts are inherited by recipe_id from the "
                    f"2024 credentials run — never recomputed on forward dates, which have no "
                    f"truth. Nothing here is approved: every recipe's status is *candidate*.")
    else:
        _t = st.selectbox("Target", _all_targets, index=0 if _all_targets else None)

        @st.cache_data(show_spinner=False, ttl=300)
        def _model_pool() -> list:
            out = _sp.run([str(_BACKEND_PY), "-c",
                           "import sys;sys.path.insert(0,'backend');"
                           "from b_ml_pipeline import available_models;"
                           "print('\\n'.join(sorted(available_models())))"],
                          cwd=str(REPOROOT), capture_output=True, text=True, timeout=120)
            return [l for l in out.stdout.split("\n") if l.strip()] if out.returncode == 0 else []

        _pool = _model_pool()
        if not _pool:
            st.warning("**The model pool could not be read** from the backend interpreter. "
                       "Published forecasts above are unaffected — they come from artifacts.")
        _mdl = st.selectbox("Model", _pool,
                            help="Any model in the pool, including ones never ablated on this "
                                 "target. Read live from the backend, so it cannot go stale.")
        _h = st.slider("Horizon (business days)", 1, 10, VALIDATED_HORIZON)
        if _h != VALIDATED_HORIZON:
            st.warning(f"**Horizon {_h} is exploratory.** The benchmark, recipe selection and "
                       f"every gate were measured at {VALIDATED_HORIZON} business days. At "
                       f"horizon {_h} no recipe was selected and no gate was measured.")
        st.error(f"**{EXPLORATORY_LABEL}.** Nothing below is published, enters the track record, "
                 f"or carries a gate verdict.")
        if st.button("Run (exploratory)", disabled=not (_t and _mdl)):
            with st.spinner("Running …"):
                _r = _dispatch(["--mode", "exploratory", "--target", _t, "--model", _mdl,
                                "--horizon", str(_h), "--data", str(_DATA)])
            if not _r.get("ok"):
                st.error(_r.get("reason", "unknown failure"))
            else:
                st.markdown(_r["banner"])
                _f = pd.DataFrame(_r["forecasts"])
                _f["target_date"] = pd.to_datetime(_f["target_date"])
                st.dataframe(pd.DataFrame({
                    "Date": _f["target_date"].dt.strftime("%a %d %b"),
                    "Low": _f["p10"].map(m), "Central": _f["p50"].map(m),
                    "High": _f["p90"].map(m)}), hide_index=True, use_container_width=True)
                st.caption(f"{UNIT_LABEL.capitalize()}. Exploratory — not written to "
                           f"`forecasts/published/`, not exportable as official.")

st.divider()

# ── Verdict reconciliation: what an old issue said, and what it would say now ──
#
# A published issue is immutable: its gates.json records the verdict at issue time, and
# rewriting it would destroy the only record of what was actually said on that date. But the
# gates themselves changed in P2 -- MASE became binding, the signal threshold was calibrated
# 1.50 -> 1.15, and vs_ruler stopped deciding -- so verdicts moved underneath issues already
# published. A reader of the 2025-08-06 issue would otherwise see only the verdict it was
# issued under, with nothing saying it no longer holds.
#
# Both statements are true; they answer different questions. So show both, never overwrite.
st.markdown(section_header("Verdict history",
                          "What each published issue said then, and what it would say now"),
            unsafe_allow_html=True)

_rec = data.get("reconciliation") or []
if not _rec:
    st.caption("No published issue to reconcile yet.")
else:
    _changed = [r for r in _rec if r.get("changed")]
    if _changed:
        st.warning(
            f"**{len(_changed)} published verdict(s) would differ today.** The forecast numbers "
            f"in those issues have not changed — only the verdict attached to them, because the "
            f"publication gates were corrected. The published files are left exactly as issued."
        )
    else:
        st.success("Every published verdict still holds under the current gates.")

    for _issue in sorted({r["issue_date"] for r in _rec}, reverse=True):
        _rows = [r for r in _rec if r["issue_date"] == _issue]
        _n_changed = sum(1 for r in _rows if r.get("changed"))
        _label = (f"Issue {_issue} — {_n_changed} of {len(_rows)} verdict(s) changed"
                  if _n_changed else f"Issue {_issue} — unchanged")
        with st.expander(_label, expanded=bool(_n_changed)):
            for r in _rows:
                _then, _now = r["verdict_at_issue"], r["verdict_today"]
                if r.get("changed"):
                    st.markdown(
                        f"**{r['target']}** &nbsp; `{_then}` &nbsp;→&nbsp; `{_now}`",
                        unsafe_allow_html=True)
                    # The one actionable sentence: it names only the gates that drove the
                    # change, not every gate that differs.
                    st.caption(r["why"])
                else:
                    st.markdown(f"**{r['target']}** &nbsp; `{_then}` &nbsp;(unchanged)",
                                unsafe_allow_html=True)
            st.caption("The published issue is immutable and its gates.json correctly records "
                       "what was decided on the issue date. This is a comparison, not a "
                       "correction to it.")

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
