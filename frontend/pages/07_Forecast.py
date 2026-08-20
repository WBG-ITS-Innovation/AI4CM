# pages/07_Forecast.py — Forward forecast: the next five working days.
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
from model_shelf import shelf_for as _shelf_for  # noqa: E402
from ui_styles import help_text  # tooltips, translated at render time
from ui_styles import COLORS, inject_global_css, page_header, section_header  # noqa: E402
from ui_styles import TOK as _TOK  # noqa: E402
from ui_styles import HELP, reading_this_chart  # noqa: E402

from ui_styles import inject_design_system, plotly_chrome  # presentation only
from ui_styles import glossary_note  # plain-language definitions, on demand
from ui_styles import definition  # the one wording of a term, for use inline
from i18n import install as install_language  # language toggle + pending-review note
# Imported as `_translate`, deliberately not as `_t`. It was `_t`, and three lines in the
# "Generate a forecast" block below used `_t` for a Treasury line name. Those lines sit in a
# module-level `if`, so they rebound the module global and the translator became a string:
# the page then died on `TypeError: 'str' object is not callable` the moment it reached the
# verdict history. `test_no_i18n_shadowing` fails if any page rebinds its i18n import again.
from i18n import t as _translate  # verdict sentences are fixed copy and are translated
from ui_styles import page_intro  # the one-or-two-sentence intro every page opens with
from ui_styles import render_app_header  # presentation only
from ui_styles import render_brand  # the one brand header, in the sidebar
st.set_page_config(page_title="Forecast · Treasury Forecast", layout="wide")
inject_global_css()

inject_design_system()

# The language toggle and, in Georgian, the standing note that the translation has
# not been reviewed by a native speaker. One call per page; everything else the
# reader sees is translated inside the shared helpers.
render_brand()
install_language()

render_app_header("Forward forecast", "The next working days, which are dates beyond the end of the data")
page_intro(
    "This page holds the forecast itself: what each Treasury line is expected to do over "
    "the next few working days, which model produced each figure, and what earned that "
    "model its place."
)
glossary_note("champion", "exploratory", "holdout", "withheld", "sealed window",
              "P10", "P50", "P90", "baseline", "skill")
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

# ══════════════════════════════════════════════════════════════════════════════
# ONE INTERPRETER OWNS THE MODELS
#
# The modelling stack (sklearn, lightgbm, xgboost, catboost, matplotlib) lives in the BACKEND
# interpreter. Importing the pipeline from here crashed on matplotlib and would then have crashed
# on sklearn in turn -- so this page dispatches to that interpreter and reads JSON, the same
# pattern the Lab page uses.
#
# This block used to sit further down, next to the "Generate a forecast" section that was its
# only caller. The per-target Models panel needs it too -- to ask which estimators this build can
# actually fit, and to run a comparison -- so it moves above the first use rather than being
# duplicated.
# ══════════════════════════════════════════════════════════════════════════════
import json as _json
import subprocess as _sp

_DATA = REPOROOT / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
_BACKEND_PY = next((p for p in (REPOROOT / "backend" / ".venv" / "bin" / "python",
                                REPOROOT / "backend" / ".venv" / "Scripts" / "python.exe")
                    if p.exists()), None)
_modes_ok = _BACKEND_PY is not None

VALIDATED_HORIZON = 5
EXPLORATORY_LABEL = "exploratory, not gated and not published"

#: Verdict codes in the words a reader uses. The codes themselves are terms of art.
_VERDICT_WORDS_EN = {
    "publishable": "usable as a forecast",
    "withheld_as_forecast": "shown as a guide only",
    "withheld": "not usable",
    "unknown": "not decided",
}


def _verdict_words(code: str) -> str:
    """A verdict in the reader's own words, in the reader's own language."""
    return _translate(_VERDICT_WORDS_EN.get(code, code))


class _VerdictWords(dict):
    """Kept as a mapping so the two call sites read unchanged, translated on lookup."""

    def get(self, code, default=None):
        return _verdict_words(code) if code in _VERDICT_WORDS_EN else default


_VERDICT_WORDS = _VerdictWords()


def _dispatch(args: list, timeout: int = 600) -> dict:
    """Run backend/forecast_modes.py and return its JSON, or an explained failure."""
    if not _modes_ok:
        return {"ok": False, "refused": False,
                "reason": "the backend interpreter (backend/.venv) was not found"}
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
def _model_pool() -> list:
    """Estimator names this build can actually fit, read live from the backend interpreter."""
    if not _modes_ok:
        return []
    out = _sp.run([str(_BACKEND_PY), "-c",
                   "import sys;sys.path.insert(0,'backend');"
                   "from b_ml_pipeline import available_models;"
                   "print('\\n'.join(sorted(available_models())))"],
                  cwd=str(REPOROOT), capture_output=True, text=True, timeout=120)
    return [l for l in out.stdout.split("\n") if l.strip()] if out.returncode == 0 else []


@st.cache_data(show_spinner=False, ttl=300)
def _shelf_state(target: str) -> dict:
    """The champion and its runners-up for one target, read from the recorded evidence.

    Cached because it re-reads the whole experiment ledger and recomputes the Treasury
    method's own 2024 error, and neither changes while a reader is on the page.
    """
    return _shelf_for(target, data_path=_DATA, runnable_models=set(_model_pool()))


def _render_compare_alternatives(target: str, shelf: dict) -> None:
    """Run the champion and its runners-up side by side, exploratorily.

    Every run here goes through ``forecast_modes.exploratory_run``, which returns a type with
    no publish path at all. That is the reason this button is safe to offer: the separation is
    enforced in the backend, so forgetting a flag on this page cannot leak a comparison into
    the published record, the official artifacts or the scorecard.
    """
    comparable = shelf.get("comparable") or []
    if not comparable:
        return

    with st.expander(f"Compare alternatives for {target}", expanded=False):
        st.warning(
            "**EXPLORATORY.** Anything produced here is a side-by-side experiment. It is not "
            "published, not written to the official forecast, and not entered in the "
            "scorecard. The official forecast always uses the champion."
        )
        st.caption(
            "Each model is refitted on all data through the end of the file and asked for the "
            "same working days as the official forecast. Comparing them here shows how much "
            "the choice of model actually moves the number."
        )
        if st.button(f"Run the comparison for {target}",
                     key=f"cmp_{target}", disabled=not _modes_ok):
            _names = [shelf["champion_model"]] + list(comparable)
            _frames, _failed = [], []
            for _name in _names:
                with st.spinner(f"Running {_name} on {target} …"):
                    _res = _dispatch(["--mode", "exploratory", "--target", target,
                                      "--model", _name, "--horizon", str(VALIDATED_HORIZON),
                                      "--data", str(_DATA)])
                if not _res.get("ok"):
                    _failed.append((_name, _res.get("reason", "no reason reported")))
                    continue
                _df = pd.DataFrame(_res["forecasts"])
                _df["target_date"] = pd.to_datetime(_df["target_date"])
                _frames.append((_name, _df))

            for _name, _why in _failed:
                st.error(f"**{_name} did not produce a comparison.** {_why}")

            if _frames:
                _out = pd.DataFrame({
                    "Date": _frames[0][1]["target_date"].dt.strftime("%a %d %b")})
                for _name, _df in _frames:
                    _label = (f"{_name} (champion)" if _name == shelf["champion_model"]
                              else _name)
                    _out[_label] = _df["p50"].map(m).to_list()
                st.dataframe(_out, hide_index=True, use_container_width=True)
                st.caption(
                    f"{UNIT_LABEL.capitalize()}. Central estimates only. These are exploratory "
                    "results and no gate verdict attaches to any of them, including the "
                    "champion's column, because the gates were measured on 2024 and these "
                    "dates have no actual value yet."
                )
                st.warning(
                    "**EXPLORATORY.** Nothing above was published or scored. The official "
                    "forecast for this line uses the champion."
                )



# ══════════════════════════════════════════════════════════════════════════════
# TWO TABS
#
# The page was one column roughly eight hundred lines long, and "generate a forecast" sat two
# thirds of the way down it, below every published figure and every piece of evidence. The one
# thing somebody comes here to DO was the last thing they could reach.
#
# So: running a forecast is the first tab and the first thing on it. Everything that explains
# and evidences the published forecasts is the second.
# ══════════════════════════════════════════════════════════════════════════════
_tab_run, _tab_read = st.tabs([_translate("Forecast"),
                               _translate("How to read this page")])

with _tab_run:
    # ── What the two kinds are, before the button that makes one ──────────────────
    #
    # Above the run controls rather than behind an expander, because choosing the Mode IS
    # choosing which kind, and a reader who does not know the difference cannot choose. Kept to
    # a table rather than a column of paragraphs so the run controls stay on the first screen.
    st.markdown(section_header(
        _translate("Two kinds of forecast"),
        _translate("Which one you are making is the most useful thing to know here")),
        unsafe_allow_html=True)

    st.markdown(_translate(
        "This page can produce either kind. The Mode switch below is where you choose. The Lab "
        "page only ever produces the exploratory kind."
    ))

    st.markdown(
        f"| | **{_translate('Official')}** | **{_translate('Exploratory')}** |\n"
        "|---|---|---|\n"
        f"| {_translate('What it is')} | {_translate('The forecast for a Treasury line')} "
        f"| {_translate('Any model on any line, to see what it does')} |\n"
        f"| {_translate('Who picks the model')} "
        f"| {_translate('Nobody here. Its evidence did, once')} "
        f"| {_translate('You do, from a list')} |\n"
        f"| {_translate('Checks measured')} | {_translate('All of them, each with a reason')} "
        f"| {_translate('None')} |\n"
        f"| {_translate('Can be published')} | {_translate('Yes')} "
        f"| {_translate('No, and there is no setting for it')} |\n"
        f"| {_translate('Scored against reality')} | {_translate('Yes')} "
        f"| {_translate('No')} |\n"
    )

    st.caption(f"**{_translate('What published means.')}** " + _translate(
        "The forecast is written to a folder named for the day it was issued, and never edited "
        "afterwards. That is what lets us score it later against what actually happened."
    ))

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

    if not _modes_ok:
        st.warning(
            "**Forecast generation needs the backend interpreter** (`backend/.venv`), which was not "
            "found. The models and their libraries live there, not in the interpreter running this "
            "page. Published forecasts above are unaffected, because they are read from artifacts.")


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
                  "immutably. The model is not selectable, because the champion was chosen on "
                  "recorded evidence.  Exploratory: any model, any target, any horizon, shown "
                  "but never published and never scored."))

        _tr = _targets_and_recipes()
        _all_targets, _reg = _tr["targets"], _tr["recipes"]

        if _mode == "Official":
            _sel = st.multiselect("Target(s)", _all_targets,
                                  default=[t for t in _all_targets if t in _reg][:1])
            st.caption(f"Horizon is fixed at {VALIDATED_HORIZON} business days, the only horizon at "
                       f"which the benchmark, recipe selection and gates were measured.")
            _runnable = [t for t in _sel if t in _reg]
            for _tgt in [t for t in _sel if t not in _reg]:
                st.error(f"**{_tgt} has no champion recipe, so no official forecast can be issued for "
                         f"it.** Substituting another target's recipe would attach five folds of "
                         f"evidence to a model it was never measured on. Use exploratory mode, where "
                         f"nothing is published and no gate is claimed.")
            if _runnable:
                st.caption("Will run: " + ", ".join(
                    f"**{t}** uses `{_reg[t]['recipe_id']}` ({_reg[t]['model']})" for t in _runnable))
            _pub = st.checkbox("Publish to forecasts/published/ under a new issue date", value=False)
            st.caption(
                "The model is not selectable in this mode, and that is deliberate. An official "
                "forecast is the champion recipe, which was chosen once on recorded evidence. "
                "To try a different model, use the comparison in each target's Models panel "
                "above, or Exploratory mode here."
            )
            if st.button("Run the champion recipe", disabled=not _runnable, type="primary"):
                for _tgt in _runnable:
                    with st.spinner(f"Running {_tgt} …"):
                        _args = ["--mode", "official", "--target", _tgt, "--data", str(_DATA)]
                        if _pub:
                            _args.append("--publish")
                        _r = _dispatch(_args)
                    if not _r.get("ok"):
                        st.error(f"**{_tgt}**: {_r.get('reason', 'unknown failure')}")
                        continue
                    _appr = _r.get("approved_by")
                    st.success(f"**{_tgt}** · recipe `{_r['recipe_id']}` · model `{_r['model']}` · "
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
                        f"2024 credentials run and are never recomputed on forward dates, which "
                        f"have no truth yet. Nothing here is approved: every recipe's status is "
                        f"*candidate*.")
        else:
            _tgt = st.selectbox("Target", _all_targets, index=0 if _all_targets else None)

            _pool = _model_pool()
            if not _pool:
                st.warning("**The model pool could not be read** from the backend interpreter. "
                           "Published forecasts above are unaffected, because they come from "
                           "stored artifacts.")
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
            if st.button("Run (exploratory)", disabled=not (_tgt and _mdl)):
                with st.spinner("Running …"):
                    _r = _dispatch(["--mode", "exploratory", "--target", _tgt, "--model", _mdl,
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
                    st.caption(f"{UNIT_LABEL.capitalize()}. This is exploratory. It is not written to "
                               f"`forecasts/published/` and cannot be exported as official.")

    st.divider()

with _tab_read:
    # ── Orientation, before the numbers ──────────────────────────────────────────
    st.markdown(section_header(
        _translate("How to read this page"),
        _translate("Three things worth knowing before you read the figures below")),
        unsafe_allow_html=True)

    st.markdown(f"**1. {_translate('Some lines are withheld.')}** " + definition("withheld"))
    st.markdown(f"**2. {_translate('The sealed window is not read here.')}** "
                + definition("sealed window"))
    st.markdown(f"**3. {_translate('A published forecast is never edited.')}** " + _translate(
        "It is written once, in a folder named for the day it was issued. Where a check has "
        "changed since, this page shows both what the issue said then and what it would say now, "
        "and it changes neither."
    ))

    st.divider()


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
                  help="How far ahead the forecast runs, counted in Georgian working days. "
                       "Weekends and public holidays are skipped.")
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

        # Verdict banner — never hidden. Each verdict now opens with one sentence saying what
        # the verdict IS, because "withheld as a forecast" and "withheld" are terms of art here
        # and a reader meeting them for the first time cannot tell them apart.
        if publishable:
            st.success(
                _translate("**Usable as a forecast.** Every check this model was put through passed, so "
                   "the numbers below are the best estimate we have for these days.")
                + " " + pub["reason_plain"]
            )
        elif pub["verdict"] == "withheld":
            # P2 made these two mean different things, and the page used to render both as
            # "shown as a guide to the typical level". That is wrong for `withheld`: it means a
            # documented trivial benchmark is MORE accurate, so the numbers are not a guide to
            # anything and presenting them as one would invite a worse decision.
            st.error(
                _translate("**Do not use these numbers.** A simple rule of thumb was more accurate than "
                   "this model on data it had never seen, so acting on its figures would be "
                   "worse than acting on the rule of thumb.")
                + "\n\n" + pub["reason_plain"]
            )
        else:
            st.error(
                _translate("**Shown as a guide to the typical level, not as a forecast.** The model is "
                   "about as accurate as we would want, but it could not show that it "
                   "anticipates individual days rather than tracking the usual level, so we "
                   "will not call it a forecast.")
                + "\n\n" + pub["reason_plain"]
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
                          help=help_text("ruler") + "  " + str(_bm.get("ruler_note", "")))
            with _h3:
                st.metric("Model is better by",
                          pct_points(_bm["skill_pct"]) if _bm.get("skill_pct") is not None
                          else NOT_REPORTED,
                          help=help_text("skill"))
            st.caption(f"Both figures come from run `{_bm['run_id']}` on {_bm.get('window')} "
                       f"(n={_bm.get('n')}). They describe how this recipe performed on 2024. They do "
                       f"not describe the forecast below, which has no actual values to be scored "
                       f"against yet.")
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
                "prediction to compare against rather than an error."), unsafe_allow_html=True)

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
        st.markdown("**Checks**, tested on 2024")
        gcols = st.columns(len(gates))
        for col, (key, g) in zip(gcols, gates.items()):
            with col:
                verdict = _translate("passed") if g.get("passed") else _translate("failed")
                st.markdown(f"**{g.get('name', key)}**: {verdict}")
                st.caption(g.get("reason_plain", ""))

        # ══════════════════════════════════════════════════════════════════════
        # MODELS PANEL — the champion, and what came second
        #
        # Before this the page showed one model per target and nothing else, which reads as
        # "here is our model" rather than "here is the best of what we measured". The runners-up
        # come from experiments/log.csv, judged by the same publication_gates code that decided
        # the champion's own verdict, so nothing here is a second opinion about the gates.
        #
        # Nothing in this panel selects anything. The champion is read from the registry, which
        # is hand-edited; displaying a ranking of past measurements does not change it.
        # ══════════════════════════════════════════════════════════════════════
        _shelf = _shelf_state(target)
        st.markdown("**Models measured on this line**")
        st.caption(
            "The champion is the model the official forecast uses. It was chosen once, on "
            "recorded evidence, and no run re-chooses it. The alternatives below are the next "
            "best models that also cleared the accuracy gate, shown so the choice can be "
            "checked rather than taken on trust."
        )

        _mc1, _mc2 = st.columns([1.15, 0.85], gap="large")
        with _mc1:
            st.markdown(f"**{_shelf['champion_model']}** is the champion")
            st.caption(_shelf["champion_sentence"])
        with _mc2:
            _ops = _shelf["ops"]
            if _ops.get("available") and _ops.get("skill_pct") is not None:
                st.metric("Better than the Treasury's current method by",
                          pct_points(_ops["skill_pct"]),
                          help="Both errors are averages over the same year, 2024. The current "
                               "method is the Treasury's published planning construction: a "
                               "three-year average annual total, split by month share and spread "
                               "over working days.")
                st.caption(f"Measured over {_ops['n']} working days in 2024.")
            else:
                st.metric("Better than the Treasury's current method by", NOT_REPORTED,
                          help="No comparison is possible here. The reason is stated below.")
                st.caption(f"Not comparable: {_ops.get('reason', 'reason not recorded')}.")

        _alts = _shelf["alternatives"]
        if not _alts:
            st.info(_shelf["no_alternatives_reason"])
        else:
            st.dataframe(pd.DataFrame([{
                "Model": a.model,
                "Better than the naive rule by": f"{(1.0 - a.mase) * 100:.1f}%",
                "Better than carrying forward by": (f"{a.skill_vs_ruler_pct:.1f}%"
                                                    if a.skill_vs_ruler_pct is not None
                                                    else NOT_REPORTED),
                "Verdict if published": _VERDICT_WORDS.get(a.verdict, a.verdict),
                "Can be re-run here": "yes" if a.runnable else "no",
            } for a in _alts]), hide_index=True, use_container_width=True)
            st.caption(
                "Every figure is read from the recorded run that produced it, on 2024 data the "
                "model was never fitted on. A model marked \"no\" in the last column was "
                "measured but cannot be re-run as a forward forecast in this build."
            )

        with st.expander("What does this mean?"):
            st.markdown(
                "- **Champion** means the one model the official forecast uses for this line. "
                "It was chosen on recorded evidence from 2024, and loading new data refits it "
                "but never re-chooses it.\n"
                "- **The naive rule** is repeating what happened on the same weekday last week. "
                "A model that cannot beat it has no business being published.\n"
                "- **Carrying forward** means assuming the value from five working days ago "
                "repeats. It is the single shared benchmark every model family here is scored "
                "against, which is what makes their numbers comparable.\n"
                "- **The Treasury's current method** is the planning construction in use today: "
                "a three-year average annual total, split by month share and spread evenly over "
                "working days. It is not defined for a balance level, which has no annual total.\n"
                "- An alternative appearing here has **not** replaced the champion and is not "
                "published. Comparing them below runs them exploratorily, and nothing that runs "
                "there is written to the official record."
            )

        _render_compare_alternatives(target, _shelf)

        if sec:
            with st.expander("In plain language", expanded=not publishable):
                for p in sec["paragraphs"]:
                    st.markdown(p)

        # The honest "not the best" disclosure, where it applies.
        nb = rec["dev_credentials"].get("not_the_dev_best")
        if nb:
            with st.expander("This is not the single best 2024 result, and here is why it was chosen"):
                st.markdown(
                    f"A different model did better on 2024: **{nb['better_option']}**, with a "
                    f"typical error of {m(nb['its_dev_mae'])} million lari versus "
                    f"{m(nb['this_dev_mae'])} million for the model shown "
                    f"({nb['gap_pct']:.1f}% apart).\n\n{nb['why_promoted_anyway']}"
                )

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
                f"**{len(_changed)} published verdict(s) would differ today.** A check was "
                f"corrected after those issues went out, and under the corrected check the "
                f"verdict on them would read differently.\n\n"
                f"**What this means for you.** The forecast figures in those issues have not "
                f"changed and never will: a published file is written once and left exactly as "
                f"issued. The corrected check applies from now on. Nothing here needs doing, "
                f"and the comparison below shows both readings side by side so an old issue "
                f"cannot be read as though it still said what it said then."
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
                    # `withheld_as_forecast` and its siblings are codes in the registry, not
                    # words. Printing them raw asked the reader to learn a vocabulary in order
                    # to read a verdict history.
                    _then_w = _VERDICT_WORDS.get(_then, _then)
                    _now_w = _VERDICT_WORDS.get(_now, _now)
                    if r.get("changed"):
                        st.markdown(
                            f"**{r['target']}** &nbsp; was {_then_w}, now {_now_w}",
                            unsafe_allow_html=True)
                        # The one actionable sentence: it names only the gates that drove the
                        # change, not every gate that differs.
                        st.caption(r["why"])
                    else:
                        st.markdown(f"**{r['target']}** &nbsp; {_then_w} &nbsp;(unchanged)",
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
            "A published forecast is scored only once its actual value arrives in the data. "
            "The scorer refuses to evaluate a date whose truth we do not yet hold, which is "
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
        st.code("No, still sealed" if prov.get("test_window_touched") is False else "CHECK",
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
