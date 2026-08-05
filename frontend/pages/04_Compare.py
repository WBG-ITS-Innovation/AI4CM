# pages/04_Compare.py — Cross-Run Comparison
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import streamlit as st
from ui_styles import (inject_design_system, gate_badge_tri, reading_this_chart,
                       empty_state, plotly_layout, ds_metric, HELP, COLORS)
from format_gel import NOT_REPORTED, UNIT_LABEL, gel_millions, pct_points, ratio


from utils_frontend import list_runs, load_run_outputs, RUNS_ROOT

try:
    from ui_styles import inject_global_css, page_header
except ImportError:
    def inject_global_css(): pass
    def page_header(t, s=""): return f"<h1>{t}</h1><p>{s}</p>"

st.set_page_config(page_title="Compare · Treasury Forecast", page_icon="⚖️", layout="wide")
inject_global_css()
inject_design_system()
st.markdown(
    page_header("📊 Compare Runs",
                "Select 2-6 runs to compare forecasts, metrics, and find the best model"),
    unsafe_allow_html=True,
)

# -------------------- helpers --------------------
@st.cache_data(show_spinner=False, ttl=15)
def _cached_runs() -> List[str]:
    return [r.name for r in list_runs() if (r / "outputs").exists()]


@st.cache_data(show_spinner=False, ttl=15)
def _cached_load(run_name: str) -> dict:
    return load_run_outputs(RUNS_ROOT / run_name)


# -------------------- run selector --------------------
available = _cached_runs()
if len(available) < 2:
    st.info("You need at least **2 completed runs** to compare. Use the **Lab** to create more runs.")
    st.stop()

selected = st.multiselect(
    "Select runs to compare (2-6)",
    available,
    default=available[:min(2, len(available))],
    max_selections=6,
    help="Pick 2-6 runs. They should share the same target and cadence for meaningful comparison.",
)

if len(selected) < 2:
    st.warning("Select at least **2** runs to compare.")
    st.stop()

# -------------------- load data --------------------
runs_data: Dict[str, dict] = {}
for name in selected:
    runs_data[name] = _cached_load(name)

# Check which runs have predictions
runs_with_preds = {k: v for k, v in runs_data.items() if v["pred"] is not None and not v["pred"].empty}
if len(runs_with_preds) < 2:
    st.error("Need at least 2 runs with predictions. Some selected runs have no outputs yet.")
    st.stop()

# -------------------- auto-detect shared target / horizon --------------------
all_targets = set()
all_horizons = set()
for name, data in runs_with_preds.items():
    pred = data["pred"]
    if "target" in pred.columns:
        all_targets.update(pred["target"].unique())
    if "horizon" in pred.columns:
        all_horizons.update(pred["horizon"].unique())

if not all_targets:
    all_targets = {"Series"}
if not all_horizons:
    all_horizons = {1}

c1, c2 = st.columns(2)
with c1:
    tgt = st.selectbox("Target to compare", sorted(all_targets))
with c2:
    hz = st.selectbox("Horizon to compare", sorted(all_horizons))

# -------------------- filter to shared target/horizon --------------------
filtered: Dict[str, pd.DataFrame] = {}
for name, data in runs_with_preds.items():
    pred = data["pred"].copy()
    if "target" in pred.columns:
        pred = pred[pred["target"] == tgt]
    if "horizon" in pred.columns:
        pred = pred[pred["horizon"] == hz]
    if not pred.empty:
        # Tag with run ID for easy identification
        pred["_run"] = name
        filtered[name] = pred

if len(filtered) < 2:
    st.warning(f"Fewer than 2 runs have predictions for target='{tgt}', horizon={hz}. "
               "Try a different combination.")
    st.stop()

st.success(f"Comparing **{len(filtered)} runs** for target='{tgt}', horizon={hz}")

# -------------------- tabs --------------------
tab_ruler, tab_overlay, tab_metrics, tab_winner, tab_intervals = st.tabs(
    ["Skill vs the shared ruler", "Overlay Chart", "Metric Comparison",
     "Winner Summary", "Interval Comparison"]
)

# ── Tab: cross-family skill on the ONE shared ruler ───────────────────────────
#
# The point of this tab is that every family is measured against the same h-step persistence
# benchmark, computed by one shared function. Without that, skill numbers from different
# families are not comparable and a bar chart of them is misleading. Two things are shown
# alongside each bar because neither means much alone:
#   * the sentinel reading -- a model can post large skill while its inputs carry no
#     information about the target, which is central tendency rather than forecasting;
#   * a note when a run's evaluation window or horizon unit differs from the others, since
#     that alone can move skill by tens of points.
with tab_ruler:
    st.subheader("Skill against the shared persistence benchmark")

    _rows = []
    for _name, _pred in filtered.items():
        _rd = RUNS_ROOT / _name
        _integ = {}
        for _cand in (_rd / "outputs" / "artifacts" / "integrity_report.json",
                      _rd / "artifacts" / "integrity_report.json"):
            if _cand.exists():
                try:
                    _integ = json.loads(_cand.read_text(encoding="utf-8"))
                except Exception:
                    _integ = {}
                break
        # The ruler and skill are READ from the run's own integrity report, never
        # recomputed here. The backend computes them with the single shared function; a
        # second implementation in the frontend could silently diverge, which is exactly
        # the failure mode that made an earlier tuning harness produce incomparable skill
        # figures. Where a run does not record them, they render as not reported.
        _best = _integ.get("best_model")
        _sub = _pred.copy()
        if _best and "model" in _sub.columns and (_sub["model"] == _best).any():
            _sub = _sub[_sub["model"] == _best]
        _td = (pd.to_datetime(_sub["target_date"], errors="coerce")
               if "target_date" in _sub.columns else None)
        _rows.append({
            "run": _name,
            "model": _best or NOT_REPORTED,
            "skill": _integ.get("skill_pct", np.nan),
            "ruler": _integ.get("mae_persistence", np.nan),
            "mae": _integ.get("mae_best", np.nan),
            "n": int(len(_sub)),
            "sentinel": _integ.get("shuffled_to_normal_ratio", np.nan),
            "family": _integ.get("pipeline", NOT_REPORTED),
            "win_from": _td.min().date() if _td is not None and _td.notna().any() else None,
            "win_to": _td.max().date() if _td is not None and _td.notna().any() else None,
            "horizon": _integ.get("horizon", np.nan),
        })
    _cmp = pd.DataFrame(_rows)

    if _cmp.empty or _cmp["skill"].isna().all():
        st.markdown(
            empty_state(
                "No run has the columns needed to compute skill on the shared ruler.",
                filename="artifacts/integrity_report.json (needs skill_pct and mae_persistence)",
                looked_in=str(RUNS_ROOT),
                command="Re-run any family; the runner writes the shared benchmark into its integrity report",
            ), unsafe_allow_html=True)
    else:
        _MIN_SENTINEL = 1.50
        _plot = _cmp.dropna(subset=["skill"]).sort_values("skill", ascending=False)
        _cols = [COLORS["trust"] if (pd.notna(sv) and sv >= _MIN_SENTINEL)
                 else COLORS["caution"] for sv in _plot["sentinel"]]
        _labels = [f"{s:.1f}%  (signal {('x%.2f' % sv) if pd.notna(sv) else 'not reported'})"
                   for s, sv in zip(_plot["skill"], _plot["sentinel"])]
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=_plot["skill"], y=_plot["run"], orientation="h", marker_color=_cols,
            text=_labels, textposition="outside", name="Skill",
            customdata=np.stack([
                _plot["sentinel"].fillna(-1.0), _plot["ruler"].fillna(0) / 1e6,
                _plot["n"], _plot["model"].astype(str)], axis=-1),
            hovertemplate=("<b>%{y}</b><br>Model: %{customdata[3]}<br>"
                           "Skill vs shared ruler: %{x:.2f}%<br>"
                           "Signal check: x%{customdata[0]:.2f}<br>"
                           "Ruler: %{customdata[1]:,.1f} M GEL<br>"
                           "Days scored: %{customdata[2]:,}<extra></extra>")))
        fig_r.add_vline(x=0, line_color=COLORS["neutral"])
        plotly_layout(fig_r, height=90 + 42 * len(_plot),
                      xtitle="Skill vs the shared persistence benchmark (%)",
                      legend_bottom=False)
        st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

        _weak = _plot[_plot["sentinel"].fillna(0) < _MIN_SENTINEL]
        st.markdown(reading_this_chart(
            "Every bar is measured against the <b>same</b> benchmark — predict that the value "
            "five working days ago repeats — computed by one shared function, so these numbers "
            "are comparable across model families. The figure in brackets is the signal "
            "self-test.<br><br>"
            f"<b>Amber bars fail the signal test</b> (below x{_MIN_SENTINEL:.2f}). For those "
            "runs the error really is smaller than the benchmark, but shuffling the historical "
            "answers barely hurt the model — so it is tracking a typical level rather than "
            "anticipating individual days. A large skill number next to a failing signal test "
            "is not a forecast; it is regression to the mean against a spiky benchmark. The "
            "two numbers only mean something together."), unsafe_allow_html=True)

        # ── differing window or horizon unit invalidates a direct comparison ──
        _notes = []
        _hs = set(int(h) for h in _cmp["horizon"].dropna().unique())
        if len(_hs) > 1:
            _notes.append(f"Runs use **different horizons** ({sorted(_hs)}). Skill at "
                          f"different horizons is not directly comparable — a shorter "
                          f"horizon is an easier problem.")
        _wins = {(r["win_from"], r["win_to"]) for _, r in _cmp.iterrows()
                 if r.get("win_from") is not None}
        if len(_wins) > 1:
            _notes.append("Runs cover **different evaluation windows**. The benchmark is "
                          "recomputed per window, so a run evaluated on a calmer period "
                          "faces an easier benchmark and its skill is not comparable.")
        _fams = set(str(f) for f in _cmp["family"].dropna().unique())
        if len(_fams) > 1:
            _notes.append(f"Families present: {', '.join(sorted(_fams))}. Skill is "
                          f"comparable across families *because* the benchmark is shared — "
                          f"that is the point of this view.")
        for _n in _notes:
            st.markdown(f"- {_n}")

        st.dataframe(pd.DataFrame({
            "Run": _cmp["run"],
            "Family": _cmp["family"],
            "Best model": _cmp["model"],
            "Skill vs ruler": [pct_points(v) for v in _cmp["skill"]],
            "Signal check": [ratio(v) for v in _cmp["sentinel"]],
            f"Ruler ({UNIT_LABEL})": [gel_millions(v) for v in _cmp["ruler"]],
            "Days scored": [f"{int(v):,}" if pd.notna(v) else NOT_REPORTED
                            for v in _cmp["n"]],
        }), hide_index=True, use_container_width=True)

# -------------------- Tab 1: Overlay Chart --------------------
with tab_overlay:
    st.subheader("Forecast Overlay Across Runs")
    st.caption("Each line shows one run's best model forecast versus Actuals.")

    fig = go.Figure()

    # Plot actuals from first run (they should be the same across runs)
    first_key = list(filtered.keys())[0]
    base = filtered[first_key].dropna(subset=["y_true"]).sort_values("date")
    fig.add_scatter(
        x=base["date"], y=base["y_true"],
        name="Actual", mode="lines",
        line=dict(color="black", width=2.5),
    )

    # Color palette
    colors = px.colors.qualitative.Plotly
    for i, (run_name, df) in enumerate(filtered.items()):
        df = df.dropna(subset=["y_pred"]).sort_values("date")
        if df.empty:
            continue
        # Use best model (lowest MAE) from this run
        if "model" in df.columns and df["model"].nunique() > 1:
            model_mae = df.groupby("model").apply(
                lambda g: (g["y_true"] - g["y_pred"]).abs().mean()
            )
            best_model = model_mae.idxmin()
            df = df[df["model"] == best_model]
            label = f"{run_name} ({best_model})"
        else:
            label = run_name

        fig.add_scatter(
            x=df["date"], y=df["y_pred"],
            name=label, mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5),
        )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Date", yaxis_title=tgt,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# -------------------- Tab 2: Metric Comparison --------------------
with tab_metrics:
    st.subheader("Metric Comparison Table")
    st.caption("Side-by-side metrics for each run's best model.")

    metric_rows = []
    for run_name, data in runs_with_preds.items():
        metr = data.get("metr")
        if metr is None or metr.empty:
            continue
        m = metr.copy()
        if "target" in m.columns:
            m = m[m["target"] == tgt]
        if "horizon" in m.columns:
            m = m[m["horizon"] == hz]
        if m.empty:
            continue

        # Get best model by MAE
        if "model" in m.columns:
            best_row = m.loc[m["MAE"].idxmin()]
        else:
            best_row = m.iloc[0]

        row = {"Run": run_name, "Best Model": best_row.get("model", "N/A")}
        for metric in ["MAE", "RMSE", "sMAPE", "R2", "Monthly_TOL10_Accuracy",
                        "PI_coverage@90", "PI_width@90"]:
            if metric in best_row.index:
                val = best_row[metric]
                row[metric] = round(float(val), 4) if not pd.isna(val) else None
        metric_rows.append(row)

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        # Highlight best values
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        # Bar chart comparison
        avail_metrics = [c for c in ["MAE", "RMSE", "sMAPE", "R2"] if c in mdf.columns and mdf[c].notna().any()]
        if avail_metrics:
            metric_choice = st.selectbox("Chart metric", avail_metrics, index=0)
            ascending = metric_choice != "R2"
            chart_df = mdf[["Run", "Best Model", metric_choice]].dropna()
            chart_df = chart_df.sort_values(metric_choice, ascending=ascending)
            chart_df["label"] = chart_df["Run"].str[:30] + " (" + chart_df["Best Model"].str[:15] + ")"

            fig_m = px.bar(
                chart_df, x="label", y=metric_choice,
                title=f"{metric_choice} by Run (lower is better)" if ascending else f"{metric_choice} by Run (higher is better)",
                height=380, color="label",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_m.update_layout(showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No metrics available for the selected target/horizon combination.")

# -------------------- Tab 3: Winner Summary --------------------
with tab_winner:
    st.subheader("Winner Summary")
    st.caption("Which run produced the best model for this target and horizon?")

    # Compute per-run best MAE
    winner_rows = []
    for run_name, df in filtered.items():
        df_clean = df.dropna(subset=["y_true", "y_pred"])
        if df_clean.empty:
            continue

        if "model" in df_clean.columns:
            for model_name, g in df_clean.groupby("model"):
                mae = float((g["y_true"] - g["y_pred"]).abs().mean())
                rmse = float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean()))
                n = len(g)
                winner_rows.append({
                    "Run": run_name, "Model": model_name,
                    "MAE": mae, "RMSE": rmse, "N predictions": n,
                })
        else:
            mae = float((df_clean["y_true"] - df_clean["y_pred"]).abs().mean())
            rmse = float(np.sqrt(((df_clean["y_true"] - df_clean["y_pred"]) ** 2).mean()))
            winner_rows.append({
                "Run": run_name, "Model": "N/A",
                "MAE": mae, "RMSE": rmse, "N predictions": len(df_clean),
            })

    if winner_rows:
        wdf = pd.DataFrame(winner_rows).sort_values("MAE")

        # Podium
        st.markdown("### Podium")
        podium_cols = st.columns(min(3, len(wdf)))
        medals = ["1st", "2nd", "3rd"]
        for i, col in enumerate(podium_cols):
            if i < len(wdf):
                row = wdf.iloc[i]
                with col:
                    st.metric(
                        f"{medals[i]}: {row['Model']}",
                        f"MAE = {row['MAE']:,.0f}",
                        delta=f"RMSE = {row['RMSE']:,.0f}",
                        delta_color="off",
                    )
                    st.caption(f"Run: {row['Run'][:40]}")

        st.markdown("### Full Ranking")
        wdf_display = wdf.copy()
        wdf_display["Rank"] = range(1, len(wdf_display) + 1)
        wdf_display["MAE"] = wdf_display["MAE"].apply(lambda x: f"{x:,.2f}")
        wdf_display["RMSE"] = wdf_display["RMSE"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(
            wdf_display[["Rank", "Run", "Model", "MAE", "RMSE", "N predictions"]],
            use_container_width=True, hide_index=True,
        )

        # Recommendation
        best = wdf.iloc[0]
        second = wdf.iloc[1] if len(wdf) > 1 else None
        st.markdown("### Recommendation")
        improvement = ""
        if second is not None:
            pct = (1 - best["MAE"] / second["MAE"]) * 100 if second["MAE"] > 0 else 0
            improvement = f" ({pct:.1f}% better than runner-up)"
        st.success(
            f"**Best model:** {best['Model']} from run `{best['Run']}`{improvement}. "
            f"MAE = {best['MAE']:,.0f}, RMSE = {best['RMSE']:,.0f}."
        )
    else:
        st.info("No predictions available for comparison.")

# -------------------- Tab 4: Interval Comparison --------------------
with tab_intervals:
    st.subheader("Prediction Interval Comparison")
    st.caption("Compare PI coverage and width across runs (where available).")

    pi_rows = []
    for run_name, df in filtered.items():
        if {"y_lo", "y_hi"}.issubset(df.columns):
            dfpi = df.dropna(subset=["y_lo", "y_hi", "y_true"])
            if dfpi.empty:
                continue

            if "model" in dfpi.columns:
                for model, g in dfpi.groupby("model"):
                    covered = ((g["y_true"] >= g["y_lo"]) & (g["y_true"] <= g["y_hi"])).mean()
                    width = (g["y_hi"] - g["y_lo"]).mean()
                    pi_rows.append({
                        "Run": run_name, "Model": model,
                        "Coverage": round(float(covered), 3),
                        "Avg Width": round(float(width), 1),
                        "N": len(g),
                    })
            else:
                covered = ((dfpi["y_true"] >= dfpi["y_lo"]) & (dfpi["y_true"] <= dfpi["y_hi"])).mean()
                width = (dfpi["y_hi"] - dfpi["y_lo"]).mean()
                pi_rows.append({
                    "Run": run_name, "Model": "N/A",
                    "Coverage": round(float(covered), 3),
                    "Avg Width": round(float(width), 1),
                    "N": len(dfpi),
                })

    if pi_rows:
        pi_df = pd.DataFrame(pi_rows)
        st.dataframe(pi_df, use_container_width=True, hide_index=True)

        # Coverage chart
        fig_cov = px.bar(
            pi_df, x="Model", y="Coverage", color="Run",
            barmode="group", title="PI Coverage by Run (target: 0.90)",
            height=380,
        )
        fig_cov.add_hline(y=0.90, line_dash="dash", line_color="red",
                          annotation_text="Target 90%")
        fig_cov.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_cov, use_container_width=True, config={"displaylogo": False})

        # Width chart
        fig_w = px.bar(
            pi_df, x="Model", y="Avg Width", color="Run",
            barmode="group", title="Average PI Width by Run (narrower = more precise)",
            height=380,
        )
        st.plotly_chart(fig_w, use_container_width=True, config={"displaylogo": False})
    else:
        st.info(
            "No prediction intervals found in the selected runs. "
            "PI columns (`y_lo`, `y_hi`) are produced by ML (conformal) and Statistical (native) pipelines."
        )
