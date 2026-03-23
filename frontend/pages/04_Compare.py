# pages/04_Compare.py — Cross-Run Comparison
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils_frontend import list_runs, load_run_outputs, RUNS_ROOT

try:
    from ui_styles import inject_global_css, page_header
except ImportError:
    def inject_global_css(): pass
    def page_header(t, s=""): return f"<h1>{t}</h1><p>{s}</p>"

st.set_page_config(page_title="Compare Runs", layout="wide")
inject_global_css()
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
tab_overlay, tab_metrics, tab_winner, tab_intervals = st.tabs(
    ["Overlay Chart", "Metric Comparison", "Winner Summary", "Interval Comparison"]
)

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
