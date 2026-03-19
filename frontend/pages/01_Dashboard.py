# pages/01_Dashboard.py — Dashboard (interactive overlays, diagnostics, downloads)
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import io, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from backend_consts import QUALITY_GATE_SKILL_PCT
from recommender import recommend_model, format_scorecard_markdown

APPROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = APPROOT / "runs"

st.set_page_config(page_title="📈 Dashboard", layout="wide")
st.title("📈 Dashboard — Interactive Results & Diagnostics")

# -------------------- Caching helpers --------------------
@st.cache_data(show_spinner=False, ttl=15)
def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

@st.cache_data(show_spinner=False, ttl=15)
def _list_runs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and (p/"outputs").exists()],
                  key=lambda p: p.stat().st_mtime, reverse=True)

# -------------------- File resolution --------------------
def _base_dirs(out_root: Path) -> List[Tuple[str, Path]]:
    dirs = [("root", out_root)]
    for cad in ("daily","weekly","monthly"):
        p = out_root / cad
        if (p / "predictions_long.csv").exists():
            dirs.append((cad, p))
    return dirs

def _load_outputs(base_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    pred = metr = lb = None
    p = base_dir / "predictions_long.csv"
    m = base_dir / "metrics_long.csv"
    l = base_dir / "leaderboard.csv"
    if p.exists():
        pred = _read_csv(p)
    if m.exists():
        metr = _read_csv(m)
    if l.exists():
        lb = _read_csv(l)
    return pred, metr, lb

# -------------------- Baseline helpers --------------------
def _weekday_mean_baseline(df: pd.DataFrame) -> Optional[pd.Series]:
    """Fallback baseline derived from Actuals for display only."""
    if "date" not in df.columns or "y_true" not in df.columns:
        return None
    dfx = df[["date","y_true"]].dropna().copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx = dfx.set_index("date")
    dfx["dow"] = dfx.index.dayofweek
    prof = dfx.groupby("dow")["y_true"].mean()
    out = dfx.index.to_series().apply(lambda d: prof.get(d.dayofweek, np.nan))
    out.index = dfx.index
    return out

def _read_ops_baseline(base_dir: Path, target: str) -> Optional[pd.Series]:
    # Prefer cadence folder baseline files if they exist
    daily_file = base_dir / f"{target}_ops_baseline_daily.csv"
    monthly_file = base_dir / f"{target}_ops_baseline_monthly.csv"
    if daily_file.exists():
        df = _read_csv(daily_file)
        if {"date","forecast"}.issubset(df.columns):
            return pd.Series(df["forecast"].values, index=pd.to_datetime(df["date"]))
    if monthly_file.exists():
        dfm = _read_csv(monthly_file)
        if "date" in dfm.columns and "forecast" in dfm.columns:
            dfm["date"] = pd.to_datetime(dfm["date"])
            dfm = dfm.dropna(subset=["date"]).set_index("date").sort_index()
            # distribute monthly evenly over business days (display only)
            parts = []
            for ts, v in dfm["forecast"].items():
                days = pd.date_range(ts.replace(day=1), ts, freq="B")
                if len(days):
                    parts.append(pd.Series(float(v)/len(days), index=days))
            if parts:
                s = pd.concat(parts).sort_index()
                return s
    return None

# -------------------- UI: choose run and source folder --------------------
runs = _list_runs()
if not runs:
    st.info("No runs yet. Use the **Lab** page to create one.")
    st.stop()

sel_run = st.selectbox("Select run", [r.name for r in runs], index=0)
run_dir = RUNS_DIR / sel_run
out_root = run_dir / "outputs"

cads = _base_dirs(out_root)
label_map = {"root": "outputs (root)", "daily": "outputs/daily", "weekly": "outputs/weekly", "monthly": "outputs/monthly"}
cad_choice_label = st.radio("Use outputs from", [label_map[c] for c, _ in cads], horizontal=True)
# Resolve choice back to actual dir name
choice_key = [k for k, lbl in label_map.items() if lbl == cad_choice_label][0]
base_dir = dict(cads)[choice_key]

pred, metr, lb = _load_outputs(base_dir)
if pred is None or pred.empty:
    st.warning(f"No **predictions_long.csv** in {base_dir}")
    st.stop()

# -------------------- Normalize predictions --------------------
pred = pred.copy()
pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
pred = pred.dropna(subset=["date"]).sort_values("date")
if "target" not in pred.columns: pred["target"] = "Series"
if "cadence" not in pred.columns:
    # infer from folder choice
    pred["cadence"] = choice_key if choice_key != "root" else "unknown"

targets = sorted(pred["target"].unique())
horizons = sorted(pred["horizon"].unique()) if "horizon" in pred.columns else [1]
models_all = sorted(pred["model"].unique()) if "model" in pred.columns else ["Model"]

# -------------------- Control panel --------------------
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    tgt = st.selectbox("Target", targets, index=0)
with c2:
    hz = st.multiselect("Horizons", horizons, default=horizons)
with c3:
    models_sel = st.multiselect("Models", models_all, default=models_all[:min(3, len(models_all))])
with c4:
    gran = st.selectbox("Display granularity", ["Native", "Weekly (Fri)", "Monthly (EOM)"], index=0,
                        help="Only affects the chart view; CSVs remain at native cadence.")

# -------------------- Pipeline trust badge --------------------
# ✅ FIX SYSTEM-1: Show trust level prominently so users always know
_integrity_path = base_dir / "artifacts" / "integrity_report.json"
if not _integrity_path.exists():
    _integrity_path = out_root / "artifacts" / "integrity_report.json"
if _integrity_path.exists():
    import json as _json_mod
    try:
        _integ = _json_mod.load(open(_integrity_path))
        _pipeline = _integ.get("pipeline", "ML")
        _status = _integ.get("run_status", "UNKNOWN")
        _skill = _integ.get("skill_pct", None)
        _qg = _integ.get("quality_gate_passed", None)
        if _status == "SUCCESS" and _qg:
            st.success(f"**{_pipeline} pipeline** — Quality gate PASSED (skill {_skill:.1f}%)")
        elif _status == "FAILED_QUALITY":
            st.warning(f"**{_pipeline} pipeline** — Quality gate FAILED "
                       f"(skill {_skill:.1f}% < {QUALITY_GATE_SKILL_PCT}%). "
                       f"Outputs are real but model does not beat persistence baseline.")
        elif _status == "ERROR":
            st.error(f"**{_pipeline} pipeline** — Integrity check error: {_integ.get('error','unknown')}")
        else:
            st.info(f"**{_pipeline} pipeline** — Status: {_status}")
    except Exception:
        pass
else:
    st.caption("No integrity report found for this run. Interpret outputs with caution.")

# -------------------- Accuracy Scorecard + Recommendations --------------------
_scorecard_cfg = {}
_scorecard_cfg_path = base_dir / "artifacts" / "config.json"
if not _scorecard_cfg_path.exists():
    _scorecard_cfg_path = out_root / "artifacts" / "config.json"
if _scorecard_cfg_path.exists():
    try:
        import json as _j2
        _scorecard_cfg = _j2.loads(_scorecard_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass

_rec = recommend_model(pred, metr, lb, _scorecard_cfg, target=targets[0] if targets else "", horizon=horizons[0] if horizons else 1)

with st.expander("Accuracy Scorecard & Recommendations", expanded=True):
    _grade = _rec["accuracy_grade"]
    _color = _rec["grade_color"]
    _best = _rec["best_model"]

    g1, g2, g3, g4 = st.columns([1, 1, 1, 1])
    with g1:
        _grade_emoji = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}.get(_grade, "⚪")
        st.metric("Accuracy Grade", f"{_grade_emoji} {_grade}")
    with g2:
        st.metric("Best Model", _best or "N/A")
    with g3:
        _sc = _rec["scorecard"]
        _mae_v = _sc.get("MAE", np.nan)
        st.metric("MAE", f"{_mae_v:,.0f}" if not np.isnan(_mae_v) else "N/A")
    with g4:
        _r2_v = _sc.get("R2", np.nan)
        st.metric("R²", f"{_r2_v:.4f}" if not np.isnan(_r2_v) else "N/A")

    # Show scorecard table
    if _rec["scorecard"]:
        st.markdown(format_scorecard_markdown(_rec["scorecard"]))

    # Tips
    if _rec["risk_flags"]:
        for rf in _rec["risk_flags"]:
            st.error(f"**Risk:** {rf}")
    if _rec["tips"]:
        for tip in _rec["tips"]:
            st.info(f"**Tip:** {tip}")
    if _rec["next_steps"]:
        st.markdown("**Next steps:**")
        for ns in _rec["next_steps"]:
            st.markdown(f"- {ns}")

# Optional date filter
df_t = pred[pred["target"] == tgt]
dmin, dmax = df_t["date"].min().date(), df_t["date"].max().date()
date_range = st.slider("Limit date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))
mask_date = (df_t["date"] >= pd.Timestamp(date_range[0])) & (df_t["date"] <= pd.Timestamp(date_range[1]))
df_t = df_t[mask_date]

if "horizon" in df_t.columns:
    df_t = df_t[df_t["horizon"].isin(hz)]
if "model" in df_t.columns:
    df_t = df_t[df_t["model"].isin(models_sel)]

# Decide display resampling
_freq = None
if gran == "Weekly (Fri)": _freq = "W-FRI"
elif gran == "Monthly (EOM)": _freq = "ME"

# Detect if PI columns present AND contain at least some non-NaN values
has_pi_cols = {"y_lo","y_hi"}.issubset(set(df_t.columns))
has_pi = has_pi_cols and df_t["y_lo"].notna().any() and df_t["y_hi"].notna().any()
if has_pi_cols and not has_pi:
    st.warning("⚠️ **Prediction interval columns exist but contain no values.** "
               "This run did not produce prediction intervals (common for ML models "
               "with insufficient validation data). Point forecasts are still valid.")

# -------------------- Tabs --------------------
tab_overlay, tab_leader, tab_errors, tab_intervals, tab_integrity, tab_feat_imp, tab_ensemble, tab_downloads = st.tabs(
    ["Overlay", "Leaderboard", "Errors & Residuals", "Interval Diagnostics", "Forecast Integrity", "Feature Importance", "Ensemble", "Downloads"]
)

# -------------------- Overlay (native daily/weekly) --------------------
with tab_overlay:
    st.subheader("Actual vs Model(s) vs Treasury baseline")
    
    # ✅ Forecast integrity note
    if "horizon" in df_t.columns and df_t["horizon"].nunique() > 0:
        h_sample = int(df_t["horizon"].iloc[0]) if len(df_t) > 0 else 1
        st.caption(
            f"ℹ️ **Forecast indexing**: Predictions shown at **target dates** (origin + h={h_sample} days). "
            f"Each prediction uses only information available at its forecast origin."
        )
    
    show_pi = st.checkbox("Show prediction intervals (only if one model selected)", value=False)
    treat_as_flow = st.checkbox("Treat series as **flow** when resampling (sum). Uncheck for **stock** (EOP).", value=("balance" not in tgt.lower()))

    # Build figure
    fig = go.Figure()
    # Actual
    s_true = df_t[["date","y_true"]].dropna().set_index("date").sort_index()
    if _freq:
        if treat_as_flow:
            s_true = s_true.resample(_freq).sum()
        else:
            s_true = s_true.resample(_freq).last()
        s_true = s_true.dropna()
    fig.add_scatter(x=s_true.index, y=s_true["y_true"], name="Actual", mode="lines", line=dict(color="black", width=2))

    # Baseline (if available)
    ops = _read_ops_baseline(base_dir, tgt)
    if ops is None:
        ops = _weekday_mean_baseline(df_t)
    if ops is not None and not ops.empty:
        ops = ops.loc[(ops.index >= s_true.index.min()) & (ops.index <= s_true.index.max())]
        if _freq:
            ops = ops.resample(_freq).sum() if treat_as_flow else ops.resample(_freq).last()
        if not ops.empty:
            fig.add_scatter(x=ops.index, y=ops.values, name="Ops baseline", mode="lines",
                            line=dict(dash="dot", width=2))

    # Models
    for m in models_sel:
        gm = df_t[df_t["model"] == m].copy().sort_values("date")
        if gm.empty: 
            continue
        # ✅ Only plot true out-of-sample predictions - drop rows where y_true is NaN
        # This prevents misleading charts from including training data or missing values filled with 0
        gm = gm.dropna(subset=["y_true", "y_pred"])  # Drop NaNs, don't fill with 0
        if gm.empty:
            continue
        s_pred = gm.set_index("date")["y_pred"]
        s_pi_lo = gm.set_index("date")["y_lo"] if has_pi else None
        s_pi_hi = gm.set_index("date")["y_hi"] if has_pi else None
        if _freq:
            s_pred = s_pred.resample(_freq).sum() if treat_as_flow else s_pred.resample(_freq).last()
            if has_pi:
                s_pi_lo = s_pi_lo.resample(_freq).sum() if treat_as_flow else s_pi_lo.resample(_freq).last()
                s_pi_hi = s_pi_hi.resample(_freq).sum() if treat_as_flow else s_pi_hi.resample(_freq).last()
        # Drop NaNs after resampling too
        s_pred = s_pred.dropna()
        fig.add_scatter(x=s_pred.index, y=s_pred.values, name=m, mode="lines")

        # Optional interval band (only if a single model is selected & PI columns present)
        if show_pi and has_pi and len(models_sel) == 1:
            fig.add_traces([
                go.Scatter(
                    x=s_pi_hi.index, y=s_pi_hi.values, name="PI 90% hi", mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ),
                go.Scatter(
                    x=s_pi_lo.index, y=s_pi_lo.values, name="PI 90% lo", mode="lines",
                    fill="tonexty", line=dict(width=0), fillcolor="rgba(31,119,180,0.2)",
                    showlegend=False, hoverinfo="skip"
                )
            ])

    fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Small multiples (optional)
    st.markdown("**Per‑horizon small‑multiples (top selected model)**")
    if models_sel:
        m0 = models_sel[0]
        sub = df_t[df_t["model"] == m0].copy()
        if "horizon" in sub.columns and sub["horizon"].nunique() > 1:
            grid = px.line(sub, x="date", y="y_pred", color="model", facet_row=None, facet_col="horizon",
                           facet_col_wrap=4, height=300 + 130 * (int(np.ceil(len(sub['horizon'].unique())/4))),
                           title=f"{m0} — predictions by horizon")
            st.plotly_chart(grid, use_container_width=True, config={"displaylogo": False})

# -------------------- Leaderboard --------------------
with tab_leader:
    st.subheader("Leaderboard & model comparison")
    if metr is None or metr.empty:
        st.info("No **metrics_long.csv** for this run/folder.")
    else:
        # filter to target + selected horizons
        mf = metr.copy()
        if "target" in mf.columns:
            mf = mf[mf["target"] == tgt]
        if "horizon" in mf.columns and len(hz):
            mf = mf[mf["horizon"].isin(hz)]
        metric_choice = st.selectbox("Metric", ["MAE","RMSE","sMAPE","R2","Monthly_TOL10_Accuracy","MAE_skill_vs_Ops"])
        if metric_choice not in mf.columns:
            st.warning(f"Metric '{metric_choice}' not found in metrics file.")
        else:
            agg = mf.groupby("model", as_index=False)[metric_choice].mean().sort_values(metric_choice, ascending=(metric_choice!="R2"))
            fig_lb = px.bar(agg, x="model", y=metric_choice, title=f"Average {metric_choice} across folds", height=380)
            st.plotly_chart(fig_lb, use_container_width=True, config={"displaylogo": False})
            st.dataframe(agg, use_container_width=True)

# -------------------- Errors & residuals --------------------
with tab_errors:
    st.subheader("Error diagnostics (selected model)")
    if not models_sel:
        st.info("Select at least one model.")
    else:
        m0 = models_sel[0]
        g = df_t[df_t["model"] == m0].copy()
        g["abs_err"] = (g["y_true"] - g["y_pred"]).abs()
        g["resid"] = (g["y_true"] - g["y_pred"])
        g = g.sort_values("date")

        c1, c2 = st.columns(2)
        with c1:
            fig_e = px.line(g, x="date", y="abs_err", title=f"Absolute error over time — {m0}", height=300)
            st.plotly_chart(fig_e, use_container_width=True, config={"displaylogo": False})
        with c2:
            hm = g.copy()
            hm["month"] = hm["date"].dt.to_period("M").dt.to_timestamp()
            agg = hm.groupby("month", as_index=False)["abs_err"].mean()
            fig_h = px.density_heatmap(agg, x="month", y=[m0]*len(agg), z="abs_err",
                                       color_continuous_scale="Blues",
                                       labels={"month": "Month", "y": "Model", "abs_err": "MAE"},
                                       height=300)
            fig_h.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_h, use_container_width=True, config={"displaylogo": False})

        c3, c4 = st.columns(2)
        with c3:
            bins = st.slider("Histogram bins", 10, 200, 40)
            fig_hist = px.histogram(g, x="resid", nbins=bins, title=f"Residual distribution — {m0}", height=300)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False})
        with c4:
            # Rolling MAE
            g2 = g.set_index("date").sort_index()
            g2["roll_mae_20"] = g2["abs_err"].rolling(20, min_periods=5).mean()
            fig_r = px.line(g2.reset_index(), x="date", y="roll_mae_20", title="Rolling MAE (20 periods)", height=300)
            st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

# -------------------- Interval diagnostics --------------------
with tab_intervals:
    st.subheader("Prediction interval coverage (if available)")
    if not has_pi:
        st.info("No interval columns (`y_lo`, `y_hi`) found in predictions.")
    else:
        dfpi = df_t.dropna(subset=["y_lo","y_hi"]).copy()
        if dfpi.empty:
            st.info("Intervals present but no non‑NaN rows after filtering.")
        else:
            dfpi["covered"] = ((dfpi["y_true"] >= dfpi["y_lo"]) & (dfpi["y_true"] <= dfpi["y_hi"])).astype(int)
            cov = dfpi.groupby("model", as_index=False)["covered"].mean()
            fig_cov = px.bar(cov, x="model", y="covered", title="Empirical coverage of PI bands", height=320)
            fig_cov.update_yaxes(range=[0,1])
            st.plotly_chart(fig_cov, use_container_width=True, config={"displaylogo": False})

            # Band width
            dfpi["bandwidth"] = (dfpi["y_hi"] - dfpi["y_lo"]).astype(float)
            bw = dfpi.groupby("model", as_index=False)["bandwidth"].mean()
            fig_bw = px.bar(bw, x="model", y="bandwidth", title="Average PI width", height=320)
            st.plotly_chart(fig_bw, use_container_width=True, config={"displaylogo": False})

# -------------------- Forecast Integrity --------------------
with tab_integrity:
    st.subheader("Forecast Integrity Diagnostics")
    st.write(
        "These checks verify that forecasts are **real** (no horizon misalignment, no leakage, correct indexing) "
        "and **useful** (better than naive baselines)."
    )
    
    # Try to load integrity report
    integrity_path = base_dir / "artifacts" / "integrity_report.json"
    if not integrity_path.exists():
        # Also check parent directories
        integrity_path = out_root / "artifacts" / "integrity_report.json"
    
    if integrity_path.exists():
        import json
        try:
            with open(integrity_path, "r") as f:
                integrity = json.load(f)
            
            # ✅ Trust & Usefulness Panel
            st.markdown("---")
            st.subheader("🔒 Trust & Usefulness")
            
            alignment_ok = integrity.get("alignment_ok", True)
            skill_pct = integrity.get("skill_pct", np.nan)
            skill_threshold = QUALITY_GATE_SKILL_PCT  # Must match backend gate
            
            # Determine verdict
            if not alignment_ok:
                verdict = "❌ NOT OK — Alignment Error"
                verdict_color = "error"
            elif np.isnan(skill_pct):
                verdict = "⚠️ UNKNOWN — Skill cannot be computed"
                verdict_color = "warning"
            elif skill_pct < skill_threshold:
                verdict = f"❌ NOT USEFUL — Skill ({skill_pct:.2f}%) below threshold ({skill_threshold}%)"
                verdict_color = "error"
            else:
                verdict = f"✅ OK TO USE — Skill {skill_pct:.2f}% >= {skill_threshold}%"
                verdict_color = "success"
            
            # Display verdict prominently
            if verdict_color == "success":
                st.success(f"**Verdict:** {verdict}")
            elif verdict_color == "error":
                st.error(f"**Verdict:** {verdict}")
            else:
                st.warning(f"**Verdict:** {verdict}")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                alignment_status = "✅ OK" if alignment_ok else f"❌ {integrity.get('n_misaligned', 0)} misaligned"
                st.metric("Alignment", alignment_status)
                if not alignment_ok and integrity.get("misaligned_examples"):
                    with st.expander("Misaligned examples"):
                        st.json(integrity["misaligned_examples"][:3])
            
            with col2:
                mae_model = integrity.get("mae_model", np.nan)
                st.metric("Model MAE", f"{mae_model:,.0f}" if not np.isnan(mae_model) else "N/A")
            
            with col3:
                mae_persist = integrity.get("mae_persistence", np.nan)
                st.metric("Baseline MAE", f"{mae_persist:,.0f}" if not np.isnan(mae_persist) else "N/A",
                         delta=f"{(mae_persist - mae_model):,.0f}" if not (np.isnan(mae_persist) or np.isnan(mae_model)) else None)
            
            with col4:
                skill_display = f"{skill_pct:.2f}%" if not np.isnan(skill_pct) else "N/A"
                skill_delta = f"{skill_pct - skill_threshold:.2f}%" if not np.isnan(skill_pct) else None
                st.metric("Skill %", skill_display, 
                         delta=skill_delta if skill_delta and float(skill_delta.replace("%", "")) >= 0 else None,
                         delta_color="normal" if (not np.isnan(skill_pct) and skill_pct >= skill_threshold) else "inverse")
            
            # Run status
            run_status = integrity.get("run_status", "UNKNOWN")
            if run_status == "FAILED_QUALITY":
                st.error(f"**Run Status:** {run_status} — Model does not beat persistence baseline at horizon {integrity.get('horizon', 'N/A')}")
            elif run_status == "SUCCESS":
                st.success(f"**Run Status:** {run_status}")
            else:
                st.info(f"**Run Status:** {run_status}")
            
            st.markdown("---")
            st.subheader("Detailed Diagnostics")
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Horizon", integrity.get("horizon", "N/A"))
                st.metric("Best Shift", integrity.get("best_shift", 0))
            with col2:
                rmse_model = integrity.get("rmse_model", np.nan)
                rmse_persist = integrity.get("rmse_persistence", np.nan)
                st.metric("Model RMSE", f"{rmse_model:.2f}" if not np.isnan(rmse_model) else "N/A")
                st.metric("Baseline RMSE", f"{rmse_persist:.2f}" if not np.isnan(rmse_persist) else "N/A")
            with col3:
                r2_model = integrity.get("r2_model", np.nan)
                r2_persist = integrity.get("r2_persistence", np.nan)
                st.metric("Model R²", f"{r2_model:.3f}" if not np.isnan(r2_model) else "N/A")
                st.metric("Baseline R²", f"{r2_persist:.3f}" if not np.isnan(r2_persist) else "N/A")
            
            # Warnings
            if integrity.get("lag_warning", False):
                st.error(
                    f"⚠️ **Lag Warning**: Best error occurs at shift={integrity.get('best_shift', 0)} "
                    f"(should be 0). This suggests horizon misalignment or plotting offset."
                )
            else:
                st.success("✓ **Shift Check**: Best alignment at shift=0 (correct)")
            
            if integrity.get("leakage_warning", False):
                st.error(
                    f"⚠️ **Leakage Warning**: Shuffled target performance is suspiciously close to normal. "
                    f"This may indicate data leakage."
                )
            else:
                shuffled_mae = integrity.get("mae_shuffled_target", np.nan)
                if not np.isnan(shuffled_mae):
                    st.success("✓ **Leakage Check**: Shuffled target test passed (no leakage detected)")
            
            # Baseline comparison table
            st.subheader("Baseline Comparisons")
            baseline_data = {
                "Metric": ["MAE", "RMSE", "R²"],
                "Model": [
                    f"{integrity.get('mae_model', np.nan):,.0f}" if not np.isnan(integrity.get('mae_model', np.nan)) else "N/A",
                    f"{integrity.get('rmse_model', np.nan):.2f}" if not np.isnan(integrity.get('rmse_model', np.nan)) else "N/A",
                    f"{integrity.get('r2_model', np.nan):.3f}" if not np.isnan(integrity.get('r2_model', np.nan)) else "N/A",
                ],
                "Persistence Baseline": [
                    f"{integrity.get('mae_persistence', np.nan):,.0f}" if not np.isnan(integrity.get('mae_persistence', np.nan)) else "N/A",
                    f"{integrity.get('rmse_persistence', np.nan):.2f}" if not np.isnan(integrity.get('rmse_persistence', np.nan)) else "N/A",
                    f"{integrity.get('r2_persistence', np.nan):.3f}" if not np.isnan(integrity.get('r2_persistence', np.nan)) else "N/A",
                ],
                "Seasonal Naive": [
                    f"{integrity.get('mae_seasonal_naive', np.nan):,.0f}" if not np.isnan(integrity.get('mae_seasonal_naive', np.nan)) else "N/A",
                    "N/A",  # RMSE not computed for seasonal naive
                    "N/A",  # R² not computed for seasonal naive
                ],
            }
            baseline_df = pd.DataFrame(baseline_data)
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)
            
            # Full report
            with st.expander("View full integrity report (JSON)"):
                st.json(integrity)
                
        except Exception as e:
            st.warning(f"Could not parse integrity report: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info(
            "No **integrity_report.json** found. This report is generated by ML pipelines "
            "(b_ml_pipeline.py) and includes shift sanity checks, baseline comparisons, and leakage tests."
        )
        st.caption(
            "The integrity report helps verify that:\n"
            "- Forecasts are correctly aligned (no lag/offset)\n"
            "- Predictions use only information available at forecast origin\n"
            "- Model beats simple baselines (persistence, seasonal naive)\n"
            "- No data leakage is present"
        )

# -------------------- Feature Importance --------------------
with tab_feat_imp:
    st.subheader("Feature Importance (ML pipelines)")
    st.caption("Shows which input features the ML models rely on most. Available for tree-based and linear models.")

    fi_path = base_dir / "artifacts" / "feature_importance.csv"
    if not fi_path.exists():
        fi_path = out_root / "artifacts" / "feature_importance.csv"

    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        if fi_df.empty:
            st.info("Feature importance file is empty.")
        else:
            fi_models = sorted(fi_df["model"].unique())
            fi_model_sel = st.selectbox("Model", fi_models, index=0, key="fi_model_sel")

            fi_m = fi_df[fi_df["model"] == fi_model_sel].copy()
            fi_m = fi_m.sort_values("importance", ascending=False)

            # Top-N control
            top_n = st.slider("Show top N features", 5, min(50, len(fi_m)), min(20, len(fi_m)), key="fi_topn")
            fi_top = fi_m.head(top_n)

            # Horizontal bar chart
            fig_fi = px.bar(
                fi_top, x="importance", y="feature",
                orientation="h",
                title=f"Top {top_n} features for {fi_model_sel}",
                height=max(300, top_n * 22),
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True, config={"displaylogo": False})

            # Data table
            with st.expander("Full feature importance table"):
                st.dataframe(fi_m[["feature", "importance"]], use_container_width=True, hide_index=True)

            # Download
            st.download_button(
                "Download feature importance (CSV)",
                data=fi_df.to_csv(index=False).encode("utf-8"),
                file_name=f"feature_importance_{sel_run}.csv",
                key="fi_dl_btn",
            )
    else:
        st.info(
            "No **feature_importance.csv** found for this run. "
            "Feature importance is extracted from ML pipeline (family B) runs. "
            "Run a B-ML experiment to see this data."
        )

# -------------------- Ensemble --------------------
with tab_ensemble:
    st.subheader("Ensemble — Combine multiple runs")
    st.write(
        "Select 2+ runs below to create ensembles (median, top-K mean, inverse-MAE weighted). "
        "The ensemble combines predictions from different models into a single, often more accurate, forecast."
    )
    _all_runs = _list_runs()
    _run_names = [r.name for r in _all_runs]
    _ens_selected = st.multiselect(
        "Runs to combine", _run_names,
        default=[_run_names[0]] if _run_names else [],
        help="Select 2+ runs. Each should have predictions_long.csv.",
        key="ens_run_selector",
    )

    _ens_topk = st.number_input("Top-K for mean ensemble", 1, 20, 3, key="ens_topk",
                                help="How many best models to average in ens_mean_topK.")

    if st.button("Build Ensemble", type="primary", key="ens_build_btn"):
        if len(_ens_selected) < 2:
            st.error("Select at least 2 runs to create an ensemble.")
        else:
            import sys
            _backend_dir = APPROOT.parent / "backend"
            if str(_backend_dir) not in sys.path:
                sys.path.insert(0, str(_backend_dir))
            try:
                from ensemble_postprocess import run_ensemble_from_runs
                _ens_dirs = [str(RUNS_DIR / rn) for rn in _ens_selected]
                _ens_out = str(RUNS_DIR / f"_ensemble_{'_'.join(_ens_selected[:3])}")
                with st.spinner("Building ensembles..."):
                    result = run_ensemble_from_runs(_ens_dirs, _ens_out, cadence="daily", top_k=int(_ens_topk))
                ens_preds = result.get("predictions", pd.DataFrame())
                ens_metr = result.get("metrics", pd.DataFrame())
                ens_lb = result.get("leaderboard", pd.DataFrame())

                if ens_preds.empty:
                    st.warning("No ensemble predictions could be generated. Check that runs have compatible predictions.")
                else:
                    st.success(f"Ensemble built with {len(ens_preds)} predictions across {ens_preds['model'].nunique()} methods.")

                    # Overlay chart
                    fig_ens = go.Figure()
                    _first_model = ens_preds["model"].unique()[0]
                    _base = ens_preds[ens_preds["model"] == _first_model].sort_values("date")
                    fig_ens.add_scatter(x=pd.to_datetime(_base["date"]), y=_base["y_true"],
                                        name="Actual", mode="lines", line=dict(color="black", width=2))
                    for em in ens_preds["model"].unique():
                        _eg = ens_preds[ens_preds["model"] == em].sort_values("date")
                        fig_ens.add_scatter(x=pd.to_datetime(_eg["date"]), y=_eg["y_pred"],
                                            name=em, mode="lines")
                    fig_ens.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig_ens, use_container_width=True, config={"displaylogo": False})

                    # Leaderboard
                    if not ens_lb.empty:
                        st.subheader("Ensemble Leaderboard")
                        st.dataframe(ens_lb, use_container_width=True, hide_index=True)

                    # Download
                    st.download_button(
                        "Download ensemble predictions (CSV)",
                        data=ens_preds.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_ensemble.csv",
                        key="ens_dl_btn",
                    )
            except Exception as e:
                st.error(f"Ensemble failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# -------------------- Downloads --------------------
with tab_downloads:
    st.subheader("Download filtered data & all plots")
    # Filtered predictions snapshot
    csv_bytes = df_t.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered predictions (CSV)", data=csv_bytes, file_name=f"{sel_run}_filtered_predictions.csv", key = "button12")

    # Download available plots in selected folder as ZIP
    plots_dir = base_dir / "plots"
    if plots_dir.exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for img in sorted(plots_dir.glob("*.png")):
                zf.writestr(img.name, img.read_bytes())
        st.download_button("Download all static plots (ZIP)", data=buf.getvalue(),
                           file_name=f"plots_{sel_run}_{choice_key}.zip", key="button15")
    else:
        st.caption("No static plots found in this output folder.")
