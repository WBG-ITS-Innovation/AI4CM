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

# AFTER you compute out_root = run_dir / "outputs"
candidates = [("root", out_root), ("daily", out_root/"daily"), ("weekly", out_root/"weekly"), ("monthly", out_root/"monthly")]
available = [(name, p) for name,p in candidates if (p/"predictions_long.csv").exists()]
# default: first available; else fall back to root
default_key = available[0][0] if available else "root"

label_map = {"root": "outputs (root)", "daily": "outputs/daily", "weekly": "outputs/weekly", "monthly": "outputs/monthly"}
choice = st.radio("Use outputs from",
                  [label_map[k] for k,_ in candidates],
                  index=["root","daily","weekly","monthly"].index(default_key),
                  horizontal=True)
chosen_key = [k for k,v in label_map.items() if v == choice][0]
base_dir = dict(candidates)[chosen_key]

# load
p = base_dir / "predictions_long.csv"
if not p.exists():
    st.warning(f"No **predictions_long.csv** in {base_dir}")
    st.stop()


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

# Detect if PI columns present
has_pi = {"y_lo","y_hi"}.issubset(set(df_t.columns))

# -------------------- Tabs --------------------
tab_overlay, tab_leader, tab_errors, tab_intervals, tab_downloads = st.tabs(
    ["Overlay", "Leaderboard", "Errors & Residuals", "Interval Diagnostics", "Downloads"]
)

# -------------------- Overlay (native daily/weekly) --------------------
with tab_overlay:
    st.subheader("Actual vs Model(s) vs Treasury baseline")
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
        s_pred = gm.set_index("date")["y_pred"]
        s_pi_lo = gm.set_index("date")["y_lo"] if has_pi else None
        s_pi_hi = gm.set_index("date")["y_hi"] if has_pi else None
        if _freq:
            s_pred = s_pred.resample(_freq).sum() if treat_as_flow else s_pred.resample(_freq).last()
            if has_pi:
                s_pi_lo = s_pi_lo.resample(_freq).sum() if treat_as_flow else s_pi_lo.resample(_freq).last()
                s_pi_hi = s_pi_hi.resample(_freq).sum() if treat_as_flow else s_pi_hi.resample(_freq).last()
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
