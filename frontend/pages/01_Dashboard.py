# pages/01_Dashboard.py — Dashboard (interactive overlays, diagnostics, downloads)
# Redesigned UI: modern analytics aesthetic with strong information hierarchy
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import io, json, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from backend_consts import QUALITY_GATE_SKILL_PCT
from recommender import recommend_model, format_scorecard_markdown
from ui_styles import (
    inject_global_css, metric_card, status_badge, section_header,
    callout_box, grade_badge, COLORS,
)

APPROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = APPROOT / "runs"

st.set_page_config(page_title="Dashboard — AI4CM", layout="wide")
inject_global_css()

# ──────────────────────────────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=15)
def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

@st.cache_data(show_spinner=False, ttl=15)
def _list_runs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and (p/"outputs").exists()],
                  key=lambda p: p.stat().st_mtime, reverse=True)

# ──────────────────────────────────────────────────────────────────────
# File resolution
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# Baseline helpers
# ──────────────────────────────────────────────────────────────────────
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
            parts = []
            for ts, v in dfm["forecast"].items():
                days = pd.date_range(ts.replace(day=1), ts, freq="B")
                if len(days):
                    parts.append(pd.Series(float(v)/len(days), index=days))
            if parts:
                return pd.concat(parts).sort_index()
    return None

# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────
def _fmt_num(v, decimals=0, prefix="", suffix="") -> str:
    """Format a number with commas, or 'N/A' if NaN."""
    if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
        return "N/A"
    if decimals == 0:
        return f"{prefix}{v:,.0f}{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _fmt_pct(v, decimals=1) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}%"


def _trust_status(grade: str) -> str:
    """Map accuracy grade to trust status string."""
    return {"A": "trust", "B": "trust", "C": "caution", "D": "caution", "F": "fail"}.get(grade, "neutral")


# ======================================================================
# PAGE START
# ======================================================================

# ── Page header ──────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:28px; font-weight:700; margin-bottom:2px;">'
    '📈 Dashboard</h1>'
    '<p style="color:#64748b; font-size:14px; margin-top:0;">'
    'Interactive results, diagnostics, and model recommendations</p>',
    unsafe_allow_html=True,
)

# ── Run selector ─────────────────────────────────────────────────────
runs = _list_runs()
if not runs:
    st.markdown(
        callout_box(
            "No runs yet. Use the <b>Lab</b> page to create your first experiment.",
            "info", icon="🧪",
        ),
        unsafe_allow_html=True,
    )
    st.stop()

sel_run = st.selectbox("Select run", [r.name for r in runs], index=0, label_visibility="collapsed")
run_dir = RUNS_DIR / sel_run
out_root = run_dir / "outputs"

cads = _base_dirs(out_root)
label_map = {"root": "All outputs", "daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}
cad_choice_label = st.radio(
    "Output cadence", [label_map[c] for c, _ in cads], horizontal=True,
    label_visibility="collapsed",
)
choice_key = [k for k, lbl in label_map.items() if lbl == cad_choice_label][0]
base_dir = dict(cads)[choice_key]

pred, metr, lb = _load_outputs(base_dir)
if pred is None or pred.empty:
    st.markdown(
        callout_box(
            f"No predictions found in <code>{base_dir}</code>. "
            "The run may still be in progress or may have failed.",
            "caution", icon="⏳",
        ),
        unsafe_allow_html=True,
    )
    st.stop()

# ── Normalize predictions ────────────────────────────────────────────
pred = pred.copy()
pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
pred = pred.dropna(subset=["date"]).sort_values("date")
if "target" not in pred.columns: pred["target"] = "Series"
if "cadence" not in pred.columns:
    pred["cadence"] = choice_key if choice_key != "root" else "unknown"

targets = sorted(pred["target"].unique())
horizons = sorted(pred["horizon"].unique()) if "horizon" in pred.columns else [1]
models_all = sorted(pred["model"].unique()) if "model" in pred.columns else ["Model"]

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Trust Verdict & Scorecard (top summary area)
# ══════════════════════════════════════════════════════════════════════

# Load integrity report
_integrity_path = base_dir / "artifacts" / "integrity_report.json"
if not _integrity_path.exists():
    _integrity_path = out_root / "artifacts" / "integrity_report.json"

_integ = None
if _integrity_path.exists():
    try:
        _integ = json.loads(_integrity_path.read_text(encoding="utf-8"))
    except Exception:
        pass

# Load config for scorecard
_scorecard_cfg = {}
_scorecard_cfg_path = base_dir / "artifacts" / "config.json"
if not _scorecard_cfg_path.exists():
    _scorecard_cfg_path = out_root / "artifacts" / "config.json"
if _scorecard_cfg_path.exists():
    try:
        _scorecard_cfg = json.loads(_scorecard_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass

_rec = recommend_model(pred, metr, lb, _scorecard_cfg,
                       target=targets[0] if targets else "",
                       horizon=horizons[0] if horizons else 1)

_grade = _rec["accuracy_grade"]
_best = _rec["best_model"]
_sc = _rec["scorecard"]
_grade_status = _trust_status(_grade)

# ── Trust verdict banner ─────────────────────────────────────────────
if _integ:
    _pipeline = _integ.get("pipeline", "ML")
    _run_status = _integ.get("run_status", "UNKNOWN")
    _skill = _integ.get("skill_pct", None)
    _qg = _integ.get("quality_gate_passed", None)
    _alignment_ok = _integ.get("alignment_ok", True)

    if _run_status == "SUCCESS" and _qg:
        # Quality gate PASSED — trust the forecast
        _verdict_status = "trust"
        _verdict_icon = "✅"
        _verdict_title = "Forecast Trusted — Quality Gate Passed"
        _verdict_detail = (
            f"{_pipeline} pipeline passed all checks. "
            f"Skill over baseline: {_skill:.1f}%."
        )
    elif _run_status == "FAILED_QUALITY":
        # Quality gate FAILED — model underperforms baseline
        _verdict_status = "fail"
        _verdict_icon = "⚠️"
        _verdict_title = "Caution — Model Underperforms Baseline"
        _verdict_detail = (
            f"{_pipeline} pipeline: skill {_fmt_pct(_skill)} is below the "
            f"{QUALITY_GATE_SKILL_PCT}% threshold. Outputs exist but should not "
            "be used for decisions without review."
        )
    elif _run_status == "ERROR":
        _verdict_status = "fail"
        _verdict_icon = "❌"
        _verdict_title = "Error — Integrity Check Failed"
        _verdict_detail = f"Error: {_integ.get('error', 'unknown')}"
    elif not _alignment_ok:
        _verdict_status = "fail"
        _verdict_icon = "🔴"
        _verdict_title = "Alignment Error Detected"
        _verdict_detail = "Forecast dates may be misaligned. Review integrity tab."
    else:
        _verdict_status = "info"
        _verdict_icon = "ℹ️"
        _verdict_title = f"Pipeline Status: {_run_status}"
        _verdict_detail = f"{_pipeline} pipeline completed."

    c = COLORS
    _vbg = c.get(f"{_verdict_status}_bg", c["neutral_bg"])
    _vbdr = c.get(f"{_verdict_status}_bdr", c["neutral_bdr"])
    _vaccent = c.get(_verdict_status, c["neutral"])

    st.markdown(
        f'<div class="trust-verdict" style="background:{_vbg}; border:1.5px solid {_vbdr};">'
        f'<span class="tv-icon">{_verdict_icon}</span>'
        f'<div class="tv-text">'
        f'<div class="tv-title" style="color:{_vaccent};">{_verdict_title}</div>'
        f'<div class="tv-detail" style="color:{_vaccent};">{_verdict_detail}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        callout_box(
            "No integrity report found. Interpret outputs with caution — "
            "trust checks could not be performed.",
            "caution", icon="⚠️",
        ),
        unsafe_allow_html=True,
    )

# ── Scorecard cards row ──────────────────────────────────────────────
st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

_mae_v = _sc.get("MAE", np.nan)
_r2_v = _sc.get("R2", np.nan)
_smape_v = _sc.get("sMAPE", np.nan)
_pi_cov = _sc.get("PI Coverage", np.nan)
_monthly_acc = _sc.get("Monthly Accuracy (10% tol)", np.nan)

# Determine R² status (negative R² is a red flag)
_r2_status = "neutral"
if not np.isnan(_r2_v):
    if _r2_v < 0:
        _r2_status = "fail"
    elif _r2_v >= 0.85:
        _r2_status = "trust"
    elif _r2_v >= 0.50:
        _r2_status = "caution"
    else:
        _r2_status = "fail"

g1, g2, g3, g4, g5, g6 = st.columns(6)

with g1:
    st.markdown(grade_badge(_grade), unsafe_allow_html=True)

with g2:
    st.markdown(
        metric_card("Best Model", _best or "N/A", icon="🏆", status=_grade_status),
        unsafe_allow_html=True,
    )

with g3:
    _mae_delta = ""
    if _integ and not np.isnan(_mae_v):
        _mae_persist = _integ.get("mae_persistence", np.nan)
        if not np.isnan(_mae_persist) and _mae_persist > 0:
            _skill_v = (1 - _mae_v / _mae_persist) * 100
            _mae_delta = f"Skill: {_skill_v:.1f}%"
    st.markdown(
        metric_card("MAE", _fmt_num(_mae_v), delta=_mae_delta,
                     icon="📐", status=_grade_status),
        unsafe_allow_html=True,
    )

with g4:
    _r2_display = f"{_r2_v:.4f}" if not np.isnan(_r2_v) else "N/A"
    _r2_delta = ""
    if not np.isnan(_r2_v) and _r2_v < 0:
        _r2_delta = "Worse than mean"
    st.markdown(
        metric_card("R-Squared", _r2_display, delta=_r2_delta,
                     icon="📊", status=_r2_status),
        unsafe_allow_html=True,
    )

with g5:
    _pi_display = f"{_pi_cov:.0%}" if not np.isnan(_pi_cov) else "N/A"
    _pi_status = "neutral"
    if not np.isnan(_pi_cov):
        _pi_status = "trust" if _pi_cov >= 0.85 else ("caution" if _pi_cov >= 0.70 else "fail")
    st.markdown(
        metric_card("PI Coverage", _pi_display, icon="🎯", status=_pi_status),
        unsafe_allow_html=True,
    )

with g6:
    _ma_display = f"{_monthly_acc:.0%}" if not np.isnan(_monthly_acc) else "N/A"
    _ma_status = "neutral"
    if not np.isnan(_monthly_acc):
        _ma_status = "trust" if _monthly_acc >= 0.80 else ("caution" if _monthly_acc >= 0.50 else "fail")
    st.markdown(
        metric_card("Monthly Acc", _ma_display, delta="within 10% tol",
                     icon="📅", status=_ma_status),
        unsafe_allow_html=True,
    )

# ── Scorecard detail table + warnings ────────────────────────────────
st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

with st.expander("Detailed Scorecard & Recommendations", expanded=False):
    if _sc:
        # Build a clean styled table
        _sc_rows = []
        _metric_descriptions = {
            "MAE": "Mean Absolute Error — average magnitude of prediction errors",
            "RMSE": "Root Mean Squared Error — penalises large errors more",
            "MAPE": "Mean Absolute Percentage Error — percentage-based error",
            "sMAPE": "Symmetric MAPE — handles near-zero actuals better",
            "R2": "R-Squared — fraction of variance explained (1.0 = perfect)",
            "Monthly Accuracy (10% tol)": "Share of months where forecast is within 10% of actual",
            "PI Coverage": "Share of actuals falling inside prediction intervals (target: 90%)",
            "PI Avg Width": "Average width of prediction intervals (narrower = more precise)",
            "N predictions": "Number of out-of-sample prediction points used",
        }
        for k, v in _sc.items():
            if isinstance(v, float) and np.isnan(v):
                formatted = "N/A"
            elif k in ("MAPE", "sMAPE", "Monthly Accuracy (10% tol)", "PI Coverage"):
                formatted = f"{v:.1%}"
            elif k in ("MAE", "RMSE", "PI Avg Width"):
                formatted = f"{v:,.0f}"
            elif k == "R2":
                formatted = f"{v:.4f}"
            elif isinstance(v, float):
                formatted = f"{v:.4f}"
            else:
                formatted = str(v)

            _sc_rows.append({
                "Metric": k,
                "Value": formatted,
                "Description": _metric_descriptions.get(k, ""),
            })

        sc_df = pd.DataFrame(_sc_rows)
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # Risk flags
    if _rec["risk_flags"]:
        st.markdown(section_header("Risk Flags", "Issues that need attention"), unsafe_allow_html=True)
        for rf in _rec["risk_flags"]:
            st.markdown(callout_box(rf, "fail", icon="🔴"), unsafe_allow_html=True)

    # Tips
    if _rec["tips"]:
        st.markdown(section_header("Tips", "Observations about this run"), unsafe_allow_html=True)
        for tip in _rec["tips"]:
            st.markdown(callout_box(tip, "info", icon="💡"), unsafe_allow_html=True)

    # Next steps
    if _rec["next_steps"]:
        st.markdown(section_header("Next Steps"), unsafe_allow_html=True)
        for ns in _rec["next_steps"]:
            st.markdown(callout_box(ns, "neutral", icon="→"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Controls
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
st.markdown(section_header("Analysis Controls", "Filter by target, horizon, model, and date range"), unsafe_allow_html=True)

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

# Date filter
df_t = pred[pred["target"] == tgt]
dmin, dmax = df_t["date"].min().date(), df_t["date"].max().date()
date_range = st.slider("Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))
mask_date = (df_t["date"] >= pd.Timestamp(date_range[0])) & (df_t["date"] <= pd.Timestamp(date_range[1]))
df_t = df_t[mask_date]

if "horizon" in df_t.columns:
    df_t = df_t[df_t["horizon"].isin(hz)]
if "model" in df_t.columns:
    df_t = df_t[df_t["model"].isin(models_sel)]

# Resampling
_freq = None
if gran == "Weekly (Fri)": _freq = "W-FRI"
elif gran == "Monthly (EOM)": _freq = "ME"

# PI detection
has_pi_cols = {"y_lo","y_hi"}.issubset(set(df_t.columns))
has_pi = has_pi_cols and df_t["y_lo"].notna().any() and df_t["y_hi"].notna().any()
if has_pi_cols and not has_pi:
    st.markdown(
        callout_box(
            "<b>Prediction intervals exist but contain no values.</b> "
            "This run did not produce intervals (common for ML models with "
            "insufficient validation data). Point forecasts are still valid.",
            "caution", icon="📉",
        ),
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Tabs
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

tab_overlay, tab_leader, tab_errors, tab_intervals, tab_integrity, tab_feat_imp, tab_ensemble, tab_downloads = st.tabs(
    ["📈 Overlay", "🏆 Leaderboard", "📉 Errors", "🎯 Intervals",
     "🔒 Integrity", "🔍 Features", "🔗 Ensemble", "📥 Downloads"]
)

# ── Tab: Overlay ─────────────────────────────────────────────────────
with tab_overlay:
    st.markdown(
        section_header(
            "Actual vs Model(s) vs Baseline",
            "Compare model predictions against actuals and treasury baselines",
        ),
        unsafe_allow_html=True,
    )

    if "horizon" in df_t.columns and df_t["horizon"].nunique() > 0:
        h_sample = int(df_t["horizon"].iloc[0]) if len(df_t) > 0 else 1
        st.caption(
            f"Predictions shown at target dates (origin + h={h_sample}). "
            "Each prediction uses only information available at forecast origin."
        )

    _ov_c1, _ov_c2 = st.columns(2)
    with _ov_c1:
        show_pi = st.checkbox("Show prediction intervals", value=False,
                              help="Only works when a single model is selected")
    with _ov_c2:
        treat_as_flow = st.checkbox(
            "Treat as flow (sum when resampling)",
            value=("balance" not in tgt.lower()),
            help="Uncheck for stock/balance series (use end-of-period)",
        )

    fig = go.Figure()
    s_true = df_t[["date","y_true"]].dropna().set_index("date").sort_index()
    if _freq:
        s_true = s_true.resample(_freq).sum() if treat_as_flow else s_true.resample(_freq).last()
        s_true = s_true.dropna()
    fig.add_scatter(
        x=s_true.index, y=s_true["y_true"],
        name="Actual", mode="lines",
        line=dict(color="#1e293b", width=2.5),
    )

    # Baseline
    ops = _read_ops_baseline(base_dir, tgt)
    if ops is None:
        ops = _weekday_mean_baseline(df_t)
    if ops is not None and not ops.empty:
        ops = ops.loc[(ops.index >= s_true.index.min()) & (ops.index <= s_true.index.max())]
        if _freq:
            ops = ops.resample(_freq).sum() if treat_as_flow else ops.resample(_freq).last()
        if not ops.empty:
            fig.add_scatter(
                x=ops.index, y=ops.values,
                name="Ops baseline", mode="lines",
                line=dict(dash="dot", width=2, color="#94a3b8"),
            )

    # Model traces
    _model_colors = px.colors.qualitative.Set2
    for i, m in enumerate(models_sel):
        gm = df_t[df_t["model"] == m].copy().sort_values("date")
        if gm.empty:
            continue
        gm = gm.dropna(subset=["y_true", "y_pred"])
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
        s_pred = s_pred.dropna()
        color = _model_colors[i % len(_model_colors)]
        fig.add_scatter(x=s_pred.index, y=s_pred.values, name=m, mode="lines",
                        line=dict(color=color, width=2))

        if show_pi and has_pi and len(models_sel) == 1:
            fig.add_traces([
                go.Scatter(
                    x=s_pi_hi.index, y=s_pi_hi.values, name="PI hi", mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ),
                go.Scatter(
                    x=s_pi_lo.index, y=s_pi_lo.values, name="PI lo", mode="lines",
                    fill="tonexty", line=dict(width=0),
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                    showlegend=False, hoverinfo="skip",
                ),
            ])

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#f1f5f9", showgrid=True),
        yaxis=dict(gridcolor="#f1f5f9", showgrid=True, tickformat=","),
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Small multiples
    with st.expander("Per-horizon breakdown (first selected model)", expanded=False):
        if models_sel:
            m0 = models_sel[0]
            sub = df_t[df_t["model"] == m0].copy()
            if "horizon" in sub.columns and sub["horizon"].nunique() > 1:
                grid = px.line(
                    sub, x="date", y="y_pred", color="model",
                    facet_col="horizon", facet_col_wrap=4,
                    height=300 + 130 * int(np.ceil(len(sub["horizon"].unique()) / 4)),
                    title=f"{m0} — predictions by horizon",
                )
                grid.update_layout(font=dict(family="Inter, -apple-system, sans-serif"))
                st.plotly_chart(grid, use_container_width=True, config={"displaylogo": False})
            else:
                st.caption("Only one horizon available — nothing to compare.")

# ── Tab: Leaderboard ─────────────────────────────────────────────────
with tab_leader:
    st.markdown(
        section_header("Leaderboard", "Compare models side-by-side on key metrics"),
        unsafe_allow_html=True,
    )

    if metr is None or metr.empty:
        st.markdown(
            callout_box("No metrics file found for this run.", "info", icon="ℹ️"),
            unsafe_allow_html=True,
        )
    else:
        mf = metr.copy()
        if "target" in mf.columns:
            mf = mf[mf["target"] == tgt]
        if "horizon" in mf.columns and len(hz):
            mf = mf[mf["horizon"].isin(hz)]

        metric_choice = st.selectbox(
            "Metric", ["MAE","RMSE","sMAPE","R2","Monthly_TOL10_Accuracy","MAE_skill_vs_Ops"],
            help="Select the metric to rank models by",
        )

        if metric_choice not in mf.columns:
            st.markdown(
                callout_box(f"Metric '{metric_choice}' not found in this run's output.", "caution", icon="⚠️"),
                unsafe_allow_html=True,
            )
        else:
            ascending = metric_choice != "R2"
            agg = mf.groupby("model", as_index=False)[metric_choice].mean().sort_values(
                metric_choice, ascending=ascending
            )

            # Styled bar chart
            color_scale = "Teal" if ascending else "Teal_r"
            fig_lb = px.bar(
                agg, x="model", y=metric_choice,
                title=f"Average {metric_choice} across folds",
                height=400,
                color=metric_choice,
                color_continuous_scale=color_scale,
            )
            fig_lb.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9"),
                yaxis=dict(gridcolor="#f1f5f9", tickformat=","),
                font=dict(family="Inter, -apple-system, sans-serif"),
            )
            st.plotly_chart(fig_lb, use_container_width=True, config={"displaylogo": False})

            # Highlight winner
            if len(agg) > 0:
                winner_row = agg.iloc[0]
                direction = "lowest" if ascending else "highest"
                st.markdown(
                    callout_box(
                        f"<b>{winner_row['model']}</b> has the {direction} {metric_choice}: "
                        f"<b>{_fmt_num(winner_row[metric_choice], decimals=2)}</b>",
                        "trust", icon="🏆",
                    ),
                    unsafe_allow_html=True,
                )

            st.dataframe(agg, use_container_width=True, hide_index=True)

# ── Tab: Errors & Residuals ──────────────────────────────────────────
with tab_errors:
    st.markdown(
        section_header("Error Diagnostics", "Residual analysis for the selected model"),
        unsafe_allow_html=True,
    )

    if not models_sel:
        st.markdown(callout_box("Select at least one model.", "info", icon="ℹ️"), unsafe_allow_html=True)
    else:
        m0 = models_sel[0]
        g = df_t[df_t["model"] == m0].copy()
        g["abs_err"] = (g["y_true"] - g["y_pred"]).abs()
        g["resid"] = g["y_true"] - g["y_pred"]
        g = g.sort_values("date")

        c1, c2 = st.columns(2)
        with c1:
            fig_e = px.line(g, x="date", y="abs_err", title=f"Absolute error — {m0}", height=340)
            fig_e.update_layout(plot_bgcolor="white", yaxis_tickformat=",",
                                font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_e, use_container_width=True, config={"displaylogo": False})

        with c2:
            hm = g.copy()
            hm["month"] = hm["date"].dt.to_period("M").dt.to_timestamp()
            agg_h = hm.groupby("month", as_index=False)["abs_err"].mean()
            fig_h = px.density_heatmap(
                agg_h, x="month", y=[m0]*len(agg_h), z="abs_err",
                color_continuous_scale="Teal",
                labels={"month": "Month", "y": "Model", "abs_err": "MAE"},
                height=340,
            )
            fig_h.update_yaxes(showticklabels=False)
            fig_h.update_layout(font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_h, use_container_width=True, config={"displaylogo": False})

        c3, c4 = st.columns(2)
        with c3:
            bins = st.slider("Histogram bins", 10, 200, 40)
            fig_hist = px.histogram(g, x="resid", nbins=bins, title=f"Residual distribution — {m0}", height=340)
            fig_hist.update_layout(plot_bgcolor="white",
                                   font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False})

        with c4:
            g2 = g.set_index("date").sort_index()
            g2["roll_mae_20"] = g2["abs_err"].rolling(20, min_periods=5).mean()
            fig_r = px.line(g2.reset_index(), x="date", y="roll_mae_20",
                            title="Rolling MAE (20 periods)", height=340)
            fig_r.update_layout(plot_bgcolor="white", yaxis_tickformat=",",
                                font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

# ── Tab: Interval Diagnostics ────────────────────────────────────────
with tab_intervals:
    st.markdown(
        section_header("Prediction Interval Coverage", "Are uncertainty bands well-calibrated?"),
        unsafe_allow_html=True,
    )

    if not has_pi:
        st.markdown(
            callout_box(
                "No prediction interval columns found. Intervals are produced by "
                "ML (conformal) and Statistical (native) pipelines.",
                "info", icon="ℹ️",
            ),
            unsafe_allow_html=True,
        )
    else:
        dfpi = df_t.dropna(subset=["y_lo","y_hi"]).copy()
        if dfpi.empty:
            st.markdown(
                callout_box("Interval columns exist but contain no data after filtering.", "caution", icon="⚠️"),
                unsafe_allow_html=True,
            )
        else:
            dfpi["covered"] = ((dfpi["y_true"] >= dfpi["y_lo"]) & (dfpi["y_true"] <= dfpi["y_hi"])).astype(int)
            cov = dfpi.groupby("model", as_index=False)["covered"].mean()

            fig_cov = px.bar(cov, x="model", y="covered",
                             title="Empirical coverage (target: 90%)", height=380,
                             color="covered", color_continuous_scale="Teal")
            fig_cov.add_hline(y=0.90, line_dash="dash", line_color=COLORS["fail"],
                              annotation_text="Target 90%")
            fig_cov.update_yaxes(range=[0, 1.05])
            fig_cov.update_layout(plot_bgcolor="white", showlegend=False,
                                  font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_cov, use_container_width=True, config={"displaylogo": False})

            dfpi["bandwidth"] = (dfpi["y_hi"] - dfpi["y_lo"]).astype(float)
            bw = dfpi.groupby("model", as_index=False)["bandwidth"].mean()
            fig_bw = px.bar(bw, x="model", y="bandwidth",
                            title="Average PI width (narrower = more precise)", height=380,
                            color="bandwidth", color_continuous_scale="Teal_r")
            fig_bw.update_layout(plot_bgcolor="white", showlegend=False, yaxis_tickformat=",",
                                 font=dict(family="Inter, -apple-system, sans-serif"))
            st.plotly_chart(fig_bw, use_container_width=True, config={"displaylogo": False})

# ── Tab: Forecast Integrity ──────────────────────────────────────────
with tab_integrity:
    st.markdown(
        section_header(
            "Forecast Integrity",
            "Verify alignment, detect leakage, and compare against baselines",
        ),
        unsafe_allow_html=True,
    )

    integrity_path = base_dir / "artifacts" / "integrity_report.json"
    if not integrity_path.exists():
        integrity_path = out_root / "artifacts" / "integrity_report.json"

    if integrity_path.exists():
        try:
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))

            alignment_ok = integrity.get("alignment_ok", True)
            skill_pct = integrity.get("skill_pct", np.nan)
            skill_threshold = QUALITY_GATE_SKILL_PCT

            # Verdict
            if not alignment_ok:
                _iv_status, _iv_text = "fail", "NOT OK — Alignment Error"
            elif np.isnan(skill_pct):
                _iv_status, _iv_text = "caution", "UNKNOWN — Skill cannot be computed"
            elif skill_pct < skill_threshold:
                _iv_status, _iv_text = "fail", f"NOT USEFUL — Skill {skill_pct:.2f}% < {skill_threshold}%"
            else:
                _iv_status, _iv_text = "trust", f"OK TO USE — Skill {skill_pct:.2f}% ≥ {skill_threshold}%"

            _iv_icon = {"trust": "✅", "caution": "⚠️", "fail": "❌"}.get(_iv_status, "ℹ️")
            st.markdown(
                callout_box(f"<b>Verdict:</b> {_iv_icon} {_iv_text}", _iv_status),
                unsafe_allow_html=True,
            )

            # Key metrics row
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                _align_text = "Aligned" if alignment_ok else f"{integrity.get('n_misaligned', 0)} misaligned"
                _align_st = "trust" if alignment_ok else "fail"
                st.markdown(
                    metric_card("Alignment", _align_text,
                                icon="✅" if alignment_ok else "❌", status=_align_st),
                    unsafe_allow_html=True,
                )
                if not alignment_ok and integrity.get("misaligned_examples"):
                    with st.expander("Misaligned examples"):
                        st.json(integrity["misaligned_examples"][:3])

            with col2:
                mae_model = integrity.get("mae_model", np.nan)
                st.markdown(
                    metric_card("Model MAE", _fmt_num(mae_model), icon="📐", status="neutral"),
                    unsafe_allow_html=True,
                )

            with col3:
                mae_persist = integrity.get("mae_persistence", np.nan)
                _bm_delta = ""
                if not (np.isnan(mae_persist) or np.isnan(mae_model)):
                    _bm_delta = f"Δ {_fmt_num(mae_persist - mae_model)}"
                st.markdown(
                    metric_card("Baseline MAE", _fmt_num(mae_persist), delta=_bm_delta,
                                icon="📏", status="neutral"),
                    unsafe_allow_html=True,
                )

            with col4:
                _sk_display = _fmt_pct(skill_pct, 2) if not np.isnan(skill_pct) else "N/A"
                _sk_st = "trust" if (not np.isnan(skill_pct) and skill_pct >= skill_threshold) else "fail"
                st.markdown(
                    metric_card("Skill %", _sk_display, icon="📊", status=_sk_st),
                    unsafe_allow_html=True,
                )

            # Run status
            run_status = integrity.get("run_status", "UNKNOWN")
            if run_status == "FAILED_QUALITY":
                st.markdown(
                    callout_box(
                        f"<b>Run Status: {run_status}</b> — Model does not beat persistence "
                        f"baseline at horizon {integrity.get('horizon', 'N/A')}.",
                        "fail", icon="⚠️",
                    ),
                    unsafe_allow_html=True,
                )
            elif run_status == "SUCCESS":
                st.markdown(
                    callout_box(f"<b>Run Status: {run_status}</b>", "trust", icon="✅"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    callout_box(f"<b>Run Status: {run_status}</b>", "info", icon="ℹ️"),
                    unsafe_allow_html=True,
                )

            # Detailed diagnostics
            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            st.markdown(section_header("Detailed Diagnostics"), unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Horizon", integrity.get("horizon", "N/A"))
                st.metric("Best Shift", integrity.get("best_shift", 0))
            with col2:
                rmse_model = integrity.get("rmse_model", np.nan)
                rmse_persist = integrity.get("rmse_persistence", np.nan)
                st.metric("Model RMSE", _fmt_num(rmse_model, 2))
                st.metric("Baseline RMSE", _fmt_num(rmse_persist, 2))
            with col3:
                r2_model = integrity.get("r2_model", np.nan)
                r2_persist = integrity.get("r2_persistence", np.nan)
                st.metric("Model R²", _fmt_num(r2_model, 3))
                st.metric("Baseline R²", _fmt_num(r2_persist, 3))

            # Checks
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            if integrity.get("lag_warning", False):
                st.markdown(
                    callout_box(
                        f"<b>Lag Warning:</b> Best error at shift={integrity.get('best_shift', 0)} "
                        "(should be 0). This suggests horizon misalignment.",
                        "fail", icon="🔴",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    callout_box("<b>Shift Check:</b> Best alignment at shift=0 (correct)", "trust", icon="✅"),
                    unsafe_allow_html=True,
                )

            if integrity.get("leakage_warning", False):
                st.markdown(
                    callout_box(
                        "<b>Leakage Warning:</b> Shuffled target performance suspiciously close "
                        "to normal. This may indicate data leakage.",
                        "fail", icon="🔴",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                shuffled_mae = integrity.get("mae_shuffled_target", np.nan)
                if not np.isnan(shuffled_mae):
                    st.markdown(
                        callout_box("<b>Leakage Check:</b> Passed (no leakage detected)", "trust", icon="✅"),
                        unsafe_allow_html=True,
                    )

            # Baseline comparison table
            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            st.markdown(section_header("Baseline Comparisons"), unsafe_allow_html=True)

            baseline_data = {
                "Metric": ["MAE", "RMSE", "R²"],
                "Model": [
                    _fmt_num(integrity.get("mae_model", np.nan)),
                    _fmt_num(integrity.get("rmse_model", np.nan), 2),
                    _fmt_num(integrity.get("r2_model", np.nan), 3),
                ],
                "Persistence Baseline": [
                    _fmt_num(integrity.get("mae_persistence", np.nan)),
                    _fmt_num(integrity.get("rmse_persistence", np.nan), 2),
                    _fmt_num(integrity.get("r2_persistence", np.nan), 3),
                ],
                "Seasonal Naive": [
                    _fmt_num(integrity.get("mae_seasonal_naive", np.nan)),
                    "N/A",
                    "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(baseline_data), use_container_width=True, hide_index=True)

            with st.expander("View full integrity report (JSON)"):
                st.json(integrity)

        except Exception as e:
            st.markdown(
                callout_box(f"Could not parse integrity report: {e}", "fail", icon="❌"),
                unsafe_allow_html=True,
            )
            import traceback
            st.code(traceback.format_exc())
    else:
        st.markdown(
            callout_box(
                "No integrity report found. This report is generated by ML pipelines "
                "and includes shift checks, baseline comparisons, and leakage tests.",
                "info", icon="ℹ️",
            ),
            unsafe_allow_html=True,
        )

# ── Tab: Feature Importance ──────────────────────────────────────────
with tab_feat_imp:
    st.markdown(
        section_header("Feature Importance", "Which input features matter most to the model?"),
        unsafe_allow_html=True,
    )

    fi_path = base_dir / "artifacts" / "feature_importance.csv"
    if not fi_path.exists():
        fi_path = out_root / "artifacts" / "feature_importance.csv"

    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        if fi_df.empty:
            st.markdown(callout_box("Feature importance file is empty.", "info", icon="ℹ️"), unsafe_allow_html=True)
        else:
            fi_models = sorted(fi_df["model"].unique())
            fi_model_sel = st.selectbox("Model", fi_models, index=0, key="fi_model_sel")

            fi_m = fi_df[fi_df["model"] == fi_model_sel].copy()
            fi_m = fi_m.sort_values("importance", ascending=False)

            top_n = st.slider("Show top N features", 5, min(50, len(fi_m)), min(20, len(fi_m)), key="fi_topn")
            fi_top = fi_m.head(top_n)

            fig_fi = px.bar(
                fi_top, x="importance", y="feature", orientation="h",
                title=f"Top {top_n} features — {fi_model_sel}",
                height=max(320, top_n * 24),
                color="importance", color_continuous_scale="Teal",
            )
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"), showlegend=False,
                plot_bgcolor="white",
                font=dict(family="Inter, -apple-system, sans-serif"),
            )
            st.plotly_chart(fig_fi, use_container_width=True, config={"displaylogo": False})

            with st.expander("Full feature importance table"):
                st.dataframe(fi_m[["feature", "importance"]], use_container_width=True, hide_index=True)

            st.download_button(
                "Download feature importance (CSV)",
                data=fi_df.to_csv(index=False).encode("utf-8"),
                file_name=f"feature_importance_{sel_run}.csv",
                key="fi_dl_btn",
            )
    else:
        st.markdown(
            callout_box(
                "No feature importance data found. Run a B-ML experiment to generate this.",
                "info", icon="ℹ️",
            ),
            unsafe_allow_html=True,
        )

# ── Tab: Ensemble ────────────────────────────────────────────────────
with tab_ensemble:
    st.markdown(
        section_header("Ensemble Builder", "Combine predictions from multiple runs"),
        unsafe_allow_html=True,
    )

    _all_runs = _list_runs()
    _run_names = [r.name for r in _all_runs]
    _ens_selected = st.multiselect(
        "Runs to combine", _run_names,
        default=[_run_names[0]] if _run_names else [],
        help="Select 2+ runs with compatible predictions.",
        key="ens_run_selector",
    )

    _ens_topk = st.number_input("Top-K for mean ensemble", 1, 20, 3, key="ens_topk",
                                help="How many best models to average.")

    if st.button("Build Ensemble", type="primary", key="ens_build_btn"):
        if len(_ens_selected) < 2:
            st.markdown(
                callout_box("Select at least 2 runs to create an ensemble.", "fail", icon="⚠️"),
                unsafe_allow_html=True,
            )
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
                ens_lb = result.get("leaderboard", pd.DataFrame())

                if ens_preds.empty:
                    st.markdown(
                        callout_box("No ensemble predictions generated. Check run compatibility.", "caution", icon="⚠️"),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        callout_box(
                            f"Ensemble built: <b>{len(ens_preds)}</b> predictions across "
                            f"<b>{ens_preds['model'].nunique()}</b> methods.",
                            "trust", icon="✅",
                        ),
                        unsafe_allow_html=True,
                    )

                    fig_ens = go.Figure()
                    _first_model = ens_preds["model"].unique()[0]
                    _base = ens_preds[ens_preds["model"] == _first_model].sort_values("date")
                    fig_ens.add_scatter(
                        x=pd.to_datetime(_base["date"]), y=_base["y_true"],
                        name="Actual", mode="lines", line=dict(color="#1e293b", width=2.5),
                    )
                    for em in ens_preds["model"].unique():
                        _eg = ens_preds[ens_preds["model"] == em].sort_values("date")
                        fig_ens.add_scatter(
                            x=pd.to_datetime(_eg["date"]), y=_eg["y_pred"],
                            name=em, mode="lines",
                        )
                    fig_ens.update_layout(
                        height=460, margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        plot_bgcolor="white",
                        font=dict(family="Inter, -apple-system, sans-serif"),
                    )
                    st.plotly_chart(fig_ens, use_container_width=True, config={"displaylogo": False})

                    if not ens_lb.empty:
                        st.markdown(section_header("Ensemble Leaderboard"), unsafe_allow_html=True)
                        st.dataframe(ens_lb, use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download ensemble predictions (CSV)",
                        data=ens_preds.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_ensemble.csv",
                        key="ens_dl_btn",
                    )
            except Exception as e:
                st.markdown(
                    callout_box(f"Ensemble failed: {e}", "fail", icon="❌"),
                    unsafe_allow_html=True,
                )
                import traceback
                st.code(traceback.format_exc())

# ── Tab: Downloads ───────────────────────────────────────────────────
with tab_downloads:
    st.markdown(
        section_header("Downloads", "Export filtered data and all generated plots"),
        unsafe_allow_html=True,
    )

    csv_bytes = df_t.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download filtered predictions (CSV)",
        data=csv_bytes,
        file_name=f"{sel_run}_filtered_predictions.csv",
        key="button12",
        use_container_width=True,
    )

    plots_dir = base_dir / "plots"
    if plots_dir.exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for img in sorted(plots_dir.glob("*.png")):
                zf.writestr(img.name, img.read_bytes())
        st.download_button(
            "📥 Download all static plots (ZIP)",
            data=buf.getvalue(),
            file_name=f"plots_{sel_run}_{choice_key}.zip",
            key="button15",
            use_container_width=True,
        )
    else:
        st.caption("No static plots found in this output folder.")
