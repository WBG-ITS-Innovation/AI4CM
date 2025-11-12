# pages/00_Lab.py — Lab (run models + live log + overlay)
from __future__ import annotations
import html, json, re, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend_bridge import launch_backend
from utils_frontend import load_paths, new_run_folders, UPLOADS_ROOT

st.set_page_config(page_title="🧪 Lab — Forecast Runner", layout="wide")

APPROOT = Path(__file__).resolve().parent
RUNS_DIR = APPROOT / "runs"; RUNS_DIR.mkdir(exist_ok=True)

def _scroll_term(container, text: str, height: int = 340):
    container.markdown(
        f"<div style='height:{height}px;overflow:auto;background:#0b0b0b;color:#e6e6e6; padding:8px; border-radius:8px;"
        "font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; white-space:pre;'>"
        f"{html.escape(text)}</div>", unsafe_allow_html=True)

def _baseline_series(out_root: Path, target: str, cadence: str) -> Optional[pd.Series]:
    cad = cadence.lower()
    def _read(p: Path):
        if not p.exists(): return None
        df = pd.read_csv(p)
        if not {"date","forecast"}.issubset(df.columns): return None
        s = pd.Series(df["forecast"].values, index=pd.to_datetime(df["date"], errors="coerce")).dropna()
        return s if not s.empty else None
    daily = _read(out_root / f"{target}_ops_baseline_daily.csv") or _read(out_root / cad / f"{target}_ops_baseline_daily.csv")
    if daily is not None:
        return daily if cad=="daily" else (daily.resample("W-FRI").sum() if cad=="weekly" else daily.resample("ME").sum())
    monthly = _read(out_root / f"{target}_ops_baseline_monthly.csv") or _read(out_root / cad / f"{target}_ops_baseline_monthly.csv")
    if monthly is not None:
        parts = []
        for ts, val in monthly.items():
            days = pd.date_range(ts.replace(day=1), ts, freq="B")
            if len(days): parts.append(pd.Series(float(val)/len(days), index=days))
        if parts:
            dd = pd.concat(parts).sort_index()
            return dd if cad=="daily" else (dd.resample("W-FRI").sum() if cad=="weekly" else monthly)
    return None

# Sidebar: auto-detected backend paths (no manual pasting)
st.sidebar.header("Backend (auto)")
paths = load_paths()
st.session_state["backend_py"]  = paths.get("backend_python","") or st.session_state.get("backend_py","")
st.session_state["backend_dir"] = paths.get("backend_dir","") or st.session_state.get("backend_dir","")
st.sidebar.caption(f"Python: `{st.session_state.get('backend_py','') or 'MISSING'}`")
st.sidebar.caption(f"Backend: `{st.session_state.get('backend_dir','') or 'MISSING'}`")

# 1) Data
st.title("🧪 Forecast Lab")
st.header("1) Data")
up = st.file_uploader("Upload CSV (must contain a `date` column + numeric target/exog)", type=["csv"])
if up:
    dest = UPLOADS_ROOT / "uploaded.csv"
    dest.write_bytes(up.read())
    st.success(f"Saved: `{dest}`")

with st.expander("Create a quick demo sample (optional)"):
    n = st.slider("Rows to keep (from end)", 200, 200_000, 2_000, step=200)
    if st.button("Create / Refresh sample from uploaded.csv"):
        src = UPLOADS_ROOT / "uploaded.csv"
        if not src.exists():
            st.warning("Please upload a CSV first.")
        else:
            pd.read_csv(src).tail(n).to_csv(UPLOADS_ROOT / "quick_sample.csv", index=False)
            st.info(f"Sample saved: `{(UPLOADS_ROOT/'quick_sample.csv').resolve()}`")

sources = []
if (UPLOADS_ROOT / "uploaded.csv").exists():
    sources.append(("Full uploaded dataset", str((UPLOADS_ROOT / "uploaded.csv").resolve())))
if (UPLOADS_ROOT / "quick_sample.csv").exists():
    sources.append(("Quick sample", str((UPLOADS_ROOT / "quick_sample.csv").resolve())))
if not sources:
    st.stop()

use_label = st.radio("Select data source", [lbl for lbl, _ in sources], index=0, horizontal=True)
data_path = Path(dict(sources)[use_label])
df = pd.read_csv(data_path)

with st.expander("Preview (first 200 rows) • parsed date span", expanded=True):
    head = df.head(200).copy()
    dcol = "date" if "date" in head.columns else head.columns[0]
    dtry = pd.to_datetime(head[dcol], errors="coerce")
    st.caption(f"Rows: {len(df):,} • date span: {dtry.min()} → {dtry.max()}")
    st.dataframe(head, use_container_width=True)

# 2) Configure
st.header("2) Configure")
L, R = st.columns([1.08, 0.92], gap="large")
with L:
    date_col = st.selectbox("Date column", list(df.columns), index=(list(df.columns).index("date") if "date" in df.columns else 0))
    num_cols = [c for c in df.columns if c != date_col]
    target = st.selectbox("Target", num_cols)
    cadence = st.selectbox("Cadence", ["Daily","Weekly","Monthly"], index=0)
    horizon = st.slider("Horizon", 1, 24, 6)
    with st.expander(f"Target preview — {target} @ {cadence}", expanded=True):
        tmp = df[[date_col, target]].dropna()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).set_index(date_col)
        s_prev = tmp[target]
        if cadence == "Weekly":   s_prev = s_prev.resample("W-FRI").sum()
        if cadence == "Monthly":  s_prev = s_prev.resample("ME").sum()
        st.plotly_chart(px.line(s_prev, title=f"{target} • {cadence} (preview)"),
                        use_container_width=True, config={"displaylogo": False})

with R:
    fam_label = st.selectbox("Family", ["A · Statistical","B · Machine Learning","C · Deep Learning","E · Quantile"])
    family = {"A · Statistical":"A_STAT","B · Machine Learning":"B_ML","C · Deep Learning":"C_DL","E · Quantile":"E_QUANTILE"}[fam_label]
    variant = "Univariate" if family == "A_STAT" else st.radio("Variant", ["Univariate","Multivariate"], horizontal=True)

st.subheader("Run profile")
profile = st.radio("Profile", ["Demo (fast)","Balanced","Thorough"], horizontal=True, index=0)
ov: Dict[str, Any] = {
    "folds": 0 if profile=="Demo (fast)" else (3 if profile=="Balanced" else 5),
    "min_train_years": 0 if profile=="Demo (fast)" else (2 if profile=="Balanced" else 4),
    "demo_clip_months": 12 if profile=="Demo (fast)" else None
}

if family == "B_ML":
    model = st.selectbox("Model (ML)", ["Ridge","Lasso","ElasticNet","RandomForest","ExtraTrees","HistGBDT","XGBoost","LightGBM"], index=0)
    if cadence == "Daily":
        ov["lags_daily"]    = st.multiselect("lags_daily", [1,2,3,5,7,14,21], default=[1,3,7])
        ov["windows_daily"] = st.multiselect("windows_daily", [3,5,7,14,21,28], default=[3,7])
    elif cadence == "Weekly":
        ov["lags_weekly"]   = st.multiselect("lags_weekly", [1,2,3,4,8,12,26], default=[1,4,12])
        ov["windows_weekly"]= st.multiselect("windows_weekly", [2,4,8,12,26], default=[4,12])
    else:
        ov["lags_monthly"]  = st.multiselect("lags_monthly", [1,2,3,6,12], default=[1,3])
        ov["windows_monthly"]= st.multiselect("windows_monthly", [3,6,12], default=[3,6])
    if variant == "Multivariate":
        ov["exog_top_k"] = st.number_input("exog_top_k", 0, 64, 8)
elif family == "A_STAT":
    model = st.selectbox("Model (Stat)", ["ETS","SARIMAX","STL_ARIMA","THETA","NAIVE","WEEKDAY_MEAN","MOVAVG"], index=0)
elif family == "C_DL":
    model = st.selectbox("Model (DL)", ["GRU","LSTM","TCN","Transformer","MLP"], index=0)
    ov["lookback"] = st.number_input("lookback", 4, 365, 48)
    ov["batch_size"] = st.number_input("batch_size", 8, 2048, 128, step=8)
    ov["max_epochs"] = st.number_input("max_epochs", 1, 200, 3)
    ov["valid_frac"] = st.number_input("valid_frac", 0.05, 0.9, 0.2, 0.05)
    ov["conformal_calib_frac"] = st.number_input("conformal_calib_frac", 0.05, 0.9, 0.2, 0.05)
    ov["device"] = st.selectbox("device", ["auto","cpu","cuda"], index=0)
else:
    model = st.selectbox("Model (Quantile)", ["GBQuantile"], index=0)
    q_text = st.text_input("quantiles (comma-separated)", "0.1,0.5,0.9")
    try: ov["quantiles"] = [float(x) for x in q_text.split(",") if x.strip()!=""]
    except Exception: ov["quantiles"] = [0.1,0.5,0.9]

st.subheader("Advanced overrides (JSON)")
ov_text = st.text_area("OVERRIDES_JSON", json.dumps(ov, indent=2), height=200)
try:
    ov_final: Dict[str, Any] = json.loads(ov_text)
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    ov_final = ov

# 3) Launch
st.header("3) Launch & live log")

# Build an INFORMATIVE run name so the Dashboard list stays readable
ts = datetime.now().strftime("%Y%m%d_%H%M")
short_fam = {"A_STAT":"A","B_ML":"B","C_DL":"C","E_QUANTILE":"E"}[family]
short_var = "uni" if variant=="Univariate" else "multi"
run_label = f"run_{short_fam}_{short_var}_{model}_{target}_{cadence}_h{int(horizon)}_{ts}"

run_id, run_dir, out_root = new_run_folders(run_label)

if family == "A_STAT":
    runner = Path(st.session_state["backend_dir"]) / "run_a_stat.py"
elif family == "B_ML":
    runner = Path(st.session_state["backend_dir"]) / ("run_b_ml_univariate.py" if short_var=="uni" else "run_b_ml_multivariate.py")
elif family == "C_DL":
    runner = Path(st.session_state["backend_dir"]) / ("run_c_dl_univariate.py" if short_var=="uni" else "run_c_dl_multivariate.py")
else:
    runner = Path(st.session_state["backend_dir"]) / ("run_e_quantile_daily_univariate.py" if short_var=="uni" else "run_e_quantile_daily_multivariate.py")

env = {
    "TG_FAMILY": family,
    "TG_MODEL_FILTER": model,
    "TG_TARGET": target,
    "TG_CADENCE": cadence,
    "TG_HORIZON": str(int(horizon)),
    "TG_DATA_PATH": str(Path(data_path).resolve()),
    "TG_DATE_COL": str([c for c in df.columns if c.lower()==date_col.lower()][0]),
    "TG_PARAM_OVERRIDES": json.dumps(ov_final),
    "TG_OUT_ROOT": str(out_root.resolve()),
}

log_box = st.empty()
status = st.empty()

def _on_progress(tail: str, elapsed: float):
    status.info(f"Elapsed: {elapsed:.1f}s")
    _scroll_term(log_box, tail)

if st.button("🚀 Run experiment", type="primary", use_container_width=True):
    py   = st.session_state.get("backend_py","")
    back = st.session_state.get("backend_dir","")
    if not py or not Path(py).exists():
        st.error("Backend Python missing/invalid. Create a .venv under TreasuryGeorgiaBackEnd.")
        st.stop()
    if not back or not Path(back).exists():
        st.error("Backend directory missing/invalid.")
        st.stop()

    rc, elapsed, log_path, out_real = launch_backend(
        backend_py=py, runner_script=str(runner), backend_dir=back,
        env_vars=env, run_dir=run_dir, on_progress=_on_progress,
    )

    if rc != 0:
        st.error(f"Run failed (rc={rc}). See log below.")
    else:
        st.success(f"Finished in {elapsed:.1f}s • outputs in `{out_real}`")
        # Overlay preview (collapsible)
        with st.expander("Overlay preview — Actual vs Models (and Ops baseline if available)", expanded=True):
            p = Path(out_real) / "predictions_long.csv"
            if not p.exists():
                for cad in ("daily","weekly","monthly"):
                    cand = Path(out_real) / cad / "predictions_long.csv"
                    if cand.exists(): p = cand; break
            if p.exists():
                pred = pd.read_csv(p)
                pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
                pred = pred.dropna(subset=["date"]).sort_values("date")
                models = sorted(pred["model"].unique())
                sel = st.multiselect("Models to visualize", models, default=models[:1])
                fig = go.Figure()
                fig.add_scatter(x=pred["date"], y=pred["y_true"], name="Actual", mode="lines", line=dict(color="black"))
                for m in sel:
                    g = pred[pred["model"] == m]
                    fig.add_scatter(x=g["date"], y=g["y_pred"], name=m, mode="lines")
                base = _baseline_series(Path(out_real), target, cadence)
                if base is not None and len(base):
                    rng = (pred["date"].min(), pred["date"].max())
                    b = base[(base.index >= rng[0]) & (base.index <= rng[1])]
                    if not b.empty:
                        fig.add_scatter(x=b.index, y=b.values, name="Ops baseline", mode="lines", line=dict(dash="dot"))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.caption("No predictions_long.csv found.")

    st.markdown("**Log tail**")
    st.code(Path(log_path).read_text(encoding="utf-8")[-4000:], language="text")
