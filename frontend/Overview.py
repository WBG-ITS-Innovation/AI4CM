# Overview.py — Landing / Overview (stable)
from __future__ import annotations
from pathlib import Path
import datetime as dt
import streamlit as st
import pandas as pd

from utils_frontend import load_paths, save_paths, list_runs, zip_outputs

st.set_page_config(page_title="Overview • Treasury Forecast Lab", page_icon="📊", layout="wide")

APPROOT = Path(__file__).resolve().parent
RUNS_DIR = APPROOT / "runs"

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.title("Georgia Treasury • Forecast Lab")
st.caption("A production-grade multi-model forecasting sandbox — Statistical • ML • DL • Quantile")

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1:
    st.page_link("pages/00_Lab.py", label="🧪 Open Lab", help="Configure a run and launch the backend with live logs.")
with c2:
    st.page_link("pages/01_Dashboard.py", label="📈 Open Dashboard", help="Explore Actual vs Baseline vs Predictions.")
with c3:
    st.page_link("pages/02_History.py", label="🕒 See History", help="Browse and download artifacts from past runs.")
with c4:
    st.page_link("pages/03_Models.py", label="📚 Read about Models", help="Deep guide to model families and parameters.")
with c5:
    st.page_link("pages/00_Data_Preprocessing.py", label="🧺 Data Pre-processing", help="Standardize and clean source data files.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# BACKEND PATHS (view + save)
# ─────────────────────────────────────────────────────────────
st.subheader("Backend paths (auto-detected)")
paths = load_paths()
bp_default = paths.get("backend_python","")
bd_default = paths.get("backend_dir","")

colp, cold, cols = st.columns([1.1, 1.1, 0.6])
with colp:
    backend_py = st.text_input("Python executable", value=bp_default,
        help=r"Example (Windows): C:\...\TreasuryGeorgiaBackEnd\.venv\Scripts\python.exe")
with cold:
    backend_dir = st.text_input("Backend directory", value=bd_default,
        help=r"Folder that contains run_a_stat.py, run_b_ml_*.py, run_c_dl_*.py, run_e_quantile_*.py")
with cols:
    if st.button("Save as default", key="ov_save_paths_btn"):
        save_paths(backend_py, backend_dir)
        st.success("Saved. All pages will reuse these paths.")

# ─────────────────────────────────────────────────────────────
# WHAT THIS PROTOTYPE DOES
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1])
with left:
    st.subheader("What this prototype is")
    st.markdown(
        """
- **End-to-end sandbox** to compare forecasting families on Treasury time series:
  **A** (Statistical), **B** (Machine Learning), **C** (Deep Learning), **E** (Quantile).
- Standard **outputs & plots** per run so results are easy to compare and share.
- **Lab → Dashboard → History** workflow:
  1) configure & run, 2) inspect accuracy and overlays, 3) download artifacts.
        """
    )
    st.subheader("How experiments work")
    st.markdown(
        """
1. **Lab** builds a config (target, cadence, horizon, overrides), then launches a family runner.  
2. **Backend** trains/validates with time-series CV and writes standard artifacts.  
3. **Dashboard** loads those artifacts and renders **Actual vs Treasury Baseline vs Predictions**.  
4. **History** keeps your runs (log + outputs) for later.
        """
    )
with right:
    st.subheader("Standard outputs per run")
    st.markdown(
        """
- `predictions_long.csv`  
- `metrics_long.csv`  
- `leaderboard.csv`  
- `plots/*` (overlay & extras)  
- `artifacts/*` (config, params)
        """
    )
    st.info("All files are under `runs/<run_id>/outputs[/cadence]/` (inside the **frontend** folder).")

# ─────────────────────────────────────────────────────────────
# QUICK START (no conda required)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Quick Start (no conda required)")
tabs = st.tabs(["Windows (PowerShell)", "macOS / Linux (bash)"])
with tabs[0]:
    st.code(
        """\
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r TreasuryGeorgiaBackEnd/requirements.txt
pip install -r frontend/requirements.txt  # if you keep a separate list for Streamlit
streamlit run Overview.py
"""
    )
with tabs[1]:
    st.code(
        """\
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r TreasuryGeorgiaBackEnd/requirements.txt
pip install -r frontend/requirements.txt  # if you keep a separate list for Streamlit
streamlit run Overview.py
"""
    )

# ─────────────────────────────────────────────────────────────
# RECENT RUNS (latest → oldest)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Recent runs")
runs = list_runs()
if not runs:
    st.caption("No runs yet — use **Open Lab** above to start your first experiment.")
else:
    # show up to 5 latest runs as compact cards
    for idx, run in enumerate(runs[:5], start=1):
        out_dir = run / "outputs"
        log_file = run / "backend_run.log"
        ts = dt.datetime.fromtimestamp(run.stat().st_mtime)
        with st.container(border=True):
            top = st.columns([4, 2, 2, 2])
            with top[0]:
                st.markdown(f"**Run:** `{run.name}`")
                st.caption(ts.strftime("Finished: %Y-%m-%d %H:%M"))
                if log_file.exists():
                    st.code(log_file.read_text(encoding="utf-8")[-800:], language="text")
                else:
                    st.caption("_No log found for this run._")
            with top[1]:
                p = out_dir / "predictions_long.csv"
                st.markdown("**predictions_long.csv**" + (" ✅" if p.exists() else " —"))
                st.page_link("pages/01_Dashboard.py", label="➡️ View in Dashboard")
            with top[2]:
                m = out_dir / "metrics_long.csv"
                st.markdown("**metrics_long.csv**" + (" ✅" if m.exists() else " —"))
                if m.exists():
                    st.download_button("Download", data=m.read_bytes(), file_name="metrics_long.csv",
                                       use_container_width=True, key=f"ov_dl_metrics_{idx}")
            with top[3]:
                if out_dir.exists():
                    st.download_button("⬇️ All artifacts (.zip)", data=zip_outputs(out_dir),
                                       file_name=f"{run.name}_artifacts.zip",
                                       use_container_width=True, key=f"ov_dl_zip_{idx}")
                else:
                    st.caption("No artifacts folder yet.")

# ─────────────────────────────────────────────────────────────
# HOW TO READ RESULTS (brief)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("How to read results")
col_a, col_b = st.columns([1,1])
with col_a:
    st.markdown(
        """
**Dashboard overlays**  
- Compare **Actual**, **Treasury Baseline**, and **Prediction**; resample to weekly/monthly for display.  
- Use the **Model selector** to switch; optional **PI band** if available.
"""
    )
with col_b:
    st.markdown(
        """
**History**  
- Every run is stored under `runs/<run_id>/` with its **log**, **outputs**, and **plots**.  
- You can download the whole run as a **ZIP** from this page or from **History**.
"""
    )

st.markdown("---")
st.caption("Need a refresher on models and parameters? Open **Models** for defaults and tuning tips.")
