# Overview.py — Landing / Overview (stable)
from __future__ import annotations
from pathlib import Path
import datetime as dt
import re
import streamlit as st
import pandas as pd

from utils_frontend import load_paths, save_paths, list_runs, zip_outputs

try:
    from ui_styles import inject_global_css, page_header
except ImportError:
    def inject_global_css(): pass
    def page_header(t, s=""): return f"<h1>{t}</h1><p>{s}</p>"

from ui_styles import inject_design_system  # presentation only
from i18n import install as install_language  # language toggle + pending-review note
from ui_styles import page_intro  # the one-or-two-sentence intro every page opens with
from ui_styles import render_app_header  # presentation only
from ui_styles import render_brand  # the one brand header, in the sidebar
st.set_page_config(page_title="Overview · Treasury Forecast", layout="wide")
inject_global_css()

inject_design_system()

# The language toggle and, in Georgian, the standing note that the translation has
# not been reviewed by a native speaker. One call per page; everything else the
# reader sees is translated inside the shared helpers.
render_brand()
install_language()

render_app_header("Overview", "What this lab does, and what it does not claim")
page_intro(
    "This is the landing page. It confirms the Lab can find its backend, shows what has "
    "been run so far, and states plainly what this project does and does not claim."
)
APPROOT = Path(__file__).resolve().parent
from paths import runs_dir
RUNS_DIR = runs_dir()

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────

# The guide comes first and is called out on its own, because the row of six links below it
# was the whole of the app's navigation and none of the labels tells a new reader which page
# answers their question.
st.info(
    "**New here?** The Start here page says what each page is for, what you can do there, and "
    "one thing to try, and it explains the difference between an official forecast and an "
    "experiment. It is the shortest way in."
)
st.page_link("pages/01_Start_here.py", label="Start here: the guide to this Lab")
st.write("")

c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1,1])
with c1:
    st.page_link("pages/07_Forecast.py", label="Forecast", help="The forecast for the next working days, and the evidence behind it.")
with c2:
    st.page_link("pages/08_Scorecard.py", label="Scorecard", help="How past published forecasts actually did, and where new actuals are uploaded.")
with c3:
    st.page_link("pages/03_Lab.py", label="Lab", help="Configure a run and launch the backend with live logs.")
with c4:
    st.page_link("pages/04_Dashboard.py", label="Dashboard", help="Explore Actual vs Baseline vs Predictions.")
with c5:
    st.page_link("pages/05_Compare.py", label="Compare runs", help="Side-by-side comparison of 2 to 6 runs.")
with c6:
    st.page_link("pages/06_History.py", label="History", help="Browse and download artifacts from past runs.")
with c7:
    st.page_link("pages/09_Documentation.py", label="Documentation", help="Every model on the shelf, and whether anyone has measured it.")
st.page_link("pages/02_Data_Preprocessing.py", label="Data pre-processing", help="Standardise and clean source data files.")

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
        help=r"Example on Windows: C:\Projects\AI4CM\backend\.venv\Scripts\python.exe")
with cold:
    backend_dir = st.text_input("Backend directory", value=bd_default,
        help=r"Folder that contains the runner scripts, such as run_a_stat.py and run_b_ml_univariate.py")
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
- **Run profiles** (Demo/Balanced/Thorough) with family-specific tuning.
- **Batch runs**: run ALL models and/or multiple horizons in one session.
- **Data quality pre-flight** checks before every run.
- **Ensemble builder**: combine 2+ runs into optimized ensembles.
- **Cross-run comparison**: side-by-side metrics, overlays, winner podium.
- **Accuracy scorecard**: letter grade (A-F), tips, and next steps.
        """
    )
    st.subheader("How experiments work")
    st.markdown(
        """
1. **Lab** validates data quality, then launches backend with your config.
2. **Backend** trains/validates with time-series CV and writes standard artifacts.
3. **Dashboard** renders overlays, metrics, feature importance, and a scorecard.
4. **Compare** lets you pick 2-6 runs for side-by-side analysis.
5. **History** keeps your runs (log + outputs) for later.
        """
    )
with right:
    st.subheader("Standard outputs per run")
    st.markdown(
        """
- `predictions_long.csv` (with prediction intervals)
- `metrics_long.csv`
- `leaderboard.csv`
- `plots/*` (overlay & extras)
- `artifacts/*` (config, integrity report, feature importance)
        """
    )
    st.info("All files are under `runs/<run_id>/outputs[/cadence]/` (inside the **frontend** folder).")

    st.subheader("Key capabilities")
    st.markdown(
        """
| Feature | Status |
|---|---|
| Statistical PIs (ETS/SARIMAX) | Yes |
| Conformal PIs (ML) | Yes |
| Ensemble (median/top-K/weighted) | Yes |
| Feature importance (ML) | Yes |
| Quality gate (5% skill) | Yes |
| Data pre-flight checks | Yes |
| Multi-horizon batch | Yes |
        """
    )

# ─────────────────────────────────────────────────────────────
# QUICK START (no conda required)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Quick Start (no conda required)")
tabs = st.tabs(["Windows (PowerShell)", "macOS / Linux (bash)"])
with tabs[0]:
    st.code(
        r"""\
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
    st.caption("No runs yet. Use **Open Lab** above to start your first experiment.")
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
                st.markdown("**predictions_long.csv**" + (" written" if p.exists() else " (not written)"))
                st.page_link("pages/04_Dashboard.py", label="View in Dashboard")
            with top[2]:
                m = out_dir / "metrics_long.csv"
                st.markdown("**metrics_long.csv**" + (" written" if m.exists() else " (not written)"))
                if m.exists():
                    st.download_button("Download", data=m.read_bytes(), file_name="metrics_long.csv",
                                       use_container_width=True, key=f"ov_dl_metrics_{idx}")
            with top[3]:
                if out_dir.exists():
                    st.download_button("All artifacts (.zip)", data=zip_outputs(out_dir),
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

# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS SINCE THE LAST REVIEW  (Part 8)
#
# A SECTION on this page, not a new page and not a nav item. Reads
# reports/PROGRESS_SINCE_LAST_REVIEW.md so the lab and the written record cannot drift: there
# is one source and this renders it.
# ══════════════════════════════════════════════════════════════════════════════
_progress = REPOROOT_PROGRESS = (Path(__file__).resolve().parent.parent
                                 / "reports" / "PROGRESS_SINCE_LAST_REVIEW.md")
st.divider()
st.markdown("## Progress since the last review")
if _progress.exists():
    _txt = _progress.read_text(encoding="utf-8")
    # Drop the H1 so it does not compete with the section heading above.
    _body = re.sub(r"^#\s+.*?$", "", _txt, count=1, flags=re.M).lstrip()
    _head, _sep, _rest = _body.partition("## 2 ·")
    st.markdown(_head)
    with st.expander("The rest of the record: levers, model pool, correctness work, "
                     "and what we still cannot claim", expanded=False):
        st.markdown(_sep + _rest if _sep else _rest)
    st.caption(f"Rendered from `reports/PROGRESS_SINCE_LAST_REVIEW.md` "
               f"(updated {pd.Timestamp(_progress.stat().st_mtime, unit='s').date()}). "
               f"Every figure in it traces to a row in `experiments/log.csv`.")
else:
    st.info("No progress record found. Expected `reports/PROGRESS_SINCE_LAST_REVIEW.md`.")
