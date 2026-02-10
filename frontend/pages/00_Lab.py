# pages/00_Lab.py — Lab (run models + live log + overlay + REAL hover help + batch runs)
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend_bridge import launch_backend
from utils_frontend import load_paths, new_run_folders, UPLOADS_ROOT

st.set_page_config(page_title="🧪 Lab — Forecast Runner", layout="wide")

APPROOT = Path(__file__).resolve().parent
RUNS_DIR = APPROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)


# ---------------------------- small UI helpers ----------------------------
def _scroll_term(container, text: str, height: int = 340):
    """Scrollable terminal-style log box."""
    container.markdown(
        f"<div style='height:{height}px;overflow:auto;background:#0b0b0b;color:#e6e6e6; padding:8px; border-radius:8px;"
        "font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; white-space:pre;'>"
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _baseline_series(out_root: Path, target: str, cadence: str) -> Optional[pd.Series]:
    """
    Tries to load Ops baseline forecast series if present.
    Supports both daily baseline and monthly baseline that can be spread over business days.
    """
    cad = cadence.lower()

    def _read(p: Path):
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if not {"date", "forecast"}.issubset(df.columns):
            return None
        s = pd.Series(df["forecast"].values, index=pd.to_datetime(df["date"], errors="coerce")).dropna()
        return s if not s.empty else None

    daily = _read(out_root / f"{target}_ops_baseline_daily.csv") or _read(out_root / cad / f"{target}_ops_baseline_daily.csv")
    if daily is not None:
        return daily if cad == "daily" else (daily.resample("W-FRI").sum() if cad == "weekly" else daily.resample("ME").sum())

    monthly = _read(out_root / f"{target}_ops_baseline_monthly.csv") or _read(out_root / cad / f"{target}_ops_baseline_monthly.csv")
    if monthly is not None:
        parts = []
        for ts, val in monthly.items():
            days = pd.date_range(ts.replace(day=1), ts, freq="B")
            if len(days):
                parts.append(pd.Series(float(val) / len(days), index=days))
        if parts:
            dd = pd.concat(parts).sort_index()
            return dd if cad == "daily" else (dd.resample("W-FRI").sum() if cad == "weekly" else monthly)

    return None


def _safe_date_col(df_cols: List[str], selected: str) -> str:
    """Return exact column name in df that matches selected (case-insensitive)."""
    for c in df_cols:
        if c.lower() == selected.lower():
            return c
    return selected


def _read_predictions_any(out_real: Path) -> Optional[pd.DataFrame]:
    """
    Find predictions_long.csv in standard locations (root or cadence folder).
    Returns parsed dataframe with date parsed if found.
    """
    candidates = [
        out_real / "predictions_long.csv",
        out_real / "daily" / "predictions_long.csv",
        out_real / "weekly" / "predictions_long.csv",
        out_real / "monthly" / "predictions_long.csv",
    ]
    for p in candidates:
        if p.exists():
            pred = pd.read_csv(p)
            if "date" in pred.columns:
                pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
                pred = pred.dropna(subset=["date"]).sort_values("date")
            return pred
    return None


# ---------------------------- Sidebar: backend auto paths ----------------------------
st.sidebar.header("Backend (auto)")
paths = load_paths()
st.session_state["backend_py"] = paths.get("backend_python", "") or st.session_state.get("backend_py", "")
st.session_state["backend_dir"] = paths.get("backend_dir", "") or st.session_state.get("backend_dir", "")
st.sidebar.caption(f"Python: `{st.session_state.get('backend_py','') or 'MISSING'}`")
st.sidebar.caption(f"Backend: `{st.session_state.get('backend_dir','') or 'MISSING'}`")


# ---------------------------- Page intro ----------------------------
st.title("🧪 Forecast Lab")
st.write(
    """
This page is where you run forecasting experiments.

**What happens when you click Run:**
1) We take your dataset + configuration (target, cadence, horizon, model parameters)  
2) We launch a backend runner script using the backend virtual environment  
3) The run produces logs + outputs (predictions, metrics, plots) saved under `frontend/runs/`  
4) You can compare runs later in **Dashboard** and **History**
"""
)

with st.expander("How to use this tab (quick overview)", expanded=False):
    st.markdown(
        """
**Typical workflow**
- Upload your CSV (or use the quick sample)
- Choose a target and cadence
- Pick a model family + model
- (Optional) adjust parameters (lags/lookback/epochs etc.)
- Run the experiment and review the overlay + logs

**Important note**
- The Lab runs do **not** overwrite each other. Every run is saved with a timestamped folder.
"""
    )


# ---------------------------- 1) Data ----------------------------
st.header("1) Data")

up = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help=(
        "Upload a CSV file that includes a date column and at least one numeric column to forecast.\n\n"
        "Minimum requirement:\n"
        "- one column representing dates (usually named 'date')\n"
        "- one numeric column (your target)\n"
        "- optional additional numeric columns (exogenous inputs for multivariate ML/DL)\n"
    ),
)

if up:
    dest = UPLOADS_ROOT / "uploaded.csv"
    dest.write_bytes(up.read())
    st.success(f"Saved: `{dest}`")

with st.expander("Create a quick sample from the uploaded CSV (optional)", expanded=False):
    st.write(
        "This creates a smaller dataset (last N rows) so demo runs finish quickly while keeping recent behavior."
    )
    n = st.slider(
        "Rows to keep (from end)",
        200,
        200_000,
        2_000,
        step=200,
        help="Keeps only the most recent N rows from uploaded.csv and saves as quick_sample.csv.",
    )
    if st.button("Create / Refresh sample", help="Creates frontend/runs_uploads/quick_sample.csv"):
        src = UPLOADS_ROOT / "uploaded.csv"
        if not src.exists():
            st.warning("Please upload a CSV first.")
        else:
            pd.read_csv(src).tail(n).to_csv(UPLOADS_ROOT / "quick_sample.csv", index=False)
            st.info(f"Sample saved: `{(UPLOADS_ROOT / 'quick_sample.csv').resolve()}`")

sources = []
if (UPLOADS_ROOT / "uploaded.csv").exists():
    sources.append(("Full uploaded dataset", str((UPLOADS_ROOT / "uploaded.csv").resolve())))
if (UPLOADS_ROOT / "quick_sample.csv").exists():
    sources.append(("Quick sample", str((UPLOADS_ROOT / "quick_sample.csv").resolve())))

if not sources:
    st.info("Upload a CSV to begin (or generate quick_sample.csv).")
    st.stop()

use_label = st.radio(
    "Select data source",
    [lbl for lbl, _ in sources],
    index=0,
    horizontal=True,
    help="Choose whether to run on the full dataset or the smaller quick sample.",
)
data_path = Path(dict(sources)[use_label])
df = pd.read_csv(data_path)

with st.expander("Preview data (first 200 rows)", expanded=True):
    head = df.head(200).copy()
    dcol_guess = "date" if "date" in head.columns else head.columns[0]
    dtry = pd.to_datetime(head[dcol_guess], errors="coerce")
    st.caption(f"Rows: {len(df):,} • date span (first 200 rows): {dtry.min()} → {dtry.max()}")
    st.dataframe(head, use_container_width=True)


# ---------------------------- 2) Configure ----------------------------
st.header("2) Configure")

L, R = st.columns([1.08, 0.92], gap="large")

with L:
    cols = list(df.columns)

    date_col = st.selectbox(
        "Date column",
        cols,
        index=(cols.index("date") if "date" in cols else 0),
        help="Which column contains the dates. This will be parsed into a time index.",
    )

    num_cols = [c for c in cols if c != date_col]
    target = st.selectbox(
        "Target (what we forecast)",
        num_cols,
        help=(
            "The numeric series the model will predict.\n\n"
            "Examples: daily revenue flow, daily expense flow, a balance level, etc."
        ),
    )

    cadence = st.selectbox(
        "Cadence (aggregation level)",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help=(
            "How we aggregate the data before training/forecasting.\n\n"
            "- Daily: uses each day\n"
            "- Weekly: resamples to weekly totals (W-FRI)\n"
            "- Monthly: resamples to monthly totals (month end)\n"
        ),
    )

    horizon = st.slider(
        "Horizon (how far ahead to predict)",
        1,
        24,
        6,
        help=(
            "Number of future periods to forecast.\n\n"
            "Examples:\n"
            "- Daily horizon=7 means forecasting the next 7 days\n"
            "- Monthly horizon=6 means forecasting the next 6 months"
        ),
    )

    with st.expander("Target preview chart", expanded=True):
        st.write(
            "This chart shows the selected target series after applying the cadence aggregation."
        )
        tmp = df[[date_col, target]].dropna()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).set_index(date_col)
        s_prev = tmp[target]

        if cadence == "Weekly":
            s_prev = s_prev.resample("W-FRI").sum()
        if cadence == "Monthly":
            s_prev = s_prev.resample("ME").sum()

        st.plotly_chart(
            px.line(s_prev, title=f"{target} • {cadence} (preview)"),
            use_container_width=True,
            config={"displaylogo": False},
        )

with R:
    fam_label = st.selectbox(
        "Model family",
        ["A · Statistical", "B · Machine Learning", "C · Deep Learning", "E · Quantile"],
        help=(
            "Choose the type of forecasting approach.\n\n"
            "A · Statistical: traditional time-series models (ETS/SARIMAX/etc.)\n"
            "B · ML: feature-based models using lags/rolling windows\n"
            "C · DL: neural networks over time windows (LSTM/GRU/TCN/etc.)\n"
            "E · Quantile: predicts uncertainty ranges (P10/P50/P90)"
        ),
    )

    family = {
        "A · Statistical": "A_STAT",
        "B · Machine Learning": "B_ML",
        "C · Deep Learning": "C_DL",
        "E · Quantile": "E_QUANTILE",
    }[fam_label]

    variant = "Univariate" if family == "A_STAT" else st.radio(
        "Variant",
        ["Univariate", "Multivariate"],
        horizontal=True,
        help=(
            "Univariate: uses only the target series history.\n"
            "Multivariate: also uses additional numeric columns as exogenous signals (if supported by the runner)."
        ),
    )

st.subheader("Run profile")

profile = st.radio(
    "Profile",
    ["Demo (fast)", "Balanced", "Thorough"],
    horizontal=True,
    index=0,
    help=(
        "Controls how heavy the run is.\n\n"
        "Demo (fast): minimal training / minimal cross-validation\n"
        "Balanced: reasonable default for exploration\n"
        "Thorough: slower, more robust evaluation"
    ),
)

# base overrides from profile
ov: Dict[str, Any] = {
    "folds": 1 if profile == "Demo (fast)" else (3 if profile == "Balanced" else 5),  # Thorough = 5 folds
    "min_train_years": 0 if profile == "Demo (fast)" else (2 if profile == "Balanced" else 4),
    "demo_clip_months": 12 if profile == "Demo (fast)" else None,
}


# ---------------------------- model selection (+ batch option) ----------------------------
st.subheader("Model selection")

run_multiple = st.checkbox(
    "Run multiple models (batch)",
    value=False,
    help=(
        "If enabled, you can select multiple models and the app will run them one-by-one automatically.\n\n"
        "This is the safest way to compare multiple models without changing backend code."
    ),
)

model_options: List[str]
if family == "B_ML":
    model_options = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "ExtraTrees", "HistGBDT", "XGBoost", "LightGBM"]
elif family == "A_STAT":
    model_options = ["ETS", "SARIMAX", "STL_ARIMA", "THETA", "NAIVE", "WEEKDAY_MEAN", "MOVAVG"]
elif family == "C_DL":
    model_options = ["GRU", "LSTM", "TCN", "Transformer", "MLP"]
else:
    model_options = ["GBQuantile"]

# choose one vs many
if run_multiple and len(model_options) > 1:
    models_selected = st.multiselect(
        "Models to run",
        model_options,
        default=model_options[:2],
        help="Select multiple models. Each will create its own run folder and outputs.",
    )
    if not models_selected:
        st.warning("Select at least one model (or disable batch mode).")
        st.stop()
else:
    model_single = st.selectbox(
        "Model",
        model_options,
        index=0,
        help="Select the model to run for this experiment.",
    )
    models_selected = [model_single]


# ---------------------------- parameter widgets (with REAL hover help) ----------------------------
st.subheader("Model parameters")

if family == "B_ML":
    # Lag/window parameter names vary with cadence in your backend
    if cadence == "Daily":
        ov["lags_daily"] = st.multiselect(
            "lags_daily",
            [1, 2, 3, 5, 7, 14, 21],
            default=[1, 3, 7],
            help=(
                "Which past days to include as features.\n\n"
                "Example: [1,3,7] means we use yesterday, 3 days ago, and 1 week ago as inputs."
            ),
        )
        ov["windows_daily"] = st.multiselect(
            "windows_daily",
            [3, 5, 7, 14, 21, 28],
            default=[3, 7],
            help=(
                "Rolling window statistics (e.g., trailing mean) used as features.\n\n"
                "Example: 7 means a trailing 7-day average."
            ),
        )
    elif cadence == "Weekly":
        ov["lags_weekly"] = st.multiselect(
            "lags_weekly",
            [1, 2, 3, 4, 8, 12, 26],
            default=[1, 4, 12],
            help="Past weeks used as input features (e.g., 1 week ago, 4 weeks ago, 12 weeks ago).",
        )
        ov["windows_weekly"] = st.multiselect(
            "windows_weekly",
            [2, 4, 8, 12, 26],
            default=[4, 12],
            help="Rolling window sizes in weeks used for trailing stats (mean/std etc.).",
        )
    else:
        ov["lags_monthly"] = st.multiselect(
            "lags_monthly",
            [1, 2, 3, 6, 12],
            default=[1, 3],
            help="Past months used as input features (1 month ago, 3 months ago, etc.).",
        )
        ov["windows_monthly"] = st.multiselect(
            "windows_monthly",
            [3, 6, 12],
            default=[3, 6],
            help="Rolling window sizes in months used for trailing stats (mean/std etc.).",
        )

    if variant == "Multivariate":
        ov["exog_top_k"] = st.number_input(
            "exog_top_k",
            0,
            64,
            8,
            help=(
                "In multivariate mode, we can select the most useful external signals.\n\n"
                "exog_top_k = how many exogenous columns to keep (based on importance/selection in the backend)."
            ),
        )

elif family == "C_DL":
    ov["lookback"] = st.number_input(
        "lookback",
        4,
        365,
        48,
        help=(
            "How many past time steps the network sees to make a forecast.\n\n"
            "Daily example: lookback=60 means the model uses the last 60 days to predict the next horizon."
        ),
    )
    ov["batch_size"] = st.number_input(
        "batch_size",
        8,
        2048,
        128,
        step=8,
        help="Training batch size. Larger batches can be faster but use more memory.",
    )
    ov["max_epochs"] = st.number_input(
        "max_epochs",
        1,
        200,
        3,
        help="Maximum training epochs (passes through training data). More epochs = slower but can improve fit.",
    )
    ov["valid_frac"] = st.number_input(
        "valid_frac",
        0.05,
        0.9,
        0.2,
        0.05,
        help="Fraction of data held out as validation (used to monitor training quality).",
    )
    ov["conformal_calib_frac"] = st.number_input(
        "conformal_calib_frac",
        0.05,
        0.9,
        0.2,
        0.05,
        help="Fraction used to calibrate prediction intervals (uncertainty bands) using conformal prediction.",
    )
    ov["device"] = st.selectbox(
        "device",
        ["auto", "cpu", "cuda"],
        index=0,
        help="Training device. 'auto' chooses best available; 'cuda' requires a CUDA GPU.",
    )

elif family == "E_QUANTILE":
    q_text = st.text_input(
        "quantiles (comma-separated)",
        "0.1,0.5,0.9",
        help="Quantiles to predict. 0.1=lower bound (P10), 0.5=median (P50), 0.9=upper bound (P90).",
    )
    try:
        ov["quantiles"] = [float(x) for x in q_text.split(",") if x.strip() != ""]
    except Exception:
        ov["quantiles"] = [0.1, 0.5, 0.9]
        st.warning("Could not parse quantiles; using default [0.1, 0.5, 0.9].")

else:
    # A_STAT has no extra widget knobs here because actual parameters are driven by backend
    st.caption(
        "Statistical models are configured primarily in the backend; use Advanced overrides below if you want to pass extra knobs."
    )


# ---------------------------- Advanced overrides ----------------------------
st.subheader("Advanced overrides (JSON)")

st.write(
    "This is the exact JSON passed to the backend runner. You can edit it if you want to force specific settings."
)

ov_text = st.text_area(
    "OVERRIDES_JSON",
    json.dumps(ov, indent=2),
    height=200,
    help=(
        "Advanced users only. This JSON is passed to the backend as TG_PARAM_OVERRIDES.\n\n"
        "If you edit this, keep valid JSON format."
    ),
)

try:
    ov_final: Dict[str, Any] = json.loads(ov_text)
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    ov_final = ov


# ---------------------------- 3) Launch & live log ----------------------------
st.header("3) Launch & live log")

st.write(
    """
When you run an experiment, the Lab creates a new run folder and launches the backend runner script.
You will see the live backend log below.
"""
)

log_box = st.empty()
status = st.empty()

def _on_progress(tail: str, elapsed: float):
    status.info(f"Elapsed: {elapsed:.1f}s")
    _scroll_term(log_box, tail)

def _runner_for(family: str, short_var: str) -> Path:
    back = Path(st.session_state["backend_dir"])
    if family == "A_STAT":
        return back / "run_a_stat.py"
    if family == "B_ML":
        return back / ("run_b_ml_univariate.py" if short_var == "uni" else "run_b_ml_multivariate.py")
    if family == "C_DL":
        return back / ("run_c_dl_univariate.py" if short_var == "uni" else "run_c_dl_multivariate.py")
    # E_QUANTILE
    return back / ("run_e_quantile_daily_univariate.py" if short_var == "uni" else "run_e_quantile_daily_multivariate.py")

def _run_one(model_name: str):
    """Run a single model experiment and return result dict."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    short_fam = {"A_STAT": "A", "B_ML": "B", "C_DL": "C", "E_QUANTILE": "E"}[family]
    short_var = "uni" if variant == "Univariate" else "multi"
    run_label = f"run_{short_fam}_{short_var}_{model_name}_{target}_{cadence}_h{int(horizon)}_{ts}"

    run_id, run_dir, out_root = new_run_folders(run_label)
    runner = _runner_for(family, short_var)

    env = {
        "TG_FAMILY": family,
        "TG_MODEL_FILTER": model_name,
        "TG_TARGET": target,
        "TG_CADENCE": cadence,
        "TG_HORIZON": str(int(horizon)),
        "TG_DATA_PATH": str(Path(data_path).resolve()),
        "TG_DATE_COL": str(_safe_date_col(list(df.columns), date_col)),
        "TG_PARAM_OVERRIDES": json.dumps(ov_final),
        "TG_OUT_ROOT": str(out_root.resolve()),
    }

    rc, elapsed, log_path, out_real = launch_backend(
        backend_py=st.session_state["backend_py"],
        runner_script=str(runner),
        backend_dir=st.session_state["backend_dir"],
        env_vars=env,
        run_dir=run_dir,
        on_progress=_on_progress,
    )

    return {
        "model": model_name,
        "rc": rc,
        "elapsed": elapsed,
        "log_path": log_path,
        "out_real": out_real,
        "run_label": run_label,
    }


if st.button("🚀 Run experiment(s)", type="primary", use_container_width=True):
    py = st.session_state.get("backend_py", "")
    back = st.session_state.get("backend_dir", "")

    if not py or not Path(py).exists():
        st.error("Backend Python missing/invalid. Create backend/.venv via scripts/setup_* and ensure .tg_paths.json is set.")
        st.stop()
    if not back or not Path(back).exists():
        st.error("Backend directory missing/invalid.")
        st.stop()

    results = []
    prog = st.progress(0, text="Starting…")

    for i, m in enumerate(models_selected, start=1):
        prog.progress((i - 1) / max(1, len(models_selected)), text=f"Running {m} ({i}/{len(models_selected)})…")
        res = _run_one(m)
        results.append(res)

        # ✅ FIX 5: Check for failure: return code != 0 OR error.json exists OR FAILED_QUALITY
        error_json_path = Path(res["out_real"]) / "artifacts" / "error.json"
        integrity_json_path = Path(res["out_real"]) / "artifacts" / "integrity_report.json"
        has_error_file = error_json_path.exists()
        
        # Check integrity report for FAILED_QUALITY status
        run_status = None
        if integrity_json_path.exists():
            try:
                integrity = json.loads(integrity_json_path.read_text(encoding="utf-8"))
                run_status = integrity.get("run_status", "UNKNOWN")
            except Exception:
                pass
        
        if res["rc"] != 0 or has_error_file:
            status_msg = f"❌ {m} failed"
            if res["rc"] != 0:
                status_msg += f" (rc={res['rc']})"
            if has_error_file:
                status_msg += " (error.json found)"
            st.error(status_msg)
        elif run_status == "FAILED_QUALITY":
            # ✅ FIX 5: Show FAILED_QUALITY status (outputs still written, but quality gate failed)
            st.warning(f"⚠️ {m} finished but FAILED_QUALITY — Model does not beat persistence baseline. "
                      f"Outputs written to `{res['out_real']}` (elapsed: {res['elapsed']:.1f}s)")
        else:
            st.success(f"✅ {m} finished in {res['elapsed']:.1f}s • outputs in `{res['out_real']}`")

    prog.progress(1.0, text="Done.")

    # Summary table
    st.subheader("Run summary")
    summary_rows = []
    for r in results:
        error_json_path = Path(r["out_real"]) / "artifacts" / "error.json"
        integrity_json_path = Path(r["out_real"]) / "artifacts" / "integrity_report.json"
        has_error = error_json_path.exists()
        run_status = None
        if integrity_json_path.exists():
            try:
                integrity = json.loads(integrity_json_path.read_text(encoding="utf-8"))
                run_status = integrity.get("run_status", "UNKNOWN")
            except Exception:
                pass
        
        if r["rc"] != 0 or has_error:
            status = f"FAIL (rc={r['rc']})"
        elif run_status == "FAILED_QUALITY":
            status = "FAILED_QUALITY"  # ✅ FIX 5: Show quality gate failure
        else:
            status = "OK"
        
        summary_rows.append({
            "model": r["model"],
            "status": status,
            "elapsed_sec": round(float(r["elapsed"]), 2),
            "run_label": r["run_label"],
            "out_dir": r["out_real"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Overlay comparison across runs
    st.subheader("Overlay preview — compare models")
    # ✅ FIX 5: Include FAILED_QUALITY runs in overlay (outputs still exist)
    ok_runs = [r for r in results if r["rc"] == 0 and not (Path(r["out_real"]) / "artifacts" / "error.json").exists()]
    
    # Check for FAILED_QUALITY and show warning banner
    failed_quality_runs = []
    for r in ok_runs:
        integrity_json_path = Path(r["out_real"]) / "artifacts" / "integrity_report.json"
        if integrity_json_path.exists():
            try:
                integrity = json.loads(integrity_json_path.read_text(encoding="utf-8"))
                if integrity.get("run_status") == "FAILED_QUALITY":
                    failed_quality_runs.append(r["model"])
            except Exception:
                pass
    
    if failed_quality_runs:
        st.warning(f"⚠️ **Model underperforms baseline:** {', '.join(failed_quality_runs)}. "
                  f"Plots shown below, but model does not beat persistence baseline.")

    if not ok_runs:
        st.warning("No successful runs to visualize.")
    else:
        # read predictions from each output folder
        preds_by_model: Dict[str, pd.DataFrame] = {}
        for r in ok_runs:
            pred = _read_predictions_any(Path(r["out_real"]))
            if pred is not None and {"y_true", "y_pred", "model"}.issubset(pred.columns) and "date" in pred.columns:
                preds_by_model[r["model"]] = pred

        if not preds_by_model:
            st.caption("No predictions_long.csv found in outputs.")
        else:
            compare_models = st.multiselect(
                "Select runs to visualize",
                list(preds_by_model.keys()),
                default=list(preds_by_model.keys()),
                help="Choose which model runs to overlay in the chart.",
            )

            # Build figure
            fig = go.Figure()

            # Use first selected model's y_true as "Actual" (they should all be identical if same target/cadence)
            first_key = compare_models[0] if compare_models else list(preds_by_model.keys())[0]
            base_pred = preds_by_model[first_key].copy()
            # ✅ Only plot true out-of-sample predictions - drop rows where y_true is NaN
            base_pred = base_pred.dropna(subset=["y_true"])  # Don't fill with 0
            fig.add_scatter(
                x=base_pred["date"],
                y=base_pred["y_true"],
                name="Actual",
                mode="lines",
                line=dict(color="black"),
            )

            for m in compare_models:
                p = preds_by_model[m].copy()
                # ✅ Only plot true out-of-sample predictions - drop rows where y_true or y_pred is NaN
                p = p.dropna(subset=["y_true", "y_pred"])  # Don't fill with 0
                if p.empty:
                    continue
                # In your output format, column "model" often contains internal model name; we label by run model selection
                fig.add_scatter(
                    x=p["date"],
                    y=p["y_pred"],
                    name=m,
                    mode="lines",
                )

            # Ops baseline if available (from first run)
            base = _baseline_series(Path(ok_runs[0]["out_real"]), target, cadence)
            if base is not None and len(base):
                rng = (base_pred["date"].min(), base_pred["date"].max())
                b = base[(base.index >= rng[0]) & (base.index <= rng[1])]
                if not b.empty:
                    fig.add_scatter(
                        x=b.index,
                        y=b.values,
                        name="Ops baseline",
                        mode="lines",
                        line=dict(dash="dot"),
                    )

            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Show log tails for each run (collapsed)
    st.subheader("Logs (tail)")
    for r in results:
        with st.expander(f"Log tail — {r['model']} ({'OK' if r['rc']==0 else 'FAILED'})", expanded=False):
            try:
                st.code(Path(r["log_path"]).read_text(encoding="utf-8")[-4000:], language="text")
            except Exception as e:
                st.caption(f"Could not read log: {e}")
