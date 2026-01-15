# pages/00_Lab.py — Lab (run models + live log + overlay)
from __future__ import annotations

import html
import json
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
RUNS_DIR = APPROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Help strings (used as widget hover tooltips)
# -------------------------------------------------------------------
HELP: Dict[str, str] = {
    # Data
    "upload_csv": (
        "Upload a CSV file used for forecasting.\n\n"
        "Requirements:\n"
        "• A date column (e.g., 'date')\n"
        "• At least one numeric column (your forecast Target)\n"
        "• Optional: additional numeric columns (potential drivers / exogenous features)\n\n"
        "Tip: for best results, ensure the date column is consistently formatted and sorted."
    ),
    "quick_sample": (
        "Creates a smaller dataset from the end of the uploaded file (most recent rows).\n\n"
        "Why it exists:\n"
        "• Faster runs during demos\n"
        "• Faster iteration while tuning parameters\n\n"
        "This does not change the original uploaded dataset."
    ),
    "rows_keep": (
        "Number of rows kept in the quick sample.\n\n"
        "Practical guidance:\n"
        "• 2,000–10,000 rows is typically enough for a demo\n"
        "• Larger samples improve training signal but increase runtime"
    ),
    "data_source": (
        "Select which dataset file to run.\n\n"
        "• Full uploaded dataset: best fidelity, slower\n"
        "• Quick sample: faster, best for demos and iteration"
    ),
    "preview": (
        "Preview of the dataset. The chart preview below reflects your chosen cadence (Daily / Weekly / Monthly).\n\n"
        "If you see gaps or invalid dates, check date parsing and column selection."
    ),
    # Configure
    "date_col": (
        "The column used as time index.\n\n"
        "The runner:\n"
        "• Parses this column into timestamps\n"
        "• Sorts the data chronologically\n"
        "• Aggregates to Weekly/Monthly if selected\n\n"
        "Select the column that represents observation dates."
    ),
    "target": (
        "The numeric series you want to forecast.\n\n"
        "Examples:\n"
        "• Revenues, Expenditures, Balance, Net flows\n\n"
        "In Univariate mode, models use only the target history.\n"
        "In Multivariate mode, models may use other columns as additional inputs."
    ),
    "cadence": (
        "Controls the time resolution of forecasting.\n\n"
        "• Daily: observations by day\n"
        "• Weekly: aggregated by week (W-FRI)\n"
        "• Monthly: aggregated by month-end (ME)\n\n"
        "Cadence affects:\n"
        "• seasonality settings (7/52/12)\n"
        "• feature engineering choices (lags/windows)\n"
        "• horizon interpretation (steps ahead)"
    ),
    "horizon": (
        "How many steps ahead to forecast at the selected cadence.\n\n"
        "Examples:\n"
        "• Monthly horizon=6 → forecast 6 months ahead\n"
        "• Daily horizon=14 → forecast 14 days ahead\n\n"
        "Longer horizons are harder and typically increase error."
    ),
    "family": (
        "Select the forecasting model family.\n\n"
        "A · Statistical:\n"
        "• ETS, SARIMAX, STL-ARIMA, Theta, baselines\n"
        "• Transparent and strong baselines\n\n"
        "B · Machine Learning:\n"
        "• Regression/trees/boosting using engineered features\n\n"
        "C · Deep Learning:\n"
        "• LSTM/GRU/TCN/Transformer style models\n\n"
        "E · Quantile:\n"
        "• Produces P10/P50/P90 (risk-aware ranges)"
    ),
    "variant": (
        "Univariate vs Multivariate.\n\n"
        "• Univariate: uses only target history\n"
        "• Multivariate: also uses additional columns as potential predictors\n\n"
        "Multivariate can help when extra columns are meaningful drivers.\n"
        "If extra columns are noisy, it can reduce generalization."
    ),
    "profile": (
        "Defines how heavy the run is.\n\n"
        "Demo (fast):\n"
        "• minimal validation\n"
        "• optional data clipping\n\n"
        "Balanced:\n"
        "• moderate cross-validation\n"
        "• reasonable training history\n\n"
        "Thorough:\n"
        "• more cross-validation\n"
        "• slower but more robust comparisons"
    ),
    # ML knobs
    "lags": (
        "Lags are past target values used as input features.\n\n"
        "Example:\n"
        "• lag 7 (daily) uses the value from 7 days ago.\n\n"
        "Guidance:\n"
        "• Include lags around seasonality: 7/14/28 (daily), 4/12 (weekly), 3/12 (monthly)\n"
        "• Avoid excessively dense lag sets when data is limited"
    ),
    "windows": (
        "Rolling windows are trailing summaries (e.g., rolling mean/std) used as features.\n\n"
        "Example:\n"
        "• window 7 (daily) uses stats over the last 7 days.\n\n"
        "Guidance:\n"
        "• Use a few windows that match business cycles\n"
        "• Ensure windows are trailing (past-only) to prevent leakage"
    ),
    "exog_top_k": (
        "In Multivariate mode, selects the top-K most useful exogenous columns.\n\n"
        "Purpose:\n"
        "• Controls feature explosion\n"
        "• Helps reduce overfitting\n\n"
        "Guidance:\n"
        "• Start small (5–15)\n"
        "• Increase only if models are stable and data is sufficiently large"
    ),
    # DL knobs
    "lookback": (
        "Deep learning: how much history is fed into the network per prediction.\n\n"
        "Example:\n"
        "• lookback 48 (daily) = use last 48 days to predict future.\n\n"
        "Guidance:\n"
        "• Cover at least 1–2 major seasonal cycles when possible\n"
        "• Larger lookback increases runtime and memory"
    ),
    "batch_size": (
        "Deep learning: mini-batch size used during training.\n\n"
        "Higher:\n"
        "• faster per epoch\n"
        "• more memory usage\n\n"
        "Lower:\n"
        "• safer on CPU\n"
        "• slower but stable"
    ),
    "max_epochs": (
        "Deep learning: maximum training epochs.\n\n"
        "Guidance:\n"
        "• Use small values when iterating\n"
        "• Increase when you have a stable configuration and want best accuracy\n\n"
        "If training is unstable, reduce learning rate (configured in backend) or reduce model size."
    ),
    "valid_frac": (
        "Deep learning: fraction of data reserved for validation monitoring.\n\n"
        "Purpose:\n"
        "• track generalization during training\n"
        "• enable early stopping (if implemented in backend)"
    ),
    "conformal_calib": (
        "Fraction reserved for calibrating prediction intervals (conformal).\n\n"
        "Purpose:\n"
        "• improves reliability of uncertainty bands\n\n"
        "Tradeoff:\n"
        "• more calibration data = less training data"
    ),
    "device": (
        "Compute device selection.\n\n"
        "• auto: choose best available\n"
        "• cpu: safest / portable\n"
        "• cuda: GPU (only if CUDA is available on the system)"
    ),
    # Quantile knobs
    "quantiles": (
        "Quantiles for risk-aware forecasts.\n\n"
        "Example:\n"
        "0.1,0.5,0.9 → P10 (low), P50 (median), P90 (high)\n\n"
        "Use cases:\n"
        "• planning under uncertainty\n"
        "• conservative vs optimistic scenarios"
    ),
    # Advanced / Run
    "overrides": (
        "Advanced configuration payload passed to the backend as JSON.\n\n"
        "This is exposed for transparency and power-users.\n"
        "If you do not need custom tuning, keep defaults."
    ),
    "run_button": (
        "Runs the selected experiment.\n\n"
        "The system:\n"
        "1) writes a run folder under frontend/runs/\n"
        "2) launches the backend runner script\n"
        "3) streams log output here\n"
        "4) saves predictions, metrics, plots, and artifacts\n\n"
        "If a run fails, check backend_run.log for details."
    ),
}

# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------
def _scroll_term(container, text: str, height: int = 360):
    container.markdown(
        f"<div style='height:{height}px;overflow:auto;background:#0b0b0b;color:#e6e6e6; padding:10px; border-radius:10px;"
        "font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; white-space:pre;'>"
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _baseline_series(out_root: Path, target: str, cadence: str) -> Optional[pd.Series]:
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


# -------------------------------------------------------------------
# Sidebar: backend paths
# -------------------------------------------------------------------
st.sidebar.header("Backend (auto-detected)")
paths = load_paths()
st.session_state["backend_py"] = paths.get("backend_python", "") or st.session_state.get("backend_py", "")
st.session_state["backend_dir"] = paths.get("backend_dir", "") or st.session_state.get("backend_dir", "")

st.sidebar.caption(f"Python: `{st.session_state.get('backend_py','') or 'MISSING'}`")
st.sidebar.caption(f"Backend: `{st.session_state.get('backend_dir','') or 'MISSING'}`")

with st.sidebar.expander("What the backend settings mean", expanded=False):
    st.markdown(
        """
The UI launches model runs by calling a **backend Python environment**.

- **Python** points to the backend virtual environment interpreter (backend/.venv)
- **Backend** is the directory containing the runner scripts (run_a_stat.py, run_b_ml_*.py, ...)

If either path is missing, go to the **Overview** page and re-run setup scripts.
"""
    )

# -------------------------------------------------------------------
# Page header
# -------------------------------------------------------------------
st.title("🧪 Forecast Lab")
st.caption(
    "Run forecasting experiments (statistical, ML, deep learning, quantiles) and review outputs immediately. "
    "Each experiment is saved as a run folder for reproducibility."
)

with st.expander("How this page is structured", expanded=True):
    st.markdown(
        """
This page is organized into three steps:

**1) Data**  
Upload a dataset and choose which file (full vs quick sample) to run.

**2) Configure**  
Select the target series, cadence, horizon, model family, and model-specific parameters.

**3) Launch & live log**  
Run the experiment, monitor logs in real-time, and review an immediate overlay plot (Actual vs predictions).

Outputs are saved under `frontend/runs/` in a time-stamped run folder.
"""
    )

# -------------------------------------------------------------------
# 1) Data
# -------------------------------------------------------------------
st.header("1) Data")

st.markdown(
    """
Upload a CSV used as input for forecasting.  
If you are iterating or running a live demo, consider creating a quick sample for faster execution.
"""
)

up = st.file_uploader("Upload CSV", type=["csv"], help=HELP["upload_csv"])
if up:
    dest = UPLOADS_ROOT / "uploaded.csv"
    dest.write_bytes(up.read())
    st.success(f"Saved: `{dest}`")

with st.expander("Optional: create a quick sample for faster runs", expanded=False):
    st.markdown(
        """
This creates a smaller dataset from the end of your upload (the most recent rows).  
It is intended for faster experimentation and demonstrations.
"""
    )
    n = st.slider("Rows to keep (from end)", 200, 200_000, 2_000, step=200, help=HELP["rows_keep"])
    if st.button("Create / Refresh sample from uploaded.csv", help=HELP["quick_sample"]):
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
    st.warning("Upload a CSV to continue.")
    st.stop()

use_label = st.radio(
    "Select data source",
    [lbl for lbl, _ in sources],
    index=0,
    horizontal=True,
    help=HELP["data_source"],
)

data_path = Path(dict(sources)[use_label])
df = pd.read_csv(data_path)

with st.expander("Dataset preview", expanded=True):
    st.caption("Preview is limited to the first 200 rows. The full dataset is used in the run.")
    head = df.head(200).copy()
    dcol_guess = "date" if "date" in head.columns else head.columns[0]
    dtry = pd.to_datetime(head[dcol_guess], errors="coerce")
    st.caption(f"Rows: {len(df):,} • date span (preview parse): {dtry.min()} → {dtry.max()}")
    st.dataframe(head, use_container_width=True)

# -------------------------------------------------------------------
# 2) Configure
# -------------------------------------------------------------------
st.header("2) Configure")

st.markdown(
    """
Select the forecast target and run configuration.  
Key choices that most strongly affect runtime and outputs:
- **Cadence** (Daily/Weekly/Monthly)
- **Horizon** (steps ahead)
- **Family** and **Model**
- **Run profile** (validation depth)
"""
)

L, R = st.columns([1.08, 0.92], gap="large")

with L:
    date_col = st.selectbox(
        "Date column",
        list(df.columns),
        index=(list(df.columns).index("date") if "date" in df.columns else 0),
        help=HELP["date_col"],
    )

    num_cols = [c for c in df.columns if c != date_col]
    target = st.selectbox("Target", num_cols, help=HELP["target"])
    cadence = st.selectbox("Cadence", ["Daily", "Weekly", "Monthly"], index=0, help=HELP["cadence"])
    horizon = st.slider("Horizon", 1, 24, 6, help=HELP["horizon"])

    with st.expander(f"Target preview — {target} @ {cadence}", expanded=True):
        st.caption(HELP["preview"])
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
        "Family",
        ["A · Statistical", "B · Machine Learning", "C · Deep Learning", "E · Quantile"],
        help=HELP["family"],
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
        help=HELP["variant"],
    )

st.subheader("Run profile")
profile = st.radio(
    "Profile",
    ["Demo (fast)", "Balanced", "Thorough"],
    horizontal=True,
    index=0,
    help=HELP["profile"],
)

# Profile → default override payload
ov: Dict[str, Any] = {
    "folds": 0 if profile == "Demo (fast)" else (3 if profile == "Balanced" else 5),
    "min_train_years": 0 if profile == "Demo (fast)" else (2 if profile == "Balanced" else 4),
    "demo_clip_months": 12 if profile == "Demo (fast)" else None,
}

# Model selection + family-specific knobs
if family == "B_ML":
    model = st.selectbox(
        "Model (ML)",
        ["Ridge", "Lasso", "ElasticNet", "RandomForest", "ExtraTrees", "HistGBDT", "XGBoost", "LightGBM"],
        index=0,
    )

    st.markdown(
        """
**Feature engineering for ML models**  
ML models require explicit features. This prototype uses lags (past values) and rolling windows (trailing summaries) as core features.
"""
    )

    if cadence == "Daily":
        ov["lags_daily"] = st.multiselect("lags_daily", [1, 2, 3, 5, 7, 14, 21], default=[1, 3, 7], help=HELP["lags"])
        ov["windows_daily"] = st.multiselect("windows_daily", [3, 5, 7, 14, 21, 28], default=[3, 7], help=HELP["windows"])
    elif cadence == "Weekly":
        ov["lags_weekly"] = st.multiselect("lags_weekly", [1, 2, 3, 4, 8, 12, 26], default=[1, 4, 12], help=HELP["lags"])
        ov["windows_weekly"] = st.multiselect("windows_weekly", [2, 4, 8, 12, 26], default=[4, 12], help=HELP["windows"])
    else:
        ov["lags_monthly"] = st.multiselect("lags_monthly", [1, 2, 3, 6, 12], default=[1, 3], help=HELP["lags"])
        ov["windows_monthly"] = st.multiselect("windows_monthly", [3, 6, 12], default=[3, 6], help=HELP["windows"])

    if variant == "Multivariate":
        st.markdown(
            """
**Multivariate configuration**  
When multivariate is enabled, the runner can incorporate other numeric columns as candidate predictors.
To reduce overfitting and keep the run tractable, you can limit how many are used.
"""
        )
        ov["exog_top_k"] = st.number_input("exog_top_k", 0, 64, 8, help=HELP["exog_top_k"])

elif family == "A_STAT":
    model = st.selectbox(
        "Model (Stat)",
        ["ETS", "SARIMAX", "STL_ARIMA", "THETA", "NAIVE", "WEEKDAY_MEAN", "MOVAVG"],
        index=0,
    )
    with st.expander("Notes on statistical models", expanded=False):
        st.markdown(
            """
Statistical models are typically strong baselines and are easier to explain and audit.

- **ETS**: exponential smoothing with trend/seasonality (often a top baseline)
- **SARIMAX**: ARIMA-family with optional seasonal terms
- **STL_ARIMA**: decomposition (STL) + ARIMA on remainder
- **THETA**: robust competition-grade baseline
- **NAIVE / WEEKDAY_MEAN / MOVAVG**: fast baselines useful for comparison
"""
        )

elif family == "C_DL":
    model = st.selectbox("Model (DL)", ["GRU", "LSTM", "TCN", "Transformer", "MLP"], index=0)

    st.markdown(
        """
**Deep learning configuration**  
Deep learning models typically require more training time. The key parameters are: lookback (history length), batch size, and epochs.
"""
    )
    ov["lookback"] = st.number_input("lookback", 4, 365, 48, help=HELP["lookback"])
    ov["batch_size"] = st.number_input("batch_size", 8, 2048, 128, step=8, help=HELP["batch_size"])
    ov["max_epochs"] = st.number_input("max_epochs", 1, 200, 3, help=HELP["max_epochs"])
    ov["valid_frac"] = st.number_input("valid_frac", 0.05, 0.9, 0.2, 0.05, help=HELP["valid_frac"])
    ov["conformal_calib_frac"] = st.number_input("conformal_calib_frac", 0.05, 0.9, 0.2, 0.05, help=HELP["conformal_calib"])
    ov["device"] = st.selectbox("device", ["auto", "cpu", "cuda"], index=0, help=HELP["device"])

else:
    model = st.selectbox("Model (Quantile)", ["GBQuantile"], index=0)

    st.markdown(
        """
**Quantile forecasting configuration**  
Quantile models output multiple forecast levels (e.g., P10/P50/P90) which can be used for scenario-based planning.
"""
    )

    q_text = st.text_input("quantiles (comma-separated)", "0.1,0.5,0.9", help=HELP["quantiles"])
    try:
        ov["quantiles"] = [float(x) for x in q_text.split(",") if x.strip() != ""]
    except Exception:
        ov["quantiles"] = [0.1, 0.5, 0.9]

# Advanced overrides
st.subheader("Advanced overrides (JSON)")
st.caption(
    "This JSON is passed directly to the backend. It is exposed for transparency and advanced tuning. "
    "Most users can keep defaults unless they are experimenting with model behavior."
)
ov_text = st.text_area("OVERRIDES_JSON", json.dumps(ov, indent=2), height=200, help=HELP["overrides"])
try:
    ov_final: Dict[str, Any] = json.loads(ov_text)
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    ov_final = ov

# -------------------------------------------------------------------
# 3) Launch
# -------------------------------------------------------------------
st.header("3) Launch & live log")

with st.expander("What happens during a run (run folder, outputs, logs)", expanded=False):
    st.markdown(
        """
When you run an experiment, the UI:

1. Creates a new run folder under `frontend/runs/` (time-stamped, includes configuration).
2. Launches a backend runner script based on the selected family/model/variant.
3. Streams the backend log to this page while the model is training/evaluating.
4. Saves outputs, typically including:
   - `predictions_long.csv` (actuals + predictions by model/date)
   - `metrics_long.csv` (MAE/RMSE/etc.)
   - `leaderboard.csv` (ranking models by metric)
   - `plots/` and `artifacts/` for visuals and serialized config

If a run fails, the log will usually contain the exception and the failing step.
"""
    )

# Build readable run name for History/Dashboard lists
ts = datetime.now().strftime("%Y%m%d_%H%M")
short_fam = {"A_STAT": "A", "B_ML": "B", "C_DL": "C", "E_QUANTILE": "E"}[family]
short_var = "uni" if variant == "Univariate" else "multi"
run_label = f"run_{short_fam}_{short_var}_{model}_{target}_{cadence}_h{int(horizon)}_{ts}"

run_id, run_dir, out_root = new_run_folders(run_label)

# Choose runner script
if family == "A_STAT":
    runner = Path(st.session_state["backend_dir"]) / "run_a_stat.py"
elif family == "B_ML":
    runner = Path(st.session_state["backend_dir"]) / ("run_b_ml_univariate.py" if short_var == "uni" else "run_b_ml_multivariate.py")
elif family == "C_DL":
    runner = Path(st.session_state["backend_dir"]) / ("run_c_dl_univariate.py" if short_var == "uni" else "run_c_dl_multivariate.py")
else:
    runner = Path(st.session_state["backend_dir"]) / ("run_e_quantile_daily_univariate.py" if short_var == "uni" else "run_e_quantile_daily_multivariate.py")

env = {
    "TG_FAMILY": family,
    "TG_MODEL_FILTER": model,
    "TG_TARGET": target,
    "TG_CADENCE": cadence,
    "TG_HORIZON": str(int(horizon)),
    "TG_DATA_PATH": str(Path(data_path).resolve()),
    "TG_DATE_COL": str([c for c in df.columns if c.lower() == date_col.lower()][0]),
    "TG_PARAM_OVERRIDES": json.dumps(ov_final),
    "TG_OUT_ROOT": str(out_root.resolve()),
}

log_box = st.empty()
status = st.empty()

def _on_progress(tail: str, elapsed: float):
    status.info(f"Elapsed: {elapsed:.1f}s")
    _scroll_term(log_box, tail)

if st.button("🚀 Run experiment", type="primary", use_container_width=True, help=HELP["run_button"]):
    py = st.session_state.get("backend_py", "")
    back = st.session_state.get("backend_dir", "")

    if not py or not Path(py).exists():
        st.error("Backend Python missing/invalid. Go to Overview and ensure backend/.venv exists.")
        st.stop()
    if not back or not Path(back).exists():
        st.error("Backend directory missing/invalid. Go to Overview and confirm backend path.")
        st.stop()
    if not runner.exists():
        st.error(f"Runner script missing: `{runner}`")
        st.stop()

    rc, elapsed, log_path, out_real = launch_backend(
        backend_py=py,
        runner_script=str(runner),
        backend_dir=back,
        env_vars=env,
        run_dir=run_dir,
        on_progress=_on_progress,
    )

    if rc != 0:
        st.error(f"Run failed (rc={rc}). See log below.")
    else:
        st.success(f"Finished in {elapsed:.1f}s • outputs in `{out_real}`")

        with st.expander("Overlay preview — Actual vs selected model(s) (and Ops baseline if available)", expanded=True):
            p = Path(out_real) / "predictions_long.csv"
            if not p.exists():
                for cad in ("daily", "weekly", "monthly"):
                    cand = Path(out_real) / cad / "predictions_long.csv"
                    if cand.exists():
                        p = cand
                        break

            if p.exists():
                pred = pd.read_csv(p)
                pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
                pred = pred.dropna(subset=["date"]).sort_values("date")

                models = sorted(pred["model"].unique())
                sel = st.multiselect("Models to visualize", models, default=models[:1])

                fig = go.Figure()
                fig.add_scatter(
                    x=pred["date"], y=pred["y_true"],
                    name="Actual", mode="lines",
                    line=dict(color="black")
                )
                for m in sel:
                    g = pred[pred["model"] == m]
                    fig.add_scatter(x=g["date"], y=g["y_pred"], name=m, mode="lines")

                base = _baseline_series(Path(out_real), target, cadence)
                if base is not None and len(base):
                    rng = (pred["date"].min(), pred["date"].max())
                    b = base[(base.index >= rng[0]) & (base.index <= rng[1])]
                    if not b.empty:
                        fig.add_scatter(
                            x=b.index, y=b.values,
                            name="Ops baseline", mode="lines",
                            line=dict(dash="dot")
                        )

                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.caption("No predictions_long.csv found in the outputs folder.")

    st.markdown("**Log tail**")
    st.code(Path(log_path).read_text(encoding="utf-8")[-5000:], language="text")
