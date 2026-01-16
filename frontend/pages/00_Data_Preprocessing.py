# pages/00_Data_Preprocessing.py — Data Pre-processing (upload + auto backend paths)
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_frontend import new_run_folders, load_paths, UPLOADS_ROOT
from backend_bridge import launch_backend

st.set_page_config(page_title="Data Pre-processing • Georgia Treasury", layout="wide")
st.title("🧺 Data Pre-processing")

st.write(
    """
This page converts a Treasury **Balance_by_Day** file into a standardized daily dataset
that the Forecast Lab can use consistently.

You can generate three variants of the same dataset:
- **raw** (minimal transformation)
- **clean_conservative** (safe cleaning)
- **clean_treasury** (Treasury-friendly rules for modeling & demos)
"""
)

with st.expander("What do the three variants mean?", expanded=False):
    st.markdown(
        """
### raw
Minimal processing. We parse dates, ensure numeric columns are numeric, and standardize the output format.
This is useful for transparency and debugging, but may include anomalies that make forecasts unstable.

### clean_conservative
A “safe cleaning” version. It fixes issues that are very likely to be data-quality problems (duplicates,
inconsistent parsing, basic missingness handling) without applying strong assumptions about how Treasury
*should* behave.

### clean_treasury
A “Treasury-friendly” version. It applies additional assumptions and rules intended to make the dataset
behave consistently for forecasting and demos—especially around **business days**, **weekend/holiday behavior**,
and patterns that matter operationally.

> The exact transformations are implemented in the backend preprocessing runner. This UI is designed to make the intent
and choices easy to understand and to reproduce.
"""
    )

# Use the BACKEND .venv python (autodetected). Do NOT fall back to frontend venv.
paths = load_paths()
backend_py = paths.get("backend_python", "")
backend_dir = paths.get("backend_dir", "")

st.sidebar.header("Backend (auto)")
st.sidebar.caption(f"Backend Python: `{backend_py or 'MISSING'}`")
st.sidebar.caption(f"Backend dir: `{backend_dir or 'MISSING'}`")

# ---------------------------- Source file ----------------------------
st.subheader("1) Source file")

up = st.file_uploader(
    "Upload input file (.xlsx or .csv)",
    type=["xlsx", "csv"],
    help=(
        "Upload the source file.\n\n"
        "Typical input: Balance_by_Day_2015-2025.xlsx\n"
        "You can also upload a CSV if it already contains a date column."
    ),
)

manual_path = st.text_input(
    "…or enter a local file path",
    value="",
    help=(
        "If the file is already on your machine/server, you can paste the full path instead of uploading.\n\n"
        "Windows example: C:\\Users\\Name\\Downloads\\Balance_by_Day_2015-2025.xlsx\n"
        "macOS example: /Users/name/Downloads/Balance_by_Day_2015-2025.xlsx"
    ),
)

input_path = ""
if up:
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    saved = UPLOADS_ROOT / up.name
    saved.write_bytes(up.read())
    input_path = str(saved)

if manual_path.strip():
    input_path = manual_path.strip()

if input_path:
    st.success(f"Using: `{input_path}`")

# Optional Excel-specific knobs
with st.expander("Excel options (only if needed)", expanded=False):
    sheet_name = st.text_input(
        "Sheet name (optional)",
        value="",
        help="If the Excel file has multiple tabs, specify the one that contains the data. Leave blank to use the default.",
    )
    header_row = st.text_input(
        "Header row (optional)",
        value="",
        help="If headers are not on the first row, provide the row number (e.g., 2). Leave blank for default.",
    )

# ---------------------------- Column mapping ----------------------------
st.subheader("2) Column mapping (optional)")

st.write(
    "If your file uses non-standard column names, you can specify them here. "
    "If you leave these blank, the backend will try to detect them automatically."
)

col1, col2 = st.columns(2)
with col1:
    date_col = st.text_input(
        "Date column name (optional)",
        value="",
        help="Name of the date column in the file. Leave blank to auto-detect.",
    )
with col2:
    balance_col = st.text_input(
        "Balance column name (optional)",
        value="",
        help="Name of the main balance/level column (if applicable). Leave blank to auto-detect.",
    )

# ---------------------------- Variant & cleaning rules ----------------------------
st.subheader("3) Output variant (raw vs cleaned)")

variant = st.selectbox(
    "Variant",
    ["raw", "clean_conservative", "clean_treasury"],
    index=2,
    help="Choose how aggressively to clean/standardize the data for modeling.",
)

VARIANT_EXPLAIN = {
    "raw": "Minimal transformation. Best for transparency/debugging. May contain anomalies that destabilize models.",
    "clean_conservative": "Safe cleaning only. Fixes likely data issues but avoids strong domain assumptions.",
    "clean_treasury": "Treasury-friendly cleaning. Applies business-day logic and other assumptions for consistent modeling/demos.",
}
st.info(VARIANT_EXPLAIN[variant])

colA, colB, colC = st.columns(3)

with colA:
    business_zero = st.checkbox(
        "Set flows to 0 on non-business days",
        value=(variant != "raw"),
        help=(
            "When enabled, the preprocessing will set flow-type series to 0 on non-business days (e.g., weekends).\n\n"
            "This is often useful for Treasury flows, because activity may be concentrated on business days.\n"
            "If you are preprocessing balances/levels, this may be less relevant."
        ),
    )

with colB:
    weekday_weeks = st.number_input(
        "Weekday-mean reference window (weeks)",
        2,
        52,
        8,
        help=(
            "Used in Treasury-friendly cleaning to estimate typical values for a weekday based on recent history.\n\n"
            "Example: with 8 weeks, Monday behavior is estimated from the past 8 Mondays.\n"
            "This can help stabilize flows that have strong weekday patterns."
        ),
    )

with colC:
    save_parquet = st.checkbox(
        "Also save Parquet output",
        value=False,
        help="If enabled, the processed dataset is saved as both CSV and Parquet. Parquet is faster for large files.",
    )

expected_csv = st.text_input(
    "Compare to an expected CSV (optional)",
    value="",
    help=(
        "Optional regression test.\n\n"
        "If you provide a path to an 'expected' output CSV, the backend can compare the produced output to it "
        "and report differences."
    ),
)

# ---------------------------- Run preprocessing ----------------------------
st.markdown("---")
st.subheader("4) Run preprocessing")

st.write(
    """
When you run preprocessing, the backend will:
- read the input file
- apply the selected cleaning variant and rules
- write the processed dataset to `data_preprocessed/<variant>/`
- write a small JSON report + preview that this page displays
"""
)

if st.button("▶️ Run preprocessing", type="primary", use_container_width=True):
    if not backend_py or not Path(backend_py).exists():
        st.error(
            "Backend Python missing.\n\n"
            "Run scripts/setup_windows.bat (Windows) or scripts/setup_unix.sh (macOS/Linux) first, "
            "then restart the app so the backend path can be detected automatically."
        )
        st.stop()

    runner_script = Path(backend_dir) / "run_preprocess.py"
    if not backend_dir or not Path(backend_dir).exists() or not runner_script.exists():
        st.error("Backend directory invalid or `run_preprocess.py` missing in backend/")
        st.stop()

    if not input_path:
        st.error("Upload a file or enter a local file path.")
        st.stop()

    run_id, run_dir, out_dir = new_run_folders()
    repo_root = Path(backend_dir).parent  # backend/.. = repo root
    pp_out_root = str((repo_root / "data_preprocessed").resolve())

    env = {
        "PP_INPUT_PATH": input_path,
        "PP_DATE_COL": date_col.strip(),
        "PP_BALANCE_COL": balance_col.strip(),
        "PP_VARIANT": variant,
        "PP_BUSINESS_ZERO_FLOWS": str(bool(business_zero)).lower(),
        "PP_WEEKDAY_WEEKS": str(int(weekday_weeks)),
        "PP_OUT_ROOT": pp_out_root,
        "PP_RUN_OUTPUTS": str(out_dir),
        "PP_EXPECTED_CSV": expected_csv.strip(),
        "PP_SAVE_PARQUET": str(bool(save_parquet)).lower(),
        "PP_SHEET_NAME": sheet_name.strip(),
        "PP_HEADER_ROW": header_row.strip(),
    }

    rc, elapsed, log_file, _ = launch_backend(
        backend_py=backend_py,
        runner_script=str(runner_script),
        backend_dir=backend_dir,
        env_vars=env,
        run_dir=run_dir,
    )

    st.caption(f"Run finished in {elapsed:.1f}s (return code={rc}).")

    st.subheader("Run log (tail)")
    st.code(Path(log_file).read_text(encoding="utf-8")[-8000:], language="text")

    report = out_dir / "preprocess_report.json"
    if rc == 0 and report.exists():
        meta = json.loads(report.read_text(encoding="utf-8"))
        st.success("✅ Pre-processing completed.")
        st.subheader("Preprocess report")
        st.json(meta)

        prev = Path(meta.get("preview_path", ""))
        if prev.exists():
            st.subheader("Preview (first 200 rows)")
            st.dataframe(pd.read_csv(prev).head(200), use_container_width=True)

        out_csv = Path(meta.get("output_csv", ""))
        if out_csv.exists():
            st.download_button(
                "⬇️ Download processed CSV",
                data=out_csv.read_bytes(),
                file_name=out_csv.name,
                key=f"dl_pp_csv_{out_csv.name}",
                use_container_width=True,
            )

        out_parq = meta.get("output_parquet")
        if out_parq and Path(out_parq).exists():
            p = Path(out_parq)
            st.download_button(
                "⬇️ Download processed Parquet",
                data=p.read_bytes(),
                file_name=p.name,
                key=f"dl_pp_parq_{p.name}",
                use_container_width=True,
            )
    else:
        st.error("❌ Pre-processing failed. See log above.")
