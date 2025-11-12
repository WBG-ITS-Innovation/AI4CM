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

st.markdown("""
Convert **Balance_by_Day_*.xlsx** into a standardized daily dataset
(**Raw**, **Clean — Conservative**, **Clean — Treasury**).
""")

# Use the BACKEND .venv python (autodetected). Do NOT fall back to frontend venv.
paths = load_paths()
backend_py  = paths.get("backend_python","")
backend_dir = paths.get("backend_dir","")

st.caption(f"Backend Python: `{backend_py or 'MISSING'}`")
st.caption(f"Backend dir   : `{backend_dir or 'MISSING'}`")

st.subheader("Source file")
up = st.file_uploader("Upload (.xlsx or .csv)", type=["xlsx","csv"])
manual_path = st.text_input("…or enter a file path", value="")
input_path = ""
if up:
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    saved = UPLOADS_ROOT / up.name
    saved.write_bytes(up.read())
    input_path = str(saved)
if manual_path.strip():
    input_path = manual_path.strip()

col1, col2 = st.columns(2)
with col1:
    date_col    = st.text_input("Date column (optional)", value="")
    balance_col = st.text_input("Balance column (optional)", value="")
with col2:
    variant       = st.selectbox("Variant", ["raw","clean_conservative","clean_treasury"])
    business_zero = st.checkbox("Set flows to 0 on non-business days", value=(variant!="raw"))
    weekday_weeks = st.number_input("Weekday-mean baseline (weeks)", 2, 52, 8)
    expected_csv  = st.text_input("Compare to expected CSV (optional)", value="")
    save_parquet  = st.checkbox("Also save a Parquet copy", value=False)

st.markdown("---")
if st.button("▶️ Run preprocessing", type="primary"):
    if not backend_py or not Path(backend_py).exists():
        st.error("Backend Python missing. Create `…TreasuryGeorgiaBackEnd\\.venv` and restart the app.\n"
                 "Then the resolver will find it automatically.")
        st.stop()
    runner_script = Path(backend_dir) / "run_preprocess.py"
    if not backend_dir or not Path(backend_dir).exists() or not runner_script.exists():
        st.error("Backend directory invalid or `run_preprocess.py` missing.")
        st.stop()
    if not input_path:
        st.error("Upload a file or enter a path.")
        st.stop()

    run_id, run_dir, out_dir = new_run_folders("run_PREPROC_" + Path(input_path).stem)
    pp_out_root = str((Path(backend_dir).parent / "data_preprocessed").resolve())

    env = {
        "PP_INPUT_PATH": input_path,
        "PP_DATE_COL": date_col.strip(),
        "PP_BALANCE_COL": balance_col.strip(),
        "PP_VARIANT": variant,
        "PP_BUSINESS_ZERO_FLOWS": str(business_zero).lower(),
        "PP_WEEKDAY_WEEKS": str(int(weekday_weeks)),
        "PP_OUT_ROOT": pp_out_root,
        "PP_RUN_OUTPUTS": str(out_dir),
        "PP_EXPECTED_CSV": expected_csv.strip(),
        "PP_SAVE_PARQUET": str(save_parquet).lower(),
    }

    rc, elapsed, log_file, _ = launch_backend(
        backend_py=backend_py,
        runner_script=str(runner_script),
        backend_dir=backend_dir,
        env_vars=env,
        run_dir=run_dir,
    )

    st.subheader("Run log (tail)")
    st.code(Path(log_file).read_text(encoding="utf-8")[-8000:], language="text")

    report = out_dir / "preprocess_report.json"
    if rc == 0 and report.exists():
        meta = json.loads(report.read_text(encoding="utf-8"))
        st.success("Completed.")
        st.json(meta)
        prev = Path(meta.get("preview_path",""))
        if prev.exists():
            st.subheader("Preview (first 200 rows)")
            st.dataframe(pd.read_csv(prev), use_container_width=True)
        out_csv = Path(meta.get("output_csv",""))
        if out_csv.exists():
            st.download_button("⬇️ Download processed CSV", data=out_csv.read_bytes(),
                               file_name=out_csv.name, key=f"dl_pp_csv_{out_csv.name}")
        out_parq = meta.get("output_parquet")
        if out_parq and Path(out_parq).exists():
            p = Path(out_parq)
            st.download_button("⬇️ Download processed Parquet", data=p.read_bytes(),
                               file_name=p.name, key=f"dl_pp_parq_{p.name}")
    else:
        st.error("Pre-processing failed. See log above.")
