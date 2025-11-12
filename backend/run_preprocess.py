from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime
from preprocess_data import PreprocessConfig, run_preprocess

def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    env = os.environ
    cfg = PreprocessConfig(
        input_path = env.get("PP_INPUT_PATH", ""),
        date_col   = env.get("PP_DATE_COL") or None,
        balance_col= env.get("PP_BALANCE_COL") or None,
        variant    = env.get("PP_VARIANT", "raw"),
        business_days_zero_flows = env.get("PP_BUSINESS_ZERO_FLOWS", "true").lower() in {"1","true","yes"},
        weekday_weeks = int(env.get("PP_WEEKDAY_WEEKS", "8")),
        out_root   = env.get("PP_OUT_ROOT", str((Path.cwd().parents[1] / "data_preprocessed").resolve())),
        run_outputs_dir = env.get("PP_RUN_OUTPUTS", str(Path.cwd() / "outputs")),
        expected_csv = env.get("PP_EXPECTED_CSV") or None,
        save_parquet = env.get("PP_SAVE_PARQUET", "false").lower() in {"1","true","yes"},
        sheet_name  = env.get("PP_SHEET_NAME") or None,
        header_row  = int(env["PP_HEADER_ROW"]) if env.get("PP_HEADER_ROW") not in (None,"","null","None") else None,
    )

    run_dir = Path(cfg.run_outputs_dir).parent; run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "backend_run.log"
    log.write_text("="*80 + f"\n[{_ts()}] Data preprocessing run starting\n" + json.dumps(cfg.__dict__, indent=2) + "\n" + "="*80 + "\n", encoding="utf-8")

    try:
        report = run_preprocess(cfg)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"[{_ts()}] Completed OK\n")
            f.write(json.dumps(report, indent=2) + "\n")
            f.write("="*80 + "\n")
        print(json.dumps(report, indent=2))
    except Exception as e:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"[{_ts()}] ERROR: {e}\n")
        raise
