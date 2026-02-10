# run_b_ml_univariate.py
import json
import os
import sys
from pathlib import Path
from b_ml_pipeline import ConfigBML, run_pipeline_ml
import b_ml_pipeline

def _env(name: str, default=None):
    v = os.environ.get(name, default)
    return None if v in ("", "null", "None") else v

if __name__ == "__main__":
    cfg = ConfigBML(
        data_path=_env("TG_DATA_PATH"),
        date_col=_env("TG_DATE_COL", "date"),
        target=_env("TG_TARGET"),
        cadence=_env("TG_CADENCE", "Daily"),
        horizon=int(_env("TG_HORIZON", 5)),
        variant="uni",
        model_filter=_env("TG_MODEL_FILTER"),
        out_root=_env("TG_OUT_ROOT"),
    )

    # Apply JSON overrides
    overrides = _env("TG_PARAM_OVERRIDES", "{}")
    try:
        ov = json.loads(overrides) if isinstance(overrides, str) else (overrides or {})
    except Exception:
        ov = {}
    for k, v in ov.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    
    # ✅ Provenance tracking
    provenance = {
        "pipeline_module_path": str(b_ml_pipeline.__file__),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "env_vars": {
            "TG_FAMILY": os.environ.get("TG_FAMILY"),
            "TG_MODEL_FILTER": os.environ.get("TG_MODEL_FILTER"),
            "TG_TARGET": os.environ.get("TG_TARGET"),
            "TG_CADENCE": os.environ.get("TG_CADENCE"),
            "TG_HORIZON": os.environ.get("TG_HORIZON"),
            "TG_DATA_PATH": os.environ.get("TG_DATA_PATH"),
            "TG_DATE_COL": os.environ.get("TG_DATE_COL"),
            "TG_PARAM_OVERRIDES": os.environ.get("TG_PARAM_OVERRIDES"),
            "TG_OUT_ROOT": os.environ.get("TG_OUT_ROOT"),
        },
        "config": {
            "folds": cfg.folds,
            "min_train_years": cfg.min_train_years,
            "demo_clip_months": cfg.demo_clip_months,
        }
    }
    
    provenance_path = Path(cfg.out_root) / "artifacts" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    
    print("[runner] ===== Georgia B·ML (univariate) runner =====")
    print(f"[runner] Pipeline module: {b_ml_pipeline.__file__}")
    print(f"[runner] Python: {sys.executable}")
    print(f"[runner] CWD: {Path.cwd()}")
    for key in ("TG_FAMILY","TG_MODEL_FILTER","TG_TARGET","TG_CADENCE","TG_HORIZON","TG_DATA_PATH","TG_DATE_COL","TG_PARAM_OVERRIDES","TG_OUT_ROOT"):
        print(f"[runner] {key} = {os.environ.get(key)}")
    print("[runner] START pipeline for target='%s' cadence=%s horizon=%s (univariate)" % (cfg.target, cfg.cadence, cfg.horizon))
    
    try:
        run_pipeline_ml(cfg)
        print("[runner] DONE")
    except SystemExit as e:
        raise
    except Exception as e:
        import traceback
        print(f"[runner] ERROR: {e}")
        print("[runner] Full traceback:")
        traceback.print_exc()
        # Write error to artifacts/error.json
        error_path = Path(cfg.out_root) / "artifacts" / "error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_path, "w", encoding="utf-8") as f:
            import json
            json.dump({
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }, f, indent=2)
        print(f"[runner] Error details saved to: {error_path}")
        sys.exit(1)
