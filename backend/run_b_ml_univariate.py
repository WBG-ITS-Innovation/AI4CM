# run_b_ml_univariate.py
import json
import os
from pathlib import Path
from b_ml_pipeline import ConfigBML, run_pipeline_ml

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
    print("[runner] ===== Georgia B·ML (univariate) runner =====")
    for key in ("TG_FAMILY","TG_MODEL_FILTER","TG_TARGET","TG_CADENCE","TG_HORIZON","TG_DATA_PATH","TG_DATE_COL","TG_PARAM_OVERRIDES","TG_OUT_ROOT"):
        print(f"[runner] {key} = {os.environ.get(key)}")
    print("[runner] START pipeline for target='%s' cadence=%s horizon=%s (univariate)" % (cfg.target, cfg.cadence, cfg.horizon))
    run_pipeline_ml(cfg)
    print("[runner] DONE")
