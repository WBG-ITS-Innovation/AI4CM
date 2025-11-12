# run_c_dl_multivariate.py — Georgia C·DL (multivariate) runner
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Import the DL pipeline (must be in the same folder)
import c_dl_pipeline as pipe
from c_dl_pipeline import ConfigDL

def _log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)

def _get_env() -> Dict[str, str]:
    keys = [
        "TG_FAMILY", "TG_MODEL_FILTER", "TG_TARGET", "TG_CADENCE", "TG_HORIZON",
        "TG_DATA_PATH", "TG_DATE_COL", "TG_PARAM_OVERRIDES", "TG_OUT_ROOT"
    ]
    return {k: os.environ.get(k, "") for k in keys}

def _parse_overrides(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        _log(f"[runner] WARN: TG_PARAM_OVERRIDES is not valid JSON: {e}. Ignoring.")
        return {}

def _models_from_filter(model_filter: str) -> List[str]:
    # Valid model names in pipeline are lowercase
    # (lstm, gru, dcnn, transformer, mlp)
    if not model_filter:
        return ["lstm", "gru", "dcnn", "transformer", "mlp"]
    return [model_filter.lower()]

def main() -> None:
    _log("===== Georgia C·DL (multivariate) runner =====")
    env = _get_env()

    # Print env (like other families)
    for k in ["TG_FAMILY","TG_MODEL_FILTER","TG_TARGET","TG_CADENCE","TG_HORIZON",
              "TG_DATA_PATH","TG_DATE_COL","TG_PARAM_OVERRIDES","TG_OUT_ROOT"]:
        _log(f"{k} = {env.get(k)}")

    # Required bits
    target   = env["TG_TARGET"] or "State budget balance"
    cadence  = (env["TG_CADENCE"] or "Daily").lower()   # daily|weekly|monthly
    horizon  = int(env["TG_HORIZON"] or 5)
    data_p   = env["TG_DATA_PATH"]
    date_col = env["TG_DATE_COL"] or "date"
    out_root = env["TG_OUT_ROOT"] or "./outputs/dl_multivariate"

    # Overrides
    ov = _parse_overrides(env["TG_PARAM_OVERRIDES"])

    # Build a ConfigDL consistent with c_dl_pipeline
    cfg = ConfigDL(
        data_path=data_p,
        date_col=date_col,
        holidays_csv=ov.get("holidays_csv"),
        out_root_multi=out_root,
        targets=[target],
        cadences=[cadence],
        seq_len_daily=int(ov.get("lookback", 64)),
        seq_len_weekly=int(ov.get("lookback", 52)),
        seq_len_monthly=int(ov.get("lookback", 36)),
        epochs=int(ov.get("max_epochs", 30)),
        batch_size=int(ov.get("batch_size", 128)),
        lr=float(ov.get("lr", 1e-3)),
        weight_decay=float(ov.get("weight_decay", 1e-4)),
        dropout=float(ov.get("dropout", 0.1)),
        valid_frac=float(ov.get("valid_frac", 0.1)),
        conformal_calib_frac=float(ov.get("conformal_calib_frac", 0.2)),
        nominal_pi=float(ov.get("nominal_pi", 0.90)),
        sample_weight_scheme=str(ov.get("sample_weight_scheme", "none")),
        eom_boost_weight=float(ov.get("eom_boost_weight", 3.0)),
        target_transform=str(ov.get("target_transform", "none")),
        min_train_years=int(ov.get("min_train_years", 4)),
        device=str(ov.get("device", "auto")),
        quick_mode=bool(ov.get("quick_mode", False)),
    )

    # Horizons (only the active cadence matters)
    if cadence == "daily":
        cfg.horizons_daily = [horizon]
    elif cadence == "weekly":
        cfg.horizons_weekly = [horizon]
    else:
        cfg.horizons_monthly = [horizon]

    # If user specifies just one model via TG_MODEL_FILTER => set as multivariate-only
    models = _models_from_filter(env["TG_MODEL_FILTER"])
    cfg.models_multivariate = models

    # Multivariate feature cap (how many exogenous columns to auto-include)
    cfg.mv_max_auto_features = int(ov.get("exog_top_k", cfg.mv_max_auto_features))

    try:
        _log(f"[runner] START pipeline for target='{target}' cadence={cadence.capitalize()} horizon={horizon} (multivariate)")
        pipe.run_pipeline(config=cfg, run_univariate=False, run_multivariate=True)
        _log(f"[OK] Master outputs in: {Path(out_root).resolve()}")
        _log("[runner] DONE")
    except Exception as e:
        _log(f"[runner] ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
