# run_e_quantile_daily_multivariate.py
from __future__ import annotations
import os, json, time
from e_quantile_daily_pipeline import Config, run_pipeline

def _getenv(k, default=None):
    v = os.environ.get(k, default)
    return v

def main():
    t0 = time.time()
    print("[runner] ===== Georgia E·Quantile (multivariate) runner =====")
    overrides = json.loads(_getenv("TG_PARAM_OVERRIDES", "{}"))
    cfg = Config(
        target=_getenv("TG_TARGET", "State budget balance"),
        cadence=_getenv("TG_CADENCE", "Daily"),
        horizon=int(_getenv("TG_HORIZON", "5")),
        data_path=_getenv("TG_DATA_PATH"),
        date_col=_getenv("TG_DATE_COL", "date"),
        folds=overrides.get("folds", 3),  # None = use ALL folds (thorough mode)
        min_train_years=int(overrides.get("min_train_years", 4)),
        model_filter=_getenv("TG_MODEL_FILTER", "").strip() or None,
        quantiles=tuple(overrides.get("quantiles", [0.1,0.5,0.9])),
        lags_daily=tuple(overrides.get("lags_daily", [1,5,20])),
        windows_daily=tuple(overrides.get("windows_daily", [5,20])),
        exog_top_k=overrides.get("exog_top_k", 20),
        out_root=_getenv("TG_OUT_ROOT", "outputs"),
        demo_clip_months=overrides.get("demo_clip_months", None),
        variant="multivariate"
    )

    print(f"[runner] TG_FAMILY = E_QUANTILE")
    print(f"[runner] TG_MODEL_FILTER = {cfg.model_filter or '(all)'}")
    print(f"[runner] TG_TARGET = {cfg.target}")
    print(f"[runner] TG_CADENCE = {cfg.cadence}")
    print(f"[runner] TG_HORIZON = {cfg.horizon}")
    print(f"[runner] TG_DATA_PATH = {cfg.data_path}")
    print(f"[runner] TG_DATE_COL = {cfg.date_col}")
    print(f"[runner] TG_PARAM_OVERRIDES = {os.environ.get('TG_PARAM_OVERRIDES','{}')}")
    print(f"[runner] TG_OUT_ROOT = {cfg.out_root}")
    print(f"[runner] RUNNER = {__file__}")
    print("[runner] Loading data…")

    run_pipeline(cfg)
    print(f"[runner] Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
