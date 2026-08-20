# backend_consts.py

QUICK_DL_DEFAULTS = {
    "quick_mode": True,
    "max_epochs": 2,
    "lookback": 32,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "demo_clip_months": 1,
}

# ---------------------------------------------------------------------------
# Canonical model-name mappings  (UI label -> backend filter value)
# ---------------------------------------------------------------------------
# The backend ``run_a_stat.py`` upper-cases TG_MODEL_FILTER and passes it to ``_fc()``. The
# names below are the canonical UI labels *and* the values the backend recognises.
#
# READ FROM THE REGISTRY, NOT TYPED HERE. This used to be three hand-written lists with a
# comment asking whoever adds a model to remember to update them. Measured on 2026-08-19,
# nobody had: the registry offered 15 machine-learning models and this file named 8, the
# quantile family offered 3 and this file named 2, and ETS_DAMPED was missing entirely. So
# **10 registered models could not be run from the Lab at all**, while the Models page told
# readers they could run any untested model there. A list that must be kept in sync by hand
# is a list that is silently wrong, and the stale copy is the one the reader sees.
#
# ``backend/model_catalog.py`` is deliberately free of sklearn, pandas and the boosters at
# module level precisely so this interpreter can read it. See its docstring.
#
# ONE TRAP, WORTH STATING. ``ModelSpec.installed`` calls ``find_spec`` in *whichever*
# interpreter asks, and this is the Streamlit one. XGBoost, LightGBM and CatBoost live in
# ``backend/.venv``, so from here they all report installed=False. Filtering on that flag would
# delete XGBoost and LightGBM from the Lab, which is exactly backwards: the run executes in the
# backend interpreter, where they are present. So these lists are UNFILTERED, and a genuinely
# missing package is reported by the run rather than guessed at here.
try:
    import sys as _sys
    from pathlib import Path as _Path

    _BACKEND = _Path(__file__).resolve().parent.parent / "backend"
    if str(_BACKEND) not in _sys.path:
        _sys.path.insert(0, str(_BACKEND))

    from model_catalog import (  # noqa: E402
        FAMILY_ML as _FAMILY_ML,
        QUANTILE_MODEL_NAMES as _QUANTILE_NAMES,
        STAT_MODEL_NAMES as _STAT_NAMES,
        specs as _specs,
    )

    STAT_MODEL_OPTIONS = [(name, name) for name in _STAT_NAMES]
    ML_MODEL_OPTIONS = [spec.name for spec in _specs(_FAMILY_ML)]
    QUANTILE_MODEL_OPTIONS = list(_QUANTILE_NAMES)
    MODEL_OPTIONS_SOURCE = "registry"
except Exception:                                  # noqa: BLE001 - the Lab must still open
    # A checkout without the backend tree, or a syntax error in the catalogue, must not take the
    # Lab down with it. The fallback is the smallest set known to exist in every build, and it
    # says so, so a short list is legible as a degraded read rather than as the whole shelf.
    STAT_MODEL_OPTIONS = [(n, n) for n in ("ETS", "SARIMAX", "STL_ARIMA", "THETA",
                                           "NAIVE", "WEEKDAY_MEAN", "MOVAVG")]
    ML_MODEL_OPTIONS = ["Ridge", "Lasso", "ElasticNet", "RandomForest",
                        "ExtraTrees", "HistGBDT", "XGBoost", "LightGBM"]
    QUANTILE_MODEL_OPTIONS = ["GBQuantile", "ResidualRF"]
    MODEL_OPTIONS_SOURCE = "fallback"

#: C_DL is not in the catalogue: its models are named in ``backend/c_dl_registry.py`` and the
#: Lab's own labels differ from those names (TCN/Transformer here, DCNN/TRANSFORMER there). Left
#: as it was rather than half-migrated, and noted so the inconsistency is visible.
DL_MODEL_OPTIONS = ["GRU", "LSTM", "TCN", "Transformer", "MLP"]

# Quality gate — minimum skill (%) over persistence baseline.
# Must be > 0.  Values below this threshold cause a FAILED_QUALITY status.
QUALITY_GATE_SKILL_PCT = 5.0

# ---------------------------------------------------------------------------
# Run-profile defaults  (profile, family) -> parameter overrides
# ---------------------------------------------------------------------------
# These centralize what "Demo", "Balanced", and "Thorough" mean for every
# model family so that Lab, Models-doc, and tests can all reference one source.

HORIZON_PRESETS = {
    "Daily":   [1, 5, 10, 20],
    "Weekly":  [1, 4, 8, 12],
    "Monthly": [1, 3, 6, 12],
}

PROFILE_DEFAULTS = {
    # ── A · Statistical ──────────────────────────────────────────
    ("Demo", "A_STAT"): {
        "folds": 1, "min_train_years": 0, "demo_clip_months": 12,
    },
    ("Balanced", "A_STAT"): {
        "folds": 3, "min_train_years": 2,
    },
    ("Thorough", "A_STAT"): {
        "folds": None, "min_train_years": 4,   # None = use ALL available folds
    },

    # ── B · Machine Learning ─────────────────────────────────────
    ("Demo", "B_ML"): {
        "folds": 1, "min_train_years": 0, "demo_clip_months": 12,
        "lags_daily": [1, 3, 7], "windows_daily": [3, 7],
        "lags_weekly": [1, 4], "windows_weekly": [4],
        "lags_monthly": [1, 3], "windows_monthly": [3],
    },
    ("Balanced", "B_ML"): {
        "folds": 3, "min_train_years": 2,
        "lags_daily": [1, 2, 3, 7, 14], "windows_daily": [3, 7, 14],
        "lags_weekly": [1, 4, 12], "windows_weekly": [4, 8, 12],
        "lags_monthly": [1, 3, 12], "windows_monthly": [3, 6, 12],
    },
    ("Thorough", "B_ML"): {
        "folds": 5, "min_train_years": 4,
        "lags_daily": [1, 2, 3, 5, 7, 14, 21, 28],
        "windows_daily": [3, 5, 7, 14, 21, 28],
        "lags_weekly": [1, 2, 4, 8, 12, 26],
        "windows_weekly": [4, 8, 12, 26],
        "lags_monthly": [1, 2, 3, 6, 12],
        "windows_monthly": [3, 6, 12],
    },

    # ── C · Deep Learning ────────────────────────────────────────
    ("Demo", "C_DL"): {
        "folds": 1, "min_train_years": 0, "demo_clip_months": 12,
        "max_epochs": 3, "lookback": 32, "batch_size": 128,
        "quick_mode": True,
    },
    ("Balanced", "C_DL"): {
        "folds": 3, "min_train_years": 2,
        "max_epochs": 20, "lookback": 64, "batch_size": 64,
    },
    ("Thorough", "C_DL"): {
        "folds": 5, "min_train_years": 4,
        "max_epochs": 50, "lookback": 128, "batch_size": 32,
        "valid_frac": 0.15, "thorough_mode": True,
    },

    # ── E · Quantile ─────────────────────────────────────────────
    ("Demo", "E_QUANTILE"): {
        "folds": 1, "min_train_years": 0, "demo_clip_months": 12,
    },
    ("Balanced", "E_QUANTILE"): {
        "folds": 3, "min_train_years": 2,
        "lags_daily": [1, 5, 20], "windows_daily": [5, 20],
    },
    ("Thorough", "E_QUANTILE"): {
        "folds": 5, "min_train_years": 4,
        "lags_daily": [1, 2, 3, 5, 7, 10, 14, 20],
        "windows_daily": [3, 5, 7, 14, 20],
    },
}
