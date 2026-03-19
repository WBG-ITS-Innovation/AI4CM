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
# The backend ``run_a_stat.py`` upper-cases TG_MODEL_FILTER and passes it to
# ``_fc()``.  The names below are the canonical UI labels *and* the values
# that the backend recognises after upper-casing.  Keep this mapping in sync
# whenever models are added/removed.

STAT_MODEL_OPTIONS = [
    # (UI label, backend filter sent via TG_MODEL_FILTER)
    ("ETS",          "ETS"),
    ("SARIMAX",      "SARIMAX"),
    ("STL_ARIMA",    "STL_ARIMA"),
    ("THETA",        "THETA"),
    ("NAIVE",        "NAIVE"),
    ("WEEKDAY_MEAN", "WEEKDAY_MEAN"),
    ("MOVAVG",       "MOVAVG"),
]

ML_MODEL_OPTIONS = [
    "Ridge", "Lasso", "ElasticNet", "RandomForest",
    "ExtraTrees", "HistGBDT", "XGBoost", "LightGBM",
]

DL_MODEL_OPTIONS = ["GRU", "LSTM", "TCN", "Transformer", "MLP"]

QUANTILE_MODEL_OPTIONS = ["GBQuantile", "ResidualRF"]

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
