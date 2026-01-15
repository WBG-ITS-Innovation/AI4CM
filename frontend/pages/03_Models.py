# pages/04_Models.py — Model Families & Parameters (documentation-style)
import json
from typing import Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Models • Georgia Treasury", layout="wide")
st.title("🧩 Model Families & Parameters")

RUNTIME_LEGEND = "⚡ very fast · ⏱ medium · 🐢 slower"

# ---------------------------- helpers ----------------------------
def dl_csv_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

def codejson(obj: Dict):
    st.code(json.dumps(obj, indent=2), language="json")

def filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q.strip():
        return df
    ql = q.strip().lower()
    mask = False
    for col in df.columns:
        mask = mask | df[col].astype(str).str.lower().str.contains(ql, na=False)
    return df[mask].copy()

# ---------------------------- Page intro ----------------------------
st.markdown(
    """
This page documents the model families available in the prototype and explains the meaning of the most common parameters.

**How to use this page**
- If you want conceptual understanding: start with the family overview under each tab.
- If you want practical tuning: use the parameter tables and the detailed notes under “Details”.
- If you want to run models: use the **Lab** page (Family + Model + Variant + Horizon). This page explains what those choices do.

The goal is to make model behavior and configuration auditable and reproducible.
"""
)

with st.expander("How this maps to the Lab page (Family / Model / Variant / Overrides)", expanded=False):
    st.markdown(
        """
In the **Lab** page you choose:

- **Family**: A / B / C / E  
- **Model**: e.g., ETS, Ridge, LightGBM, LSTM, GBQuantile  
- **Variant**:
  - **Univariate** uses only the target history
  - **Multivariate** can incorporate additional columns as candidate predictors (where supported)
- **Overrides JSON**: an optional configuration object passed to the backend runner.  
  This is exposed for transparency and power-users.

This Models page describes those parameters and common defaults.
"""
    )

st.markdown("---")

# ---------------------------- Defaults + tables ----------------------------
def defaults_a():
    return {
        "daily":   {"seasonal_periods": 7,  "cv_folds": 5, "min_train_years": 3},
        "weekly":  {"seasonal_periods": 52, "cv_folds": 5, "min_train_years": 4},
        "monthly": {"seasonal_periods": 12, "cv_folds": 5, "min_train_years": 5},
    }

def table_a() -> pd.DataFrame:
    rows = [
        dict(Model="NaiveLast", Parameter="—",
             Meaning="Forecast equals the last observed value.",
             WhyItMatters="Establishes a sanity baseline and helps detect random-walk behavior.",
             Suggested="No tuning.", Runtime="⚡"),
        dict(Model="WeekdayMean", Parameter="weeks_back",
             Meaning="Average of same weekday over the last N weeks (daily flows).",
             WhyItMatters="Captures weekly operational patterns without complex modeling.",
             Suggested="4–8", Runtime="⚡"),
        dict(Model="MovingAverage", Parameter="window",
             Meaning="Simple moving mean over the last W periods.",
             WhyItMatters="Smooths noise; reduces sensitivity to day-to-day spikes.",
             Suggested="Daily: 7/14/28; Weekly: 4/8/12; Monthly: 3/6/12", Runtime="⚡"),
        dict(Model="ETS", Parameter="trend, seasonal, seasonal_periods, damped_trend",
             Meaning="Exponential smoothing with level/trend/seasonality components.",
             WhyItMatters="Often strong baseline with interpretable components.",
             Suggested="trend='add', seasonal='add', sp=7|52|12, damped=True", Runtime="⏱"),
        dict(Model="SARIMAX", Parameter="order=(p,d,q), seasonal_order=(P,D,Q,s)",
             Meaning="ARIMA-family with seasonal terms; can support exogenous inputs.",
             WhyItMatters="Captures autocorrelation structure; flexible but can be slower to tune.",
             Suggested="Keep (p,q,P,Q) ≤ 2; set s=7|52|12", Runtime="⏱/🐢"),
        dict(Model="STL-ARIMA", Parameter="stl_season_length, arima_order, robust",
             Meaning="Decompose seasonality/trend via STL, then ARIMA on remainder.",
             WhyItMatters="Useful when seasonality is stable and outliers exist (robust STL).",
             Suggested="stl=7|12; arima=(0,1,1) or (1,0,1); robust=True for spikes", Runtime="⏱"),
        dict(Model="Theta", Parameter="theta, seasonal_periods",
             Meaning="Competition-grade method: trend + smoothing.",
             WhyItMatters="Strong baseline, low tuning burden.",
             Suggested="theta=2; sp=7|52|12", Runtime="⚡"),
    ]
    return pd.DataFrame(rows)

def overrides_a() -> Dict:
    return {
        "models": ["NaiveLast", "WeekdayMean", "MovingAverage", "ETS", "SARIMAX", "STL-ARIMA", "Theta"],
        "seasonal_periods": 7,
        "damped_trend": True,
        "cv_folds": 5,
        "min_train_years": 3,
        "use_stock_flow_rules": True,
    }

def defaults_b():
    return {
        "daily":   {"lags": [1,2,3,7,14,21,28], "rolling_windows": [7,14,28], "cv_folds": 5, "min_train_years": 2},
        "weekly":  {"lags": [1,2,3,4,8,12,26],  "rolling_windows": [4,8,12],  "cv_folds": 5, "min_train_years": 3},
        "monthly": {"lags": [1,2,3,6,12,24],    "rolling_windows": [3,6,12],  "cv_folds": 5, "min_train_years": 4},
    }

def table_b() -> pd.DataFrame:
    rows = [
        dict(Model="Preprocessing", Parameter="lags",
             Meaning="Past target values used as features.",
             WhyItMatters="Captures seasonality/momentum; main signal in many ML setups.",
             Suggested="Daily: 1,2,3,7,14,21,28; Weekly: 1,4,12; Monthly: 1,3,12", Runtime="⚡"),
        dict(Model="Preprocessing", Parameter="rolling_windows",
             Meaning="Trailing window summaries (rolling mean/std, etc.).",
             WhyItMatters="Adds stability and context, especially on noisy flows.",
             Suggested="Daily: 7/14/28; Weekly: 4/8/12; Monthly: 3/6/12", Runtime="⚡"),
        dict(Model="Preprocessing (multi)", Parameter="exog_top_k",
             Meaning="Top-K exogenous columns to include (multivariate).",
             WhyItMatters="Controls dimensionality and reduces overfitting risk.",
             Suggested="Start 5–15, increase cautiously", Runtime="⏱"),
        dict(Model="Ridge", Parameter="alpha",
             Meaning="L2 regularization strength.",
             WhyItMatters="Higher alpha reduces overfit on noisy/collinear features.",
             Suggested="0.1 / 1.0 / 10.0", Runtime="⚡"),
        dict(Model="Lasso", Parameter="alpha",
             Meaning="L1 regularization strength (sparsity).",
             WhyItMatters="Can select features automatically; may underfit if too strong.",
             Suggested="0.01 / 0.1 / 1.0", Runtime="⚡"),
        dict(Model="ElasticNet", Parameter="alpha, l1_ratio",
             Meaning="Mix of L1 and L2 regularization.",
             WhyItMatters="Balances shrinkage and feature selection.",
             Suggested="alpha=0.1–1.0, l1_ratio=0.2–0.8", Runtime="⚡"),
        dict(Model="RandomForest", Parameter="n_estimators, max_depth, min_samples_leaf",
             Meaning="Bagged trees for nonlinear effects.",
             WhyItMatters="Robust but slower; can overfit without min_samples_leaf.",
             Suggested="n=400–800; leaf=1–10", Runtime="⏱"),
        dict(Model="ExtraTrees", Parameter="n_estimators, max_depth",
             Meaning="More randomized tree ensemble.",
             WhyItMatters="Can perform well on noisy data; similar tuning to RF.",
             Suggested="n=400–800; depth=None", Runtime="⏱"),
        dict(Model="HistGBDT", Parameter="learning_rate, max_depth, max_leaf_nodes",
             Meaning="Efficient gradient boosting (sklearn).",
             WhyItMatters="Strong performance with manageable tuning.",
             Suggested="lr=0.05–0.1; depth=3–7", Runtime="⏱"),
        dict(Model="XGBoost", Parameter="n_estimators, max_depth, eta, subsample, colsample_bytree, reg_lambda",
             Meaning="Boosted trees (xgboost).",
             WhyItMatters="Often top accuracy; slower; requires regularization to prevent overfit.",
             Suggested="depth=3–6; eta=0.03–0.1; subs/cols=0.7–0.9; λ=1–5", Runtime="⏱/🐢"),
        dict(Model="LightGBM", Parameter="n_estimators, num_leaves, learning_rate, feature_fraction, bagging_fraction, lambda_l2, min_data_in_leaf",
             Meaning="Boosting with leaf-wise growth (lightgbm).",
             WhyItMatters="Very strong but can overfit if leaves are large and min_data_in_leaf is small.",
             Suggested="leaves=31–127; lr=0.03–0.07; minleaf=20–60", Runtime="⏱/🐢"),
    ]
    return pd.DataFrame(rows)

def overrides_b() -> Dict:
    return {
        "lags": [1,2,3,7,14,21,28],
        "rolling_windows": [7,14,28],
        "models": ["Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"],
        "feature_scaler": "StandardScaler",
        "target_transform": None,
        "cv_folds": 5,
        "min_train_years": 2,
        "exog_top_k": 20,
    }

def defaults_c():
    return {
        "daily":   {"lookback": 90,  "horizon": 14, "epochs": 50,  "batch_size": 64},
        "weekly":  {"lookback": 104, "horizon": 8,  "epochs": 80,  "batch_size": 32},
        "monthly": {"lookback": 60,  "horizon": 12, "epochs": 100, "batch_size": 16},
    }

def table_c() -> pd.DataFrame:
    rows = [
        dict(Model="Global", Parameter="lookback",
             Meaning="Sequence length fed to the model.",
             WhyItMatters="Must be long enough to capture seasonal cycles and regime changes.",
             Suggested="Daily: 60–120; Weekly: 80–120; Monthly: 36–60", Runtime="⏱"),
        dict(Model="Global", Parameter="batch_size",
             Meaning="Mini-batch size during training.",
             WhyItMatters="Impacts speed and memory; too large can cause memory errors.",
             Suggested="CPU: 16–64; GPU: 32–128", Runtime="—"),
        dict(Model="Global", Parameter="max_epochs / early stopping",
             Meaning="Training duration and stopping behavior.",
             WhyItMatters="More epochs can improve accuracy but increases runtime and overfit risk.",
             Suggested="Exploration: 3–15; Final: 30–100", Runtime="🐢"),
        dict(Model="LSTM/GRU", Parameter="hidden_size, num_layers, dropout",
             Meaning="Capacity and regularization knobs.",
             WhyItMatters="Higher capacity fits complex patterns but increases overfit risk.",
             Suggested="hidden=64–128; layers=1–2; dropout=0.1–0.3", Runtime="⏱/🐢"),
        dict(Model="TCN", Parameter="levels, kernel_size, dropout",
             Meaning="Causal convolutions with dilation (receptive field).",
             WhyItMatters="Controls how far back the model can “see” effectively.",
             Suggested="levels=5–7; kernel=3–5; dropout=0.1–0.3", Runtime="⏱"),
        dict(Model="Transformer", Parameter="d_model, nhead, num_layers, dim_ff, dropout",
             Meaning="Attention-based sequence model configuration.",
             WhyItMatters="Powerful but can overfit on small datasets; heavier runtime.",
             Suggested="d=64–128; heads=4–8; layers=2; dropout=0.1", Runtime="🐢"),
        dict(Model="MLP", Parameter="hidden_dims",
             Meaning="Feed-forward network on flattened windows.",
             WhyItMatters="Fast to train, but may struggle with long seasonal dependencies.",
             Suggested="[128, 64] + dropout 0.2", Runtime="⏱"),
    ]
    return pd.DataFrame(rows)

def overrides_c() -> Dict:
    return {
        "architecture": "LSTM",
        "lookback": 90,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 64,
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "early_stopping": True,
        "conformal_alpha": 0.1,
    }

def defaults_e():
    return {
        "daily":   {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
        "weekly":  {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 600, "max_depth": 3, "learning_rate": 0.05},
        "monthly": {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 800, "max_depth": 3, "learning_rate": 0.03},
    }

def table_e() -> pd.DataFrame:
    rows = [
        dict(Model="GBQuantile", Parameter="quantiles",
             Meaning="Probability levels for lower/median/upper forecasts.",
             WhyItMatters="Used for scenario-based planning; not just point predictions.",
             Suggested="[0.1, 0.5, 0.9] (P10/P50/P90)", Runtime="⏱"),
        dict(Model="GBQuantile", Parameter="n_estimators, learning_rate",
             Meaning="Number of trees and boosting step size.",
             WhyItMatters="Main bias/variance tradeoff; lower lr typically needs more trees.",
             Suggested="n=400–800; lr=0.03–0.07", Runtime="⏱/🐢"),
        dict(Model="GBQuantile", Parameter="max_depth",
             Meaning="Tree depth / interaction complexity.",
             WhyItMatters="Deeper trees fit noise easily on small datasets.",
             Suggested="3–5 (prefer 3–4 on treasury flows)", Runtime="⏱"),
        dict(Model="GBQuantile", Parameter="min_samples_leaf / min_child_weight",
             Meaning="Minimum leaf size regularization.",
             WhyItMatters="Higher values reduce variance/overfit risk.",
             Suggested="5–20", Runtime="⏱"),
    ]
    return pd.DataFrame(rows)

def overrides_e() -> Dict:
    return {
        "quantiles": [0.1, 0.5, 0.9],
        "lags": [1,2,3,7,14,21,28],
        "rolling_windows": [7,14,28],
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 3,
        "cv_folds": 5,
    }

# ---------------------------- UI: tabs + filtering ----------------------------
tabs = st.tabs(["A · Statistical", "B · Machine Learning", "C · Deep Learning", "E · Quantile", "Glossary"])

with tabs[0]:
    st.header("A · Statistical")
    st.markdown(
        """
Statistical models explicitly represent trend and seasonality (when applicable).  
They are often strong baselines and are usually easier to audit and explain.
"""
    )
    st.subheader("Defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_a()).T.rename_axis("cadence"), use_container_width=True)

    st.subheader("Parameter reference")
    q = st.text_input("Filter A-family parameters (search)", "", key="filter_a")
    a_df = filter_df(table_a(), q)
    st.dataframe(a_df, use_container_width=True)
    dl_csv_button(a_df, "Download A-family table (CSV)", "A_stat_parameters.csv")

    with st.expander("Details and guidance", expanded=False):
        st.markdown(
            """
**ETS**
- `trend`: how the level evolves over time (additive/multiplicative/None)
- `seasonal`: repeating pattern form (additive/multiplicative/None)
- `seasonal_periods`: 7 (daily), 52 (weekly), 12 (monthly)
- `damped_trend`: prevents explosive long-run trend projections

**SARIMAX**
- `order=(p,d,q)` controls autoregressive and moving-average structure
- `seasonal_order=(P,D,Q,s)` is the seasonal analogue; `s` is the seasonal period
- Tuning guidance: keep orders small unless you have very long history and strong autocorrelation structure

**STL-ARIMA**
- Helps when seasonality is stable and the series includes occasional spikes
- `robust=True` reduces the impact of outliers during decomposition

**Theta**
- Strong baseline with minimal tuning in many real-world forecasting tasks
"""
        )

    st.subheader("Example Overrides JSON")
    st.caption("These examples are intended as reference. The Lab page constructs overrides automatically based on your selections.")
    codejson(overrides_a())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[1]:
    st.header("B · Machine Learning")
    st.markdown(
        """
Machine Learning models rely on engineered features such as lags and rolling windows.  
They can capture nonlinearities and interactions that statistical models may miss, but require careful feature control to avoid overfitting.
"""
    )
    st.subheader("Defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_b()).T.rename_axis("cadence"), use_container_width=True)

    st.subheader("Parameter reference")
    q = st.text_input("Filter B-family parameters (search)", "", key="filter_b")
    b_df = filter_df(table_b(), q)
    st.dataframe(b_df, use_container_width=True)
    dl_csv_button(b_df, "Download B-family table (CSV)", "B_ml_parameters.csv")

    with st.expander("Details and guidance", expanded=False):
        st.markdown(
            """
**Feature engineering**
- `lags`: past target values used as features; include lags around seasonality (e.g., 7/14/28 daily)
- `rolling_windows`: trailing summaries (rolling mean/std); stabilizes signal on volatile flows
- Multivariate: `exog_top_k` constrains how many extra columns enter the model

**Regularization**
- Linear models depend heavily on regularization:
  - Ridge: shrink coefficients when features are noisy/collinear
  - Lasso: encourages sparsity (feature selection)
  - ElasticNet: combines both

**Boosting models (XGBoost/LightGBM)**
- Most important controls:
  - model complexity: `max_depth` (XGBoost), `num_leaves` (LightGBM)
  - regularization: `min_data_in_leaf`, `lambda_l2`
  - generalization: `subsample`, `feature_fraction`

A common failure mode is overfitting: training error improves but out-of-sample performance degrades.  
Feature control (`exog_top_k`, `min_data_in_leaf`) is often more impactful than adding complexity.
"""
        )

    st.subheader("Example Overrides JSON")
    st.caption("These examples are intended as reference. The Lab page constructs overrides automatically based on your selections.")
    codejson(overrides_b())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[2]:
    st.header("C · Deep Learning")
    st.markdown(
        """
Deep learning models can learn complex temporal patterns, but they generally require:
- sufficient history (data volume)
- careful regularization
- more runtime for training

In small or noisy datasets, simpler families may outperform deep learning.
"""
    )
    st.subheader("Defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_c()).T.rename_axis("cadence"), use_container_width=True)

    st.subheader("Parameter reference")
    q = st.text_input("Filter C-family parameters (search)", "", key="filter_c")
    c_df = filter_df(table_c(), q)
    st.dataframe(c_df, use_container_width=True)
    dl_csv_button(c_df, "Download C-family table (CSV)", "C_dl_parameters.csv")

    with st.expander("Details and guidance", expanded=False):
        st.markdown(
            """
**Core parameters**
- `lookback`: controls how much history the model sees; too short misses seasonality, too long increases runtime
- `batch_size`: affects speed and memory; reduce if you see memory errors
- `max_epochs`: determines training duration; start small and increase once stable

**Architectures**
- LSTM/GRU: general-purpose sequence models
- TCN: convolutional architecture with controllable receptive field
- Transformer: attention-based; can overfit on small datasets unless constrained

**Uncertainty**
If conformal calibration is enabled in the backend, additional parameters control how uncertainty bands are calibrated.
"""
        )

    st.subheader("Example Overrides JSON")
    st.caption("These examples are intended as reference. The Lab page constructs overrides automatically based on your selections.")
    codejson(overrides_c())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[3]:
    st.header("E · Quantile")
    st.markdown(
        """
Quantile models produce a distributional forecast rather than a single point prediction.  
This is useful when decisions depend on downside/upside risk (e.g., conservative vs optimistic planning).
"""
    )
    st.subheader("Defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_e()).T.rename_axis("cadence"), use_container_width=True)

    st.subheader("Parameter reference")
    q = st.text_input("Filter E-family parameters (search)", "", key="filter_e")
    e_df = filter_df(table_e(), q)
    st.dataframe(e_df, use_container_width=True)
    dl_csv_button(e_df, "Download E-family table (CSV)", "E_quantile_parameters.csv")

    with st.expander("Details and guidance", expanded=False):
        st.markdown(
            """
**Quantiles**
- `[0.1, 0.5, 0.9]` corresponds to P10 / P50 / P90
- Wider bands (e.g., 0.05 and 0.95) can be used when you want more conservative uncertainty

**Boosting controls**
- `n_estimators` and `learning_rate` trade off speed and accuracy
- Keep trees shallow (`max_depth`) on treasury flows unless you have long history and clean signals
"""
        )

    st.subheader("Example Overrides JSON")
    st.caption("These examples are intended as reference. The Lab page constructs overrides automatically based on your selections.")
    codejson(overrides_e())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[4]:
    st.header("Glossary")
    st.markdown(
        """
- **Target**: the time series you want to forecast (e.g., Revenues)  
- **Cadence**: time resolution (Daily / Weekly / Monthly)  
- **Horizon**: how many steps ahead to predict at the chosen cadence  
- **Seasonal period**: repeating cycle length (7 daily, 52 weekly, 12 monthly)  
- **Lags**: past target values used as model inputs  
- **Rolling windows**: trailing summary features (e.g., mean over last 7 periods)  
- **Cross-validation folds**: repeated evaluation over multiple time splits  
- **Exogenous features**: additional columns used as candidate predictors (multivariate)  
- **Quantiles (P10/P50/P90)**: distributional forecasts for uncertainty-aware planning  
"""
    )
