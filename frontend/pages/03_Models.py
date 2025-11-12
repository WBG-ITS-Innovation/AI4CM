# pages/04_Models.py
import json
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Models • Georgia Treasury", layout="wide")
st.title("🧩 Model Families & Parameters")

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

RUNTIME_LEGEND = "⚡ very fast · ⏱ medium · 🐢 slower"

# ---------------------------- A · Statistical ----------------------------
def defaults_a():
    return {
        "daily":   {"seasonal_periods": 7,  "cv_folds": 5, "min_train_years": 3},
        "weekly":  {"seasonal_periods": 52, "cv_folds": 5, "min_train_years": 4},
        "monthly": {"seasonal_periods": 12, "cv_folds": 5, "min_train_years": 5},
    }

def table_a() -> pd.DataFrame:
    rows = [
        dict(Model="NaiveLast", Parameter="—",
             What="Forecast equals last observed value.", Suggested="No knobs.",
             Tuning="Sanity baseline; catches random-walk behavior.", Runtime="⚡"),
        dict(Model="WeekdayMean", Parameter="weeks_back",
             What="Avg of same weekday over last N weeks (flows).", Suggested="4 (daily flows)",
             Tuning="Increase to 6–8 if volatile.", Runtime="⚡"),
        dict(Model="MovingAverage", Parameter="window",
             What="Simple moving mean over last W periods.",
             Suggested="7/14/28 (daily), 4/8 (weekly), 3/6 (monthly)",
             Tuning="Shorter=responsive; longer=smoother.", Runtime="⚡"),
        dict(Model="ETS", Parameter="trend, seasonal, seasonal_periods, damped_trend",
             What="Exp smoothing with trend + seasonality.",
             Suggested="trend='add', seasonal='add', sp=7|52|12, damped=True",
             Tuning="Use seasonal='mul' if positive with scaling seasonality.", Runtime="⏱"),
        dict(Model="SARIMAX", Parameter="order=(p,d,q), seasonal_order=(P,D,Q,s)",
             What="ARIMA + seasonal part; exog optional.",
             Suggested="Daily flows: (1,0,1)(1,1,0,7); Monthly: (0,1,1)(0,1,1,12)",
             Tuning="Keep (p,q),(P,Q) ≤ 2; raise D only for seasonal unit root.", Runtime="⏱/🐢"),
        dict(Model="STL-ARIMA", Parameter="stl_season_length, arima_order",
             What="Decompose seasonal/trend (STL), model remainder with ARIMA.",
             Suggested="stl=7|12; arima=(0,1,1) or (1,0,1)",
             Tuning="robust=True for outliers; excels with stable seasonality.", Runtime="⏱"),
        dict(Model="Theta", Parameter="theta, seasonal_periods",
             What="Linear trend + SES (M3 winner).",
             Suggested="theta=2, sp=7|52|12",
             Tuning="Strong, simple baseline; rarely needs tuning.", Runtime="⚡"),
    ]
    return pd.DataFrame(rows)

def overrides_a() -> Dict:
    return {
        "models": ["NaiveLast","WeekdayMean","MovingAverage","ETS","SARIMAX","STL-ARIMA","Theta"],
        "seasonal_periods": 7, "damped_trend": True,
        "cv_folds": 5, "min_train_years": 3,
        "use_stock_flow_rules": True
    }

# ---------------------------- B · Machine Learning ----------------------------
def defaults_b():
    return {
        "daily":   {"lags": [1,2,3,7,14,21,28], "windows": [7,14,28], "cv_folds": 5, "min_train_years": 2},
        "weekly":  {"lags": [1,2,3,4,8,12,26],  "windows": [4,8,12],  "cv_folds": 5, "min_train_years": 3},
        "monthly": {"lags": [1,2,3,6,12,24],    "windows": [3,6,12],  "cv_folds": 5, "min_train_years": 4},
    }

def table_b() -> pd.DataFrame:
    rows = [
        dict(Model="Preprocessing", Parameter="lags",
             What="Past target values as features.",
             Suggested="Daily: 1,2,3,7,14,21,28",
             Tuning="Add around dominant seasonalities; avoid overly dense sets.", Runtime="⚡"),
        dict(Model="Preprocessing", Parameter="rolling_windows",
             What="Trailing window stats (mean/std).",
             Suggested="7/14/28 (daily), 4/8/12 (weekly), 3/6/12 (monthly)",
             Tuning="Use trailing windows only (no leakage).", Runtime="⚡"),
        dict(Model="Preprocessing (multi)", Parameter="exog_top_k",
             What="Top-K exogenous features by importance.",
             Suggested="10–30", Tuning="Start 10; increase if stable & sample size large.", Runtime="⏱"),
        dict(Model="Ridge", Parameter="alpha",
             What="L2 regularization strength.",
             Suggested="0.1 / 1.0 / 10.0", Tuning="Higher for noisy/collinear features.", Runtime="⚡"),
        dict(Model="Lasso", Parameter="alpha",
             What="L1 regularization (sparsity).",
             Suggested="0.01 / 0.1 / 1.0", Tuning="Great for selection; avoid underfit.", Runtime="⚡"),
        dict(Model="ElasticNet", Parameter="alpha, l1_ratio",
             What="Mix of L1 and L2.", Suggested="alpha=0.1–1.0, l1_ratio=0.2–0.8",
             Tuning="When both selection & shrinkage help.", Runtime="⚡"),
        dict(Model="RandomForest", Parameter="n_estimators, max_depth, min_samples_leaf",
             What="Bagged trees for nonlinearities.",
             Suggested="n=400–800, depth=None, leaf=1–5",
             Tuning="Increase leaf for less variance; depth=None often fine.", Runtime="⏱"),
        dict(Model="ExtraTrees", Parameter="n_estimators, max_depth",
             What="Extremely randomized trees.",
             Suggested="n=400–800, depth=None",
             Tuning="Often slightly better than RF on noisy series.", Runtime="⏱"),
        dict(Model="HistGradientBoosting", Parameter="learning_rate, max_depth, max_leaf_nodes",
             What="Efficient gradient boosting.",
             Suggested="lr=0.05–0.1, depth=3–7, leaves=31–127",
             Tuning="Lower lr → higher n_estimators.", Runtime="⏱"),
        dict(Model="XGBoost", Parameter="n_estimators, max_depth, eta, subsample, colsample_bytree, reg_lambda",
             What="Tree boosting (xgboost).",
             Suggested="n=600–1200, depth=3–6, eta=0.03–0.1, subs=0.7–0.9, cols=0.7–0.9, λ=1–5",
             Tuning="Small depth with more trees; early stopping.", Runtime="⏱/🐢"),
        dict(Model="LightGBM", Parameter="n_estimators, num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction, lambda_l2, min_data_in_leaf",
             What="Leaf-wise boosting (lightgbm).",
             Suggested="n=800–1500, leaves=31–127, depth=-1, lr=0.03–0.07, feat=0.8, bag=0.8, λ2=1–5, minleaf=20–60",
             Tuning="Raise minleaf to fight overfit; tune leaves with lr.", Runtime="⏱/🐢"),
    ]
    return pd.DataFrame(rows)

def overrides_b() -> Dict:
    return {
        "lags": [1,2,3,7,14,21,28],
        "rolling_windows": [7,14,28],
        "models": ["Ridge","ElasticNet","RandomForest","XGBoost","LightGBM"],
        "feature_scaler": "StandardScaler",
        "target_transform": None,
        "cv_folds": 5,
        "min_train_years": 2,
        "exog_top_k": 20
    }

# ---------------------------- C · Deep Learning ----------------------------
def defaults_c():
    return {
        "daily":   {"lookback": 90,  "horizon": 14, "epochs": 50,  "batch_size": 64},
        "weekly":  {"lookback": 104, "horizon": 8,  "epochs": 80,  "batch_size": 32},
        "monthly": {"lookback": 60,  "horizon": 12, "epochs": 100, "batch_size": 16},
    }

def table_c() -> pd.DataFrame:
    rows = [
        dict(Model="Global", Parameter="lookback",
             What="Sequence length fed to the net.",
             Suggested="Daily 60–120; Weekly 80–120; Monthly 36–60",
             Tuning="Cover ≥ 2 key seasonal cycles.", Runtime="⏱"),
        dict(Model="Global", Parameter="batch_size",
             What="Mini-batch size for SGD.",
             Suggested="32–128 (GPU), 16–64 (CPU)",
             Tuning="Larger → smoother gradients; watch memory.", Runtime="—"),
        dict(Model="Global", Parameter="epochs, early_stopping",
             What="Training iterations + patience.",
             Suggested="30–100 epochs, patience=5–10",
             Tuning="Quick runs (8–15) for selection; longer to finalize.", Runtime="🐢"),
        dict(Model="Global", Parameter="learning_rate, weight_decay",
             What="Optimizer step & L2 regularization.",
             Suggested="lr=1e-3; wd=1e-4",
             Tuning="Reduce lr to 1e-4 if loss noisy/unstable.", Runtime="—"),
        dict(Model="LSTM/GRU", Parameter="hidden_size, num_layers, dropout",
             What="Capacity & regularization.",
             Suggested="hidden=64–128, layers=2, dropout=0.1–0.3",
             Tuning="Increase hidden_size first; add dropout as needed.", Runtime="⏱/🐢"),
        dict(Model="TCN", Parameter="levels, kernel_size, dropout",
             What="Dilated causal conv net.",
             Suggested="levels=5–7, k=3–5, dropout=0.1–0.3",
             Tuning="Tune receptive field to cover lookback.", Runtime="⏱"),
        dict(Model="Transformer", Parameter="d_model, nhead, num_layers, dim_ff, dropout",
             What="Self-attention encoder.",
             Suggested="d=64–128, heads=4–8, layers=2, ff=128–256, dropout=0.1",
             Tuning="Keep small on limited data; regularize.", Runtime="🐢"),
        dict(Model="MLP", Parameter="hidden_dims",
             What="Feed-forward on flattened window.",
             Suggested="[128, 64] + dropout 0.2",
             Tuning="Fast; weaker on long seasonalities.", Runtime="⏱"),
    ]
    return pd.DataFrame(rows)

def overrides_c() -> Dict:
    return {
        "architecture": "LSTM",
        "lookback": 90,
        "hidden_size": 64, "num_layers": 2, "dropout": 0.1,
        "batch_size": 64, "max_epochs": 50, "learning_rate": 1e-3,
        "early_stopping": True, "conformal_alpha": 0.1
    }

# ---------------------------- E · Quantile ----------------------------
def defaults_e():
    return {
        "daily":   {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
        "weekly":  {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 600, "max_depth": 3, "learning_rate": 0.05},
        "monthly": {"quantiles": [0.1, 0.5, 0.9], "n_estimators": 800, "max_depth": 3, "learning_rate": 0.03},
    }

def table_e() -> pd.DataFrame:
    rows = [
        dict(Model="GBQuantile", Parameter="quantiles",
             What="Probability levels for lower/median/upper.",
             Suggested="[0.1, 0.5, 0.9]",
             Tuning="Add 0.05/0.95 for wider bands; calibrate with conformal.", Runtime="⏱"),
        dict(Model="GBQuantile", Parameter="n_estimators, learning_rate",
             What="#trees and step size.",
             Suggested="n=400–800, lr=0.03–0.07",
             Tuning="Lower lr → more trees; early stop if available.", Runtime="⏱/🐢"),
        dict(Model="GBQuantile", Parameter="max_depth",
             What="Tree depth (interaction complexity).",
             Suggested="3–5",
             Tuning="Prefer 3–4 on treasury series.", Runtime="⏱"),
        dict(Model="GBQuantile", Parameter="min_samples_leaf / min_child_weight",
             What="Regularization via minimum leaf size.",
             Suggested="5–20", Tuning="Raise to reduce variance.", Runtime="⏱"),
    ]
    return pd.DataFrame(rows)

def overrides_e() -> Dict:
    return {
        "quantiles": [0.1, 0.5, 0.9],
        "lags": [1,2,3,7,14,21,28], "rolling_windows": [7,14,28],
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3,
        "cv_folds": 5
    }

# ---------------------------- Page layout ----------------------------
tabs = st.tabs(["A · Statistical", "B · Machine Learning", "C · Deep Learning", "E · Quantile", "Glossary"])

with tabs[0]:
    st.header("A · Statistical (ETS, SARIMAX, STL-ARIMA, Theta)")
    st.write("Transparent models with seasonality/trend; strong on short horizons with clear calendars.")
    st.subheader("Recommended defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_a()).T.rename_axis("cadence"), use_container_width=True)
    st.subheader("Parameter cheat sheet")
    a_df = table_a(); st.dataframe(a_df, use_container_width=True); dl_csv_button(a_df, "Download A-family table (CSV)", "A_stat_parameters.csv")
    with st.expander("Details (definitions & tips)"):
        st.markdown("""
- **ETS**  
  - `trend` (*add/mul/None*): how level evolves; *mul* only if series is positive and seasonality scales.  
  - `seasonal` (*add/mul/None*): repeating pattern form.  
  - `seasonal_periods` (*int*): 7 (daily), 52 (weekly), 12 (monthly).  
  - `damped_trend` (*bool*): slows trend to avoid runaway forecasts.
- **SARIMAX**  
  - `order=(p,d,q)`: AR lags, differencing, MA lags.  
  - `seasonal_order=(P,D,Q,s)`: seasonal analogue with period `s`.  
  - Keep `(p,q),(P,Q) ≤ 2`; raise `D` only if seasonal unit root obvious.
- **STL-ARIMA**  
  - `stl_season_length`: match cadence period; `robust=True` down-weights outliers.
- **Theta**  
  - `theta`: slope weight; `2` is classic.
        """)
    st.subheader("Copy-ready OVERRIDES JSON")
    codejson(overrides_a())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[1]:
    st.header("B · Machine Learning (linear + trees/boosting)")
    st.write("Feature-based forecasting using lags, rolling stats, holidays, and (optionally) exogenous signals.")
    st.subheader("Recommended defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_b()).T.rename(columns={"windows":"rolling_windows"}).rename_axis("cadence"), use_container_width=True)
    st.subheader("Parameter cheat sheet")
    b_df = table_b(); st.dataframe(b_df, use_container_width=True); dl_csv_button(b_df, "Download B-family table (CSV)", "B_ml_parameters.csv")
    with st.expander("Details (definitions & tips)"):
        st.markdown("""
- **Preprocessing**  
  - `lags` (*list[int]*): past target values; pick around the main seasonalities (7/14/28).  
  - `rolling_windows` (*list[int]*): trailing mean/std; **no leakage** (strictly past).  
  - `exog_top_k` (*int*): cap exogenous features to control variance.
- **Linear models**  
  - `alpha` (Ridge/Lasso): L2 (shrinkage) vs L1 (sparsity).  
  - `l1_ratio` (ElasticNet): mix of L1/L2.
- **Trees/Boosting**  
  - `n_estimators`: more trees with smaller `learning_rate`.  
  - `max_depth`/`num_leaves`: interaction complexity; keep small on small data.  
  - `min_samples_leaf`/`min_child_weight`: raise to reduce variance.  
  - `subsample`/`feature_fraction`: stochasticity for generalization.
        """)
    st.subheader("Copy-ready OVERRIDES JSON")
    codejson(overrides_b())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[2]:
    st.header("C · Deep Learning (sequence models)")
    st.write("Use LSTM/GRU for general cases, TCN for long receptive fields, Transformers sparingly on small datasets.")
    st.subheader("Recommended defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_c()).T.rename_axis("cadence"), use_container_width=True)
    st.subheader("Parameter cheat sheet")
    c_df = table_c(); st.dataframe(c_df, use_container_width=True); dl_csv_button(c_df, "Download C-family table (CSV)", "C_dl_parameters.csv")
    with st.expander("Details (definitions & tips)"):
        st.markdown("""
- **Common**  
  - `lookback`: sequence length; cover ≥ 2 major seasonal cycles.  
  - `batch_size`: 32–128 (GPU), 16–64 (CPU).  
  - `max_epochs` + `early_stopping`: quick (8–15) for selection; longer to finalize.  
  - `learning_rate`/`weight_decay`: 1e-3 / 1e-4 start points.
- **LSTM/GRU**  
  - `hidden_size`, `num_layers`, `dropout`: increase hidden before layers; add dropout to regularize.
- **TCN**  
  - `levels`, `kernel_size`, `dropout`: tune receptive field (levels×dilation×kernel) to cover lookback.
- **Transformer**  
  - `d_model`, `nhead`, `num_layers`, `dim_ff`, `dropout`: keep small (e.g., d_model 64–128) with strong regularization.
        """)
    st.subheader("Copy-ready OVERRIDES JSON")
    codejson(overrides_c())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[3]:
    st.header("E · Quantile (risk-aware ranges)")
    st.write("Outputs P10/P50/P90 for planning under uncertainty.")
    st.subheader("Recommended defaults by cadence")
    st.dataframe(pd.DataFrame(defaults_e()).T.rename_axis("cadence"), use_container_width=True)
    st.subheader("Parameter cheat sheet")
    e_df = table_e(); st.dataframe(e_df, use_container_width=True); dl_csv_button(e_df, "Download E-family table (CSV)", "E_quantile_parameters.csv")
    with st.expander("Details (definitions & tips)"):
        st.markdown("""
- `quantiles`: e.g., `[0.1, 0.5, 0.9]` → P10/P50/P90.  
- `n_estimators` & `learning_rate`: lower lr → more trees; prefer shallow trees.  
- `max_depth`: 3–5; deeper overfits easily on small data.  
- `min_samples_leaf`/`min_child_weight`: raise to reduce variance.
        """)
    st.subheader("Copy-ready OVERRIDES JSON")
    codejson(overrides_e())
    st.caption(f"*Runtime guide:* {RUNTIME_LEGEND}")

with tabs[4]:
    st.header("Glossary")
    st.markdown("""
**Horizon**: steps ahead to predict · **Lookback**: past steps fed to the model · **Seasonal period**: 7/52/12  
**CV folds**: time-ordered train/validation splits · **Exogenous features**: external regressors  
**Stock vs Flow**: point-in-time level vs amount per interval · **Conformal PIs**: calibrated prediction intervals  
**MAE / RMSE / MAPE**: error metrics; prefer MAE/SMAPE when values near zero occur.
""")
