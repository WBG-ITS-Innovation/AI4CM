# pages/09_Documentation.py — the reference: every model, every parameter, every promoted
# recipe, and how to add a model.
#
# Renamed from 06_Models.py. This is the one page whose URL moved, because Streamlit derives
# a page URL from its name and not from its number prefix: /Models became /Documentation.
# Every in-app link is a page_link to the file, so none of them carried the old URL.
import json
from typing import Dict

import pandas as pd
import streamlit as st

try:
    from ui_styles import inject_global_css, page_header
except ImportError:
    def inject_global_css(): pass
    def page_header(t, s=""): return f"<h1>{t}</h1><p>{s}</p>"

from ui_styles import help_text  # tooltips, translated at render time
from ui_styles import inject_design_system  # presentation only
from ui_styles import glossary_note  # plain-language definitions, on demand
from i18n import install as install_language  # language toggle + pending-review note
from i18n import t as _t  # the shelf's status labels are fixed copy
from ui_styles import page_intro  # the one-or-two-sentence intro every page opens with
from ui_styles import render_app_header  # presentation only
from ui_styles import render_brand  # the one brand header, in the sidebar
st.set_page_config(page_title="Documentation · Treasury Forecast", layout="wide")
inject_global_css()
inject_design_system()

# The language toggle and, in Georgian, the standing note that the translation has
# not been reviewed by a native speaker. One call per page; everything else the
# reader sees is translated inside the shared helpers.
render_brand()
install_language()
render_app_header("Documentation",
                  "Every model, its settings, the promoted recipes and their evidence")
page_intro(
    "This page is the shelf: every model available here, what it does in plain language, "
    "and whether anybody has recorded a measured result for it."
)
glossary_note("MASE", "champion", "withheld", "gate", "baseline")

# ──────────────────────────────────────────────────────────────────────
# PROMOTED RECIPES (live, from registry/recipes.json)
#
# The rest of this page is a static reference for every model family available. This
# section is different: it is what has actually been promoted per target, with the
# evidence, read live from the registry. If it disagrees with the reference below, the
# registry wins -- it is the thing tied to logged runs.
# ──────────────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "backend"))


from format_gel import gel_millions as _gel_m  # noqa: E402
from format_gel import (NOT_REPORTED, UNIT_LABEL, gel_millions, number, pct,  # noqa: E402
                        pct_points, ratio)
from ui_styles import (empty_state, gate_badge_tri, section_header, HELP)  # noqa: E402


def _render_registry() -> None:
    try:
        from registry import load_registry, verify_against_log
    except Exception as exc:  # pragma: no cover - import guard for a demo machine
        st.info(f"Registry unavailable ({exc}).")
        return
    try:
        reg = load_registry()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    st.subheader("Promoted recipes, one per target")
    st.caption(
        "Champions selected on training folds and confirmed on 2024. "
        "**Nothing here is approved**: no approval workflow exists yet, and neither "
        "hyperparameter tuning nor target scaling has been run."
    )

    rows = []
    for r in reg["recipes"]:
        cred = r["dev_credentials"]
        pub = r["publication"]
        rows.append({
            "Target": r["target"],
            "Model": r["point_model"],
            "Intervals": r.get("interval_model", "not recorded"),
            "Typical error 2024 (M GEL)": _gel_m(cred["dev_mae"]),
            "vs benchmark": f"{cred['skill_vs_ruler_pct']:.1f}% better",
            "Verdict": ("forecast" if pub["verdict"] == "publishable"
                        else "withheld as forecast"),
            "Status": r["status"],
            "Approved by": r["approved_by"] or "nobody",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    chk = verify_against_log()
    if chk["ok"]:
        st.success(
            f"All {chk['metrics_checked']} quoted figures reconcile against "
            f"`experiments/log.csv`. Every number here is traceable to a logged run."
        )
    else:
        st.error("Registry does not reconcile with the experiments log:\n" +
                 "\n".join(f"- {p}" for p in chk["problems"]))

    for r in reg["recipes"]:
        cred, pub = r["dev_credentials"], r["publication"]
        verdict = ("usable as a forecast" if pub["verdict"] == "publishable"
                   else "WITHHELD as a forecast")
        with st.expander(f"{r['target']}: {r['point_model']} · {verdict}"):
            st.markdown(f"**Recipe id** `{r['id']}`  \n"
                        f"**Family** {r['family']} · **Intervals** "
                        f"{r.get('interval_model', 'not recorded')}  \n"
                        f"**Target scaling** {r['scaling']}  \n"
                        f"**Fiscal calendar version** `{r['calendar_version']}`")
            st.markdown("**Feature groups**: " + ", ".join(r["feature_groups"]) +
                        (("  \n**Exogenous blocks**: " + ", ".join(r["exog_blocks"]))
                         if r.get("exog_blocks") else ""))
            st.markdown(f"**Why this recipe.** {r['provenance_note']}")

            st.markdown("**Evidence (2024 confirmation)**")
            e1, e2, e3 = st.columns(3)
            e1.metric("Typical error", f"{_gel_m(cred['dev_mae'])} M GEL")
            e2.metric("vs simple benchmark", f"{cred['skill_vs_ruler_pct']:.1f}%")
            e3.metric("Scaled error (1.0 = benchmark)", f"{cred['mase']:.2f}")
            st.caption(f"Logged run `{cred['run_id']}` · window {cred['window']} · "
                       f"n={cred['n']}")

            st.markdown("**Checks**")
            for key, g in cred["gates"].items():
                verdict = _t("passed") if g.get("passed") else _t("failed")
                st.markdown(f"- **{g.get('name', key)}**, {verdict}. "
                            f"{g.get('reason_plain', '')}")
                if g.get("corroboration"):
                    st.caption(f"  {g['corroboration']}")

            if pub["verdict"] != "publishable":
                st.error(f"**Withheld as a forecast.** {pub['reason_plain']}")
                if pub.get("named_fix"):
                    st.warning(f"**Named fix:** {pub['named_fix']}")

            nb = cred.get("not_the_dev_best")
            if nb:
                st.info(
                    f"**Not the single best 2024 result.** {nb['better_option']} scored "
                    f"{_gel_m(nb['its_dev_mae'])} M GEL versus {_gel_m(nb['this_dev_mae'])} "
                    f"M GEL here ({nb['gap_pct']:.1f}% apart). {nb['why_promoted_anyway']}"
                )

    st.caption("Pending: " + " · ".join(reg["pending_workstreams"]))
    st.divider()


_render_registry()

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
        dict(Model="NaiveLast", Parameter="none",
             Meaning="Forecast equals the last observed value.",
             WhyItMatters="Establishes a sanity baseline and helps detect random-walk behavior.",
             Suggested="No tuning.", Runtime="very fast"),
        dict(Model="WeekdayMean", Parameter="weeks_back",
             Meaning="Average of same weekday over the last N weeks (daily flows).",
             WhyItMatters="Captures weekly operational patterns without complex modeling.",
             Suggested="4–8", Runtime="very fast"),
        dict(Model="MovingAverage", Parameter="window",
             Meaning="Simple moving mean over the last W periods.",
             WhyItMatters="Smooths noise; reduces sensitivity to day-to-day spikes.",
             Suggested="Daily: 7/14/28; Weekly: 4/8/12; Monthly: 3/6/12", Runtime="very fast"),
        dict(Model="ETS", Parameter="trend, seasonal, seasonal_periods, damped_trend",
             Meaning="Exponential smoothing with level/trend/seasonality components.",
             WhyItMatters="Often strong baseline with interpretable components.",
             Suggested="trend='add', seasonal='add', sp=7|52|12, damped=True", Runtime="medium"),
        dict(Model="SARIMAX", Parameter="order=(p,d,q), seasonal_order=(P,D,Q,s)",
             Meaning="ARIMA-family with seasonal terms; can support exogenous inputs.",
             WhyItMatters="Captures autocorrelation structure; flexible but can be slower to tune.",
             Suggested="Keep (p,q,P,Q) ≤ 2; set s=7|52|12", Runtime="medium to slower"),
        dict(Model="STL-ARIMA", Parameter="stl_season_length, arima_order, robust",
             Meaning="Decompose seasonality/trend via STL, then ARIMA on remainder.",
             WhyItMatters="Useful when seasonality is stable and outliers exist (robust STL).",
             Suggested="stl=7|12; arima=(0,1,1) or (1,0,1); robust=True for spikes", Runtime="medium"),
        dict(Model="Theta", Parameter="theta, seasonal_periods",
             Meaning="Competition-grade method: trend + smoothing.",
             WhyItMatters="Strong baseline, low tuning burden.",
             Suggested="theta=2; sp=7|52|12", Runtime="very fast"),
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
             Suggested="Daily: 1,2,3,7,14,21,28; Weekly: 1,4,12; Monthly: 1,3,12", Runtime="very fast"),
        dict(Model="Preprocessing", Parameter="rolling_windows",
             Meaning="Trailing window summaries (rolling mean/std, etc.).",
             WhyItMatters="Adds stability and context, especially on noisy flows.",
             Suggested="Daily: 7/14/28; Weekly: 4/8/12; Monthly: 3/6/12", Runtime="very fast"),
        dict(Model="Preprocessing (multi)", Parameter="exog_top_k",
             Meaning="Top-K exogenous columns to include (multivariate).",
             WhyItMatters="Controls dimensionality and reduces overfitting risk.",
             Suggested="Start 5–15, increase cautiously", Runtime="medium"),
        dict(Model="Ridge", Parameter="alpha",
             Meaning="L2 regularization strength.",
             WhyItMatters="Higher alpha reduces overfit on noisy/collinear features.",
             Suggested="0.1 / 1.0 / 10.0", Runtime="very fast"),
        dict(Model="Lasso", Parameter="alpha",
             Meaning="L1 regularization strength (sparsity).",
             WhyItMatters="Can select features automatically; may underfit if too strong.",
             Suggested="0.01 / 0.1 / 1.0", Runtime="very fast"),
        dict(Model="ElasticNet", Parameter="alpha, l1_ratio",
             Meaning="Mix of L1 and L2 regularization.",
             WhyItMatters="Balances shrinkage and feature selection.",
             Suggested="alpha=0.1–1.0, l1_ratio=0.2–0.8", Runtime="very fast"),
        dict(Model="RandomForest", Parameter="n_estimators, max_depth, min_samples_leaf",
             Meaning="Bagged trees for nonlinear effects.",
             WhyItMatters="Robust but slower; can overfit without min_samples_leaf.",
             Suggested="n=400–800; leaf=1–10", Runtime="medium"),
        dict(Model="ExtraTrees", Parameter="n_estimators, max_depth",
             Meaning="More randomized tree ensemble.",
             WhyItMatters="Can perform well on noisy data; similar tuning to RF.",
             Suggested="n=400–800; depth=None", Runtime="medium"),
        dict(Model="HistGBDT", Parameter="learning_rate, max_depth, max_leaf_nodes",
             Meaning="Efficient gradient boosting (sklearn).",
             WhyItMatters="Strong performance with manageable tuning.",
             Suggested="lr=0.05–0.1; depth=3–7", Runtime="medium"),
        dict(Model="XGBoost", Parameter="n_estimators, max_depth, eta, subsample, colsample_bytree, reg_lambda",
             Meaning="Boosted trees (xgboost).",
             WhyItMatters="Often top accuracy; slower; requires regularization to prevent overfit.",
             Suggested="depth=3–6; eta=0.03–0.1; subs/cols=0.7–0.9; λ=1–5", Runtime="medium to slower"),
        dict(Model="LightGBM", Parameter="n_estimators, num_leaves, learning_rate, feature_fraction, bagging_fraction, lambda_l2, min_data_in_leaf",
             Meaning="Boosting with leaf-wise growth (lightgbm).",
             WhyItMatters="Very strong but can overfit if leaves are large and min_data_in_leaf is small.",
             Suggested="leaves=31–127; lr=0.03–0.07; minleaf=20–60", Runtime="medium to slower"),
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
             Suggested="Daily: 60–120; Weekly: 80–120; Monthly: 36–60", Runtime="medium"),
        dict(Model="Global", Parameter="batch_size",
             Meaning="Mini-batch size during training.",
             WhyItMatters="Impacts speed and memory; too large can cause memory errors.",
             Suggested="CPU: 16 to 64; GPU: 32 to 128", Runtime="not recorded"),
        dict(Model="Global", Parameter="max_epochs / early stopping",
             Meaning="Training duration and stopping behavior.",
             WhyItMatters="More epochs can improve accuracy but increases runtime and overfit risk.",
             Suggested="Exploration: 3–15; Final: 30–100", Runtime="slower"),
        dict(Model="LSTM/GRU", Parameter="hidden_size, num_layers, dropout",
             Meaning="Capacity and regularization knobs.",
             WhyItMatters="Higher capacity fits complex patterns but increases overfit risk.",
             Suggested="hidden=64–128; layers=1–2; dropout=0.1–0.3", Runtime="medium to slower"),
        dict(Model="TCN", Parameter="levels, kernel_size, dropout",
             Meaning="Causal convolutions with dilation (receptive field).",
             WhyItMatters="Controls how far back the model can “see” effectively.",
             Suggested="levels=5–7; kernel=3–5; dropout=0.1–0.3", Runtime="medium"),
        dict(Model="Transformer", Parameter="d_model, nhead, num_layers, dim_ff, dropout",
             Meaning="Attention-based sequence model configuration.",
             WhyItMatters="Powerful but can overfit on small datasets; heavier runtime.",
             Suggested="d=64–128; heads=4–8; layers=2; dropout=0.1", Runtime="slower"),
        dict(Model="MLP", Parameter="hidden_dims",
             Meaning="Feed-forward network on flattened windows.",
             WhyItMatters="Fast to train, but may struggle with long seasonal dependencies.",
             Suggested="[128, 64] + dropout 0.2", Runtime="medium"),
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
             Suggested="[0.1, 0.5, 0.9] (P10/P50/P90)", Runtime="medium"),
        dict(Model="GBQuantile", Parameter="n_estimators, learning_rate",
             Meaning="Number of trees and boosting step size.",
             WhyItMatters="Main bias/variance tradeoff; lower lr typically needs more trees.",
             Suggested="n=400–800; lr=0.03–0.07", Runtime="medium to slower"),
        dict(Model="GBQuantile", Parameter="max_depth",
             Meaning="Tree depth / interaction complexity.",
             WhyItMatters="Deeper trees fit noise easily on small datasets.",
             Suggested="3–5 (prefer 3–4 on treasury flows)", Runtime="medium"),
        dict(Model="GBQuantile", Parameter="min_samples_leaf / min_child_weight",
             Meaning="Minimum leaf size regularization.",
             WhyItMatters="Higher values reduce variance/overfit risk.",
             Suggested="5–20", Runtime="medium"),
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

# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL DETAIL VIEW
#
# Assembled by backend/model_reference.py and read as JSON, because the modelling libraries live in
# the backend interpreter and not the one running Streamlit.
#
# Three kinds of content, labelled differently on purpose:
#   * descriptions  -- prose, marked "general description, not a measured claim";
#   * hyperparameters -- read LIVE from available_models() via get_params(), diffed against a fresh
#     instance so only what the pipeline actually SETS is highlighted. Change a value in the
#     pipeline and this page moves with it; nothing is transcribed;
#   * measured performance -- experiments/log.csv only, every figure carrying its run_id.
#
# Coverage reads "not reported" for point models because those runs never wrote it. Measured, not
# assumed: 0 of the point-model rows in the log carry a coverage figure.
# ══════════════════════════════════════════════════════════════════════════════

def _render_model_detail() -> None:
    import subprocess as _sp

    st.markdown(section_header("Model reference",
                               "What each model is, what it is configured with, and what it "
                               "actually measured"),
                unsafe_allow_html=True)

    _repo = _Path(__file__).resolve().parents[2]

    @st.cache_data(show_spinner="Reading the model pool from the backend…", ttl=300)
    def _reference() -> dict:
        for _py in (_repo / "backend" / ".venv" / "bin" / "python",
                    _repo / "backend" / ".venv" / "Scripts" / "python.exe"):
            if not _py.exists():
                continue
            try:
                out = _sp.run([str(_py), "backend/model_reference.py"], cwd=str(_repo),
                              capture_output=True, text=True, timeout=180)
            except Exception:
                continue
            if out.returncode == 0:
                line = next((l for l in reversed(out.stdout.splitlines())
                             if l.strip().startswith("{")), "")
                if line:
                    return json.loads(line)
        return {}

    ref = _reference()
    if not ref:
        st.markdown(
            empty_state(
                "The model reference could not be assembled.",
                filename="backend/model_reference.py",
                looked_in=str(_repo / "backend" / ".venv"),
                command="./backend/.venv/bin/python backend/model_reference.py"),
            unsafe_allow_html=True)
        return

    models = ref["models"]
    perf = ref.get("performance", {})
    champs = ref.get("champions", {})

    _names = sorted(models)
    _champ_names = {n for n in _names if n in champs}
    _n_evaluated = sum(1 for m in models.values() if m.get("status") == "evaluated")
    _n_untested = sum(1 for m in models.values() if m.get("status") == "untested")
    st.caption(f"{len(_names)} models on the shelf · {len(_champ_names)} promoted as a champion "
               f"({', '.join(sorted(_champ_names))}) · "
               f"{sum(1 for m in models.values() if not m['available'])} unavailable here")

    # ── The shelf, with what has and has not been measured ────────────────────
    #
    # This table is the reason the status field exists. Before it, an untested model
    # appeared in the list beside a champion with nothing distinguishing them, and the
    # Lab's default selection was Ridge, which has no recorded result on any target.
    st.markdown("**Every model on the shelf, and whether anyone has measured it**")
    st.markdown(
        f"{_n_evaluated} of these have a recorded result that can be quoted and traced back "
        f"to the run that produced it. {_n_untested} are registered candidates with no "
        f"recorded result yet. A candidate can be run as an experiment and cannot become "
        f"the model behind an official forecast until a result has been recorded for it."
    )
    # Translated at lookup, like every other fixed label. These are the words the shelf
    # table shows in its Status column, so leaving them out of the translation layer would
    # have left the one column a reader scans in English while the rest of the page moved.
    _STATUS_LABEL = {
        "evaluated": _t("measured"),
        "untested": _t("UNTESTED"),
        "baseline": _t("reference rule"),
        "unavailable": _t("not installed here"),
    }
    st.dataframe(pd.DataFrame([{
        "Model": n,
        "Family": models[n].get("pipeline", "not recorded"),
        "Status": _STATUS_LABEL.get(models[n].get("status"), models[n].get("status", "not recorded")),
        "Measured on": ", ".join(models[n].get("measured_on") or []) or "nothing yet",
        "Can be a champion": "yes" if models[n].get("gate_eligible") else "no",
    } for n in _names]), hide_index=True, use_container_width=True)

    with st.expander("What does UNTESTED mean?"):
        st.markdown(
            "- It means no measured result has been recorded for that model on any target, "
            "so there is no number to show for it and nothing for the publication checks to "
            "read.\n"
            "- It does **not** mean the model is bad, or that it has never been executed. It "
            "means nobody has yet run it through the evaluation that produces a quotable "
            "figure.\n"
            "- You can run an untested model as an experiment from the Lab or from the "
            "comparison on the Forecast page. Nothing you run there is published.\n"
            "- It cannot become the model behind an official forecast until a result has "
            "been recorded, because an official recipe cites the run that measured it and "
            "that citation is checked.\n"
            "- A **reference rule** is not a candidate at all. Carrying the last value "
            "forward is the yardstick the others are measured against, so asking whether it "
            "was measured is the wrong question.\n"
            "- Adding a model is one entry in `backend/model_catalog.py`. See "
            "`docs/ADDING_A_MODEL.md`."
        )

    _pick = st.selectbox("Model", _names,
                         index=_names.index("LightGBM_L1") if "LightGBM_L1" in _names else 0)
    m = models[_pick]

    # ── availability ──────────────────────────────────────────────────────────
    if not m["available"]:
        st.error(f"**{_pick} is unavailable in this environment.** Its library "
                 f"(`{m.get('missing_library', 'unknown')}`) is not installed, so the model cannot "
                 f"be run or configured. It is listed here rather than hidden, so the pool's "
                 f"contents do not silently change with the environment.")

    # ── status, before anything that looks like a number ──────────────────────
    if m.get("status") == "untested":
        st.warning(f"**UNTESTED.** {m.get('status_note', '')}")
    elif m.get("status") == "baseline":
        st.info(f"**Reference rule.** {m.get('status_note', '')}")
    elif m.get("status") == "evaluated":
        st.success(
            "**Measured.** This model has a recorded result on "
            + ", ".join(m.get("measured_on") or [])
            + ", so the figures below can be traced back to the runs that produced them."
        )

    # ── description: general, not measured ────────────────────────────────────
    st.markdown(f"#### {_pick}")
    st.caption(f"{m.get('family', 'not recorded')} · {m.get('pipeline')} · class "
               f"`{m.get('class', 'not recorded')}`")
    if m.get("summary"):
        st.markdown(m["summary"])
        st.caption(f"_{m['description_kind']}._ It describes how the model works; it says nothing "
                   f"about how well it performed here. For that, see the measured table below.")

    # ── champion record, exactly as stored ────────────────────────────────────
    for rec in champs.get(_pick, []):
        cred = rec["dev_credentials"]
        with st.container():
            st.success(f"**Promoted as champion for {rec['target']}** · recipe `{rec['id']}`")
            st.markdown(
                f"**Status** {rec['status']}  \n"
                f"**Approved by** {rec['approved_by'] or 'nobody, because no approval workflow exists yet'}  \n"
                f"**Target scaling** {rec['scaling']}  \n"
                f"**Feature groups** {', '.join(rec['feature_groups'])}"
                + (f"  \n**Exogenous blocks** {', '.join(rec['exog_blocks'])}"
                   if rec.get("exog_blocks") else "")
                + f"  \n**Fiscal calendar version** `{rec['calendar_version']}`")
            st.caption(f"Evidence: run `{cred['run_id']}` · {cred['window']} · n={cred['n']}")
            for key, g in cred["gates"].items():
                st.markdown(gate_badge_tri(g.get("passed", None),
                                           label=g.get("name", key)),
                            unsafe_allow_html=True)
                if g.get("reason_plain"):
                    st.caption(g["reason_plain"])
            if rec["publication"]["verdict"] != "publishable":
                st.error(f"**Withheld as a forecast.** {rec['publication']['reason_plain']}")

    # ── hyperparameters, read live ────────────────────────────────────────────
    hp = m["hyperparameters"]
    st.markdown("##### Configuration")
    if hp.get("note"):
        st.info(hp["note"])
    _set = hp.get("set_by_pipeline", [])
    if _set:
        st.caption(f"{len(_set)} of {hp['n_total']} parameters are set by the pipeline; the rest "
                   f"are library defaults. Read live from the code, so if a value changes in the "
                   f"pipeline, it changes here.")
        st.dataframe(pd.DataFrame([{
            "Parameter": p["name"], "Current value": p["value"],
            "What it controls": p["controls"] or NOT_REPORTED,
            "Sensible range": p["range"] or NOT_REPORTED} for p in _set]),
            hide_index=True, use_container_width=True)
    elif m["available"]:
        st.caption("This model holds no explicitly set parameters. It runs on library defaults, "
                   "or is constructed per fold.")
    if hp.get("library_default"):
        with st.expander(f"Inherited library defaults ({len(hp['library_default'])})"):
            st.dataframe(pd.DataFrame([{
                "Parameter": p["name"], "Value": p["value"],
                "What it controls": p["controls"] or ""} for p in hp["library_default"]]),
                hide_index=True, use_container_width=True)

    # ── measured performance, from the log only ───────────────────────────────
    st.markdown("##### Measured performance")
    rows = perf.get(_pick, [])
    if not rows:
        st.markdown(
            empty_state(
                f"{_pick} has no logged runs, so nothing measured can be shown for it.",
                filename="experiments/log.csv",
                looked_in=str(_repo / "experiments"),
                command="Run an ablation for this model; runs append to the log automatically"),
            unsafe_allow_html=True)
        st.caption("A model with no logged run is not a bad model. It is an unmeasured one. "
                   "Nothing here is inferred from a sibling model.")
    else:
        _tsel = st.multiselect("Target", sorted({r["target"] for r in rows}), default=[])
        _view = [r for r in rows if not _tsel or r["target"] in _tsel]
        st.caption(f"{len(_view)} of {len(rows)} logged runs. Every figure carries the run that "
                   f"produced it; nothing is recomputed on this page.")
        st.dataframe(pd.DataFrame([{
            "Target": r["target"],
            "Window": r["window"],
            f"MAE ({UNIT_LABEL})": gel_millions(r["mae"]),
            "MASE": number(r["mase"]),
            "Skill vs ruler": pct_points(r["skill_vs_ruler_pct"]),
            "Signal": ratio(r["sentinel"]),
            "Coverage low": pct(r["coverage_low"]),
            "Coverage mid": pct(r["coverage_mid"]),
            "Coverage high": pct(r["coverage_high"]),
            "run_id": r["run_id"],
        } for r in _view]), hide_index=True, use_container_width=True,
            column_config={
                "Skill vs ruler": st.column_config.TextColumn(help=help_text("skill")),
                "Signal": st.column_config.TextColumn(help=help_text("sentinel")),
                "MASE": st.column_config.TextColumn(help=help_text("mase")),
                "Coverage high": st.column_config.TextColumn(help=help_text("tercile_coverage")),
                "run_id": st.column_config.TextColumn(
                    help="The logged run this row came from. Its full record, including data and "
                         "code fingerprints, is in experiments/runs/<run_id>.json."),
            })
        if all(r["coverage_high"] is None for r in _view):
            st.info(ref.get("coverage_note", ""))

    st.divider()


_render_model_detail()


# ══════════════════════════════════════════════════════════════════════════════
# ADDING A MODEL
#
# Rendered from docs/ADDING_A_MODEL.md rather than restated here, on the same grounds as the
# Overview page's progress section: one source, so the page and the written procedure cannot
# drift. If the file is absent the section says where it should be instead of vanishing.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(section_header("Adding a model",
                           "The procedure for putting a new model on the shelf"),
            unsafe_allow_html=True)

_ADDING = _Path(__file__).resolve().parents[2] / "docs" / "ADDING_A_MODEL.md"
st.markdown(_t(
    "Every model on this page was registered the same way, and the procedure is written down "
    "rather than passed on by word of mouth. It covers where the entry goes, which fields need "
    "thought, why the import must sit inside the function, and how to check the model can "
    "actually be reached from the Lab."
))
if _ADDING.exists():
    st.caption(f"Rendered from `docs/ADDING_A_MODEL.md`.")
    with st.expander(_t("Read the procedure"), expanded=False):
        st.markdown(_ADDING.read_text(encoding="utf-8"))
else:
    st.info(_t(
        "The procedure is not on this machine. It belongs at `docs/ADDING_A_MODEL.md` in the "
        "repository."
    ))
