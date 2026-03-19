# frontend/recommender.py — Smart model recommendation engine
"""
Analyzes completed run results and provides actionable recommendations
for improving accuracy, choosing models, and tuning parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_float(v, default=np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def recommend_model(
    pred: pd.DataFrame,
    metr: Optional[pd.DataFrame],
    lb: Optional[pd.DataFrame],
    config: dict,
    target: str = "",
    horizon: int = 1,
) -> Dict:
    """Analyze run outputs and return structured recommendations.

    Returns dict with keys:
        best_model     : str
        accuracy_grade : str (A/B/C/D/F)
        grade_color    : str (success/warning/error)
        scorecard      : dict of metric name -> value
        tips           : list[str]
        next_steps     : list[str]
        risk_flags     : list[str]
    """
    result = {
        "best_model": "",
        "accuracy_grade": "N/A",
        "grade_color": "info",
        "scorecard": {},
        "tips": [],
        "next_steps": [],
        "risk_flags": [],
    }

    if pred is None or pred.empty:
        result["tips"].append("No predictions found. Run an experiment first.")
        return result

    # Filter to target/horizon if available
    df = pred.copy()
    if "target" in df.columns and target:
        df = df[df["target"] == target]
    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]

    if df.empty:
        result["tips"].append(f"No predictions for target='{target}', horizon={horizon}.")
        return result

    # ---- Best model identification ----
    if lb is not None and not lb.empty:
        lb_f = lb.copy()
        if "target" in lb_f.columns and target:
            lb_f = lb_f[lb_f["target"] == target]
        if "horizon" in lb_f.columns:
            lb_f = lb_f[lb_f["horizon"] == horizon]
        # Skip baseline rows
        lb_f = lb_f[~lb_f["model"].str.contains("baseline|Persistence", case=False, na=False)]
        if not lb_f.empty and "MAE" in lb_f.columns:
            best_row = lb_f.loc[lb_f["MAE"].idxmin()]
            result["best_model"] = str(best_row["model"])

    if not result["best_model"] and "model" in df.columns:
        model_mae = df.groupby("model").apply(
            lambda g: (g["y_true"] - g["y_pred"]).abs().mean()
        )
        result["best_model"] = model_mae.idxmin()

    # ---- Scorecard ----
    best = result["best_model"]
    if best and "model" in df.columns:
        bdf = df[df["model"] == best].dropna(subset=["y_true", "y_pred"])
    else:
        bdf = df.dropna(subset=["y_true", "y_pred"])

    if bdf.empty:
        result["tips"].append("No valid predictions with both y_true and y_pred.")
        return result

    y_true = bdf["y_true"].values
    y_pred = bdf["y_pred"].values
    err = y_true - y_pred
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_abs_true = float(np.mean(np.abs(y_true)))
    mape = float(np.mean(abs_err / np.maximum(np.abs(y_true), 1e-9))) if mean_abs_true > 0 else np.nan
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    smape = float(np.mean(2 * abs_err / denom))

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan

    # Monthly accuracy (within 10%)
    idx = pd.to_datetime(bdf["date"])
    s_true = pd.Series(y_true, index=idx)
    s_pred = pd.Series(y_pred, index=idx)
    m_true = s_true.resample("ME").sum()
    m_pred = s_pred.resample("ME").sum()
    monthly_tol10 = float(np.mean(
        (np.abs(m_true - m_pred) / np.maximum(np.abs(m_true), 1e-9)) <= 0.10
    )) if len(m_true) > 0 else np.nan

    # PI coverage
    pi_coverage = np.nan
    pi_width = np.nan
    if {"y_lo", "y_hi"}.issubset(bdf.columns):
        pidf = bdf.dropna(subset=["y_lo", "y_hi"])
        if not pidf.empty:
            covered = (pidf["y_true"] >= pidf["y_lo"]) & (pidf["y_true"] <= pidf["y_hi"])
            pi_coverage = float(covered.mean())
            pi_width = float((pidf["y_hi"] - pidf["y_lo"]).mean())

    result["scorecard"] = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "Monthly Accuracy (10% tol)": monthly_tol10,
        "PI Coverage": pi_coverage,
        "PI Avg Width": pi_width,
        "N predictions": len(bdf),
    }

    # ---- Accuracy grade ----
    # Grade based on sMAPE and R2
    if not np.isnan(smape) and not np.isnan(r2):
        if smape < 0.05 and r2 > 0.95:
            grade, color = "A", "success"
        elif smape < 0.10 and r2 > 0.85:
            grade, color = "B", "success"
        elif smape < 0.20 and r2 > 0.70:
            grade, color = "C", "warning"
        elif smape < 0.35 and r2 > 0.50:
            grade, color = "D", "warning"
        else:
            grade, color = "F", "error"
    elif not np.isnan(r2):
        if r2 > 0.90:
            grade, color = "A", "success"
        elif r2 > 0.75:
            grade, color = "B", "success"
        elif r2 > 0.50:
            grade, color = "C", "warning"
        else:
            grade, color = "D", "warning"
    else:
        grade, color = "N/A", "info"

    result["accuracy_grade"] = grade
    result["grade_color"] = color

    # ---- Tips and recommendations ----
    tips = result["tips"]
    next_steps = result["next_steps"]
    risk_flags = result["risk_flags"]

    # Model-specific tips
    family = config.get("family", config.get("TG_FAMILY", ""))
    folds = config.get("folds", 1)
    profile_was_demo = folds and int(folds) <= 1

    if profile_was_demo:
        next_steps.append(
            "This was a Demo run (1 fold). Re-run with **Balanced** (3 folds) or "
            "**Thorough** (5+ folds) for production-quality results."
        )

    if grade in ("D", "F"):
        tips.append(
            "Accuracy is low. Consider: (1) using more training data, "
            "(2) trying a different model family, (3) checking for data quality issues."
        )
        if "B_ML" in family or "ML" in str(best).lower():
            next_steps.append("Try adding more lag features or switching to tree-based models (XGBoost/LightGBM).")
        if "A_STAT" in family:
            next_steps.append("Statistical models may be limited. Try ML (family B) or DL (family C) for nonlinear patterns.")

    if grade == "C":
        tips.append("Decent accuracy but room for improvement. Consider ensemble methods or more folds.")

    # R2 check
    if not np.isnan(r2) and r2 < 0:
        risk_flags.append(
            f"R2 is negative ({r2:.3f}), meaning the model is worse than predicting the mean. "
            "This suggests a fundamental issue with the model or data."
        )

    # PI check
    if not np.isnan(pi_coverage):
        if pi_coverage < 0.80:
            risk_flags.append(
                f"PI coverage ({pi_coverage:.1%}) is well below the 90% target. "
                "Intervals are too narrow; consider widening conformal calibration."
            )
        elif pi_coverage > 0.98:
            tips.append(
                f"PI coverage ({pi_coverage:.1%}) is very high. "
                "Intervals may be too wide (conservative). Tighter intervals would be more useful."
            )

    # Monthly accuracy
    if not np.isnan(monthly_tol10):
        if monthly_tol10 >= 0.80:
            tips.append(f"Monthly forecasts are within 10% of actuals {monthly_tol10:.0%} of the time. Good for planning.")
        elif monthly_tol10 < 0.50:
            risk_flags.append(
                f"Monthly accuracy is only {monthly_tol10:.0%} (within 10%). "
                "Consider: more data, different model, or checking for structural breaks."
            )

    # Horizon guidance
    if horizon >= 10:
        tips.append(
            f"Long horizon (h={horizon}). Consider ensemble of short-horizon models "
            "or iterative forecasting for better accuracy at long horizons."
        )

    # Next steps
    if not next_steps:
        if grade in ("A", "B"):
            next_steps.append("Results look good! Consider running an ensemble of your best models for even better accuracy.")
            next_steps.append("Try the Compare page to see how this run stacks up against others.")
        else:
            next_steps.append("Try running multiple models and compare in the Dashboard.")

    return result


def format_scorecard_markdown(scorecard: dict) -> str:
    """Format scorecard dict as a clean markdown table."""
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in scorecard.items():
        if isinstance(v, float):
            if np.isnan(v):
                lines.append(f"| {k} | N/A |")
            elif k in ("MAPE", "sMAPE", "Monthly Accuracy (10% tol)", "PI Coverage"):
                lines.append(f"| {k} | {v:.1%} |")
            elif k in ("MAE", "RMSE", "PI Avg Width"):
                lines.append(f"| {k} | {v:,.0f} |")
            elif k == "R2":
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
