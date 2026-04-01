"""
Overfitting detection for forecast pipeline outputs.

Takes predictions from any pipeline family and checks for:
- Train vs test error ratio (per fold)
- Error variance across folds (stability)
- Performance degradation on later folds (distribution shift)

Returns a structured report with risk level and details.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def check_overfitting(
    predictions_df: pd.DataFrame,
    train_errors: Optional[List[Dict]] = None,
    model_col: str = "model",
    split_col: str = "split_id",
) -> Dict:
    """Analyze predictions for overfitting signals.

    Parameters
    ----------
    predictions_df : DataFrame
        Must contain columns: y_true, y_pred, and a split/fold identifier.
    train_errors : list of dicts, optional
        Each dict should have keys: {"fold", "model", "train_mae", "test_mae"}.
        If provided, used for train/test ratio analysis.  If not provided,
        the function derives per-fold test errors and flags instability.
    model_col : str
        Column identifying the model name.
    split_col : str
        Column identifying the fold/split.

    Returns
    -------
    dict with keys:
        overfitting_risk : "low" | "medium" | "high"
        details : list of human-readable finding strings
        fold_errors : list of per-fold error dicts
        train_test_ratios : list of ratios (if train_errors provided)
        error_cv : coefficient of variation of test MAE across folds
        trend_slope : normalized slope of MAE across folds (positive = degrading)
    """
    result = {
        "overfitting_risk": "low",
        "details": [],
        "fold_errors": [],
        "train_test_ratios": [],
        "error_cv": np.nan,
        "trend_slope": np.nan,
    }

    df = predictions_df.copy()
    if df.empty or "y_true" not in df.columns or "y_pred" not in df.columns:
        result["details"].append("No valid predictions to analyze.")
        return result

    # Use first model if multiple present
    if model_col in df.columns:
        models = df[model_col].unique()
        # Skip baseline rows
        models = [m for m in models if "baseline" not in str(m).lower()]
        if models:
            df = df[df[model_col] == models[0]]

    # ── Per-fold test errors ──
    fold_maes = []
    if split_col in df.columns:
        for fold_id, group in df.groupby(split_col, sort=False):
            valid = group.dropna(subset=["y_true", "y_pred"])
            if len(valid) < 2:
                continue
            mae = float(np.mean(np.abs(valid["y_true"].values - valid["y_pred"].values)))
            fold_maes.append({"fold": fold_id, "test_mae": mae, "n_points": len(valid)})
    else:
        # Single fold
        valid = df.dropna(subset=["y_true", "y_pred"])
        if len(valid) >= 2:
            mae = float(np.mean(np.abs(valid["y_true"].values - valid["y_pred"].values)))
            fold_maes.append({"fold": "all", "test_mae": mae, "n_points": len(valid)})

    result["fold_errors"] = fold_maes

    if len(fold_maes) == 0:
        result["details"].append("No folds with enough data for analysis.")
        return result

    test_mae_values = np.array([f["test_mae"] for f in fold_maes])

    # ── Check 1: Train vs test error ratio ──
    risk_signals = []
    if train_errors:
        ratios = []
        for te in train_errors:
            train_mae = te.get("train_mae", np.nan)
            test_mae = te.get("test_mae", np.nan)
            if train_mae > 0 and np.isfinite(test_mae):
                ratio = test_mae / train_mae
                ratios.append(ratio)
                if ratio > 2.0:
                    risk_signals.append("high")
                    result["details"].append(
                        f"Fold '{te.get('fold', '?')}': test/train MAE ratio = {ratio:.2f} "
                        f"(train={train_mae:.2f}, test={test_mae:.2f}) — likely overfitting"
                    )
                elif ratio < 0.5:
                    risk_signals.append("medium")
                    result["details"].append(
                        f"Fold '{te.get('fold', '?')}': train MAE > 2× test MAE "
                        f"(ratio={ratio:.2f}) — unusual, check for data issues"
                    )
        result["train_test_ratios"] = ratios

    # ── Check 2: Error variance across folds ──
    if len(test_mae_values) >= 2:
        mean_mae = float(np.mean(test_mae_values))
        std_mae = float(np.std(test_mae_values, ddof=1))
        cv = std_mae / mean_mae if mean_mae > 0 else np.nan
        result["error_cv"] = float(cv) if np.isfinite(cv) else np.nan

        if np.isfinite(cv):
            if cv > 0.5:
                risk_signals.append("high")
                result["details"].append(
                    f"High error variance across folds: CV = {cv:.2f} "
                    f"(MAEs: {', '.join(f'{m:.1f}' for m in test_mae_values)}) — unstable model"
                )
            elif cv > 0.3:
                risk_signals.append("medium")
                result["details"].append(
                    f"Moderate error variance across folds: CV = {cv:.2f}"
                )

    # ── Check 3: Performance degradation on later folds ──
    if len(test_mae_values) >= 3:
        # Fit a simple linear trend: MAE vs fold index
        x = np.arange(len(test_mae_values), dtype=float)
        # Normalize x and y for a meaningful slope
        x_norm = x / max(x.max(), 1)
        y_norm = test_mae_values / max(np.mean(test_mae_values), 1e-9)
        slope = float(np.polyfit(x_norm, y_norm, 1)[0])
        result["trend_slope"] = slope

        if slope > 0.5:
            risk_signals.append("high")
            result["details"].append(
                f"MAE increases across folds (slope={slope:.2f}): "
                f"later folds are harder — possible distribution shift"
            )
        elif slope > 0.25:
            risk_signals.append("medium")
            result["details"].append(
                f"Slight MAE increase across folds (slope={slope:.2f})"
            )

    # ── Aggregate risk ──
    if "high" in risk_signals:
        result["overfitting_risk"] = "high"
    elif "medium" in risk_signals:
        result["overfitting_risk"] = "medium"
    else:
        result["overfitting_risk"] = "low"
        if not result["details"]:
            result["details"].append("No overfitting signals detected.")

    return result
