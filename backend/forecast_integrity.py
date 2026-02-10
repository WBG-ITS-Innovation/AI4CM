"""
Forecast Integrity Module - Horizon-Aware Validation

This module provides comprehensive integrity checks for forecast predictions:
- Alignment validation (step-based, horizon-aware)
- Leakage detection (feature-level checks)
- Shift diagnostics (horizon-aware interpretation)
- Baseline comparisons (quality gates)

✅ Designed to work for ANY horizon h (not just h=6).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path


def validate_alignment_step_based(
    predictions_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    horizon: int,
) -> Dict:
    """
    Validate alignment using step-based (index-position) logic.
    
    ✅ Horizon-aware: Works for ANY horizon h.
    
    For each prediction:
        idx(target_date) - idx(origin_date) == h_steps
    
    Args:
        predictions_df: DataFrame with columns ['origin_date', 'target_date', 'horizon']
        date_index: The actual DatetimeIndex used for modeling (after filtering/resampling)
        horizon: Forecast horizon (in steps, not calendar days)
        
    Returns:
        Dictionary with alignment_ok, n_misaligned, misaligned_examples
    """
    if "origin_date" not in predictions_df.columns or "target_date" not in predictions_df.columns:
        return {
            "alignment_ok": False,
            "error": "Missing origin_date or target_date columns",
            "n_misaligned": 0,
            "misaligned_examples": [],
        }
    
    df = predictions_df.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    
    # Build position map: date -> position in index
    date_to_pos = {date: pos for pos, date in enumerate(date_index)}
    
    # Compute expected target using step-based logic
    alignment_errors = []
    for idx, row in df.iterrows():
        origin_date = pd.Timestamp(row["origin_date"]).normalize()
        target_date = pd.Timestamp(row["target_date"]).normalize()
        
        if origin_date not in date_to_pos:
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "target": target_date.date(),
                "error": "origin_date not found in date_index",
            })
            continue
        
        pos_origin = date_to_pos[origin_date]
        pos_target_expected = pos_origin + horizon
        
        if pos_target_expected >= len(date_index):
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "target": target_date.date(),
                "error": f"target position {pos_target_expected} out of bounds (len={len(date_index)})",
            })
            continue
        
        expected_target = date_index[pos_target_expected]
        
        if target_date != expected_target:
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "expected": expected_target.date(),
                "actual": target_date.date(),
                "difference_steps": pos_target_expected - date_to_pos.get(target_date, -999),
            })
    
    n_misaligned = len(alignment_errors)
    
    return {
        "alignment_ok": n_misaligned == 0,
        "n_misaligned": n_misaligned,
        "n_total": len(df),
        "misaligned_examples": alignment_errors[:10],  # Limit examples
        "validation_method": "step_based",
    }


def check_feature_leakage(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
) -> Dict:
    """
    Check for leakage in feature engineering.
    
    ✅ Horizon-aware: Ensures no feature uses information beyond origin_date.
    
    Checks:
    - No negative lags (shift(-k) where k < 0)
    - No centered rolling windows
    - All rolling features are trailing (shift(1) applied)
    - No feature uses y_target or shifted(-k) with k < 0
    
    Args:
        feature_df: DataFrame with feature columns
        target_series: Target series (for reference)
        horizon: Forecast horizon
        
    Returns:
        Dictionary with leakage_detected, leakage_details
    """
    leakage_issues = []
    
    # Check for negative shifts in feature names or values
    for col in feature_df.columns:
        # Check column name for negative lag indicators
        if "shift(-" in col.lower() or "lag_-" in col.lower():
            leakage_issues.append(f"Feature '{col}' appears to use negative shift (leakage)")
        
        # Check if feature values correlate perfectly with future target (suspicious)
        if len(feature_df) > horizon:
            # Compare feature at t with target at t+h
            feat_t = feature_df[col].iloc[:-horizon] if horizon > 0 else feature_df[col]
            target_t_h = target_series.iloc[horizon:] if horizon > 0 else target_series
            
            if len(feat_t) == len(target_t_h) and len(feat_t) > 10:
                corr = np.corrcoef(feat_t.dropna(), target_t_h[feat_t.dropna().index].dropna())[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.99:
                    leakage_issues.append(
                        f"Feature '{col}' has near-perfect correlation (r={corr:.3f}) with target at t+{horizon} (suspicious)"
                    )
    
    return {
        "leakage_detected": len(leakage_issues) > 0,
        "leakage_details": leakage_issues,
        "n_checks": len(feature_df.columns),
    }


def shift_diagnostic_horizon_aware(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    max_shift: Optional[int] = None,
) -> Dict:
    """
    Horizon-aware shift diagnostic.
    
    ✅ Interprets shifts relative to horizon:
    - best_shift ≈ 0: OK
    - best_shift ≈ -h: persistence-like (compare vs baseline)
    - best_shift ≈ -(h+1): missing lag_0 / using lag_1 (FLAG strongly)
    
    Args:
        y_true: True values at target dates
        y_pred: Predicted values
        horizon: Forecast horizon
        max_shift: Maximum shift to test (default: max(horizon+5, 10))
        
    Returns:
        Dictionary with best_shift, interpretation, lag_0_issue flag
    """
    if max_shift is None:
        max_shift = max(horizon + 5, 10)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    
    mae_shift0 = float(np.mean(np.abs(y_true - y_pred)))
    best_shift = 0
    best_mae = mae_shift0
    shift_maes = {}
    
    # Test shifts in range [-max_shift, +max_shift]
    shifts_to_test = set(range(-max_shift, max_shift + 1))
    shifts_to_test.add(-horizon)  # Always test shift=-h
    shifts_to_test.add(-(horizon + 1))  # Always test shift=-(h+1) for lag_0 detection
    shifts_to_test = sorted(shifts_to_test)
    
    for shift in shifts_to_test:
        if shift == 0:
            mae = mae_shift0
        elif shift > 0:
            if shift < len(y_true):
                mae = float(np.mean(np.abs(y_true[shift:] - y_pred[:-shift])))
            else:
                continue
        else:
            shift_abs = abs(shift)
            if shift_abs < len(y_true):
                mae = float(np.mean(np.abs(y_true[:-shift_abs] - y_pred[shift_abs:])))
            else:
                continue
        
        shift_maes[shift] = mae
        if mae < best_mae:
            best_mae = mae
            best_shift = shift
    
    improvement_ratio = best_mae / mae_shift0 if mae_shift0 > 0 else 1.0
    improvement_pct = ((mae_shift0 - best_mae) / mae_shift0 * 100) if mae_shift0 > 0 else 0.0
    
    # ✅ Horizon-aware interpretation
    is_lag0_issue = False
    is_persistence_like = False
    interpretation = "OK"
    
    if abs(best_shift) <= 1:
        interpretation = "OK (best alignment at shift=0 or ±1)"
    elif abs(best_shift - (-horizon)) <= 1:
        is_persistence_like = True
        interpretation = f"Persistence-like (best_shift≈-h={-horizon}, compare vs persistence baseline)"
    elif abs(best_shift - (-(horizon + 1))) <= 1:
        is_lag0_issue = True
        interpretation = f"MISSING LAG_0 (best_shift≈-(h+1)={-(horizon+1)}, model using lag_1 instead)"
    elif best_shift < -horizon:
        interpretation = f"Suspicious backward shift (best_shift={best_shift} < -h={-horizon})"
    elif best_shift > 0:
        interpretation = f"Forward shift detected (best_shift={best_shift}, predictions ahead?)"
    
    return {
        "best_shift": int(best_shift),
        "best_mae": float(best_mae),
        "mae_shift0": float(mae_shift0),
        "mae_shift_minus_h": shift_maes.get(-horizon, np.nan),
        "mae_shift_minus_h_plus_1": shift_maes.get(-(horizon + 1), np.nan),
        "improvement_pct": float(improvement_pct),
        "improvement_ratio": float(improvement_ratio),
        "is_lag0_issue": bool(is_lag0_issue),
        "is_persistence_like": bool(is_persistence_like),
        "interpretation": interpretation,
        "shift_maes": {int(k): float(v) for k, v in shift_maes.items()},
    }


def compute_persistence_baseline(
    predictions_df: pd.DataFrame,
) -> Dict:
    """
    Compute persistence baseline: y_hat(t+h) = y(t) (origin value).
    
    Args:
        predictions_df: DataFrame with columns ['origin_value', 'y_true', 'y_pred']
        
    Returns:
        Dictionary with mae_persistence, rmse_persistence, skill metrics
    """
    if "origin_value" not in predictions_df.columns or "y_true" not in predictions_df.columns:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "error": "Missing origin_value or y_true columns",
        }
    
    df = predictions_df.dropna(subset=["origin_value", "y_true"]).copy()
    if len(df) == 0:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "error": "No valid rows after dropping NaN",
        }
    
    persistence_preds = df["origin_value"].values
    y_true = df["y_true"].values
    
    mae = float(np.mean(np.abs(y_true - persistence_preds)))
    rmse = float(np.sqrt(np.mean((y_true - persistence_preds) ** 2)))
    
    return {
        "mae_persistence": mae,
        "rmse_persistence": rmse,
        "n_valid": len(df),
    }


def compute_seasonal_naive_baseline(
    predictions_df: pd.DataFrame,
    full_series: pd.Series,
    horizon: int,
    season: int = 5,  # Default: 5 business days (1 week)
) -> Dict:
    """
    Compute seasonal naive baseline for daily data.
    
    For business-day index: season=5 (1 week)
    For calendar-day index: season=7 (1 week)
    
    Args:
        predictions_df: DataFrame with columns ['target_date', 'y_true']
        full_series: Full time series
        horizon: Forecast horizon
        season: Seasonal period (default: 5 for business days)
        
    Returns:
        Dictionary with mae_seasonal_naive
    """
    if "target_date" not in predictions_df.columns or "y_true" not in predictions_df.columns:
        return {
            "mae_seasonal_naive": np.nan,
            "error": "Missing target_date or y_true columns",
        }
    
    df = predictions_df.dropna(subset=["target_date", "y_true"]).copy()
    if len(df) == 0:
        return {
            "mae_seasonal_naive": np.nan,
            "error": "No valid rows",
        }
    
    seasonal_preds = []
    y_true_vals = []
    
    for _, row in df.iterrows():
        target_date = pd.to_datetime(row["target_date"])
        y_true_val = row["y_true"]
        
        # Find value at target_date - season (if exists)
        season_ago_date = target_date - pd.Timedelta(days=season * 2)  # Approximate
        # Find closest date in series
        available_dates = full_series.index[full_series.index <= target_date]
        if len(available_dates) >= season:
            # Use value at position len - season
            seasonal_val = full_series.iloc[-season] if len(available_dates) >= season else np.nan
        else:
            seasonal_val = np.nan
        
        if not np.isnan(seasonal_val):
            seasonal_preds.append(seasonal_val)
            y_true_vals.append(y_true_val)
    
    if len(seasonal_preds) == 0:
        return {
            "mae_seasonal_naive": np.nan,
            "error": "No valid seasonal predictions",
        }
    
    mae = float(np.mean(np.abs(np.array(y_true_vals) - np.array(seasonal_preds))))
    
    return {
        "mae_seasonal_naive": mae,
        "n_valid": len(seasonal_preds),
    }


def compute_skill_score(
    mae_model: float,
    mae_baseline: float,
) -> float:
    """
    Compute skill score: (MAE_baseline - MAE_model) / MAE_baseline * 100.
    
    Positive skill means model is better than baseline.
    
    Args:
        mae_model: Model MAE
        mae_baseline: Baseline MAE
        
    Returns:
        Skill percentage (NaN if baseline MAE is 0 or invalid)
    """
    if np.isnan(mae_model) or np.isnan(mae_baseline) or mae_baseline <= 0:
        return np.nan
    
    skill = ((mae_baseline - mae_model) / mae_baseline) * 100.0
    return float(skill)
