"""Forecast integrity checks to detect horizon misalignment, leakage, and plotting issues."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.linear_model import Ridge


def target_date_from_origin(origin_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """
    Helper to ensure predictions are timestamped at target_date = origin + horizon.
    
    Args:
        origin_date: Date when forecast is made
        horizon: Forecast horizon (days)
        
    Returns:
        Target date (origin + horizon)
    """
    return origin_date + pd.Timedelta(days=horizon)


def shift_sanity_check(
    y_true: np.ndarray, y_pred: np.ndarray, horizon: int, max_shift: int = None
) -> Dict:
    """
    Test A: Shift sanity check (detects lag/offset).
    
    ✅ FIXED: Uses array-step shifts (not calendar-day shifts).
    For business-day indexed data, this correctly tests positional offsets.
    
    Compute error when comparing y_true(t+h) vs y_pred(t+h) and also
    if you artificially shift predictions by ±k steps in the array.
    
    - shift=0: y_pred[t] vs y_true[t] (correct alignment)
    - shift=-k: y_pred[t] vs y_true[t-k] (detects if predictions are actually past values)
    - shift=+k: y_pred[t] vs y_true[t+k] (detects if predictions are too far ahead)
    
    Args:
        y_true: True values at target dates (array, ordered by target_date)
        y_pred: Predicted values (array, same order)
        horizon: Forecast horizon (in steps, not calendar days)
        max_shift: Maximum shift to test in steps (default: max(horizon, 10))
        
    Returns:
        Dictionary with best_shift, best_mae, mae_shift0, lag_warning
    """
    if max_shift is None:
        # ✅ FIX 4: Horizon-aware shift detection window
        # Test shifts around horizon to detect lag_0 issues (best_shift ≈ -(h+1))
        max_shift = max(horizon + 5, 10)  # Test at least [-(h+5)..+(h+5)] steps
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    
    # Calculate MAE for shift=0 (correct alignment: y_pred[t] vs y_true[t])
    mae_shift0 = float(np.mean(np.abs(y_true - y_pred)))
    
    # ✅ FIX 2: Use array-step shifts (not calendar-day shifts)
    # This correctly handles business-day indexed data
    best_shift = 0
    best_mae = mae_shift0
    shift_maes = {}
    
    # Ensure we test shift=-h specifically
    shifts_to_test = set(range(-max_shift, max_shift + 1))
    shifts_to_test.add(-horizon)  # Always test shift=-h
    shifts_to_test = sorted(shifts_to_test)
    
    for shift in shifts_to_test:
        if shift == 0:
            mae = mae_shift0
        elif shift > 0:
            # Shift y_true forward by k steps: compare y_pred[t] with y_true[t+k]
            # This tests if predictions are actually for dates k steps in the past
            # y_pred[0:-k] vs y_true[k:]
            if shift < len(y_true):
                mae = float(np.mean(np.abs(y_true[shift:] - y_pred[:-shift])))
            else:
                continue
        else:
            # Shift y_true backward by k steps: compare y_pred[t] with y_true[t-k]
            # This matches user's test: y_pred[t] vs y_true[t-h] when shift=-h
            # y_pred[k:] vs y_true[:-k]
            shift_abs = abs(shift)
            if shift_abs < len(y_true):
                mae = float(np.mean(np.abs(y_true[:-shift_abs] - y_pred[shift_abs:])))
            else:
                continue
        
        shift_maes[shift] = mae
        if mae < best_mae:
            best_mae = mae
            best_shift = shift
    
    # ✅ Enhanced warning: if best_shift != 0 and mae(best_shift) < 0.85 * mae(0)
    # This indicates significant improvement from shifting (possible misalignment or persistence-like behavior)
    improvement_ratio = best_mae / mae_shift0 if mae_shift0 > 0 else 1.0
    improvement_pct = ((mae_shift0 - best_mae) / mae_shift0 * 100) if mae_shift0 > 0 else 0.0
    
    # ✅ FIX 4: Detect lag_0 issues (best_shift ≈ -(h+1) pattern)
    # This indicates model is effectively using lag_1 instead of lag_0
    is_lag0_issue = (best_shift <= -(horizon - 1)) and (best_shift >= -(horizon + 3))
    
    # ✅ FIX B: Only flag as "critical timestamping bug" if improvement is near-perfect (>95% AND mae_best < 0.2*mae0)
    # Otherwise, it's likely just autocorrelation/persistence (common in financial series)
    is_critical_timestamping_bug = (
        best_shift != 0 and 
        improvement_pct > 95.0 and 
        best_mae < 0.2 * mae_shift0
    )
    
    # Lag warning if best shift is not 0 and improvement is > 10% OR if improvement ratio < 0.85
    # But NOT if it's a critical timestamping bug (that gets handled separately)
    lag_warning = (best_shift != 0) and ((improvement_pct > 10.0) or (improvement_ratio < 0.85)) and not is_critical_timestamping_bug
    
    return {
        "best_shift": int(best_shift),
        "best_mae": float(best_mae),
        "mae_shift0": float(mae_shift0),
        "mae_shift_minus_h": shift_maes.get(-horizon, np.nan),  # Specific check for shift=-h
        "mae_shift_minus_h_plus_1": shift_maes.get(-(horizon + 1), np.nan),  # ✅ FIX 4: Check for lag_0 issue pattern
        "improvement_pct": float(improvement_pct),
        "improvement_ratio": float(improvement_ratio),
        "lag_warning": bool(lag_warning),
        "is_critical_timestamping_bug": bool(is_critical_timestamping_bug),  # ✅ FIX B: Flag for critical bugs
        "is_lag0_issue": bool(is_lag0_issue),  # ✅ FIX 4: Flag for lag_0 missing pattern
        "shift_maes": {int(k): float(v) for k, v in shift_maes.items()},
    }


def compute_baselines(
    series: pd.Series, target_dates: pd.DatetimeIndex, horizon: int
) -> Dict:
    """
    Test B: Baseline check (detects "previous value as prediction").
    
    Computes persistence and seasonal naive baselines.
    
    Args:
        series: Full time series (indexed by date)
        target_dates: Dates where we want predictions
        horizon: Forecast horizon
        
    Returns:
        Dictionary with persistence and seasonal naive predictions and MAEs
    """
    persistence_preds = []
    seasonal_naive_preds = []
    
    for target_date in target_dates:
        origin_date = target_date - pd.Timedelta(days=horizon)
        
        # Persistence: y_hat(t+h) = y(t)
        if origin_date in series.index:
            persistence_preds.append(float(series.loc[origin_date]))
        else:
            persistence_preds.append(np.nan)
        
        # Seasonal naive: y_hat(t+h) = y(t+h-7) (weekly seasonality)
        seasonal_date = target_date - pd.Timedelta(days=7)
        if seasonal_date in series.index:
            seasonal_naive_preds.append(float(series.loc[seasonal_date]))
        else:
            seasonal_naive_preds.append(np.nan)
    
    return {
        "persistence": persistence_preds,
        "seasonal_naive": seasonal_naive_preds,
    }


def compute_baseline_maes(
    y_true: np.ndarray, persistence_preds: List[float], seasonal_naive_preds: List[float]
) -> Dict:
    """Compute MAEs for baselines."""
    y_true = np.asarray(y_true)
    
    # Persistence MAE
    persistence_valid = ~np.isnan(persistence_preds)
    if persistence_valid.any():
        mae_persistence = float(
            np.mean(np.abs(y_true[persistence_valid] - np.array(persistence_preds)[persistence_valid]))
        )
    else:
        mae_persistence = np.nan
    
    # Seasonal naive MAE
    seasonal_valid = ~np.isnan(seasonal_naive_preds)
    if seasonal_valid.any():
        mae_seasonal_naive = float(
            np.mean(
                np.abs(y_true[seasonal_valid] - np.array(seasonal_naive_preds)[seasonal_valid])
            )
        )
    else:
        mae_seasonal_naive = np.nan
    
    return {
        "mae_persistence": mae_persistence,
        "mae_seasonal_naive": mae_seasonal_naive,
    }


def leakage_sentinel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> Dict:
    """
    Test C: Leakage sentinel (detects unreal predictions / future peek).
    
    Shuffle y_train randomly, retrain a lightweight model (Ridge), evaluate.
    Performance should collapse if there's no leakage.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        horizon: Forecast horizon
        
    Returns:
        Dictionary with shuffled_target_mae and leakage_warning
    """
    if len(X_train) < 10 or len(X_test) < 5:
        return {
            "mae_shuffled_target": np.nan,
            "leakage_warning": False,
            "note": "Insufficient data for leakage test",
        }
    
    # Train normal model
    model_normal = Ridge(random_state=42, alpha=1.0)
    model_normal.fit(X_train, y_train)
    y_pred_normal = model_normal.predict(X_test)
    mae_normal = float(np.mean(np.abs(y_test - y_pred_normal)))
    
    # Shuffle targets and retrain
    y_train_shuffled = y_train.copy()
    y_train_shuffled = y_train_shuffled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    y_train_shuffled.index = y_train.index  # Restore original index
    
    model_shuffled = Ridge(random_state=42, alpha=1.0)
    model_shuffled.fit(X_train, y_train_shuffled)
    y_pred_shuffled = model_shuffled.predict(X_test)
    mae_shuffled = float(np.mean(np.abs(y_test - y_pred_shuffled)))
    
    # Warning if shuffled performance is suspiciously close to normal
    # (within 20% of normal performance suggests leakage or very weak signal)
    if mae_normal > 0:
        ratio = mae_shuffled / mae_normal
        leakage_warning = ratio < 1.5  # If shuffled is less than 1.5x worse, suspicious
    else:
        leakage_warning = False
    
    return {
        "mae_normal": float(mae_normal),
        "mae_shuffled_target": float(mae_shuffled),
        "shuffled_to_normal_ratio": float(ratio) if mae_normal > 0 else np.nan,
        "leakage_warning": bool(leakage_warning),
    }


def validate_alignment(
    predictions_df: pd.DataFrame,
    horizon: int,
    cadence: str = "Daily",
    date_index: pd.DatetimeIndex = None,
) -> Dict:
    """
    Validate that origin_date + h steps = target_date for each prediction.
    
    ✅ FIXED: Uses STEP-BASED logic (positional indexing) instead of calendar days.
    For business-day indexed data, horizon is "steps" not calendar days.
    
    Args:
        predictions_df: DataFrame with columns ['origin_date', 'target_date', 'horizon']
        horizon: Forecast horizon (in steps, not calendar days)
        cadence: "Daily" | "Weekly" | "Monthly"
        date_index: The actual DatetimeIndex used in the pipeline (for step-based validation)
                    If None, falls back to calendar-day logic (less accurate for business-day data)
        
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
    
    # ✅ STEP-BASED VALIDATION: Use positional indexing if date_index is provided
    # ✅ FIX A: Also detect business-day frequency and use BDay offsets
    if date_index is not None and len(date_index) > 0:
        # Detect if index is business-day (skips weekends)
        is_business_day = False
        if len(date_index) > 1:
            # Check if index frequency is 'B' (business day) or if weekends are skipped
            try:
                inferred_freq = pd.infer_freq(date_index)
                if inferred_freq == 'B' or inferred_freq == 'C':
                    is_business_day = True
                else:
                    # Check manually: if there are gaps > 1 day, likely business-day
                    diffs = date_index.to_series().diff().dropna()
                    if len(diffs) > 0 and (diffs > pd.Timedelta(days=1)).any():
                        # More than 1 day gaps suggest business-day
                        is_business_day = True
            except:
                pass
        
        # Build position map: date -> position in index
        date_to_pos = {date: pos for pos, date in enumerate(date_index)}
        
        # Compute expected target using step-based logic OR BusinessDay offsets
        expected_targets = []
        missing_origins = []
        
        for idx, row in df.iterrows():
            origin_date = pd.Timestamp(row["origin_date"]).normalize()
            if origin_date in date_to_pos:
                pos_origin = date_to_pos[origin_date]
                pos_target = pos_origin + horizon
                if 0 <= pos_target < len(date_index):
                    # Use positional indexing (most accurate)
                    expected_target = date_index[pos_target]
                    expected_targets.append(expected_target)
                else:
                    expected_targets.append(pd.NaT)
                    missing_origins.append(idx)
            else:
                # Origin not in index - use BusinessDay offset if business-day data
                if is_business_day or cadence.lower() == "daily":
                    # Use BusinessDay offset for business-day indexed data
                    expected_target = (origin_date + BDay(int(horizon))).normalize()
                    expected_targets.append(expected_target)
                else:
                    expected_targets.append(pd.NaT)
                    missing_origins.append(idx)
        
        expected_target = pd.Series(expected_targets, index=df.index)
        
        # Check alignment: target_date should equal expected_target (step-based or BDay)
        misaligned_mask = expected_target.notna() & (df["target_date"].dt.normalize() != expected_target.dt.normalize())
        n_misaligned = int(misaligned_mask.sum())
        
        # Also flag rows where origin_date wasn't found in index
        if len(missing_origins) > 0:
            print(f"[WARN] {len(missing_origins)} predictions have origin_date not found in date_index")
    else:
        # Fallback: Use BusinessDay offsets for daily cadence (treat as business-day)
        if cadence.lower() == "daily":
            # ✅ FIX A: Use BusinessDay offset instead of calendar days
            expected_target = df["origin_date"].apply(lambda x: (pd.Timestamp(x).normalize() + BDay(int(horizon))).normalize())
        elif cadence.lower() == "weekly":
            expected_target = df["origin_date"] + pd.Timedelta(weeks=horizon)
        elif cadence.lower() == "monthly":
            expected_target = df["origin_date"] + pd.DateOffset(months=horizon)
        else:
            expected_target = df["origin_date"] + pd.Timedelta(days=horizon)
        
        # Check alignment (normalize dates for comparison)
        misaligned_mask = (df["target_date"].dt.normalize() != expected_target.dt.normalize())
        n_misaligned = int(misaligned_mask.sum())
    
    misaligned_examples = []
    if n_misaligned > 0:
        # ✅ FIX 6: Defensive assertion for boolean mask length
        assert len(misaligned_mask) == len(df), f"Boolean mask length mismatch: mask={len(misaligned_mask)}, df={len(df)}"
        misaligned_df = df[misaligned_mask].head(5)
        for idx, row in misaligned_df.iterrows():
            if date_index is not None and idx in expected_target.index:
                exp_target = expected_target.loc[idx]
                if pd.notna(exp_target):
                    misaligned_examples.append({
                        "origin_date": str(row["origin_date"].date()),
                        "expected_target": str(exp_target.date()),
                        "actual_target": str(row["target_date"].date()),
                        "difference_days": int((row["target_date"] - exp_target).days),
                    })
            else:
                # Fallback for calendar-day logic
                exp_target = expected_target.loc[idx] if idx in expected_target.index else pd.NaT
                if pd.notna(exp_target):
                    misaligned_examples.append({
                        "origin_date": str(row["origin_date"].date()),
                        "expected_target": str(exp_target.date()),
                        "actual_target": str(row["target_date"].date()),
                        "difference_days": int((row["target_date"] - exp_target).days),
                    })
    
    return {
        "alignment_ok": n_misaligned == 0,
        "n_misaligned": n_misaligned,
        "n_total": len(df),
        "misaligned_examples": misaligned_examples,
        "validation_method": "step_based" if date_index is not None else "calendar_days",
    }


def compute_persistence_baseline_from_origin(
    predictions_df: pd.DataFrame,
) -> Dict:
    """
    Compute persistence baseline using origin_value (y_hat_base = origin_value).
    This is the mandatory baseline comparison.
    
    Args:
        predictions_df: DataFrame with columns ['origin_value', 'y_true', 'y_pred']
        
    Returns:
        Dictionary with persistence predictions, MAE, RMSE, R2
    """
    if "origin_value" not in predictions_df.columns:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "r2_persistence": np.nan,
            "error": "Missing origin_value column",
        }
    
    df = predictions_df.copy()
    y_true = df["y_true"].to_numpy()
    y_pred_persistence = df["origin_value"].to_numpy()  # y_hat_base = origin_value
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_persistence))
    if valid_mask.sum() == 0:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "r2_persistence": np.nan,
            "error": "No valid data points",
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_persistence[valid_mask]
    
    mae = float(np.mean(np.abs(y_true_valid - y_pred_valid)))
    rmse = float(np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)))
    
    # R2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
    
    return {
        "mae_persistence": mae,
        "rmse_persistence": rmse,
        "r2_persistence": r2,
        "n_valid": int(valid_mask.sum()),
    }


def compute_integrity_report(
    predictions_df: pd.DataFrame,
    full_series: pd.Series,
    horizon: int,
    model_name: str = None,
    cadence: str = "Daily",
    date_index: pd.DatetimeIndex = None,
) -> Dict:
    """
    Compute comprehensive integrity report for forecast predictions.
    
    Args:
        predictions_df: DataFrame with columns ['date', 'y_true', 'y_pred', 'model', 'origin_date', 'target_date', 'origin_value']
        full_series: Full time series (for baselines)
        horizon: Forecast horizon
        model_name: Specific model to analyze (if None, uses first model)
        cadence: "Daily" | "Weekly" | "Monthly"
        
    Returns:
        Integrity report dictionary
    """
    if model_name:
        df = predictions_df[predictions_df["model"] == model_name].copy()
    else:
        df = predictions_df.copy()
        if "model" in df.columns:
            model_name = df["model"].iloc[0]
        else:
            model_name = "unknown"
    
    if len(df) == 0:
        return {"error": "No predictions found for integrity check"}
    
    df = df.sort_values("date")
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    target_dates = pd.to_datetime(df["date"])
    
    # ✅ Test 0: Alignment validation (step-based for business-day data)
    # Use date_index from full_series if available, otherwise None (falls back to calendar days)
    date_idx = date_index if date_index is not None else (full_series.index if isinstance(full_series, pd.Series) else None)
    alignment_check = validate_alignment(df, horizon, cadence, date_index=date_idx)
    
    # Test A: Shift sanity check
    shift_check = shift_sanity_check(y_true, y_pred, horizon)
    
    # ✅ Test B: Mandatory persistence baseline (using origin_value)
    persistence_baseline = compute_persistence_baseline_from_origin(df)
    
    # Test C: Additional baselines (seasonal naive)
    baselines = compute_baselines(full_series, target_dates, horizon)
    baseline_maes = compute_baseline_maes(y_true, baselines["persistence"], baselines["seasonal_naive"])
    
    # Model MAE, RMSE, R2
    mae_model = float(np.mean(np.abs(y_true - y_pred)))
    rmse_model = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_model = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
    
    # ✅ Skill score: (MAE_base - MAE_model) / MAE_base
    # Positive skill means model is better than baseline
    if not np.isnan(persistence_baseline.get("mae_persistence", np.nan)) and persistence_baseline["mae_persistence"] > 0:
        skill_pct = ((persistence_baseline["mae_persistence"] - mae_model) / persistence_baseline["mae_persistence"]) * 100.0
    else:
        skill_pct = np.nan
    
    # Improvement percentages (for backward compatibility)
    improvement_vs_persistence = skill_pct if not np.isnan(skill_pct) else np.nan
    
    improvement_vs_seasonal = (
        ((baseline_maes["mae_seasonal_naive"] - mae_model) / baseline_maes["mae_seasonal_naive"] * 100)
        if not np.isnan(baseline_maes["mae_seasonal_naive"]) and baseline_maes["mae_seasonal_naive"] > 0
        else np.nan
    )
    
    report = {
        "horizon": int(horizon),
        "model": str(model_name),
        "prediction_indexing": "target_date",  # Assert correct indexing
        "n_predictions": int(len(df)),
        # ✅ Alignment check
        "alignment_ok": alignment_check["alignment_ok"],
        "n_misaligned": alignment_check.get("n_misaligned", 0),
        "misaligned_examples": alignment_check.get("misaligned_examples", []),
        # Shift check
        "best_shift": shift_check["best_shift"],
        "mae_shift0": shift_check["mae_shift0"],
        "mae_best": shift_check["best_mae"],
        "mae_shift_minus_h": shift_check.get("mae_shift_minus_h", np.nan),
        "improvement_pct_vs_shift0": shift_check["improvement_pct"],
        "improvement_ratio": shift_check.get("improvement_ratio", np.nan),
        "lag_warning": shift_check["lag_warning"],
        "is_critical_timestamping_bug": shift_check.get("is_critical_timestamping_bug", False),
        "is_lag0_issue": shift_check.get("is_lag0_issue", False),  # ✅ FIX 4: Lag_0 issue pattern
        # Model performance
        "mae_model": float(mae_model),
        "rmse_model": float(rmse_model),
        "r2_model": float(r2_model),
        # ✅ Mandatory persistence baseline (using origin_value)
        "mae_persistence": persistence_baseline.get("mae_persistence", np.nan),
        "rmse_persistence": persistence_baseline.get("rmse_persistence", np.nan),
        "r2_persistence": persistence_baseline.get("r2_persistence", np.nan),
        # ✅ Skill score
        "skill_pct": float(skill_pct) if not np.isnan(skill_pct) else np.nan,
        # Additional baselines (for backward compatibility)
        "mae_seasonal_naive": baseline_maes["mae_seasonal_naive"],
        "improvement_pct_vs_persistence": float(improvement_vs_persistence) if not np.isnan(improvement_vs_persistence) else np.nan,
        "improvement_pct_vs_seasonal_naive": float(improvement_vs_seasonal) if not np.isnan(improvement_vs_seasonal) else np.nan,
        # Leakage (will be added separately)
        "mae_shuffled_target": np.nan,
        "leakage_warning": False,
    }
    
    return report
