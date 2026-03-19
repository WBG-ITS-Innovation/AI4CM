# data_preflight.py — Pre-run data quality validation
"""
Validates a dataset before launching a forecasting pipeline.
Returns blockers (prevent run), warnings (inform but allow), and summary info.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def run_preflight(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    cadence: str,
    horizon: int,
) -> Dict[str, Any]:
    """Run data quality checks and return a structured report.

    Returns
    -------
    dict with keys:
        blockers : list[str]  — Fatal issues that prevent a run.
        warnings : list[str]  — Non-fatal issues the user should know about.
        info     : dict       — Summary statistics for display.
    """
    blockers: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    n_rows = len(df)
    info["rows"] = n_rows

    # ── 1. Date column checks ────────────────────────────────────────
    if date_col not in df.columns:
        blockers.append(f"Date column '{date_col}' not found in dataset.")
        return {"blockers": blockers, "warnings": warnings, "info": info}

    dates = pd.to_datetime(df[date_col], errors="coerce")
    n_unparseable = int(dates.isna().sum())
    if n_unparseable == n_rows:
        blockers.append(f"Date column '{date_col}' could not be parsed as dates (all NaT).")
        return {"blockers": blockers, "warnings": warnings, "info": info}
    if n_unparseable > 0:
        warnings.append(f"{n_unparseable} rows ({n_unparseable/n_rows*100:.1f}%) have unparseable dates.")

    valid_dates = dates.dropna().sort_values()
    info["date_min"] = str(valid_dates.min().date())
    info["date_max"] = str(valid_dates.max().date())
    info["date_span_days"] = int((valid_dates.max() - valid_dates.min()).days)

    # ── 2. Target column checks ──────────────────────────────────────
    if target not in df.columns:
        blockers.append(f"Target column '{target}' not found in dataset.")
        return {"blockers": blockers, "warnings": warnings, "info": info}

    target_vals = pd.to_numeric(df[target], errors="coerce")
    n_non_numeric = int(target_vals.isna().sum() - df[target].isna().sum())
    if n_non_numeric > n_rows * 0.5:
        blockers.append(f"Target column '{target}' is mostly non-numeric ({n_non_numeric} non-numeric values).")
        return {"blockers": blockers, "warnings": warnings, "info": info}

    valid_target = target_vals.dropna()
    info["target_count"] = int(len(valid_target))
    info["target_mean"] = float(valid_target.mean()) if len(valid_target) > 0 else None
    info["target_std"] = float(valid_target.std()) if len(valid_target) > 1 else None
    info["target_min"] = float(valid_target.min()) if len(valid_target) > 0 else None
    info["target_max"] = float(valid_target.max()) if len(valid_target) > 0 else None

    # ── 3. Sufficient data check ─────────────────────────────────────
    min_required = horizon * 3 + 30
    if len(valid_target) < min_required:
        blockers.append(
            f"Insufficient data: {len(valid_target)} valid rows but need at least "
            f"{min_required} (horizon*3 + 30) for meaningful cross-validation."
        )

    # ── 4. Missing values ────────────────────────────────────────────
    n_missing = int(target_vals.isna().sum())
    missing_pct = n_missing / n_rows * 100 if n_rows > 0 else 0
    info["missing_count"] = n_missing
    info["missing_pct"] = round(missing_pct, 1)
    if missing_pct > 5:
        warnings.append(f"Target has {missing_pct:.1f}% missing values ({n_missing} rows).")

    # ── 5. Duplicate dates ───────────────────────────────────────────
    n_dup_dates = int(valid_dates.duplicated().sum())
    if n_dup_dates > 0:
        warnings.append(f"{n_dup_dates} duplicate dates detected. The pipeline will use the last value per date.")

    # ── 6. Date gaps ─────────────────────────────────────────────────
    if cadence.lower() == "daily" and len(valid_dates) > 2:
        diffs = valid_dates.diff().dropna()
        # Ignore weekends (2-3 day gaps are normal for business-day data)
        large_gaps = diffs[diffs > pd.Timedelta(days=5)]
        if len(large_gaps) > 0:
            max_gap = large_gaps.max()
            warnings.append(
                f"{len(large_gaps)} date gap(s) > 5 days detected (max: {max_gap.days} days). "
                f"The pipeline fills missing business days automatically."
            )

    # ── 7. Outlier detection ─────────────────────────────────────────
    if len(valid_target) > 60:
        # Rolling z-score
        s = valid_target.reset_index(drop=True)
        roll_mean = s.rolling(60, min_periods=20).mean()
        roll_std = s.rolling(60, min_periods=20).std()
        z = ((s - roll_mean) / roll_std.replace(0, np.nan)).abs()
        n_outliers = int((z > 4.0).sum())
        if n_outliers > 0:
            warnings.append(
                f"{n_outliers} potential outlier(s) detected (|z-score| > 4.0 vs 60-period rolling stats). "
                f"Consider reviewing extreme values."
            )

    # ── 8. Constant series ───────────────────────────────────────────
    if len(valid_target) > 1 and valid_target.std() == 0:
        blockers.append("Target series is constant (std = 0). Forecasting a constant series is meaningless.")

    # ── 9. Frequency detection ───────────────────────────────────────
    if len(valid_dates) > 5:
        freq_guess = pd.infer_freq(valid_dates.head(100))
        info["detected_freq"] = freq_guess or "irregular"
    else:
        info["detected_freq"] = "too few dates"

    # ── 10. Exogenous columns count ──────────────────────────────────
    numeric_cols = [c for c in df.columns if c != date_col and c != target
                    and pd.api.types.is_numeric_dtype(df[c])]
    info["exog_columns"] = len(numeric_cols)

    return {"blockers": blockers, "warnings": warnings, "info": info}
