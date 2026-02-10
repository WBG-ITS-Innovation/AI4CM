"""Main preprocessing module for Treasury data.

Handles Balance_by_Day Excel format (multi-sheet, dates as columns, metrics as rows)
and CSV inputs with date columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .holidays import georgian_holidays_range


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing run."""

    input_path: str
    date_col: Optional[str] = None
    balance_col: Optional[str] = None
    variant: str = "raw"
    business_days_zero_flows: bool = True
    weekday_weeks: int = 8
    out_root: str = "data_preprocessed"
    run_outputs_dir: str = "outputs"
    expected_csv: Optional[str] = None
    save_parquet: bool = False
    sheet_name: Optional[str] = None
    header_row: Optional[int] = None


def parse_balance_by_day_excel(
    excel_path: Path, sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse Balance_by_Day Excel format.
    
    Format:
    - Row containing dates starts at column C (index 2)
    - First 2 columns are labels (geo, english)
    - Subsequent rows contain numeric values per date
    - Duplicate metric names must be SUMMED per date
    - Blank numeric cells = 0.0 (not NaN)
    
    Args:
        excel_path: Path to Excel file
        sheet_name: Specific sheet to read (None = all year sheets)
        
    Returns:
        DataFrame with date as index, metric names as columns
    """
    import openpyxl

    _log(f"Reading Excel file: {excel_path}")

    # Get sheet names
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    if sheet_name:
        sheet_names = [sheet_name] if sheet_name in wb.sheetnames else []
        if not sheet_names:
            raise ValueError(f"Sheet '{sheet_name}' not found in Excel file")
    else:
        # Auto-detect year sheets (common pattern: "2015", "2016", etc.)
        sheet_names = [s for s in wb.sheetnames if s.strip().isdigit() or "year" in s.lower()]
        if not sheet_names:
            # Fallback to all sheets except very common non-data sheets
            exclude = {"Sheet1", "Sheet", "Summary", "Overview", "Notes"}
            sheet_names = [s for s in wb.sheetnames if s not in exclude]
            if not sheet_names:
                sheet_names = wb.sheetnames

    if not sheet_names:
        raise ValueError(f"No sheets found in {excel_path}")

    _log(f"Processing {len(sheet_names)} sheet(s): {sheet_names}")

    all_data = []

    for sname in sheet_names:
        _log(f"  Processing sheet: {sname}")
        try:
            df = pd.read_excel(excel_path, sheet_name=sname, header=None, engine="openpyxl")
        except Exception as e:
            _log(f"    Warning: Could not read sheet {sname}: {e}, skipping")
            continue

        if len(df) < 2 or len(df.columns) < 3:
            _log(f"    Warning: Sheet {sname} too small, skipping")
            continue

        # Find row with dates
        # Dates can start at column C (index 2) or column D (index 3)
        # Look for row that has date-like values
        date_row_idx = None
        date_start_col = None
        
        for i in range(min(20, len(df))):  # Check first 20 rows
            # Try column 2 (index 2) first, then column 3 (index 3)
            for col_idx in [2, 3]:
                if len(df.columns) <= col_idx:
                    continue
                val = df.iloc[i, col_idx]
                if pd.notna(val):
                    try:
                        # Try to parse as date
                        test_date = pd.to_datetime(str(val), errors="raise")
                        # Check if next few columns also look like dates
                        if len(df.columns) > col_idx + 1:
                            val2 = df.iloc[i, col_idx + 1]
                            if pd.notna(val2):
                                try:
                                    pd.to_datetime(str(val2), errors="raise")
                                    date_row_idx = i
                                    date_start_col = col_idx
                                    break
                                except Exception:
                                    pass
                    except Exception:
                        continue
            if date_row_idx is not None:
                break

        if date_row_idx is None:
            _log(f"    Warning: Could not find date row in sheet {sname}, skipping")
            continue

        # Extract dates from row (starting at detected column)
        date_row = df.iloc[date_row_idx, date_start_col:].values
        dates = []
        for val in date_row:
            if pd.isna(val):
                break
            try:
                dt = pd.to_datetime(str(val), errors="raise")
                dates.append(dt)
            except Exception:
                # Stop at first non-date
                break

        if not dates:
            _log(f"    Warning: No valid dates found in sheet {sname}, skipping")
            continue

        _log(f"    Found {len(dates)} dates from {dates[0]} to {dates[-1]}")

        # Extract metric rows (rows after date_row_idx)
        # Column 0 (index 0): Georgian labels (optional)
        # Column 1 (index 1): English metric names
        # Values start at date_start_col (same column where dates start)
        metrics_data = {}
        for i in range(date_row_idx + 1, len(df)):
            row = df.iloc[i]
            if len(row) < date_start_col + 1:
                continue

            # Get metric name from English label (column 1, index 1)
            metric_name = None
            if len(row) > 1:
                name_val = row.iloc[1]
                if pd.notna(name_val):
                    metric_name = str(name_val).strip()
                    if metric_name.lower() in {"nan", "none", "", "null"}:
                        continue

            if not metric_name:
                continue

            # Extract values for this metric (starting at date_start_col, same as dates)
            values = []
            for j in range(date_start_col, min(len(row), date_start_col + len(dates))):
                val = row.iloc[j]
                # Blank cells = 0.0 (not NaN)
                if pd.isna(val):
                    val = 0.0
                else:
                    try:
                        val = float(val)
                        if not np.isfinite(val):
                            val = 0.0
                    except (ValueError, TypeError):
                        val = 0.0
                values.append(val)

            if len(values) != len(dates):
                # Pad or truncate to match dates
                if len(values) < len(dates):
                    values.extend([0.0] * (len(dates) - len(values)))
                else:
                    values = values[: len(dates)]

            # Handle duplicates: SUM them
            if metric_name in metrics_data:
                # Sum with existing values
                existing = metrics_data[metric_name]
                metrics_data[metric_name] = [
                    existing[j] + values[j] for j in range(len(dates))
                ]
            else:
                metrics_data[metric_name] = values

        # Build DataFrame for this sheet
        if metrics_data:
            sheet_df = pd.DataFrame(metrics_data, index=pd.DatetimeIndex(dates))
            all_data.append(sheet_df)
            _log(f"    Extracted {len(metrics_data)} metrics")

    if not all_data:
        raise ValueError("No data extracted from Excel file")

    # Combine all sheets
    combined = pd.concat(all_data, axis=0, sort=False)
    # Sum duplicates across sheets (same metric name, same date)
    combined = combined.groupby(combined.index).sum()

    # Sort by date
    combined = combined.sort_index()

    _log(f"Final: {len(combined)} unique dates, {len(combined.columns)} metrics")
    return combined


def parse_csv_input(
    csv_path: Path, date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse CSV input with date column.
    
    Args:
        csv_path: Path to CSV file
        date_col: Name of date column (auto-detect if None)
        
    Returns:
        DataFrame with date as index, other columns as metrics
    """
    _log(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find date column
    if date_col:
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in CSV")
        date_col_found = date_col
    else:
        # Auto-detect: look for columns with date-like values
        date_col_found = None
        for col in df.columns:
            try:
                test = pd.to_datetime(df[col], errors="coerce")
                if test.notna().mean() > 0.8:  # 80% valid dates
                    date_col_found = col
                    break
            except Exception:
                continue

        if not date_col_found:
            raise ValueError("Could not auto-detect date column in CSV")

    # Set date as index
    df[date_col_found] = pd.to_datetime(df[date_col_found], errors="coerce")
    df = df.dropna(subset=[date_col_found])
    df = df.set_index(date_col_found)
    df.index.name = "date"

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    _log(f"Parsed CSV: {len(df)} dates, {len(df.columns)} columns")
    return df


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_weekend and is_holiday flags to DataFrame.
    
    Args:
        df: DataFrame with date index
        
    Returns:
        DataFrame with added flags
    """
    idx = pd.DatetimeIndex(df.index)
    df = df.copy()

    # Weekend flag
    df["is_weekend"] = (idx.weekday >= 5).astype(int)

    # Georgian holidays
    if len(idx) > 0:
        holidays = georgian_holidays_range(idx.min().date(), idx.max().date())
        holiday_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in holidays])
        df["is_holiday"] = idx.normalize().isin(holiday_dates.normalize()).astype(int)
    else:
        df["is_holiday"] = 0

    return df


def reindex_to_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to full daily calendar (including weekends/holidays).
    
    On non-business days, metric columns remain NaN in raw output.
    
    Args:
        df: DataFrame with date index
        
    Returns:
        DataFrame reindexed to daily calendar
    """
    if len(df) == 0:
        return df

    # Create full daily calendar
    start = df.index.min().normalize()
    end = df.index.max().normalize()
    full_calendar = pd.date_range(start=start, end=end, freq="D")

    # Reindex (non-business days will have NaN for metrics)
    df_reindexed = df.reindex(full_calendar)

    # Add calendar flags to all rows
    df_reindexed = add_calendar_flags(df_reindexed)

    return df_reindexed


def apply_variant_raw(
    df: pd.DataFrame, business_days_zero_flows: bool
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply raw variant: minimal transformation.
    
    - Reindex to full daily calendar
    - Metrics remain NaN on non-business days
    - Include is_weekend/is_holiday flags
    
    Args:
        df: DataFrame with date index and metric columns
        business_days_zero_flows: If True, set flows to 0 on non-business days
        
    Returns:
        (processed_df, summary_dict)
    """
    _log("Applying variant: raw")

    # Reindex to daily calendar
    df_out = reindex_to_daily_calendar(df)

    # Identify level columns (e.g., "State budget balance")
    level_cols = []
    for col in df.columns:
        if "balance" in str(col).lower() or "level" in str(col).lower():
            level_cols.append(col)

    # For flow columns on non-business days
    flow_cols = [c for c in df.columns if c not in level_cols and c not in ["is_weekend", "is_holiday"]]
    
    if business_days_zero_flows:
        # Set flows to 0 on non-business days
        non_business = (df_out["is_weekend"] == 1) | (df_out["is_holiday"] == 1)
        for col in flow_cols:
            df_out.loc[non_business, col] = 0.0

    summary = {
        "variant": "raw",
        "level_columns": level_cols,
        "flow_columns": flow_cols,
        "business_days_zero_flows": business_days_zero_flows,
    }

    return df_out, summary


def apply_variant_clean_conservative(
    df: pd.DataFrame, business_days_zero_flows: bool
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply clean_conservative variant: safe cleaning.
    
    - Level columns: forward-fill across ALL calendar days
    - Flow columns: set to 0 on non-business days if flag is True, else leave NaN
    
    Args:
        df: DataFrame with date index and metric columns
        business_days_zero_flows: If True, set flows to 0 on non-business days
        
    Returns:
        (processed_df, summary_dict)
    """
    _log("Applying variant: clean_conservative")

    # Start from raw
    df_out, _ = apply_variant_raw(df, business_days_zero_flows)

    # Identify level columns
    level_cols = []
    for col in df.columns:
        if "balance" in str(col).lower() or "level" in str(col).lower():
            level_cols.append(col)

    # Forward-fill level columns across ALL days
    for col in level_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].ffill()

    # Flow columns: handle non-business days
    flow_cols = [c for c in df.columns if c not in level_cols and c not in ["is_weekend", "is_holiday"]]
    
    if business_days_zero_flows:
        non_business = (df_out["is_weekend"] == 1) | (df_out["is_holiday"] == 1)
        for col in flow_cols:
            if col in df_out.columns:
                df_out.loc[non_business, col] = 0.0

    summary = {
        "variant": "clean_conservative",
        "level_columns": level_cols,
        "flow_columns": flow_cols,
        "business_days_zero_flows": business_days_zero_flows,
        "imputations": {"level_forward_fill": len(level_cols)},
    }

    return df_out, summary


def apply_variant_clean_treasury(
    df: pd.DataFrame, business_days_zero_flows: bool, weekday_weeks: int
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply clean_treasury variant: Treasury-friendly cleaning.
    
    - Start from clean_conservative
    - For flow columns on BUSINESS DAYS only:
      - Compute weekday reference (median of past N weeks of same weekday)
      - Impute NaN with weekday reference
      - Clip extreme outliers using MAD (8*MAD threshold)
    
    Args:
        df: DataFrame with date index and metric columns
        business_days_zero_flows: If True, set flows to 0 on non-business days
        weekday_weeks: Number of weeks to use for weekday reference
        
    Returns:
        (processed_df, summary_dict)
    """
    _log("Applying variant: clean_treasury")

    # Start from clean_conservative
    df_out, base_summary = apply_variant_clean_conservative(df, business_days_zero_flows)

    # Identify columns
    level_cols = base_summary.get("level_columns", [])
    flow_cols = [c for c in df.columns if c not in level_cols and c not in ["is_weekend", "is_holiday"]]

    # Business days mask
    is_business = (df_out["is_weekend"] == 0) & (df_out["is_holiday"] == 0)

    imputation_counts = {}
    clipping_counts = {}

    for col in flow_cols:
        if col not in df_out.columns:
            continue

        series = df_out[col].copy()
        business_series = series[is_business]

        # Compute weekday reference for each weekday
        weekday_refs = {}
        for dow in range(7):  # 0=Monday, 6=Sunday
            # Get past N weeks of this weekday (on business days only)
            dow_mask = (df_out.index.weekday == dow) & is_business
            dow_values = series[dow_mask]

            if len(dow_values) >= weekday_weeks:
                # Use median of last N occurrences of this weekday
                ref = dow_values.tail(weekday_weeks).median()
            elif len(dow_values) > 0:
                # Fallback to mean if too few
                ref = dow_values.mean()
            else:
                ref = np.nan

            weekday_refs[dow] = ref

        # Impute NaN on business days with weekday reference
        imputed = 0
        for idx in business_series.index:
            if pd.isna(series.loc[idx]):
                dow = idx.weekday()
                ref = weekday_refs.get(dow, np.nan)
                if not pd.isna(ref):
                    series.loc[idx] = ref
                    imputed += 1

        # Clip outliers using MAD (on business days only)
        clipped = 0
        for dow in range(7):
            dow_mask = (df_out.index.weekday == dow) & is_business
            dow_values = series[dow_mask]

            if len(dow_values) < 2:
                continue

            median = dow_values.median()
            mad = (dow_values - median).abs().median()

            if mad > 0 and np.isfinite(mad):
                threshold = 8 * mad
                upper = median + threshold
                lower = median - threshold

                # Clip values outside threshold
                before = series.loc[dow_mask].copy()
                series.loc[dow_mask] = series.loc[dow_mask].clip(lower=lower, upper=upper)
                after = series.loc[dow_mask]

                clipped += int((before != after).sum())
            elif len(dow_values) > 0:
                # Fallback to std-based clipping
                std = dow_values.std()
                if std > 0 and np.isfinite(std):
                    median = dow_values.median()
                    threshold = 8 * std
                    upper = median + threshold
                    lower = median - threshold
                    before = series.loc[dow_mask].copy()
                    series.loc[dow_mask] = series.loc[dow_mask].clip(lower=lower, upper=upper)
                    after = series.loc[dow_mask]
                    clipped += int((before != after).sum())

        df_out[col] = series
        imputation_counts[col] = imputed
        clipping_counts[col] = int(clipped)

    summary = {
        "variant": "clean_treasury",
        "level_columns": level_cols,
        "flow_columns": flow_cols,
        "business_days_zero_flows": business_days_zero_flows,
        "weekday_weeks": weekday_weeks,
        "imputations": imputation_counts,
        "clipped_outliers": clipping_counts,
    }

    return df_out, summary


def compare_with_expected(output_df: pd.DataFrame, expected_path: Path) -> Dict:
    """
    Compare output with expected CSV for regression testing.
    
    Args:
        output_df: Output DataFrame
        expected_path: Path to expected CSV
        
    Returns:
        Comparison summary dict
    """
    if not expected_path.exists():
        return {"error": f"Expected file not found: {expected_path}"}

    try:
        expected_df = pd.read_csv(expected_path)
        expected_df["date"] = pd.to_datetime(expected_df["date"])
        expected_df = expected_df.set_index("date")

        # Compare structure
        output_cols = set(output_df.columns)
        expected_cols = set(expected_df.columns)

        # Compare values for common columns and dates
        common_cols = output_cols & expected_cols
        common_dates = output_df.index.intersection(expected_df.index)

        diffs = {}
        for col in common_cols:
            if col in ["is_weekend", "is_holiday"]:
                continue  # Skip flags for comparison
            output_vals = output_df.loc[common_dates, col]
            expected_vals = expected_df.loc[common_dates, col]

            # Compare numeric values
            numeric_mask = output_vals.notna() & expected_vals.notna()
            if numeric_mask.any():
                diff = (output_vals[numeric_mask] - expected_vals[numeric_mask]).abs()
                max_diff = float(diff.max()) if len(diff) > 0 else 0.0
                mean_diff = float(diff.mean()) if len(diff) > 0 else 0.0
                diffs[col] = {"max_diff": max_diff, "mean_diff": mean_diff, "n_diffs": int((diff > 1e-6).sum())}

        return {
            "expected_file": str(expected_path),
            "output_rows": len(output_df),
            "expected_rows": len(expected_df),
            "output_cols": list(output_cols),
            "expected_cols": list(expected_cols),
            "common_cols": list(common_cols),
            "common_dates": len(common_dates),
            "differences": diffs,
        }
    except Exception as e:
        return {"error": f"Comparison failed: {e}"}


def run_preprocess(cfg: PreprocessConfig) -> Dict:
    """
    Main preprocessing function.
    
    Args:
        cfg: Preprocessing configuration
        
    Returns:
        Report dictionary
    """
    _log("=" * 80)
    _log("Starting preprocessing")
    _log(f"Input: {cfg.input_path}")
    _log(f"Variant: {cfg.variant}")

    input_path = Path(cfg.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Parse input
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = parse_balance_by_day_excel(input_path, cfg.sheet_name)
    elif input_path.suffix.lower() == ".csv":
        df = parse_csv_input(input_path, cfg.date_col)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Apply variant
    variant = cfg.variant.lower().strip()
    if variant == "raw":
        df_out, summary = apply_variant_raw(df, cfg.business_days_zero_flows)
    elif variant == "clean_conservative":
        df_out, summary = apply_variant_clean_conservative(df, cfg.business_days_zero_flows)
    elif variant == "clean_treasury":
        df_out, summary = apply_variant_clean_treasury(
            df, cfg.business_days_zero_flows, cfg.weekday_weeks
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be: raw, clean_conservative, clean_treasury")

    # Reset index to have date as column
    # The index should be a DatetimeIndex from reindex_to_daily_calendar
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out = df_out.reset_index()
        # The reset_index will create a column from the index
        # Find the date column (should be the first column or named 'date')
        date_col_name = None
        for col in df_out.columns:
            if col == "date" or pd.api.types.is_datetime64_any_dtype(df_out[col]):
                date_col_name = col
                break
        
        if date_col_name and date_col_name != "date":
            df_out = df_out.rename(columns={date_col_name: "date"})
        elif not date_col_name:
            # If no date column found, the index might not have been reset properly
            # Create date column from index
            df_out["date"] = df_out.index
            df_out = df_out.reset_index(drop=True)
    
    # Ensure date column exists and is datetime
    if "date" not in df_out.columns:
        raise ValueError("Date column not found after processing")
    df_out["date"] = pd.to_datetime(df_out["date"])

    # Write output
    out_root = Path(cfg.out_root)
    variant_dir = out_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    output_csv = variant_dir / f"{base_name}__{variant}.csv"
    df_out.to_csv(output_csv, index=False)
    _log(f"Saved: {output_csv}")

    output_parquet = None
    if cfg.save_parquet:
        output_parquet = variant_dir / f"{base_name}__{variant}.parquet"
        df_out.to_parquet(output_parquet, index=False)
        _log(f"Saved: {output_parquet}")

    # Write preview
    run_outputs = Path(cfg.run_outputs_dir)
    run_outputs.mkdir(parents=True, exist_ok=True)
    preview_path = run_outputs / "preprocess_preview.csv"
    df_out.head(500).to_csv(preview_path, index=False)
    _log(f"Saved preview: {preview_path}")

    # Build report
    report = {
        "output_csv": str(output_csv),
        "output_parquet": str(output_parquet) if output_parquet else None,
        "preview_path": str(preview_path),
        "variant": variant,
        "input_path": str(input_path),
        "row_count": len(df_out),
        "col_count": len(df_out.columns),
        "summary": summary,
    }

    # Compare with expected if provided
    if cfg.expected_csv:
        expected_path = Path(cfg.expected_csv)
        comparison = compare_with_expected(df_out.set_index("date"), expected_path)
        report["comparison"] = comparison

    # Write report
    report_path = run_outputs / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Saved report: {report_path}")

    _log("Preprocessing completed successfully")
    return report
