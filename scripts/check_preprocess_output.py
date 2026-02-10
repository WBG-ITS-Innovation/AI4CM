#!/usr/bin/env python3
"""
Sanity checks for Georgia Treasury preprocessing outputs.

Usage examples:
  python scripts/check_preprocess_output.py \
    --treasury data_preprocessed/clean_treasury/Balance_by_Day_2015-2025__clean_treasury.csv \
    --conservative data_preprocessed/clean_conservative/Balance_by_Day_2015-2025__clean_conservative.csv

Options:
  --max_missing_biz_days_conservative 1   # allow up to N business-days with any missing flows
  --expect_zero_flows_on_nonbiz true      # enforce flows are 0/NaN on weekends/holidays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_BASE_COLS = {"date", "is_weekend", "is_holiday"}
DEFAULT_LEVEL_COL = "State budget balance"


def _die(msg: str, code: int = 1) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(code)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        _die(f"File not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        _die(f"Missing 'date' column in {path.name}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        _die(f"Some dates could not be parsed in {path.name}. Example rows:\n{bad}")
    return df


def assert_daily_continuous(df: pd.DataFrame, name: str) -> None:
    df = df.sort_values("date")
    # duplicates
    if df["date"].duplicated().any():
        dups = df[df["date"].duplicated(keep=False)]["date"].head(10).tolist()
        _die(f"{name}: duplicate dates found (examples): {dups}")

    # continuous daily range
    dmin, dmax = df["date"].min(), df["date"].max()
    expected_n = (dmax - dmin).days + 1
    if len(df) != expected_n:
        # find gaps
        full = pd.date_range(dmin, dmax, freq="D")
        missing = full.difference(df["date"])
        _die(
            f"{name}: date range not continuous daily. "
            f"Expected {expected_n} rows from {dmin.date()} to {dmax.date()}, got {len(df)}. "
            f"Missing dates example: {list(missing[:10].date)}"
        )

    _ok(f"{name}: daily continuity OK ({dmin.date()} → {dmax.date()}, {len(df)} rows)")


def assert_flags(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        _die(f"{name}: missing required columns: {sorted(missing)}")

    # weekend correctness
    expected_weekend = (df["date"].dt.weekday >= 5).astype(int)
    mism = (df["is_weekend"].astype(int) != expected_weekend)
    if mism.any():
        ex = df.loc[mism, ["date", "is_weekend"]].head(10)
        _die(f"{name}: is_weekend mismatch examples:\n{ex}")

    # holiday must be 0/1
    hol = df["is_holiday"]
    if not set(pd.unique(hol.dropna())).issubset({0, 1, False, True}):
        _die(f"{name}: is_holiday contains values outside {{0,1}}: {sorted(set(pd.unique(hol)))}")

    _ok(f"{name}: flags OK (is_weekend matches calendar; is_holiday is binary)")


def assert_known_holidays(df: pd.DataFrame, name: str) -> None:
    # Minimal "sentinel" checks to catch Easter regression.
    # If these fail, your Orthodox Easter logic is likely wrong again.
    must_be_holiday = [
        "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06",
        "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21",
        "2024-01-01", "2025-01-01",
    ]
    must_not_be_holiday = ["2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19"]

    df_idx = df.set_index("date")
    date_range = (df_idx.index.min(), df_idx.index.max())

    for d in must_be_holiday:
        ts = pd.Timestamp(d)
        if ts < date_range[0] or ts > date_range[1]:
            # Date is outside the data range, skip validation
            continue
        if ts not in df_idx.index:
            _die(f"{name}: missing date {d} (cannot validate holidays)")
        if int(df_idx.loc[ts, "is_holiday"]) != 1:
            _die(f"{name}: expected is_holiday=1 on {d}, got {df_idx.loc[ts, 'is_holiday']}")

    for d in must_not_be_holiday:
        ts = pd.Timestamp(d)
        if ts < date_range[0] or ts > date_range[1]:
            # Date is outside the data range, skip validation
            continue
        if ts not in df_idx.index:
            _die(f"{name}: missing date {d} (cannot validate holidays)")
        if int(df_idx.loc[ts, "is_holiday"]) != 0:
            _die(f"{name}: expected is_holiday=0 on {d}, got {df_idx.loc[ts, 'is_holiday']}")

    _ok(f"{name}: known holiday sentinels OK")


def get_flow_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in ["date", "is_weekend", "is_holiday"]]
    # level col may or may not exist; treat it as non-flow if present
    if DEFAULT_LEVEL_COL in cols:
        cols.remove(DEFAULT_LEVEL_COL)
    return cols


def business_mask(df: pd.DataFrame) -> pd.Series:
    return (df["is_weekend"].astype(int) == 0) & (df["is_holiday"].astype(int) == 0)


def assert_missingness_rules(
    df: pd.DataFrame,
    name: str,
    max_missing_biz_days: int,
    expect_zero_flows_on_nonbiz: bool,
) -> None:
    flow_cols = get_flow_cols(df)
    if not flow_cols:
        _die(f"{name}: could not infer flow columns")

    biz = business_mask(df)

    # business-day NaNs in flows
    any_missing = df.loc[biz, flow_cols].isna().any(axis=1)
    n_missing_days = int(any_missing.sum())
    if n_missing_days > max_missing_biz_days:
        examples = df.loc[biz & any_missing, ["date"]].head(10)["date"].dt.date.tolist()
        _die(
            f"{name}: too many business-days with missing flow values: {n_missing_days} "
            f"(max allowed {max_missing_biz_days}). Examples: {examples}"
        )
    _ok(f"{name}: business-day missing flows OK (missing biz-days={n_missing_days}, allowed={max_missing_biz_days})")

    # Non-business day flow behavior (only enforce if configured)
    if expect_zero_flows_on_nonbiz:
        nonbiz = ~biz
        sub = df.loc[nonbiz, flow_cols]
        # allow NaN or 0; flag nonzero
        bad = ~(sub.isna() | (sub == 0))
        if bad.any().any():
            # show first few cells that violate
            bad_rows = df.loc[nonbiz, ["date"]].copy()
            bad_rows["bad_count"] = bad.sum(axis=1).values
            bad_rows = bad_rows[bad_rows["bad_count"] > 0].sort_values("bad_count", ascending=False).head(10)
            _die(
                f"{name}: found nonzero flow values on non-business days (expected 0 or NaN). "
                f"Examples (date, bad_count):\n{bad_rows.to_string(index=False)}"
            )
        _ok(f"{name}: non-business-day flows are 0/NaN as expected")


def assert_same_schema(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> None:
    if list(df_a.columns) != list(df_b.columns):
        _warn(f"{name_a} vs {name_b}: column order differs")
    set_a, set_b = set(df_a.columns), set(df_b.columns)
    if set_a != set_b:
        _die(f"{name_a} vs {name_b}: column sets differ.\nOnly in {name_a}: {sorted(set_a-set_b)}\nOnly in {name_b}: {sorted(set_b-set_a)}")
    _ok(f"{name_a} vs {name_b}: schema sets match")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treasury", type=str, required=True, help="Path to clean_treasury CSV")
    ap.add_argument("--conservative", type=str, required=True, help="Path to clean_conservative CSV")
    ap.add_argument("--max_missing_biz_days_conservative", type=int, default=1)
    ap.add_argument("--max_missing_biz_days_treasury", type=int, default=0)
    ap.add_argument("--expect_zero_flows_on_nonbiz", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args()

    expect_zero = args.expect_zero_flows_on_nonbiz.lower() == "true"

    p_t = Path(args.treasury)
    p_c = Path(args.conservative)

    df_t = load_csv(p_t)
    df_c = load_csv(p_c)

    assert_same_schema(df_t, df_c, "clean_treasury", "clean_conservative")

    for df, nm in [(df_t, "clean_treasury"), (df_c, "clean_conservative")]:
        assert_daily_continuous(df, nm)
        assert_flags(df, nm)
        assert_known_holidays(df, nm)

    assert_missingness_rules(
        df_t,
        "clean_treasury",
        max_missing_biz_days=args.max_missing_biz_days_treasury,
        expect_zero_flows_on_nonbiz=expect_zero,
    )
    assert_missingness_rules(
        df_c,
        "clean_conservative",
        max_missing_biz_days=args.max_missing_biz_days_conservative,
        expect_zero_flows_on_nonbiz=expect_zero,
    )

    _ok("All preprocessing checks passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
