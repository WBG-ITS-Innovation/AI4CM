from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------- Logging ----------------
def _ts() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _log(msg: str): print(f"[{_ts()}] {msg}", flush=True)

# ---------------- Reading helpers ----------------
DATE_NAMES = {
    # en
    "date","dt","as_of","as_of_date","posting_date","value_date",
    # fr
    "date valeur","date_op","date opération","date de valeur",
    # ru
    "дата","дата операции","дата поступления","дата списания",
}
BAL_NAMES = {
    # en
    "balance","ending_balance","net cash balance","cash_balance","closing_balance",
    "closing bal","ending bal","cash","amount","net_flow","netflow",
    # fr
    "solde","solde fin","solde fin de jour","encaisse","montant",
    # ru
    "остаток","остаток на конец дня","сальдо","сумма","остаток средств"
}

def _read_excel(src: Path, sheet: Optional[str], header_row: Optional[int]) -> pd.DataFrame:
    import openpyxl  # ensure available
    kw = {}
    if sheet: kw["sheet_name"] = sheet
    if header_row is not None: kw["header"] = header_row
    df = pd.read_excel(src, engine="openpyxl", **kw)
    # if merged headers or unnamed, try to flatten one level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in t if str(x)!="nan"]).strip() for t in df.columns.to_list()]
    return df

def _read_any(src: Path, sheet: Optional[str], header_row: Optional[int]) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    ext = src.suffix.lower()
    if ext in {".xlsx",".xls"}:
        return _read_excel(src, sheet, header_row)
    if ext == ".csv":
        kw = {}
        if header_row is not None: kw["header"] = header_row
        return pd.read_csv(src, **kw)
    # try parquet as a last resort
    return pd.read_parquet(src)

def _pick_by_name(cols: List[str], name_set: set[str]) -> Optional[str]:
    low = [c.lower().strip() for c in cols]
    for i, c in enumerate(low):
        if c in name_set: return cols[i]
        # fuzzy contains
        if any(tok in c for tok in name_set):
            return cols[i]
    return None

def _infer_date_col(df: pd.DataFrame) -> Optional[str]:
    best_c = None; best_score = -1.0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        frac = s.notna().mean()
        if frac < 0.60:  # not date-like enough
            continue
        # monotonicity score (allow 5% violations)
        diffs_ok = (s.dropna().diff().dt.days.fillna(1) >= 0).mean()
        score = 0.7*frac + 0.3*diffs_ok
        if score > best_score:
            best_score = score; best_c = c
    return best_c

def _infer_balance_col(df: pd.DataFrame, date_col: str | None) -> Optional[str]:
    candidates = []
    for c in df.columns:
        if c == date_col: continue
        try:
            v = pd.to_numeric(df[c], errors="coerce")
            ratio = v.notna().mean()
            if ratio < 0.60:  # too sparse / non-numeric
                continue
            # prefer big-magnitude, stable series
            mag = np.nanquantile(np.abs(v), 0.9)
            candidates.append((mag, c))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def _auto_find_columns(df: pd.DataFrame,
                       date_col: Optional[str],
                       balance_col: Optional[str]) -> Tuple[str, str, Dict]:
    cols = list(df.columns)
    info = {"columns": cols}

    # 1) exact/fuzzy by name
    dcol = date_col or _pick_by_name(cols, DATE_NAMES)
    bcol = balance_col or _pick_by_name(cols, BAL_NAMES)

    # 2) heuristic fallback
    if not dcol:
        dcol = _infer_date_col(df)
    if not bcol:
        bcol = _infer_balance_col(df, dcol)

    if not dcol or not bcol:
        raise ValueError("Could not auto-detect date/balance columns. Please set PP_DATE_COL and PP_BALANCE_COL.")

    info["picked_date_col"] = dcol
    info["picked_balance_col"] = bcol
    return dcol, bcol, info

# ---------------- Business logic ----------------
def _calendar_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    cal = USFederalHolidayCalendar().holidays(start=idx.min(), end=idx.max())
    f = pd.DataFrame(index=idx)
    f["is_weekend"] = idx.weekday >= 5
    f["is_holiday"] = idx.normalize().isin(cal)
    f["is_business_day"] = (~f["is_weekend"]) & (~f["is_holiday"])
    f["dow"] = idx.dayofweek
    f["month_end"] = idx.is_month_end
    f["quarter_end"] = idx.is_quarter_end
    return f

def _reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    base = df.set_index("date").reindex(idx); base.index.name = "date"
    base["balance"] = base["balance"].ffill()
    flags = _calendar_flags(idx); base = base.join(flags)
    base["net_flow"] = base["balance"].diff().fillna(0.0)
    return base.reset_index()

def _clip_flows_mad(flow: pd.Series, k: float = 6.0) -> tuple[pd.Series, float]:
    med = flow.median(); mad = (flow - med).abs().median()
    thr = (1.4826 * mad * k) if (mad and np.isfinite(mad)) else float(flow.abs().quantile(0.99))
    return flow.clip(lower=-thr, upper=thr), float(thr)

def _weekday_mean(series: pd.Series, weeks: int = 8) -> pd.Series:
    s = series.copy()
    out = []
    for _, sub in s.groupby(s.index.dayofweek):
        out.append(sub.rolling(weeks, min_periods=1).mean())
    return pd.concat(out).sort_index()

# -------------- Variants --------------
def build_raw(df: pd.DataFrame, business_days_zero_flows: bool) -> pd.DataFrame:
    out = _reindex_daily(df)
    if business_days_zero_flows:
        out.loc[~out["is_business_day"], "net_flow"] = 0.0
    return out

def build_clean_conservative(df: pd.DataFrame, business_days_zero_flows: bool):
    out = _reindex_daily(df)
    if business_days_zero_flows:
        out.loc[~out["is_business_day"], "net_flow"] = 0.0
    flows_c, thr = _clip_flows_mad(out["net_flow"].fillna(0.0), k=6.0)
    out["net_flow_conservative"] = flows_c
    bal0 = float(out["balance"].iloc[0])
    out["balance_conservative"] = bal0 + out["net_flow_conservative"].cumsum()
    return out, {"outlier_method":"MAD clipping","mad_k":6.0,"flow_threshold_abs":thr,
                 "business_days_zero_flows":bool(business_days_zero_flows)}

def build_clean_treasury(df: pd.DataFrame, business_days_zero_flows: bool, weekday_weeks: int = 8):
    out = _reindex_daily(df)
    out.loc[~out["is_business_day"], "net_flow"] = 0.0
    tmp = out.set_index("date")
    tmp["weekday_baseline_8w"] = _weekday_mean(tmp["net_flow"], weeks=weekday_weeks)
    return tmp.reset_index(), {"weekday_mean_weeks":int(weekday_weeks),"business_days_zero_flows":True}

# -------------- Orchestrator --------------
@dataclass
class PreprocessConfig:
    input_path: str
    date_col: Optional[str]
    balance_col: Optional[str]
    variant: str
    business_days_zero_flows: bool
    weekday_weeks: int
    out_root: str
    run_outputs_dir: str
    expected_csv: Optional[str] = None
    save_parquet: bool = False
    sheet_name: Optional[str] = None
    header_row: Optional[int] = None

def run_preprocess(cfg: PreprocessConfig) -> Dict:
    _log("Loading input…")
    src = Path(cfg.input_path)
    df_in = _read_any(src, cfg.sheet_name, cfg.header_row)

    # Normalize columns to strings (avoid Int64Index etc.)
    df_in.columns = [str(c).strip() for c in df_in.columns]

    # Detect columns (or use provided)
    dcol, bcol, info = _auto_find_columns(df_in, cfg.date_col, cfg.balance_col)

    # Select & standardize
    df = df_in[[dcol, bcol]].rename(columns={dcol:"date", bcol:"balance"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")

    _log(f"Detected columns: date='{dcol}', balance='{bcol}'")

    variant = cfg.variant.lower().strip()
    notes = {}
    if variant == "raw":
        df2 = build_raw(df, cfg.business_days_zero_flows)
    elif variant == "clean_conservative":
        df2, notes = build_clean_conservative(df, cfg.business_days_zero_flows)
    elif variant == "clean_treasury":
        df2, notes = build_clean_treasury(df, cfg.business_days_zero_flows, cfg.weekday_weeks)
    else:
        raise ValueError("PP_VARIANT must be one of: raw | clean_conservative | clean_treasury")

    out_root = Path(cfg.out_root); out_root.mkdir(parents=True, exist_ok=True)
    sub = out_root / variant; sub.mkdir(parents=True, exist_ok=True)
    base = src.stem
    outfile = sub / f"{base}__{variant}.csv"; df2.to_csv(outfile, index=False)

    parquet_file = None
    if cfg.save_parquet:
        parquet_file = sub / f"{base}__{variant}.parquet"; df2.to_parquet(parquet_file, index=False)

    run_out = Path(cfg.run_outputs_dir); run_out.mkdir(parents=True, exist_ok=True)
    df2.head(200).to_csv(run_out / "preview.csv", index=False)

    profile = {
        "input_file": str(src),
        "sheet_name": cfg.sheet_name,
        "header_row": cfg.header_row,
        "variant": variant,
        "detected": info,
        "rows": int(len(df2)),
        "date_range": [df2["date"].min().isoformat() if "date" in df2.columns else None,
                       df2["date"].max().isoformat() if "date" in df2.columns else None],
        "business_days_zero_flows": cfg.business_days_zero_flows,
        "weekday_weeks": cfg.weekday_weeks if variant == "clean_treasury" else None,
        "notes": notes,
        "output_csv": str(outfile),
        "output_parquet": str(parquet_file) if parquet_file else None,
        "preview_path": str(run_out / "preview.csv")
    }
    (run_out / "preprocess_report.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _log(f"Saved: {outfile}")
    if parquet_file: _log(f"Saved: {parquet_file}")
    return profile
