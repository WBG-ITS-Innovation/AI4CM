#!/usr/bin/env python3
"""
Write a short plain-text summary of a daily forecast run.

Called by scripts/run_daily_forecast.sh after every pipeline family has run.
For each family it reports the models that ran, the best model's error, the
skill vs. the persistence baseline, and — importantly — any *leakage* or
*shift* flags raised.  It also checks data freshness so a perfect run on a
stale data file cannot pass unnoticed.

The exit code matters: a non-zero exit tells the shell script the run should
be treated as failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The leakage/shift diagnostics live in the backend package.  Add backend/ to
# the import path based on this file's location (scripts/ is a sibling of
# backend/), so the helper works no matter what directory it is called from.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import detect_lagged_copy  # noqa: E402  (import after sys.path edit)


def _find_one(family_dir: Path, filename: str) -> Path | None:
    """Return the first matching file under family_dir, or None.

    We search recursively because the DL pipeline nests its outputs one level
    deeper (in a `daily/` subfolder) than the other families.
    """
    direct = family_dir / filename
    if direct.exists():
        return direct
    matches = sorted(family_dir.rglob(filename))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict:
    """Read a JSON file into a dict, tolerating a missing file."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fmt_money(x: float) -> str:
    """Format a large number with thousands separators (or 'n/a')."""
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def summarize_family(name: str, family_dir: Path) -> dict:
    """Collect the summary facts for one pipeline family.

    Returns a dict with the numbers we want to print plus two booleans:
    leakage_flag and shift_flag.
    """
    info: dict = {
        "name": name,
        "ok": False,
        "models": [],
        "best_model": None,
        "best_mae": None,
        "skill_pct": None,
        "run_status": None,
        "leakage_flag": False,
        "leakage_detail": "",
        "shift_flag": False,
        "shift_detail": "",
        "notes": [],
    }

    pred_path = _find_one(family_dir, "predictions_long.csv")
    if pred_path is None:
        info["notes"].append("no predictions_long.csv found")
        return info

    preds = pd.read_csv(pred_path)
    if preds.empty:
        info["notes"].append("predictions_long.csv is empty")
        return info
    info["ok"] = True

    # ── Models run + best model, from the leaderboard if present ──
    lb_path = _find_one(family_dir, "leaderboard.csv")
    if lb_path is not None:
        lb = pd.read_csv(lb_path)
        if "model" in lb.columns:
            info["models"] = [str(m) for m in lb["model"].tolist()]
        if "model" in lb.columns and "MAE" in lb.columns:
            # Ignore baseline rows when choosing the "best" real model.
            real = lb[~lb["model"].astype(str).str.contains("baseline", case=False, na=False)]
            real = real.dropna(subset=["MAE"])
            if not real.empty:
                best = real.loc[real["MAE"].idxmin()]
                info["best_model"] = str(best["model"])
                info["best_mae"] = float(best["MAE"])
    elif "model" in preds.columns:
        info["models"] = [str(m) for m in preds["model"].unique().tolist()]

    # ── Skill vs persistence + run status, from the integrity report ──
    report = _read_json(_find_one(family_dir, "integrity_report.json"))
    if "skill_pct" in report:
        info["skill_pct"] = report.get("skill_pct")
    if "run_status" in report:
        info["run_status"] = report.get("run_status")

    # ── Leakage flag: any prediction that references the future ──
    if "origin_date" in preds.columns and "target_date" in preds.columns:
        origin = pd.to_datetime(preds["origin_date"], errors="coerce")
        target = pd.to_datetime(preds["target_date"], errors="coerce")
        violations = int((origin >= target).sum())
        if violations > 0:
            info["leakage_flag"] = True
            info["leakage_detail"] = f"{violations} row(s) with origin_date >= target_date"

    # ── Shift flag: models that look like a lagged copy of the target ──
    if {"y_true", "y_pred"}.issubset(preds.columns):
        result = detect_lagged_copy(preds)
        flagged = [m["model"] for m in result.get("per_model", []) if m.get("flagged")]
        if flagged:
            info["shift_flag"] = True
            info["shift_detail"] = "; ".join(result.get("details", []))

    return info


def check_freshness(data_file: Path, date_col: str, run_date: str, stale_days: int) -> tuple[str, bool]:
    """Return (message, is_stale) describing how current the data file is."""
    try:
        dates = pd.read_csv(data_file, usecols=[date_col])[date_col]
        latest = pd.to_datetime(dates, errors="coerce").max()
    except Exception as exc:
        return (f"Could not read latest data date from {data_file.name}: {exc}", True)

    if pd.isna(latest):
        return (f"No valid dates found in {data_file.name}", True)

    run_ts = pd.to_datetime(run_date)
    gap_days = (run_ts.normalize() - latest.normalize()).days
    line = f"Latest data date: {latest.date()} ({gap_days} day(s) before run date {run_ts.date()})"
    return (line, gap_days > stale_days)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a daily forecast run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target", required=True)
    parser.add_argument("--cadence", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--families", required=True, help="space-separated family names")
    parser.add_argument("--stale-days", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_file = Path(args.data_file)
    families = args.families.split()

    # ── Data freshness ──
    fresh_line, is_stale = check_freshness(data_file, args.date_col, args.run_date, args.stale_days)

    # ── Per-family facts ──
    summaries = [summarize_family(fam, run_dir / fam.lower()) for fam in families]

    n_ok = sum(1 for s in summaries if s["ok"])
    n_leak = sum(1 for s in summaries if s["leakage_flag"])
    n_shift = sum(1 for s in summaries if s["shift_flag"])

    # ── Build the report text ──
    lines: list[str] = []
    lines.append("AI4CM Daily Forecast Summary")
    lines.append("=" * 40)
    lines.append(f"Run date:   {args.run_date}")
    lines.append(f"Data file:  {data_file.name}")
    lines.append(fresh_line)
    if is_stale:
        lines.append(f"WARNING: data appears STALE (older than {args.stale_days} day(s)) — "
                     f"forecasts may be based on out-of-date inputs.")
    lines.append(f"Target: {args.target} | Cadence: {args.cadence} | Horizon: {args.horizon}")
    lines.append(f"Families requested: {', '.join(families)}")
    lines.append("")

    for s in summaries:
        lines.append(f"[{s['name']}]")
        if not s["ok"]:
            lines.append(f"  STATUS: no usable output ({'; '.join(s['notes']) or 'unknown'})")
            lines.append("")
            continue
        models = ", ".join(s["models"]) if s["models"] else "(unknown)"
        lines.append(f"  Models run: {models}")
        if s["best_model"] is not None:
            lines.append(f"  Best model: {s['best_model']} (MAE {_fmt_money(s['best_mae'])})")
        if s["skill_pct"] is not None:
            try:
                lines.append(f"  Skill vs persistence: {float(s['skill_pct']):.2f}%")
            except (TypeError, ValueError):
                pass
        if s["run_status"]:
            lines.append(f"  Run status: {s['run_status']}")
        lines.append(f"  Leakage flag: {'YES — ' + s['leakage_detail'] if s['leakage_flag'] else 'none'}")
        lines.append(f"  Shift flag:   {'YES — ' + s['shift_detail'] if s['shift_flag'] else 'none'}")
        lines.append("")

    lines.append("-" * 40)
    lines.append(f"Overall: {n_ok}/{len(families)} families produced output.")
    lines.append(f"Flags raised: {n_leak} leakage, {n_shift} shift, "
                 f"data {'STALE' if is_stale else 'fresh'}.")

    report = "\n".join(lines) + "\n"
    (run_dir / "SUMMARY.txt").write_text(report)
    # Also echo to stdout so the shell script's log captures it.
    print(report)

    # A family that was requested but produced nothing is a real failure.
    if n_ok < len(families):
        print("ERROR: one or more requested families produced no output.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
