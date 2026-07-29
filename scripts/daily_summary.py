#!/usr/bin/env python3
"""
Write a short plain-text summary of a daily forecast run.

Called by scripts/run_daily_forecast.sh after every pipeline family has run.
For each family it reports the models that ran, the best model's error, the
skill vs. the persistence baseline, and any leakage or shift flags.

Two sources of flags are combined, and both are shown:
  1. Warnings the *pipeline itself* recorded in artifacts/integrity_report.json
     (leakage_warning, shift_interpretation, and the "model ≈ naive baseline"
     guard).  These are surfaced verbatim — not re-thresholded here.
  2. Two independent checks this summary computes from the predictions
     (origin_date >= target_date, and detect_lagged_copy).

Every field is always printed; where a family did not produce a value
(e.g. C_DL quick mode writes no integrity report), the line reads
"n/a (not produced)" rather than being silently dropped.

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

NA = "n/a (not produced)"


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


def _find_integrity_report(family_dir: Path) -> Path | None:
    """Locate a family's integrity report.

    Most families write `integrity_report.json`.  The DL pipeline instead
    writes `integrity_<Target>_h<H>.json` (e.g. integrity_Revenues_h5.json),
    nested under daily/artifacts/.  Try the standard name first (unchanged
    behaviour for the other families), then fall back to any integrity_*.json,
    sorted so the choice is deterministic when several match.
    """
    standard = _find_one(family_dir, "integrity_report.json")
    if standard is not None:
        return standard
    matches = sorted(family_dir.rglob("integrity_*.json"))
    return matches[0] if matches else None


def gate_reasons(report: dict, leakage_flag: bool) -> list[str]:
    """Reasons a family fails the quality gate (empty list = no failure).

    A family fails when the pipeline recorded run_status=FAILED_QUALITY (or
    quality_gate_passed=false in older artifacts), or when any leakage flag
    was raised — a leaky model is not usable no matter how good it looks.
    """
    reasons: list[str] = []
    if report:
        status = str(report.get("run_status", "")).strip().upper()
        if status == "FAILED_QUALITY":
            reasons.append("run_status=FAILED_QUALITY")
        elif report.get("quality_gate_passed") is False:
            reasons.append("quality_gate_passed=false")
    if leakage_flag:
        reasons.append("leakage flag raised")
    return reasons


def _fmt_money(x) -> str:
    """Format a large number with thousands separators (or 'n/a')."""
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def pipeline_leakage(report: dict) -> tuple[str, bool]:
    """Leakage warning the pipeline itself recorded (surfaced verbatim).

    Returns (text, is_flag).  If the field was never produced, text is the
    "n/a" marker and is_flag is False.
    """
    if "leakage_warning" not in report:
        return (NA, False)
    if report.get("leakage_warning") is True:
        ratio = report.get("shuffled_to_normal_ratio")
        ratio_s = f"{float(ratio):.2f}" if _is_number(ratio) else "n/a"
        return (f"leakage_warning=true (shuffled_to_normal_ratio={ratio_s})", True)
    return ("none", False)


def pipeline_shift(report: dict) -> tuple[list[str], bool]:
    """Shift warnings the pipeline itself recorded (surfaced verbatim).

    Returns (lines, is_flag).  Combines the pipeline's own shift_interpretation
    string with its "model ≈ naive baseline" guard.  The guard reproduces the
    pipeline's exact comparison (b_ml_pipeline.py GUARD B: model MAE within 10%
    of the shift=-h MAE) using the numbers the pipeline stored, so the numbers
    shown are the pipeline's own.
    """
    has_fields = ("shift_interpretation" in report) or ("best_shift" in report)
    if not has_fields:
        return ([NA], False)

    lines: list[str] = []

    interp = report.get("shift_interpretation")
    if isinstance(interp, str) and not interp.strip().upper().startswith("OK"):
        lines.append(interp)  # verbatim

    if report.get("is_critical_timestamping_bug") is True:
        lines.append("is_critical_timestamping_bug=true")

    # GUARD B: model performance ≈ naive (shift=-h) baseline.
    mae_model = report.get("mae_model")
    naive_mae = report.get("mae_shift_minus_h")
    if _is_number(mae_model) and _is_number(naive_mae):
        denom = max(mae_model, naive_mae, 1.0)
        if abs(mae_model - naive_mae) / denom < 0.1:
            lines.append(
                f"model performance ≈ naive baseline (shift=-h): "
                f"Model MAE={_fmt_money(mae_model)} vs Naive MAE={_fmt_money(naive_mae)}"
            )

    if not lines:
        return (["none"], False)
    return (lines, True)


def summary_leakage_check(preds: pd.DataFrame) -> tuple[str, bool]:
    """This summary's own leakage check: any prediction that peeks ahead."""
    if not {"origin_date", "target_date"}.issubset(preds.columns):
        return (NA, False)
    origin = pd.to_datetime(preds["origin_date"], errors="coerce")
    target = pd.to_datetime(preds["target_date"], errors="coerce")
    violations = int((origin >= target).sum())
    if violations > 0:
        return (f"{violations} row(s) with origin_date >= target_date", True)
    return ("none", False)


def summary_shift_check(preds: pd.DataFrame) -> tuple[str, bool]:
    """This summary's own shift check via detect_lagged_copy."""
    if not {"y_true", "y_pred"}.issubset(preds.columns):
        return (NA, False)
    result = detect_lagged_copy(preds)
    flagged = [m["model"] for m in result.get("per_model", []) if m.get("flagged")]
    if flagged:
        return ("; ".join(result.get("details", [])), True)
    return ("none", False)


def summarize_family(name: str, family_dir: Path) -> dict:
    """Collect the summary facts for one pipeline family."""
    info: dict = {
        "name": name,
        "ok": False,
        "models": NA,
        "best_model": NA,
        "skill_pct": NA,
        "run_status": NA,
        "quality": NA,
        "gate_passed": None, "gate_reasons": [],
        "integrity_found": False,
        "pipe_leak": NA, "pipe_leak_flag": False,
        "pipe_shift": [NA], "pipe_shift_flag": False,
        "chk_leak": NA, "chk_leak_flag": False,
        "chk_shift": NA, "chk_shift_flag": False,
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
            info["models"] = ", ".join(str(m) for m in lb["model"].tolist())
        if "model" in lb.columns and "MAE" in lb.columns:
            real = lb[~lb["model"].astype(str).str.contains("baseline", case=False, na=False)]
            real = real.dropna(subset=["MAE"])
            if not real.empty:
                best = real.loc[real["MAE"].idxmin()]
                info["best_model"] = f"{best['model']} (MAE {_fmt_money(best['MAE'])})"
    elif "model" in preds.columns:
        info["models"] = ", ".join(str(m) for m in preds["model"].unique().tolist())

    # ── Skill vs persistence + run status, from the integrity report ──
    report = _read_json(_find_integrity_report(family_dir))
    info["integrity_found"] = bool(report)
    if _is_number(report.get("skill_pct")):
        info["skill_pct"] = f"{float(report['skill_pct']):.2f}%"
    if report.get("run_status"):
        info["run_status"] = str(report["run_status"])

    # ── Flags: pipeline-recorded (verbatim) and this summary's own checks ──
    info["pipe_leak"], info["pipe_leak_flag"] = pipeline_leakage(report)
    info["pipe_shift"], info["pipe_shift_flag"] = pipeline_shift(report)
    info["chk_leak"], info["chk_leak_flag"] = summary_leakage_check(preds)
    info["chk_shift"], info["chk_shift_flag"] = summary_shift_check(preds)

    # ── Quality gate: computed last, because leakage flags feed into it ──
    info["gate_reasons"] = gate_reasons(report, family_leakage_flag(info))
    if info["gate_reasons"]:
        info["gate_passed"] = False
    elif info["integrity_found"]:
        info["gate_passed"] = True
    else:
        info["gate_passed"] = None  # never verified — must not look like a pass

    return info


def family_leakage_flag(s: dict) -> bool:
    return bool(s["pipe_leak_flag"] or s["chk_leak_flag"])


def family_shift_flag(s: dict) -> bool:
    return bool(s["pipe_shift_flag"] or s["chk_shift_flag"])


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

    fresh_line, is_stale = check_freshness(data_file, args.date_col, args.run_date, args.stale_days)
    summaries = [summarize_family(fam, run_dir / fam.lower()) for fam in families]

    n_ok = sum(1 for s in summaries if s["ok"])
    n_leak = sum(1 for s in summaries if family_leakage_flag(s))
    n_shift = sum(1 for s in summaries if family_shift_flag(s))
    n_quality = sum(1 for s in summaries if s["gate_passed"] is False)

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
        lines.append(f"  Models run: {s['models']}")
        if s["gate_passed"] is False:
            reasons = "; ".join(s["gate_reasons"])
            best_display = (f"WITHHELD — {reasons}, not usable; "
                            f"{s['best_model']} for diagnosis only")
        elif s["gate_passed"] is None:
            best_display = f"{s['best_model']} (integrity not verified)"
        else:
            best_display = s["best_model"]
        s["best_model_display"] = best_display
        lines.append(f"  Best model: {best_display}")
        lines.append(f"  Skill vs persistence: {s['skill_pct']}")
        lines.append(f"  Run status: {s['run_status']}")
        if s["gate_passed"] is True:
            lines.append("  Quality gate: PASSED")
        elif s["gate_passed"] is False:
            lines.append(f"  Quality gate: FAILED ({'; '.join(s['gate_reasons'])})")
        else:
            lines.append(f"  Quality gate: {NA}")

        leak_flag = "YES" if family_leakage_flag(s) else "none"
        lines.append(f"  Leakage flag: {leak_flag}")
        lines.append(f"    - pipeline: {s['pipe_leak']}")
        lines.append(f"    - summary check (origin_date >= target_date): {s['chk_leak']}")

        shift_flag = "YES" if family_shift_flag(s) else "none"
        lines.append(f"  Shift flag: {shift_flag}")
        for i, w in enumerate(s["pipe_shift"]):
            label = "pipeline" if i == 0 else "pipeline (cont.)"
            lines.append(f"    - {label}: {w}")
        lines.append(f"    - summary check (detect_lagged_copy): {s['chk_shift']}")
        lines.append("")

    lines.append("-" * 40)
    lines.append(f"Overall: {n_ok}/{len(families)} families produced output.")
    lines.append(f"Flags raised: {n_leak} leakage, {n_shift} shift, "
                 f"{n_quality} quality, data {'STALE' if is_stale else 'fresh'}.")

    report = "\n".join(lines) + "\n"
    (run_dir / "SUMMARY.txt").write_text(report)

    # Machine-readable twin of the text report, for downstream tooling.
    payload = {
        "run_date": args.run_date,
        "target": args.target,
        "cadence": args.cadence,
        "horizon": args.horizon,
        "families": [
            {
                "name": s["name"],
                "ok": s["ok"],
                "models": s["models"],
                "best_model": s["best_model"],
                "best_model_display": s.get("best_model_display", s["best_model"]),
                "skill_pct": s["skill_pct"],
                "run_status": s["run_status"],
                "integrity_verified": s["integrity_found"],
                "gate_passed": s["gate_passed"],
                "gate_reasons": s["gate_reasons"],
                "leakage_flag": family_leakage_flag(s),
                "shift_flag": family_shift_flag(s),
            }
            for s in summaries
        ],
        "overall": {
            "families_requested": len(families),
            "families_ok": n_ok,
            "families_gate_passed": sum(
                1 for s in summaries if s["gate_passed"] is True
            ),
            "leakage_flags": n_leak,
            "shift_flags": n_shift,
            "quality_gate_failures": n_quality,
        },
        "freshness": {"line": fresh_line, "stale": bool(is_stale)},
    }
    (run_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2))

    print(report)

    if n_ok < len(families):
        print("ERROR: one or more requested families produced no output.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())