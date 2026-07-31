#!/usr/bin/env python3
"""
Build a cross-family backtest report from a completed forecast run.

Reads the machine-readable SUMMARY.json written by daily_summary.py plus each
family's integrity report, and emits BACKTEST_REPORT.md: one table comparing
every family on the same window, with per-model detail where a family reports
it (E_QUANTILE), and the quality-gate verdict shown alongside every number.

Design rule: a withheld or gate-failing family still appears in the table,
marked as not usable.  A report that quietly drops its failures is how this
project lost the client's trust the first time; the point of the backtest is
to show what holds up *and* what does not.

Usage:
    python scripts/backtest_report.py --run-dir backend/forecast_runs/2026-07-29
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from evaluation_windows import describe_split  # noqa: E402


def _find_integrity(family_dir: Path) -> dict:
    """Locate a family's integrity report (same fallback as daily_summary)."""
    if not family_dir.exists():
        return {}
    direct = family_dir / "integrity_report.json"
    candidates = [direct] if direct.exists() else sorted(family_dir.rglob("integrity_*.json"))
    if not candidates:
        candidates = sorted(family_dir.rglob("integrity_report.json"))
    for c in candidates:
        try:
            return json.loads(c.read_text())
        except Exception:
            continue
    return {}


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_cov(x) -> str:
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return "—"


def _fmt_money(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _verdict(fam: dict) -> str:
    if fam.get("gate_passed") is True:
        return "USABLE"
    if fam.get("gate_passed") is False:
        reasons = "; ".join(fam.get("gate_reasons") or []) or "quality gate failed"
        return f"NOT USABLE ({reasons})"
    return "UNVERIFIED (no integrity report)"


def build_report(run_dir: Path) -> str:
    payload = json.loads((run_dir / "SUMMARY.json").read_text())
    families = payload.get("families", [])
    overall = payload.get("overall", {})
    fresh = payload.get("freshness", {})

    lines: list[str] = []
    lines.append("# AI4CM Backtest Report")
    lines.append("")
    lines.append(f"**Run date:** {payload.get('run_date', 'n/a')}  ")
    lines.append(f"**Target:** {payload.get('target', 'n/a')} · "
                 f"**Cadence:** {payload.get('cadence', 'n/a')} · "
                 f"**Horizon:** {payload.get('horizon', 'n/a')} business days")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "This is a **historical backtest**, not a live forecast. Models are fitted "
        "only on data before each forecast origin and scored on days they never saw, "
        "so the errors below are out-of-sample."
    )
    lines.append("")
    lines.append(f"**Data split:** {describe_split()}")
    lines.append("")
    lines.append(f"_{fresh.get('line', '')}_")
    lines.append("")

    # ── cross-family table ────────────────────────────────────────────────
    lines.append("## Family comparison")
    lines.append("")
    lines.append("| Family | Models run | Best model | MAE | Skill vs persistence | Interval coverage | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for fam in families:
        name = fam.get("name", "?")
        report = _find_integrity(run_dir / name.lower())
        n_models = len([m for m in str(fam.get("models", "")).split(",") if m.strip()])
        best = fam.get("best_model", "n/a")
        if fam.get("gate_passed") is False:
            best = f"~~{best}~~"
        cov = report.get("coverage_p10_p90")
        lines.append(
            f"| {name} | {n_models} | {best} | "
            f"{_fmt_money(report.get('mae_model') or report.get('mae_p50'))} | "
            f"{_fmt_pct(fam.get('skill_pct', '').rstrip('%') if isinstance(fam.get('skill_pct'), str) else fam.get('skill_pct'))} | "
            f"{_fmt_cov(cov)} | {_verdict(fam)} |"
        )
    lines.append("")

    # ── per-model detail where available ─────────────────────────────────
    detail_written = False
    for fam in families:
        report = _find_integrity(run_dir / str(fam.get("name", "")).lower())
        models = report.get("models")
        if not isinstance(models, dict) or not models:
            continue
        if not detail_written:
            lines.append("## Per-model detail")
            lines.append("")
            detail_written = True
        lines.append(f"### {fam.get('name')}")
        lines.append("")
        lines.append("| Model | n | MAE (P50) | Skill | Coverage (P10–P90) | Gate |")
        lines.append("|---|---|---|---|---|---|")
        for mname, m in sorted(models.items()):
            gate = "PASS" if m.get("gate_passed") else \
                   "FAIL — " + "; ".join(m.get("gate_reasons") or [])
            lines.append(
                f"| {mname} | {m.get('n_predictions', '—')} | "
                f"{_fmt_money(m.get('mae_p50'))} | {_fmt_pct(m.get('skill_pct'))} | "
                f"{_fmt_cov(m.get('coverage_p10_p90'))} | {gate} |"
            )
        lines.append("")

    # ── honest bottom line ───────────────────────────────────────────────
    usable = [f.get("name") for f in families if f.get("gate_passed") is True]
    not_usable = [f.get("name") for f in families if f.get("gate_passed") is False]
    lines.append("## Bottom line")
    lines.append("")
    lines.append(f"- Families producing output: {overall.get('families_ok', '?')}"
                 f"/{overall.get('families_requested', '?')}")
    lines.append(f"- Passing the quality gate: **{', '.join(usable) if usable else 'none'}**")
    if not_usable:
        lines.append(f"- Withheld as not usable: **{', '.join(not_usable)}** "
                     f"(reasons in the table above)")
    lines.append(f"- Flags raised: {overall.get('leakage_flags', 0)} leakage, "
                 f"{overall.get('shift_flags', 0)} shift, "
                 f"{overall.get('quality_gate_failures', 0)} quality")
    lines.append("")
    lines.append("### How to read the numbers")
    lines.append("")
    lines.append(
        "**Skill vs persistence** compares each model against the naive forecast "
        "\"tomorrow's balance equals today's\", measured on one shared baseline for "
        "all families. Positive skill means the model beats that naive rule; daily "
        "cash balances behave close to a random walk, which makes it a demanding "
        "benchmark rather than an easy one."
    )
    lines.append("")
    lines.append(
        "**Interval coverage** applies to families that publish prediction intervals: "
        "the share of actual values that landed inside the P10–P90 band, which should "
        "be near 80%. Coverage far below that means the intervals are too narrow and "
        "would understate risk in liquidity planning."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a cross-family backtest report.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="output path (default: <run-dir>/BACKTEST_REPORT.md)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_json = run_dir / "SUMMARY.json"
    if not summary_json.exists():
        print(f"ERROR: {summary_json} not found — run scripts/run_daily_forecast.sh first.",
              file=sys.stderr)
        return 1

    report = build_report(run_dir)
    out = Path(args.out) if args.out else run_dir / "BACKTEST_REPORT.md"
    out.write_text(report)
    print(report)
    print(f"[backtest] Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
