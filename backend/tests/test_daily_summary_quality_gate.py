"""M-1: the daily summary must never present a FAILED_QUALITY family as a
clean "best model", and must say so when integrity was never verified.

These tests run scripts/daily_summary.py as a subprocess (the same way
scripts/run_daily_forecast.sh invokes it) against a synthetic run directory
that replicates the real C_DL layout: the integrity report is named
integrity_<Target>_h<H>.json and nested under daily/artifacts/, exactly like
backend/forecast_runs/<date>/c_dl/daily/artifacts/integrity_Revenues_h5.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "daily_summary.py"

# Field-for-field replica of a real C_DL integrity report that failed the
# quality gate (skill is strongly negative: worse than persistence).
CDL_FAILED_REPORT = {
    "pipeline": "DL",
    "target": "Revenues",
    "horizon": 5,
    "mae_model": 73425359.70126116,
    "mae_persistence": 52957744.22403093,
    "skill_pct": -38.64895640313647,
    "best_shift": -5,
    "is_lag0_issue": False,
    "is_persistence_like": True,
    "alignment_ok": True,
    "mask_target_at_origin": False,
    "quality_gate_passed": False,
    "run_status": "FAILED_QUALITY",
}


def _write_family_outputs(out_dir: Path, model: str, mae: float) -> None:
    """Write a minimal predictions_long.csv + leaderboard.csv into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(6):
        rows.append(
            {
                "model": model,
                "origin_date": f"2026-07-{10 + i:02d}",
                "target_date": f"2026-07-{15 + i:02d}",
                "y_true": 100.0 + 9.0 * i,
                # Correlates with y_true at shift 0 (honest), small alternating
                # error so it is not a lagged copy and not exactly persistence.
                "y_pred": 100.0 + 9.0 * i + (3.0 if i % 2 else -3.0),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "predictions_long.csv", index=False)
    pd.DataFrame([{"model": model, "MAE": mae}]).to_csv(
        out_dir / "leaderboard.csv", index=False
    )


def _run_summary(run_dir: Path, data_file: Path, families: str):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir", str(run_dir),
            "--data-file", str(data_file),
            "--target", "Revenues",
            "--cadence", "daily",
            "--horizon", "5",
            "--run-date", "2026-07-21",
            "--families", families,
        ],
        capture_output=True,
        text=True,
    )


def _write_data_file(path: Path) -> None:
    pd.DataFrame(
        {"date": ["2026-07-20", "2026-07-21"], "Revenues": [1.0, 2.0]}
    ).to_csv(path, index=False)


def test_failed_quality_family_is_marked_not_usable(tmp_path):
    run_dir = tmp_path / "run"

    # A_STAT: clean family, integrity_report.json directly in its folder.
    a_dir = run_dir / "a_stat"
    _write_family_outputs(a_dir, "SARIMAX", 5.0e7)
    (a_dir / "integrity_report.json").write_text(
        json.dumps(
            {
                "skill_pct": 27.5,
                "run_status": "OK",
                "quality_gate_passed": True,
                "leakage_warning": False,
                "shift_interpretation": "OK: no shift detected",
            }
        )
    )

    # C_DL: real layout — outputs under daily/, integrity report named
    # integrity_Revenues_h5.json nested under daily/artifacts/.
    c_out = run_dir / "c_dl" / "daily"
    _write_family_outputs(c_out, "GRU", 7.3e7)
    artifacts = c_out / "artifacts"
    artifacts.mkdir()
    (artifacts / "integrity_Revenues_h5.json").write_text(
        json.dumps(CDL_FAILED_REPORT)
    )

    data_file = tmp_path / "data.csv"
    _write_data_file(data_file)

    proc = _run_summary(run_dir, data_file, "A_STAT C_DL")

    # Quality failure is a *result*, not a crash: exit code stays 0.
    assert proc.returncode == 0, proc.stderr

    summary = (run_dir / "SUMMARY.txt").read_text()
    a_section, c_section = summary.split("[C_DL]")

    # The failed family is visibly unusable...
    assert "FAILED_QUALITY, not usable" in c_section
    # ...and its nested integrity report was actually found (skill populated).
    assert "Skill vs persistence: -38.65%" in c_section
    assert "Run status: FAILED_QUALITY" in c_section

    # The clean family's best-model line stays clean.
    a_best = next(
        line for line in a_section.splitlines()
        if line.strip().startswith("Best model:")
    )
    assert "FAILED_QUALITY" not in a_best
    assert "not verified" not in a_best

    # The overall footer counts the quality failure.
    assert "1 quality" in summary


def test_missing_integrity_report_is_annotated_not_silent(tmp_path):
    """'We did not check' must never look identical to 'we checked and it passed'."""
    run_dir = tmp_path / "run"
    _write_family_outputs(run_dir / "b_ml", "LightGBM", 6.0e7)  # no integrity file

    data_file = tmp_path / "data.csv"
    _write_data_file(data_file)

    proc = _run_summary(run_dir, data_file, "B_ML")

    assert proc.returncode == 0, proc.stderr
    summary = (run_dir / "SUMMARY.txt").read_text()

    best = next(
        line for line in summary.splitlines()
        if line.strip().startswith("Best model:")
    )
    assert "(integrity not verified)" in best
    assert "0 quality" in summary
