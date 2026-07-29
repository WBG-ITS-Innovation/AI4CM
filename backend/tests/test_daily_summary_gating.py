"""M-1: daily_summary must not present FAILED_QUALITY families as clean winners.

Builds a synthetic run directory with two families:
  * A_STAT  — SUCCESS, no flags      -> gate PASSED, best model shown
  * B_ML    — FAILED_QUALITY         -> gate FAILED, best model WITHHELD
then runs scripts/daily_summary.py as a subprocess (exactly how the shell
runner invokes it) and checks SUMMARY.txt and SUMMARY.json agree.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "daily_summary.py"
RUN_DATE = "2026-07-29"


def _write_family(fam_dir: Path, run_status: str) -> None:
    fam_dir.mkdir(parents=True)
    pd.DataFrame({
        "model": ["m1", "m1"],
        "origin_date": ["2026-07-27", "2026-07-27"],
        "target_date": ["2026-07-28", "2026-07-29"],
        "y_true": [100.0, 110.0],
        "y_pred": [101.0, 108.0],
    }).to_csv(fam_dir / "predictions_long.csv", index=False)
    pd.DataFrame({
        "model": ["m1", "persistence_baseline"],
        "MAE": [1.5, 3.0],
    }).to_csv(fam_dir / "leaderboard.csv", index=False)
    (fam_dir / "integrity_report.json").write_text(json.dumps({
        "run_status": run_status,
        "skill_pct": 12.3,
        "leakage_warning": False,
    }))


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / RUN_DATE
    _write_family(rd / "a_stat", "SUCCESS")
    _write_family(rd / "b_ml", "FAILED_QUALITY")
    # Data file whose latest date equals the run date -> fresh.
    pd.DataFrame({"date": ["2026-07-28", RUN_DATE], "y": [1.0, 2.0]}).to_csv(
        tmp_path / "master.csv", index=False)
    return rd


def _run_summary(run_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--run-dir", str(run_dir),
         "--data-file", str(run_dir.parent / "master.csv"),
         "--target", "State budget balance",
         "--cadence", "daily",
         "--horizon", "5",
         "--run-date", RUN_DATE,
         "--families", "A_STAT B_ML"],
        capture_output=True, text=True)


def test_failed_quality_best_model_is_withheld(run_dir: Path):
    proc = _run_summary(run_dir)
    assert proc.returncode == 0, proc.stderr
    txt = (run_dir / "SUMMARY.txt").read_text()

    a_stat, b_ml = txt.split("[B_ML]")
    assert "Best model: m1 (MAE 2)" in a_stat  # clean family keeps its line
    assert "Quality gate: PASSED" in a_stat
    assert "WITHHELD" in b_ml
    assert "run_status=FAILED_QUALITY" in b_ml
    assert "Quality gate: FAILED" in b_ml
    assert "diagnosis only" in b_ml  # raw number still visible, labelled


def test_summary_json_matches_text_gate(run_dir: Path):
    proc = _run_summary(run_dir)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((run_dir / "SUMMARY.json").read_text())

    fams = {f["name"]: f for f in payload["families"]}
    assert fams["A_STAT"]["gate_passed"] is True
    assert fams["A_STAT"]["gate_reasons"] == []
    assert fams["B_ML"]["gate_passed"] is False
    assert fams["B_ML"]["gate_reasons"] == ["run_status=FAILED_QUALITY"]
    assert "WITHHELD" in fams["B_ML"]["best_model_display"]
    assert payload["overall"]["families_gate_passed"] == 1
    assert payload["freshness"]["stale"] is False


def test_leakage_flag_also_fails_gate(tmp_path: Path):
    rd = tmp_path / RUN_DATE
    _write_family(rd / "a_stat", "SUCCESS")
    # Overwrite integrity report: status SUCCESS but leakage_warning true.
    (rd / "a_stat" / "integrity_report.json").write_text(json.dumps({
        "run_status": "SUCCESS", "skill_pct": 9.9,
        "leakage_warning": True, "shuffled_to_normal_ratio": 1.02,
    }))
    pd.DataFrame({"date": [RUN_DATE], "y": [1.0]}).to_csv(
        tmp_path / "master.csv", index=False)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(rd),
         "--data-file", str(tmp_path / "master.csv"),
         "--target", "t", "--cadence", "daily", "--horizon", "5",
         "--run-date", RUN_DATE, "--families", "A_STAT"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((rd / "SUMMARY.json").read_text())
    fam = payload["families"][0]
    assert fam["gate_passed"] is False
    assert fam["gate_reasons"] == ["leakage flag raised"]
