"""The reported best model must be the one whose numbers are reported.

Observed on a real run: the family line read
    Best model: ResidualRF (MAE 33,245,860)
    Skill vs persistence: 48.45%   Quality gate: PASSED
while the per-model detail showed ResidualRF FAILING on coverage (69.8%) and
those skill/coverage figures belonging to GBQuantile.  Cause: the summary took
the *name* from leaderboard.csv (lowest MAE) but the *numbers* from the
integrity report, whose best_model is chosen among gate-passing models only.
A report that names one model and describes another is precisely the failure
mode this project exists to eliminate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "scripts" / "daily_summary.py"


def _family(out_dir: Path, models_mae: dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models_mae:
        rows += [{"model": model,
                  "origin_date": f"2025-03-{10 + i:02d}",
                  "target_date": f"2025-03-{15 + i:02d}",
                  "y_true": 100.0 + 9.0 * i,
                  "y_pred": 100.0 + 9.0 * i + (3.0 if i % 2 else -3.0)}
                 for i in range(6)]
    pd.DataFrame(rows).to_csv(out_dir / "predictions_long.csv", index=False)
    pd.DataFrame([{"model": m, "MAE": v} for m, v in models_mae.items()]).to_csv(
        out_dir / "leaderboard.csv", index=False)


def _run(run_dir: Path, data_file: Path, families: str):
    return subprocess.run(
        [sys.executable, str(SUMMARY),
         "--run-dir", str(run_dir), "--data-file", str(data_file),
         "--target", "Revenues", "--cadence", "daily", "--horizon", "5",
         "--run-date", "2026-07-30", "--families", families, "--mode", "backtest"],
        capture_output=True, text=True)


def test_summary_names_the_gate_passing_best_model(tmp_path):
    run_dir = tmp_path / "run"
    eq = run_dir / "e_quantile"
    # ResidualRF has the LOWER MAE but fails coverage; GBQuantile is the
    # family's declared best because it is the only one passing the gate.
    _family(eq, {"GBQuantile": 33_964_365.0, "ResidualRF": 33_245_860.0})
    (eq / "integrity_report.json").write_text(json.dumps({
        "best_model": "GBQuantile",
        "skill_pct": 48.45,
        "coverage_p10_p90": 0.780,
        "mae_p50": 33_964_365.0,
        "quality_gate_passed": True,
        "run_status": "SUCCESS",
        "models": {
            "GBQuantile": {"n_predictions": 205, "mae_p50": 33_964_365.0,
                           "skill_pct": 48.45, "coverage_p10_p90": 0.780,
                           "gate_passed": True, "gate_reasons": []},
            "ResidualRF": {"n_predictions": 205, "mae_p50": 33_245_860.0,
                           "skill_pct": 49.54, "coverage_p10_p90": 0.698,
                           "gate_passed": False,
                           "gate_reasons": ["coverage 69.8% outside [70%, 90%]"]},
        },
    }))
    data_file = tmp_path / "master.csv"
    pd.DataFrame({"date": ["2025-08-05", "2025-08-06"],
                  "Revenues": [1.0, 2.0]}).to_csv(data_file, index=False)

    proc = _run(run_dir, data_file, "E_QUANTILE")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()

    best_line = next(l for l in text.splitlines() if l.strip().startswith("Best model:"))
    assert "GBQuantile" in best_line, best_line
    assert "ResidualRF" not in best_line, best_line
    # The MAE shown must be the named model's, not the rejected model's.
    assert "33,964,365" in best_line
    assert "33,245,860" not in best_line


def test_leaderboard_still_used_when_family_declares_no_best(tmp_path):
    """Families without a best_model field keep the old leaderboard behaviour."""
    run_dir = tmp_path / "run"
    a = run_dir / "a_stat"
    _family(a, {"ETS": 44_199_748.0})
    (a / "integrity_report.json").write_text(json.dumps({
        "skill_pct": 27.51, "quality_gate_passed": True, "run_status": "SUCCESS",
    }))
    data_file = tmp_path / "master.csv"
    pd.DataFrame({"date": ["2025-08-05", "2025-08-06"],
                  "Revenues": [1.0, 2.0]}).to_csv(data_file, index=False)

    proc = _run(run_dir, data_file, "A_STAT")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()
    assert "ETS" in text and "44,199,748" in text
