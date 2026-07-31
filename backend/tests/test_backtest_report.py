"""Backtest deliverable: locked split, backtest labelling, cross-family report.

Three concerns:
  1. evaluation_windows is the single source of truth for train/dev/test, and
     asking it for an eval_start forces you to say whether you are tuning
     (dev) or reporting (locked test).
  2. daily_summary --mode backtest labels a historical run instead of warning
     that the data is stale: a backtest is *supposed* to end in the past.
  3. backtest_report.py produces a client-facing comparison that still shows
     the families which failed the quality gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from evaluation_windows import (  # noqa: E402
    DEV_START,
    TEST_START,
    describe_split,
    eval_start_for,
    window_for,
)

SUMMARY = REPO_ROOT / "scripts" / "daily_summary.py"
REPORT = REPO_ROOT / "scripts" / "backtest_report.py"


# ── 1. windows ────────────────────────────────────────────────────────────

def test_window_boundaries():
    assert window_for("2023-06-15") == "train"
    assert window_for("2024-01-01") == "dev"
    assert window_for("2024-12-31") == "dev"
    assert window_for("2025-01-01") == "test"
    assert window_for("2025-08-06") == "test"


def test_eval_start_distinguishes_tuning_from_reporting():
    assert eval_start_for("tuning") == DEV_START
    assert eval_start_for("report") == TEST_START


def test_eval_start_rejects_ambiguous_purpose():
    # Being forced to declare intent is what keeps the holdout clean.
    with pytest.raises(ValueError):
        eval_start_for("whatever")


def test_describe_split_mentions_all_three_windows():
    text = describe_split("2025-08-06")
    assert "Train" in text and "Dev" in text and "Test" in text
    assert "2025-08-06" in text


# ── shared fixture: a small completed run directory ───────────────────────

def _write_family(out_dir: Path, model: str, mae: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"model": model,
             "origin_date": f"2025-03-{10 + i:02d}",
             "target_date": f"2025-03-{15 + i:02d}",
             "y_true": 100.0 + 9.0 * i,
             "y_pred": 100.0 + 9.0 * i + (3.0 if i % 2 else -3.0)}
            for i in range(6)]
    pd.DataFrame(rows).to_csv(out_dir / "predictions_long.csv", index=False)
    pd.DataFrame([{"model": model, "MAE": mae}]).to_csv(
        out_dir / "leaderboard.csv", index=False)


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"

    a = run_dir / "a_stat"
    _write_family(a, "ETS", 4.4e7)
    (a / "integrity_report.json").write_text(json.dumps({
        "skill_pct": 27.51, "run_status": "SUCCESS", "quality_gate_passed": True,
        "mae_model": 44199748.0, "leakage_warning": False,
        "shift_interpretation": "OK: no shift",
    }))

    # Quantile family with per-model detail and one failing model.
    e = run_dir / "e_quantile"
    _write_family(e, "GBQuantile", 3.5e7)
    (e / "integrity_report.json").write_text(json.dumps({
        "skill_pct": 46.39, "run_status": "SUCCESS", "quality_gate_passed": True,
        "mae_p50": 35381573.0, "coverage_p10_p90": 0.781,
        "best_model": "GBQuantile",
        "models": {
            "GBQuantile": {"n_predictions": 155, "mae_p50": 35381573.0,
                           "skill_pct": 46.39, "coverage_p10_p90": 0.781,
                           "gate_passed": True, "gate_reasons": []},
            "ResidualRF": {"n_predictions": 155, "mae_p50": 36538055.0,
                           "skill_pct": 44.64, "coverage_p10_p90": 0.265,
                           "gate_passed": False,
                           "gate_reasons": ["coverage 26.5% outside [70%, 90%]"]},
        },
    }))

    # A family that must be shown as withheld, not dropped.
    b = run_dir / "b_ml"
    _write_family(b, "Lasso", 4.3e7)
    (b / "integrity_report.json").write_text(json.dumps({
        "skill_pct": 28.71, "run_status": "SUCCESS", "quality_gate_passed": True,
        "mae_model": 43472942.0, "leakage_warning": True,
        "shuffled_to_normal_ratio": 0.83,
    }))

    data_file = tmp_path / "master.csv"
    pd.DataFrame({"date": ["2025-08-05", "2025-08-06"],
                  "Revenues": [1.0, 2.0]}).to_csv(data_file, index=False)
    return run_dir, data_file


def _run_summary(run_dir: Path, data_file: Path, mode: str):
    return subprocess.run(
        [sys.executable, str(SUMMARY),
         "--run-dir", str(run_dir), "--data-file", str(data_file),
         "--target", "Revenues", "--cadence", "daily", "--horizon", "5",
         "--run-date", "2026-07-30", "--families", "A_STAT B_ML E_QUANTILE",
         "--mode", mode],
        capture_output=True, text=True)


# ── 2. backtest labelling ────────────────────────────────────────────────

def test_backtest_mode_labels_instead_of_warning_stale(tmp_path):
    run_dir, data_file = _make_run(tmp_path)
    proc = _run_summary(run_dir, data_file, "backtest")
    assert proc.returncode == 0, proc.stderr

    text = (run_dir / "SUMMARY.txt").read_text()
    assert "MODE: BACKTEST" in text
    assert "out-of-sample" in text
    assert "appears STALE" not in text          # not a production failure
    assert "data backtest window" in text

    payload = json.loads((run_dir / "SUMMARY.json").read_text())
    assert payload["mode"] == "backtest"
    assert payload["freshness"]["backtest"] is True
    # The underlying freshness fact is still recorded, just not alarmed on.
    assert payload["freshness"]["stale"] is True


def test_production_mode_still_warns(tmp_path):
    run_dir, data_file = _make_run(tmp_path)
    proc = _run_summary(run_dir, data_file, "production")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()
    assert "appears STALE" in text
    assert "MODE: BACKTEST" not in text


# ── 3. cross-family report ───────────────────────────────────────────────

def test_report_shows_usable_and_withheld_families(tmp_path):
    run_dir, data_file = _make_run(tmp_path)
    assert _run_summary(run_dir, data_file, "backtest").returncode == 0

    proc = subprocess.run([sys.executable, str(REPORT), "--run-dir", str(run_dir)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    md = (run_dir / "BACKTEST_REPORT.md").read_text()

    # Framed as a backtest, with the split declared.
    assert "historical backtest" in md
    assert "locked holdout" in md

    # Every family appears — including the one that failed the gate.
    for fam in ("A_STAT", "B_ML", "E_QUANTILE"):
        assert fam in md
    assert "NOT USABLE" in md            # B_ML withheld on its leakage flag
    assert "USABLE" in md

    # Per-model detail surfaces the miscalibrated model rather than hiding it.
    assert "ResidualRF" in md and "26.5%" in md
    assert "GBQuantile" in md and "78.1%" in md

    # Reader guidance is present so numbers aren't misread.
    assert "Skill vs persistence" in md
    assert "Interval coverage" in md


def test_report_fails_loudly_without_summary_json(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run([sys.executable, str(REPORT), "--run-dir", str(empty)],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert "SUMMARY.json" in proc.stderr
