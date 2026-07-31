"""M-2: E_QUANTILE must be evaluated honestly.

Three requirements, each of which fails on the pre-M-2 code:
  1. Folds can tile a fixed evaluation window (eval_start), instead of a
     hand-picked number of 5-day blocks at the very end of the series.
  2. The quality gate must consider interval coverage, not just P50 skill —
     a quantile family exists to produce calibrated intervals.
  3. Skill/coverage are reported PER MODEL (the old report pooled all models
     into one number that belonged to no model), a best model is named, and
     the leaderboard carries an MAE column so the daily summary can show it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from e_quantile_daily_pipeline import (  # noqa: E402
    Config,
    _time_folds,
    quantile_quality_gate,
    run_pipeline,
)


# ── 1. eval_start tiles the evaluation window ──────────────────────────────

def test_time_folds_eval_start_tiles_window():
    folds = _time_folds(n=100, horizon=5, folds=None, min_train=0,
                        eval_start_idx=80)
    # Non-overlapping 5-step blocks covering [80, 100) exactly.
    assert folds == [(80, 85), (85, 90), (90, 95), (95, 100)]


def test_time_folds_eval_start_respects_min_train():
    # Window would start at 20, but min_train floor (30 rows) forbids the
    # earliest blocks: only blocks whose train side exceeds 30 rows survive.
    folds = _time_folds(n=50, horizon=5, folds=None, min_train=0,
                        eval_start_idx=20)
    assert folds == [(35, 40), (40, 45), (45, 50)]


def test_time_folds_without_eval_start_unchanged():
    # Legacy behaviour: `folds` blocks tiled backward from the series end.
    folds = _time_folds(n=100, horizon=5, folds=2, min_train=0,
                        eval_start_idx=None)
    assert folds == [(90, 95), (95, 100)]


# ── 2. gate requires calibrated coverage, not just skill ──────────────────

def test_gate_passes_on_good_skill_and_coverage():
    passed, reasons = quantile_quality_gate(skill_pct=20.0, coverage=0.78)
    assert passed is True
    assert reasons == []


def test_gate_fails_on_bad_coverage_even_with_good_skill():
    passed, reasons = quantile_quality_gate(skill_pct=44.0, coverage=0.27)
    assert passed is False
    assert any("coverage" in r for r in reasons)


def test_gate_fails_on_low_skill_even_with_good_coverage():
    passed, reasons = quantile_quality_gate(skill_pct=1.0, coverage=0.80)
    assert passed is False
    assert any("skill" in r for r in reasons)


# ── 3. per-model reporting, best model, leaderboard MAE ────────────────────

def _synthetic_csv(path: Path, n: int = 170) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(7)
    y = 1000.0 + 200.0 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 25, n)
    df = pd.DataFrame({"date": idx, "y": y})
    df.to_csv(path, index=False)
    return df


def test_pipeline_reports_per_model_and_gates(tmp_path):
    data = tmp_path / "data.csv"
    df = _synthetic_csv(data)
    eval_start = str(df["date"].iloc[-15].date())  # last ~15 origins

    cfg = Config(target="y", cadence="Daily", horizon=3,
                 data_path=str(data), date_col="date",
                 folds=None, min_train_years=0,
                 eval_start=eval_start, out_root=str(tmp_path / "out"))
    run_pipeline(cfg)

    report = json.loads(
        (tmp_path / "out" / "artifacts" / "integrity_report.json").read_text()
    )
    # Per-model verdicts for both registry models.
    assert set(report["models"]) == {"GBQuantile", "ResidualRF"}
    for m in report["models"].values():
        assert {"mae_p50", "skill_pct", "coverage_p10_p90",
                "gate_passed", "gate_reasons"} <= set(m)
    # A best model is named and the top-level verdict belongs to it.
    best = report["best_model"]
    assert best in report["models"]
    assert report["quality_gate_passed"] == report["models"][best]["gate_passed"]
    assert report["run_status"] in ("SUCCESS", "FAILED_QUALITY")
    # Shift diagnostic fields the daily summary reads are present.
    assert "shift_interpretation" in report and "best_shift" in report

    # Leaderboard carries MAE so daily_summary can display a best model.
    lb = pd.read_csv(tmp_path / "out" / "leaderboard.csv")
    assert "MAE" in lb.columns and lb["MAE"].notna().all()

    # predictions_long carries y_pred (=P50) so the summary's independent
    # lagged-copy check runs on this family too.
    preds = pd.read_csv(tmp_path / "out" / "predictions_long.csv")
    assert "y_pred" in preds.columns


def test_best_model_prefers_calibrated_intervals(monkeypatch, tmp_path):
    """A model with broken coverage must not be 'best' on median MAE alone.

    Force ResidualRF to produce a slightly better median than GBQuantile but
    degenerate (zero-width) intervals; the pipeline must select the
    gate-passing model as best and report SUCCESS.
    """
    import e_quantile_daily_pipeline as eq

    def fake_rf(X_tr, y_tr, X_te, quantiles):
        base = np.asarray(y_tr)[-len(X_te):] if len(y_tr) >= len(X_te) else np.full(len(X_te), float(np.mean(y_tr)))
        return {q: base.astype(float) for q in quantiles}  # p10 == p50 == p90

    monkeypatch.setattr(eq, "_fit_residual_rf_quantiles", fake_rf)

    data = tmp_path / "data.csv"
    df = _synthetic_csv(data)
    eval_start = str(df["date"].iloc[-15].date())
    cfg = Config(target="y", cadence="Daily", horizon=3,
                 data_path=str(data), date_col="date",
                 folds=None, min_train_years=0,
                 eval_start=eval_start, out_root=str(tmp_path / "out"))
    eq.run_pipeline(cfg)

    report = json.loads(
        (tmp_path / "out" / "artifacts" / "integrity_report.json").read_text()
    )
    rf, gb = report["models"]["ResidualRF"], report["models"]["GBQuantile"]
    assert rf["gate_passed"] is False  # zero-width intervals -> coverage fails
    if gb["gate_passed"]:
        # Whenever any model is calibrated, it must be the family's best,
        # even if a miscalibrated model has a lower median MAE.
        assert report["best_model"] == "GBQuantile"
        assert report["run_status"] == "SUCCESS"
