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
    """The min_train floor still wins over eval_start, but the grid changed.

    Item 1f made pinned folds tile FORWARD from eval_start, so that the evaluation
    window begins exactly there. Previously they tiled backward from the end of the
    series, which left a remainder at the START of the window whenever its length was
    not a multiple of the horizon -- with 156 target dates and h=5 the family
    evaluated 150 of them, beginning 2025-01-09 rather than 2025-01-01, i.e. a
    different window from every other family.

    Here eval_start=20 is below the 30-row training floor, so the window cannot start
    where asked. Forward tiling begins at the earliest LEGAL index (31) instead of the
    first end-anchored grid point above the floor (35), covering 19 rows of the window
    rather than 15. Same floor, strictly more evaluation data.
    """
    folds = _time_folds(n=50, horizon=5, folds=None, min_train=0,
                        eval_start_idx=20)
    assert folds == [(31, 36), (36, 41), (41, 46), (46, 50)]
    assert folds[0][0] > 30, "the min_train floor must still be respected"


def test_pinned_folds_cover_the_whole_window_even_when_not_a_multiple_of_horizon():
    """The defect item 1f's one-ruler check caught.

    A 156-row window at h=5 is 31 whole blocks plus a remainder. Backward tiling
    dropped the remainder from the front; forward tiling keeps it as a shorter final
    block. A partial block is a smaller sample, not a wrong one -- and it is what makes
    E_QUANTILE's window identical to the other three families'.
    """
    n, h, window = 2763, 5, 156
    folds = _time_folds(n=n, horizon=h, folds=None, min_train=4,
                        eval_start_idx=n - window)
    assert folds[0][0] == n - window, (
        f"first block starts at {folds[0][0]}, not at eval_start {n - window}"
    )
    assert folds[-1][1] == n, "the window is not covered to its end"
    assert folds[-1][1] - folds[0][0] == window, (
        f"covered {folds[-1][1] - folds[0][0]} rows, expected {window}"
    )
    # Blocks are contiguous and non-overlapping.
    for a, b in zip(folds, folds[1:]):
        assert a[1] == b[0], f"gap or overlap between {a} and {b}"


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


# ── 4. the evaluation window needs an UPPER bound, not just a lower one ─────

def test_eval_end_caps_the_window():
    """Regression: `eval_start` alone only set a floor, so folds ran to the series end.

    Item 1f made pinned folds tile forward from `eval_start`. Correct for the 2025
    benchmark, where the window happens to end at the series end -- but it left the
    upper edge unbounded. Pinning to DEV_START then tiled straight through DEV and on
    into the 2025 holdout: 418 target dates where DEV has 262. Any "DEV" figure produced
    that way silently included TEST, which is exactly what the sealed-holdout rule
    forbids.

    `eval_end` is the cap. With it, a DEV-scoped run stops at DEV's last row.
    """
    unbounded = _time_folds(n=100, horizon=5, folds=None, min_train=0,
                            eval_start_idx=60)
    assert unbounded[-1][1] == 100, "without a cap, tiling runs to the series end"

    capped = _time_folds(n=100, horizon=5, folds=None, min_train=0,
                         eval_start_idx=60, eval_end_idx=79)
    assert capped == [(60, 65), (65, 70), (70, 75), (75, 80)]
    assert capped[-1][1] <= 80, "rows beyond eval_end must not be evaluated"


def test_eval_end_keeps_a_partial_final_block():
    """A cap that is not a whole number of horizons must truncate, not overshoot.

    Overshooting by even one block is a holdout read; dropping the remainder would
    quietly shrink the window. Truncate.
    """
    folds = _time_folds(n=200, horizon=5, folds=None, min_train=0,
                        eval_start_idx=100, eval_end_idx=112)
    assert folds[0][0] == 100
    assert folds[-1][1] == 113, f"expected the window to end at 113, got {folds[-1][1]}"
    assert folds[-1][1] - folds[-1][0] == 3, "the final block is the 3-row remainder"


def test_eval_end_below_eval_start_yields_no_folds():
    """A contradictory window must evaluate nothing rather than fall back to the end.

    Silently ignoring an impossible cap is how an unbounded run gets re-introduced.
    """
    folds = _time_folds(n=100, horizon=5, folds=None, min_train=0,
                        eval_start_idx=80, eval_end_idx=60)
    assert folds == []


def test_config_carries_eval_end():
    cfg = Config(target="y", cadence="Daily", horizon=5, data_path="unused.csv",
                 eval_start="2024-01-01", eval_end="2024-12-31")
    assert cfg.eval_end == "2024-12-31"
