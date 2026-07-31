"""M-5: the shuffled-target sentinel measures SIGNAL, not leakage.

The old sentinel reported a low shuffled/real MAE ratio as
``leakage_warning=true``.  That reading is inverted.  Leakage lets a model see
the future, which makes its real-target error implausibly SMALL and therefore
makes this ratio LARGE.  A low ratio means shuffling the targets barely hurt —
the features were never predicting the target.

On the real Revenues run this mattered: B_ML was withheld with the reason
"leakage flag raised" at ratio 0.83, when 0.83 (< 1.0) actually says the
shuffled model beat the real one, i.e. no usable signal.  The verdict
"not usable" was right; the stated reason was wrong, and a wrong reason
destroys trust as surely as a wrong number.

Also covered: the sentinel now scores itself on a held-out tail (it used to
evaluate on rows from its own training window), standardises features (raw
treasury magnitudes made Ridge ill-conditioned at rcond ~1e-18), and the
summary gate fails a family for "no signal" and for "persistence-like"
rather than mislabelling either as leakage.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.integrity import (  # noqa: E402
    MIN_SIGNAL_RATIO,
    leakage_sentinel,
    signal_sentinel,
)

SUMMARY = REPO_ROOT / "scripts" / "daily_summary.py"


# ── the sentinel's own behaviour ──────────────────────────────────────────

def _with_signal(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"lag_1": rng.normal(0, 1, n),
                      "lag_7": rng.normal(0, 1, n),
                      "rmean_7": rng.normal(0, 1, n)})
    y = pd.Series(3.0 * X["lag_1"] + 2.0 * X["lag_7"] + rng.normal(0, 0.5, n))
    return X, y


def _without_signal(n=300, seed=1):
    """Features are pure noise, unrelated to the target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = pd.Series(rng.normal(0, 1, n))
    return X, y


def test_signal_detected_when_features_predict_target():
    X, y = _with_signal()
    r = signal_sentinel(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], horizon=5)
    assert r["signal_detected"] is True
    assert r["shuffled_to_normal_ratio"] >= MIN_SIGNAL_RATIO
    assert "signal present" in r["signal_verdict"]


def test_no_signal_when_features_are_noise():
    X, y = _without_signal()
    r = signal_sentinel(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], horizon=5)
    assert r["signal_detected"] is False
    assert r["shuffled_to_normal_ratio"] < MIN_SIGNAL_RATIO
    assert "SIGNAL" in r["signal_verdict"].upper()


def test_sentinel_never_claims_leakage():
    """Whatever the ratio, this check must not assert leakage."""
    for builder in (_with_signal, _without_signal):
        X, y = builder()
        r = signal_sentinel(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], horizon=5)
        assert r["leakage_warning"] is False


def test_ratio_is_scale_invariant():
    """Standardisation means treasury-sized magnitudes give the same verdict.

    Without scaling, Ridge on raw values ~1e8 is ill-conditioned and the two
    error figures (hence the ratio) become unreliable.
    """
    X, y = _with_signal(seed=3)
    small = signal_sentinel(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], horizon=5)
    big = signal_sentinel(X.iloc[:200] * 1e8, y.iloc[:200] * 1e8,
                          X.iloc[200:] * 1e8, y.iloc[200:] * 1e8, horizon=5)
    assert big["signal_detected"] == small["signal_detected"]
    assert np.isclose(big["shuffled_to_normal_ratio"],
                      small["shuffled_to_normal_ratio"], rtol=0.05)


def test_deprecated_name_still_works():
    X, y = _with_signal(seed=4)
    r = leakage_sentinel(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], horizon=5)
    assert r["signal_detected"] is True
    assert r["leakage_warning"] is False


def test_insufficient_data_is_not_measurable_not_a_pass():
    X, y = _with_signal(n=12, seed=5)
    r = signal_sentinel(X.iloc[:8], y.iloc[:8], X.iloc[8:], y.iloc[8:], horizon=5)
    # "Could not measure" must never be recorded as a clean result.
    assert r["signal_detected"] is None
    assert "not measurable" in r["signal_verdict"]


# ── the gate's use of it ──────────────────────────────────────────────────

def _family(out_dir: Path, report: dict, model="Lasso", mae=4.3e7,
            lagged_copy=False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 24
    if lagged_copy:
        # An oscillating target makes the shift test discriminating: a linear
        # ramp correlates equally well at every shift, so it cannot reveal a
        # lagged copy.  y_pred replays the previous actual with a small
        # constant bias, so its correlation peaks at shift +1 (a constant
        # offset does not change correlation) and it cannot beat the lag-1
        # baseline.  Without the bias the first row would be an accidental
        # perfect hit, which is enough to edge past the baseline.
        y_true = [100.0 + 40.0 * np.sin(np.pi * i / 3.0) for i in range(n)]
        y_pred = [y_true[0] + 6.0] + [v + 6.0 for v in y_true[:-1]]
    else:
        y_true = [100.0 + 9.0 * i for i in range(n)]
        y_pred = [v + (3.0 if i % 2 else -3.0) for i, v in enumerate(y_true)]
    pd.DataFrame({
        "model": [model] * n,
        "origin_date": [f"2025-03-{10 + i:02d}" for i in range(n)],
        "target_date": [f"2025-04-{1 + i:02d}" for i in range(n)],
        "y_true": y_true, "y_pred": y_pred,
    }).to_csv(out_dir / "predictions_long.csv", index=False)
    pd.DataFrame([{"model": model, "MAE": mae}]).to_csv(
        out_dir / "leaderboard.csv", index=False)
    (out_dir / "integrity_report.json").write_text(json.dumps(report))


def _run(run_dir: Path, tmp_path: Path, families: str):
    data_file = tmp_path / "master.csv"
    pd.DataFrame({"date": ["2025-08-05", "2025-08-06"],
                  "Revenues": [1.0, 2.0]}).to_csv(data_file, index=False)
    return subprocess.run(
        [sys.executable, str(SUMMARY),
         "--run-dir", str(run_dir), "--data-file", str(data_file),
         "--target", "Revenues", "--cadence", "daily", "--horizon", "5",
         "--run-date", "2026-07-30", "--families", families, "--mode", "backtest"],
        capture_output=True, text=True)


def test_gate_fails_for_no_signal_not_for_leakage(tmp_path):
    """Mirrors the real B_ML case: ratio 0.83, SUCCESS, no leakage evidence."""
    run_dir = tmp_path / "run"
    _family(run_dir / "b_ml", {
        "run_status": "SUCCESS",
        "skill_pct": 28.71,
        "quality_gate_passed": True,
        "leakage_warning": False,
        "shuffled_to_normal_ratio": 0.83,
        "signal_detected": False,
        "signal_verdict": "NO SIGNAL: shuffling the targets improved held-out error",
        "shift_interpretation": "OK: no shift detected",
    })
    proc = _run(run_dir, tmp_path, "B_ML")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()

    assert "WITHHELD" in text
    assert "no signal beyond shuffled targets (ratio 0.83)" in text
    # The reason must no longer be mislabelled as leakage.
    assert "Quality gate: FAILED" in text
    assert "leakage flag raised" not in text


def test_gate_fails_a_persistence_like_forecast(tmp_path):
    """Mirrors the real C_DL case: skill clears the bar, yet it is a lagged copy."""
    run_dir = tmp_path / "run"
    _family(run_dir / "c_dl", {
        "run_status": "SUCCESS",
        "skill_pct": 10.84,
        "quality_gate_passed": True,
        "leakage_warning": False,
    }, model="MLP", mae=4.7e7, lagged_copy=True)
    proc = _run(run_dir, tmp_path, "C_DL")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()

    assert "Shift flag: YES" in text
    assert "persistence-like" in text
    assert "WITHHELD" in text, "a lagged copy must not be offered as usable"


def test_clean_family_still_passes(tmp_path):
    run_dir = tmp_path / "run"
    _family(run_dir / "a_stat", {
        "run_status": "SUCCESS",
        "skill_pct": 27.51,
        "quality_gate_passed": True,
        "leakage_warning": False,
        "signal_detected": True,
        "shift_interpretation": "OK: no shift detected",
    }, model="ETS", mae=4.4e7)
    proc = _run(run_dir, tmp_path, "A_STAT")
    assert proc.returncode == 0, proc.stderr
    text = (run_dir / "SUMMARY.txt").read_text()
    assert "Quality gate: PASSED" in text
    assert "WITHHELD" not in text
