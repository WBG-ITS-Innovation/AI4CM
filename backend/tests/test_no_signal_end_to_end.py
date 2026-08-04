"""M-5 (end to end): a no-signal run must be withheld, for the no-signal reason.

test_signal_sentinel_semantics.py exercises the summary gate by hand-writing an
``integrity_report.json``.  That proves the gate reads the field correctly but
not that any pipeline ever *writes* it, nor that the value survives the trip.

The chain M-5 actually delivered has four links:

    b_ml_pipeline runs signal_sentinel
      -> writes signal_detected / signal_verdict into integrity_report.json
        -> daily_summary.gate_reasons turns that into a failure reason
          -> SUMMARY.txt withholds the best model and names the accurate reason

The gate-critical field is ``signal_detected``: it is the only one
``daily_summary.gate_reasons`` acts on.  It is also the most fragile, because
the sentinel block sits behind five nested guards (a fold must exist, the
training sample must exceed 20 rows, the test slice 5, the feature frame 10,
and ``train_stop`` must reach 10).  Any of those failing skips the sentinel
silently, leaving no ``signal_detected`` key — and a family with no key is not
gated, so it passes.  "Not measured" quietly becomes "fine".

(For the record, one hazard I expected here is *not* real: the pipeline merges
``compute_integrity_report``'s legacy dict over its own report via
``integrity_report.update(legacy_report)``.  Reordering that merge after the
sentinel was measured to overwrite only ``leakage_warning`` (to False, already
its value) and ``mae_shuffled_target`` (to NaN, display only).
``signal_detected``, ``signal_verdict`` and ``shuffled_to_normal_ratio`` are
absent from the legacy dict and survive any merge, so the gate is not at risk
from that ordering.)

This test drives the whole chain on a target with no forecastable structure, and
asserts the run is withheld **and** that the stated reason is "no signal" rather
than the pre-M-5 "leakage".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from b_ml_pipeline import ConfigBML, run_pipeline_ml  # noqa: E402

SUMMARY = REPO_ROOT / "scripts" / "daily_summary.py"
HORIZON = 5


def _unforecastable_csv(path: Path) -> None:
    """I.i.d. noise: no autocorrelation, no weekly or monthly shape.

    Lags and rolling means of white noise carry no information about its next
    value, so the shuffled-target control should find nothing — while ordinary
    skill-vs-persistence still looks respectable, because predicting the mean
    beats chasing a random walk.  That combination is the whole point: the
    number looks fine and the model is still useless.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2016-01-01", "2023-08-06")
    values = rng.normal(7.0e7, 1.2e7, size=len(dates))
    pd.DataFrame({"date": dates, "Revenues": np.maximum(values, 0.0)}).to_csv(
        path, index=False
    )


@pytest.fixture(scope="module")
def no_signal_run(tmp_path_factory):
    """Run B_ML on unforecastable data, then summarise it the way cron does."""
    run_dir = tmp_path_factory.mktemp("nosignal")
    csv = run_dir / "master.csv"
    _unforecastable_csv(csv)

    run_pipeline_ml(ConfigBML(
        data_path=str(csv), date_col="date", target="Revenues", cadence="Daily",
        horizon=HORIZON, variant="uni", model_filter="Ridge",
        out_root=str(run_dir / "b_ml"), folds=1, min_train_years=4,
    ))

    proc = subprocess.run(
        [sys.executable, str(SUMMARY),
         "--run-dir", str(run_dir), "--data-file", str(csv),
         "--target", "Revenues", "--cadence", "Daily", "--horizon", str(HORIZON),
         "--run-date", "2026-08-04", "--families", "B_ML", "--mode", "backtest"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    return {
        "report": json.loads(
            (run_dir / "b_ml" / "artifacts" / "integrity_report.json").read_text()),
        "text": (run_dir / "SUMMARY.txt").read_text(),
        "json": json.loads((run_dir / "SUMMARY.json").read_text()),
    }


# ── link 1-2: the pipeline writes the sentinel's verdict ──────────────────

def test_pipeline_publishes_the_signal_verdict(no_signal_run):
    r = no_signal_run["report"]
    assert "signal_detected" in r, (
        "the pipeline did not publish signal_detected — either the sentinel was "
        "skipped, or a later dict merge overwrote its fields"
    )
    assert r["signal_detected"] is False, (
        f"expected no signal on i.i.d. noise, got {r['signal_detected']!r} "
        f"(ratio {r.get('shuffled_to_normal_ratio')!r})"
    )
    assert isinstance(r.get("signal_verdict"), str) and r["signal_verdict"]


def test_pipeline_does_not_call_it_leakage(no_signal_run):
    """The pre-M-5 inversion, asserted on a real artifact this time."""
    assert no_signal_run["report"].get("leakage_warning") is False


def test_skill_still_looks_respectable(no_signal_run):
    """Guards the premise: this must be a no-signal case, not a low-skill one.

    If skill were also below the gate the run would be withheld for skill and
    this test would stop exercising the signal path at all.
    """
    skill = no_signal_run["report"].get("skill_pct")
    assert skill is not None and skill >= 5.0, (
        f"fixture no longer exercises the intended path: skill={skill!r} "
        f"already fails the skill gate"
    )


# ── link 3-4: the summary withholds, for the accurate reason ──────────────

def test_summary_withholds_the_best_model(no_signal_run):
    text = no_signal_run["text"]
    assert "WITHHELD" in text, (
        "a model with no measurable signal was offered as usable:\n" + text
    )
    assert "Quality gate: FAILED" in text


def test_summary_reason_is_no_signal_not_leakage(no_signal_run):
    text = no_signal_run["text"]
    assert "no signal beyond shuffled targets" in text, text
    assert "leakage flag raised" not in text, (
        "the no-signal condition was reported as leakage — the M-5 inversion "
        "has returned to the summary layer"
    )


def test_summary_json_agrees_with_the_text(no_signal_run):
    fam = no_signal_run["json"]["families"][0]
    assert fam["gate_passed"] is False
    assert any("no signal" in r.lower() for r in fam["gate_reasons"]), fam
    assert not any("leak" in r.lower() for r in fam["gate_reasons"]), fam
    assert fam["leakage_flag"] is False
