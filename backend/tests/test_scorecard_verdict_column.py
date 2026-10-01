"""The scorecard's ``publication_verdict`` holds the verdict at issue, not the recipe's status.

Why this file exists
--------------------
``SCORECARD_COLUMNS`` documents ``publication_verdict`` as "the gate verdict in force at issue".
The scorer filled it from the ``status`` field of the issue's ``gates.json``, and both writers
put the recipe's registry *status* there ("candidate -- pre-tuning"), never a verdict. Measured
in the disposable clone on 2026-09-30: every one of twenty scored rows carried the status under
the verdict's name (scoring-loop audit, finding F1). The real store would do the same on all
twenty-five of its rows.

The verdict at issue is derivable from the same file: ``gates.json`` records each gate's
``passed`` flag, and ``_verdict_from_recorded_gates`` already reconstructs the verdict from them
for ``reconcile_verdicts``. So the scorer writes that, and carries the status under its own
name, ``recipe_status``.

Every fixture here is built in a temporary directory. The real store is never read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from published_forecasts import (  # noqa: E402
    SCORECARD_COLUMNS,
    SCORECARD_SCHEMA_VERSION,
    _verdict_from_recorded_gates,
    score_published,
)

STATUS = "candidate -- pre-tuning"


# ── fixtures: an issue whose gates.json records flags, as the writers record them ──

def _actuals(path: Path, start="2024-01-02", n=300) -> Path:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(3)
    pd.DataFrame({"date": idx,
                  "Revenues": 4.0e7 + 5.0e6 * rng.normal(0, 1, n)}).to_csv(path, index=False)
    return path


def _issue(root: Path, issue_date: str, gates: dict, status: str = STATUS) -> Path:
    """One target, five horizons, and a gates.json in the shape both publish paths write."""
    d = root / issue_date
    d.mkdir(parents=True, exist_ok=True)
    origin = "2024-06-28"
    dates = pd.bdate_range("2024-07-01", periods=5)
    pd.DataFrame([{
        "target": "Revenues", "horizon": h, "origin_date": origin, "origin_value": 4.6e7,
        "target_date": str(td.date()), "p10": 1.0e7, "p50": 4.0e7, "p90": 9.0e7,
        "point_model": "LightGBM_L1", "interval_model": "GBQuantile",
        "target_transform": "ratio",
    } for h, td in enumerate(dates, start=1)]).to_csv(d / "forecast.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({
        "issue_date": issue_date, "data_sha_at_issue": "sha_aaa", "git_sha_at_issue": "git_bbb",
        "recipes": [{"target": "Revenues", "recipe_id": "rev-v1"}],
    }), encoding="utf-8")
    (d / "gates.json").write_text(json.dumps({
        "rev-v1": {"target": "Revenues", "gates": gates, "status": status},
    }), encoding="utf-8")
    return d


def _passed(**flags) -> dict:
    return {name: {"passed": v, "name": name} for name, v in flags.items()}


def _score(tmp_path: Path, gates: dict) -> pd.DataFrame:
    data = _actuals(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _issue(root, "2026-08-13", gates)
    out = score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 5, out
    return pd.read_csv(tmp_path / "sc.csv")


# ── the column holds a verdict ───────────────────────────────────────────────

def test_a_failed_signal_gate_is_written_as_withheld_as_forecast_not_as_the_status(tmp_path):
    """The pre-P2 shape the 2025-08-06 issue carries: signal failed, the rest passed."""
    sc = _score(tmp_path, _passed(signal=False, overfitting=True, vs_ruler=True))
    assert set(sc["publication_verdict"]) == {"withheld_as_forecast"}, (
        f"publication_verdict holds {set(sc['publication_verdict'])}; the status is not a verdict")


def test_all_gates_passing_is_written_as_publishable(tmp_path):
    """The post-P2 shape the 2026-08-13 and 2026-08-16 issues carry."""
    sc = _score(tmp_path, _passed(accuracy_vs_naive=True, signal=True, leakage=True,
                                   persistence_mimicry=True, coverage=None, overfitting=True))
    assert set(sc["publication_verdict"]) == {"publishable"}


def test_the_recipe_status_is_carried_under_its_own_name(tmp_path):
    sc = _score(tmp_path, _passed(signal=True))
    assert "recipe_status" in SCORECARD_COLUMNS
    assert set(sc["recipe_status"]) == {STATUS}


def test_an_issue_with_no_recorded_gate_flags_gets_no_verdict_rather_than_a_pass(tmp_path):
    """An empty gates dict is 'nothing measured', and nothing measured is never a pass."""
    sc = _score(tmp_path, {})
    assert sc["publication_verdict"].isna().all() or set(sc["publication_verdict"]) == {""}, (
        f"an issue with no gate flags was given the verdict {set(sc['publication_verdict'])}")
    assert set(sc["recipe_status"]) == {STATUS}


def test_a_failed_accuracy_gate_reads_as_withheld_under_the_gate_set_that_records_it():
    """Severity follows the gate set the artifact records.

    ``accuracy_vs_naive`` exists only in post-P2 issues, and under P2 a failed accuracy gate
    means ``withheld`` (publication_gates._SEVERITY), not ``withheld_as_forecast``. Before this
    the derivation applied the pre-P2 severity to every issue, so a post-P2 issue that failed
    accuracy would have been reconstructed one step too lenient.
    """
    gates = _passed(accuracy_vs_naive=False, signal=True, leakage=True,
                    persistence_mimicry=True, overfitting=True)
    assert _verdict_from_recorded_gates(gates) == "withheld"
    # And the pre-P2 derivations are unchanged.
    assert _verdict_from_recorded_gates(_passed(signal=False, vs_ruler=True)) == "withheld_as_forecast"
    assert _verdict_from_recorded_gates(_passed(leakage=False, signal=True)) == "withheld"
    assert _verdict_from_recorded_gates(_passed(signal=True, vs_ruler=True)) == "publishable"


# ── the schema says what changed ─────────────────────────────────────────────

def test_the_schema_version_was_bumped_for_the_new_column():
    assert SCORECARD_SCHEMA_VERSION >= 3
    src = (BACKEND / "published_forecasts.py").read_text(encoding="utf-8")
    i = src.index("SCORECARD_SCHEMA_VERSION = ")
    preamble = src[max(0, i - 2500):i]
    assert "3 --" in preamble and "recipe_status" in preamble, (
        "version 3 must say it added recipe_status, so a bump is decodable")


def test_recipe_status_sits_beside_the_verdict_in_the_column_order():
    cols = list(SCORECARD_COLUMNS)
    assert cols.index("recipe_status") == cols.index("publication_verdict") + 1
