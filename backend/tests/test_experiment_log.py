"""Ground rule 2: the experiments log must actually support "reproducible from a log row".

These pin the two properties that failed in practice while the log was being populated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import experiment_log as el  # noqa: E402


@pytest.fixture()
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "LOG_DIR", tmp_path)
    monkeypatch.setattr(el, "LOG_CSV", tmp_path / "log.csv")
    monkeypatch.setattr(el, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def _log(**over):
    kw = dict(target="Revenues", model="LightGBM_L1", git_sha="a" * 40, data_sha="b" * 64,
              feature_names=["y_lag_1"], params={"x": 1}, seed=0,
              fold_scheme="scheme", dev_mae=1.0, mase=0.5, skill_vs_ruler=10.0)
    kw.update(over)
    return el.log_run(**kw)


def test_run_ids_are_unique_within_a_burst(log_dir):
    """Regression: ids were stamped to whole seconds.

    A sweep logs tens of rows inside one second and `target_model` repeats across
    windows, so every row after the first collided -- meaning they shared one detail
    JSON and all but one run became unrecoverable. Caught by verify_log_integrity
    reporting 'duplicate run_id values' on a 36-row batch, not by review.
    """
    ids = [_log()["run_id"] for _ in range(40)]
    assert len(set(ids)) == 40, f"{40 - len(set(ids))} collision(s)"
    assert el.verify_log_integrity()["ok"], el.verify_log_integrity()["problems"]


def test_every_row_has_its_detail_json(log_dir):
    for _ in range(3):
        _log()
    for row in el.read_log():
        assert (el.RUNS_DIR / f"{row['run_id']}.json").exists()


def test_integrity_notices_a_missing_detail_json(log_dir):
    d = _log()
    (el.RUNS_DIR / f"{d['run_id']}.json").unlink()
    out = el.verify_log_integrity()
    assert out["ok"] is False
    assert any("missing detail JSON" in p for p in out["problems"])


def test_integrity_notices_header_drift(log_dir):
    _log()
    el.LOG_CSV.write_text("run_id,timestamp\nx,y\n")
    assert el.verify_log_integrity()["ok"] is False


def test_header_is_exactly_the_brief_columns(log_dir):
    _log()
    header = el.LOG_CSV.read_text().splitlines()[0].split(",")
    for required in ("timestamp", "git_sha", "data_sha", "target", "feature_set_hash",
                     "params", "seed", "fold_scheme", "dev_mae", "mase",
                     "skill_vs_ruler", "sentinel_ratio", "coverage_low",
                     "coverage_mid", "coverage_high"):
        assert required in header, f"brief column {required!r} missing"


def test_feature_hash_is_over_names_not_order(log_dir):
    assert el.feature_set_hash(["a", "b"]) == el.feature_set_hash(["b", "a"])
    assert el.feature_set_hash(["a", "b"]) != el.feature_set_hash(["a", "c"])


def test_log_is_append_only_across_calls(log_dir):
    _log()
    _log(target="Expenditure")
    rows = el.read_log()
    assert len(rows) == 2
    assert {r["target"] for r in rows} == {"Revenues", "Expenditure"}
