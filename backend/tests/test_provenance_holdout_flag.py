"""``test_window_touched`` in a forward run's provenance is measured, not written.

Why this file exists
--------------------
``build_provenance`` wrote ``"test_window_touched": False`` as a literal. The claim was true by
design (a forward run reads no truth) but nothing computed it, so a holdout read introduced by
any future change would have been published under a provenance that still said "no"
(inference-horizon map, §1.3, "One thing to read as it is").

The holdout ledger, ``experiments/test_access.log``, records every read of the sealed window.
The flag is now the difference between the ledger's length before the run and after it. These
tests point the ledger at a temporary file, so they neither read nor append to the real one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import evaluation_windows  # noqa: E402
import forward_forecast as ff  # noqa: E402
from evaluation_windows import PURPOSE_REPORT, require_test_access  # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A private ledger with two prior entries, standing in for the real one."""
    p = tmp_path / "test_access.log"
    p.write_text('{"purpose": "report", "reason": "earlier"}\n' * 2, encoding="utf-8")
    monkeypatch.setattr(evaluation_windows, "TEST_ACCESS_LOG", p)
    return p


def test_the_flag_is_false_when_the_ledger_did_not_grow(ledger, tmp_path):
    before = ff.holdout_ledger_length()
    prov = ff.build_provenance(str(tmp_path / "data.csv"), [], ledger_before=before)
    assert prov["test_window_touched"] is False
    assert prov["holdout_ledger"]["reads_during_run"] == 0


def test_a_planted_holdout_read_flips_the_flag_to_true(ledger, tmp_path):
    before = ff.holdout_ledger_length()
    require_test_access("planted by the test", caller="test_provenance_holdout_flag",
                        purpose=PURPOSE_REPORT)
    prov = ff.build_provenance(str(tmp_path / "data.csv"), [], ledger_before=before)
    assert prov["test_window_touched"] is True
    assert prov["holdout_ledger"]["reads_during_run"] == 1
    assert not any("no truth was read" in n for n in prov["notes"]), (
        "the notes still claim no truth was read after the ledger grew")


def test_the_delta_is_recorded_so_the_flag_can_be_audited(ledger, tmp_path):
    before = ff.holdout_ledger_length()
    assert before == 2
    prov = ff.build_provenance(str(tmp_path / "data.csv"), [], ledger_before=before)
    rec = prov["holdout_ledger"]
    assert rec["lines_before"] == 2 and rec["lines_after"] == 2
    assert rec["path"] == str(ledger)


def test_a_missing_ledger_counts_as_zero_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_windows, "TEST_ACCESS_LOG", tmp_path / "absent.log")
    assert ff.holdout_ledger_length() == 0


def test_the_flag_is_not_a_literal_in_the_writer():
    src = (BACKEND / "forward_forecast.py").read_text(encoding="utf-8")
    assert '"test_window_touched": False' not in src, "the flag is still written as a literal"


def test_both_official_callers_snapshot_the_ledger_before_the_fit():
    """The runner and the page path must take the 'before' reading before any model is fit,
    or a read during fitting would land before the snapshot and be missed."""
    for name in ("run_forward_forecast.py", "forecast_modes.py"):
        src = (BACKEND / name).read_text(encoding="utf-8")
        snap = src.find("holdout_ledger_length()")
        fit = src.find("run_forward(")
        assert snap != -1, f"{name} never snapshots the ledger"
        assert fit != -1 and snap < fit, f"{name} snapshots the ledger after calling run_forward"
        assert "ledger_before=" in src, f"{name} does not pass the snapshot to build_provenance"
