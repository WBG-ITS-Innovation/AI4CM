"""`data_file` in SUMMARY.json — review C1.

`SUMMARY.txt` has printed `Data file: <name>` on line 4 since it was written. The JSON twin
carried no such key, so two artifacts of the *same run* disagreed about whether the input was
knowable, and a consumer reaching for `data_file` got `None` and rendered the word "None" on a
dashboard. The writer had the value the whole time — `daily_summary.py` takes `--data-file` and
already uses it for the freshness check.

These tests run the real writer and read what it produced, rather than grepping its source. The
contract itself records why that distinction matters: `test_artifact_contract.py` asserted
`run_id` and `schema_version` for months while no artifact on disk carried either, because it
was checking the writer's text and not the file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "daily_summary.py"


def _write_family_outputs(out_dir: Path, model: str, mae: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "origin_date": "2026-07-16", "target_date": "2026-07-21", "origin_value": 1.0,
        "target": "Revenues", "horizon": 5, "model": model,
        "y_true": 2.0, "y_pred": 1.5,
    }]).to_csv(out_dir / "predictions_long.csv", index=False)
    pd.DataFrame([{"model": model, "MAE": mae}]).to_csv(
        out_dir / "leaderboard.csv", index=False)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    fam = d / "a_stat"
    _write_family_outputs(fam, "SARIMAX", 5.0e7)
    (fam / "integrity_report.json").write_text(json.dumps({
        "skill_pct": 27.5, "run_status": "OK", "quality_gate_passed": True,
        "leakage_warning": False, "shift_interpretation": "OK: no shift detected",
    }))
    return d


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    p = tmp_path / "master_daily_clean_treasury.csv"
    pd.DataFrame({"date": ["2026-07-20", "2026-07-21"],
                  "Revenues": [1.0, 2.0]}).to_csv(p, index=False)
    return p


def _run(run_dir: Path, data_file: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--run-dir", str(run_dir), "--data-file", str(data_file),
         "--target", "Revenues", "--cadence", "daily", "--horizon", "5",
         "--run-date", "2026-07-21", "--families", "A_STAT"],
        capture_output=True, text=True)


@pytest.fixture
def summary(run_dir: Path, data_file: Path) -> dict:
    proc = _run(run_dir, data_file)
    assert proc.returncode == 0, proc.stderr
    return json.loads((run_dir / "SUMMARY.json").read_text())


def test_summary_json_records_the_input_file(summary):
    assert summary["data_file"] == "master_daily_clean_treasury.csv"


def test_it_is_the_bare_name_not_a_path(summary):
    """The absolute path is machine-specific and does not belong in a published interface."""
    assert "/" not in summary["data_file"]
    assert "\\" not in summary["data_file"]


def test_the_json_and_the_text_report_now_agree(run_dir, data_file, summary):
    """The defect was the two disagreeing, so the fix is tested as agreement."""
    text = (run_dir / "SUMMARY.txt").read_text()
    assert f"Data file:  {summary['data_file']}" in text


def test_a_consumer_never_has_to_render_none(summary):
    """The literal failure C1 produced downstream."""
    assert summary["data_file"] is not None
    assert str(summary["data_file"]).strip() != ""
    assert str(summary["data_file"]).lower() != "none"


def test_the_validator_accepts_a_run_that_carries_it(run_dir, data_file, summary):
    sys.path.insert(0, str(REPO / "backend"))
    from artifact_validation import validate_run

    rep = validate_run(run_dir)
    complaints = [f for f in rep.findings if "data_file" in f.message]
    assert not complaints, complaints


def test_the_validator_reports_a_run_that_omits_it(run_dir, data_file, summary):
    """A historical artifact stays readable, but the absence is stated, not ignored."""
    sys.path.insert(0, str(REPO / "backend"))
    from artifact_validation import validate_run

    stripped = dict(summary)
    stripped.pop("data_file")
    (run_dir / "SUMMARY.json").write_text(json.dumps(stripped, indent=2))

    rep = validate_run(run_dir)
    msgs = [f.message for f in rep.findings if "data_file" in f.message]
    assert msgs, "an absent data_file must be reported"
    assert any("cannot say which dataset" in m for m in msgs), msgs


def test_the_validator_rejects_a_path_where_a_name_belongs(run_dir, data_file, summary):
    sys.path.insert(0, str(REPO / "backend"))
    from artifact_validation import ERROR, validate_run

    bad = dict(summary)
    bad["data_file"] = "/Users/someone/data/processed/master.csv"
    (run_dir / "SUMMARY.json").write_text(json.dumps(bad, indent=2))

    rep = validate_run(run_dir)
    errs = [f for f in rep.findings
            if f.severity == ERROR and "data_file" in f.message]
    assert errs, "a full path must be rejected, not published"


def test_the_contract_documents_the_field():
    """Writer and contract have to agree; that is the whole point of the document."""
    contract = (REPO / "docs" / "AGENT_ARTIFACT_CONTRACT.md").read_text()
    assert "`data_file`" in contract
    assert "bare name" in contract
