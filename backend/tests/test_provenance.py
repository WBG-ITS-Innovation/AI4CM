"""Item 1e: a run must be identifiable after the fact.

Before this, provenance existed for exactly one of four families, no run recorded the
git SHA or the input's hash, and `run_daily_forecast.sh` selected its input with
`ls -t | head -1` -- by modification time. Two runs on the same date could use
different inputs and nothing said which (review §7.2, §7.4). `touch` on any file in
the processed directory silently changed what the pipeline forecast.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from provenance import (  # noqa: E402
    STALE_OVERRIDE_ENV,
    build_provenance,
    describe_code,
    describe_input,
    record_run,
    sha256_of,
    verify_expected_sha,
)

RUNNERS = (
    "run_a_stat.py",
    "run_b_ml_univariate.py",
    "run_c_dl_univariate.py",
    "run_c_dl_multivariate.py",
    "run_e_quantile_daily_univariate.py",
    "run_e_quantile_daily_multivariate.py",
)


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "data.csv"
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=50),
                  "Revenues": range(50)}).to_csv(p, index=False)
    return p


# ── identifying the input by content ──────────────────────────────────────

def test_sha256_matches_hashlib(csv):
    assert sha256_of(csv) == hashlib.sha256(csv.read_bytes()).hexdigest()


def test_describe_input_records_the_four_required_fields(csv):
    """The Phase-2 brief names exactly these: name, sha256, latest_data_date, n_rows."""
    info = describe_input(csv)
    assert info["name"] == "data.csv"
    assert len(info["sha256"]) == 64
    assert info["n_rows"] == 50
    assert info["latest_data_date"] == str(pd.bdate_range("2024-01-01", periods=50)[-1].date())
    assert Path(info["path"]).is_absolute()


def test_a_changed_file_gets_a_different_hash(csv):
    before = describe_input(csv)["sha256"]
    df = pd.read_csv(csv)
    df.loc[0, "Revenues"] = 999999
    df.to_csv(csv, index=False)
    assert describe_input(csv)["sha256"] != before, (
        "the hash did not change, so it cannot identify content"
    )


def test_missing_input_is_reported_not_raised():
    info = describe_input("/nonexistent/nope.csv")
    assert info["exists"] is False and info["sha256"] is None


def test_unreadable_dates_are_reported_not_raised(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("not,a,valid\ncsv,for,dates\n")
    info = describe_input(p)
    assert info["sha256"] is not None, "the hash must still be computed"
    assert "read_error" in info


# ── pinning to specific bytes ─────────────────────────────────────────────

def test_matching_expected_sha_passes(csv):
    verify_expected_sha(csv, expected=sha256_of(csv))


def test_mismatched_expected_sha_fails_closed(csv):
    """Selecting by name is necessary but not sufficient: a file can be regenerated."""
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_expected_sha(csv, expected="0" * 64)


def test_no_expectation_means_no_check(csv, monkeypatch):
    monkeypatch.delenv("AI4CM_EXPECTED_DATA_SHA256", raising=False)
    verify_expected_sha(csv)          # must not raise


def test_expected_sha_read_from_the_environment(csv, monkeypatch):
    monkeypatch.setenv("AI4CM_EXPECTED_DATA_SHA256", "1" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_expected_sha(csv)


# ── the record ────────────────────────────────────────────────────────────

def test_provenance_carries_code_data_and_environment(csv):
    rec = build_provenance("B_ML", csv, config={"horizon": 5}, seed=42)
    assert rec["family"] == "B_ML"
    assert rec["seed"] == 42
    assert rec["data_file"]["sha256"] is not None
    assert "git_sha" in rec["code"]
    assert "packages" in rec["environment"]
    assert rec["config"]["horizon"] == 5
    assert "timestamp_utc" in rec


def test_environment_captures_the_optional_boosters(csv):
    """xgboost / lightgbm availability silently changes B_ML's model set."""
    rec = build_provenance("B_ML", csv, config={})
    pkgs = rec["environment"]["packages"]
    for name in ("xgboost", "lightgbm", "numpy", "pandas"):
        assert name in pkgs, f"{name} version not recorded"


def test_git_dirty_is_recorded_so_a_sha_is_not_over_trusted():
    """A dirty tree means the SHA alone does not identify what ran."""
    code = describe_code()
    assert "git_dirty" in code and "git_dirty_files" in code


def test_stale_override_is_recorded_when_used(csv, monkeypatch):
    """Q5: stale blocks publication, overridable only explicitly -- and never silently."""
    monkeypatch.delenv(STALE_OVERRIDE_ENV, raising=False)
    assert build_provenance("A", csv, config={})["stale_override"] is False
    monkeypatch.setenv(STALE_OVERRIDE_ENV, "1")
    assert build_provenance("A", csv, config={})["stale_override"] is True


def test_record_run_writes_the_artifact(csv, tmp_path):
    out = tmp_path / "run"
    rec = record_run(out, "A_STAT", csv, config={"model": "ETS"}, seed=None)
    path = out / "artifacts" / "provenance.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["family"] == "A_STAT"
    assert on_disk["data_file"]["sha256"] == rec["data_file"]["sha256"]


def test_record_run_never_aborts_a_successful_run(tmp_path):
    """Provenance is a record, not a gate: a write failure must not lose the run."""
    rec = record_run(tmp_path / "run", "X", "/nonexistent/x.csv", config={})
    assert isinstance(rec, dict)      # returned rather than raised


# ── the wiring ────────────────────────────────────────────────────────────

def test_all_four_families_record_provenance():
    """One implementation, used everywhere -- not four copies that drift."""
    for runner in RUNNERS:
        src = (BACKEND_DIR / runner).read_text()
        assert "record_run(" in src, f"{runner} does not record provenance"
        assert "from provenance import" in src, (
            f"{runner} does not use the shared helper"
        )


def test_every_runner_verifies_a_pinned_sha():
    for runner in RUNNERS:
        src = (BACKEND_DIR / runner).read_text()
        assert "verify_expected_sha(" in src, (
            f"{runner} does not check AI4CM_EXPECTED_DATA_SHA256, so it cannot be "
            f"pinned to specific bytes"
        )


def test_the_daily_script_no_longer_selects_by_mtime():
    """The specific defect: `ls -t | head -1`.

    Checks EXECUTABLE lines only. The script deliberately documents the old
    behaviour in a comment, and a naive substring search over the whole file would
    match that explanation -- flagging the fix as the bug.
    """
    src = (REPO_ROOT / "scripts" / "run_daily_forecast.sh").read_text()
    code = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    offenders = [ln.strip() for ln in code if "ls -t" in ln]
    assert not offenders, f"the daily script still selects its input by mtime: {offenders}"

    joined = "\n".join(code)
    assert "DATA_FILE_NAME" in joined, "no explicit input name"
    assert "shasum -a 256" in joined or "sha256sum" in joined, "the script records no hash"
    assert "AI4CM_EXPECTED_DATA_SHA256" in joined, "the script cannot be pinned"
