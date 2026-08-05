"""Part 3a: the artifact contract, tested rather than described.

Six defects, each one a place two artifacts disagreed or a consumer could not read what a
producer wrote. The tests are named for the defect they prevent, because the failure message is
the only documentation anyone reads at 2am.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forecast_integrity import (  # noqa: E402
    GATE_KEY,
    GATE_KEY_LEGACY,
    gate_check,
    read_gate,
    write_gate,
)


# ── X9: two gate keys of opposite polarity ────────────────────────────────────

def test_x9_legacy_inverted_key_is_read_correctly():
    """B_ML wrote `quality_gate_failed`; the other three wrote `quality_gate_passed`.

    daily_summary.gate_reasons() checked only the positive key, so a B_ML run that FAILED its
    gate was reported as PASSING. That is a silent inversion, not a naming inconsistency.
    """
    assert read_gate({GATE_KEY_LEGACY: True}) is False, "a failed B_ML run must read as failed"
    assert read_gate({GATE_KEY_LEGACY: False}) is True


def test_x9_absence_of_any_gate_key_is_never_a_pass():
    assert read_gate({}) is None
    assert read_gate({"skill_pct": 40.0}) is None
    assert read_gate(None) is None


def test_x9_explicit_failure_outranks_a_success_status():
    """An explicit false must beat run_status=SUCCESS in both key spellings."""
    assert read_gate({"run_status": "SUCCESS", GATE_KEY: False}) is False
    assert read_gate({"run_status": "SUCCESS", GATE_KEY_LEGACY: True}) is False
    assert read_gate({"run_status": "FAILED_QUALITY", GATE_KEY: True}) is False


def test_x9_writer_keeps_both_keys_consistent():
    """The legacy key is DERIVED, never set independently -- that is what stopped the two from
    contradicting each other."""
    r = write_gate({}, False)
    assert r[GATE_KEY] is False and r[GATE_KEY_LEGACY] is True
    r = write_gate({}, True)
    assert r[GATE_KEY] is True and r[GATE_KEY_LEGACY] is False
    r = write_gate({GATE_KEY_LEGACY: True}, None)
    assert r[GATE_KEY] is None and GATE_KEY_LEGACY not in r, (
        "an unverified verdict must not leave a stale legacy value behind"
    )


def test_x9_b_ml_no_longer_writes_the_inverted_key_alone():
    src = (BACKEND / "b_ml_pipeline.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert 'integrity_report["quality_gate_failed"] =' not in code, (
        "b_ml sets the inverted key directly again; the canonical key will go unset"
    )
    assert "write_gate(" in code


# ── F1: measured and threshold must both be numeric ──────────────────────────

def test_f1_gate_check_yields_numeric_measured_and_threshold():
    g = gate_check("signal", 1.2255, 1.5, False, "why")
    assert isinstance(g["measured"], float) and isinstance(g["threshold"], float)
    assert g["measured"] == pytest.approx(1.2255)


def test_f1_strings_are_coerced_or_nulled_never_passed_through():
    """A string in either field makes the pair impossible to compare or plot, which is how a
    gate ends up displayed without the number that justified it."""
    g = gate_check("a", "1.23", "1.50", True)
    assert g["measured"] == pytest.approx(1.23) and g["threshold"] == pytest.approx(1.50)
    g2 = gate_check("b", "n/a", "—", None)
    assert g2["measured"] is None and g2["threshold"] is None
    g3 = gate_check("c", float("nan"), 1.5, None)
    assert g3["measured"] is None, "NaN must become None, not a number that fails comparisons"


# ── X10: one publisher, everything else derives ──────────────────────────────

def test_x10_summary_derives_the_gate_and_records_its_source():
    """SUMMARY.json contradicted integrity_report.json on the 2026-08-04 C_DL run because both
    computed the verdict independently. The summary must derive and say so."""
    src = (Path(BACKEND).parent / "scripts" / "daily_summary.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "read_gate(report)" in code, "the summary no longer reads the published verdict"
    assert '"gate_source"' in code, "the summary does not record which source produced it"
    assert '"gate_published_by_family"' in code, (
        "the family's own verdict is not carried through, so a disagreement would be invisible"
    )


def test_x10_summary_does_not_assert_a_pass_from_mere_existence():
    """`integrity_found -> True` was the old rule: a report that existed but recorded no gate
    verdict became a pass."""
    src = (Path(BACKEND).parent / "scripts" / "daily_summary.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert 'elif info["integrity_found"]:\n        info["gate_passed"] = True' not in code


# ── run_id in SUMMARY.json ───────────────────────────────────────────────────

def test_summary_json_carries_run_id_and_schema_version():
    """Without an identifier a consumer holding the file cannot say which run produced it."""
    src = (Path(BACKEND).parent / "scripts" / "daily_summary.py").read_text()
    assert '"run_id": run_dir.name' in src
    assert '"schema_version"' in src


# ── C8: the interval's advertised level is recorded as data ──────────────────

def test_c8_b_ml_captures_the_nominal_interval_level():
    """ConfigBML.nominal_pi configured the conformal interval and was written nowhere, so a
    consumer had y_lo/y_hi with no idea what coverage they claimed."""
    src = (BACKEND / "b_ml_pipeline.py").read_text()
    assert "_nominal_pi_used" in src, "the advertised level is still not captured"


# ── X11: a baseline over zero prediction rows is not a baseline ──────────────

def test_x11_summary_flags_a_baseline_with_no_prediction_rows():
    src = (Path(BACKEND).parent / "scripts" / "daily_summary.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '"n_prediction_rows"' in code
    assert '"baseline_without_predictions"' in code
    assert "persistence baseline reported with zero prediction rows" in src, (
        "the condition is recorded but never raised as a gate reason"
    )
