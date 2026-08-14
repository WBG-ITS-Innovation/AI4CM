"""Reconciling a published verdict with today's, and the guards that changed with it.

P2 moved every publication verdict. A published issue is immutable — its `gates.json` records what
was decided on the issue date, and rewriting it would destroy the only record of what was actually
said. So the two must be *reconciled*, not merged: both are true, and they answer different
questions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from published_forecasts import (  # noqa: E402
    _verdict_from_recorded_gates,
    reconcile_verdicts,
)

PUBLISHED = REPO / "forecasts" / "published"


# ── the published record is never touched ────────────────────────────────────

def test_reconciling_does_not_modify_any_published_issue():
    before = {p: p.read_bytes() for p in sorted(PUBLISHED.rglob("gates.json"))}
    reconcile_verdicts()
    after = {p: p.read_bytes() for p in sorted(PUBLISHED.rglob("gates.json"))}
    assert before == after, "reconciliation must read, never write"
    assert before, "there should be at least one published issue to reconcile"


def test_the_issue_time_verdict_is_derived_from_what_the_artifact_records():
    """`gates.json` stores gate outcomes but never stored a verdict, so it is reconstructed."""
    assert _verdict_from_recorded_gates(
        {"signal": {"passed": True}, "overfitting": {"passed": True}}) == "publishable"
    assert _verdict_from_recorded_gates(
        {"signal": {"passed": False}}) == "withheld_as_forecast"
    assert _verdict_from_recorded_gates(
        {"signal": {"passed": True}, "leakage": {"passed": False}}) == "withheld"


# ── the three real reconciliations ───────────────────────────────────────────

def _by_target(issue: str = "2025-08-06"):
    return {r["target"]: r for r in reconcile_verdicts() if r["issue_date"] == issue}


@pytest.mark.parametrize("target,at_issue,today", [
    ("Revenues", "withheld_as_forecast", "publishable"),
    ("Expenditure", "withheld_as_forecast", "withheld"),
    ("State budget balance", "publishable", "withheld"),
])
def test_every_published_target_reports_both_verdicts(target, at_issue, today):
    r = _by_target()[target]
    assert r["verdict_at_issue"] == at_issue
    assert r["verdict_today"] == today
    assert r["changed"] is True


def test_the_reason_names_the_gate_that_actually_changed_it():
    """Not every gate that differs — only the ones that drove the verdict.

    An earlier draft listed all four added gates including the three that pass, which buried the
    one that mattered.
    """
    rev = _by_target()["Revenues"]["why"]
    assert "signal was re-thresholded from 1.5 to 1.15" in rev
    assert "1.2255" in rev, "the measurement is unchanged and should be quoted as such"
    assert "coverage" not in rev and "leakage" not in rev, (
        "gates that pass did not change the verdict and must not be listed")

    stock = _by_target()["State budget balance"]["why"]
    assert "accuracy_vs_naive" in stock and "1.57832" in stock
    assert "signal" not in stock, "the stock target's signal gate passed both before and after"


def test_the_reconciliation_says_the_numbers_did_not_change():
    """The forecast is the same; only the verdict attached to it moved."""
    for r in _by_target().values():
        assert "numbers in the published issue have not changed" in r["why"]
        assert "immutable" in r["note"]


def test_gate_changes_distinguish_added_from_rethresholded():
    changes = {c["gate"]: c for c in _by_target()["Revenues"]["gate_changes"]}
    assert changes["accuracy_vs_naive"]["change"] == "added"
    assert changes["signal"]["change"] == "rethresholded"
    assert changes["signal"]["threshold_at_issue"] == 1.5
    assert changes["signal"]["threshold_now"] == 1.15
    assert changes["vs_ruler"]["change"] == "removed", (
        "vs_ruler was a gate at issue time and is not one now")


def test_a_recipe_missing_from_the_registry_is_reported_not_dropped(tmp_path):
    issue = tmp_path / "2025-01-01"
    issue.mkdir(parents=True)
    (issue / "forecast.csv").write_text("target,horizon\nX,5\n")
    (issue / "gates.json").write_text(json.dumps(
        {"gone-v1": {"target": "X", "gates": {"signal": {"passed": True}}}}))
    r = reconcile_verdicts(published_root=tmp_path)[0]
    assert r["verdict_today"] is None and r["changed"] is None
    assert "no longer in the registry" in r["why"]


# ── the re-issue ─────────────────────────────────────────────────────────────

REISSUE = PUBLISHED / "2026-08-13"


@pytest.mark.skipif(not REISSUE.exists(), reason="no re-issue committed")
def test_the_reissue_carries_the_new_gate_set():
    """At least one published issue now reflects the P2 verdicts."""
    g = json.loads((REISSUE / "gates.json").read_text())
    entry = next(iter(g.values()))
    gates = entry["gates"]
    assert "accuracy_vs_naive" in gates, "the re-issue must record the gate P2 added"
    assert "vs_ruler" not in gates, "vs_ruler is no longer a gate"
    assert gates["signal"]["threshold"] == 1.15
    assert gates["signal"]["passed"] is True
    assert gates["accuracy_vs_naive"]["passed"] is True


@pytest.mark.skipif(not REISSUE.exists(), reason="no re-issue committed")
def test_the_reissue_is_the_publishable_target_only():
    fc = pd.read_csv(REISSUE / "forecast.csv")
    assert set(fc["target"]) == {"Revenues"}, (
        "only the target whose current verdict is publishable should be re-issued")
    assert "y_true" not in fc.columns


@pytest.mark.skipif(not REISSUE.exists(), reason="no re-issue committed")
def test_the_reissue_reconciles_as_unchanged():
    """Issued under the current gates, so its two verdicts agree."""
    r = [x for x in reconcile_verdicts() if x["issue_date"] == "2026-08-13"]
    assert r, "the re-issue should appear in the reconciliation"
    assert all(x["changed"] is False for x in r), [x["why"] for x in r]
    assert "Unchanged" in r[0]["why"]


def test_publishing_a_withheld_recipe_is_refused():
    """`withheld` means a trivial benchmark is more accurate, so the numbers must not go out.

    Distinct from `withheld_as_forecast`, which still publishes: there the numbers remain the
    best estimate available and only the event claim is withheld.
    """
    from forecast_modes import NotOfficial, OfficialResult

    res = OfficialResult(target="State budget balance", recipe_id="x", model="m", horizon=5,
                         forecasts=pd.DataFrame(), provenance={}, gates={})
    with pytest.raises(NotOfficial, match="verdict is 'withheld'"):
        from forecast_modes import publish_official
        publish_official(res)


# ── the holdout read is recorded, not silent ─────────────────────────────────

def test_the_two_read_purposes_are_distinct():
    """Reporting over the holdout is what it is FOR; consulting it to choose is not.

    The module's own discipline draws this line -- "TEST is run at the end of a milestone to
    report what would have happened... Each time TEST is consulted **to make a choice**, it stops
    being a clean holdout" -- and the code did not, so a reporting read went through neither the
    gate nor the log.
    """
    from evaluation_windows import PURPOSE_REPORT, PURPOSE_SELECTION

    assert PURPOSE_SELECTION == "selection" and PURPOSE_REPORT == "report"


def test_a_reporting_read_is_logged_without_raising(tmp_path, monkeypatch):
    import evaluation_windows as ew

    log = tmp_path / "access.log"
    monkeypatch.setattr(ew, "TEST_ACCESS_LOG", log)
    ew.require_test_access("covering 156 holdout dates", caller="unit",
                           purpose=ew.PURPOSE_REPORT)

    entry = json.loads(log.read_text().strip())
    assert entry["purpose"] == "report"
    assert entry["caller"] == "unit"
    assert "156 holdout dates" in entry["reason"]


def test_a_selection_read_still_raises_when_the_holdout_is_closed(tmp_path, monkeypatch):
    import evaluation_windows as ew

    monkeypatch.setattr(ew, "TEST_ACCESS_LOG", tmp_path / "access.log")
    monkeypatch.delenv(ew.TEST_ACCESS_ENV, raising=False)
    with pytest.raises(ew.TestWindowAccessError):
        ew.require_test_access("choosing a model", caller="unit")


def test_an_unknown_purpose_is_refused(tmp_path, monkeypatch):
    import evaluation_windows as ew

    monkeypatch.setattr(ew, "TEST_ACCESS_LOG", tmp_path / "access.log")
    with pytest.raises(ValueError, match="purpose must be"):
        ew.require_test_access("something", purpose="whatever")


@pytest.mark.parametrize("module,fn,caller", [
    ("run_a_stat", "main", "run_a_stat.main"),
    ("c_dl_pipeline", "build_yearly_folds", "c_dl_pipeline.yearly_folds"),
])
def test_the_families_that_report_over_the_holdout_record_the_read(module, fn, caller):
    """A_STAT folds over every full year; C_DL's runners default eval_start to TEST_START.

    Both reached 2025 with no gate and no log entry. Neither gets a *selection* guard, because
    neither is choosing at that point -- A_STAT runs one model per invocation, and building folds
    chooses nothing.
    """
    import importlib

    src = Path(importlib.import_module(module).__file__).read_text()
    assert "PURPOSE_REPORT" in src, f"{module} does not record its holdout read"
    assert caller in src, f"{module} does not identify itself in the log entry"
    assert "assert_selection_free" not in src, (
        f"{module} makes no selection at this point and must not carry a selection guard")
