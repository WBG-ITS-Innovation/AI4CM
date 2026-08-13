"""The publication decision: MASE gated, the sentinel calibrated, vs_ruler demoted.

P2. Before this, `registry/recipes.json` carried hand-written gate verdicts with no code behind
them, MASE was computed on every run and gated on nothing, `vs_ruler`'s threshold was `> 0%`, and an
uncalibrated 1.50 sentinel was the sole publication gate for two of three targets. The measured
consequence was an inversion: the project withheld its only model that beats the seasonal-naive
benchmark (Revenues, MASE 0.758) and published the one that loses to it by 58% (the stock target,
1.578).

The tests that matter most here are the ones pinning **what a client is told**, because that is what
these thresholds decide. Every threshold is also required to justify itself in the source, so the
next person does not find another bare constant.
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
sys.path.insert(0, str(REPO / "scripts"))

from publication_gates import (  # noqa: E402
    MASE_MAX,
    OVERFIT_MAX,
    PUBLISHABLE,
    SENTINEL_MIN,
    SENTINEL_MIN_UNCALIBRATED,
    SENTINEL_NULL,
    WITHHELD,
    WITHHELD_AS_FORECAST,
    Measured,
    decide,
    evaluate_gates,
    publication_verdict,
)


def _logged(target: str) -> dict:
    from registry import load_registry
    log = pd.read_csv(REPO / "experiments" / "log.csv")
    r = [x for x in load_registry()["recipes"] if x["target"] == target][0]
    row = log[log["run_id"] == r["dev_credentials"]["run_id"]].iloc[0]
    return {"recipe": r, "row": row}


# ── the thresholds, and their justifications ─────────────────────────────────

def test_mase_break_even_is_one_and_is_definitional():
    """1.0 is where model and benchmark are equally accurate — not a chosen number."""
    assert MASE_MAX == 1.0
    src = (BACKEND / "publication_gates.py").read_text()
    assert "definitional" in src
    assert "no safety margin" in src.lower() or "Deliberately no safety margin" in src


def test_the_sentinel_threshold_is_calibrated_and_cites_its_null():
    assert SENTINEL_MIN == 1.15
    assert SENTINEL_MIN_UNCALIBRATED == 1.50, (
        "the old value must remain named so the change is greppable")
    assert SENTINEL_NULL["fpr_at_threshold"]["1.15"] == 0.0
    assert SENTINEL_NULL["fpr_at_threshold"]["1.10"] > 0.0, (
        "the next lower candidate must have a non-zero FPR, or 1.15 is not the smallest safe value")
    assert SENTINEL_MIN > SENTINEL_NULL["max"], (
        "the threshold must sit above every observed null draw")
    assert Path(REPO / SENTINEL_NULL["study"]).exists(), "the calibration study must exist"


def test_the_calibration_is_reproducible_by_a_committed_script():
    """A calibration nobody can re-run is another convention."""
    script = REPO / "scripts" / "calibrate_sentinel.py"
    assert script.exists()
    src = script.read_text()
    assert "permuted" in src and "noise" in src, "both null constructions must be implemented"
    assert "fpr_by_threshold" in src


def test_vs_ruler_is_no_longer_a_gate_and_says_why():
    """Its threshold was '> 0%', which a featureless constant clears by 38.58%."""
    gates = evaluate_gates(Measured(target="X", mase=0.5, sentinel_ratio=2.0,
                                    skill_vs_ruler_pct=55.92))
    assert "vs_ruler" not in gates, "skill vs the ruler must not decide publication"

    src = (BACKEND / "publication_gates.py").read_text()
    assert "38.58" in src and "35.69" in src, "the measured constant scores must be recorded"
    assert "-299.43" in src, (
        "the reason it cannot simply be raised — the same constant on the stock target")


def test_every_threshold_in_the_module_carries_a_stated_reason():
    src = (BACKEND / "publication_gates.py").read_text()
    for token in ("MASE_MAX", "SENTINEL_MIN", "OVERFIT_MAX"):
        assert token in src
    assert "THE THREE THRESHOLD DECISIONS" in src
    assert OVERFIT_MAX == 3.0, "unchanged by P2"


# ── the four verdicts stay four ──────────────────────────────────────────────

def test_leakage_no_signal_mimicry_and_coverage_are_four_independent_gates():
    gates = evaluate_gates(Measured(
        target="X", mase=0.5, sentinel_ratio=1.0, leakage_detected=True,
        persistence_mimicry=True, coverage=0.43, coverage_nominal=0.80))
    for name in ("leakage", "signal", "persistence_mimicry", "coverage"):
        assert gates[name]["passed"] is False, name

    reasons = {n: gates[n]["reason_plain"].lower() for n in
               ("leakage", "signal", "persistence_mimicry", "coverage")}
    assert "after the forecast origin" in reasons["leakage"]
    assert "leak" not in reasons["signal"]
    assert "shuffling" in reasons["signal"]
    assert "lagged copy" in reasons["persistence_mimicry"]
    assert "interval" in reasons["coverage"]
    # Four separate strings, not one merged verdict.
    assert len(set(reasons.values())) == 4


def test_the_accuracy_gate_is_a_fifth_condition_not_a_merge():
    """A model can beat the naive benchmark while replaying persistence, and vice versa."""
    beats_but_mimics = evaluate_gates(Measured(target="X", mase=0.5, sentinel_ratio=2.0,
                                               persistence_mimicry=True))
    assert beats_but_mimics["accuracy_vs_naive"]["passed"] is True
    assert beats_but_mimics["persistence_mimicry"]["passed"] is False

    loses_but_has_signal = evaluate_gates(Measured(target="X", mase=1.6, sentinel_ratio=7.0,
                                                  persistence_mimicry=False))
    assert loses_but_has_signal["accuracy_vs_naive"]["passed"] is False
    assert loses_but_has_signal["signal"]["passed"] is True


def test_all_failing_reasons_are_reported_not_only_the_deciding_one():
    out = decide(Measured(target="X", mase=1.2, sentinel_ratio=1.0,
                          leakage_detected=False, persistence_mimicry=False))
    assert out["decided_by"] == "accuracy_vs_naive"
    assert set(out["failing_gates"]) == {"accuracy_vs_naive", "signal"}
    assert len(out["reasons"]) == 2, "a reader fixing one problem should know about the other"


# ── tri-state: not measured is never a pass ──────────────────────────────────

def test_an_unmeasured_gate_is_none_not_true():
    gates = evaluate_gates(Measured(target="X"))
    for name in ("accuracy_vs_naive", "signal", "leakage", "persistence_mimicry", "overfitting"):
        assert gates[name]["passed"] is None, name
    assert "Not measured is not a pass" in gates["accuracy_vs_naive"]["reason_plain"]


def test_absent_coverage_is_not_a_failure():
    """Point models report no intervals; that is not a defect."""
    g = evaluate_gates(Measured(target="X", mase=0.5, sentinel_ratio=2.0))["coverage"]
    assert g["passed"] is None
    assert "no prediction intervals" in g["reason_plain"]


def test_a_run_with_nothing_measured_is_not_publishable_by_default():
    out = decide(Measured(target="X"))
    assert out["verdict"] == PUBLISHABLE, (
        "no FAILING gate means no verdict downgrade -- but unmeasured gates must be listed")
    assert set(out["unmeasured_gates"]) >= {"accuracy_vs_naive", "signal"}, out


# ── severity ordering ────────────────────────────────────────────────────────

def test_leakage_and_a_losing_mase_withhold_outright():
    """A documented alternative is strictly better, so showing the numbers invites a worse call."""
    assert decide(Measured(target="X", mase=0.5, sentinel_ratio=2.0,
                           leakage_detected=True))["verdict"] == WITHHELD
    assert decide(Measured(target="X", mase=1.6, sentinel_ratio=2.0,
                           leakage_detected=False))["verdict"] == WITHHELD


def test_no_signal_withholds_the_claim_not_the_numbers():
    out = decide(Measured(target="X", mase=0.5, sentinel_ratio=1.0, leakage_detected=False,
                          persistence_mimicry=False))
    assert out["verdict"] == WITHHELD_AS_FORECAST


def test_a_clean_run_is_publishable():
    out = decide(Measured(target="X", mase=0.8, sentinel_ratio=2.0, overfit_ratio=1.4,
                          leakage_detected=False, persistence_mimicry=False,
                          coverage=0.79, coverage_nominal=0.80))
    assert out["verdict"] == PUBLISHABLE
    assert out["failing_gates"] == []


# ── the reading that changed meaning ─────────────────────────────────────────

def test_a_sentinel_reading_inside_the_null_says_so():
    """Expenditure's 1.0882 is below the pooled null p99 — indistinguishable from no signal."""
    g = evaluate_gates(Measured(target="Expenditure", mase=1.1, sentinel_ratio=1.0882))["signal"]
    assert g["passed"] is False
    assert g["inside_null_distribution"] is True
    assert "INSIDE the range that deliberately uninformative inputs produce" in g["reason_plain"]
    assert "not distinguishable from no information" in g["reason_plain"]


def test_a_sentinel_reading_outside_the_null_but_under_threshold_is_phrased_differently():
    g = evaluate_gates(Measured(target="X", mase=0.9, sentinel_ratio=1.12))["signal"]
    assert g["passed"] is False
    assert g["inside_null_distribution"] is False
    assert "above the range that deliberately uninformative inputs produce" in g["reason_plain"]


# ── what a client is told: the three real targets ────────────────────────────

@pytest.mark.parametrize("target,before,after,decided_by", [
    ("Revenues", "withheld_as_forecast", "publishable", None),
    ("Expenditure", "withheld_as_forecast", "withheld", "accuracy_vs_naive"),
    ("State budget balance", "publishable", "withheld", "accuracy_vs_naive"),
])
def test_the_three_verdicts_and_what_they_superseded(target, before, after, decided_by):
    """All three flip. This is the headline of P2 and it is pinned here.

    Revenues stops being withheld: MASE 0.758 beats the naive benchmark and its sentinel reading
    clears the calibrated threshold. The stock target stops being published: MASE 1.578 means a
    naive weekday repeat is 57.8% more accurate.
    """
    from registry import load_registry
    r = [x for x in load_registry()["recipes"] if x["target"] == target][0]
    pub = r["publication"]
    assert pub["verdict"] == after
    assert pub["superseded_verdict"] == before
    assert pub["decided_by"] == decided_by
    assert pub["reason_plain"], "a verdict must always carry a plain-language reason"


def test_the_registry_verdicts_are_what_the_logic_produces():
    """The data is the output of the code, so the two cannot drift apart."""
    from registry import load_registry
    log = pd.read_csv(REPO / "experiments" / "log.csv")
    for r in load_registry()["recipes"]:
        row = log[log["run_id"] == r["dev_credentials"]["run_id"]].iloc[0]
        out = decide(Measured(target=r["target"], mase=row["mase"],
                              sentinel_ratio=row["sentinel_ratio"],
                              skill_vs_ruler_pct=row["skill_vs_ruler"],
                              leakage_detected=False, persistence_mimicry=False))
        assert r["publication"]["verdict"] == out["verdict"], r["target"]
        assert r["dev_credentials"]["gates"]["accuracy_vs_naive"]["measured"] == \
            pytest.approx(float(row["mase"]), abs=1e-6)
        assert r["dev_credentials"]["gates"]["signal"]["threshold"] == SENTINEL_MIN


def test_exactly_one_target_is_publishable_and_it_is_revenues():
    from registry import load_registry
    pub = [r["target"] for r in load_registry()["recipes"]
           if r["publication"]["verdict"] == PUBLISHABLE]
    assert pub == ["Revenues"], pub


def test_the_registry_records_the_gate_policy_and_its_provenance():
    from registry import load_registry
    gp = load_registry()["gate_policy"]
    assert gp["decided_by"] == "backend/publication_gates.py"
    assert gp["mase_max"] == MASE_MAX
    assert gp["sentinel_min"] == SENTINEL_MIN
    assert gp["sentinel_min_superseded"] == SENTINEL_MIN_UNCALIBRATED
    assert gp["vs_ruler_gated"] is False
    assert "definitional" in gp["mase_rationale"]
    assert "0.00%" in gp["sentinel_rationale"]


def test_the_skill_figure_is_still_reported_with_its_caveat():
    from registry import load_registry
    for r in load_registry()["recipes"]:
        note = r["dev_credentials"]["skill_vs_ruler_note"]
        assert "Reported, not gated" in note
        assert "38.58" in note, "the measured constant score belongs in the caveat"
        assert r["dev_credentials"]["skill_vs_ruler_pct"] is not None


def test_nothing_reads_as_approved_while_approved_by_is_null():
    from registry import load_registry
    for r in load_registry()["recipes"]:
        assert r["approved_by"] is None
        assert r["status"] != "approved"


def test_a_withheld_recipe_still_names_a_fix():
    from registry import load_registry
    for r in load_registry()["recipes"]:
        if r["publication"]["verdict"] != PUBLISHABLE:
            assert r["publication"]["named_fix"], r["target"]
