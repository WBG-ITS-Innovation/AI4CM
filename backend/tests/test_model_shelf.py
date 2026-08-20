"""The Forecast page must show what came second, and must not choose anything by doing so.

Why this exists
---------------
The page showed one model per target. A reader could see that the champion beat a
benchmark but not what else had been tried, so "this is the best model" had to be taken
on trust. ``model_shelf`` reads the run ledger and answers that, which introduces two
risks worth pinning:

1. It could quietly become a *selector*. Ranking runs by MASE is one line away from
   crowning one. These tests assert that nothing here writes the registry, that the
   champion is read rather than computed, and that the eligibility judgement comes from
   ``publication_gates`` rather than from a threshold retyped in this module.
2. It could flatter. A shelf that lists a model twice, or lists a model that loses to the
   naive benchmark as an "alternative", makes the work look deeper than it is. These tests
   assert deduplication and the accuracy-gate filter, and that an empty shelf explains
   itself rather than rendering blank.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import model_shelf  # noqa: E402
from model_shelf import (  # noqa: E402
    MEASURED_WINDOW,
    alternatives_for,
    base_model_name,
    champion_model,
    champion_sentence,
    measured_models_for,
    ops_comparison,
    shelf_for,
)
from registry import load_registry  # noqa: E402

TARGETS = [r["target"] for r in load_registry()["recipes"]]
FLOW_TARGETS = ["Revenues", "Expenditure"]
STOCK_TARGET = "State budget balance"


# ---------------------------------------------------------------------------
# It reads. It does not choose.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_champion_is_read_from_the_registry_not_computed(target):
    """The best-MASE model is frequently NOT the champion, and must not become it.

    On Expenditure the registry deliberately promotes a model that is not the DEV best, and
    records why. If this module derived the champion by ranking, that recorded decision
    would be silently overridden on the page.
    """
    registry_champion = next(r["point_model"] for r in load_registry()["recipes"]
                             if r["target"] == target)
    assert champion_model(target) == registry_champion


def test_the_shelf_does_not_write_anything():
    """No write path exists. Asserted at the source, because a future edit is the risk.

    Reads are fine and there are several; what must never appear is a write. The list below
    is every way this module could acquire one without anyone noticing in review.
    """
    src = (BACKEND_DIR / "model_shelf.py").read_text(encoding="utf-8")
    for forbidden in ("write_text(", "write_bytes(", "to_csv(", "to_json(", "json.dump(",
                      ".unlink(", ".mkdir(", "shutil.", '"w"', "'w'", '"a"', "'a'"):
        assert forbidden not in src, f"model_shelf must not write: found {forbidden}"


def test_eligibility_comes_from_the_real_gate_code():
    """A threshold retyped here is a second gate policy that will drift from the first."""
    src = (BACKEND_DIR / "model_shelf.py").read_text(encoding="utf-8")
    assert "from publication_gates import" in src
    assert "1.0" not in src.split("def _judge")[1].split("def ")[1], (
        "the MASE threshold must not be hardcoded in the eligibility judgement"
    )


def test_the_registry_file_is_untouched_by_building_every_shelf():
    """End to end: build all three shelves and the registry bytes are identical."""
    path = REPO / "registry" / "recipes.json"
    before = path.read_bytes()
    for target in TARGETS:
        shelf_for(target)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# It does not flatter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_no_model_appears_twice_on_a_shelf(target):
    """The ledger holds reproductions and re-logs of the same configuration."""
    names = [base_model_name(e.model) for e in measured_models_for(target)]
    assert len(names) == len(set(names)), f"duplicate models on the {target} shelf: {names}"


@pytest.mark.parametrize("target", TARGETS)
def test_every_alternative_cleared_the_accuracy_gate(target):
    for alt in alternatives_for(target):
        assert alt.gate_eligible
        assert alt.mase is not None and alt.mase < 1.0, (
            f"{alt.model} is listed as an alternative but loses to the naive benchmark"
        )


@pytest.mark.parametrize("target", TARGETS)
def test_the_champion_is_never_listed_as_its_own_alternative(target):
    champ = base_model_name(champion_model(target) or "")
    for alt in alternatives_for(target):
        assert base_model_name(alt.model) != champ


@pytest.mark.parametrize("target", TARGETS)
def test_alternatives_are_ordered_best_first(target):
    scores = [a.mase for a in alternatives_for(target, k=5)]
    assert scores == sorted(scores)


def test_a_target_with_no_eligible_alternative_says_why():
    """State budget balance has 12 measured models and none beats the naive benchmark.

    Rendering that as an empty table would read as "nothing was tried". It is the opposite:
    a great deal was tried and none of it cleared the bar, which is the more useful fact.
    """
    shelf = shelf_for(STOCK_TARGET)
    assert shelf["alternatives"] == []
    reason = shelf["no_alternatives_reason"]
    assert reason and reason.endswith(".")
    assert "naive benchmark" in reason


@pytest.mark.parametrize("target", TARGETS)
def test_the_reason_field_is_empty_only_when_there_are_alternatives(target):
    shelf = shelf_for(target)
    assert bool(shelf["no_alternatives_reason"]) == (not shelf["alternatives"])


def test_alternatives_are_capped_at_what_was_asked_for():
    assert len(alternatives_for("Expenditure", k=1)) <= 1
    assert len(alternatives_for("Expenditure", k=2)) <= 2


# ---------------------------------------------------------------------------
# The comparison against the method actually in use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", FLOW_TARGETS)
def test_ops_comparison_is_measured_on_the_same_year_as_the_champion(target):
    ops = ops_comparison(target)
    assert ops["available"]
    assert ops["window"] == MEASURED_WINDOW
    assert ops["n"] > 200, "a 2024 comparison should cover roughly a year of working days"
    registry_n = next(r["dev_credentials"]["n"] for r in load_registry()["recipes"]
                      if r["target"] == target)
    assert ops["n"] == registry_n, (
        "the Treasury method and the champion must be averaged over the same days, "
        "or the percentage between them means nothing"
    )


@pytest.mark.parametrize("target", FLOW_TARGETS)
def test_ops_skill_is_consistent_with_the_two_errors_it_compares(target):
    ops = ops_comparison(target)
    expected = (1.0 - ops["model_mae"] / ops["ops_mae"]) * 100.0
    assert abs(ops["skill_pct"] - expected) < 1e-6


def test_ops_comparison_on_a_stock_target_says_why_it_cannot_be_made():
    """The Treasury method aggregates a flow to an annual total. A balance has none.

    Returning a blank here would let a reader infer the model tied with the current method.
    """
    ops = ops_comparison(STOCK_TARGET)
    assert ops["available"] is False
    assert ops["skill_pct"] is None
    assert "annual total" in ops["reason"]


def test_ops_comparison_on_a_missing_file_explains_rather_than_raises(tmp_path):
    ops = ops_comparison("Revenues", data_path=tmp_path / "absent.csv")
    assert ops["available"] is False
    assert "not found" in ops["reason"]


# ---------------------------------------------------------------------------
# The one sentence a reader actually reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_the_champion_sentence_is_prose(target):
    sentence = champion_sentence(target, ops_comparison(target))
    assert sentence.endswith(".")
    assert "--" not in sentence and "—" not in sentence
    assert "MASE" not in sentence and "dev_mae" not in sentence
    assert "never fitted on" in sentence, "the sentence must state the evidence, not the model"


@pytest.mark.parametrize("target", TARGETS)
def test_the_sentence_does_not_overclaim_a_withheld_model(target):
    """A model that loses to the naive rule must not be described as more accurate than it."""
    rec = next(r for r in load_registry()["recipes"] if r["target"] == target)
    sentence = champion_sentence(target, ops_comparison(target))
    if rec["dev_credentials"]["mase"] > 1.0:
        assert "less accurate than repeating the same weekday" in sentence
        assert "which is why it is withheld" in sentence


def test_a_target_with_no_recipe_gets_a_sentence_rather_than_a_crash():
    sentence = champion_sentence("Nonexistent line")
    assert sentence.endswith(".")
    assert "no champion" in sentence


# ---------------------------------------------------------------------------
# Runnability, so a comparison button cannot offer a model that will fail
# ---------------------------------------------------------------------------

def test_runnable_is_unknown_until_the_pool_is_supplied():
    for alt in alternatives_for("Expenditure"):
        assert alt.runnable is None


def test_runnable_marks_only_models_this_build_can_fit():
    from b_ml_pipeline import available_models

    pool = set(available_models())
    shelf = shelf_for("Revenues", runnable_models=pool)
    for alt in shelf["alternatives"]:
        assert alt.runnable == (base_model_name(alt.model) in pool)
    assert all(name in pool for name in shelf["comparable"])


def test_base_model_name_strips_transforms_and_relog_suffixes():
    assert base_model_name("LightGBM_L1+ratio") == "LightGBM_L1"
    assert base_model_name("GBQuantile_tuned_recomputed") == "GBQuantile_tuned"
    assert base_model_name("Ridge") == "Ridge"
    assert base_model_name("") == ""


def test_a_run_with_no_sidecar_is_skipped_rather_than_guessed(monkeypatch):
    """``log.csv`` records no model name, so a missing sidecar means the model is unknown.

    Parsing it back out of the run_id would be guessing at a string format, and a guess that
    lands on the wrong name puts one model's numbers under another model's label.
    """
    monkeypatch.setattr(model_shelf, "_run_detail", lambda run_id: {})
    assert measured_models_for("Revenues") == []
