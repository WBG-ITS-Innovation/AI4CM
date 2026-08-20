"""The model composition, derived — and every champion-crowning family enumerable.

Two defects of the same shape sit behind this file.

**Enumerability.** `DESCRIPTIONS` once described `ETS` and `Theta` while no pool exposed
them, so the Models page could not look up a model that had won. That was fixed for A_STAT
by giving it a `registry_models()`. It then recurred in C_DL, and was fixed the same way in
item 6 (`c_dl_registry.py`, split out so the catalogue reads without torch). Fixing the same
defect twice in two families is a sign it will happen in a third, so
`test_every_champion_crowning_family_is_enumerable` derives the family list from
`run_daily_forecast.sh` — the actual source of truth for what runs — and fails if a family
can be run, and can therefore publish a `best_model`, without a registry behind it.

**The composition.** The sentence we give a client was a literal string in a pinned test.
When C_DL became enumerable the pool went 23 → 28 and the sentence did not move, so we were
describing a pool that no longer matched the registry and nothing detected it. The counts
are now derived from `model_pool()` and the sentence is written from them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from model_reference import (                                       # noqa: E402
    CATEGORY_ORDER, CHAMPION_POOL_CATEGORY, COMPETING_CATEGORIES,
    client_category, client_framing, composition, model_pool,
)

RUNNER = REPO / "scripts" / "run_daily_forecast.sh"


@pytest.fixture(scope="module")
def pool():
    return model_pool()


@pytest.fixture(scope="module")
def comp(pool):
    return composition(pool)


def runnable_families() -> list[str]:
    """The families `run_daily_forecast.sh` runs by default.

    Read from the script rather than restated here: a family added to the runner must not be
    able to slip past this file by not being listed in it.
    """
    text = RUNNER.read_text()
    m = re.search(r'^FAMILIES="\$\{FAMILIES:-([^}"]+)\}"', text, re.MULTILINE)
    assert m, "could not read the default FAMILIES from run_daily_forecast.sh"
    families = m.group(1).split()
    assert families, "the runner's default FAMILIES is empty"
    return families


# ── enumerability ───────────────────────────────────────────────────────────────────────

def test_the_runner_family_list_is_readable():
    assert runnable_families() == ["A_STAT", "B_ML", "E_QUANTILE", "C_DL"]


def test_every_champion_crowning_family_is_enumerable(pool):
    """A family that can publish a `best_model` must have a lookupable model catalogue.

    `daily_summary.py` writes `best_model` for every family it is asked to summarise, so
    every runnable family can crown one. If such a family contributes nothing to
    `model_pool()`, a consumer reading its leaderboard cannot look up what won — which is
    exactly the defect fixed once for A_STAT and again for C_DL.
    """
    by_pipeline: dict[str, list[str]] = {}
    for name, entry in pool.items():
        by_pipeline.setdefault(entry["pipeline"], []).append(name)

    missing = [f for f in runnable_families() if not by_pipeline.get(f)]
    assert not missing, (
        f"{missing} can be run by run_daily_forecast.sh, and therefore can publish a "
        f"best_model, but contribute no entries to model_pool(). Give each a "
        f"registry_models() (see c_dl_registry.py) so a consumer can look up what won."
    )


def test_no_pool_entry_comes_from_a_family_the_runner_cannot_run(pool):
    """The converse: the pool must not advertise a family that never runs."""
    runnable = set(runnable_families())
    stray = sorted({e["pipeline"] for e in pool.values()} - runnable)
    assert not stray, f"model_pool() offers {stray}, which run_daily_forecast.sh cannot run"


def test_every_enumerable_model_has_a_name_and_a_pipeline(pool):
    for name, entry in pool.items():
        assert entry.get("name") == name
        assert entry.get("pipeline"), f"{name} has no pipeline"


def test_a_family_with_no_client_category_is_a_hard_error():
    """The guard that stops a fifth family being silently uncounted.

    This is the anti-staleness property: a new pipeline cannot reach the pool without
    somebody deciding, in code, whether its models compete.
    """
    with pytest.raises(ValueError, match="no client-facing category"):
        client_category({"name": "SOMETHING", "pipeline": "F_NEW"})


def test_every_real_pool_entry_has_a_client_category(pool):
    for name, entry in pool.items():
        cat = client_category(entry)
        assert cat in CATEGORY_ORDER, f"{name} -> {cat!r}"


# ── the composition, derived ────────────────────────────────────────────────────────────

def test_composition_covers_every_pool_entry(pool, comp):
    assert comp["total"] == len(pool)
    assert sum(comp["counts"].values()) == len(pool)


def test_composition_members_partition_the_pool(pool, comp):
    seen = [n for names in comp["members"].values() for n in names]
    assert sorted(seen) == sorted(pool)
    assert len(seen) == len(set(seen)), "a model was counted in two categories"


def test_the_derived_counts_are_what_we_tell_a_client(comp):
    """The numbers in the client sentence, derived rather than asserted.

    Adding a model changes one of these and fails here, which is the whole point: the
    previous pinned string went stale silently when C_DL was added.
    """
    # 2026-08-19: eleven models registered across three families, none of them measured. The
    # machine-learning count went 15 -> 21, statistical 5 -> 7, quantile 3 -> 6. What did NOT
    # change is `evaluated_total`, still 8, which is the number that says whether any of this
    # made the project stronger.
    assert comp["counts"] == {
        "machine-learning models": 21,
        "deep-learning models": 5,
        "statistical models": 7,
        "quantile methods": 6,
        "reference baselines": 3,
    }, comp["counts"]


def test_baselines_and_interval_methods_do_not_compete(comp):
    """The ruler is not a rival, and an interval method is not a point forecaster."""
    assert "reference baselines" not in COMPETING_CATEGORIES
    assert "quantile methods" not in COMPETING_CATEGORIES
    assert comp["competing_total"] == comp["total"] - comp["counts"]["quantile methods"] \
        - comp["counts"]["reference baselines"]


def test_the_client_sentence_is_written_from_the_counts(comp):
    sentence = client_framing()
    for cat in COMPETING_CATEGORIES:
        assert f"{comp['counts'][cat]} {cat}" in sentence
    assert f"{comp['counts']['quantile methods']} quantile methods" in sentence
    assert f"{comp['counts']['reference baselines']} further entries" in sentence
    assert "not competitors" in sentence


def test_the_client_sentence_names_every_family_that_competes(comp):
    """C_DL was absent from this sentence for the whole of item 6.

    It runs daily, publishes a best_model, and can pass the gate, so a sentence about what
    competes on each target that omits it is not true.
    """
    sentence = client_framing()
    assert "deep-learning models" in sentence
    assert "5 deep-learning models" in sentence


def test_the_sentence_never_offers_a_single_headline_total(comp):
    """Summing the categories presents the ruler and the interval methods as rivals."""
    sentence = client_framing()
    for wrong in (str(comp["total"]), str(comp["competing_total"])):
        assert f"{wrong} models compete" not in sentence
        assert f"a total of {wrong}" not in sentence


# ── the two meanings of "champion" ──────────────────────────────────────────────────────

def test_the_registry_champion_pool_is_the_machine_learning_models(comp):
    assert comp["champion_pool_category"] == CHAMPION_POOL_CATEGORY
    # 21 since 2026-08-19. Widening the shelf widens the pool a recipe MAY draw from, and that
    # is not the same as widening what it may draw: an untested model is not gate-eligible, so
    # nothing here can reach a champion slot without a recorded measurement first.
    assert comp["champion_pool_size"] == 21
    assert comp["champion_pool"] == sorted(comp["members"]["machine-learning models"])


def test_every_model_the_registry_promotes_is_in_that_pool(comp):
    """If a recipe ever promotes a model from another family, the sentence "the
    champion-eligible pool is the machine-learning models" stops being true, and this
    fails before a client is told it."""
    assert comp["promoted_by_registry"], "the registry promotes no model at all"
    assert comp["promoted_outside_champion_pool"] == [], (
        f"{comp['promoted_outside_champion_pool']} are promoted by registry/recipes.json but "
        f"are not in the {CHAMPION_POOL_CATEGORY} pool"
    )


def test_the_daily_best_model_families_are_broader_than_the_champion_pool(comp):
    """The word "champion" means two things, and they are not the same size.

    `registry/recipes.json` selects a point model from the 13. `daily_summary.py` writes a
    `best_model` for all four families. A consumer ranking families — which is what the
    Agent does — is choosing across four families, not across the registry pool, and the
    contract has to say so.
    """
    assert comp["daily_best_model_families"] == ["A_STAT", "B_ML", "C_DL", "E_QUANTILE"]
    assert len(comp["daily_best_model_families"]) > 1
    assert comp["champion_pool_category"] == "machine-learning models"


# ── the shelf is not the same thing as the evidence ─────────────────────────────────────

def test_the_sentence_says_how_many_have_a_recorded_result(comp):
    """Counting the shelf and counting the evidence are different claims.

    Before this clause the sentence said "13 machine-learning models compete on each
    target", which reads as thirteen measured contenders. Measured at the time, 8 of the 28
    non-baseline entries had a row in experiments/log.csv and 20 had none. A shelf is worth
    having and worth describing; it is not a body of evidence and must not read as one.
    """
    sentence = client_framing()
    assert f"{comp['evaluated_total']} have a recorded result" in sentence
    assert f"{comp['untested_total']} are registered candidates" in sentence


def test_evaluated_and_untested_partition_everything_except_the_baselines(comp):
    """A baseline is the ruler, so asking whether it was measured is the wrong question."""
    counted = comp["evaluated_total"] + comp["untested_total"]
    assert counted == comp["total"] - comp["counts"]["reference baselines"]
    assert not set(comp["evaluated"]) & set(comp["untested"])


def test_every_model_the_registry_promotes_has_a_recorded_result(comp):
    """A champion with nothing in the ledger would have credentials nobody can check.

    registry.verify_against_log already checks that each recipe's quoted metrics match its
    logged run. This asserts the same thing from the other direction, at the level of the
    model rather than the recipe.
    """
    for model in comp["promoted_by_registry"]:
        assert model in comp["evaluated"], (
            f"{model} is promoted by registry/recipes.json with no recorded result")


def test_a_newly_registered_candidate_is_untested_until_it_is_run(comp):
    """The three candidates added in the MVP consolidation, and both CatBoost entries.

    If any of these ever shows as evaluated without a run being logged, the status is being
    declared somewhere rather than derived.
    """
    for candidate in ("Huber", "GBDT_L1", "ETS_DAMPED", "CatBoost_L1", "CatBoost_Quantile"):
        assert candidate in comp["untested"], f"{candidate} claims a result it does not have"
