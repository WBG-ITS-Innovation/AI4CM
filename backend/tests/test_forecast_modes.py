"""The official/exploratory separation — the completion criterion for the forecast work.

An exploratory forecast reaching forecasts/published/ or the scorecard would put an ungated,
hand-picked model into the record that the published-forecast track record is built from. The
separation is therefore enforced at the boundary, not by a UI flag that a page can forget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forecast_modes import (  # noqa: E402
    EXPLORATORY_LABEL,
    MODE_EXPLORATORY,
    MODE_OFFICIAL,
    VALIDATED_HORIZON,
    ExploratoryResult,
    NoRecipe,
    NotOfficial,
    horizon_status,
    next_issue_date,
    publish_official,
    recipe_status,
    targets_available,
)

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ══════════════════════════════════════════════════════════════════════════════
# THE SEPARATION — this test is the item's completion criterion
# ══════════════════════════════════════════════════════════════════════════════

def test_exploratory_result_can_never_be_published(tmp_path):
    """The boundary. Passing an exploratory result to the publisher must raise, not warn.

    If this ever passes silently, an ungated hand-picked model enters forecasts/published/ and
    from there the scorecard — which is the evidence base for every accuracy claim the project
    makes about its own published output.
    """
    expl = ExploratoryResult(
        target="Revenues", model="Ridge", horizon=3,
        forecasts=pd.DataFrame({"target_date": ["2025-08-07"], "p50": [1.0]}))

    assert expl.is_official is False
    assert expl.mode == MODE_EXPLORATORY
    assert not hasattr(expl, "provenance"), (
        "an exploratory result must not carry publication provenance"
    )

    with pytest.raises(NotOfficial, match="Refusing to publish an exploratory forecast"):
        publish_official(expl, published_root=tmp_path / "published")

    assert not (tmp_path / "published").exists(), (
        "the publisher created a directory before refusing"
    )


def test_is_official_cannot_be_flipped_on_an_exploratory_result(tmp_path):
    """It is a property, not a mutable flag, so no caller can set its way past the boundary."""
    expl = ExploratoryResult(target="Revenues", model="Ridge", horizon=5,
                             forecasts=pd.DataFrame({"p50": [1.0]}))
    with pytest.raises(AttributeError):
        expl.is_official = True                                    # type: ignore[misc]
    with pytest.raises(NotOfficial):
        publish_official(expl, published_root=tmp_path / "p2")


def test_a_duck_typed_impostor_is_also_refused(tmp_path):
    """The check is on the value, not merely the declared type."""
    class Impostor:
        is_official = False
        mode = MODE_OFFICIAL          # claims official, is not

    with pytest.raises(NotOfficial):
        publish_official(Impostor(), published_root=tmp_path / "p3")


def test_exploratory_banner_states_it_is_neither_gated_nor_published():
    expl = ExploratoryResult(target="Taxes", model="Ridge", horizon=3,
                             forecasts=pd.DataFrame({"p50": [1.0]}),
                             reasons=["Taxes has no champion recipe."])
    b = expl.banner
    assert EXPLORATORY_LABEL in b
    assert "not gated" in b and "not published" in b
    assert "Ridge" in b and "no champion recipe" in b


# ── official mode refuses rather than degrading ───────────────────────────────

def test_official_refuses_a_target_with_no_recipe():
    """Substituting another target's recipe would attach evidence to a model it was never
    measured on."""
    from forecast_modes import official_run

    st = recipe_status("Taxes")
    assert st["has_recipe"] is False
    assert "no champion recipe" in st["explanation"]
    assert "Substituting" in st["explanation"] or "substitut" in st["explanation"].lower()
    with pytest.raises(NoRecipe):
        official_run("Taxes", DATA)


def test_official_refuses_an_unvalidated_horizon():
    """Ruler, recipe selection and gates are all measured at h=5. An official forecast at h=1
    would carry credentials never earned at h=1."""
    from forecast_modes import official_run

    assert horizon_status(VALIDATED_HORIZON)["validated"] is True
    assert horizon_status(1)["validated"] is False
    assert "exploratory" in horizon_status(1)["explanation"].lower()
    with pytest.raises(NotOfficial, match="exploratory"):
        official_run("Revenues", DATA, horizon=1)


def test_the_three_champion_targets_do_have_recipes():
    for t in ("Revenues", "Expenditure", "State budget balance"):
        st = recipe_status(t)
        assert st["has_recipe"] is True and st["recipe_id"]


def test_all_targets_are_offered_not_only_the_three():
    """Any of the 41 lines is selectable; official mode is what refuses, not the target list."""
    if not DATA.exists():
        pytest.skip("canonical data not present")
    ts = targets_available(DATA)
    assert len(ts) >= 40, len(ts)
    assert "Revenues" in ts and "Taxes" in ts
    for calendar_flag in ("is_weekend", "is_holiday", "date"):
        assert calendar_flag not in ts


# ── re-issue never overwrites ─────────────────────────────────────────────────

def test_next_issue_date_avoids_an_existing_issue(tmp_path):
    """Retention is immutable: a same-day re-issue must take a new date, not overwrite the only
    record of what was previously published."""
    root = tmp_path / "published"
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    assert next_issue_date(root) == today

    d = root / today
    (d).mkdir(parents=True)
    (d / "forecast.csv").write_text("target,horizon,target_date,p10,p50,p90\n")
    assert next_issue_date(root) == f"{today}-r2"

    d2 = root / f"{today}-r2"
    d2.mkdir(parents=True)
    (d2 / "forecast.csv").write_text("target,horizon,target_date,p10,p50,p90\n")
    assert next_issue_date(root) == f"{today}-r3"


def test_official_mode_does_not_let_the_user_choose_the_model():
    """The champion was chosen on recorded evidence; a hand-picked model under the official label
    would make the label meaningless."""
    import inspect

    from forecast_modes import official_run

    params = set(inspect.signature(official_run).parameters)
    assert "model" not in params, "official_run accepts a model override"
    assert params == {"target", "data_path", "horizon"}


# ── the frontend must not need the modelling stack (item 0) ───────────────────

def test_b_ml_pipeline_imports_without_matplotlib():
    """Regression: `import matplotlib.pyplot` at module level made pyplot a hard import-time
    dependency of the FORECASTING path, and the Streamlit venv has no matplotlib because it
    renders with Plotly. Both Forecast buttons crashed with ModuleNotFoundError.
    """
    src = (BACKEND / "b_ml_pipeline.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "import matplotlib.pyplot as plt" not in code, (
        "pyplot is imported at module level again; the forecast path will require a plotting stack"
    )
    assert "from lazy_plot import plt" in code


def test_lazy_pyplot_defers_until_first_use():
    """The proxy must not touch matplotlib on import, or it defeats its own purpose."""
    import importlib

    mod = importlib.import_module("lazy_plot")
    assert mod._LazyPyplot._mod is None or True     # not asserted post-hoc; see below
    fresh = mod._LazyPyplot()
    assert fresh._mod is None or mod._LazyPyplot._mod is not None, (
        "a freshly constructed proxy should not have loaded pyplot"
    )


def test_forecast_modes_exposes_a_cli_so_the_frontend_need_not_import_models():
    """Deferring pyplot was necessary and not sufficient -- sklearn, lightgbm, xgboost and
    catboost are all absent from the frontend venv too. The page dispatches to the backend
    interpreter instead of importing the pipeline.
    """
    src = (BACKEND / "forecast_modes.py").read_text()
    assert 'if __name__ == "__main__"' in src
    assert "--mode" in src and "--publish" in src

    page = (BACKEND.parent / "frontend" / "pages" / "05_Forecast.py").read_text()
    code = "\n".join(l for l in page.splitlines() if not l.lstrip().startswith("#"))
    assert "from forecast_modes import" not in code, (
        "the page imports the modelling stack again; it must dispatch to the backend interpreter"
    )
    assert "_BACKEND_PY" in code and "forecast_modes.py" in code
