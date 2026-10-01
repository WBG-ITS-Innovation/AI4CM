"""An official run refuses a recipe whose declared horizon is not the validated one.

Why this file exists
--------------------
Everything that makes a forecast official was measured at ``VALIDATED_HORIZON``: the ruler,
the recipe selection and every gate. ``official_run`` already refuses a *requested* horizon
other than that one. It did not look at the horizon the *recipe* declares in its parameters
(``registry/recipes.json``, ``params.horizon``), so a recipe credentialed at ten days could be
run and published as official at five with nothing comparing the two numbers
(``docs/sessions/2026-09-30-inference-horizon-map.md``, §2.7, first bullet).

The recipe is doctored in memory. The registry on disk is never changed.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import forecast_modes  # noqa: E402
from forecast_modes import VALIDATED_HORIZON, NotOfficial, official_run, recipe_status  # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
DOCTORED_HORIZON = 10


def _doctored(monkeypatch, horizon):
    """recipe_status() answering with the real Revenues recipe at a different horizon."""
    real = recipe_status("Revenues")
    assert real["has_recipe"]
    st = copy.deepcopy(real)
    st["recipe"]["params"]["horizon"] = horizon
    monkeypatch.setattr(forecast_modes, "recipe_status", lambda target: st)
    return st


def test_a_recipe_declared_at_another_horizon_is_refused_before_any_data_is_read(
        tmp_path, monkeypatch):
    _doctored(monkeypatch, DOCTORED_HORIZON)
    missing = tmp_path / "no_such_file.csv"          # read only if the guard did not fire
    with pytest.raises(NotOfficial) as refused:
        official_run("Revenues", missing)
    assert not missing.exists()


def test_the_reason_names_both_horizons(monkeypatch, tmp_path):
    _doctored(monkeypatch, DOCTORED_HORIZON)
    with pytest.raises(NotOfficial) as refused:
        official_run("Revenues", tmp_path / "no_such_file.csv")
    reason = str(refused.value)
    assert str(DOCTORED_HORIZON) in reason and str(VALIDATED_HORIZON) in reason, reason
    assert "Revenues" in reason


def test_the_real_recipes_declare_the_validated_horizon():
    """The guard is inert on the registry as committed; this is what makes that checkable."""
    for target in ("Revenues", "Expenditure", "State budget balance"):
        st = recipe_status(target)
        assert int(st["recipe"]["params"]["horizon"]) == VALIDATED_HORIZON, target


def test_a_recipe_that_declares_no_horizon_is_not_refused_on_that_ground(monkeypatch, tmp_path):
    """Absence is not a mismatch. A recipe with no declared horizon is refused, if at all, on
    other grounds; here the data file is missing, so the failure is the ordinary one."""
    st = _doctored(monkeypatch, VALIDATED_HORIZON)
    del st["recipe"]["params"]["horizon"]
    with pytest.raises(FileNotFoundError):
        official_run("Revenues", tmp_path / "no_such_file.csv")
