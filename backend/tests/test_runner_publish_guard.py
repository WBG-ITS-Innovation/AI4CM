"""The runner's ``--publish`` applies the same verdict refusal as the page path.

Why this file exists
--------------------
Two publish paths existed. The Forecast page's path, ``forecast_modes.publish_official``,
refuses a target whose registry verdict is ``withheld``: a documented trivial benchmark is more
accurate than the model, so publishing its numbers invites a worse decision than publishing
nothing. The runner, ``run_forward_forecast.py --publish``, called ``published_forecasts.publish``
directly and applied no such check. Measured in the disposable clone on 2026-09-30: the page
refused Expenditure and State budget balance; the runner published both
(``docs/sessions/2026-09-30-scoring-loop-audit.md``, finding F3).

The rule is one guard, shared. A second copy of the wording in the runner would be a second
place for the two paths to drift apart, which is how the gap opened in the first place.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import published_forecasts  # noqa: E402
import run_forward_forecast as rff  # noqa: E402
from forecast_modes import NotOfficial, OfficialResult, publish_official  # noqa: E402
from registry import load_registry  # noqa: E402

TARGETS = ("Revenues", "Expenditure", "State budget balance")
ORIGIN = "2025-08-06"

#: The sentence that makes a refusal a refusal. Asserted to live in exactly one module.
REFUSAL_PHRASE = "documented trivial benchmark is more accurate"


def _rows(target: str) -> pd.DataFrame:
    dates = pd.bdate_range(pd.Timestamp(ORIGIN) + pd.Timedelta(days=1), periods=5)
    return pd.DataFrame({
        "target": target, "horizon": range(1, 6), "origin_date": ORIGIN,
        "origin_value": 1.0, "target_date": [str(d.date()) for d in dates],
        "p10": 1.0, "p50": 2.0, "p90": 3.0, "point_model": "LightGBM_L1",
        "interval_model": "GBQuantile", "target_transform": "raw",
    })


def _forecasts() -> pd.DataFrame:
    return pd.concat([_rows(t) for t in TARGETS], ignore_index=True)


def _prov() -> dict:
    return {
        "run_kind": "forward_forecast",
        "data": {"name": "synthetic.csv", "sha256": "abc", "n_rows": 1,
                 "latest_data_date": ORIGIN},
        "code": {"git_sha": "deadbeef"}, "calendar_version": "cal1",
        "test_window_touched": False,
        "recipes": [{"target": t, "recipe_id": f"{t}-v1", "point_model": "LightGBM_L1",
                     "target_transform": "raw"} for t in TARGETS],
    }


def _gates() -> dict:
    return {f"{t}-v1": {"target": t, "gates": {}, "status": "candidate -- pre-tuning"}
            for t in TARGETS}


def _registry_with(verdicts: dict) -> dict:
    """The real registry with the named targets' verdicts doctored. Never written to disk."""
    reg = copy.deepcopy(load_registry())
    for r in reg["recipes"]:
        if r["target"] in verdicts:
            r["publication"]["verdict"] = verdicts[r["target"]]
    return reg


def _official(target: str) -> OfficialResult:
    return OfficialResult(target=target, recipe_id=f"{target}-v1", model="LightGBM_L1",
                          horizon=5, forecasts=_rows(target), provenance={"recipes": []},
                          gates={f"{target}-v1": {"target": target, "gates": {}}})


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """The runner's staging and publishing redirected away from the real store and vault."""
    monkeypatch.setattr(rff, "DEFAULT_OUT", tmp_path / "forward" / "latest")

    def _never(*_a, **_k):
        raise AssertionError("the vault must not be touched when published_root is redirected")

    monkeypatch.setattr(published_forecasts, "retain_to_vault", _never)
    return tmp_path / "published"


# ── the runner refuses what the page refuses ─────────────────────────────────

def test_the_runner_refuses_the_targets_the_page_refuses(redirected):
    """Today's registry: Revenues publishable, the other two withheld (recipes.json)."""
    out = rff._publish_and_retain(_forecasts(), _prov(), _gates(), [], published_root=redirected)

    assert {t for t, _ in out["refused"]} == {"Expenditure", "State budget balance"}
    assert out["published"] == ["Revenues"]
    fc = pd.read_csv(Path(out["dest"]) / "forecast.csv")
    assert set(fc["target"]) == {"Revenues"}, "a refused target's rows reached the record"


def test_the_refusal_reason_is_the_page_paths_wording(redirected, tmp_path):
    """Same guard, same words: a reader of either log sees one explanation."""
    with pytest.raises(NotOfficial) as page:
        publish_official(_official("Expenditure"), published_root=tmp_path / "page")
    out = rff._publish_and_retain(_forecasts(), _prov(), _gates(), [], published_root=redirected)
    runner = dict(out["refused"])["Expenditure"]
    assert runner == str(page.value)
    assert REFUSAL_PHRASE in runner


def test_withheld_as_forecast_still_publishes_through_the_runner(redirected):
    """'withheld_as_forecast' keeps the numbers as the best estimate; only the claim is withheld."""
    reg = _registry_with({"Expenditure": "withheld_as_forecast"})
    out = rff._publish_and_retain(_forecasts(), _prov(), _gates(), [],
                                  published_root=redirected, registry=reg)

    assert set(out["published"]) == {"Revenues", "Expenditure"}
    assert {t for t, _ in out["refused"]} == {"State budget balance"}
    fc = pd.read_csv(Path(out["dest"]) / "forecast.csv")
    assert set(fc["target"]) == {"Revenues", "Expenditure"}


def test_nothing_is_published_when_every_target_is_refused(redirected):
    reg = _registry_with({t: "withheld" for t in TARGETS})
    out = rff._publish_and_retain(_forecasts(), _prov(), _gates(), [],
                                  published_root=redirected, registry=reg)

    assert out["dest"] is None and out["published"] == []
    assert {t for t, _ in out["refused"]} == set(TARGETS)
    assert not redirected.exists() or not any(redirected.iterdir()), (
        "an issue directory was created for a run with nothing to publish")


def test_the_published_manifest_names_only_the_targets_that_were_published(redirected):
    """The manifest's recipe list is read by the scorer; a refused recipe must not be in it."""
    import json

    out = rff._publish_and_retain(_forecasts(), _prov(), _gates(), [], published_root=redirected)
    manifest = json.loads((Path(out["dest"]) / "manifest.json").read_text())
    assert [r["target"] for r in manifest["recipes"]] == ["Revenues"]
    assert manifest["targets"] == ["Revenues"]
    gates = json.loads((Path(out["dest"]) / "gates.json").read_text())
    assert {g["target"] for g in gates.values()} == {"Revenues"}


# ── one guard, not two ───────────────────────────────────────────────────────

def test_the_refusal_wording_lives_in_one_place():
    """A copy of the sentence in the runner is a second place for the two paths to drift."""
    counts = {p.name: p.read_text(encoding="utf-8").count(REFUSAL_PHRASE)
              for p in BACKEND.glob("*.py")}
    assert counts.get("forecast_modes.py") == 1, counts
    assert sum(counts.values()) == 1, f"the refusal wording appears in more than one module: {counts}"


def test_the_runner_does_not_publish_the_shared_forward_directory_directly():
    """The direct call is what bypassed the verdict. Checked at the source."""
    src = (BACKEND / "run_forward_forecast.py").read_text(encoding="utf-8")
    assert "publish(DEFAULT_OUT" not in src, (
        "run_forward_forecast.py still publishes the whole forward directory unguarded")
    assert "refuse_withheld" in src, "the runner does not call the shared guard"
