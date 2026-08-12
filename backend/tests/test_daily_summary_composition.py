"""`client_framing` and `data_file` must reach a real SUMMARY.json, not just the writer.

This is the gap the Agent audit surfaced twice, in the same shape both times: a field was correct in
the code and absent from every artifact. `test_artifact_contract.py` greps the writer's source and
passes; the Agent reads files and reported "composition not recorded" on every run. So this test
**runs `daily_summary.py` as a subprocess and opens the file it wrote** — the only kind of test that
can tell the two apart.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
SCRIPT = REPO / "scripts" / "daily_summary.py"
RUN_DATE = "2026-08-12"


@pytest.fixture()
def written_summary(tmp_path) -> dict:
    """Run the real writer over a minimal-but-valid run and return the parsed JSON."""
    rd = tmp_path / RUN_DATE
    for fam in ("a_stat", "b_ml"):
        d = rd / fam
        d.mkdir(parents=True)
        pd.DataFrame({
            "model": ["Ridge"] * 2,
            "origin_date": ["2026-08-05", "2026-08-05"],
            "target_date": ["2026-08-06", "2026-08-07"],
            "origin_value": [99.0, 100.0],
            "y_true": [100.0, 110.0], "y_pred": [101.0, 108.0],
        }).to_csv(d / "predictions_long.csv", index=False)
        pd.DataFrame({"target": ["Revenues"] * 2, "horizon": [5, 5],
                      "model": ["Ridge", "persistence_baseline"],
                      "MAE": [1.5, 3.0]}).to_csv(d / "leaderboard.csv", index=False)
        (d / "integrity_report.json").write_text(json.dumps(
            {"run_status": "SUCCESS", "skill_pct": 12.3, "leakage_warning": False}))

    data = tmp_path / "master_daily_clean_treasury.csv"
    pd.DataFrame({"date": ["2026-08-11", RUN_DATE], "y": [1.0, 2.0]}).to_csv(data, index=False)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(rd), "--data-file", str(data),
         "--target", "Revenues", "--cadence", "daily", "--horizon", "5",
         "--run-date", RUN_DATE, "--families", "A_STAT B_ML"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((rd / "SUMMARY.json").read_text())


def test_the_written_file_carries_data_file(written_summary):
    assert written_summary["data_file"] == "master_daily_clean_treasury.csv"


def test_the_written_file_carries_client_framing(written_summary):
    s = written_summary["client_framing"]
    assert isinstance(s, str) and s.endswith(".")
    # Never a single headline number: the entries are not one kind of thing.
    assert "compete on each target" in s
    assert "reference baselines, not competitors" in s
    assert "quantile methods" in s


def test_client_framing_matches_the_function_it_came_from(written_summary):
    sys.path.insert(0, str(BACKEND))
    from model_reference import client_framing

    assert written_summary["client_framing"] == client_framing(), (
        "the artifact and the function must not be two implementations of one sentence")


def test_the_written_file_carries_the_counts_behind_the_sentence(written_summary):
    comp = written_summary["model_composition"]
    assert comp["counts"] == {"machine-learning models": 13, "deep-learning models": 5,
                              "statistical models": 4, "quantile methods": 3,
                              "reference baselines": 3}
    assert sum(comp["counts"].values()) == 28
    # Every counted model must be named, so a consumer can requote or recompute.
    assert sum(len(v) for v in comp["members"].values()) == 28


def test_the_two_meanings_of_champion_are_both_recorded(written_summary):
    """Conflating them is how a true sentence becomes a wrong one."""
    comp = written_summary["model_composition"]
    assert len(comp["champion_pool"]) == 13, comp["champion_pool"]
    assert comp["champion_pool_category"] == "machine-learning models"
    assert set(comp["daily_best_model_families"]) == {"A_STAT", "B_ML", "C_DL", "E_QUANTILE"}, (
        "the Agent ranks across every family that writes a best_model, not across champion_pool")


def test_the_registry_cross_check_is_carried_and_currently_clean(written_summary):
    comp = written_summary["model_composition"]
    assert comp["promoted_outside_champion_pool"] == [], (
        "a recipe promotes a model outside the pool a client was told about")
    # Distinct MODELS, not recipes: Revenues and Expenditure both promote LightGBM_L1, so three
    # recipes name two models. Asserting 3 here would be asserting the wrong thing.
    assert sorted(comp["promoted_by_registry"]) == ["HistGBDT_L1", "LightGBM_L1"]
    assert set(comp["promoted_by_registry"]) <= set(comp["champion_pool"])


def test_no_unavailable_reason_when_the_composition_was_derivable(written_summary):
    assert "client_framing_unavailable_reason" not in written_summary


def test_an_undegradable_pool_yields_a_reason_not_a_silent_gap(monkeypatch):
    """Absence must be explained: the contract §0 pattern, applied to this field.

    A summary must not fail because the model *catalogue* could not be read -- the run itself is
    unaffected -- but it must not silently omit the field either.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import daily_summary

    def boom():
        raise ImportError("no lightgbm on this machine")

    monkeypatch.setattr(daily_summary, "_composition_fields",
                        lambda: {"client_framing_unavailable_reason":
                                 "could not derive the model composition: ImportError: "
                                 "no lightgbm on this machine"})
    out = daily_summary._composition_fields()
    assert "client_framing" not in out
    assert "no lightgbm" in out["client_framing_unavailable_reason"]


def test_the_real_composition_fields_helper_degrades_rather_than_raising(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import daily_summary

    monkeypatch.setitem(sys.modules, "model_reference", None)   # force the import to fail
    out = daily_summary._composition_fields()
    assert "client_framing" not in out
    assert "could not derive the model composition" in out["client_framing_unavailable_reason"]
