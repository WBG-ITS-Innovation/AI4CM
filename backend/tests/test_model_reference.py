"""The Models reference: live hyperparameters, traceable metrics, and no fabricated coverage.

Three properties that would rot silently if untested:
  * a hyperparameter transcribed instead of read would drift from the pipeline without anything
    failing;
  * a metric shown without its run_id cannot be audited, and there would be no signal that it had
    become untraceable;
  * a coverage figure attached to a point model would be fabrication, and it would look identical
    to a real one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from model_reference import (  # noqa: E402
    DESCRIPTIONS,
    PARAM_MEANING,
    build,
    live_hyperparameters,
    measured_performance,
    model_pool,
)


@pytest.fixture(scope="module")
def ref():
    return build()


# ── hyperparameters are READ, not transcribed ─────────────────────────────────

def test_hyperparameters_come_from_get_params_not_a_literal_table():
    """If a value were hardcoded, changing the pipeline would leave the page stale and silent."""
    from b_ml_pipeline import available_models

    est = available_models()["HistGBDT_L1"]
    hp = live_hyperparameters(est)
    got = {p["name"]: p["value"] for p in hp["set_by_pipeline"]}
    live = est.get_params()
    for name, shown in got.items():
        assert str(live[name]) == shown, f"{name} shown as {shown!r}, live value {live[name]!r}"


def test_a_changed_pipeline_value_moves_the_page():
    """The property that matters: mutate the estimator and the reported value follows."""
    from b_ml_pipeline import available_models

    est = available_models()["HistGBDT_L1"]
    before = {p["name"]: p["value"] for p in live_hyperparameters(est)["set_by_pipeline"]}
    assert before["l2_regularization"] == "1.0"

    est.set_params(l2_regularization=7.5)
    after = {p["name"]: p["value"] for p in live_hyperparameters(est)["set_by_pipeline"]}
    assert after["l2_regularization"] == "7.5", (
        "the reported value did not follow the estimator — it is not being read live"
    )


def test_only_pipeline_set_params_are_highlighted():
    """HistGBDT_L1 sets 3 of its ~21 params. Showing all 21 as 'configuration' would imply the
    project chose values it merely inherited."""
    from b_ml_pipeline import available_models

    hp = live_hyperparameters(available_models()["HistGBDT_L1"])
    names = {p["name"] for p in hp["set_by_pipeline"]}
    assert names == {"loss", "l2_regularization", "random_state"}, names
    assert hp["n_total"] > len(names)
    assert len(hp["library_default"]) >= 10


def test_pipeline_wrapped_models_expose_their_inner_params():
    """Ridge is inside a Pipeline, so its alpha appears as est__alpha. A naive shallow read would
    show no configuration at all for the linear models."""
    from b_ml_pipeline import available_models

    hp = live_hyperparameters(available_models()["Ridge"])
    allnames = {p["name"] for p in hp["set_by_pipeline"] + hp["library_default"]}
    assert any(n.startswith("est__") for n in allnames), sorted(allnames)[:8]


def test_every_reported_parameter_has_a_plain_language_meaning_or_says_none():
    """A parameter name with no explanation is not a reference, it is a dump."""
    pool = model_pool()
    missing = []
    for name, m in pool.items():
        for p in m["hyperparameters"].get("set_by_pipeline", []):
            if not p["controls"]:
                missing.append(f"{name}.{p['name']}")
    assert not missing, f"parameters set by the pipeline with no explanation: {missing}"


# ── every model is described ──────────────────────────────────────────────────

def test_every_model_in_the_pool_has_a_description(ref):
    undescribed = [n for n, m in ref["models"].items() if not m.get("summary")]
    assert not undescribed, undescribed
    assert len(ref["models"]) >= 13, len(ref["models"])


def test_descriptions_are_labelled_as_general_not_measured(ref):
    """A description must not read as evidence."""
    for n, m in ref["models"].items():
        assert m["description_kind"] == "general description, not a measured claim", n


def test_descriptions_make_no_numeric_performance_claim():
    """Prose is the one unverifiable content here, so it must not carry a figure that looks
    measured."""
    import re

    for name, d in DESCRIPTIONS.items():
        text = d["summary"]
        offenders = re.findall(r"\b\d+(?:\.\d+)?\s*%", text)
        assert not offenders, f"{name} description states a percentage: {offenders}"


# ── measured performance is traceable ────────────────────────────────────────

def test_every_measured_row_carries_its_run_id(ref):
    for model, rows in ref["performance"].items():
        for r in rows:
            assert r["run_id"], f"{model} has a row with no run_id"


def test_every_run_id_resolves_to_a_real_run_record(ref):
    runs = REPO / "experiments" / "runs"
    missing = [r["run_id"] for rows in ref["performance"].values() for r in rows
               if not (runs / f"{r['run_id']}.json").exists()]
    assert not missing, f"run_ids with no record: {missing[:5]}"


def test_measured_values_match_the_log_exactly(ref):
    """Nothing is recomputed on the page. A second implementation of a published number is how one
    quantity ends up with two values."""
    from experiment_log import read_log

    log = {r["run_id"]: r for r in read_log()}
    checked = 0
    for rows in ref["performance"].values():
        for r in rows:
            src = log[r["run_id"]]
            if r["mae"] is not None:
                assert abs(float(src["dev_mae"]) - r["mae"]) < 1e-6, r["run_id"]
                checked += 1
    assert checked > 50, f"only {checked} values cross-checked"


def test_window_is_derived_from_the_logged_fold_scheme(ref):
    for rows in ref["performance"].values():
        for r in rows:
            assert r["window"] in ("DEV (2024)", "TRAIN (<=2023)", "not reported")
            if r["window"] == "DEV (2024)":
                assert "dev" in r["fold_scheme"].lower()


# ── the coverage constraint: never fabricated ────────────────────────────────

def test_point_model_runs_report_no_coverage(ref):
    """Measured, not assumed: point-model runs never wrote coverage, so it must come back None.

    A coverage figure attached to a model that produced no interval would be indistinguishable
    from a real one on the page.
    """
    offenders = []
    for model, rows in ref["performance"].items():
        if "Quantile" in model or "cqr" in model.lower():
            continue
        for r in rows:
            if any(r[k] is not None for k in ("coverage_low", "coverage_mid", "coverage_high")):
                offenders.append(f"{model}/{r['run_id']}")
    assert not offenders, f"coverage fabricated for point models: {offenders[:5]}"


def test_quantile_runs_do_report_coverage(ref):
    """The complement — otherwise the test above would pass on an empty payload."""
    have = [r for model, rows in ref["performance"].items() if "cqr" in model.lower()
            for r in rows if r["coverage_high"] is not None]
    assert have, "no run reports coverage at all; the constraint test proves nothing"


def test_coverage_note_explains_the_absence(ref):
    note = ref["coverage_note"]
    assert "not reported" in note and "not backfilled" in note


# ── availability is stated, not hidden ───────────────────────────────────────

def test_unavailable_models_are_listed_with_the_missing_library():
    """A model that vanishes when a library is absent makes the pool's contents depend on the
    environment without saying so."""
    import b_ml_pipeline as bml

    pool = model_pool()
    if bml.HAVE_CATBOOST:
        assert pool["CatBoost_L1"]["available"] is True
    else:
        assert pool["CatBoost_L1"]["available"] is False
        assert pool["CatBoost_L1"]["missing_library"] == "catboost"


def test_champions_are_keyed_by_model_and_keep_a_null_approver(ref):
    for model, recs in ref["champions"].items():
        for rec in recs:
            assert rec["approved_by"] is None, f"{model} claims an approver"
            assert rec["status"].startswith("candidate")


def test_cli_emits_valid_json():
    """The frontend reads this as JSON from a subprocess; a stray print would break it."""
    import subprocess

    py = BACKEND / ".venv" / "bin" / "python"
    if not py.exists():
        pytest.skip("backend venv not present")
    out = subprocess.run([str(py), "backend/model_reference.py"], cwd=str(REPO),
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-400:]
    line = next(l for l in reversed(out.stdout.splitlines()) if l.strip().startswith("{"))
    d = json.loads(line)
    assert {"models", "performance", "champions", "coverage_note"} <= set(d)
