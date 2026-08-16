"""The artifact contract validator, tested against deliberately corrupted artifacts.

A validator that has only ever seen clean input is an untested validator. Every check here is
exercised twice: once on an artifact that should pass, and once on the same artifact with one
specific thing broken. The corruption tests are the ones that matter — they are what prove the
check can fail.

The three defect classes are kept apart on purpose, because the response differs:

* **malformed** — unparseable or wrong-typed. The writer is broken.
* **incomplete** — parseable but a value is missing with nothing saying why. Note that absence is
  *legitimate* on this project in several places (a point model logs no coverage), so this class is
  reserved for absence that no explicit marker accounts for.
* **inconsistent** — every field is fine and they contradict each other. These are the dangerous
  ones: no per-field check finds them.
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

from artifact_validation import (  # noqa: E402
    ERROR,
    WARNING,
    ArtifactContractError,
    ValidationReport,
    validate_champion_is_in_the_pool,
    validate_coverage_block,
    validate_family_tables,
    validate_published_issue,
    validate_run,
    validate_skill_reconciles,
    validate_summary,
)


# ── fixtures: a clean run, and helpers to break one thing at a time ───────────

def _clean_summary() -> dict:
    return {
        "run_id": "2026-08-11", "schema_version": 2, "run_date": "2026-08-11",
        "target": "Revenues", "cadence": "Daily", "horizon": "5",
        # review C1: a clean run says which dataset produced it. Bare name, not a path.
        "data_file": "master_daily_clean_treasury.csv",
        "families": [
            {"name": "B_ML", "ok": True, "models": "Ridge, Lasso",
             "best_model": "Ridge (MAE 40,000,000)", "skill_pct": "20.00%",
             "run_status": "SUCCESS", "gate_passed": True, "gate_reasons": [],
             "leakage_flag": False, "shift_flag": False,
             "mae_model": 40_000_000.0, "mae_persistence": 50_000_000.0,
             "n_prediction_rows": 262},
            {"name": "A_STAT", "ok": True, "models": "ETS",
             "best_model": "ETS", "skill_pct": "n/a (not produced)",
             "run_status": "SUCCESS", "gate_passed": None, "gate_reasons": [],
             "leakage_flag": False, "shift_flag": False},
        ],
        "overall": {"families_requested": 2, "families_ok": 2, "families_gate_passed": 1,
                    "leakage_flags": 0, "shift_flags": 0, "quality_gate_failures": 0},
        "mode": "backtest",
        "freshness": {"line": "...", "stale": False, "backtest": True},
    }


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """A clean run: SUMMARY.json plus one well-formed family."""
    d = tmp_path / "2026-08-11"
    (d / "b_ml").mkdir(parents=True)
    (d / "SUMMARY.json").write_text(json.dumps(_clean_summary(), indent=2))

    origins = pd.bdate_range("2024-01-01", periods=40)
    targets = pd.bdate_range("2024-01-08", periods=40)
    pd.DataFrame({
        "origin_date": list(origins) * 2, "target_date": list(targets) * 2,
        "origin_value": 1e8, "y_true": 1.1e8, "y_pred": 1.05e8,
        "model": ["Ridge"] * 40 + ["Lasso"] * 40,
        "target": "Revenues", "horizon": 5,
    }).to_csv(d / "b_ml" / "predictions_long.csv", index=False)

    pd.DataFrame([
        {"target": "Revenues", "horizon": 5, "model": "Ridge", "MAE": 4e7, "rank": 1},
        {"target": "Revenues", "horizon": 5, "model": "Lasso", "MAE": 4.3e7, "rank": 2},
    ]).to_csv(d / "b_ml" / "leaderboard.csv", index=False)

    pd.DataFrame([
        {"target": "Revenues", "horizon": 5, "model": "Ridge", "MAE": 4e7, "RMSE": 5e7},
        {"target": "Revenues", "horizon": 5, "model": "Lasso", "MAE": 4.3e7, "RMSE": 5.2e7},
    ]).to_csv(d / "b_ml" / "metrics_long.csv", index=False)
    return d


def _write_summary(run_dir: Path, mutate) -> Path:
    d = _clean_summary()
    mutate(d)
    (run_dir / "SUMMARY.json").write_text(json.dumps(d, indent=2))
    return run_dir


def _kinds(rep: ValidationReport, kind: str) -> list[str]:
    return [f.message for f in rep.by_kind(kind)]


# ── the clean case, so the corruption cases mean something ────────────────────

def test_a_clean_run_passes_with_no_errors(run_dir):
    rep = validate_run(run_dir)
    assert rep.ok, rep.summary()
    assert rep.errors == [], rep.summary()


def test_a_clean_run_passes_even_under_strict(run_dir):
    """Strict promotes warnings to errors, so a genuinely clean run must have neither."""
    rep = validate_run(run_dir, strict=True)
    assert rep.ok, rep.summary()


# ── MALFORMED ────────────────────────────────────────────────────────────────

def test_missing_summary_is_an_error(tmp_path):
    rep = ValidationReport()
    validate_summary(tmp_path / "SUMMARY.json", rep)
    assert not rep.ok
    assert any("missing" in m for m in _kinds(rep, "malformed"))


def test_unparseable_summary_is_an_error(run_dir):
    (run_dir / "SUMMARY.json").write_text("{not json at all")
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("not valid JSON" in m for m in _kinds(rep, "malformed"))


def test_a_missing_required_key_is_an_error(run_dir):
    _write_summary(run_dir, lambda d: d.pop("target"))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("'target' absent" in m for m in _kinds(rep, "malformed"))


def test_a_non_tristate_gate_verdict_is_an_error(run_dir):
    """gate_passed must be true / false / null. "unknown" would read as truthy."""
    _write_summary(run_dir, lambda d: d["families"][0].update(gate_passed="unknown"))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("tri-state" in m for m in _kinds(rep, "malformed"))


def test_a_skill_figure_that_is_neither_number_nor_marker_is_an_error(run_dir):
    _write_summary(run_dir, lambda d: d["families"][0].update(skill_pct="pretty good"))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("neither a number" in m for m in _kinds(rep, "malformed"))


def test_an_explicit_not_applicable_marker_is_accepted(run_dir):
    """A_STAT's "n/a (not produced)" is an answer, not a defect."""
    _write_summary(run_dir, lambda d: d["families"][0].update(skill_pct="not reported"))
    rep = validate_run(run_dir)
    assert rep.ok, rep.summary()


def test_an_unreadable_csv_is_an_error(run_dir):
    (run_dir / "b_ml" / "predictions_long.csv").write_bytes(b'a,b\n"unterminated\x00\n1')
    rep = validate_run(run_dir)
    assert not rep.ok or rep.warnings, rep.summary()


def test_a_missing_required_prediction_column_is_an_error(run_dir):
    p = run_dir / "b_ml" / "predictions_long.csv"
    pd.read_csv(p).drop(columns=["target_date"]).to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("'target_date' absent" in m for m in _kinds(rep, "malformed"))


def test_a_coverage_proportion_outside_zero_one_is_an_error():
    rep = ValidationReport()
    validate_coverage_block({"coverage_p10_p90": 1.4, "coverage_nominal": 0.8}, "x", rep)
    assert not rep.ok
    assert any("must lie in [0, 1]" in m for m in _kinds(rep, "malformed"))


# ── INCOMPLETE ───────────────────────────────────────────────────────────────

def test_absent_run_id_is_a_warning_not_a_blocker(run_dir):
    """A historical artifact missing a later field is a fact about it, not a reason to refuse."""
    _write_summary(run_dir, lambda d: (d.pop("run_id"), d.pop("schema_version")))
    rep = validate_run(run_dir)
    assert rep.ok, "an old artifact must remain readable"
    assert any("run_id" in f.message for f in rep.warnings)


def test_absent_run_id_becomes_an_error_under_strict(run_dir):
    """A NEW run is held to the current contract."""
    _write_summary(run_dir, lambda d: (d.pop("run_id"), d.pop("schema_version")))
    rep = validate_run(run_dir, strict=True)
    assert not rep.ok
    assert any("run_id" in m for m in _kinds(rep, "incomplete"))


def test_an_empty_families_list_is_an_error(run_dir):
    _write_summary(run_dir, lambda d: d.update(families=[]))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("empty" in m for m in _kinds(rep, "incomplete"))


def test_a_leaderboard_with_no_rows_is_an_error(run_dir):
    p = run_dir / "b_ml" / "leaderboard.csv"
    pd.read_csv(p).iloc[0:0].to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("no model" in m for m in _kinds(rep, "incomplete"))


def test_an_entirely_empty_metric_column_is_reported(run_dir):
    """The absent-vs-failed distinction: an all-null column expresses neither."""
    p = run_dir / "b_ml" / "metrics_long.csv"
    df = pd.read_csv(p)
    df["MAE_skill_vs_Ops"] = None
    df.to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert any("MAE_skill_vs_Ops" in f.message for f in rep.findings), rep.summary()
    assert not validate_run(run_dir, strict=True).ok


# ── INCONSISTENT — the dangerous class ───────────────────────────────────────

def test_a_withheld_family_with_no_reason_is_an_error(run_dir):
    """A model withheld without a stated reason is unactionable."""
    _write_summary(run_dir, lambda d: d["families"][0].update(gate_passed=False,
                                                             gate_reasons=[]))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("must say why" in m for m in _kinds(rep, "inconsistent"))


def test_a_passing_gate_with_reasons_is_an_error(run_dir):
    _write_summary(run_dir, lambda d: d["families"][0].update(
        gate_passed=True, gate_reasons=["coverage 43.2% outside [70%, 90%]"]))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("disagree" in m for m in _kinds(rep, "inconsistent"))


def test_the_overall_block_must_reconcile_with_the_families(run_dir):
    """`overall` is derived, so it can contradict what it derives from."""
    _write_summary(run_dir, lambda d: d["overall"].update(families_gate_passed=2))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("families_gate_passed is 2" in m for m in _kinds(rep, "inconsistent"))


def test_duplicate_family_names_are_an_error(run_dir):
    def dup(d):
        d["families"].append(dict(d["families"][0]))
        d["overall"].update(families_requested=3, families_ok=3, families_gate_passed=2)
    _write_summary(run_dir, dup)
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("duplicate family names" in m for m in _kinds(rep, "inconsistent"))


def test_a_coverage_value_with_no_nominal_is_an_error():
    """Audit field #2, enforced at the contract boundary."""
    rep = ValidationReport()
    validate_coverage_block({"coverage_p10_p90": 0.78}, "x", rep)
    assert not rep.ok
    assert any("no numeric 'coverage_nominal'" in m for m in _kinds(rep, "inconsistent"))


def test_a_coverage_nominal_that_contradicts_its_quantiles_is_an_error():
    rep = ValidationReport()
    validate_coverage_block({"coverage_p5_p95": 0.9, "coverage_nominal": 0.80,
                             "coverage_lower_quantile": 0.05,
                             "coverage_upper_quantile": 0.95}, "x", rep)
    assert not rep.ok
    assert any("imply 0.9" in m for m in _kinds(rep, "inconsistent"))


def test_a_consistent_coverage_block_passes():
    rep = ValidationReport()
    validate_coverage_block({"coverage_p5_p95": 0.91, "coverage_nominal": 0.90,
                             "coverage_lower_quantile": 0.05,
                             "coverage_upper_quantile": 0.95}, "x", rep)
    assert rep.ok, rep.summary()


def test_a_skill_figure_that_does_not_follow_from_its_maes_is_an_error():
    """The WS2 harness bug: a non-canonical ruler inflated skill while both MAEs sat beside it."""
    rep = ValidationReport()
    validate_skill_reconciles(
        {"skill_pct": "32.62%", "mae_model": 40_000_000.0,
         "mae_persistence": 56_600_000.0}, "x", rep)         # implies 29.33%
    assert not rep.ok
    msg = _kinds(rep, "inconsistent")[0]
    assert "29.3" in msg and "32.62" in msg, msg
    assert "different baseline" in msg


def test_a_skill_figure_that_does_reconcile_passes():
    rep = ValidationReport()
    validate_skill_reconciles(
        {"skill_pct": "20.00%", "mae_model": 40_000_000.0,
         "mae_persistence": 50_000_000.0}, "x", rep)
    assert rep.ok, rep.summary()


def test_a_non_positive_persistence_baseline_is_an_error():
    rep = ValidationReport()
    validate_skill_reconciles({"skill_pct": 10.0, "mae_model": 1.0,
                               "mae_persistence": 0.0}, "x", rep)
    assert not rep.ok


def test_a_skill_figure_over_zero_prediction_rows_is_an_error(run_dir):
    """X11: a baseline computed over nothing is not a baseline."""
    _write_summary(run_dir, lambda d: d["families"][0].update(n_prediction_rows=0))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("n_prediction_rows=0" in m for m in _kinds(rep, "inconsistent"))


def test_baseline_without_predictions_contradicting_a_skill_figure_is_an_error(run_dir):
    _write_summary(run_dir,
                   lambda d: d["families"][0].update(baseline_without_predictions=True))
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("zero prediction rows" in m for m in _kinds(rep, "inconsistent"))


def test_a_champion_absent_from_the_pool_is_an_error():
    rep = ValidationReport()
    validate_champion_is_in_the_pool("NoSuchModel (MAE 1)", "x", rep, pool=["Ridge"])
    assert not rep.ok
    assert any("not in the model pool" in m for m in _kinds(rep, "inconsistent"))


def test_a_real_champion_is_accepted_against_the_real_pool():
    """Every family's champion label must resolve against model_pool() -- this is what caught
    C_DL's "MLP" being in no enumerable pool."""
    rep = ValidationReport()
    for name in ("Ridge (MAE 40,000,000)", "MLP (MAE 47,217,031)", "ETS",
                 "GBQuantile (MAE 33,964,365)", "LightGBM_L1"):
        validate_champion_is_in_the_pool(name, "x", rep)
    assert rep.ok, rep.summary()


def test_a_baseline_label_is_not_treated_as_a_missing_model():
    rep = ValidationReport()
    validate_champion_is_in_the_pool("⚡ Persistence (baseline)", "x", rep, pool=["Ridge"])
    assert rep.ok, rep.summary()


def test_an_origin_at_or_after_its_target_date_is_an_error(run_dir):
    p = run_dir / "b_ml" / "predictions_long.csv"
    df = pd.read_csv(p)
    df.loc[0, "target_date"] = df.loc[0, "origin_date"]
    df.to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert not rep.ok
    assert any("origin_date >= target_date" in m for m in _kinds(rep, "inconsistent"))


def test_a_partially_identified_leaderboard_is_an_error(run_dir):
    """The real A_STAT defect: identity columns on the baseline row and not the winner."""
    p = run_dir / "b_ml" / "leaderboard.csv"
    df = pd.read_csv(p)
    df.loc[0, "target"] = None
    df.to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert not rep.ok
    msg = " ".join(_kinds(rep, "inconsistent"))
    assert "populated on 1 of 2 rows" in msg, msg


def test_a_leaderboard_model_missing_from_predictions_is_reported(run_dir):
    p = run_dir / "b_ml" / "leaderboard.csv"
    df = pd.read_csv(p)
    df.loc[len(df)] = {"target": "Revenues", "horizon": 5, "model": "Ghost",
                       "MAE": 1e7, "rank": 0}
    df.to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert any("Ghost" in f.message for f in rep.findings), rep.summary()
    assert not validate_run(run_dir, strict=True).ok


def test_a_decorated_model_name_used_as_a_join_key_is_reported(run_dir):
    p = run_dir / "b_ml" / "leaderboard.csv"
    df = pd.read_csv(p)
    df.loc[0, "model"] = "⚡ Ridge"
    df.to_csv(p, index=False)
    rep = validate_run(run_dir)
    assert any("decorated model name" in f.message for f in rep.findings), rep.summary()


# ── published issues ─────────────────────────────────────────────────────────

def _published(tmp_path, **over) -> Path:
    d = tmp_path / "published" / "2025-08-06"
    d.mkdir(parents=True)
    fc = pd.DataFrame([{"target": "Revenues", "horizon": h, "origin_date": "2025-08-06",
                        "origin_value": 1e8, "target_date": "2025-08-13",
                        "p10": 9e7, "p50": 1e8, "p90": 1.1e8} for h in range(1, 6)])
    for k, v in over.items():
        fc[k] = v
    fc.to_csv(d / "forecast.csv", index=False)
    for name in ("provenance.json", "manifest.json", "gates.json"):
        (d / name).write_text("{}")
    return d


def test_a_clean_published_issue_passes(tmp_path):
    rep = ValidationReport()
    validate_published_issue(_published(tmp_path), rep)
    assert rep.ok, rep.summary()


def test_a_published_forecast_carrying_truth_is_an_error(tmp_path):
    """A forecast issued before its truth existed cannot have a y_true column."""
    rep = ValidationReport()
    validate_published_issue(_published(tmp_path, y_true=1.05e8), rep)
    assert not rep.ok
    assert any("y_true" in m for m in _kinds(rep, "inconsistent"))


def test_crossed_quantiles_in_a_published_forecast_are_an_error(tmp_path):
    d = _published(tmp_path)
    fc = pd.read_csv(d / "forecast.csv")
    fc.loc[0, "p10"] = 2e8                      # above p50
    fc.to_csv(d / "forecast.csv", index=False)
    rep = ValidationReport()
    validate_published_issue(d, rep)
    assert not rep.ok
    assert any("crossed quantiles" in m for m in _kinds(rep, "inconsistent"))


def test_a_missing_provenance_file_is_an_error(tmp_path):
    d = _published(tmp_path)
    (d / "provenance.json").unlink()
    rep = ValidationReport()
    validate_published_issue(d, rep)
    assert not rep.ok
    assert any("provenance.json absent" in m for m in _kinds(rep, "incomplete"))


def test_absent_estimators_are_a_warning_pruned_ones_stay_a_warning(tmp_path):
    """"Cannot be re-derived" is legitimate; it must be distinguishable from a broken manifest."""
    d = _published(tmp_path)
    rep = ValidationReport()
    validate_published_issue(d, rep)
    assert rep.ok
    assert any("cannot be re-derived" in f.message for f in rep.warnings)

    est = d / "estimators"
    est.mkdir()
    (est / "manifest.json").write_text(json.dumps({
        "retention": {"pruned": True, "pruned_at": "2026-08-11"},
        "estimators": [{"fit_id": "x", "file": "r/h5_point.joblib", "sha256": "a" * 64}]}))
    rep2 = ValidationReport()
    validate_published_issue(d, rep2)
    assert rep2.ok, rep2.summary()


def test_missing_blobs_with_no_prune_marker_are_an_error(tmp_path):
    """Unexplained absence: the manifest lists blobs, they are gone, nothing says why."""
    d = _published(tmp_path)
    est = d / "estimators"
    est.mkdir()
    (est / "manifest.json").write_text(json.dumps({
        "retention": {"pruned": False},
        "estimators": [{"fit_id": "x", "file": "r/h5_point.joblib", "sha256": "a" * 64}]}))
    rep = ValidationReport()
    validate_published_issue(d, rep)
    assert not rep.ok
    assert any("absence is unexplained" in m for m in _kinds(rep, "inconsistent"))


def test_an_estimator_without_a_digest_is_an_error(tmp_path):
    d = _published(tmp_path)
    est = d / "estimators" / "r"
    est.mkdir(parents=True)
    (est / "h5_point.joblib").write_bytes(b"x")
    (d / "estimators" / "manifest.json").write_text(json.dumps({
        "retention": {"pruned": False},
        "estimators": [{"fit_id": "x", "file": "r/h5_point.joblib"}]}))
    rep = ValidationReport()
    validate_published_issue(d, rep)
    assert not rep.ok
    assert any("no sha256" in m for m in _kinds(rep, "incomplete"))


# ── the gate itself ──────────────────────────────────────────────────────────

def test_raise_if_invalid_raises_and_names_every_finding(run_dir):
    _write_summary(run_dir, lambda d: d["families"][0].update(gate_passed=False,
                                                             gate_reasons=[]))
    rep = validate_run(run_dir)
    with pytest.raises(ArtifactContractError, match="must say why"):
        rep.raise_if_invalid()


def test_raise_if_invalid_is_silent_on_a_clean_run(run_dir):
    validate_run(run_dir).raise_if_invalid()


def test_the_pipeline_wires_the_gate_in_before_publication():
    src = (REPO / "scripts" / "daily_summary.py").read_text()
    assert "from artifact_validation import validate_run" in src
    assert "must not be published" in src
    assert "--no-validate" in src, "the escape hatch must exist and be explicit"


def test_the_cli_returns_nonzero_on_a_bad_artifact(run_dir):
    import subprocess
    _write_summary(run_dir, lambda d: d["families"][0].update(gate_passed=False,
                                                             gate_reasons=[]))
    r = subprocess.run([sys.executable, str(BACKEND / "artifact_validation.py"), str(run_dir)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "must say why" in r.stdout


def test_the_cli_returns_zero_on_a_clean_artifact(run_dir):
    import subprocess
    r = subprocess.run([sys.executable, str(BACKEND / "artifact_validation.py"), str(run_dir)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ── the real committed artifacts ─────────────────────────────────────────────

REAL_RUN = BACKEND / "forecast_runs" / "2026-08-04"

#: Both tests below call ``validate_run(REAL_RUN)``, which reads the row-level artifacts -- and
#: those are gitignored, so they exist only where the run was made. Gating on ``REAL_RUN.exists()``
#: was right only while the whole directory was absent. Now that SUMMARY.json is tracked again the
#: directory exists in every clone, so a directory check would let these run against artifacts that
#: are not there and fail for a reason that is not a defect. Gate on what is actually read, the way
#: the reference-run test below already does.
REAL_RUN_ROWS = REAL_RUN / "a_stat" / "leaderboard.csv"
_NO_ROWS = "row-level artifacts are gitignored; present only where the run was made"


@pytest.mark.skipif(not REAL_RUN_ROWS.exists(), reason=_NO_ROWS)
def test_the_real_run_is_readable_and_its_findings_are_recorded():
    """Documents where the shipped artifacts actually stand.

    They are NOT clean, and the contract document says so. This test exists to make the set of
    known defects explicit: if a fix lands, this fails and the document gets updated.

    It has already done that job once. The `run_id` / `schema_version` warnings were pinned here
    until 2026-08-12, when the run's SUMMARY.json was regenerated by the current writer to carry
    `data_file` and `client_framing`. Those two warnings are consequently gone, and this test
    failing is what forced this note.
    """
    rep = validate_run(REAL_RUN)
    msgs = " | ".join(f.message for f in rep.findings)
    assert "run_id" not in msgs, (
        "run_id is missing again -- SUMMARY.json was rewritten by an older writer")
    assert any("Persistence (baseline)" in f.message for f in rep.findings), (
        "the derived baseline row is still absent from predictions_long")

    # The one remaining ERROR class: A_STAT's partially-identified leaderboard. The writer is
    # fixed; this artifact predates the fix and was deliberately not regenerated. If this stops
    # failing, the run was regenerated -- update docs/AGENT_ARTIFACT_CONTRACT.md §8.
    ident = [f for f in rep.errors if "populated on 1 of 2 rows" in f.message]
    assert len(ident) == 3, (
        f"expected the 3 known a_stat identity errors, got {len(ident)}: {rep.summary()}")
    assert len(rep.errors) == 3, (
        f"a NEW error class appeared in the committed artifacts:\n{rep.summary()}")

    # ...and the two fields the Agent was blocked on now genuinely reach the artifact.
    summary = json.loads((REAL_RUN / "SUMMARY.json").read_text())
    assert summary["data_file"] == "master_daily_clean_treasury.csv"
    assert "compete on each target" in summary["client_framing"]
    assert summary["model_composition"]["counts"]["machine-learning models"] == 13


@pytest.mark.skipif(not REAL_RUN_ROWS.exists(), reason=_NO_ROWS)
def test_the_remaining_errors_are_a_pre_existing_csv_defect_not_a_regression():
    """The verdict on why `daily_summary` exits 2 on this run, held as an assertion.

    Every remaining ERROR is on ``a_stat/leaderboard.csv`` -- a file written 2026-08-04, seven days
    before the writer bug was fixed (03ad619, 2026-08-11). The 2026-08-12 SUMMARY.json regeneration
    could not have caused them and did not: replaying the pre-regeneration shape yields the **same
    three errors with the same messages**, and the regeneration only removed warnings.

    Pinned so "the validator fails on this run" can never be mistaken for a regression introduced
    by a later change, and so that regenerating the a_stat family forces this note to be revisited.
    """
    import shutil
    import tempfile

    rep = validate_run(REAL_RUN)

    # 1. Every error is in the one CSV. Nothing the regeneration touched is implicated.
    assert {f.artifact for f in rep.errors} == {"a_stat/leaderboard.csv"}, rep.summary()
    assert not any("SUMMARY" in f.artifact for f in rep.errors), rep.summary()

    # 2. Stripping the five regenerated keys reproduces the pre-regeneration artifact, and the
    #    errors are unchanged -- so they pre-date it.
    tmp = Path(tempfile.mkdtemp()) / REAL_RUN.name
    shutil.copytree(REAL_RUN, tmp)
    p = tmp / "SUMMARY.json"
    d = json.loads(p.read_text())
    for k in ("run_id", "schema_version", "data_file", "client_framing", "model_composition"):
        d.pop(k, None)
    p.write_text(json.dumps(d, indent=2))

    before = validate_run(tmp)
    assert [f.message for f in before.errors] == [f.message for f in rep.errors], (
        "the regeneration changed the error set, so it is implicated after all")
    assert len(before.warnings) > len(rep.warnings), (
        "the regeneration should have removed warnings (run_id, schema_version, data_file)")

    # 3. The writer is fixed: the same metrics_long now yields a fully-identified leaderboard.
    import numpy as np
    metr = pd.read_csv(REAL_RUN / "a_stat" / "metrics_long.csv")
    lb = (metr.groupby("model", as_index=False)[["MAE", "RMSE"]].mean()
          .assign(target="Revenues", horizon=5, cadence="Daily")
          .assign(rank=lambda x: np.arange(1, len(x) + 1)))
    assert all(lb[c].notna().all() for c in ("target", "horizon", "cadence")), (
        "the fixed writer still leaves identity columns blank")
    assert lb["RMSE"].notna().all(), (
        "the fixed writer should also recover RMSE, which the on-disk CSV lost")


# ── the reference artifact ────────────────────────────────────────────────────

REFERENCE_RUN = BACKEND / "forecast_runs" / "2026-08-12"


@pytest.mark.skipif(not (REFERENCE_RUN / "SUMMARY.json").exists(),
                    reason="no reference run committed")
def test_the_reference_summary_is_contract_clean_and_carries_every_field():
    """`2026-08-12` is the reference artifact: the first summary with no contract errors.

    Only `SUMMARY.json` and `SUMMARY.txt` are tracked -- the row-level artifacts stay ignored
    because they carry Treasury predictions -- so this validates the summary alone rather than the
    whole run. That is what a clone actually has, which is the point of having a reference at all.

    Committed after `2026-08-04` needed its summary regenerated and still fails on a pre-fix a_stat
    leaderboard. This one was produced entirely by the current writers, so it is what a consumer
    should be pointed at.
    """
    rep = ValidationReport()
    validate_summary(REFERENCE_RUN / "SUMMARY.json", rep, strict=True)
    assert rep.ok, rep.summary()
    assert rep.findings == [], rep.summary()

    d = json.loads((REFERENCE_RUN / "SUMMARY.json").read_text())
    assert d["run_id"] == "2026-08-12"
    assert d["schema_version"] == 2
    assert d["data_file"] == "master_daily_clean_treasury.csv"
    assert "compete on each target" in d["client_framing"]
    assert d["model_composition"]["counts"]["machine-learning models"] == 13
    assert d["model_composition"]["promoted_outside_champion_pool"] == []


@pytest.mark.skipif(not (REFERENCE_RUN / "a_stat" / "leaderboard.csv").exists(),
                    reason="row-level artifacts are gitignored; present only where the run was made")
def test_the_reference_a_stat_leaderboard_is_fully_identified():
    """The writer fix, demonstrated on a run made after it rather than argued from a diff.

    `2026-08-04` has identity columns on 1 of 2 rows and an all-null RMSE. This run, same writer,
    same columns, has both -- which is what settles "pre-existing defect, not a regression".
    """
    d = pd.read_csv(REFERENCE_RUN / "a_stat" / "leaderboard.csv")
    for col in ("target", "horizon", "cadence"):
        assert d[col].notna().all(), f"{col} is not populated on every row"
    assert d["RMSE"].notna().any(), "RMSE is all-null again"
