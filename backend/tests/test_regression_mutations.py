"""Mutation tests: reintroduce each fixed bug and require the existing guard to catch it.

A regression test that only exercises the *fixed* behaviour proves the code works today. It does
not prove the test would notice if the bug came back -- and a test that cannot fail is worse than
no test, because it reads as coverage.

So each block here does the opposite: it puts the original defect back and asserts that the guard
which is supposed to catch it **actually fails**. Where the guard is an existing test function,
that test is imported and called directly under the mutation; ``pytest.raises(AssertionError)``
around it is the proof that it discriminates. Where the guard is production logic, the mutated and
real implementations are compared on the same input and required to disagree.

The six bugs, all previously fixed and all previously covered only by behaviour tests:

1. shuffled-target sentinel inversion (a LOW ratio reported as leakage)
2. zero train MAE from ExtraTrees memorisation crowning a memoriser
3. collapsed E_QUANTILE intervals passing a skill-only gate
4. inverted leakage-sentinel semantics via the deprecated alias
5. the duplicated persistence baseline shadowing the shared one
6. ``alignment_ok`` written as an unconditional literal

Mutations are applied with ``monkeypatch``, so they are undone at the end of every test. Nothing
here leaves the mutation in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast_integrity as fi  # noqa: E402


def _assert_test_fails(fn, *args, **kwargs):
    """Call an existing test under a mutation and require it to fail.

    ``AssertionError`` is the expected outcome. Anything else -- including passing -- means the
    test does not discriminate against this bug, which is the thing being measured.
    """
    try:
        fn(*args, **kwargs)
    except AssertionError:
        return
    pytest.fail(f"{fn.__module__}.{fn.__name__} PASSED with the bug reintroduced -- "
                f"it does not actually guard against this regression")


# ── 1. shuffled-target sentinel inversion ─────────────────────────────────────

def _pre_m5_inverted_sentinel(*args, **kwargs):
    """The original bug: a LOW shuffled/real ratio reported as leakage.

    Backwards. Leakage makes the *real*-target error implausibly small, which drives the ratio UP.
    A low ratio means shuffling barely hurt, i.e. the features never carried signal. On the real
    Revenues run this printed "WITHHELD -- leakage flag raised" at ratio 0.83.
    """
    out = dict(fi.signal_sentinel(*args, **kwargs))
    ratio = out.get("shuffled_to_normal_ratio")
    if ratio is not None and np.isfinite(ratio) and ratio < fi.MIN_SIGNAL_RATIO:
        out["leakage_warning"] = True
        out["signal_verdict"] = f"LEAKAGE SUSPECTED (ratio {ratio:.2f})"
    return out


def test_mutation_sentinel_inversion_is_caught(monkeypatch):
    import test_failure_mode_distinctness as tfmd

    monkeypatch.setattr(tfmd, "signal_sentinel", _pre_m5_inverted_sentinel)
    _assert_test_fails(tfmd.test_no_signal_drives_the_ratio_DOWN_and_is_never_called_leakage)


def test_mutation_sentinel_inversion_is_caught_by_the_semantics_suite(monkeypatch):
    import test_signal_sentinel_semantics as tsss

    monkeypatch.setattr(tsss, "signal_sentinel", _pre_m5_inverted_sentinel)
    _assert_test_fails(tsss.test_no_signal_when_features_are_noise)


def test_mutation_sentinel_asserting_leakage_at_any_ratio_is_caught(monkeypatch):
    """The stronger invariant: this instrument must never claim leakage, high ratio or low."""
    import test_failure_mode_distinctness as tfmd

    def always_claims_leakage(*a, **k):
        out = dict(fi.signal_sentinel(*a, **k))
        out["leakage_warning"] = True
        return out

    monkeypatch.setattr(tfmd, "signal_sentinel", always_claims_leakage)
    _assert_test_fails(tfmd.test_the_sentinel_never_asserts_leakage_at_any_ratio)


def test_mutation_lowering_the_signal_threshold_to_pass_everything_is_caught(monkeypatch):
    """Relaxing MIN_SIGNAL_RATIO to 1.0 would let a no-signal feature set read as signal."""
    import test_failure_mode_distinctness as tfmd

    def permissive(*a, **k):
        out = dict(fi.signal_sentinel(*a, **k))
        r = out.get("shuffled_to_normal_ratio")
        if r is not None and np.isfinite(r):
            out["signal_detected"] = bool(r >= 1.0)   # was MIN_SIGNAL_RATIO = 1.5
        return out

    monkeypatch.setattr(tfmd, "signal_sentinel", permissive)
    _assert_test_fails(tfmd.test_no_signal_drives_the_ratio_DOWN_and_is_never_called_leakage)


# ── 2. zero train MAE from ExtraTrees memorisation ────────────────────────────

def test_mutation_ignoring_the_overfit_ratio_crowns_a_memoriser(monkeypatch):
    """The bug: pick the lowest DEV MAE and never look at train/dev.

    ExtraTrees with unlimited depth reaches train MAE 0 -- an infinite ratio -- and posts the best
    DEV MAE by memorising. Selection has to exclude it.
    """
    import test_b_ml_overfitting as tbo

    def lowest_mae_wins(leaderboard, overfit_ratios=None, *a, **k):
        lb = leaderboard[~leaderboard["model"].str.contains("aseline|ersistence", regex=True)]
        return str(lb.sort_values("MAE")["model"].iloc[0]), []

    monkeypatch.setattr(tbo, "select_best_model", lowest_mae_wins)
    _assert_test_fails(tbo.test_memorising_model_is_excluded)
    _assert_test_fails(tbo.test_overfitting_model_is_not_crowned_best)


def test_mutation_treating_an_infinite_ratio_as_missing_is_caught(monkeypatch):
    """The subtler original defect: train MAE of 0 recorded as *no* ratio.

    A missing ratio is correctly treated as "not measured, do not blame the model" -- so silently
    turning an infinite ratio into a missing one converts the loudest possible overfit signal into
    a free pass.
    """
    import test_b_ml_overfitting as tbo
    from b_ml_pipeline import select_best_model as real_select

    def drops_infinities(leaderboard, overfit_ratios=None, *a, **k):
        cleaned = {m: r for m, r in (overfit_ratios or {}).items() if np.isfinite(r)}
        return real_select(leaderboard, cleaned, *a, **k)

    monkeypatch.setattr(tbo, "select_best_model", drops_infinities)
    _assert_test_fails(tbo.test_memorising_model_is_excluded)


def test_the_real_selector_excludes_an_infinite_ratio():
    """The positive half, so the mutation above is measured against a working baseline."""
    from b_ml_pipeline import select_best_model

    lb = pd.DataFrame([
        {"model": "ExtraTrees", "MAE": 39_000_000.0},
        {"model": "Ridge", "MAE": 43_500_000.0},
    ])
    best, excluded = select_best_model(lb, {"ExtraTrees": float("inf"), "Ridge": 1.49})
    assert best == "Ridge" and "ExtraTrees" in excluded


# ── 3. collapsed E_QUANTILE intervals ────────────────────────────────────────

def _skill_only_gate(skill_pct, coverage, min_skill=5.0, **kwargs):
    """The pre-M-2 gate: skill only. Coverage was reported but never gated on."""
    reasons = []
    if not np.isfinite(skill_pct) or skill_pct < min_skill:
        reasons.append(f"skill {skill_pct:.2f}% < {min_skill:.1f}% required")
    return (len(reasons) == 0, reasons)


def test_mutation_a_skill_only_gate_passes_a_collapsed_interval():
    """A collapsed interval is p10 == p50 == p90: an interval of zero width.

    Coverage goes to ~0 while median skill can stay excellent, so a skill-only gate publishes a
    forecast whose "80% interval" contains the truth almost never.
    """
    from e_quantile_daily_pipeline import quantile_quality_gate

    collapsed_coverage, strong_skill = 0.0, 45.0

    mutated_passed, _ = _skill_only_gate(strong_skill, collapsed_coverage)
    real_passed, real_reasons = quantile_quality_gate(strong_skill, collapsed_coverage)

    assert mutated_passed is True, "the mutation must reproduce the original permissive behaviour"
    assert real_passed is False, (
        "the real gate let a zero-width interval through -- coverage is not being gated")
    assert any("coverage" in r for r in real_reasons), (
        f"a collapsed interval must fail for a COVERAGE reason, not a skill one: {real_reasons}")
    assert not any("skill" in r for r in real_reasons), (
        f"skill was excellent; naming it would be the wrong reason: {real_reasons}")


def test_mutation_a_collapsed_interval_is_detectable_in_the_predictions():
    """Independently of the gate: zero width must be visible in the artifact itself."""
    n = 200
    rng = np.random.default_rng(0)
    y = pd.Series(1e8 + rng.normal(0, 1e6, n))
    point = y + rng.normal(0, 1e5, n)

    collapsed = pd.DataFrame({"y_true": y, "yhat_p10": point,
                              "yhat_p50": point, "yhat_p90": point})
    healthy = pd.DataFrame({"y_true": y, "yhat_p10": point - 2e6,
                            "yhat_p50": point, "yhat_p90": point + 2e6})

    def coverage(df):
        return float(np.mean((df["y_true"] >= df["yhat_p10"])
                             & (df["y_true"] <= df["yhat_p90"])))

    assert (collapsed["yhat_p90"] - collapsed["yhat_p10"]).abs().max() == 0.0
    assert coverage(collapsed) < 0.05, "a zero-width interval cannot cover the truth"
    assert coverage(healthy) > 0.5


def test_mutation_widening_the_coverage_band_to_pass_anything_is_caught():
    """Relaxing the band rather than the term: [0, 1] accepts every coverage value."""
    from e_quantile_daily_pipeline import quantile_quality_gate

    permissive, _ = quantile_quality_gate(45.0, 0.0, coverage_band=(0.0, 1.0))
    real, reasons = quantile_quality_gate(45.0, 0.0)
    assert permissive is True and real is False, reasons


# ── 4. inverted leakage-sentinel semantics (the deprecated alias) ─────────────

def test_mutation_the_alias_claiming_leakage_is_caught(monkeypatch):
    """``leakage_sentinel`` is only a name. It must behave exactly like ``signal_sentinel``."""
    import test_signal_sentinel_semantics as tsss

    def alias_that_claims_leakage(*a, **k):
        out = dict(fi.signal_sentinel(*a, **k))
        out["leakage_warning"] = True
        return out

    monkeypatch.setattr(tsss, "leakage_sentinel", alias_that_claims_leakage)
    _assert_test_fails(tsss.test_deprecated_name_still_works)


def test_the_alias_is_the_same_object_not_a_copy():
    """A copy is a second implementation waiting to diverge."""
    from preprocessing.integrity import leakage_sentinel as shim_alias

    assert fi.leakage_sentinel is not fi.signal_sentinel, (
        "the alias is a thin wrapper by design, so it should not BE the same function object")
    assert shim_alias is fi.leakage_sentinel, "the shim re-exports rather than re-implements"

    # ...and it must agree on real data, which is what actually matters.
    rng = np.random.default_rng(7)
    n = 300
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 1e8)
    label = y.shift(-5)
    X = pd.DataFrame({"lag_1": y.shift(1),
                      "rm_7": y.rolling(7, min_periods=1).mean().shift(1)})
    ok = X.notna().all(axis=1) & label.notna()
    X, label = X[ok], label[ok]
    k = int(len(X) * 0.7)
    args = (X.iloc[:k], label.iloc[:k], X.iloc[k:], label.iloc[k:])

    a = fi.signal_sentinel(*args, horizon=5)
    b = fi.leakage_sentinel(*args, horizon=5)
    assert a["shuffled_to_normal_ratio"] == b["shuffled_to_normal_ratio"]
    assert a["leakage_warning"] is False and b["leakage_warning"] is False


# ── 5. the duplicated persistence baseline ───────────────────────────────────

def test_mutation_a_shim_level_copy_of_the_baseline_is_caught(monkeypatch):
    """The bug: `preprocessing.integrity` carried its own baseline and won the merge.

    ``b_ml_pipeline`` did ``integrity_report.update(legacy_report)``, so the duplicate's number
    reached integrity_report.json, the Dashboard, the daily summary and the backtest report --
    while the tests guarded the shared function nobody was publishing.
    """
    import preprocessing.integrity as legacy
    import test_published_baseline_is_shared as tpbs

    def divergent_baseline(df, *a, **k):
        # lag-1 instead of the h-step origin: plausible, different, and would silently move
        # every skill figure in the project.
        y = np.asarray(df["y_true"], dtype=float)
        return {"mae_persistence": float(np.mean(np.abs(np.diff(y)))),
                "n": len(y), "definition": "lag-1 (WRONG: not the h-step ruler)"}

    monkeypatch.setattr(legacy, "compute_persistence_baseline", divergent_baseline)
    _assert_test_fails(tpbs.test_the_deprecated_module_still_re_exports_the_sentinel)


def test_mutation_resurrecting_a_retired_duplicate_is_caught(monkeypatch):
    import preprocessing.integrity as legacy
    import test_published_baseline_is_shared as tpbs

    monkeypatch.setattr(legacy, "compute_baselines", lambda *a, **k: {}, raising=False)
    _assert_test_fails(tpbs.test_the_duplicate_persistence_implementation_is_gone)


def test_there_is_exactly_one_persistence_baseline_definition():
    """A repo-level guard: a second `def compute_persistence_baseline` anywhere is the bug."""
    defs = []
    for p in sorted(BACKEND.rglob("*.py")):
        if "tests" in p.parts or ".venv" in p.parts or "forecast_runs" in p.parts:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("def compute_persistence_baseline"):
                defs.append(f"{p.relative_to(REPO)}:{i}")
    assert defs == ["backend/forecast_integrity.py:261"], (
        f"the h-step ruler must have exactly one definition; found {defs}")


# ── 6. the alignment_ok literal ──────────────────────────────────────────────

def test_mutation_an_always_true_alignment_check_is_caught(monkeypatch):
    """The bug: ``"alignment_ok": True`` written as a literal with no check behind it."""
    import test_dl_alignment_integrity as tdai

    monkeypatch.setattr(tdai, "validate_alignment_step_based",
                        lambda *a, **k: {"alignment_ok": True, "n_misaligned": 0,
                                         "misaligned_examples": []})
    _assert_test_fails(tdai.test_deliberately_misaligned_predictions_yield_alignment_ok_false)


def test_mutation_reinstating_the_literal_in_the_source_is_caught(monkeypatch, tmp_path):
    """The source-level guard, mutated by feeding it a file that contains the literal."""
    import test_dl_alignment_integrity as tdai

    src = (BACKEND / "c_dl_pipeline.py").read_text()
    (tmp_path / "c_dl_pipeline.py").write_text(src + '\n_bad = {"alignment_ok": True}\n')

    # The guard reads BACKEND / "c_dl_pipeline.py", so pointing BACKEND at a directory holding a
    # mutated copy is how the bug gets reintroduced without touching the real file.
    monkeypatch.setattr(tdai, "BACKEND", tmp_path)
    _assert_test_fails(tdai.test_c_dl_no_longer_writes_an_unconditional_true)


# ── 7. the fourth verdict collapsing into the generic one (item 5) ───────────

def test_mutation_folding_coverage_into_the_generic_gate_reason_is_caught(monkeypatch):
    """The pre-item-5 behaviour: a coverage failure arrives only as "quality gate failed"."""
    import daily_summary
    import test_failure_mode_distinctness as tfmd

    monkeypatch.setattr(daily_summary, "_coverage_failure_reasons", lambda report: [])
    _assert_test_fails(
        tfmd.test_coverage_failure_is_its_own_verdict_not_a_generic_gate_failure)
    _assert_test_fails(tfmd.test_all_four_conditions_yield_four_separate_reasons)


def test_mutation_assuming_a_nominal_80_percent_is_caught(monkeypatch):
    """Audit field #2 at the verdict layer: assuming the level withholds a good 90% interval."""
    import daily_summary
    import test_failure_mode_distinctness as tfmd

    real = daily_summary._coverage_failure_reasons

    def ignores_the_recorded_level(report):
        stripped = {k: v for k, v in report.items()
                    if k not in ("coverage_nominal", "coverage_band")}
        return real(stripped)

    monkeypatch.setattr(daily_summary, "_coverage_failure_reasons", ignores_the_recorded_level)
    _assert_test_fails(tfmd.test_the_nominal_level_is_read_as_data_not_assumed)


def test_mutation_a_hardcoded_coverage_band_is_caught(monkeypatch):
    """Audit field #2 at the gate layer, on production code rather than a test."""
    from e_quantile_daily_pipeline import quantile_quality_gate

    # A correctly calibrated 90% interval, measured at 91.0%.
    assumed_80, reasons_80 = quantile_quality_gate(30.0, 0.910)
    own_level, _ = quantile_quality_gate(30.0, 0.910, nominal=0.90)
    assert own_level is True, "a 91% coverage on a 90% interval must pass"
    assert assumed_80 is False and "nominal 80%" in reasons_80[0], (
        "assuming 80% must be what fails it -- that is the bug being guarded")

    # ...and the converse: a badly calibrated 90% interval must not be excused.
    bad_own, bad_reasons = quantile_quality_gate(30.0, 0.720, nominal=0.90)
    bad_assumed, _ = quantile_quality_gate(30.0, 0.720)
    assert bad_own is False and "nominal 90%" in bad_reasons[0]
    assert bad_assumed is True, "the assumed band would have passed a broken 90% interval"


def test_mutation_emitting_the_legacy_coverage_key_for_other_alphas_is_caught():
    """The key name is a claim about the alphas, so it must not survive a reconfiguration."""
    from e_quantile_daily_pipeline import (LEGACY_COVERAGE_KEY, CoverageLevelMismatch,
                                           assert_coverage_key_matches_alphas, emit_coverage)

    into = {}
    emit_coverage(into, 0.88, (0.05, 0.50, 0.95))
    assert into["coverage_key"] == "coverage_p5_p95"
    assert into["coverage_nominal"] == 0.9
    assert LEGACY_COVERAGE_KEY not in into, (
        "a 90% coverage figure was published under a key that says 80%")
    assert "legacy_coverage_key_omitted" in into, "the omission must be recorded, not silent"

    with pytest.raises(CoverageLevelMismatch, match="mislabelled measurement"):
        assert_coverage_key_matches_alphas(LEGACY_COVERAGE_KEY, (0.05, 0.50, 0.95))

    # The default configuration still emits it, so nothing existing moved.
    ok = {}
    emit_coverage(ok, 0.78, (0.10, 0.50, 0.90))
    assert ok[LEGACY_COVERAGE_KEY] == 0.78 and ok["coverage_nominal"] == 0.8


# ── 7b. the persistence-mimicry gap, pinned rather than forgotten ────────────

def _copy_frame(y: np.ndarray, shift: int) -> pd.DataFrame:
    """A forecast that is exactly ``y`` delayed by ``shift`` steps."""
    idx = pd.bdate_range("2024-01-01", periods=len(y))
    return pd.DataFrame({"date": idx, "model": "FakeModel", "y_true": y,
                         "y_pred": pd.Series(y).shift(shift).values}).dropna()


def _level_series(n: int = 300) -> np.ndarray:
    """A random-walk level -- what a stock target looks like."""
    return np.cumsum(np.random.default_rng(0).normal(0, 1, n)) + 1e8


def _flow_series(n: int = 300) -> np.ndarray:
    """Near-white noise -- what Revenues and Expenditure look like."""
    return np.random.default_rng(0).normal(5e7, 1e7, n)


def _flagged(df: pd.DataFrame) -> bool:
    return bool(fi.detect_lagged_copy(df)["per_model"][0]["flagged"])


def test_the_horizon_aware_diagnostic_catches_a_perfect_h_step_copy():
    """The detector that does work at the production horizon, on the harder (level) case."""
    df = _copy_frame(_level_series(), 5)
    s = fi.shift_diagnostic_horizon_aware(df["y_true"].values, df["y_pred"].values, 5)
    assert s["best_shift"] == -5
    assert "persistence-like" in str(s["interpretation"]).lower()


def test_detect_lagged_copy_has_two_measured_blind_spots():
    """KNOWN GAPS, pinned with the numbers that produce them.

    Flagging needs BOTH ``(corr_best - corr_at_0) > 0.05`` AND ``mae_pred >= 0.99 * mae_lag1``.
    Each condition fails on a different real target shape:

    * **A lag-1 copy of a LEVEL series is missed.** A random walk correlates ~0.98 with itself at
      shift 0, so the correlation margin is only ~0.018 -- under the 0.05 requirement.
    * **An h=5 copy of a FLOW series is missed.** A flow has no autocorrelation, so shifting it
      makes the forecast genuinely worse than lag-1 (ratio ~0.979 < 0.99) and the second condition
      fails.

    Neither is a live hole for B_ML / C_DL / E_QUANTILE: ``daily_summary.family_shift_flag`` ORs
    this with the pipeline's horizon-aware diagnostic, which catches the h-step case robustly
    (test above). It IS a live hole for a family that writes no shift fields, where this is the
    only detector running.

    If this test fails, a gap has closed -- update it and reports/gate_audit.md.
    """
    assert not _flagged(_copy_frame(_level_series(), 1)), (
        "detect_lagged_copy now catches a lag-1 copy on a level series -- gap closed, update this")
    assert not _flagged(_copy_frame(_flow_series(), 5)), (
        "detect_lagged_copy now catches an h=5 copy on a flow -- gap closed, update this")

    # Positive controls: it is narrow, not broken.
    assert _flagged(_copy_frame(_flow_series(), 1)), (
        "a lag-1 copy of a flow is squarely what this detector is for")
    assert _flagged(_copy_frame(_level_series(), 5)), (
        "an h=5 copy of a level series is caught -- but see the audit: by a 0.01 margin")


def test_a_family_writing_no_shift_fields_raises_no_persistence_flag():
    """Why the gap above matters: no fields means the pipeline detector never runs."""
    from daily_summary import pipeline_shift

    _, flag = pipeline_shift({"run_status": "SUCCESS"})
    assert flag is False, (
        "a family that never measured shift must not be blamed for it -- but it also means "
        "detect_lagged_copy is the only detector, and it misses an h-step copy")


# ── 8. the model-count composition (the open question) ───────────────────────

def test_the_model_counts_are_pinned_so_a_headline_number_cannot_drift():
    """31 enumerated: 15 B_ML + 3 E_QUANTILE + 8 A_STAT (5 stat, 3 baselines) + 5 C_DL.

    Pinned because the page presents a count to a client. If a model is added the number must
    change deliberately, with the composition still stated, rather than silently.

    Item 6 raised this from 16 to 28. A_STAT was added because ETS and Theta were described but
    listed nowhere; C_DL was added because the new artifact validator errored on a real published
    champion ("MLP") that no enumerable pool contained. The deliberate change is the point of this
    test existing.

    The MVP consolidation raised it to 31: Huber and GBDT_L1 in B_ML and ETS_DAMPED in A_STAT,
    all three registered as candidates with no recorded result. Which is why the count alone is
    no longer the whole story, and why the assertion below now also pins how many of these
    anybody has actually measured. Growing the shelf and growing the evidence are different
    events, and only one of them makes the project stronger.
    """
    from collections import Counter

    from b_ml_pipeline import available_models
    import model_reference as mr

    pool = mr.model_pool()
    by_pipeline = Counter(v["pipeline"] for v in pool.values())

    assert len(set(available_models())) == 15, "B_ML point pool changed"
    assert by_pipeline == {"B_ML": 15, "E_QUANTILE": 3, "A_STAT": 8, "C_DL": 5}, dict(by_pipeline)
    assert len(pool) == 31, f"model_pool() changed: {len(pool)}"

    comp = mr.composition(pool)
    assert comp["evaluated_total"] == 8, (
        f"the number of models with a recorded result changed to {comp['evaluated_total']}; "
        f"say so deliberately rather than letting the shelf count speak for the evidence")
    assert comp["untested_total"] == 20, comp["untested_total"]

    quantile_only = {k for k, v in pool.items() if v["pipeline"] == "E_QUANTILE"}
    assert quantile_only == {"GBQuantile", "ResidualRF", "LGBMQuantile"}, sorted(quantile_only)

    # The three A_STAT references are baselines, not competitors: a headline count that sums them
    # in would present the ruler as a rival to the models measured against it.
    baselines = {k for k, v in pool.items() if v.get("role") == "baseline"}
    assert baselines == {"NAIVE", "WEEKDAY_MEAN", "MOVAVG"}, sorted(baselines)


def test_every_description_is_reachable_and_every_pool_entry_described():
    """Item 6 part 3: DESCRIPTIONS and the enumerable pool must be the same set.

    Previously DESCRIPTIONS had 18 entries while model_pool() enumerated 16, and the two extra
    ("ETS", "Theta") matched neither the pool nor the family's own dispatch names -- so they were
    dead text nothing could surface. This pins both directions, so a model added to a pipeline
    without a description, or a description written for a model that does not exist, fails here.
    """
    import model_reference as mr

    pool = set(mr.model_pool())
    described = set(mr.DESCRIPTIONS)

    assert described - pool == set(), (
        f"descriptions unreachable from the Models page: {sorted(described - pool)}")
    assert pool - described == set(), (
        f"pool entries with no description: {sorted(pool - described)}")


def test_a_stat_registry_matches_what_the_dispatch_implements():
    """A name in the registry with no branch in ``_fc`` would be an advertised model that cannot run.

    Checked by dispatching rather than by searching the source for ``== "NAME"``. The source
    check broke the moment two names legitimately shared one branch: ETS_DAMPED differs from ETS
    in a single setting, so giving it a second copy of that block would have duplicated the
    seasonal fallback and the interval extraction purely to satisfy a test. Running each name is
    also the stronger claim, since it catches a branch that exists and raises.
    """
    import numpy as _np
    import pandas as _pd

    import run_a_stat as astat

    idx = _pd.bdate_range("2021-01-04", periods=520)
    y = _pd.Series(_np.sin(_np.arange(520) / 5.0) * 8.0 + _np.arange(520) * 0.05 + 100.0,
                   index=idx)
    y.index.freq = "B"
    future = _pd.bdate_range(idx[-1] + _pd.tseries.offsets.BDay(1), periods=5)

    for name in astat.registry_models():
        pred, _lo, _hi = astat._fc(name, y, future, {}, "Daily")
        assert len(pred) == len(future), f"{name} returned {len(pred)} values for 5 dates"
        assert _np.isfinite(pred).all(), f"{name} produced a non-finite forecast"


def test_a_stat_refuses_an_unknown_model_instead_of_forecasting_naively():
    """It used to fall through to a carried-forward last value under the requested name."""
    import numpy as _np
    import run_a_stat as astat

    y = pd.Series(_np.arange(50.0), index=pd.bdate_range("2024-01-01", periods=50))
    idx = pd.bdate_range("2024-03-12", periods=5)

    with pytest.raises(astat.UnknownAStatModel, match="does not implement"):
        astat._fc("XGBoost", y, idx, {}, "Daily")

    # ...and a real one still works.
    pred, lo, hi = astat._fc("NAIVE", y, idx, {}, "Daily")
    assert len(pred) == 5 and pred[0] == 49.0


def test_no_integrity_verdict_is_written_as_an_unconditional_literal():
    """Generalise bug 6 beyond the one field that was caught.

    Any of these written as a bare ``True`` is the same defect: a field that reads as a measured
    verdict with nothing behind it.
    """
    verdicts = ("alignment_ok", "leakage_detected", "signal_detected",
                "quality_gate_passed", "gate_passed")
    offenders = []
    for p in sorted(BACKEND.glob("*.py")):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for v in verdicts:
                if f'"{v}": True' in s or f"'{v}': True" in s:
                    offenders.append(f"{p.relative_to(REPO)}:{i}  {s[:70]}")
    assert offenders == [], (
        "an integrity verdict is hardcoded True, which is a field that cannot fail:\n  "
        + "\n  ".join(offenders))
