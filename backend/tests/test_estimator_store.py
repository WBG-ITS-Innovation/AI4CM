"""Persisted estimators must reproduce their published numbers, or refuse to load.

Three failure modes are guarded here, in order of how badly each would mislead:

1. **A scrub that changes predictions.** Removing training data from an estimator is only
   acceptable if it is arithmetically invisible. That is proved by comparing predictions before
   and after -- on a batch large enough that the scale check actually consumes the attribute
   being scrubbed -- not asserted in a docstring.
2. **Capturing the estimator changing the forecast.** ``_fit_predict_point`` used to return a
   bare float and discard the fit. Threading the object out must not move a published number.
3. **A blob loading when it should not.** scikit-learn only *warns* on a version mismatch and
   loads anyway, so silence here would mean wrong numbers under a published label.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from estimator_store import (  # noqa: E402
    DEFAULT_KEEP_LAST,
    ESTIMATOR_DIR,
    MANIFEST_NAME,
    EstimatorDigestMismatch,
    EstimatorMissing,
    EstimatorVersionMismatch,
    FittedEstimator,
    ReproductionUnavailable,
    check_versions,
    load_estimator,
    prune_estimators,
    read_manifest,
    reproduce_prediction,
    save_estimators,
    scrub_for_persistence,
)
from target_scaling import MIN_SANITY_BATCH, ScaledRegressor, trailing_level  # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ── fixtures ──────────────────────────────────────────────────────────────────

def _ratio_fit(n: int = 300, n_features: int = 4):
    """A fitted ``ScaledRegressor`` on the ``ratio`` transform -- the wrapper that holds data."""
    from sklearn.tree import DecisionTreeRegressor

    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, n_features)),
                     columns=[f"f{i}" for i in range(n_features)], index=idx)
    y = pd.Series(1e7 + rng.normal(scale=1e5, size=n), index=idx)
    level = trailing_level(y)
    ok = level.notna()
    X, y, level = X[ok], y[ok], level[ok]
    est = ScaledRegressor(base=DecisionTreeRegressor(random_state=0),
                          transform="ratio", level=level)
    est.fit(X, y.to_numpy(float))
    return est, X, y


@pytest.fixture(scope="module")
def issue(tmp_path_factory):
    """A published issue with retained estimators, built from a real fit."""
    est, X, y = _ratio_fit()
    origin = X.index[-1:]
    d = tmp_path_factory.mktemp("published") / "2025-08-06"
    d.mkdir(parents=True)

    pred = float(np.asarray(est.predict(X.loc[origin])).ravel()[0])
    pd.DataFrame([{"target": "Revenues", "horizon": 5, "origin_date": str(origin[0].date()),
                   "origin_value": float(y.iloc[-1]), "target_date": "2025-08-13",
                   "p10": pred * 0.9, "p50": pred, "p90": pred * 1.1,
                   "modelled_as": "level"}]).to_csv(d / "forecast.csv", index=False)

    save_estimators(
        d,
        [FittedEstimator(target="Revenues", horizon=5, kind="point", estimator=est,
                         feature_names=list(X.columns), n_train_rows=len(X),
                         origin_date=str(origin[0].date()), target_transform="ratio",
                         recipe_id="revenues-lgbm-l1-ws3-v1",
                         selection_run_id="20260805T090357.040566_Revenues_LightGBM_L1_ratio",
                         fiscal_groups=("A_deadline",))],
        keep_index=origin, provenance={"data": {"sha256": "deadbeef"},
                                       "code": {"git_sha": "cafe1234"}})
    return d, pred, X, origin


# ── 1. the scrub is behaviour-neutral, proved not asserted ────────────────────

def test_scrub_does_not_change_predictions_on_a_batch_that_uses_the_scrubbed_attribute():
    """The load-bearing proof.

    ``y_train_ref_`` is consumed by ``sanity_check_prediction_scale``, which is *skipped* for
    batches under ``MIN_SANITY_BATCH``. A single-row check would therefore prove nothing about
    the attribute being scrubbed. This predicts a batch well over that threshold, so the scrubbed
    value is genuinely used, and requires bitwise-equal output.
    """
    est, X, _ = _ratio_fit()
    batch = X.iloc[-(MIN_SANITY_BATCH * 2):]
    assert len(batch) > MIN_SANITY_BATCH, "the batch must be large enough to run the scale check"

    before = np.asarray(est.predict(batch), dtype=float)
    scrubbed, disclosure = scrub_for_persistence(est, batch.index)
    after = np.asarray(scrubbed.predict(batch), dtype=float)

    assert np.array_equal(before, after), (
        f"scrubbing changed predictions: max abs diff "
        f"{np.max(np.abs(before - after)):.6g}")
    assert set(disclosure) == {"y_train_ref_", "level"}


def test_scrub_is_behaviour_neutral_at_the_single_origin_a_published_issue_uses():
    est, X, _ = _ratio_fit()
    origin = X.index[-1:]
    before = float(np.asarray(est.predict(X.loc[origin])).ravel()[0])
    scrubbed, _ = scrub_for_persistence(est, origin)
    after = float(np.asarray(scrubbed.predict(X.loc[origin])).ravel()[0])
    assert before == after


def test_scrub_actually_removes_the_training_data():
    est, X, y = _ratio_fit()
    origin = X.index[-1:]
    assert len(np.asarray(est.y_train_ref_)) == len(X) > 1, "fixture must hold the full target"
    assert len(est.level) > 1

    scrubbed, _ = scrub_for_persistence(est, origin)
    assert len(np.asarray(scrubbed.y_train_ref_)) == 1, "full training target still present"
    assert len(scrubbed.level) == 1, "full trailing-level series still present"
    # And the one number that survives is the aggregate the check needs, nothing more.
    assert np.isclose(float(scrubbed.y_train_ref_[0]),
                      float(np.nanmedian(np.abs(y.to_numpy(float)))))


def test_scrubbed_estimator_refuses_an_origin_it_was_not_retained_for():
    """A replay artifact must not silently forward-fill a divisor for the wrong date."""
    est, X, _ = _ratio_fit()
    scrubbed, _ = scrub_for_persistence(est, X.index[-1:])
    with pytest.raises(ValueError, match="trailing level is missing"):
        scrubbed.predict(X.iloc[:10])


def test_a_raw_transform_estimator_carries_no_target_series():
    """Only the ``ratio`` recipe wraps; Expenditure and the balance target do not."""
    from sklearn.tree import DecisionTreeRegressor
    est = DecisionTreeRegressor(random_state=0).fit([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0])
    scrubbed, disclosure = scrub_for_persistence(est, pd.Index([0]))
    assert disclosure == {}, "nothing to scrub on an unwrapped estimator"
    assert not hasattr(scrubbed, "y_train_ref_")


# ── 2. capturing the estimator does not move a published number ───────────────

@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_capturing_estimators_leaves_the_forecast_identical():
    """``_fit_predict_point`` now returns the fit as well as the float. Prove nothing moved."""
    from forward_forecast import Champion, run_forward
    from preprocessing.fiscal_calendar import GROUP_A, GROUP_C

    raw = pd.read_csv(DATA)
    champ = Champion(target="Revenues", point_model="LightGBM_L1",
                     fiscal_groups=(GROUP_A, GROUP_C), recipe_id="test")

    without = run_forward(raw, champ, horizons=(1, 2))
    sink: list = []
    with_capture = run_forward(raw, champ, horizons=(1, 2), estimator_sink=sink)

    cols = ["p10", "p50", "p90", "p50_quantile_model", "origin_value"]
    pd.testing.assert_frame_equal(without[cols], with_capture[cols])
    # 2 horizons x (1 point + 3 quantile)
    assert len(sink) == 8, [f.fit_id for f in sink]
    assert {f.kind for f in sink} == {"point", "q10", "q50", "q90"}


def test_fit_predict_point_returns_the_fitted_estimator_not_just_a_number():
    from forward_forecast import _fit_predict_point
    X = pd.DataFrame({"a": np.arange(40.0)}, index=pd.bdate_range("2021-01-01", periods=40))
    y = pd.Series(np.arange(40.0) * 3.0, index=X.index)
    pred, est = _fit_predict_point(X, y, X.iloc[-1:], "Ridge")
    assert isinstance(pred, float)
    assert hasattr(est, "predict"), "the fitted estimator must be returned, not discarded"
    assert float(np.asarray(est.predict(X.iloc[-1:])).ravel()[0]) == pred


# ── 3. loading refuses rather than guessing ───────────────────────────────────

def test_round_trip_reproduces_the_published_prediction_to_tolerance(issue):
    d, published, X, origin = issue
    est, meta = load_estimator(d, "Revenues", 5, "point")
    got = float(np.asarray(est.predict(X.loc[origin])).ravel()[0])
    assert got == pytest.approx(published, rel=1e-12, abs=1e-9), (
        f"a saved estimator did not reproduce its published number: {got} vs {published}")
    assert meta["verified"] is True and meta["version_diffs"] == {}
    assert meta["fit_id"] == "2025-08-06/revenues/h5/point", (
        "fit_id must be keyed on the issue date, so a -r2 re-issue gets a distinct id")


@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_reproduce_prediction_rebuilds_the_design_from_the_canonical_file():
    """End to end through the real forward path: publish, then re-derive from the blob."""
    from forecast_modes import official_run, publish_official

    res = official_run("Revenues", DATA)
    root = Path(pytest.importorskip("tempfile").mkdtemp())
    dest = publish_official(res, published_root=root,
                            forward_dir=root / "_work")

    out = reproduce_prediction(dest, DATA, "Revenues", 5, "point")
    assert out["data_sha_matches_issue"] is True
    assert out["verified_environment"] is True
    assert out["rel_diff"] < 1e-9, out


@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_a_feature_that_cannot_be_rebuilt_refuses_instead_of_substituting_nan(tmp_path):
    """Regression: the first draft of ``reproduce_prediction`` built the design with no feature
    groups and then reindexed to the stored names, filling the difference with NaN. The estimator
    happily predicted on the gaps and returned **106,407,564 against a published 58,319,423** --
    82% wrong, and indistinguishable from a real answer. A shortfall must raise.
    """
    import shutil

    from forecast_modes import official_run, publish_official

    res = official_run("Revenues", DATA)
    dest = publish_official(res, published_root=tmp_path / "pub",
                            forward_dir=tmp_path / "_work")

    broken = tmp_path / "broken" / dest.name
    broken.parent.mkdir(parents=True)
    shutil.copytree(dest, broken)
    mpath = broken / ESTIMATOR_DIR / MANIFEST_NAME
    m = json.loads(mpath.read_text())
    # estimators[] runs h=1..5 x (point, q10, q50, q90), so pick the entry actually reproduced.
    entry = next(e for e in m["estimators"]
                 if int(e["horizon"]) == 5 and e["kind"] == "point")
    assert entry["fiscal_groups"], "the recipe's feature groups must be recorded to begin with"
    entry["fiscal_groups"] = []                       # the design can no longer be rebuilt
    mpath.write_text(json.dumps(m, indent=2))

    with pytest.raises(ReproductionUnavailable, match="Refusing to substitute NaN"):
        reproduce_prediction(broken, DATA, "Revenues", 5, "point")


def test_digest_mismatch_is_refused_and_never_overridable(issue, tmp_path):
    import shutil
    d, _, _, _ = issue
    copy = tmp_path / "2025-08-06"
    shutil.copytree(d, copy)

    blob = copy / ESTIMATOR_DIR / "revenues" / "h5_point.joblib"
    blob.write_bytes(blob.read_bytes() + b"tampered")

    for allow in (False, True):          # a version flag must not excuse a broken digest
        with pytest.raises(EstimatorDigestMismatch, match="not the one that was published"):
            load_estimator(copy, "Revenues", 5, "point", allow_version_mismatch=allow)


def test_digest_is_checked_before_the_bytes_reach_joblib(issue, tmp_path, monkeypatch):
    """Unpickling executes code, so the refusal must come first -- not after loading."""
    import shutil

    import joblib
    d, _, _, _ = issue
    copy = tmp_path / "2025-08-06"
    shutil.copytree(d, copy)
    (copy / ESTIMATOR_DIR / "revenues" / "h5_point.joblib").write_bytes(b"not a pickle at all")

    called: list = []
    monkeypatch.setattr(joblib, "load", lambda *a, **k: called.append(a))
    with pytest.raises(EstimatorDigestMismatch):
        load_estimator(copy, "Revenues", 5, "point")
    assert called == [], "joblib.load was reached despite a bad digest"


def test_version_mismatch_raises_and_names_both_versions(issue, tmp_path):
    import shutil
    d, _, X, origin = issue
    copy = tmp_path / "2025-08-06"
    shutil.copytree(d, copy)

    mpath = copy / ESTIMATOR_DIR / MANIFEST_NAME
    manifest = json.loads(mpath.read_text())
    manifest["environment"]["packages"]["scikit-learn"] = "0.24.1"
    mpath.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(EstimatorVersionMismatch) as exc:
        load_estimator(copy, "Revenues", 5, "point")
    msg = str(exc.value)
    assert "0.24.1" in msg, msg
    import sklearn
    assert sklearn.__version__ in msg, msg

    # ...and the explicit override loads it, marked unverified.
    est, meta = load_estimator(copy, "Revenues", 5, "point", allow_version_mismatch=True)
    assert meta["verified"] is False
    assert meta["version_diffs"]["scikit-learn"]["published_with"] == "0.24.1"
    assert est.predict(X.loc[origin]) is not None


def test_sklearn_only_warns_on_a_version_mismatch_which_is_why_we_check_ourselves():
    """Pins the upstream behaviour this module exists to compensate for.

    If a future scikit-learn starts *raising*, this test fails and the loader's own check can be
    reconsidered. Until then, silence here would mean wrong numbers under a published label.
    """
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning
    from sklearn.linear_model import Ridge

    r = Ridge().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    state = r.__getstate__()
    state["_sklearn_version"] = "1.3.2"
    fresh = Ridge.__new__(Ridge)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh.__setstate__(state)                 # does not raise
    assert any(issubclass(w.category, InconsistentVersionWarning) for w in caught)
    assert fresh.predict([[4.0]]) is not None


def test_python_is_compared_at_major_minor_not_patch(issue, tmp_path):
    import shutil
    d, _, _, _ = issue
    copy = tmp_path / "2025-08-06"
    shutil.copytree(d, copy)
    mpath = copy / ESTIMATOR_DIR / MANIFEST_NAME
    manifest = json.loads(mpath.read_text())

    major_minor = ".".join(manifest["environment"]["python"].split(".")[:2])
    manifest["environment"]["python"] = f"{major_minor}.999"
    assert check_versions(manifest) == {}, "a patch bump must not block a load"

    manifest["environment"]["python"] = "2.7.18"
    assert "python" in check_versions(manifest)


def test_a_missing_manifest_reads_as_not_available(tmp_path):
    (tmp_path / "2020-01-01").mkdir()
    with pytest.raises(EstimatorMissing, match="cannot be re-derived"):
        read_manifest(tmp_path / "2020-01-01")


def test_an_unretained_combination_says_what_it_has(issue):
    d, _, _, _ = issue
    with pytest.raises(EstimatorMissing, match="no estimator for target"):
        load_estimator(d, "Revenues", 3, "point")


# ── retention ─────────────────────────────────────────────────────────────────

def _stub_issue(root: Path, name: str):
    from sklearn.tree import DecisionTreeRegressor
    d = root / name
    d.mkdir(parents=True)
    idx = pd.Index([pd.Timestamp("2020-01-01")])
    est = DecisionTreeRegressor(random_state=0).fit([[0.0], [1.0]], [0.0, 1.0])
    save_estimators(d, [FittedEstimator(target="Revenues", horizon=5, kind="point",
                                        estimator=est, feature_names=["f0"], n_train_rows=2,
                                        origin_date="2020-01-01")],
                    keep_index=idx)
    return d


def test_pruning_removes_blobs_keeps_the_manifest_and_says_so(tmp_path):
    root = tmp_path / "published"
    olds = [_stub_issue(root, f"2020-01-{i:02d}") for i in range(1, 4)]
    actions = prune_estimators(root, keep_last=1)

    assert [a["issue_date"] for a in actions] == ["2020-01-01", "2020-01-02"]
    assert all(a["bytes_freed"] > 0 for a in actions)
    assert DEFAULT_KEEP_LAST >= 1, "the default window must retain at least the latest issue"

    for d in olds[:2]:
        assert not (d / ESTIMATOR_DIR / "revenues" / "h5_point.joblib").exists()
        m = read_manifest(d)                     # the proof of what was published survives
        assert m["retention"]["pruned"] is True and m["retention"]["pruned_at"]
        with pytest.raises(EstimatorMissing, match="pruned"):
            load_estimator(d, "Revenues", 5, "point")

    assert (olds[-1] / ESTIMATOR_DIR / "revenues" / "h5_point.joblib").exists()


def test_pruning_protects_an_issue_however_old_it_is(tmp_path):
    root = tmp_path / "published"
    for i in range(1, 4):
        _stub_issue(root, f"2020-01-{i:02d}")
    actions = prune_estimators(root, keep_last=1, protect=["2020-01-01"])
    assert [a["issue_date"] for a in actions] == ["2020-01-02"]
    assert (root / "2020-01-01" / ESTIMATOR_DIR / "revenues" / "h5_point.joblib").exists()


def test_an_issue_with_unscored_horizons_is_reported_as_protectable(tmp_path):
    """The retention policy's protection rule, on its own.

    An unscored horizon is the one whose prediction is still going to be argued about, so it is
    the one that most needs to stay re-derivable.
    """
    from estimator_store import issues_with_unscored_horizons

    root = tmp_path / "published"
    for name in ("2020-01-01", "2020-01-02"):
        d = root / name
        d.mkdir(parents=True)
        pd.DataFrame({"target": ["Revenues"] * 2, "horizon": [1, 2]}).to_csv(
            d / "forecast.csv", index=False)

    # No scorecard at all -> nothing is scored -> everything is protected.
    assert issues_with_unscored_horizons(root) == ["2020-01-01", "2020-01-02"]

    sc = tmp_path / "scorecard.csv"
    pd.DataFrame({"issue_date": ["2020-01-01", "2020-01-01"],
                  "y_true": [1.0, 2.0]}).to_csv(sc, index=False)
    assert issues_with_unscored_horizons(root, sc) == ["2020-01-02"]

    # A row present but unscored still counts as unscored.
    pd.DataFrame({"issue_date": ["2020-01-02", "2020-01-02"],
                  "y_true": [1.0, None]}).to_csv(sc, index=False)
    assert "2020-01-02" in issues_with_unscored_horizons(root, sc)


def test_pruning_is_idempotent(tmp_path):
    root = tmp_path / "published"
    for i in range(1, 3):
        _stub_issue(root, f"2020-01-{i:02d}")
    assert prune_estimators(root, keep_last=1)
    assert prune_estimators(root, keep_last=1) == []


# ── manifest content and the retention divergence ─────────────────────────────

def test_manifest_records_versions_digests_and_the_scrub_disclosure(issue):
    d, _, _, _ = issue
    m = read_manifest(d)
    e = m["estimators"][0]

    assert set(m["pinned_packages"]) == {"scikit-learn", "lightgbm", "numpy", "joblib"}
    assert m["environment"]["packages"]["joblib"], "joblib version must be recorded"
    assert len(e["sha256"]) == 64 and e["bytes"] > 0
    assert e["n_features"] == len(e["feature_names"]) == 4
    assert e["scrubbed"]["y_train_ref_"].startswith("full training target")
    assert m["retention"]["blobs_tracked_in_git"] is False
    assert m["data_sha256"] == "deadbeef" and m["git_sha"] == "cafe1234"


def test_the_selection_run_id_is_labelled_as_a_different_fit(issue):
    """It must be impossible to read the DEV run_id as this fit's identity."""
    d, _, _, _ = issue
    e = read_manifest(d)["estimators"][0]
    assert e["selection_run_id"] == "20260805T090357.040566_Revenues_LightGBM_L1_ratio"
    assert "not the fit in this blob" in e["selection_run_id_note"]
    assert e["selection_run_id"] not in e["fit_id"]
    assert e["fit_id"] == "2025-08-06/revenues/h5/point"


def test_describe_environment_includes_joblib():
    from provenance import describe_environment
    assert describe_environment()["packages"]["joblib"]


def test_gitignore_excludes_the_whole_published_issue_blobs_included():
    """The blob rule was the narrow case; on 2026-08-15 it became the general one.

    This used to assert a split -- blobs ignored, manifest and forecast.csv tracked -- because
    the estimator was treated as the one artifact too dangerous to commit. The split did not
    survive contact with what forecast.csv actually holds (origin_value and the full P10/P50/P90
    path), so the whole issue directory is ignored now. The blob assertion is unchanged and must
    stay: if the published-forecast rule is ever revisited, the blobs still cannot be tracked.
    """
    def ignored(rel: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", rel],
                              cwd=REPO).returncode == 0

    assert ignored("forecasts/published/2025-08-06/estimators/revenues/h5_point.joblib"), (
        "estimator blobs must be gitignored -- they embed Treasury training data")
    assert ignored("forecasts/published/2025-08-06/estimators/manifest.json")
    assert ignored("forecasts/published/2025-08-06/forecast.csv"), (
        "forecast.csv carries row-level Treasury figures and must not be tracked")

    # The rule must not have been widened into the aggregate artifacts, which are the only
    # thing a clone can still read.
    assert not ignored("forecasts/scorecard.csv")
    assert not ignored("registry/recipes.json")


def test_why_the_blobs_can_never_be_tracked_is_documented_where_the_rule_is_made():
    """Both the code and the ignore file must carry the reason, not just the rule.

    The reason outlives the rule it was written for: it is the argument that would have to be
    defeated before an estimator could ever be committed, and it survives the 2026-08-15 change
    that absorbed the blob carve-out into a broader one.
    """
    src = (BACKEND / "estimator_store.py").read_text()
    assert "cannot be scrubbed" in src
    assert "World Bank" in src

    ig = (REPO / ".gitignore").read_text()
    section = ig.split("forecasts/published/\n", 1)[0]
    assert "CANNOT be scrubbed" in section, (
        "the blob reasoning must sit with the rule that now covers it")
    assert "World Bank" in section
    assert "must stay ignored on their own merits" in section, (
        "the ignore file must say the blob rule stands independently of the CSV rule")


def test_the_production_runner_only_publishes_when_asked():
    """Retention rides on publishing, and publishing is not reversible."""
    import inspect

    import run_forward_forecast as runner
    assert inspect.signature(runner.main).parameters["publish"].default is False
    src = inspect.getsource(runner)
    assert "estimator_sink=sink" in src, "the runner must capture what it publishes"
    assert "issues_with_unscored_horizons" in src, "pruning must protect unscored issues"


def test_exploratory_results_have_no_estimator_retention_path():
    """Nothing unpublished should leave Treasury-derived blobs on disk."""
    from forecast_modes import ExploratoryResult
    assert not hasattr(ExploratoryResult, "estimators")
    assert "estimators" not in ExploratoryResult.__dataclass_fields__
