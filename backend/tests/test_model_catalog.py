"""Adding a model is one entry, the refactor moved nothing, and an unmeasured model says so.

Three separate claims, three sections.

**The refactor moved nothing.** ``available_models`` used to be a hand-written dictionary
in the middle of ``b_ml_pipeline``; it now builds from ``model_catalog``. Every published
number in this project came out of the old construction, so if a hyperparameter moved in
the process, every one of those numbers would silently stop being reproducible. Two
independent proofs below: a snapshot of every estimator's full parameter set taken from the
pre-catalogue code, and a fit-and-predict comparison against estimators written out
literally as the old code wrote them, asserting bit-for-bit identical predictions.

**Adding a model is one entry.** Asserted structurally: nothing outside the catalogue may
name a model, so a new entry inherits the rolling-origin evaluation, both baselines, the
sentinel, the ledger and the gates without any of them being touched.

**An unmeasured model says so.** Status is derived from the experiment ledger rather than
declared, because a declared status is a claim that drifts the moment somebody adds a model
and does not run it. Measured the day this was written, seven of thirteen were in exactly
that state, Ridge among them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import model_catalog  # noqa: E402
from model_catalog import (  # noqa: E402
    EVALUATED,
    FAMILY_ML,
    MODELS,
    UNAVAILABLE,
    UNTESTED,
    is_gate_eligible,
    measured_targets,
    shelf,
    spec_for,
    specs,
    status_of,
    untested_badge,
)

#: Captured from ``available_models()`` as it stood immediately before the catalogue
#: existed. Not transcribed by hand: written by iterating the old function.
SNAPSHOT = json.loads(
    (BACKEND / "tests" / "fixtures_available_models_pre_catalog.json").read_text(encoding="utf-8"))


def _params(estimator) -> dict:
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
            for k, v in estimator.get_params(deep=True).items()}


def _same(a, b) -> bool:
    """Equality that treats two NaNs as equal, which ``==`` does not."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


# ---------------------------------------------------------------------------
# 1. The refactor moved nothing
# ---------------------------------------------------------------------------

def test_no_model_was_lost_in_the_move():
    from b_ml_pipeline import available_models

    assert set(SNAPSHOT) <= set(available_models()), (
        f"models present before the catalogue and absent after: "
        f"{sorted(set(SNAPSHOT) - set(available_models()))}"
    )


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_every_estimator_is_constructed_exactly_as_before(name):
    """Every published number came out of the old construction.

    A moved hyperparameter would make all of them silently unreproducible, and nothing else
    in the suite would notice, because every other test measures whatever the current code
    builds.
    """
    from b_ml_pipeline import available_models

    built = available_models()[name]
    assert type(built).__name__ == SNAPSHOT[name]["class"]

    now, before = _params(built), SNAPSHOT[name]["params"]
    assert set(now) == set(before), (
        f"{name} parameter set changed: added {sorted(set(now) - set(before))}, "
        f"removed {sorted(set(before) - set(now))}"
    )
    differing = [k for k in now if not _same(now[k], before[k])]
    assert not differing, (
        f"{name} changed: " + ", ".join(f"{k} {before[k]!r} -> {now[k]!r}" for k in differing)
    )


@pytest.mark.parametrize("name", ["Ridge", "HistGBDT_L1"])
def test_a_catalogue_estimator_predicts_identically_to_the_old_construction(name):
    """The strongest form of the claim: same data, same seed, identical predictions.

    The comparison estimators below are written out exactly as ``available_models`` wrote
    them before the catalogue existed. One linear model and one tree model, and the tree
    model is the champion recipe on the stock target.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from b_ml_pipeline import available_models

    if name == "Ridge":
        old = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler(with_mean=True, with_std=True)),
                        ("est", Ridge(random_state=0))])
    else:
        old = HistGradientBoostingRegressor(loss="absolute_error", random_state=0,
                                            min_samples_leaf=20, l2_regularization=1.0)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = X[:, 0] * 3.0 - X[:, 3] * 1.5 + rng.normal(scale=0.1, size=400)
    X_new = rng.normal(size=(25, 8))

    new = available_models()[name]
    old.fit(X, y)
    new.fit(X, y)
    np.testing.assert_array_equal(old.predict(X_new), new.predict(X_new))


def test_an_uninstalled_package_omits_its_models_rather_than_substituting(monkeypatch):
    """Falling back would put one fit in the table under another model's name."""
    real = model_catalog.find_spec

    def pretend_missing(pkg, *args, **kwargs):
        return None if pkg == "lightgbm" else real(pkg, *args, **kwargs)

    monkeypatch.setattr(model_catalog, "find_spec", pretend_missing)
    from b_ml_pipeline import available_models

    built = available_models()
    assert "LightGBM_L1" not in built
    assert "HistGBDT_L1" in built, "only the affected models are omitted"


# ---------------------------------------------------------------------------
# 2. Adding a model is one entry
# ---------------------------------------------------------------------------

def test_available_models_names_no_model_of_its_own():
    """The property that makes adding a model one entry rather than several.

    If ``available_models`` still carried model names, a new entry would have to be added
    here as well as in the catalogue, and one of the two would eventually be forgotten.
    """
    src = (BACKEND / "b_ml_pipeline.py").read_text(encoding="utf-8")
    body = src.split("def available_models")[1].split("\ndef select_best_model")[0]
    for name in [s.name for s in specs(FAMILY_ML)]:
        assert f'"{name}"' not in body, (
            f"available_models still names {name}; adding a model is no longer one entry")


def test_nothing_else_hardcodes_the_machine_learning_model_list():
    """A second list is a second thing to update, and the stale one wins silently."""
    src = (BACKEND / "b_ml_pipeline.py").read_text(encoding="utf-8")
    assert src.count('"LightGBM_L1"') == 0
    assert src.count('"Ridge"') == 0


@pytest.mark.parametrize("spec", [s for s in MODELS if s.family == FAMILY_ML],
                         ids=lambda s: s.name)
def test_every_catalogue_entry_is_complete(spec):
    assert spec.factory is not None, f"{spec.name} has no way to be built"
    assert spec.summary and spec.summary.endswith("."), (
        f"{spec.name} has no finished plain-language summary")
    assert "--" not in spec.summary and "—" not in spec.summary
    assert spec.added, f"{spec.name} does not record when it was added"


@pytest.mark.parametrize("spec", [s for s in MODELS if s.family == FAMILY_ML],
                         ids=lambda s: s.name)
def test_every_installed_entry_actually_builds(spec):
    if not spec.installed:
        pytest.skip(f"{', '.join(spec.requires)} is not installed here")
    spec.build()


def test_catalogue_names_are_unique():
    names = [s.name for s in MODELS]
    assert len(names) == len(set(names))


def test_a_new_model_inherits_the_evaluation_machinery():
    """Asserted at the seam: everything downstream works from the estimator, not a branch.

    ``run_forward._fit_predict_point`` and ``b_ml_pipeline``'s training loop both look the
    model up by name in ``available_models()`` and then call fit and predict. No caller
    branches on which model it is, which is what makes rolling-origin evaluation, both
    baselines, the sentinel, the ledger and the gates apply to a new entry for free.
    """
    forward = (BACKEND / "forward_forecast.py").read_text(encoding="utf-8")
    assert "models = available_models()" in forward
    assert "if model_name ==" not in forward, "a per-model branch would break inheritance"

    pipeline = (BACKEND / "b_ml_pipeline.py").read_text(encoding="utf-8")
    assert "estimator.fit(X_tr_fit, y_tr_fit)" in pipeline, (
        "the training loop must fit whatever estimator it was handed")

    # An allow-list rather than a ban, because one branch is legitimate and a flat ban
    # would have to be deleted the first time somebody hit it, taking the check with it.
    #
    # The single permitted entry prints Lasso's iteration count and coefficient norm after
    # a fit. It changes no number and touches neither fitting nor prediction. Any NEW name
    # appearing here fails this test, which is the point: a behavioural per-model branch is
    # exactly what stops a new catalogue entry inheriting the evaluation machinery.
    branched = {name for name in [s.name for s in specs(FAMILY_ML)]
                if f'== "{name}"' in pipeline}
    assert branched <= {"Lasso"}, (
        f"a per-model branch would break inheritance for {sorted(branched - {'Lasso'})}")
    lasso_branch = pipeline.split('if model_name == "Lasso":')[1].split("# PHASE 4")[0]
    assert "print(" in lasso_branch
    assert ".fit(" not in lasso_branch and ".predict(" not in lasso_branch, (
        "the permitted branch is a diagnostic and must stay one")


# ---------------------------------------------------------------------------
# 3. An unmeasured model says so
# ---------------------------------------------------------------------------

def test_status_is_derived_from_the_ledger_not_declared():
    """No entry carries a status field, so there is nothing to declare wrongly."""
    src = (BACKEND / "model_catalog.py").read_text(encoding="utf-8")
    spec_block = src.split("class ModelSpec")[1].split("# ------")[0]
    assert "status:" not in spec_block, "a declared status is a claim that drifts"
    for entry in MODELS:
        assert not hasattr(entry, "status")


@pytest.mark.parametrize("name", ["LightGBM_L1", "HistGBDT_L1", "XGBoost_L1"])
def test_a_model_with_ledger_rows_is_evaluated(name):
    if not spec_for(name).installed:
        pytest.skip("package not installed here")
    assert status_of(name) == EVALUATED
    assert measured_targets(name)
    assert is_gate_eligible(name)


@pytest.mark.parametrize("name", ["Ridge", "Lasso", "ElasticNet", "RandomForest",
                                  "ExtraTrees", "Huber", "GBDT_L1"])
def test_a_model_with_no_ledger_row_is_untested(name):
    """Seven of the thirteen models the Lab offered were in this state, Ridge included."""
    assert status_of(name) == UNTESTED
    assert measured_targets(name) == ()
    assert not is_gate_eligible(name)


def test_an_untested_model_is_not_gate_eligible():
    """The rule Task 4 asks for, asserted over the whole shelf rather than one case."""
    for row in shelf():
        assert row["gate_eligible"] == (row["status"] == EVALUATED)
        if row["status"] != EVALUATED:
            assert row["badge"], "an untested model must carry a reason, not a bare label"


def test_an_untested_model_cannot_reach_the_alternatives_list():
    """The end-to-end consequence, through the code the Forecast page actually calls."""
    from model_shelf import alternatives_for
    from registry import load_registry

    untested = {row["name"] for row in shelf() if row["status"] != EVALUATED}
    for target in [r["target"] for r in load_registry()["recipes"]]:
        for alt in alternatives_for(target, k=5):
            assert alt.model.split("+")[0] not in untested


def test_the_untested_badge_is_a_finished_sentence():
    for row in shelf():
        if row["badge"]:
            assert row["badge"].endswith(".")
            assert "--" not in row["badge"] and "—" not in row["badge"]
            assert "Traceback" not in row["badge"]


def test_an_uninstalled_model_says_which_package_it_needs(monkeypatch):
    """Not a defect, and not something to hide behind a disappearing menu item."""
    real = model_catalog.find_spec
    monkeypatch.setattr(model_catalog, "find_spec",
                        lambda pkg, *a, **k: None if pkg == "catboost" else real(pkg, *a, **k))
    model_catalog._measured.cache_clear()
    assert status_of("CatBoost_L1") == UNAVAILABLE
    assert "catboost" in untested_badge("CatBoost_L1")
    assert not is_gate_eligible("CatBoost_L1")


def test_the_shelf_is_readable_without_the_modelling_stack():
    """The Streamlit interpreter has pandas and nothing else.

    Building a LightGBM object to display its name would put the modelling stack in the
    wrong process, so the catalogue must carry no heavy import at module level.
    """
    src = (BACKEND / "model_catalog.py").read_text(encoding="utf-8")
    header = src.split("# ------")[0]
    for heavy in ("import sklearn", "from sklearn", "import lightgbm", "from lightgbm",
                  "import xgboost", "from xgboost", "import catboost", "from catboost",
                  "import numpy", "import pandas"):
        assert heavy not in header, f"{heavy} at module level makes the catalogue unreadable"


def test_the_three_new_candidates_are_registered_and_honestly_labelled():
    """Task 4 asks for new candidates on the shelf, visibly new and visibly unmeasured."""
    rows = {row["name"]: row for row in shelf()}
    for name in ("Huber", "GBDT_L1"):
        assert name in rows
        assert rows[name]["status"] == UNTESTED
        assert rows[name]["added"] == "2026-08-19"
        assert rows[name]["measured_on"] == []

    import run_a_stat

    assert "ETS_DAMPED" in run_a_stat.registry_models()
    assert run_a_stat.model_roles()["ETS_DAMPED"] == "forecast"


def test_the_new_candidates_need_no_package_that_is_not_already_here():
    """A new dependency is a decision for a person, not a side effect of adding a model."""
    for name in ("Huber", "GBDT_L1"):
        assert spec_for(name).requires == ()


def test_ets_damped_actually_damps():
    """A registered name that dispatches to the same forecast would be a lie in the list."""
    import pandas as pd

    import run_a_stat

    idx = pd.bdate_range("2022-01-03", periods=300)
    y = pd.Series(np.arange(300) * 0.5 + 100.0, index=idx)
    y.index.freq = "B"
    future = pd.bdate_range(idx[-1] + pd.tseries.offsets.BDay(1), periods=10)

    plain, _, _ = run_a_stat._fc("ETS", y, future, {}, "Daily")
    damped, _, _ = run_a_stat._fc("ETS_DAMPED", y, future, {}, "Daily")
    assert not np.allclose(plain, damped)
    # The damped forecast's step from one day to the next must shrink, which is the whole
    # behaviour the name promises.
    steps = np.diff(damped)
    assert np.all(np.abs(steps[1:]) <= np.abs(steps[:-1]) + 1e-9)


def test_every_stat_name_in_the_catalogue_is_dispatchable():
    """A name in a list with no branch behind it used to fall through to a naive forecast."""
    import run_a_stat

    assert set(model_catalog.STAT_MODEL_NAMES) == set(run_a_stat.A_STAT_MODELS)


# ══════════════════════════════════════════════════════════════════════════════
# THE 2026-08-19 WIDENING
#
# Eleven models across three families, every one a candidate with no recorded result. The point
# of these tests is not that the names exist; it is that adding a name cannot quietly mean less
# than it appears to. Three things had to be true and are asserted here: the model is honestly
# badged, it needs no package a person did not agree to, and there is a branch behind the name.
#
# That last one is a failure this project has already had. A name in a list with no dispatch
# behind it fell through to a naive forecast and reported it under the model's own name.
# ══════════════════════════════════════════════════════════════════════════════

#: Registered 2026-08-19, by family. Smoke-tested end to end through the Lab's own dispatch path
#: (same runner, same TG_* contract, same exploratory overrides) on Revenues at h=5: 11 of 11
#: produced a populated predictions_long.csv.
WIDENING_2026_08_19 = {
    "B_ML": ("BayesianRidge", "TheilSen", "KNN", "KernelRidge", "DecisionTree_L1", "AdaBoost"),
    "E_QUANTILE": ("LinearQuantile", "HistGBQuantile", "XGBQuantile"),
    "A_STAT": ("SES", "HOLT"),
}


@pytest.mark.parametrize("name", WIDENING_2026_08_19["B_ML"])
def test_a_widening_ml_model_is_registered_and_honestly_badged(name):
    rows = {row["name"]: row for row in shelf()}
    assert name in rows, f"{name} is not on the shelf"
    assert rows[name]["status"] == UNTESTED
    assert rows[name]["added"] == "2026-08-19"
    assert rows[name]["measured_on"] == [], "an unmeasured model may not claim a target"
    assert rows[name]["gate_eligible"] is False, (
        "an untested model must not be gate-eligible, or it could reach a champion slot "
        "without a measurement behind it")


@pytest.mark.parametrize("name", WIDENING_2026_08_19["B_ML"])
def test_a_widening_ml_model_needs_no_new_package(name):
    """A new dependency is a decision for a person, not a side effect of adding a model."""
    assert spec_for(name).requires == ()


@pytest.mark.parametrize("name", WIDENING_2026_08_19["B_ML"])
def test_a_widening_ml_model_actually_builds(name):
    est = spec_for(name).build()
    assert hasattr(est, "fit") and hasattr(est, "predict")


@pytest.mark.parametrize("name", WIDENING_2026_08_19["B_ML"])
def test_a_widening_ml_model_is_reachable_from_the_point_pool(name):
    """`available_models()` is what the Lab and the Forecast page's comparison offer."""
    from b_ml_pipeline import available_models

    assert name in available_models()


@pytest.mark.parametrize("name", WIDENING_2026_08_19["A_STAT"])
def test_a_widening_stat_model_is_registered_and_dispatchable(name):
    """Registered, described, and with a branch behind the name rather than a fall-through."""
    import run_a_stat

    assert name in run_a_stat.registry_models()
    assert run_a_stat.model_roles()[name] == "forecast", (
        "these are forecasters, not references; counting one as a baseline would put it in the "
        "ruler group it is supposed to be measured against")
    assert name in model_catalog.STAT_MODEL_NAMES


@pytest.mark.parametrize("name", WIDENING_2026_08_19["E_QUANTILE"])
def test_a_widening_quantile_model_is_registered_and_dispatchable(name):
    import e_quantile_daily_pipeline as eq

    assert name in eq.registry_models()
    assert name in model_catalog.QUANTILE_MODEL_NAMES


def test_every_quantile_name_in_the_catalogue_is_dispatchable():
    """The same guard `STAT_MODEL_NAMES` has, for the family that had no such list.

    `QUANTILE_MODEL_NAMES` exists because the Streamlit interpreter cannot import
    `e_quantile_daily_pipeline` (it pulls in sklearn), so the Lab had a hand-written copy that
    had drifted to one entry while the family offered three.
    """
    import e_quantile_daily_pipeline as eq

    assert set(model_catalog.QUANTILE_MODEL_NAMES) == set(eq.registry_models())


@pytest.mark.parametrize("name", WIDENING_2026_08_19["E_QUANTILE"])
def test_a_widening_quantile_model_never_emits_a_crossed_band(name):
    """A lower edge above its upper edge is not a wide interval, it is an invalid one.

    All three fit each quantile independently, which does not guarantee an order. Measured:
    HistGBQuantile and XGBQuantile cross on real rows. Nothing downstream repairs it -- the
    predictions go straight into the yhat_p10/p50/p90 columns, and the caller's `n_cross` is only
    ever printed -- so the repair has to happen in the family, and it does.
    """
    import warnings

    import numpy as np
    import e_quantile_daily_pipeline as eq

    rng = np.random.default_rng(0)
    X = rng.normal(size=(900, 10))
    y = X @ rng.normal(size=10) + rng.normal(size=900)

    class _Cfg:
        lgbm_params: dict = {}
        horizon = 5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, _crossed = eq._predict_quantiles(name, _Cfg, X[:800], y[:800], X[800:],
                                                (0.1, 0.5, 0.9))

    assert np.all(preds[0.1] <= preds[0.5]), f"{name}: p10 above p50"
    assert np.all(preds[0.5] <= preds[0.9]), f"{name}: p50 above p90"


def test_the_crossing_repair_reports_how_many_rows_it_fixed():
    """Repaired AND counted. `ResidualRF` repairs silently, so nobody learns when a model
    crosses constantly, and constant crossing is what a misconfigured quantile model looks
    like from outside. The count feeds the report the fold loop already prints.
    """
    import numpy as np
    import e_quantile_daily_pipeline as eq

    crossed_row = {0.1: np.array([5.0, 1.0]), 0.5: np.array([3.0, 2.0]),
                   0.9: np.array([4.0, 3.0])}
    fixed, n_repaired = eq._enforce_monotone(crossed_row, (0.1, 0.5, 0.9))

    assert n_repaired == 1, "one of the two rows was crossed"
    assert list(fixed[0.1]) == [5.0, 1.0]
    assert list(fixed[0.5]) == [5.0, 2.0], "the p50 is raised to the p10, never the reverse"
    assert list(fixed[0.9]) == [5.0, 3.0]


def test_the_crossing_repair_leaves_a_well_ordered_band_alone():
    import numpy as np
    import e_quantile_daily_pipeline as eq

    fine = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
    fixed, n_repaired = eq._enforce_monotone(fine, (0.1, 0.5, 0.9))
    assert n_repaired == 0
    assert (fixed[0.1], fixed[0.5], fixed[0.9]) == (1.0, 2.0, 3.0) or all(
        fixed[q][0] == v for q, v in ((0.1, 1.0), (0.5, 2.0), (0.9, 3.0)))


def test_ses_holds_the_level_and_holt_carries_a_trend():
    """The behaviour the two names promise, which is the only reason to have both.

    If SES drifted or HOLT went flat, both would still 'run' and the pair would be two names for
    one model. On a series with a clear upward trend the difference is visible in one line.
    """
    import warnings

    import pandas as pd
    import run_a_stat

    slope, steps = 10.0, 10
    idx = pd.bdate_range("2020-01-01", periods=600)
    y = pd.Series(range(len(idx)), index=idx, dtype=float) * slope + 1000.0
    future = pd.bdate_range(idx[-1] + pd.offsets.BDay(1), periods=steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses, _, _ = run_a_stat._fc("SES", y, future, {}, "daily")
        holt, _, _ = run_a_stat._fc("HOLT", y, future, {}, "daily")

    # Compared against the series' own SLOPE, not its level. A first version of this test
    # required HOLT to move by 5% of the level, which no correct trend forecast could do: the
    # most a right answer can travel over ten steps is nine increments, here 90 against a level
    # of 6990. The threshold was wrong, not the model.
    ideal_span = slope * (steps - 1)
    ses_span = float(abs(ses[-1] - ses[0]))
    holt_span = float(holt[-1] - holt[0])

    assert ses_span < 0.1 * ideal_span, (
        f"SES must hold the level, but its forecast moved by {ses_span:,.2f}")
    assert holt_span == pytest.approx(ideal_span, rel=0.1), (
        f"HOLT must carry the trend: expected about {ideal_span:,.0f} over {steps} steps, "
        f"got {holt_span:,.2f}")


def test_nothing_in_the_frontend_hardcodes_a_model_list():
    """The drift this widening found, and the reason the lists are derived rather than typed.

    Measured on 2026-08-19 before the fix: the registry offered 15 machine-learning models and
    the Lab named 8; the quantile family offered 3 and the Lab named 1; ETS_DAMPED was missing
    entirely. Ten registered models could not be run from the Lab at all, while the Models page
    told readers they could run any untested model there. `b_ml_pipeline` already had a test like
    this one; the frontend did not, which is exactly where the stale copies lived.
    """
    frontend = BACKEND.parent / "frontend"
    lab = (frontend / "pages" / "03_Lab.py").read_text(encoding="utf-8")
    for name in ("Ridge", "HistGBDT", "GBQuantile", "SARIMAX"):
        assert f'"{name}",' not in lab, (
            f"03_Lab.py names {name!r} in a list. Model lists come from backend_consts, which "
            f"reads the registry, or the Lab silently offers a stale subset of the shelf.")

    consts = (frontend / "backend_consts.py").read_text(encoding="utf-8")
    assert "from model_catalog import" in consts, (
        "backend_consts must read the registry rather than restate it")
