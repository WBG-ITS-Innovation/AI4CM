"""The shelf: every model this system can run, and whether anyone has measured it.

Why this module exists
----------------------
Adding a model used to mean editing a dictionary buried in the middle of
``b_ml_pipeline.py``, between the feature builder and the training loop, with the
conditional import blocks for LightGBM, XGBoost and CatBoost interleaved into it. Nothing
recorded what a model was for, nothing recorded which package it needed, and nothing
recorded whether it had ever actually been run.

That last omission was the expensive one. Measured the day this module was written,
**seven of the thirteen models the Lab offered had no recorded result on any target**:
Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees and both CatBoost variants had no row
in ``experiments/log.csv``. They appeared in the model list beside LightGBM_L1, which has
five folds of evidence behind it, with nothing distinguishing them. A reader picking from
that list had no way to tell a champion from a name. Ridge, the first entry and the
default selection, was one of the seven.

So a model now carries its status, and the status is **derived, never declared**.

Derived status
--------------
``status_of`` asks the experiment ledger whether a recorded result exists. It does not
read a field somebody typed. A declared status is a claim that drifts the moment a model
is added and not run, which is exactly the state this module found the project in.

The ledger is the right question to ask rather than a harsh one. A registry recipe cites a
ledger ``run_id`` and ``registry.verify_against_log`` checks that the quoted numbers match
it, so a model with no ledger row has nothing a recipe could cite and nothing the
publication checks could read, however many times it has been executed from the Lab.

Two consequences follow, and both are enforced elsewhere rather than described here:

* An untested model has no measured numbers to show, so the Models page shows none and
  badges it UNTESTED.
* An untested model is not gate-eligible. ``publication_gates`` already refuses to pass a
  gate it has no measurement for, and ``model_shelf.alternatives_for`` already filters on
  the accuracy gate having actually passed, so an untested model cannot reach a
  champion slot without first being run. Nothing new was needed for that; it falls out of
  measurement being the thing that counts.

Adding a model
--------------
One entry in :data:`MODELS`, and a factory function only if the estimator is not already
constructible from what is imported. See ``docs/ADDING_A_MODEL.md``.

Why the factories are lazy
--------------------------
Every ``factory`` builds its estimator inside the function body, importing what it needs
there. That keeps this module free of sklearn, LightGBM, XGBoost and CatBoost at import
time, so the Streamlit interpreter -- which has pandas and nothing else -- can read the
catalogue to render the model list without the modelling stack being installed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_CSV = REPO_ROOT / "experiments" / "log.csv"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

#: A model with at least one run in ``experiments/log.csv``, so a measured number exists
#: that can be quoted and traced back to the run that produced it.
EVALUATED = "evaluated"

#: A model with no row in the experiment ledger.
#:
#: The precise claim matters. It does NOT mean "never executed": A_STAT runs daily and
#: writes a leaderboard into its run folder, and anyone can launch any of these from the
#: Lab. It means no measurement has been entered in the ledger, so there is no number
#: anybody can quote, trace to a run id, or put in front of the publication gates. A
#: registry recipe cites a ledger ``run_id`` and ``registry.verify_against_log`` checks it,
#: so a model outside the ledger cannot become a champion however often it has been run.
UNTESTED = "untested"

#: A model whose package is not installed in this environment. Not a defect, and not
#: something to hide: the name stays in the catalogue with the package it needs named, so
#: "why can I not pick CatBoost" has an answer that is not "it does not exist".
UNAVAILABLE = "unavailable"

#: A reference the others are measured against, not a candidate.
#:
#: Without this, carrying a value forward would be badged UNTESTED for having no ledger
#: row, which reads as a shortcoming. It has no ledger row because nobody would enter one:
#: it is the ruler. ``model_reference.composition()`` already keeps baselines out of the
#: competing count for the same reason, and this is that distinction one layer down.
BASELINE = "baseline"

#: Families a model can belong to. Matches the runner names the Lab dispatches on.
FAMILY_STAT = "A_STAT"
FAMILY_ML = "B_ML"
FAMILY_DL = "C_DL"
FAMILY_QUANTILE = "E_QUANTILE"


@dataclass(frozen=True)
class ModelSpec:
    """One model on the shelf.

    ``factory`` is a zero-argument callable returning an unfitted estimator. It is absent
    for families whose models are not scikit-learn estimators: A_STAT dispatches on a name
    inside its runner, and its entries here exist so one list can describe the whole shelf.
    """

    name: str
    family: str
    #: One plain sentence saying what this model does and when it might help. Shown to a
    #: reader choosing from a list, so it names the behaviour rather than the algorithm.
    summary: str
    factory: Optional[Callable[[], object]] = None
    #: Import names that must be present. Empty means "nothing beyond the base install".
    requires: Tuple[str, ...] = ()
    #: When this entry was added, so a new arrival is visibly new rather than merely unmeasured.
    added: str = ""
    #: Set when a model exists only as a reference to measure others against.
    is_baseline: bool = False

    @property
    def installed(self) -> bool:
        """Is every package this model needs importable here?"""
        return all(find_spec(pkg) is not None for pkg in self.requires)

    def build(self):
        """Construct the unfitted estimator. Raises if this family has no factory."""
        if self.factory is None:
            raise NotImplementedError(
                f"{self.name} belongs to {self.family}, which does not build estimators "
                f"through this catalogue. Its runner dispatches on the model name.")
        return self.factory()


# ---------------------------------------------------------------------------
# Estimator factories
#
# Every one imports inside its body. Their parameters are the pre-catalogue values,
# unchanged: `test_model_catalog.py` compares each built estimator against a snapshot of
# `available_models()` taken before this module existed, so the refactor cannot have moved
# a hyperparameter without a test noticing.
# ---------------------------------------------------------------------------

#: Capacity floor for tree leaves. A leaf holding one observation is memorisation by
#: construction, and ExtraTrees reached train MAE 0.00 exactly that way.
MIN_SAMPLES_PER_LEAF = 5


def _linear_pipeline(estimator):
    """Impute, scale, fit. The shape every linear model here uses."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler(with_mean=True, with_std=True)),
                     ("est", estimator)])


def _ridge():
    from sklearn.linear_model import Ridge

    return _linear_pipeline(Ridge(random_state=0))


def _lasso():
    from sklearn.linear_model import Lasso

    return _linear_pipeline(Lasso(random_state=0, max_iter=20000, tol=1e-4))


def _elastic_net():
    from sklearn.linear_model import ElasticNet

    return _linear_pipeline(ElasticNet(random_state=0))


def _huber():
    from sklearn.linear_model import HuberRegressor

    # epsilon=1.35 is the scikit-learn default and the value Huber's own paper derives for
    # 95% efficiency against a Gaussian. Left at the default deliberately: this entry is a
    # candidate, not a tuned model, and a hand-picked epsilon would make its first
    # measurement uninterpretable.
    return _linear_pipeline(HuberRegressor(epsilon=1.35, max_iter=500))


def _bayesian_ridge():
    from sklearn.linear_model import BayesianRidge

    # Ridge that estimates its own shrinkage from the data instead of being handed one. There
    # is no alpha to pick, so it cannot be tuned into looking good, which is what makes it a
    # useful reading next to Ridge on the same features.
    return _linear_pipeline(BayesianRidge())


def _theil_sen():
    from sklearn.linear_model import TheilSenRegressor

    # A second robust linear fit alongside Huber, and robust in a different way: Huber
    # de-weights extreme days, this one fits many small subsets and takes the median of their
    # answers, so a run of consecutive extreme days cannot drag it. Worth a reading on a
    # series whose month-ends are all extreme together.
    #
    # max_subpopulation caps the number of subsets; without it the count grows combinatorially
    # with the feature set and the fit stops finishing. 2000 is scikit-learn's own guidance for
    # keeping the cost bounded, and measured here it fits in about 2 seconds.
    return _linear_pipeline(TheilSenRegressor(random_state=0, max_subpopulation=2000, n_jobs=-1))


def _knn():
    from sklearn.neighbors import KNeighborsRegressor

    # "What happened on the most similar days in the past." No fitted form at all, so it says
    # something the parametric models cannot: whether near-duplicate days recur in this series.
    # Scaled through _linear_pipeline, because a distance over unscaled features is dominated by
    # whichever column happens to be largest.
    return _linear_pipeline(KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1))


def _kernel_ridge():
    from sklearn.kernel_ridge import KernelRidge

    # Ridge on a curved feature space rather than a straight one, so it can bend where the
    # linear models cannot without splitting the way the trees do. Also scaled: an RBF kernel
    # is a distance, with the same sensitivity as KNN above.
    return _linear_pipeline(KernelRidge(kernel="rbf", alpha=1.0))


def _decision_tree_l1():
    from sklearn.tree import DecisionTreeRegressor

    # One tree, on absolute error. The boosted entries are ensembles whose reasoning cannot be
    # read; this one can be printed and shown to somebody who has to defend the number, which
    # is the only reason to keep a single tree on the shelf.
    return DecisionTreeRegressor(criterion="absolute_error", random_state=0,
                                 min_samples_leaf=MIN_SAMPLES_PER_LEAF)


def _adaboost():
    from sklearn.ensemble import AdaBoostRegressor

    # Boosting that reweights hard ROWS, where the gradient-boosted entries fit the residual.
    # On a series whose informative days are rare and extreme those are different behaviours,
    # and only measurement will say which suits it.
    return AdaBoostRegressor(n_estimators=300, learning_rate=0.05, loss="linear",
                             random_state=0)


def _random_forest():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1,
                                 min_samples_leaf=MIN_SAMPLES_PER_LEAF)


def _extra_trees():
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=-1,
                               min_samples_leaf=MIN_SAMPLES_PER_LEAF)


def _hist_gbdt():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=0, min_samples_leaf=20,
                                         l2_regularization=1.0)


def _hist_gbdt_l1():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(loss="absolute_error", random_state=0,
                                         min_samples_leaf=20, l2_regularization=1.0)


def _gbdt_l1():
    from sklearn.ensemble import GradientBoostingRegressor

    # The exact-split sibling of HistGBDT_L1. Slower, and on a few thousand rows that does
    # not matter; it splits on actual feature values rather than on a 255-bin histogram,
    # which is worth measuring on a series whose informative days are the extreme ones.
    return GradientBoostingRegressor(loss="absolute_error", random_state=0,
                                     min_samples_leaf=MIN_SAMPLES_PER_LEAF)


def _xgb_params() -> Dict:
    return dict(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5.0, reg_lambda=1.0,
                random_state=0, tree_method="hist", n_jobs=-1)


def _xgboost():
    from xgboost import XGBRegressor

    return XGBRegressor(**_xgb_params())


def _xgboost_l1():
    from xgboost import XGBRegressor

    # reg:absoluteerror needs XGBoost >= 1.7. Where it is missing the model is omitted
    # rather than falling back to squared error, which would put an L2 fit in the table
    # under an L1 name.
    return XGBRegressor(objective="reg:absoluteerror", **_xgb_params())


def _lgbm_params() -> Dict:
    return dict(n_estimators=800, learning_rate=0.05, num_leaves=31, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
                random_state=0, n_jobs=-1, verbose=-1)


def _lightgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**_lgbm_params())


def _lightgbm_l1():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(objective="l1", **_lgbm_params())


def _catboost_params() -> Dict:
    return dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                random_seed=0, verbose=False, allow_writing_files=False)


def _catboost_l1():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(loss_function="MAE", **_catboost_params())


def _catboost_quantile():
    from catboost import CatBoostRegressor

    # Quantile loss at alpha=0.5 is absolute error, so this is the interval-capable sibling
    # rather than a different model class.
    return CatBoostRegressor(loss_function="Quantile:alpha=0.5", **_catboost_params())


# ---------------------------------------------------------------------------
# THE CATALOGUE
#
# One entry per model. To add one, add an entry here and a factory above if the estimator
# is not already constructible. Everything downstream -- rolling-origin evaluation, both
# baselines, the sentinel, the ledger and the gates -- is inherited, because every one of
# them works from the estimator this returns rather than from a per-model branch.
# ---------------------------------------------------------------------------

MODELS: Tuple[ModelSpec, ...] = (
    # ── linear ──────────────────────────────────────────────────────────────
    ModelSpec("Ridge", FAMILY_ML, factory=_ridge, added="2026-03",
              summary="A straight-line fit that keeps every input but shrinks the weight "
                      "on each, so no single input dominates. Fast, and the easiest model "
                      "here to explain to somebody who has to defend the number."),
    ModelSpec("Lasso", FAMILY_ML, factory=_lasso, added="2026-03",
              summary="A straight-line fit that drives the weight on unhelpful inputs to "
                      "zero, so it selects its own inputs. Useful when many candidate "
                      "inputs are suspected to carry nothing."),
    ModelSpec("ElasticNet", FAMILY_ML, factory=_elastic_net, added="2026-03",
              summary="A straight-line fit that blends the two behaviours above. It "
                      "shrinks every weight and drops some entirely."),
    ModelSpec("Huber", FAMILY_ML, factory=_huber, added="2026-08-19",
              summary="A straight-line fit that stops chasing extreme days. This series "
                      "has single days ten times the local level, and an ordinary linear "
                      "fit is dragged toward them at the cost of the ordinary days it will "
                      "mostly be judged on."),
    ModelSpec("BayesianRidge", FAMILY_ML, factory=_bayesian_ridge, added="2026-08-19",
              summary="A straight-line fit that works out for itself how much to shrink each "
                      "weight, rather than being told. There is no setting to choose, so it "
                      "cannot be tuned into looking better than it is."),
    ModelSpec("TheilSen", FAMILY_ML, factory=_theil_sen, added="2026-08-19",
              summary="A straight-line fit that takes the middle answer from many small "
                      "subsets of the history. Where the model above it de-weights extreme "
                      "days one at a time, this one is unmoved by a whole run of them, which "
                      "is what a month-end looks like here."),

    # ── distance and kernels ────────────────────────────────────────────────
    ModelSpec("KNN", FAMILY_ML, factory=_knn, added="2026-08-19",
              summary="Looks up the most similar days in the past and averages what happened "
                      "on them. It fits no formula at all, so it answers a different question: "
                      "whether days like today have happened before."),
    ModelSpec("KernelRidge", FAMILY_ML, factory=_kernel_ridge, added="2026-08-19",
              summary="A fit that is allowed to curve rather than being held straight, while "
                      "still shrinking its weights. Sits between the straight-line models and "
                      "the trees."),

    # ── trees and forests ───────────────────────────────────────────────────
    ModelSpec("DecisionTree_L1", FAMILY_ML, factory=_decision_tree_l1, added="2026-08-19",
              summary="A single decision tree, trained to minimise absolute error. The only "
                      "model here whose reasoning can be printed and read, which matters when "
                      "somebody has to defend a number rather than just quote it."),
    ModelSpec("RandomForest", FAMILY_ML, factory=_random_forest, added="2026-03",
              summary="Many decision trees grown on different samples of the history, "
                      "averaged. Captures interactions between inputs without being told "
                      "to look for them."),
    ModelSpec("ExtraTrees", FAMILY_ML, factory=_extra_trees, added="2026-03",
              summary="Like a random forest, but each tree splits at random points rather "
                      "than at the best one. Faster, and less prone to fitting the "
                      "training history too closely."),

    # ── boosting ────────────────────────────────────────────────────────────
    ModelSpec("AdaBoost", FAMILY_ML, factory=_adaboost, added="2026-08-19",
              summary="Trees built one after another, where each one pays more attention to "
                      "the days the previous ones got most wrong. The other boosted entries "
                      "instead fit what is left over, which is a different behaviour on a "
                      "series whose informative days are rare."),
    ModelSpec("HistGBDT", FAMILY_ML, factory=_hist_gbdt, added="2026-03",
              summary="Trees built one after another, each correcting what the previous "
                      "ones got wrong, on a binned view of the inputs for speed. Fits the "
                      "average day."),
    ModelSpec("HistGBDT_L1", FAMILY_ML, factory=_hist_gbdt_l1, added="2026-05",
              summary="The same method, trained to minimise absolute error rather than "
                      "squared error. That fits the typical day instead of the average "
                      "one, which matters on a series a few huge days can distort."),
    ModelSpec("GBDT_L1", FAMILY_ML, factory=_gbdt_l1, added="2026-08-19",
              summary="Boosted trees trained on absolute error, splitting on actual "
                      "values rather than on a binned approximation. Slower than the "
                      "binned version, and worth measuring on a series whose informative "
                      "days are the extreme ones."),
    ModelSpec("XGBoost", FAMILY_ML, factory=_xgboost, requires=("xgboost",), added="2026-03",
              summary="A widely used boosted-tree implementation, fitting the average day."),
    ModelSpec("XGBoost_L1", FAMILY_ML, factory=_xgboost_l1, requires=("xgboost",),
              added="2026-05",
              summary="The same implementation trained on absolute error, so it fits the "
                      "typical day rather than the average one."),
    ModelSpec("LightGBM", FAMILY_ML, factory=_lightgbm, requires=("lightgbm",), added="2026-03",
              summary="A boosted-tree implementation that grows trees leaf by leaf rather "
                      "than level by level, which often reaches the same accuracy with "
                      "fewer trees."),
    ModelSpec("LightGBM_L1", FAMILY_ML, factory=_lightgbm_l1, requires=("lightgbm",),
              added="2026-05",
              summary="The same implementation trained on absolute error. It is the "
                      "champion recipe on both flow lines."),
    ModelSpec("CatBoost_L1", FAMILY_ML, factory=_catboost_l1, requires=("catboost",),
              added="2026-08",
              summary="A boosted-tree implementation built around ordered boosting, which "
                      "is designed to reduce the way a model leaks information about its "
                      "own training rows. Trained here on absolute error."),
    ModelSpec("CatBoost_Quantile", FAMILY_ML, factory=_catboost_quantile,
              requires=("catboost",), added="2026-08",
              summary="The same implementation fitted to the middle quantile, which is the "
                      "same objective as absolute error and can also produce a range "
                      "rather than a single number."),
)

#: Names the A_STAT runner dispatches on. Listed so one catalogue can describe the whole
#: shelf; the estimators themselves are not scikit-learn objects and are built inside that
#: runner, which is why these carry no factory.
STAT_MODEL_NAMES: Tuple[str, ...] = (
    "NAIVE", "WEEKDAY_MEAN", "MOVAVG", "ETS", "ETS_DAMPED", "SARIMAX", "STL_ARIMA", "THETA",
    # Added 2026-08-19. `test_every_stat_name_in_the_catalogue_is_dispatchable` asserts this
    # tuple equals `run_a_stat.A_STAT_MODELS`, so a name added here without a dispatch branch
    # fails rather than falling through to a naive forecast.
    "SES", "HOLT",
)

#: Names the E_QUANTILE runner dispatches on. Same purpose and same contract as the tuple above.
#:
#: Added because the frontend had no way to ask. ``e_quantile_daily_pipeline`` imports sklearn, so
#: the Streamlit interpreter cannot import it to enumerate the family, and the Lab's quantile
#: picker was therefore a hardcoded list that had drifted to a single entry while the family
#: offered three. ``test_model_catalog.py`` asserts this matches ``registry_models()``.
QUANTILE_MODEL_NAMES: Tuple[str, ...] = (
    "GBQuantile", "ResidualRF", "LGBMQuantile",
    "LinearQuantile", "HistGBQuantile", "XGBQuantile",
)

#: Entries that are references rather than candidates.
#:
#: Declared here rather than read from ``run_a_stat.model_roles()``, which is the authority,
#: because importing that runner pulls in matplotlib and statsmodels and this module has to
#: stay readable from the Streamlit interpreter. ``test_model_catalog.py`` asserts the two
#: agree, so the duplication cannot drift without a test failing.
BASELINE_NAMES: Tuple[str, ...] = ("NAIVE", "WEEKDAY_MEAN", "MOVAVG")


def specs(family: Optional[str] = None) -> List[ModelSpec]:
    """Every catalogued model, optionally restricted to one family."""
    return [s for s in MODELS if family is None or s.family == family]


def spec_for(name: str) -> Optional[ModelSpec]:
    return next((s for s in MODELS if s.name == name), None)


# ---------------------------------------------------------------------------
# Derived status
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _measured() -> Dict[str, Tuple[str, ...]]:
    """``{model_name: targets it has been measured on}``, read from the run ledger.

    Cached: the ledger is a committed file that does not change while a page is open, and
    this reads one sidecar JSON per logged run.
    """
    out: Dict[str, set] = {}
    if not LOG_CSV.exists():
        return {}
    with LOG_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        sidecar = RUNS_DIR / f"{row.get('run_id', '')}.json"
        if not sidecar.exists():
            continue
        try:
            model = str(json.loads(sidecar.read_text(encoding="utf-8")).get("model") or "")
        except (OSError, ValueError):
            continue
        if not model:
            continue
        base = model.split("+")[0].strip()
        for suffix in ("_recomputed", "_relogged", "_reproduction"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        out.setdefault(base.strip("_ "), set()).add(str(row.get("target", "")))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def measured_targets(name: str) -> Tuple[str, ...]:
    """Which targets this model has a logged run for. Empty means it has never been run."""
    return _measured().get(name, ())


def status_of(name: str) -> str:
    """``evaluated``, ``untested`` or ``unavailable``, derived rather than declared.

    A declared status is a claim that drifts the moment a model is added and not run. This
    asks the ledger instead, so "is there a measured number for this" is answered by the
    record of measurements rather than by a field somebody typed.
    """
    if name in BASELINE_NAMES:
        return BASELINE
    spec = spec_for(name)
    if spec is not None and not spec.installed:
        return UNAVAILABLE
    return EVALUATED if measured_targets(name) else UNTESTED


def is_gate_eligible(name: str) -> bool:
    """Can this model be considered for publication?

    Only if something has been measured for it. This is not a second gate: the publication
    gates already refuse to pass a check they have no measurement for, and this states the
    consequence at the point a page is deciding whether to offer the model.
    """
    return status_of(name) == EVALUATED


def untested_badge(name: str) -> str:
    """The one sentence shown beside an untested model. Never a bare label."""
    if name in BASELINE_NAMES:
        return ("This is one of the reference rules every model here is measured against, "
                "not a candidate to publish. It has no measured result because there would "
                "be nothing to compare it to.")
    spec = spec_for(name)
    if spec is not None and not spec.installed:
        missing = ", ".join(pkg for pkg in spec.requires if find_spec(pkg) is None)
        return (f"This model needs the {missing} package, which is not installed here. "
                f"It cannot be run until it is.")
    return ("No measured result has been recorded for this model on any target, so there is "
            "no number to show for it and nothing for the publication checks to read. You "
            "can run it as an experiment, and it cannot become the model behind an official "
            "forecast until a result has been recorded.")


def shelf(family: Optional[str] = None) -> List[Dict]:
    """The whole shelf as plain dictionaries, for a page to render.

    Deliberately free of estimators: the Streamlit interpreter has pandas and nothing else,
    and building a LightGBM object to display its name would put the modelling stack in the
    wrong process.
    """
    out = []
    for spec in specs(family):
        status = status_of(spec.name)
        out.append({
            "name": spec.name,
            "family": spec.family,
            "summary": spec.summary,
            "status": status,
            "installed": spec.installed,
            "requires": list(spec.requires),
            "added": spec.added,
            "measured_on": list(measured_targets(spec.name)),
            "gate_eligible": status == EVALUATED,
            "badge": "" if status == EVALUATED else untested_badge(spec.name),
        })
    return out
