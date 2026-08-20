"""Reference data for the Models page: descriptions, live hyperparameters, measured performance.

Three kinds of content, kept strictly separate because they carry very different authority:

1. **Descriptions** (``DESCRIPTIONS`` below) are prose. They are the only thing here that is not
   derived from code or artifacts, and every one is served with
   ``kind: "general description"`` so the page can label it as such. A description says what a
   model *is*; it never says how well it did.

2. **Hyperparameters** are read **live** from ``b_ml_pipeline.available_models()``. Nothing is
   transcribed: if a value changes in the pipeline, this changes with it. To show only what the
   pipeline actually *sets* — rather than all 20-odd library defaults — each estimator's
   ``get_params()`` is diffed against a baseline of library defaults.

   For a bare estimator that baseline is a fresh instance of its class. For a **Pipeline** it is
   not: ``Pipeline()`` is *empty*, so every prefixed parameter differs from it and the entire
   configuration reads as deliberately chosen. Measured before this was fixed: ``Ridge`` reported
   17 "set" parameters including ``copy_X`` and ``steps``. Each step is therefore compared against
   a fresh instance of its own class, the pipeline's own unprefixed defaults are added, and
   plumbing (``steps``, ``memory``) is excluded.

   Verified against the source: ``HistGBDT_L1`` → 3 of 21 (``loss``, ``l2_regularization``,
   ``random_state``); ``Ridge`` → 2 (``imp__strategy``, ``est__random_state``), correctly omitting
   the ``StandardScaler`` arguments because those are the library defaults; ``Lasso`` → 3, adding
   ``est__max_iter`` but not ``tol``, which *is* the sklearn default.

3. **Measured performance** comes only from ``experiments/log.csv`` and the per-run JSON beside
   it. Every figure is returned with its ``run_id``. Nothing is recomputed here, because a second
   implementation of a published number is how two numbers for one thing appear.

--------------------------------------------------------------------------------
THE COVERAGE CONSTRAINT, MEASURED
--------------------------------------------------------------------------------
Point-model runs write no interval coverage. Counted across the 153 logged runs: **0 of 104**
point-model rows carry ``coverage_high``; only the 6 quantile/CQR rows do. So coverage is returned
as ``None`` for point models and the page renders "not reported". It is not backfilled, not
recomputed, and not inferred from a sibling run — a coverage figure attached to a model that never
produced an interval would be fabrication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent

#: What each hyperparameter controls, in plain language. Keyed by the bare parameter name so it
#: applies across libraries that share a concept under the same name, with a sensible range.
#: Ranges are conventional starting points for a search, not measured optima.
PARAM_MEANING: Dict[str, Dict[str, str]] = {
    # ── added 2026-08-19 with the models that set them ──────────────────────────────────────
    # `test_every_reported_parameter_has_a_plain_language_meaning_or_says_none` fails when a
    # model sets a parameter with no entry here, which is the right guard: a parameter name with
    # no explanation is a dump, not a reference. These four arrived with TheilSen, KNN and
    # KernelRidge and the test caught all four.
    "max_subpopulation": {"controls": "How many small subsets the robust linear fit draws before "
                                      "taking the middle answer. Without a cap the number grows "
                                      "combinatorially with the feature set and the fit stops "
                                      "finishing.", "range": "500 – 10000"},
    "n_neighbors": {"controls": "How many similar past days are averaged to make a prediction. "
                                "Fewer follows the nearest days closely and is noisier; more "
                                "smooths toward the overall average.", "range": "3 – 50"},
    "weights": {"controls": "Whether nearer days count for more than further ones when their "
                            "outcomes are averaged. `distance` weights by closeness, `uniform` "
                            "treats every neighbour alike.", "range": "uniform | distance"},
    "kernel": {"controls": "The shape the fit is allowed to bend into. `rbf` lets it curve "
                           "smoothly around each observation; `linear` holds it straight, which "
                           "would make the model a slower Ridge.", "range": "rbf | linear | poly"},
    "n_estimators": {"controls": "How many trees are built. More trees keep reducing error until "
                                 "they stop helping; they never overfit on their own in a bagged "
                                 "model, but each one costs time.",
                     "range": "100 – 3000"},
    "iterations": {"controls": "CatBoost's name for the number of boosting rounds — the same idea "
                               "as n_estimators.", "range": "100 – 3000"},
    "learning_rate": {"controls": "How much each new tree is allowed to correct the ones before "
                                  "it. Lower learns more carefully and needs more trees.",
                      "range": "0.01 – 0.2"},
    "max_depth": {"controls": "How many times a tree may split. Deeper trees can express finer "
                              "interactions and memorise more readily.", "range": "3 – 10"},
    "depth": {"controls": "CatBoost's tree depth.", "range": "4 – 9"},
    "num_leaves": {"controls": "LightGBM grows leaf-wise rather than level-wise, so this caps "
                               "complexity directly instead of via depth.", "range": "15 – 127"},
    "min_samples_leaf": {"controls": "The fewest observations a leaf may hold. A leaf holding one "
                                     "day has memorised that day.", "range": "5 – 50"},
    "min_child_samples": {"controls": "LightGBM's minimum rows per leaf — the same capacity floor.",
                          "range": "10 – 120"},
    "min_child_weight": {"controls": "XGBoost's minimum summed weight per leaf, its capacity "
                                     "floor.", "range": "1 – 20"},
    "min_data_in_leaf": {"controls": "CatBoost's minimum rows per leaf.", "range": "5 – 100"},
    "subsample": {"controls": "Fraction of rows each tree sees. Below 1.0 the trees disagree more, "
                              "which usually helps.", "range": "0.6 – 1.0"},
    "colsample_bytree": {"controls": "Fraction of features each tree sees, for the same reason.",
                         "range": "0.5 – 1.0"},
    "subsample_freq": {"controls": "How often LightGBM redraws the row sample.", "range": "0 – 5"},
    "reg_lambda": {"controls": "L2 penalty on leaf values — pulls predictions toward the average.",
                   "range": "0.001 – 20"},
    "reg_alpha": {"controls": "L1 penalty on leaf values — can zero them out entirely.",
                  "range": "0.001 – 10"},
    "l2_regularization": {"controls": "HistGBDT's L2 penalty on leaf values.", "range": "0 – 10"},
    "l2_leaf_reg": {"controls": "CatBoost's L2 penalty on leaf values.", "range": "0.5 – 30"},
    "alpha": {"controls": "Strength of the ridge/lasso penalty on coefficients. Higher shrinks "
                          "them harder toward zero.", "range": "0.01 – 100"},
    "l1_ratio": {"controls": "In ElasticNet, how the penalty is split between L1 and L2. 0 is "
                             "pure ridge, 1 is pure lasso.", "range": "0.1 – 0.9"},
    "loss": {"controls": "What the model is optimised to minimise. `absolute_error` fits the "
                         "conditional MEDIAN, which is what this project scores on; squared error "
                         "fits the mean and is dragged by spikes.", "range": "n/a — a choice"},
    "loss_function": {"controls": "CatBoost's name for the objective. `MAE` fits the median; "
                                  "`Quantile:alpha=q` fits that quantile.",
                      "range": "n/a — a choice"},
    "objective": {"controls": "The objective. `l1`/`reg:absoluteerror` fit the median; `quantile` "
                              "with an alpha fits a band edge.", "range": "n/a — a choice"},
    "criterion": {"controls": "The split-quality measure a tree uses.", "range": "n/a — a choice"},
    "random_state": {"controls": "Seed. Fixed so a run reproduces exactly.",
                     "range": "any fixed integer"},
    "random_seed": {"controls": "CatBoost's seed.", "range": "any fixed integer"},
    "n_jobs": {"controls": "Parallel worker count. -1 uses every core. Note: parallelism makes "
                           "results differ in the last decimal places between machines.",
               "range": "-1, or a fixed count for bit-reproducibility"},
    "max_iter": {"controls": "Iteration cap for the linear solvers before they give up "
                             "converging.", "range": "1000 – 50000"},
    "tol": {"controls": "How small a change counts as converged.", "range": "1e-5 – 1e-3"},
    "verbose": {"controls": "Log noise. Silenced so pipeline output stays readable.",
                "range": "n/a"},
    "allow_writing_files": {"controls": "CatBoost writes training logs to disk unless told not "
                                        "to. Disabled to keep run folders clean.", "range": "n/a"},
    "bootstrap_type": {"controls": "How CatBoost samples rows.", "range": "n/a — a choice"},
    "tree_method": {"controls": "XGBoost's split-finding algorithm. `hist` buckets features, "
                                "which is much faster on this many rows.",
                    "range": "n/a — a choice"},
    "strategy": {"controls": "How the imputer fills a missing value.", "range": "n/a — a choice"},
    "with_mean": {"controls": "Whether the scaler centres features.", "range": "n/a"},
    "with_std": {"controls": "Whether the scaler divides by the spread.", "range": "n/a"},
}

#: Plain-language descriptions. GENERAL, not measured — the page labels them as such.
DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "Ridge": {"family": "Linear", "summary":
        "Straight-line regression with a penalty that keeps coefficients small. It can only "
        "express 'the answer is this much of feature A plus that much of feature B', so it cannot "
        "represent an interaction like 'the 15th matters, but only in a month with a holiday'. "
        "Fast, stable, and the honest floor a tree model has to beat."},
    "Huber": {"family": "Linear", "summary":
        "A straight-line fit whose loss stops growing quadratically once a residual gets large, "
        "so a single enormous day pulls it far less than it pulls an ordinary linear fit. This "
        "series has single days ten times the local level, which is exactly the situation the "
        "loss was designed for. Registered as a candidate and not yet measured."},
    "Lasso": {"family": "Linear", "summary":
        "Ridge's cousin with a penalty that can drive coefficients to exactly zero, so it selects "
        "features as it fits. Useful when most inputs are irrelevant, which is often true of a "
        "wide calendar feature set."},
    "ElasticNet": {"family": "Linear", "summary":
        "A blend of ridge and lasso. It keeps lasso's ability to discard features while handling "
        "correlated features more gracefully — and calendar features are heavily correlated, "
        "since day-of-month and business-day-of-month largely say the same thing."},
    # Added 2026-08-19. Every one of these is a candidate with no recorded result, and each
    # says so in its own summary rather than relying on the UNTESTED badge alone: a reader
    # skimming summaries should not have to cross-reference a badge to learn that.
    "BayesianRidge": {"family": "Linear", "summary":
        "Ridge that estimates its own shrinkage from the data instead of being handed a value. "
        "There is no setting to choose, so it cannot be tuned into looking good, which makes it "
        "an honest reading next to Ridge on the same features. Registered as a candidate and not "
        "yet measured."},
    "TheilSen": {"family": "Linear", "summary":
        "A robust straight-line fit that takes the middle answer from many small subsets of the "
        "history. Huber de-weights extreme days one at a time; this is unmoved by a whole run of "
        "them together, which is what a month-end looks like in this series. Registered as a "
        "candidate and not yet measured."},
    "KNN": {"family": "Distance", "summary":
        "Looks up the most similar days in the past and averages what happened on them, weighted "
        "by how similar they were. It fits no formula at all, so it answers a different question "
        "from every other model here: whether days like today have happened before. Registered as "
        "a candidate and not yet measured."},
    "KernelRidge": {"family": "Kernel", "summary":
        "Ridge on a curved feature space rather than a straight one, so it can bend where the "
        "linear models cannot without splitting the way the trees do. Sits between the two, and "
        "is measured on scaled features because its kernel is a distance. Registered as a "
        "candidate and not yet measured."},
    "GBDT_L1": {"family": "Boosted trees", "summary":
        "Boosted trees trained on absolute error, splitting on actual feature values rather than "
        "on the 255-bin histogram its faster sibling uses. Slower, and on a few thousand rows "
        "that costs little; the binning is an approximation precisely at the extreme values this "
        "series carries its information in. Registered as a candidate and not yet measured."},
    "DecisionTree_L1": {"family": "Single tree", "summary":
        "One decision tree, trained on absolute error. It is the only model on this shelf whose "
        "reasoning can be printed and read end to end, which is what matters when somebody has "
        "to defend a number rather than quote it. Expect it to be beaten by the ensembles. "
        "Registered as a candidate and not yet measured."},
    "AdaBoost": {"family": "Boosted trees", "summary":
        "Boosting that reweights the days the previous trees got most wrong, where the "
        "gradient-boosted models instead fit what is left over. On a series whose informative "
        "days are rare and extreme those are genuinely different behaviours. Registered as a "
        "candidate and not yet measured."},
    "RandomForest": {"family": "Bagged trees", "summary":
        "Many deep trees, each grown on a different random sample of rows and features, then "
        "averaged. Averaging independent errors is what makes it robust; it captures interactions "
        "a linear model cannot see, and it will not extrapolate beyond the range of values it was "
        "trained on."},
    "ExtraTrees": {"family": "Bagged trees", "summary":
        "A random forest that also picks its split points at random rather than searching for the "
        "best. More randomness means more diversity between trees and usually a slightly better "
        "average, at the cost of each individual tree being worse."},
    "HistGBDT": {"family": "Boosted trees", "summary":
        "Gradient boosting where each tree corrects the errors of those before it, using bucketed "
        "(histogram) features for speed. Boosting is generally stronger than bagging on tabular "
        "data and is more willing to overfit, which is why the capacity floor matters."},
    "HistGBDT_L1": {"family": "Boosted trees", "summary":
        "HistGBDT trained on ABSOLUTE error rather than squared error. Squared error fits the "
        "conditional mean, which a handful of month-end spikes drag upward; absolute error fits "
        "the median, which those days barely move. This project scores absolute error, so this "
        "model optimises what it is judged on."},
    "XGBoost": {"family": "Boosted trees", "summary":
        "Gradient boosting with strong built-in regularisation and its own handling of missing "
        "values. Long the default choice for tabular problems."},
    "XGBoost_L1": {"family": "Boosted trees", "summary":
        "XGBoost on absolute error (`reg:absoluteerror`), for the same reason as HistGBDT_L1 — "
        "matching the objective to the metric the project reports."},
    "LightGBM": {"family": "Boosted trees", "summary":
        "Gradient boosting that grows trees leaf-wise, expanding wherever the gain is largest "
        "rather than level by level. Usually the fastest of the three boosters and often the most "
        "accurate; the leaf-wise habit makes the leaf-count cap the important control."},
    "LightGBM_L1": {"family": "Boosted trees", "summary":
        "LightGBM on absolute error. This is the champion point model for both flow targets."},
    "CatBoost_L1": {"family": "Boosted trees", "summary":
        "CatBoost on absolute error. It builds symmetric (oblivious) trees, which regularises "
        "differently from the other boosters and is often strong out of the box. Present in the "
        "pool but never ablated on this data, so it is not promotable."},
    "CatBoost_Quantile": {"family": "Boosted trees (quantile)", "summary":
        "CatBoost fitting a chosen quantile instead of a central value, so several fits together "
        "describe a range rather than a point. Also unablated here."},
    "GBQuantile": {"family": "Quantile", "summary":
        "Gradient boosting with a pinball (quantile) loss — one model per quantile, giving a lower "
        "edge, a middle and an upper edge. Its bands were long described here as too narrow on the "
        "largest days. That description came from grouping days by how they turned out, which "
        "understates any range; grouped instead by what was known before the day began, the bands "
        "broadly hold up. The measured figures live in the run artifacts, not here."},
    "ResidualRF": {"family": "Quantile (residual)", "summary":
        "A random forest point forecast, with a band built from the spread of its own out-of-bag "
        "residuals. Distribution-free and cheap, but the band is the same shape everywhere, so it "
        "cannot widen on a day that is genuinely more uncertain."},
    "LGBMQuantile": {"family": "Quantile", "summary":
        "LightGBM with a pinball loss, one fit per quantile, with the three sorted afterwards so "
        "the edges cannot cross. Early stopping leaves a gap the size of the forecast horizon, so "
        "the stopping decision is not made against rows whose answers sit inside the validation "
        "slice."},
    # Added 2026-08-19. The family's three existing members are all tree ensembles, so it could
    # not say whether its band widths are a property of this series or of trees. These widen it
    # on purpose: one linear, one binned, one exact-split.
    "LinearQuantile": {"family": "Quantile", "summary":
        "Quantile regression on a straight line — the only member of this family that is not a "
        "tree. That is the point of it: when every other member produces bands of a similar "
        "width, this one says whether that is the data speaking or the method. Registered as a "
        "candidate and not yet measured."},
    "HistGBQuantile": {"family": "Boosted trees (quantile)", "summary":
        "The binned sibling of the gradient-boosted quantile model: same pinball loss, splitting "
        "on a histogram of the features rather than on their actual values. One fit per quantile, "
        "sorted afterwards so the edges cannot cross. Registered as a candidate and not yet "
        "measured."},
    "XGBQuantile": {"family": "Boosted trees (quantile)", "summary":
        "XGBoost's own quantile objective, one fit per quantile, sorted afterwards so the edges "
        "cannot cross. Needs XGBoost 2.0 or newer; where the package is older the model is "
        "omitted rather than quietly falling back to a mean fit under a quantile name. Registered "
        "as a candidate and not yet measured."},
    # ── A_STAT (backend/run_a_stat.py) ──────────────────────────────────────────────────────
    # These keys use the family's own UPPERCASE dispatch names. They were previously "ETS" and
    # "Theta", which matched nothing the pipeline dispatches on and nothing model_pool()
    # enumerated -- so both descriptions were unreachable from the page and no test noticed
    # (item 6 part 3). The names now come from run_a_stat.registry_models().
    "ETS": {"family": "Statistical", "summary":
        "Exponential smoothing — a weighted average of the past where recent observations count "
        "for more, with optional trend and seasonal terms. Uses only the target's own history."},
    "ETS_DAMPED": {"family": "Statistical", "summary":
        "The same method with the trend damped, so a trend it has picked up flattens out as the "
        "forecast reaches further ahead instead of continuing indefinitely. Usually the safer of "
        "the two at longer horizons, where an undamped trend can run away from the level. "
        "Registered as a candidate and not yet measured."},
    "THETA": {"family": "Statistical", "summary":
        "A classical decomposition method: de-trend the series, forecast the pieces, recombine. "
        "Strong on smooth seasonal series and a well-known competition benchmark."},
    "SES": {"family": "Statistical", "summary":
        "Exponential smoothing with no trend and no seasonal term, so it tracks the level only. "
        "It is the plainest member of this family, and it exists to say how much of the fuller "
        "method's accuracy comes from the level alone. Registered as a candidate and not yet "
        "measured."},
    "HOLT": {"family": "Statistical", "summary":
        "Exponential smoothing with a trend but no seasonal term, and the trend continues at the "
        "same slope rather than flattening out. It sits between the level-only method and the "
        "damped one, so the three together show what the trend and the damping are each worth. "
        "Registered as a candidate and not yet measured."},
    "SARIMAX": {"family": "Statistical", "summary":
        "Seasonal ARIMA with optional external regressors. Models the series through its own "
        "autocorrelation and differencing, and is the only A_STAT model that can take exogenous "
        "inputs."},
    "STL_ARIMA": {"family": "Statistical", "summary":
        "Splits the series into trend, season and remainder (STL), forecasts the remainder with "
        "ARIMA, then recombines. Useful when the seasonal shape is strong and stable."},
    "NAIVE": {"family": "Baseline", "summary":
        "Carry the last observed value forward. This is the reference every other model in the "
        "project is measured against, not a competitor — see the h-step persistence ruler."},
    "WEEKDAY_MEAN": {"family": "Baseline", "summary":
        "Predict each day with the historical average for that weekday. A calendar-only "
        "reference: it knows what Tuesdays look like and nothing else."},
    "MOVAVG": {"family": "Baseline", "summary":
        "Predict the mean of the last N observations (7 by default). A smoothing reference with "
        "no trend or seasonal term."},

    # ── C_DL (backend/c_dl_registry.py) ─────────────────────────────────────────────────────
    # Added in item 6: the artifact validator errored on a published C_DL champion ("MLP") that
    # was in no enumerable pool, so a consumer reading the leaderboard could not look up what won.
    "LSTM": {"family": "Deep learning", "summary":
        "A recurrent network with gated memory, reading the sequence in order and carrying state "
        "forward. The standard sequence baseline."},
    "GRU": {"family": "Deep learning", "summary":
        "A recurrent network like LSTM with a simpler gating scheme — fewer parameters, often "
        "comparable accuracy on short series."},
    "DCNN": {"family": "Deep learning", "summary":
        "A dilated causal convolution stack: each layer looks further back than the last, so a "
        "wide receptive field is reached without recurrence. Causal by construction."},
    "TRANSFORMER": {"family": "Deep learning", "summary":
        "Self-attention over the input window, so any position can attend to any earlier one "
        "directly rather than through carried state."},
    "MLP": {"family": "Deep learning", "summary":
        "A plain feed-forward network over the flattened window. No sequence structure at all, "
        "which makes it the honest floor the sequence models have to beat."},
}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)      # NaN == NaN for this purpose
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def live_hyperparameters(est: Any) -> Dict[str, List[Dict]]:
    """Split an estimator's params into what the pipeline SETS and what it inherits.

    Read live via ``get_params(deep=True)`` and diffed against a fresh instance of the same class.
    Nothing is transcribed, so a change in the pipeline shows up here without an edit.
    """
    params = est.get_params(deep=True)

    # Baseline of library defaults. For a bare estimator that is a fresh instance of its class.
    #
    # For a PIPELINE it is not: `Pipeline()` is EMPTY, so every prefixed parameter differs from it
    # and the whole configuration reads as "set by the pipeline". Measured before the fix: Ridge
    # reported 17 parameters as deliberately chosen, including `copy_X` and `steps`. So each step's
    # estimator is compared against a fresh instance of ITS OWN class instead.
    fresh: Dict[str, Any] = {}
    steps = getattr(est, "steps", None)
    if steps:
        for step_name, step_est in steps:
            try:
                step_fresh = type(step_est)().get_params(deep=True)
            except Exception:
                continue
            for k, v in step_fresh.items():
                fresh[f"{step_name}__{k}"] = v
        # The pipeline's OWN params (verbose, …) are not prefixed and so are missing from the
        # step-derived baseline above; without them `verbose=False` reads as a deliberate choice.
        try:
            for k, v in type(est)(steps=list(steps)).get_params(deep=False).items():
                fresh.setdefault(k, v)
        except Exception:
            pass
    else:
        try:
            fresh = type(est)().get_params(deep=True)
        except Exception:
            fresh = {}

    # Pipeline plumbing, not hyperparameters: these describe how the steps are wired, not how the
    # model behaves, and listing them as configuration would bury the three values that matter.
    STRUCTURAL = {"steps", "memory", "transform_input", "verbose_feature_names_out"}

    def entry(k: str, v: Any) -> Dict:
        bare = k.split("__")[-1]
        meaning = PARAM_MEANING.get(bare, {})
        return {"name": k, "value": ("None" if v is None else str(v)),
                "controls": meaning.get("controls", ""),
                "range": meaning.get("range", "")}

    set_by, inherited = [], []
    for k, v in sorted(params.items()):
        if callable(v) or hasattr(v, "get_params"):
            continue                              # nested estimators, listed separately
        if k in STRUCTURAL:
            continue
        (set_by if not _same(v, fresh.get(k, "<<absent>>")) else inherited).append(entry(k, v))
    return {"set_by_pipeline": set_by, "library_default": inherited,
            "n_total": len(params)}


def model_pool() -> Dict[str, Dict]:
    """Every model the pipelines offer, with availability and live parameters."""
    import sys
    sys.path.insert(0, str(BACKEND))
    import b_ml_pipeline as bml
    import e_quantile_daily_pipeline as eq
    import run_a_stat as astat

    out: Dict[str, Dict] = {}
    for name, est in bml.available_models().items():
        out[name] = {"name": name, "pipeline": "B_ML", "available": True,
                     "class": type(est).__name__,
                     "hyperparameters": live_hyperparameters(est)}
    for name, desc in eq.registry_models().items():
        out.setdefault(name, {"name": name, "pipeline": "E_QUANTILE", "available": True,
                              "class": "(constructed per fold)",
                              "registry_description": desc,
                              "hyperparameters": {"set_by_pipeline": [], "library_default": [],
                                                  "n_total": 0,
                                                  "note": ("This family builds its estimators per "
                                                           "fold rather than holding a configured "
                                                           "instance, so parameters are shown "
                                                           "only where a tuned set was logged.")}})

    # A_STAT builds its models per fold from a single dispatch, like E_QUANTILE, and runs one
    # model per invocation via TG_MODEL_FILTER. `role` distinguishes the three reference
    # baselines from the four statistical forecasters, so a count can exclude the references
    # rather than presenting them as competitors.
    _roles = astat.model_roles()
    for name, desc in astat.registry_models().items():
        out.setdefault(name, {"name": name, "pipeline": "A_STAT", "available": True,
                              "class": "(constructed per fold)",
                              "role": _roles.get(name, "forecast"),
                              "registry_description": desc,
                              "hyperparameters": {"set_by_pipeline": [], "library_default": [],
                                                  "n_total": 0,
                                                  "note": ("This family constructs its model per "
                                                           "fold from TG_PARAM_OVERRIDES, so "
                                                           "there is no configured instance to "
                                                           "introspect.")}})

    # C_DL. Guarded because it imports torch: a machine without it must see the models marked
    # unavailable, not see them disappear -- the same rule as the boosters below.
    from c_dl_registry import registry_models as _cdl_registry
    _cdl_models = _cdl_registry()
    try:
        import c_dl_pipeline  # noqa: F401 - probing whether the family can actually run
        _cdl_available, _cdl_why = True, None
    except Exception as exc:                       # noqa: BLE001 - torch may be absent
        _cdl_available, _cdl_why = False, f"{type(exc).__name__}: {exc}"
    for name, desc in _cdl_models.items():
        entry = {"name": name, "pipeline": "C_DL", "available": _cdl_available,
                 "class": "(torch module, built per fold)" if _cdl_available else "(unavailable)",
                 "registry_description": desc,
                 "hyperparameters": {"set_by_pipeline": [], "library_default": [], "n_total": 0,
                                     "note": ("Architecture sizes are set in make_model() and the "
                                              "training schedule in ConfigDL; neither is a "
                                              "configured estimator instance to introspect.")}}
        if not _cdl_available:
            entry["missing_library"] = _cdl_why
        out.setdefault(name, entry)

    # Models that exist in the code but are unavailable because a library is missing must SAY so
    # rather than vanishing from the page.
    for name, flag, lib in (("XGBoost", bml.HAVE_XGB, "xgboost"),
                            ("XGBoost_L1", bml.HAVE_XGB, "xgboost"),
                            ("LightGBM", bml.HAVE_LGBM, "lightgbm"),
                            ("LightGBM_L1", bml.HAVE_LGBM, "lightgbm"),
                            ("CatBoost_L1", bml.HAVE_CATBOOST, "catboost"),
                            ("CatBoost_Quantile", bml.HAVE_CATBOOST, "catboost")):
        if not flag:
            out[name] = {"name": name, "pipeline": "B_ML", "available": False,
                         "class": "(unavailable)", "missing_library": lib,
                         "hyperparameters": {"set_by_pipeline": [], "library_default": [],
                                             "n_total": 0}}
    # Has anybody recorded a result for this model? Derived from the experiment ledger by
    # `model_catalog`, never declared. Carried on every pool entry so the Models page can
    # badge an untested model rather than showing it beside a champion undifferentiated,
    # which is how Ridge came to be the Lab's default selection with no recorded result.
    from model_catalog import status_of, measured_targets, untested_badge, EVALUATED

    for name, d in out.items():
        d.update(DESCRIPTIONS.get(name, {"family": "—", "summary": ""}))
        d["description_kind"] = "general description, not a measured claim"
        d["status"] = status_of(name)
        d["measured_on"] = list(measured_targets(name))
        d["gate_eligible"] = d["status"] == EVALUATED
        d["status_note"] = "" if d["status"] == EVALUATED else untested_badge(name)
    return out


# ── The client-facing composition, derived rather than asserted ─────────────────────────────
#
# The sentence we give a client used to be a string in a pinned test. When C_DL became
# enumerable in item 6 the pool went from 23 to 28 and the sentence did not move, so for a
# while we were telling clients about a pool that no longer matched the registry and nothing
# detected it. The counts below are computed from `model_pool()`, and `client_framing()`
# writes the sentence from them, so adding a model changes the sentence or fails the test.

#: Which client-facing category a pool entry belongs to, keyed by the pipeline that offers it.
#: A_STAT is split by the `role` the registry records, because three of its seven models are
#: the ruler the others are measured against, and counting a ruler as a rival is the specific
#: error this whole section exists to prevent.
_CATEGORY_BY_PIPELINE = {
    "B_ML": "machine-learning models",
    "C_DL": "deep-learning models",
    "E_QUANTILE": "quantile methods",
}

#: Order the categories appear in the sentence.
CATEGORY_ORDER = ("machine-learning models", "deep-learning models", "statistical models",
                  "quantile methods", "reference baselines")

#: Categories whose models produce point forecasts and are ranked against each other on a
#: target. Quantile methods produce intervals; baselines are the ruler. Neither competes.
COMPETING_CATEGORIES = ("machine-learning models", "deep-learning models", "statistical models")

#: The category a registry recipe draws its `point_model` from. Cross-checked against
#: `registry/recipes.json` by `composition()`, so promoting a model from another family
#: fails rather than quietly widening the pool we describe to a client.
CHAMPION_POOL_CATEGORY = "machine-learning models"


def client_category(entry: Dict) -> str:
    """The client-facing category of one `model_pool()` entry.

    Raises on an unrecognised pipeline **by design**: a fifth family must not be able to
    appear in the pool while silently missing from every number we quote.
    """
    pipeline = entry.get("pipeline")
    if pipeline == "A_STAT":
        return ("reference baselines" if entry.get("role") == "baseline"
                else "statistical models")
    try:
        return _CATEGORY_BY_PIPELINE[pipeline]
    except KeyError:
        raise ValueError(
            f"{entry.get('name')!r} comes from pipeline {pipeline!r}, which has no client-facing "
            f"category. Add it to _CATEGORY_BY_PIPELINE and decide whether it competes "
            f"(COMPETING_CATEGORIES) before quoting any model count to a client."
        ) from None


def composition(pool: Optional[Dict[str, Dict]] = None) -> Dict:
    """Counts by client-facing category, derived from the pool.

    Also derives the two different things the word "champion" means here, because
    conflating them is how a true sentence becomes a wrong one:

    * `champion_pool` — the models a **registry recipe** may promote as its `point_model`,
      i.e. the set an official published forecast is selected from.
    * `daily_best_model_families` — the families `daily_summary.py` writes a per-family
      `best_model` for. Every family that produces a leaderboard is in here, so a consumer
      ranking families (as the Agent does) is choosing across all of them, not across
      `champion_pool`.
    """
    pool = model_pool() if pool is None else pool

    counts: Dict[str, int] = {c: 0 for c in CATEGORY_ORDER}
    members: Dict[str, List[str]] = {c: [] for c in CATEGORY_ORDER}
    for name, entry in pool.items():
        cat = client_category(entry)
        counts[cat] += 1
        members[cat].append(name)
    for names in members.values():
        names.sort()

    champion_pool = sorted(members[CHAMPION_POOL_CATEGORY])

    # Cross-check: every model a recipe actually promotes must be in that pool. This is the
    # check that catches "the registry promoted a C_DL model" before a client is told the
    # eligible pool is the machine-learning one.
    import sys
    sys.path.insert(0, str(BACKEND))
    from registry import load_registry
    promoted = sorted({r["point_model"] for r in load_registry()["recipes"]})
    off_pool = [m for m in promoted if m not in champion_pool]

    # How many of these have a result anybody could quote? Counted, not assumed.
    #
    # Adding this changed the picture the sentence paints. Measured the day it was added,
    # only 8 of the 28 non-baseline entries had a row in experiments/log.csv, so a sentence
    # saying "13 machine-learning models compete on each target" was describing a shelf
    # rather than a body of evidence. The counts below let the sentence say both.
    #
    # The ledger is the right test rather than a harsh one: a registry recipe cites a
    # ledger run_id and registry.verify_against_log checks it, so a model with no ledger
    # row cannot become a champion however many times it has been executed.
    from model_catalog import EVALUATED, status_of

    evaluated, untested = [], []
    for name, entry in pool.items():
        if client_category(entry) == "reference baselines":
            continue
        (evaluated if status_of(name) == EVALUATED else untested).append(name)
    # `client_category` and `model_catalog.BASELINE` must agree about which entries are
    # rulers, or one of the two counts is wrong. Checked here rather than left to trust.
    from model_catalog import BASELINE
    _rulers = {n for n, e in pool.items() if client_category(e) == "reference baselines"}
    assert _rulers == {n for n in pool if status_of(n) == BASELINE}, (
        "the client categories and the model catalogue disagree about which entries are "
        "reference baselines")

    return {
        "counts": counts,
        "members": members,
        "total": sum(counts.values()),
        "competing_total": sum(counts[c] for c in COMPETING_CATEGORIES),
        "champion_pool_category": CHAMPION_POOL_CATEGORY,
        "champion_pool": champion_pool,
        "champion_pool_size": len(champion_pool),
        "evaluated": sorted(evaluated),
        "untested": sorted(untested),
        "evaluated_total": len(evaluated),
        "untested_total": len(untested),
        "promoted_by_registry": promoted,
        "promoted_outside_champion_pool": off_pool,
        "daily_best_model_families": sorted({e["pipeline"] for e in pool.values()}),
    }


def _join(parts: List[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def client_framing(pool: Optional[Dict[str, Dict]] = None) -> str:
    """The composition sentence, written from the derived counts.

    Never a single headline number: the entries are not one kind of thing, and summing them
    presents the ruler and the interval methods as rivals to the point models.
    """
    comp = composition(pool)
    counts = comp["counts"]

    competing = [f"{counts[c]} {c}" for c in COMPETING_CATEGORIES if counts[c]]
    sentence = f"{_join(competing)} compete on each target"
    if counts["quantile methods"]:
        sentence += (f"; prediction intervals come from "
                     f"{counts['quantile methods']} quantile methods")
    if counts["reference baselines"]:
        sentence += (f"; {counts['reference baselines']} further entries are reference "
                     f"baselines, not competitors")
    # The clause that stops the sentence describing a shelf as though it were evidence.
    #
    # "Recorded result" is the precise claim and it is narrower than "has been run": A_STAT
    # runs daily and writes a leaderboard into its run folder without entering anything in
    # experiments/log.csv. What the ledger holds is what a recipe can cite and what the
    # publication checks can read, so a model outside it has no quotable number.
    if comp["untested_total"]:
        sentence += (f". Of those, {comp['evaluated_total']} have a recorded result on at "
                     f"least one target and {comp['untested_total']} are registered "
                     f"candidates with no recorded result yet")
    return sentence + "."


def measured_performance() -> Dict[str, List[Dict]]:
    """Every logged run, grouped by model. Source: experiments/log.csv + the per-run JSON.

    The CSV has no model column, so the model name is read from
    ``experiments/runs/<run_id>.json``. Nothing is recomputed: a second implementation of a
    published number is how one quantity ends up with two values.

    ``coverage_*`` is ``None`` for point-model runs because those runs never wrote it — measured,
    not assumed. See the module docstring.
    """
    import sys
    sys.path.insert(0, str(BACKEND))
    from experiment_log import read_log

    runs_dir = REPO / "experiments" / "runs"
    out: Dict[str, List[Dict]] = {}

    def num(v):
        if v is None or str(v).strip() == "":
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    for r in read_log():
        detail = {}
        jp = runs_dir / f"{r['run_id']}.json"
        if jp.exists():
            try:
                detail = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                detail = {}
        model = detail.get("model") or "(model not recorded)"
        scheme = str(r.get("fold_scheme", ""))
        window = ("DEV (2024)" if "dev" in scheme.lower()
                  else "TRAIN (<=2023)" if "train" in scheme.lower() else "not reported")
        out.setdefault(model, []).append({
            "run_id": r["run_id"],
            "target": r.get("target"),
            "window": window,
            "fold_scheme": scheme,
            "mae": num(r.get("dev_mae")),
            "mase": num(r.get("mase")),
            "skill_vs_ruler_pct": num(r.get("skill_vs_ruler")),
            "sentinel": num(r.get("sentinel_ratio")),
            "coverage_low": num(r.get("coverage_low")),
            "coverage_mid": num(r.get("coverage_mid")),
            "coverage_high": num(r.get("coverage_high")),
            "ruler": num(detail.get("ruler")),
            "calendar_version": r.get("calendar_version") or None,
            "git_sha": r.get("git_sha") or None,
            "data_sha": r.get("data_sha") or None,
            "note": r.get("note") or "",
        })
    for rows in out.values():
        rows.sort(key=lambda d: (str(d["target"]), d["window"], d["run_id"]))
    return out


def champions() -> Dict[str, Dict]:
    """Registry recipes keyed by the model they promote, exactly as stored."""
    import sys
    sys.path.insert(0, str(BACKEND))
    from forecast_integrity import read_gate                      # noqa: F401  (contract import)
    from registry import load_registry

    out: Dict[str, Dict] = {}
    for r in load_registry()["recipes"]:
        out.setdefault(r["point_model"], []).append(r)
    return out


def build() -> Dict:
    """The whole reference payload."""
    return {"models": model_pool(), "performance": measured_performance(),
            "champions": champions(),
            "coverage_note": (
                "Point-model runs do not write interval coverage: measured across the logged runs, "
                "0 of the point-model rows carry a coverage figure and only the quantile/CQR rows "
                "do. Coverage therefore reads 'not reported' for point models. It is not "
                "backfilled or inferred from a sibling run.")}


if __name__ == "__main__":
    print(json.dumps(build(), default=str))
