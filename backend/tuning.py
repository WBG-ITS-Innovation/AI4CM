"""Workstream 2 — LightGBM/CatBoost quantile models, crossing-safe, tuned with Optuna.

Two pieces that belong together because the second exists to configure the first.

--------------------------------------------------------------------------------
CROSSING SAFETY
--------------------------------------------------------------------------------
Quantiles fitted independently can cross: a p90 below a p50 is not a wide interval, it is an
invalid one, and every coverage statistic computed from it is meaningless rather than merely
pessimistic. ``crossing_safe`` sorts each row's quantile predictions, which is the minimal
repair that cannot make an interval worse. ``count_crossings`` reports how often it had to
act — a model that crosses constantly is misconfigured, and silently sorting would hide that,
so the count is logged rather than swallowed.

--------------------------------------------------------------------------------
H-GAPPED EARLY STOPPING
--------------------------------------------------------------------------------
Early stopping needs a validation slice, and on an h-step-ahead problem the naive slice leaks.
Row *t* carries the target *y(t+h)*, so the last *h* training rows have targets that fall
inside the validation block. Their answers are therefore visible to the model through the
validation set it is stopping against.

``gapped_split`` drops those *h* rows entirely — a gap, not a boundary. Without it the
stopping decision is made against partially-seen data and the chosen tree count is optimistic.
Measured on this data the gap is 5 rows out of ~2,000, so it costs almost nothing and removes
the whole argument.

--------------------------------------------------------------------------------
WHAT IS TUNED, AND ON WHAT
--------------------------------------------------------------------------------
The objective is **mean absolute error of the median prediction, averaged over
TRAIN-internal rolling-origin folds**. That is what the project reports, so it is what gets
optimised; tuning pinball loss and reporting MAE would be optimising a different thing than
we publish. Pinball at the median *is* absolute error, so the two agree at q=0.50 anyway.

TRAIN-internal only. DEV is touched exactly once, for confirmation, after the search closes.
TEST is never touched.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

QUANTILES: Tuple[float, ...] = (0.10, 0.50, 0.90)

#: Trees are capped high and controlled by early stopping rather than by the cap.
MAX_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 60

#: Fraction of the training slice held back for early stopping, before the gap is removed.
VALID_FRACTION = 0.15


def crossing_safe(q_preds: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    """Sort each row's quantile predictions so they are non-decreasing in q.

    The minimal honest repair: it cannot widen an interval incorrectly, and it cannot turn a
    valid interval invalid. Reordering is preferred to clipping because clipping would move a
    prediction to a value no model produced.
    """
    if not q_preds:
        return {}
    qs = sorted(q_preds)
    mat = np.column_stack([np.asarray(q_preds[q], dtype=float) for q in qs])
    mat = np.sort(mat, axis=1)
    return {q: mat[:, i] for i, q in enumerate(qs)}


def count_crossings(q_preds: Dict[float, np.ndarray]) -> int:
    """Rows where the raw quantile predictions were out of order.

    Reported rather than swallowed: a model that crosses on most rows is misconfigured, and
    the sort would otherwise make that invisible.
    """
    if len(q_preds) < 2:
        return 0
    qs = sorted(q_preds)
    mat = np.column_stack([np.asarray(q_preds[q], dtype=float) for q in qs])
    return int((np.diff(mat, axis=1) < 0).any(axis=1).sum())


def gapped_split(n: int, horizon: int, valid_fraction: float = VALID_FRACTION
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Indices for (fit, validation) with an ``horizon``-row gap between them.

    Row *t* holds the target *y(t+h)*, so without the gap the final *h* fit rows have answers
    lying inside the validation block. Early stopping would then be tuned against data the
    model has partially seen.
    """
    if n <= horizon + 2:
        raise ValueError(f"cannot build a gapped split from {n} rows at horizon {horizon}")
    n_valid = max(horizon + 1, int(round(n * valid_fraction)))
    n_valid = min(n_valid, n - horizon - 2)
    valid_start = n - n_valid
    fit_end = valid_start - horizon          # the gap
    if fit_end <= 1:
        raise ValueError(f"gap of {horizon} leaves no fitting rows out of {n}")
    return np.arange(fit_end), np.arange(valid_start, n)


# ── model factories ───────────────────────────────────────────────────────────

def _lgbm_quantile(alpha: float, params: Dict, horizon: int):
    from lightgbm import LGBMRegressor
    p = dict(objective="quantile", alpha=float(alpha), n_estimators=MAX_ESTIMATORS,
             random_state=0, n_jobs=-1, verbose=-1)
    p.update({k: v for k, v in params.items() if k != "model"})
    return LGBMRegressor(**p)


def _catboost_quantile(alpha: float, params: Dict, horizon: int):
    from catboost import CatBoostRegressor
    p = dict(loss_function=f"Quantile:alpha={float(alpha)}", iterations=MAX_ESTIMATORS,
             random_seed=0, verbose=False, allow_writing_files=False)
    p.update({k: v for k, v in params.items() if k != "model"})
    return CatBoostRegressor(**p)


def _catboost_l1(params: Dict, horizon: int):
    from catboost import CatBoostRegressor
    p = dict(loss_function="MAE", iterations=MAX_ESTIMATORS, random_seed=0,
             verbose=False, allow_writing_files=False)
    p.update({k: v for k, v in params.items() if k != "model"})
    return CatBoostRegressor(**p)


def fit_quantiles(model: str, X_tr, y_tr, X_te, params: Dict, horizon: int,
                  quantiles: Sequence[float] = QUANTILES) -> Tuple[Dict[float, np.ndarray], int]:
    """Fit one model per quantile with h-gapped early stopping; return crossing-safe preds.

    Returns ``(preds, n_crossings_before_repair)`` so the caller can log how often the raw
    quantiles were out of order.
    """
    X_tr = pd.DataFrame(X_tr)
    y_tr = np.asarray(y_tr, dtype=float)
    fit_ix, val_ix = gapped_split(len(X_tr), horizon)

    raw: Dict[float, np.ndarray] = {}
    for q in quantiles:
        if model == "LGBMQuantile":
            est = _lgbm_quantile(q, params, horizon)
            import lightgbm as lgb
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_tr.iloc[fit_ix], y_tr[fit_ix],
                        eval_set=[(X_tr.iloc[val_ix], y_tr[val_ix])],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        elif model == "CatBoostQuantile":
            est = _catboost_quantile(q, params, horizon)
            est.fit(X_tr.iloc[fit_ix], y_tr[fit_ix],
                    eval_set=(X_tr.iloc[val_ix], y_tr[val_ix]),
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        else:
            raise ValueError(f"unknown quantile model {model!r}")
        raw[float(q)] = np.asarray(est.predict(pd.DataFrame(X_te)), dtype=float)

    return crossing_safe(raw), count_crossings(raw)


def fit_point(model: str, X_tr, y_tr, X_te, params: Dict, horizon: int) -> np.ndarray:
    """Point (median) fit with h-gapped early stopping."""
    X_tr = pd.DataFrame(X_tr)
    y_tr = np.asarray(y_tr, dtype=float)
    fit_ix, val_ix = gapped_split(len(X_tr), horizon)

    if model == "LGBMQuantile":
        est = _lgbm_quantile(0.50, params, horizon)
        import lightgbm as lgb
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_tr.iloc[fit_ix], y_tr[fit_ix],
                    eval_set=[(X_tr.iloc[val_ix], y_tr[val_ix])],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    elif model in ("CatBoostQuantile", "CatBoost_L1"):
        est = (_catboost_quantile(0.50, params, horizon) if model == "CatBoostQuantile"
               else _catboost_l1(params, horizon))
        est.fit(X_tr.iloc[fit_ix], y_tr[fit_ix],
                eval_set=(X_tr.iloc[val_ix], y_tr[val_ix]),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    else:
        raise ValueError(f"unknown model {model!r}")
    return np.asarray(est.predict(pd.DataFrame(X_te)), dtype=float)


# ── search space ──────────────────────────────────────────────────────────────

def suggest_params(trial, model: str) -> Dict:
    """Search space per model family. Ranges are conventional, not data-peeked."""
    if model == "LGBMQuantile":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if model in ("CatBoostQuantile", "CatBoost_L1"):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "bootstrap_type": "Bernoulli",
        }
    raise ValueError(f"no search space for {model!r}")


@dataclass
class FoldData:
    """One TRAIN-internal rolling-origin fold, already materialised.

    ``y_tr`` is what the model TRAINS on -- already target-transformed where the recipe uses
    one. ``y_te`` is truth in ORIGINAL units, and ``inverse`` maps a prediction back to those
    units. Keeping the two separate is what stops the objective from optimising MAE in
    transformed space, which is a different quantity from the MAE we publish: on Revenues the
    ratio transform divides by a ~5e7 level, so an error of 0.1 in ratio space and an error of
    0.1 in lari differ by seven orders of magnitude.
    """

    X_tr: pd.DataFrame
    y_tr: np.ndarray                 # training target, possibly transformed
    X_te: pd.DataFrame
    y_te: np.ndarray                 # truth, ORIGINAL units
    inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None


def objective_mae(folds: Sequence[FoldData], model: str, params: Dict,
                  horizon: int) -> float:
    """Mean MAE of the median prediction across folds, in ORIGINAL units."""
    errs: List[float] = []
    for f in folds:
        pred = fit_point(model, f.X_tr, f.y_tr, f.X_te, params, horizon)
        if f.inverse is not None:
            pred = np.asarray(f.inverse(pred), dtype=float)
        errs.append(float(np.mean(np.abs(f.y_te - pred))))
    return float(np.mean(errs))


def run_study(folds: Sequence[FoldData], models: Sequence[str], horizon: int,
              n_trials: int, seed: int = 0,
              progress: Optional[Callable[[int, float, str], None]] = None) -> Dict:
    """Optuna search across the given model families. Returns the best configuration.

    The model family is itself a search dimension, so the study allocates trials between
    LightGBM and CatBoost rather than splitting the budget evenly by assumption.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _obj(trial):
        model = trial.suggest_categorical("model", list(models))
        params = suggest_params(trial, model)
        val = objective_mae(folds, model, params, horizon)
        if progress:
            progress(trial.number, val, model)
        return val

    study.optimize(_obj, n_trials=n_trials, catch=(Exception,))
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError("every Optuna trial failed; refusing to report a best model")
    best = study.best_trial
    return {
        "best_value": float(best.value),
        "best_model": best.params["model"],
        "best_params": {k: v for k, v in best.params.items() if k != "model"},
        "n_trials_requested": n_trials,
        "n_trials_completed": len(completed),
        "n_trials_failed": len(study.trials) - len(completed),
        "trials": [{"number": t.number, "value": t.value,
                    "model": t.params.get("model")} for t in completed],
    }
