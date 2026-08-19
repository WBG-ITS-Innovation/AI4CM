"""Evaluate a registry champion over the sealed window, for reporting only.

Why this module has to exist
---------------------------
Nothing in the repository could measure a champion recipe on the holdout:

* ``b_ml_pipeline`` has a sealed-window evaluation path but **no ``target_transform``**. The
  Revenues champion is *defined* by ``target_transform: "ratio"`` — ratio-to-trailing-level with a
  63-business-day causal median divisor, the WS4 winner — so running ``LightGBM_L1`` through that
  pipeline produces a different model wearing the champion's name.
* ``ws2_tune.design`` does implement the recipe, but its ``make_folds`` is a **selection** path: it
  accepts only train/dev and calls ``assert_selection_free`` on evaluation rows, refusing TEST by
  construction. Widening it would put holdout dates through the one function whose comment says a
  stale window "does the most damage".

So the recipe is reused and the *fold construction* is rewritten here, as reporting: it logs, it
never selects, and it embargoes.

--------------------------------------------------------------------------------
WHAT THESE FIGURES ARE, AND ARE NOT
--------------------------------------------------------------------------------
They are a measurement of the champion **recipe** — same feature groups, same exogenous blocks,
same target transform, same hyperparameters, read from ``registry/recipes.json``.

They are **not** a continuation of the champion's logged DEV credentials, and must not be
presented as one. Reconstructing those credentials from the repository does not reproduce them:

    target        recomputed DEV MAE      logged      delta       n
    Revenues              38,044,471  38,931,956      2.28%   250 vs 262
    Expenditure           47,165,998  51,602,951      8.60%   250 vs 262

The reason is that the script which produced them is gone. Those runs are stamped
``feature_names=['ws4:ratio']`` / ``['ws4:raw']`` with ``fold_scheme="DEV confirmation, single 2024
fold"`` — a string no current script emits — ``per_fold: []``, and a single recorded feature name.
There is no ``scripts/ws4_*.py``; only the two WS4 reports survive. So the credentials are not
reproducible, and figures from this module **supersede** rather than extend them.
:func:`dev_reconstruction` recomputes the DEV figure alongside, so the gap is always visible next
to any sealed-window number rather than left for a reader to discover.

--------------------------------------------------------------------------------
THE EMBARGO, WHICH THE BORROWED GEOMETRY DOES NOT HAVE
--------------------------------------------------------------------------------
``build_yearly_folds`` sets ``train_end`` to the last business day of the year before the
evaluation block, and ``design()`` builds ``y_t = s.shift(-H)`` over the whole series. So the last
``H`` training origins carry targets that fall *inside* the evaluation block: measured, 5 rows on
the sealed window, whose targets run from 2025-01-01. Training on them would fit the model to
answers from the window it is about to be scored on.

``make_folds`` in ``ws2_tune`` does not remove them, so its DEV folds carry the same 5-row leak.
That is recorded as a finding rather than changed here — it is a selection path, and altering what
the tuner trains on is not an artifact-reporting change. This module drops them:
:func:`sealed_folds` keeps a training origin only if its target date is strictly before the first
evaluation origin.

--------------------------------------------------------------------------------
A SECOND, WORSE FINDING THE LEDGER CALL EXPOSED
--------------------------------------------------------------------------------
Folds are bounded by ORIGIN, but truth is read at ``origin + H``. So DEV origins in late December
2024 are scored against target dates in early January 2025 — **inside the sealed holdout**.

``ws2_tune.make_folds`` calls ``assert_selection_free`` on the evaluation *origins*, which are all
DEV, and then scores against those targets. The guard passes because it is checking the wrong
dates. Measured: 250 DEV origins, 4 with holdout targets (2025-01-01, 01-02, 01-03, 01-06) — the
same year boundary the Ops vintage construction had to handle.

So each champion's DEV credential in ``registry/recipes.json`` includes 4 holdout observations.
Impact on the DEV MAE: Revenues **0.82%**, Expenditure **1.16%**, stock target **0.51%**. Small in
magnitude, but it is *selection* on holdout data, which is the single thing the four-window split
exists to prevent.

Not fixed here — correcting a selection path changes what the tuner optimises and what the
registry's credentials mean. ``test_sealed_window_report.py`` pins it so it cannot be lost, and
this module's own DEV reconstruction keeps those rows so its comparison against the credential is
like-for-like, while logging the read.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent
for _p in (str(BACKEND), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation_windows import (PURPOSE_REPORT, TEST_END,                 # noqa: E402
                               TEST_START, require_test_access, window_for)

#: Horizon the recipes are credentialed at.
HORIZON = 5


@dataclass
class SealedFold:
    """One reporting fold. Origins, not row positions; truth in ORIGINAL units."""

    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    X_tr: pd.DataFrame
    y_tr: np.ndarray                       # transformed where the recipe uses a transform
    X_te: pd.DataFrame
    y_te: np.ndarray                       # truth, original units
    origin_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    origin_values: np.ndarray              # h-step persistence, i.e. the shared ruler
    inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None
    n_embargoed: int = 0


def _target_date_map(index: pd.DatetimeIndex, horizon: int) -> Dict[pd.Timestamp, pd.Timestamp]:
    """Origin -> target date, positionally, exactly as ``design()`` builds ``shift(-H)``."""
    return {d: index[i + horizon] for i, d in enumerate(index) if i + horizon < len(index)}


def sealed_folds(target: str, eval_start: str = TEST_START, eval_end: str = TEST_END,
                 horizon: int = HORIZON, log: bool = True) -> Tuple[List[SealedFold], Dict]:
    """Rolling-origin folds over ``[eval_start, eval_end]``, embargoed and logged.

    Reporting only. Nothing here chooses a model, a parameter or a threshold, so
    ``assert_selection_free`` is deliberately *not* the right guard — a logged
    ``PURPOSE_REPORT`` read is, and that is what every other family's reporting path uses.
    """
    from b_ml_pipeline import build_yearly_folds
    from ws2_tune import design

    s, X, y_t, y_true, tf, lvl, stock = design(target)
    tmap = _target_date_map(s.index, horizon)

    raw_folds = build_yearly_folds(s.index, 4, None, eval_start=eval_start, eval_end=eval_end)
    ok = X.notna().all(axis=1) & y_t.notna()

    out: List[SealedFold] = []
    for tr_end, te_start, te_end in raw_folds:
        ite = X.index[(X.index >= te_start) & (X.index <= te_end) & ok]
        if not len(ite):
            continue

        # THE EMBARGO. A training origin is kept only if its target has already happened by the
        # first evaluation origin. Without this the last `horizon` rows train on answers from
        # inside the block being scored (measured: 5 rows on the sealed window).
        first_origin = ite.min()
        cand = X.index[(X.index <= tr_end) & ok]
        itr = pd.DatetimeIndex([d for d in cand
                                if d in tmap and tmap[d] < first_origin])
        n_embargoed = len(cand) - len(itr)
        if len(itr) < 200:
            continue

        ytr = y_t.loc[itr].to_numpy(dtype=float)
        yte = y_true.loc[ite].to_numpy(dtype=float)
        origin_vals = s.loc[ite].to_numpy(dtype=float)

        inverse = None
        if tf == "ratio":
            ltr = lvl.reindex(itr).to_numpy(dtype=float)
            lte = lvl.reindex(ite).to_numpy(dtype=float)
            ytr = ytr / ltr
            inverse = ((lambda p, l=lte, o=origin_vals: p * l + o) if stock
                       else (lambda p, l=lte: p * l))
        elif stock:
            inverse = (lambda p, o=origin_vals: p + o)

        tds = pd.DatetimeIndex([tmap[d] for d in ite])
        out.append(SealedFold(train_end=tr_end, eval_start=te_start, eval_end=te_end,
                              X_tr=X.loc[itr], y_tr=ytr, X_te=X.loc[ite], y_te=yte,
                              origin_dates=ite, target_dates=tds,
                              origin_values=origin_vals, inverse=inverse,
                              n_embargoed=n_embargoed))

    # Causality, asserted rather than trusted: every training target must predate every
    # evaluation origin in the same fold.
    for f in out:
        latest = max(tmap[d] for d in f.X_tr.index if d in tmap)
        if latest >= f.origin_dates.min():
            raise AssertionError(
                f"{target}: a training target ({latest.date()}) lands at or after the first "
                f"evaluation origin ({f.origin_dates.min().date()}) — the embargo did not hold")

    meta = {"target": target, "transform": tf, "stock": bool(stock), "n_folds": len(out),
            "n_eval_rows": int(sum(len(f.y_te) for f in out)),
            "n_embargoed": int(sum(f.n_embargoed for f in out))}

    if log and out:
        holdout = [d for f in out for d in f.target_dates if window_for(d) == "test"]
        if holdout:
            require_test_access(
                f"Champion sealed-window evaluation for {target!r} at h={horizon} covers "
                f"{len(holdout)} holdout target date(s) from {min(holdout).date()} to "
                f"{max(holdout).date()}",
                caller="sealed_window_report.sealed_folds", purpose=PURPOSE_REPORT)
    return out, meta


def champion_estimator(target: str):
    """The champion's model object, built by the pipeline's own definitions.

    Reused rather than reconstructed: ``available_models`` is where every family gets its
    estimators, so a champion measured here is the same object the pipeline would fit.
    """
    from b_ml_pipeline import available_models
    from registry import recipe_for

    rec = recipe_for(target)
    name = rec["point_model"]
    models = available_models()
    if name not in models:
        raise KeyError(f"{target}: champion model {name!r} is not in available_models(); "
                       f"have {sorted(models)}")
    est = models[name]

    # Parameter fidelity, checked rather than assumed. The pipeline's defaults currently equal
    # the recipe's params for every champion, so taking the estimator from available_models() is
    # exact. If either side moves, this fails instead of quietly measuring a different model
    # under the champion's name -- which is the specific error this whole module exists to avoid.
    declared = {k: v for k, v in (rec.get("params") or {}).items() if v is not None}
    try:
        actual = est.get_params()
    except AttributeError:
        actual = {}
    mismatched = {k: (actual[k], v) for k, v in declared.items()
                  if k in actual and actual[k] != v}
    if mismatched:
        raise AssertionError(
            f"{target}: {name} as built by available_models() does not match the recipe's "
            f"declared parameters: "
            + "; ".join(f"{k} pipeline={a!r} registry={b!r}" for k, (a, b) in mismatched.items())
            + ". Measuring it would report a different model as the champion.")
    return name, est, rec


def evaluate_champion(target: str, eval_start: str = TEST_START, eval_end: str = TEST_END,
                      horizon: int = HORIZON, log: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Fit the champion recipe per fold and predict its evaluation block.

    Returns per-row predictions in ORIGINAL units, plus metadata. Never crowns anything.
    """
    from sklearn.base import clone

    folds, meta = sealed_folds(target, eval_start, eval_end, horizon, log=log)
    name, proto, rec = champion_estimator(target)
    meta.update({"model": name, "recipe_id": rec["id"]})

    rows = []
    for f in folds:
        est = clone(proto)
        est.fit(f.X_tr, f.y_tr)
        pred = np.asarray(est.predict(f.X_te), dtype=float)
        if f.inverse is not None:
            pred = f.inverse(pred)
        rows.append(pd.DataFrame({
            "target": target, "model": name, "recipe_id": rec["id"], "horizon": horizon,
            "origin_date": f.origin_dates, "target_date": f.target_dates,
            "origin_value": f.origin_values, "y_true": f.y_te, "y_pred": pred,
            "n_train_rows": len(f.X_tr), "n_features": f.X_tr.shape[1],
        }))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), meta


def dev_reconstruction(target: str) -> Dict:
    """The champion recipe on DEV, beside its logged credential, so the gap stays visible.

    The credentials are not reproducible (module docstring). Reporting a sealed-window figure
    without this comparison would invite it to be read as continuous with them.
    """
    from evaluation_windows import DEV
    from registry import recipe_for

    preds, meta = evaluate_champion(target, eval_start=DEV.start, eval_end=DEV.end, log=False)
    logged = recipe_for(target)["dev_credentials"]
    if preds.empty:
        return {"target": target, "recomputed_dev_mae": None, "logged_dev_mae": logged["dev_mae"],
                "note": "no DEV fold could be built"}
    mae = float(np.mean(np.abs(preds["y_true"] - preds["y_pred"])))
    return {"target": target, "model": meta["model"],
            "recomputed_dev_mae": mae, "recomputed_n": int(len(preds)),
            "logged_dev_mae": float(logged["dev_mae"]), "logged_n": int(logged["n"]),
            "delta_pct": abs(mae - float(logged["dev_mae"])) / float(logged["dev_mae"]) * 100.0,
            "note": ("the credential's producing script is absent from the repository, so this is "
                     "a reconstruction of the recipe rather than a reproduction of the run")}
