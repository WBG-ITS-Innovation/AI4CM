"""Production forward forecast — predict dates that do not exist in the data yet.

Everything before this module backtests: it withholds a window whose truth we already
hold and scores against it. This module does the other thing, the one the Treasury
actually needs — it predicts business days **beyond the end of the data**, where there is
no truth to read.

That distinction is the whole safety argument, and it is enforced rather than asserted:

* ``assert_forward_only`` refuses to emit any target date at or before the last date in the
  data. A forward forecast that overlapped the data would be a backtest wearing the wrong
  label, and on this project it would specifically be a 2025 evaluation — which is sealed.
* No truth column is produced. There is nothing to join a ``y_true`` to, so the artifact
  cannot accidentally acquire an accuracy number.

**One model per horizon.** The rest of the project fixes h=5 and scores the fifth business
day. A forecast the Treasury can use has to cover every day between now and then, so for
each horizon h in 1..5 a separate model is fit on rows where ``y(t+h)`` is known and asked
for exactly one prediction from the final origin. Five fits, five dates. Reusing the h=5
model for nearer days would silently misstate what each number means.

Point models come from the workstream-1 winners (``LightGBM_L1`` for the flows,
``HistGBDT_L1`` for the stock) and intervals from ``GBQuantile``, on the workstream-3
winning feature sets plus the ``cross`` exogenous block adopted for the stock in
workstream 5.

**Quality gates are NOT computed here.** Gates need held-out truth, and the only honest
held-out window available is DEV (2024) — TEST stays sealed. The gate verdicts shipped
alongside a forward run therefore come from the DEV credentials run and are attached by
``recipe_id``, never recomputed against forward dates that have no truth.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocessing.fiscal_calendar import calendar_version
from preprocessing.holidays import georgian_holidays_range

#: Horizons emitted, in business days ahead of the final origin.
FORWARD_HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Nominal interval, matching every other interval in the project.
QUANTILES: Tuple[float, ...] = (0.10, 0.50, 0.90)

DEFAULT_OUT = Path(__file__).resolve().parent / "forecast_runs" / "forward" / "latest"


def business_days_after(last: date, n: int) -> List[date]:
    """The next ``n`` Georgian business days strictly after ``last``.

    Weekends and Georgian public holidays are excluded, using the same holiday source as
    preprocessing and the fiscal calendar. Using a naive weekday rule here would put a
    forecast on Mariamoba or Giorgoba, on which the Treasury does not transact.
    """
    hol = {pd.Timestamp(d).date()
           for d in georgian_holidays_range(last, last + timedelta(days=60))}
    out: List[date] = []
    cur = last
    while len(out) < n:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5 and cur not in hol:
            out.append(cur)
    return out


def assert_forward_only(target_dates: Sequence[date], data_end: date) -> None:
    """Every emitted date must lie strictly beyond the data.

    This is the check that keeps a "forward forecast" from quietly becoming an evaluation
    on held-out truth. It is deliberately a hard failure: a forward artifact containing an
    in-sample date is not a degraded forecast, it is a mislabelled one.
    """
    bad = [d for d in target_dates if d <= data_end]
    if bad:
        raise ValueError(
            f"forward forecast produced {len(bad)} target date(s) at or before the data "
            f"end {data_end}: {bad[:5]}. A forward run must predict unseen dates only."
        )


@dataclass
class Champion:
    """The per-target recipe promoted from workstreams 1, 3 and 5."""

    target: str
    point_model: str
    fiscal_groups: Tuple[str, ...]
    exog_blocks: Tuple[str, ...] = ()
    recipe_id: str = ""
    # Target scaling is deliberately raw: workstream 4 has not run. Recorded so the
    # artifact cannot imply a tuning or scaling decision that was never made.
    scaling: str = "raw (WS4 pending)"


def _is_stock(target: str) -> bool:
    from b_ml_pipeline import is_stock
    return bool(is_stock(target))


def _build_design(raw: pd.DataFrame, champ: Champion, date_col: str = "date"
                  ) -> Tuple[pd.DataFrame, pd.Series]:
    """Feature matrix at each origin, and the target series, on a business-day index.

    Reuses the pipelines' own builders rather than reimplementing them: a forward forecast
    computed from a second, subtly different feature recipe would not be the model whose
    DEV credentials we are quoting.
    """
    from b_ml_pipeline import (ConfigBML, calendar_exog, choose_recipe,
                               lag_window_features, to_business_index)

    s = to_business_index(raw, date_col, champ.target)
    cfg = ConfigBML(target=champ.target, cadence="Daily", horizon=max(FORWARD_HORIZONS),
                    data_path="", date_col=date_col, model_filter=None,
                    variant="univariate", out_root="", fiscal_groups=champ.fiscal_groups)
    lags, wins = choose_recipe(cfg)
    cal = calendar_exog(s.index, y=s, fiscal_groups=champ.fiscal_groups)
    if champ.exog_blocks:
        from preprocessing.multivariate import build_exog_features
        cal = pd.concat([cal, build_exog_features(raw, champ.target,
                                                  list(champ.exog_blocks), s.index,
                                                  date_col=date_col)], axis=1)
    L = list(lags)
    if _is_stock(champ.target) and 0 not in L:
        # Delta modelling for a level target: lag_0 is the origin value, known at the
        # origin by definition, and it supplies change-context rather than the answer.
        L = [0] + L
    X = lag_window_features(s, L, wins).join(cal)
    return X, s


def _fit_predict_point(X_tr, y_tr, X_new, model_name: str) -> float:
    from b_ml_pipeline import available_models
    models = available_models()
    if model_name not in models:
        raise KeyError(f"point model {model_name!r} not in available_models()")
    est = models[model_name]
    from sklearn.base import clone
    est = clone(est)
    est.fit(X_tr, y_tr)
    return float(np.asarray(est.predict(X_new)).ravel()[0])


def _fit_predict_quantiles(X_tr, y_tr, X_new,
                           quantiles: Sequence[float] = QUANTILES) -> Dict[float, float]:
    """GBQuantile per quantile, then enforce monotonicity.

    Independently fitted quantiles can cross -- p90 below p50 -- which is not a wide
    interval but an invalid one. Sorting the three values is the minimal honest repair and
    is what the E_QUANTILE family already does.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    out: Dict[float, float] = {}
    for q in quantiles:
        m = GradientBoostingRegressor(loss="quantile", alpha=float(q), random_state=0)
        m.fit(X_tr, y_tr)
        out[float(q)] = float(np.asarray(m.predict(X_new)).ravel()[0])
    vals = sorted(out.values())
    return {q: v for q, v in zip(sorted(out), vals)}


def run_forward(raw: pd.DataFrame,
                champ: Champion,
                horizons: Sequence[int] = FORWARD_HORIZONS,
                date_col: str = "date") -> pd.DataFrame:
    """Fit one model per horizon on all available history; predict the final origin.

    Returns one row per horizon with ``target_date``, ``p10``, ``p50``, ``p90``,
    ``origin_date``, ``origin_value``. No truth column exists by construction.
    """
    X, s = _build_design(raw, champ, date_col)
    stock = _is_stock(champ.target)

    # The final origin is the last row whose features are all present. Features are
    # backward-looking, so this is normally the last business day in the data.
    complete = X.dropna()
    if complete.empty:
        raise ValueError(f"no complete feature row for {champ.target!r}")
    origin = complete.index[-1]
    origin_value = float(s.loc[origin])
    data_end = s.index[-1].date()

    fwd_dates = business_days_after(data_end, max(horizons))
    assert_forward_only(fwd_dates, data_end)

    rows: List[Dict] = []
    for h in horizons:
        # Target for this horizon. For a level target we model the CHANGE and rebuild the
        # level as origin_value + delta, matching the pipelines; predicting a level
        # directly makes the model a persistence copy that scores well and learns nothing.
        y_h = s.shift(-h)
        y_h = (y_h - s) if stock else y_h

        usable = X.notna().all(axis=1) & y_h.notna()
        X_tr, y_tr = X[usable], y_h[usable]
        X_new = X.loc[[origin]]

        p50_raw = _fit_predict_point(X_tr, y_tr, X_new, champ.point_model)
        qs = _fit_predict_quantiles(X_tr, y_tr, X_new)

        base = origin_value if stock else 0.0
        rows.append({
            "target": champ.target,
            "horizon": int(h),
            "origin_date": pd.Timestamp(origin),
            "origin_value": origin_value,
            "target_date": pd.Timestamp(fwd_dates[h - 1]),
            "p10": base + qs[0.10],
            "p50": base + p50_raw,
            "p90": base + qs[0.90],
            "p50_quantile_model": base + qs[0.50],
            "point_model": champ.point_model,
            "interval_model": "GBQuantile",
            "n_train_rows": int(usable.sum()),
            "n_features": int(X.shape[1]),
            "modelled_as": "delta (level reconstructed)" if stock else "level",
        })

    df = pd.DataFrame(rows)
    assert_forward_only([d.date() for d in df["target_date"]], data_end)
    if "y_true" in df.columns:  # pragma: no cover - defensive
        raise AssertionError("a forward artifact must not carry a truth column")
    return df


def build_provenance(data_path: str, champions: Sequence[Champion]) -> Dict:
    from provenance import describe_code, describe_environment, describe_input
    di = describe_input(data_path)
    return {
        "run_kind": "forward_forecast",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "data": {"name": di.get("name"), "sha256": di.get("sha256"),
                 "n_rows": di.get("n_rows"), "latest_data_date": di.get("latest_data_date")},
        "code": describe_code(),
        "environment": describe_environment(),
        "calendar_version": calendar_version(),
        "quantiles": list(QUANTILES),
        "horizons": list(FORWARD_HORIZONS),
        "recipes": [{"target": c.target, "recipe_id": c.recipe_id,
                     "point_model": c.point_model, "interval_model": "GBQuantile",
                     "fiscal_groups": sorted(c.fiscal_groups),
                     "exog_blocks": sorted(c.exog_blocks),
                     "scaling": c.scaling} for c in champions],
        "test_window_touched": False,
        "notes": [
            "Forward forecast: every target date is strictly beyond the last date in the "
            "data, so no truth was read and no accuracy was computed here.",
            "Quality gate verdicts are carried from the DEV (2024) credentials run and "
            "linked by recipe_id. The 2025 holdout remains sealed.",
            "Target scaling is raw; workstream 4 (scaling comparison) has not run.",
        ],
    }


def write_artifacts(out_dir: Path, forecasts: pd.DataFrame, provenance: Dict,
                    gates: Optional[Dict] = None) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    f = out_dir / "forward_forecast.csv"
    forecasts.to_csv(f, index=False)
    paths["forecast_csv"] = str(f)
    p = out_dir / "forward_provenance.json"
    p.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    paths["provenance_json"] = str(p)
    if gates is not None:
        g = out_dir / "forward_gates.json"
        g.write_text(json.dumps(gates, indent=2, default=str), encoding="utf-8")
        paths["gates_json"] = str(g)
    return paths
