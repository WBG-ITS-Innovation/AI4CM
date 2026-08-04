"""Yardstick + D12: E_QUANTILE on business days, with an honest stock path.

Two defects fixed here, both from the review.

**The horizon meant something different.** E_QUANTILE was the only family that did
not reindex to a business-day calendar, so it ran on the raw 7-day index and its
``h=5`` meant 5 *calendar* days against every other family's 5 business days
(review §1.2, §4.3). It was therefore solving an easier problem and grading itself
against an easier ruler: measured on the pre-Phase-1 data, 66,161,268 versus the
shared 60,273,679 on the same 148 target dates, with 28% of its evaluation targets
falling on weekends no other family scored.

**It had no stock path at all.** `b_ml`, `c_dl` and `run_a_stat` all carry
`is_stock` handling; E_QUANTILE had none, so `State budget balance` could not be
forecast by the family that three Phase-2 workstreams route through. A level series
predicted directly is dominated by its own current value, which makes the model a
trivial persistence predictor that scores well while learning nothing. The fix is
delta modelling: predict ``y(t+h) - y(t)``, reconstruct as ``origin_value + delta``.

The leakage question for the stock path is specific: ``lag_0`` is now a feature.
That is the value *at the origin*, which the forecaster knows by definition — so it
is legitimate, and these tests pin the boundary rather than take it on trust.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from e_quantile_daily_pipeline import (  # noqa: E402
    Config,
    _build_features,
    is_stock,
    to_business_index,
)

H = 5
FLOW = "Revenues"
STOCK = "State budget balance"


def _frame(n_days=800, seed=0):
    """Calendar-daily frame (7 days a week) with both a flow and a stock column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    flow = np.abs(rng.normal(7e7, 1e7, n_days))
    stock = np.cumsum(rng.normal(0, 5e6, n_days)) + 1e9
    return pd.DataFrame({FLOW: flow, STOCK: stock}, index=idx)


def _cfg(target, **kw):
    return Config(target=target, cadence="Daily", horizon=H, data_path="",
                  variant="univariate", **kw)


# ── the yardstick: h means business days ──────────────────────────────────

def test_reindex_drops_weekends():
    df = _frame()
    out = to_business_index(df, FLOW)
    assert (out.index.dayofweek < 5).all(), "weekends survived the reindex"
    assert len(out) < len(df)


def test_flows_are_zero_filled_and_stocks_forward_filled():
    """A missing trading day means no flow; a balance persists."""
    df = _frame()
    df.loc[df.index[10:14], [FLOW, STOCK]] = np.nan
    flow_out = to_business_index(df, FLOW)
    stock_out = to_business_index(df, STOCK)
    assert not flow_out[FLOW].isna().any()
    assert (flow_out.loc[flow_out.index[:20], FLOW] == 0.0).any(), "no zero-fill happened"
    assert not stock_out[STOCK].isna().any()
    assert 0.0 not in set(stock_out[STOCK].iloc[:20]), "a balance was zero-filled"


def test_horizon_is_five_business_days_not_five_calendar_days():
    """The defect this file exists for.

    On the calendar index, origin -> target spanned 5 calendar days. On the
    business-day index it must span 5 business-day *steps*, i.e. 7 calendar days
    across a weekend, matching every other family.
    """
    df = to_business_index(_frame(), FLOW)
    X, y, od, ov = _build_features(df, _cfg(FLOW))
    pos = {ts: i for i, ts in enumerate(df.index)}
    steps = [pos[df.index[pos[o] + H]] - pos[o] for o in od.values[:50]]
    assert set(steps) == {H}, f"origin->target is not {H} index steps: {set(steps)}"

    gaps = {(df.index[pos[o] + H] - pd.Timestamp(o)).days for o in od.values[:200]}
    assert gaps.issubset({H, H + 1, H + 2}), (
        f"calendar gaps {sorted(gaps)} do not look like {H} BUSINESS days "
        f"(expect 7 across a weekend, 5 within a week)"
    )
    assert max(gaps) > H, "no gap exceeded 5 calendar days, so weekends are still counted"


# ── the stock path ────────────────────────────────────────────────────────

def test_is_stock_agrees_with_the_other_families():
    from b_ml_pipeline import is_stock as bml_is_stock
    for name in (STOCK, "balance", "t0", FLOW, "Expenditure", "Taxes"):
        assert is_stock(name) == bml_is_stock(name), f"families disagree about '{name}'"


def test_stock_target_is_modelled_as_a_delta():
    """The modelling target must be the change, not the level."""
    df = to_business_index(_frame(), STOCK)
    X, y, od, ov = _build_features(df, _cfg(STOCK))
    pos = {ts: i for i, ts in enumerate(df.index)}
    level = df[STOCK]

    for o in list(od.values)[:40]:
        expected = float(level.iloc[pos[o] + H]) - float(level.loc[o])
        assert y.loc[o] == pytest.approx(expected, rel=1e-12), (
            f"target at origin {pd.Timestamp(o).date()} is not y(t+h) - y(t); "
            f"the level is being predicted directly"
        )
    # A delta straddles zero; a level of ~1e9 never would.
    assert y.min() < 0 < y.max(), "target does not look like a change"
    assert abs(y).max() < 0.5 * float(level.abs().mean()), "target looks like a level"


def test_flow_target_is_still_the_level():
    """Delta modelling must apply to stocks only."""
    df = to_business_index(_frame(), FLOW)
    X, y, od, ov = _build_features(df, _cfg(FLOW))
    pos = {ts: i for i, ts in enumerate(df.index)}
    for o in list(od.values)[:40]:
        assert y.loc[o] == pytest.approx(float(df[FLOW].iloc[pos[o] + H]), rel=1e-12)


def test_origin_value_is_always_the_level():
    """Persistence and level reconstruction both need the real y(t)."""
    for target in (FLOW, STOCK):
        df = to_business_index(_frame(), target)
        X, y, od, ov = _build_features(df, _cfg(target))
        for o in list(od.values)[:40]:
            assert ov.loc[o] == pytest.approx(float(df[target].loc[o]), rel=1e-12), (
                f"origin_value for '{target}' is not the level at the origin"
            )


def test_level_reconstruction_is_exact():
    """origin_value + delta must recover the true level, or reported forecasts are wrong."""
    df = to_business_index(_frame(), STOCK)
    X, y, od, ov = _build_features(df, _cfg(STOCK))
    pos = {ts: i for i, ts in enumerate(df.index)}
    reconstructed = ov.to_numpy() + y.to_numpy()
    truth = np.array([float(df[STOCK].iloc[pos[o] + H]) for o in od.values])
    assert np.allclose(reconstructed, truth, rtol=1e-12), (
        "origin_value + delta does not equal the true level"
    )


# ── the leak boundary for lag_0 ────────────────────────────────────────────

def test_lag_0_exists_only_for_stocks_and_equals_the_origin_value():
    """lag_0 is the value AT the origin: known at forecast time, so not leakage.

    It is added only under delta modelling. For a flow target it would make the
    model a near-persistence predictor with nothing removing the level, which is
    the trap b_ml_pipeline documents.
    """
    df_s = to_business_index(_frame(), STOCK)
    Xs, ys, ods, ovs = _build_features(df_s, _cfg(STOCK))
    assert "y_lag_0" in Xs.columns, "stock path is missing lag_0 change-context"
    assert np.allclose(Xs["y_lag_0"].to_numpy(), ovs.to_numpy(), rtol=1e-12), (
        "lag_0 is not the origin value"
    )

    df_f = to_business_index(_frame(), FLOW)
    Xf, *_ = _build_features(df_f, _cfg(FLOW))
    assert "y_lag_0" not in Xf.columns, "lag_0 leaked into the flow feature set"


def test_no_feature_uses_data_after_the_origin():
    """The core leak check: perturb y strictly after an origin, features must not move."""
    for target in (FLOW, STOCK):
        df = to_business_index(_frame(), target)
        X_base, _, od, _ = _build_features(df, _cfg(target))

        cut = od.values[len(od) // 2]
        df_pert = df.copy()
        after = df_pert.index > pd.Timestamp(cut)
        df_pert.loc[after, target] = df_pert.loc[after, target] * 3.0 + 1e8
        X_pert, _, od_p, _ = _build_features(df_pert, _cfg(target))

        common = [o for o in od.values if pd.Timestamp(o) <= pd.Timestamp(cut)]
        common = [o for o in common if o in set(od_p.values)]
        assert len(common) > 50, "not enough shared origins to test"
        delta = (X_base.loc[common] - X_pert.loc[common]).abs().to_numpy().max()
        assert delta == 0.0, (
            f"'{target}': perturbing y after {pd.Timestamp(cut).date()} changed "
            f"features at or before it by up to {delta} - a feature reads the future"
        )


def test_run_pipeline_actually_reindexes(tmp_path):
    """The wiring, not just the helper.

    Every other test here calls to_business_index() directly, so all of them pass
    even if run_pipeline stops calling it -- confirmed by mutation: deleting the
    reindex line left 11/11 green. This asserts the property on the published
    artifact, which is the only place it can be checked end to end.
    """
    from e_quantile_daily_pipeline import run_pipeline

    df = _frame(n_days=1400)
    csv = tmp_path / "d.csv"
    df.rename_axis("date").to_csv(csv)
    out = tmp_path / "out"
    run_pipeline(Config(
        target=FLOW, cadence="Daily", horizon=H, data_path=str(csv),
        variant="univariate", model_filter="GBQuantile",
        out_root=str(out), folds=3, min_train_years=1,
    ))

    preds = pd.read_csv(out / "predictions_long.csv",
                        parse_dates=["origin_date", "target_date"])
    assert len(preds) > 0

    assert (preds["target_date"].dt.dayofweek < 5).all(), (
        "a weekend target date was published -- run_pipeline is not reindexing"
    )
    assert (preds["origin_date"].dt.dayofweek < 5).all()

    steps = preds.apply(
        lambda r: len(pd.bdate_range(r["origin_date"], r["target_date"])) - 1, axis=1
    )
    assert set(steps.unique()) == {H}, (
        f"origin->target spans {sorted(steps.unique())} business days, not {H}; "
        f"the family is back on a calendar-day horizon"
    )
    gaps = (preds["target_date"] - preds["origin_date"]).dt.days
    assert gaps.max() > H, (
        "no origin->target gap exceeded 5 calendar days, so weekends are being "
        "counted as forecast steps"
    )


def test_target_uses_the_future_but_features_never_do():
    """Guards the premise of the previous test.

    If perturbing the future changed nothing at all, the fixture would be inert and
    the leak test vacuous. The h-step TARGET must move; only features must not.
    """
    df = to_business_index(_frame(), FLOW)
    _, y_base, od, _ = _build_features(df, _cfg(FLOW))
    cut = od.values[len(od) // 2]
    df_pert = df.copy()
    after = df_pert.index > pd.Timestamp(cut)
    df_pert.loc[after, FLOW] = df_pert.loc[after, FLOW] * 3.0 + 1e8
    _, y_pert, od_p, _ = _build_features(df_pert, _cfg(FLOW))

    tail = [o for o in od.values[-50:] if o in set(od_p.values)]
    assert (y_base.loc[tail] != y_pert.loc[tail]).any(), (
        "the h-step target did not move, so the perturbation never landed"
    )
