"""Workstream 5: exogenous features must not read at or after the forecast origin.

The multivariate step is where leakage is easiest to introduce and hardest to see: a
same-day exogenous value looks like an ordinary column and improves every metric. These
tests pin the three properties the module claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from preprocessing.multivariate import (  # noqa: E402
    BLOCKS,
    DEBT_OPS_BLOCK,
    DEFAULT_EXOG_LAGS,
    NON_ECONOMIC,
    build_exog_features,
    exog_spec_hash,
    resolve_block,
)

IDX = pd.bdate_range("2018-01-01", "2024-12-31")


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {c: rng.normal(1e8, 2e7, len(IDX)) for c in
            list(DEBT_OPS_BLOCK) + ["Revenues", "Expenditure", "Taxes", "Value added tax"]}
    cols["is_weekend"] = 0
    cols["is_holiday"] = 0
    df = pd.DataFrame(cols, index=IDX)
    df.index.name = "date"
    return df.reset_index()


# ── the leak test that matters ────────────────────────────────────────────────

def test_future_exog_values_cannot_change_past_features():
    """Mutate every exogenous column's future; past features must be byte-identical."""
    df = _frame()
    cut = len(IDX) // 2
    base = build_exog_features(df, "Revenues", ["debt_ops", "tax"], IDX)

    tampered = df.copy()
    num = [c for c in tampered.columns if c != "date"]
    tampered.loc[cut:, num] = tampered.loc[cut:, num] * 1000.0 + 9e9
    after = build_exog_features(tampered, "Revenues", ["debt_ops", "tax"], IDX)

    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut])
    assert not base.iloc[cut:].equals(after.iloc[cut:]), (
        "the tampering had no effect anywhere -- this test cannot detect leakage"
    )


def test_no_feature_uses_the_same_day_value():
    """Perturb one row of one column; the same row's features must not move.

    This is the specific multivariate leak: an exogenous line dated at the origin is not
    reliably known at the origin for a flow series.
    """
    df = _frame()
    t = 400
    f0 = build_exog_features(df, "Revenues", ["debt_ops"], IDX)
    df2 = df.copy()
    df2.loc[t, "Domestic"] = df2.loc[t, "Domestic"] * 77.0
    f1 = build_exog_features(df2, "Revenues", ["debt_ops"], IDX)
    pd.testing.assert_series_equal(f0.iloc[t], f1.iloc[t], check_names=False)
    assert not f0.iloc[t + 1].equals(f1.iloc[t + 1]), "lag 1 should see row t at t+1"


def test_lag_zero_is_refused():
    df = _frame()
    with pytest.raises(ValueError, match="must all be >= 1"):
        build_exog_features(df, "Revenues", ["debt_ops"], IDX, lags=(0, 1))


def test_features_are_not_normalised_across_the_sample():
    """Truncating the input must not change any surviving value.

    Any fitted statistic -- a mean, a scale, an encoding -- would break this. The module
    fits nothing, which is what makes the leak argument structural.
    """
    df = _frame()
    full = build_exog_features(df, "Revenues", ["debt_ops", "cross"], IDX)
    n = 900
    part = build_exog_features(df.iloc[:n], "Revenues", ["debt_ops", "cross"], IDX[:n])
    pd.testing.assert_frame_equal(full.iloc[:n], part)


def test_aligned_lag_never_references_a_future_row():
    df = _frame()
    # make one column equal to its own row number so provenance is checkable
    df["Domestic"] = np.arange(len(df), dtype=float)
    f = build_exog_features(df, "Revenues", ["debt_ops"], IDX)
    v = f["x_Domestic_aligned_prev_month"].to_numpy()
    pos = np.arange(len(v), dtype=float)
    ok = np.isnan(v) | (v < pos)
    assert ok.all(), f"referenced a future row at {np.where(~ok)[0][:5]}"


# ── block hygiene ─────────────────────────────────────────────────────────────

def test_target_is_never_in_its_own_exog_set():
    """The target enters through the pipeline's own lag recipe; duplicating it here would
    double-count it and, for the cross block, hand the model its own column."""
    df = _frame()
    for target in ("Revenues", "Expenditure"):
        f = build_exog_features(df, target, ["cross"], IDX)
        safe = f"x_{target}".replace(" ", "_")
        assert not any(c.startswith(safe + "_") for c in f.columns), (
            f"{target} appeared in its own exog set: {list(f.columns)}"
        )


def test_calendar_flags_are_excluded_from_the_broad_pool():
    """is_weekend/is_holiday are calendar, not economic -- the fiscal calendar owns them."""
    df = _frame()
    cols = resolve_block("broad", [c for c in df.columns if c != "date"], "Revenues")
    for c in NON_ECONOMIC:
        assert c not in cols


def test_empty_block_is_an_error_not_a_silent_pass():
    """A block that resolves to nothing must fail loudly.

    Silently producing zero features would make an ablation row read as "the block did not
    help" when the truth is "the block was never applied".
    """
    df = pd.DataFrame({"date": IDX, "Revenues": 1.0})
    with pytest.raises(ValueError, match="no usable columns"):
        build_exog_features(df, "Revenues", ["debt_ops"], IDX)


def test_unknown_block_is_rejected():
    with pytest.raises(KeyError, match="unknown exog block"):
        resolve_block("nonsense", ["Revenues"], "Expenditure")


def test_debt_ops_block_names_the_measured_mechanism():
    """The hypothesis is specific; the block must contain the two lines DATA_SEMANTICS
    §1 measured as driving the negative-Revenues days."""
    assert "Increase in liabilities" in DEBT_OPS_BLOCK
    assert "Domestic" in DEBT_OPS_BLOCK


def test_spec_hash_is_order_insensitive_and_content_sensitive():
    assert exog_spec_hash(["tax", "debt_ops"], [1, 5], True) == \
           exog_spec_hash(["debt_ops", "tax"], [5, 1], True)
    assert exog_spec_hash(["debt_ops"], [1], True) != exog_spec_hash(["debt_ops"], [1], False)


def test_feature_count_is_as_specified():
    df = _frame()
    f = build_exog_features(df, "Revenues", ["debt_ops"], IDX,
                            lags=DEFAULT_EXOG_LAGS, aligned=True)
    # 4 columns x (3 lags + 1 aligned)
    assert f.shape[1] == 4 * (len(DEFAULT_EXOG_LAGS) + 1)
    f2 = build_exog_features(df, "Revenues", ["debt_ops"], IDX, lags=(1,), aligned=False)
    assert f2.shape[1] == 4


def test_pipeline_config_defaults_to_univariate():
    from b_ml_pipeline import ConfigBML, calendar_exog

    cfg = ConfigBML(target="y", cadence="Daily", horizon=5, data_path="unused.csv",
                    date_col="date", model_filter=None, variant="univariate",
                    out_root="unused")
    assert cfg.exog_blocks is None
    # and the pre-WS5 calendar frame is unchanged
    assert not any(c.startswith("x_") for c in calendar_exog(IDX).columns)
