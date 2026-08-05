"""A2 / item 1d: the seasonal-naive field must never impersonate persistence.

`preprocessing.integrity.compute_baselines` produced `mae_seasonal_naive` with
`season_steps` hardcoded to 5. The production horizon is also 5, so at h=5 the
"seasonal naive" reference was **exactly** the h-step persistence baseline, and the
Dashboard displayed the two side by side as if the second corroborated the first.

Verified on the real artifact (review §1.2):

    mae_seasonal_naive: 60976736.58082051 == mae_persistence

and confirmed to be a coincidence of h == season, not a genuine identity:

    h=3: persistence == seasonal_naive ? False
    h=5: persistence == seasonal_naive ? True     <- production horizon
    h=7: persistence == seasonal_naive ? False

Per A2 the field is aliased rather than removed, because the Dashboard reads it. The
degeneracy is now explicit: NaN plus `seasonal_naive_degenerate=True` and a note, so a
reader sees "not available, and why" instead of a duplicate under another name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import (  # noqa: E402
    SEASONAL_NAIVE_SEASON_STEPS,
    compute_persistence_baseline,
    compute_seasonal_naive_baseline,
)


def _preds(n=120, h=5, season_gap=None):
    """Predictions on a business-day index with origin h steps back."""
    idx = pd.bdate_range("2024-01-01", periods=n + h)
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 5.0, len(idx))) + 1000.0
    rows = [{"target_date": idx[i], "y_true": float(y[i]),
             "origin_value": float(y[i - h])} for i in range(h, len(idx))]
    return pd.DataFrame(rows)


def test_degenerate_when_season_equals_horizon():
    """The production case: h=5, season=5. Must NOT report a number."""
    r = compute_seasonal_naive_baseline(_preds(h=5), horizon=5, season_steps=5)
    assert r["seasonal_naive_degenerate"] is True
    assert np.isnan(r["mae_seasonal_naive"]), (
        "a season equal to the horizon reproduces persistence exactly; reporting a "
        "number here is what let a duplicate pose as corroboration"
    )
    assert "note" in " ".join(r.keys()) or "seasonal_naive_note" in r
    assert "persistence" in r["seasonal_naive_note"].lower()


def test_not_degenerate_when_season_differs_from_horizon():
    r = compute_seasonal_naive_baseline(_preds(h=5), horizon=5, season_steps=21)
    assert r["seasonal_naive_degenerate"] is False
    assert np.isfinite(r["mae_seasonal_naive"])
    assert r["mae_seasonal_naive"] > 0
    assert r["seasonal_naive_season_steps"] == 21


def test_the_degeneracy_is_real_and_not_a_coincidence_of_the_fixture():
    """Guards the premise: at season == h the two references are the same estimator.

    If they were genuinely different measurements, returning NaN would be
    over-cautious rather than necessary.

    On the real artifact the two matched **to the cent**, because the retired
    `compute_baselines` looked its season origin up in the FULL series. This function
    only receives the predictions frame, so its sample starts at the first target date
    and the two agree closely rather than exactly: persistence can pair a target with
    an origin dated before the evaluation window, and this cannot. The distinction is
    about available rows, not about what is being measured -- both are
    mean|y(t) - y(t - k)| with k = 5. Hence a tolerance, and an explicit check that a
    genuinely different season is genuinely far away.
    """
    h = 5
    preds = _preds(h=h)
    persistence = compute_persistence_baseline(preds)["mae_persistence"]

    y = preds.sort_values("target_date")["y_true"].to_numpy(dtype=float)
    seasonal_at_h = float(np.mean(np.abs(y[h:] - y[:-h])))

    assert np.isclose(seasonal_at_h, persistence, rtol=0.05), (
        f"seasonal-naive at season==h ({seasonal_at_h:,.2f}) should be within 5% of "
        f"persistence ({persistence:,.2f}) -- they are the same estimator on nearly "
        f"the same rows. If they diverge, the degeneracy guard is unnecessary."
    )

    # A different season must be far away, or "degenerate" would be meaningless.
    s21 = compute_seasonal_naive_baseline(preds, horizon=h, season_steps=21)
    assert not np.isclose(s21["mae_seasonal_naive"], persistence, rtol=0.10), (
        f"season=21 gave {s21['mae_seasonal_naive']:,.2f} vs persistence "
        f"{persistence:,.2f}; a non-degenerate season should differ materially"
    )


def test_the_field_survives_for_the_dashboard():
    """A2: alias, do not remove. The key must always be present."""
    for season in (5, 21):
        r = compute_seasonal_naive_baseline(_preds(), horizon=5, season_steps=season)
        assert "mae_seasonal_naive" in r
        assert "seasonal_naive_season_steps" in r
        assert "seasonal_naive_degenerate" in r


def test_default_season_is_documented_and_used():
    assert SEASONAL_NAIVE_SEASON_STEPS == 5
    r = compute_seasonal_naive_baseline(_preds(), horizon=20)
    assert r["seasonal_naive_season_steps"] == SEASONAL_NAIVE_SEASON_STEPS
    assert r["seasonal_naive_degenerate"] is False


def test_missing_columns_and_short_series_report_rather_than_crash():
    r = compute_seasonal_naive_baseline(pd.DataFrame({"y_true": [1.0, 2.0]}),
                                        horizon=5, season_steps=21)
    assert np.isnan(r["mae_seasonal_naive"]) and "seasonal_naive_note" in r

    short = _preds(n=3, h=1)
    r2 = compute_seasonal_naive_baseline(short, horizon=1, season_steps=21)
    assert np.isnan(r2["mae_seasonal_naive"]) and "seasonal_naive_note" in r2
