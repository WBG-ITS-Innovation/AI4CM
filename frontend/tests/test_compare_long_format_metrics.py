"""A run whose metrics cannot be compared is named, not silently dropped -- and never crashes.

The bug this holds down
-----------------------
The Compare page picked each run's best model with::

    if "model" in m.columns:
        best_row = m.loc[m["MAE"].idxmin()]

The guard tests for ``model`` and then indexes ``MAE``, which it never checked. Every
E_QUANTILE run writes a genuinely long ``metrics_long.csv``::

    model,fold,metric,quantile,value

which has ``model`` and no ``MAE``, so the page raised ``KeyError: 'MAE'``. The
``target``/``horizon`` filters above did not save it: those columns are absent from the
long format, so both filters were skipped and the frame arrived non-empty.

Why these runs are skipped rather than pivoted
----------------------------------------------
The long format carries ``pinball`` and ``coverage_p10_p90`` only. There is no MAE, RMSE,
sMAPE or R2 anywhere in it, so a pivot cannot produce the columns this table is built from
-- it would add a row of blanks and imply a comparison that never happened. Skipping and
saying so is the honest option, and the reason string is what makes it visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

COMPARE = FRONTEND / "pages" / "05_Compare.py"
SOURCE = COMPARE.read_text(encoding="utf-8")


def _helper():
    try:
        from utils_frontend import best_metric_row
    except ImportError:
        pytest.fail("utils_frontend has no `best_metric_row`: the Compare page is still "
                    "indexing m['MAE'] behind a guard that only checks for 'model'")
    return best_metric_row


def _long_frame():
    """The shape every E_QUANTILE run writes."""
    return pd.DataFrame({
        "model": ["GBQuantile", "GBQuantile", "ResidualRF", "ResidualRF"],
        "fold": [1, 2, 1, 2],
        "metric": ["pinball", "coverage_p10_p90", "pinball", "coverage_p10_p90"],
        "quantile": [0.1, 0.9, 0.1, 0.9],
        "value": [8.6e7, 0.78, 9.1e7, 0.70],
    })


def _wide_frame():
    return pd.DataFrame({
        "target": ["Revenues"] * 3,
        "horizon": [5] * 3,
        "model": ["Ridge", "LightGBM", "ETS"],
        "MAE": [3.0e8, 2.1e8, 4.4e8],
        "RMSE": [3.6e8, 2.8e8, 5.0e8],
    })


def test_a_long_format_frame_does_not_raise():
    """The reported crash, stated directly."""
    row, reason = _helper()(_long_frame(), "Revenues", 5)
    assert row is None
    assert reason


def test_a_long_format_frame_is_explained_rather_than_dropped():
    """A blank table with no reason is the failure this replaced."""
    _, reason = _helper()(_long_frame(), "Revenues", 5)
    assert "MAE" in reason
    assert reason.strip().endswith(".")


def test_a_wide_frame_still_picks_the_lowest_mae():
    row, reason = _helper()(_wide_frame(), "Revenues", 5)
    assert reason is None
    assert row["model"] == "LightGBM"
    assert row["MAE"] == 2.1e8


def test_the_target_and_horizon_filters_still_apply():
    frame = _wide_frame()
    frame.loc[1, "target"] = "Expenditure"       # the lowest MAE, now a different target
    row, reason = _helper()(frame, "Revenues", 5)
    assert reason is None
    assert row["model"] == "Ridge"


def test_a_missing_metrics_file_is_explained_rather_than_dropped():
    row, reason = _helper()(None, "Revenues", 5)
    assert row is None
    assert reason


def test_no_rows_for_this_target_is_explained_rather_than_dropped():
    row, reason = _helper()(_wide_frame(), "Grants", 5)
    assert row is None
    assert reason


def test_the_compare_page_no_longer_indexes_mae_unguarded():
    collapsed = 'best_row = m.loc[m["MAE"].idxmin()]'
    assert collapsed not in SOURCE, (
        "05_Compare.py is indexing m['MAE'] again behind a guard that only checks for "
        "'model'; a long-format metrics frame raises KeyError")
