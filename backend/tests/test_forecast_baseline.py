"""The persistence benchmark must be READ from artifacts, never recomputed for display.

Same principle as MASE: one implementation of a published number. A second computation of the
benchmark in a page would be free to drift from the evaluator's, and the drift would be invisible —
both would look like "the benchmark".

The two facts these tests hold:
  * `origin_value` in a forward artifact IS the benchmark prediction, by the same definition
    `compute_persistence_baseline()` uses;
  * the benchmark ERROR shown in the header comes from the audited DEV run, not from the forward
    dates, which have no truth to measure against.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from forecast_integrity import compute_persistence_baseline  # noqa: E402
from forward_forecast import (  # noqa: E402
    BENCHMARK_COLUMN,
    BENCHMARK_EXPLANATION,
    BENCHMARK_LABEL,
    benchmark_mae_for_target,
    benchmark_series,
)

PUBLISHED = REPO / "forecasts" / "published"


def _issue() -> Path:
    issues = sorted(p for p in PUBLISHED.iterdir()
                    if p.is_dir() and (p / "forecast.csv").exists()) if PUBLISHED.exists() else []
    if not issues:
        pytest.skip("no published forecast issue on disk")
    return issues[-1]


# ── the series is the evaluator's own column ───────────────────────────────────

def test_benchmark_column_is_the_one_the_evaluator_uses():
    """compute_persistence_baseline() documents y_hat(t+h) = y(t) and reads `origin_value`.

    If these ever diverge, the page would plot one benchmark while skill was measured against
    another, and nothing would say so.
    """
    src = (BACKEND / "forecast_integrity.py").read_text()
    fn = src[src.index("def compute_persistence_baseline"):]
    fn = fn[:fn.index("\ndef ", 1)]
    assert BENCHMARK_COLUMN in fn, (
        f"the evaluator no longer reads {BENCHMARK_COLUMN!r}; the plotted benchmark would be a "
        f"different quantity from the one skill is measured against"
    )
    assert "y_hat(t+h) = y(t)" in fn, "the benchmark definition changed"


def test_benchmark_series_reproduces_the_evaluators_prediction():
    """Not merely the same column name — the same numbers.

    Built from a frame with known truth so the evaluator can be run on it, then checked that the
    series the page plots is exactly what the evaluator scored.
    """
    y_origin = np.array([10.0, 11.0, 12.0, 13.0])
    df = pd.DataFrame({"origin_value": y_origin,
                       "y_true": np.array([10.5, 12.0, 11.0, 14.0]),
                       "y_pred": np.array([10.2, 11.5, 11.8, 13.2])})
    plotted = benchmark_series(df)
    assert plotted is not None
    assert np.allclose(plotted.to_numpy(), y_origin)

    # the evaluator's benchmark error over the same rows, from the same column
    expected_mae = float(np.mean(np.abs(df["y_true"] - plotted)))
    assert compute_persistence_baseline(df)["mae_persistence"] == pytest.approx(expected_mae)


def test_benchmark_series_returns_none_rather_than_computing_a_substitute():
    """A missing column must surface as "not reported", not trigger a fallback calculation."""
    df = pd.DataFrame({"y_true": [1.0], "p50": [1.0]})
    assert benchmark_series(df) is None
    all_nan = pd.DataFrame({BENCHMARK_COLUMN: [np.nan, np.nan], "y_true": [1.0, 2.0]})
    assert benchmark_series(all_nan) is None


# ── on the real published issue ───────────────────────────────────────────────

def test_published_issue_carries_the_benchmark_column():
    fc = pd.read_csv(_issue() / "forecast.csv")
    assert BENCHMARK_COLUMN in fc.columns, (
        "the published artifact has no benchmark column, so the page has nothing to read"
    )


def test_benchmark_is_flat_across_horizons_from_one_origin():
    """All horizons share an origin, so "today's value carried forward" is one value.

    A varying series here would mean the origin differed per horizon, which would make the
    benchmark a different thing at each point on the chart.
    """
    fc = pd.read_csv(_issue() / "forecast.csv")
    for target, g in fc.groupby("target"):
        assert g["origin_date"].nunique() == 1, f"{target} has multiple origins"
        s = benchmark_series(g)
        assert s is not None and s.nunique() == 1, (
            f"{target} benchmark is not flat: {sorted(s.unique())[:4]}"
        )


def test_benchmark_equals_the_target_value_at_the_origin():
    """The benchmark must be the actual value on the origin date, read from the canonical data."""
    data = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
    if not data.exists():
        pytest.skip("canonical data not present")
    from b_ml_pipeline import to_business_index

    raw = pd.read_csv(data)
    fc = pd.read_csv(_issue() / "forecast.csv")
    for target, g in fc.groupby("target"):
        s = to_business_index(raw, "date", target)
        origin = pd.Timestamp(g["origin_date"].iloc[0])
        assert float(benchmark_series(g).iloc[0]) == pytest.approx(float(s.loc[origin]), rel=1e-9), (
            f"{target}: the plotted benchmark is not the value at its origin date"
        )


# ── the benchmark ERROR comes from the audited run ────────────────────────────

def test_header_benchmark_mae_comes_from_the_dev_credentials_run():
    from registry import load_registry

    for rec in load_registry()["recipes"]:
        b = benchmark_mae_for_target(rec["target"])
        assert b["available"] is True, rec["target"]
        assert b["run_id"] == rec["dev_credentials"]["run_id"]
        j = json.loads((REPO / "experiments" / "runs" / f"{b['run_id']}.json").read_text())
        assert b["benchmark_mae"] == pytest.approx(j["ruler"]), (
            "the header figure is not the ruler recorded on the audited run"
        )
        assert b["model_mae"] == pytest.approx(rec["dev_credentials"]["dev_mae"])


def test_header_figures_reconcile_with_the_recorded_skill():
    """(benchmark - model) / benchmark must reproduce the logged skill.

    Not a tautology: benchmark_mae and skill come from different files -- the run JSON and the
    registry -- so this catches the two drifting apart. The tolerance is 1e-4 because the log
    stores skill to four decimals.
    """
    from registry import load_registry

    for rec in load_registry()["recipes"]:
        b = benchmark_mae_for_target(rec["target"])
        implied = (b["benchmark_mae"] - b["model_mae"]) / b["benchmark_mae"] * 100.0
        assert implied == pytest.approx(b["skill_pct"], abs=1e-4), (
            f"{rec['target']}: recorded skill {b['skill_pct']} does not follow from the header "
            f"figures ({implied})"
        )


def test_a_target_with_no_recipe_reports_no_benchmark_rather_than_borrowing_one():
    b = benchmark_mae_for_target("Taxes")
    assert b["available"] is False
    assert b["benchmark_mae"] is None
    assert "no registry recipe" in b["ruler_note"]


def test_benchmark_labels_state_it_is_a_prediction_not_an_error():
    """Forward dates have no truth. Presenting the benchmark as an error there would invite a
    reader to look for an accuracy figure that cannot exist."""
    assert "carried forward" in BENCHMARK_LABEL
    assert "no actual value yet" in BENCHMARK_EXPLANATION
    assert "not an error measurement" in BENCHMARK_EXPLANATION


# ── the page must not grow its own implementation ─────────────────────────────

def test_forecast_page_reads_the_shared_reader_and_computes_nothing():
    page = (REPO / "frontend" / "pages" / "07_Forecast.py").read_text()
    code = "\n".join(l for l in page.splitlines() if not l.lstrip().startswith("#"))
    assert "_benchmark_series(" in code and "_benchmark_mae(" in code
    for banned in ("compute_persistence_baseline", ".shift(", "mae_persistence ="):
        assert banned not in code, (
            f"the page appears to compute the benchmark itself ({banned!r}); it must read the "
            f"audited value"
        )
