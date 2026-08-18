"""The Ops comparison: causal, correctly spread, reported everywhere, and gating nothing.

The Treasury's current method is the comparison a client actually cares about — "how much better
than what we do today" — so it sits beside the naive ruler on every reported row. It must not
influence what gets published, and it must be computed without reaching forward in time.

The defect these tests exist because of
---------------------------------------
``ops_daily_from_monthly(method="profile")`` returned **identically zero** for every date: 1983
of 1983 non-NaN values exactly 0.00, while the monthly totals were correct. It built each month's
intraday shape from the same month in a previous year and then mapped it onto the current year's
dates with ``.reindex(days, fill_value=0.0)`` — labels that cannot match across years, so every
weight became zero.

Consequence, measured: every ``MAE_skill_vs_Ops`` figure the repository had produced was skill
against a *zero forecast*, i.e. against ``mean|y|``. On revenues it read 55–62% where the truth
against the stated method is −8% to +8%. A four-fold overstatement of the project's headline
client-facing claim, passing silently because nothing asserted the baseline was non-zero.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

warnings.filterwarnings("ignore")

import ops_baseline as ob                                                # noqa: E402
from c_dl_pipeline import (ops_daily_from_monthly,                       # noqa: E402
                           ops_monthly_baseline_treasury)
from evaluation_windows import TEST_END, TEST_START                      # noqa: E402
from published_forecasts import SCORECARD_COLUMNS                        # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
FLOW, STOCK = "Revenues", "State budget balance"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")


@pytest.fixture(scope="module")
def flow_series():
    return ob._flow_series(DATA, FLOW)


# ── 1. the spread is not zero, and it conserves the monthly total ────────────

@needs_data
@pytest.mark.parametrize("method", ["flat", "profile"])
def test_the_daily_spread_is_not_identically_zero(flow_series, method):
    """The regression guard for the four-fold overstatement."""
    monthly = ops_monthly_baseline_treasury(flow_series)
    daily = ops_daily_from_monthly(flow_series, monthly, method=method).dropna()
    assert len(daily) > 100, "nothing to check"
    assert (daily != 0).sum() > 0.9 * len(daily), (
        f"{method}: {(daily == 0).sum()} of {len(daily)} daily values are exactly zero. A "
        f"baseline of zero makes skill_vs_ops mean 'better than forecasting nothing'.")


@needs_data
@pytest.mark.parametrize("method", ["flat", "profile"])
def test_each_months_daily_values_sum_to_its_monthly_total(flow_series, method):
    """The property that makes the spread a *spread* rather than a rescaling.

    This is what the zero bug violated most visibly: every month summed to 0 against a non-zero
    monthly total, and nothing noticed.
    """
    monthly = ops_monthly_baseline_treasury(flow_series)
    daily = ops_daily_from_monthly(flow_series, monthly, method=method).dropna()
    got = daily.resample("ME").sum()
    want = monthly.reindex(got.index)
    both = want.notna()
    agree = np.isclose(got[both], want[both], rtol=1e-6)
    assert agree.sum() >= both.sum() - 1, (
        f"{method}: {int((~agree).sum())} month(s) whose daily values do not sum to the monthly "
        f"total")


# ── 2. rolling-origin discipline: no future month informs a past comparison ──

@needs_data
@pytest.mark.parametrize("target", [FLOW, "Expenditure"])
def test_the_baseline_is_truncation_invariant(target):
    """The causality property, proved by recomputation rather than asserted.

    Each figure depends only on COMPLETE PRIOR calendar years, so recomputing from data cut at an
    earlier origin must reproduce the same values. If it did not, a full-history average would be
    leaking later months into earlier comparisons -- the failure this test is named for.
    """
    s = ob._flow_series(DATA, target)
    full = ops_daily_from_monthly(s, ops_monthly_baseline_treasury(s), method="flat")
    checked = 0
    for cut in ("2025-01-01", "2025-03-03", "2025-06-02", TEST_END):
        st = s.loc[:cut]
        tr = ops_daily_from_monthly(st, ops_monthly_baseline_treasury(st), method="flat")
        common = full.dropna().index.intersection(tr.dropna().index)
        if not len(common):
            continue
        a, b = full.reindex(common), tr.reindex(common)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-6), (
            f"{target}: truncating at {cut} changed {(~np.isclose(a, b)).sum()} earlier values, "
            f"so the full-history computation is leaking forward")
        checked += len(common)
    assert checked > 100, f"only {checked} values compared"


@needs_data
def test_a_january_row_uses_the_vintage_its_origin_actually_had():
    """The one place causality is not automatic, and the construction that fixes it.

    At h=5, four sealed-window target dates have origins in December 2024 -- before 2024 closed.
    The 2025 baseline (built from 2022-2024) did not exist at those origins; the Treasury had the
    window ending 2023. Each row is therefore scored against the vintage in force at ITS OWN
    origin.
    """
    targets = pd.bdate_range(TEST_START, TEST_END)
    origins = targets - pd.offsets.BDay(5)
    early = [(t, o) for t, o in zip(targets, origins) if o < pd.Timestamp("2024-12-31")]
    assert len(early) == 4, f"expected 4 December-origin rows, got {len(early)}"

    for _t, o in early:
        assert ob._vintage_year(o) == 2023, (
            f"origin {o.date()} precedes 2024's close, so its latest complete year is 2023")
    for _t, o in zip(targets[10:], origins[10:]):
        assert ob._vintage_year(o) == 2024

    cache = ob.vintage_cache(DATA, FLOW, origins, targets)
    assert set(cache) == {2023, 2024}, f"expected two vintages, got {sorted(cache)}"
    values = [ob.ops_prediction_for(t, o, cache)[0] for t, o in zip(targets, origins)]
    assert np.isfinite(values).all(), (
        "the vintage construction exists so no row is dropped; "
        f"{int((~np.isfinite(values)).sum())} are unusable")


@needs_data
def test_the_flat_spread_emits_no_negative_planning_figure():
    """Why flat is canonical. A negative *revenue* baseline is not defensible to a reader, and
    the profile spread produces some because the raw series contains negative days."""
    targets = pd.bdate_range(TEST_START, TEST_END)
    origins = targets - pd.offsets.BDay(5)
    flat = ob.vintage_cache(DATA, FLOW, origins, targets, spread=ob.SPREAD_FLAT)
    prof = ob.vintage_cache(DATA, FLOW, origins, targets, spread=ob.SPREAD_PROFILE)
    fv = np.array([ob.ops_prediction_for(t, o, flat)[0] for t, o in zip(targets, origins)])
    pv = np.array([ob.ops_prediction_for(t, o, prof)[0] for t, o in zip(targets, origins)])
    assert (fv < 0).sum() == 0, f"flat produced {(fv < 0).sum()} negative baselines"
    assert (pv < 0).sum() > 0, (
        "the profile spread is expected to inherit the data's negative days; if it no longer "
        "does, the reason flat was chosen has changed and should be revisited")
    assert ob.DEFAULT_SPREAD == ob.SPREAD_FLAT


# ── 3. flows only; a stock says why, and does not get a fabricated figure ────

@needs_data
def test_a_stock_target_has_no_ops_baseline_and_says_so():
    assert ob.ops_daily_series(DATA, STOCK) is None
    assert ob.ops_series_for_vintage(DATA, STOCK, 2024, [pd.Timestamp("2025-01-31")]) is None
    assert "annual total" in ob.REASON_STOCK


# ── 4. reported in both places ──────────────────────────────────────────────

def test_skill_vs_ops_is_in_the_scorecard_schema():
    for field in ("ops_pred", "ops_abs_error", "skill_vs_ops", "ops_source"):
        assert field in SCORECARD_COLUMNS, field
    cols = list(SCORECARD_COLUMNS)
    assert cols.index("skill_vs_ops") > cols.index("skill_vs_ruler_pct"), (
        "the ops comparison should read after the naive one, as its companion")


def test_the_committed_scorecard_header_carries_the_ops_columns():
    sc = REPO / "forecasts" / "scorecard.csv"
    if not sc.exists():
        pytest.skip("no scorecard on disk")
    header = sc.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(SCORECARD_COLUMNS)
    assert "skill_vs_ops" in header


def test_the_leaderboard_writer_emits_the_ops_comparison():
    """Asserted on the writer, because no sealed leaderboard has been regenerated yet."""
    src = (BACKEND / "b_ml_pipeline.py").read_text(encoding="utf-8")
    assert "skill_vs_ops_pct" in src and "ops_MAE" in src
    assert "log_sealed_window_read" in src, "the leaderboard's holdout read must reach the ledger"


def test_both_comparators_are_present_wherever_one_is():
    """The point of the addition: never one skill figure without the other."""
    cols = set(SCORECARD_COLUMNS)
    assert {"skill_vs_ruler_pct", "skill_vs_ops"} <= cols
    assert {"persistence_pred", "ops_pred"} <= cols
    assert {"persistence_abs_error", "ops_abs_error"} <= cols
    assert {"persistence_source", "ops_source"} <= cols


# ── 5. it gates nothing ─────────────────────────────────────────────────────

def test_the_publication_gate_does_not_read_the_ops_comparison():
    """Ops skill is reporting. Publication still turns on the naive ruler alone."""
    src = (BACKEND / "publication_gates.py").read_text(encoding="utf-8")
    for token in ("ops", "Ops", "OPS"):
        assert token not in src, (
            f"publication_gates mentions {token!r}; the ops comparison must not reach gate logic")


def test_the_three_verdicts_are_unchanged_by_the_ops_addition():
    """Pinned against the registry, which predates this session's work.

    Five of eleven models measured on revenues are WORSE than the current method, so if ops skill
    ever leaked into gating it would change what publishes. It must not.
    """
    from publication_gates import Measured, decide
    from registry import load_registry

    log = pd.read_csv(REPO / "experiments" / "log.csv")
    expected = {"Revenues": "publishable", "Expenditure": "withheld",
                "State budget balance": "withheld"}
    for r in load_registry()["recipes"]:
        row = log[log["run_id"] == r["dev_credentials"]["run_id"]].iloc[0]
        out = decide(Measured(target=r["target"], mase=row["mase"],
                              sentinel_ratio=row["sentinel_ratio"],
                              skill_vs_ruler_pct=row["skill_vs_ruler"],
                              leakage_detected=False, persistence_mimicry=False))
        assert out["verdict"] == expected[r["target"]] == r["publication"]["verdict"], (
            f"{r['target']}: verdict moved to {out['verdict']}")


def test_measured_gate_inputs_contain_no_ops_field():
    """Structural: the Measured dataclass has no ops slot, so a gate cannot read one."""
    from publication_gates import Measured
    assert not [f for f in Measured.__dataclass_fields__ if "ops" in f.lower()]
