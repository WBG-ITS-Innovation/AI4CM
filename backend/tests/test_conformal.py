"""Part 4: CQR, the conditional-coverage gate, and the selection rule.

The gate is the point of this module, so most of these tests are about what it refuses to pass.
The band that motivated it covered 83% of days overall and 9.6% of the largest third; a marginal
gate called that acceptable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from conformal import (  # noqa: E402
    CONDITIONAL_FLOOR,
    DEFAULT_ALPHA,
    TERCILES,
    causal_calibration_split,
    conditional_coverage_gate,
    conformal_width,
    conformity_scores,
    coverage_by_bucket,
    cqr_calibrate,
    select_per_target,
    tercile_edges,
    volatility_terciles,
)


# ── conformity scores and the correction ──────────────────────────────────────

def test_conformity_score_is_signed_distance_outside_the_band():
    y = np.array([5.0, 15.0, 10.0])
    lo = np.array([8.0, 8.0, 8.0])
    hi = np.array([12.0, 12.0, 12.0])
    s = conformity_scores(y, lo, hi)
    assert s[0] == pytest.approx(3.0), "below the band by 3"
    assert s[1] == pytest.approx(3.0), "above the band by 3"
    assert s[2] == pytest.approx(-2.0), "inside, with 2 to spare"


def test_conformal_width_uses_the_finite_sample_rank():
    """The (n+1) is what makes the guarantee finite-sample rather than asymptotic."""
    s = np.arange(1.0, 101.0)          # 1..100
    w = conformal_width(s, alpha=0.20)
    assert w == pytest.approx(81.0), w   # ceil(101*0.8) = 81 -> 81st smallest


def test_conformal_width_refuses_when_the_sample_is_too_small():
    """With too few calibration rows no finite width can promise the level. Returning the largest
    observed score instead would imply a guarantee the data cannot support."""
    assert conformal_width(np.array([1.0, 2.0]), alpha=0.01) == float("inf")


def test_cqr_widening_lifts_coverage_to_at_least_nominal():
    rng = np.random.default_rng(0)
    n = 800
    y = rng.normal(0, 1, n)
    lo, hi = np.full(n, -0.2), np.full(n, 0.2)     # far too narrow
    cal = slice(0, 400)
    c = cqr_calibrate(y[cal], lo[cal], hi[cal], alpha=DEFAULT_ALPHA)
    lo2, hi2 = c.apply(lo[400:], hi[400:])
    cov = float(((y[400:] >= lo2) & (y[400:] <= hi2)).mean())
    assert cov >= 0.75, f"CQR failed to reach nominal: {cov:.1%}"


def test_grouped_calibration_gives_a_wider_correction_where_misses_concentrate():
    """The whole reason for grouping: a global widening adds the same absolute amount everywhere
    and so leaves the largest days under-covered."""
    n = 600
    small = np.full(n // 2, 1.0)
    large = np.full(n // 2, 100.0)
    y = np.concatenate([small, large])
    lo = y - 0.5
    hi = y + 0.5
    # make the LARGE rows miss badly
    y = y.copy()
    y[n // 2:] += 50.0
    c = cqr_calibrate(y, lo, hi, alpha=DEFAULT_ALPHA, grouped=True)
    assert c.grouped and c.per_bucket
    assert c.per_bucket[TERCILES[2]] > c.per_bucket[TERCILES[0]], (
        "grouped calibration did not concentrate the widening on the failing bucket"
    )


def test_grouped_calibration_falls_back_and_says_so_on_thin_buckets():
    y = np.arange(30.0)
    c = cqr_calibrate(y, y - 1, y + 1, grouped=True, min_per_bucket=100)
    assert "fell back to the global width" in c.note


# ── the calibration slice is causal ──────────────────────────────────────────

def test_calibration_split_leaves_exactly_h_rows_out():
    """Row t carries y(t+h), so without the gap the last h fit rows have answers inside the
    calibration slice and the correction is measured partly on data already seen."""
    for h in (1, 5, 10, 21):
        fit, cal = causal_calibration_split(1000, h)
        assert cal[0] - fit[-1] - 1 == h, f"gap is not h for h={h}"
        assert not set(fit) & set(cal)
        assert cal[-1] == 999, "calibration must be the most recent rows"


def test_calibration_split_refuses_impossible_geometry():
    with pytest.raises(ValueError):
        causal_calibration_split(6, 5)


def test_tercile_edges_come_from_calibration_and_are_reused():
    """If evaluation rows were bucketed by their own edges the buckets would shift between fit and
    use, and the correction would land in the wrong one."""
    cal_mag = np.arange(1.0, 100.0)
    e = tercile_edges(cal_mag)
    c = cqr_calibrate(cal_mag, cal_mag - 1, cal_mag + 1, grouped=True)
    assert c.bucket_edges == pytest.approx(e)


# ── the conditional-coverage gate ────────────────────────────────────────────

def _band_that_misses_the_big_days(n=300):
    y = np.linspace(1e8, 9e8, n)
    p = y.copy()
    p[200:] = y[200:] * 0.5
    half = 0.05e8
    return y, p - half, p + half


def test_gate_fails_the_band_a_marginal_gate_would_have_passed():
    """The motivating case: healthy overall, catastrophic on the largest days."""
    y, lo, hi = _band_that_misses_the_big_days()
    g = conditional_coverage_gate(y, lo, hi, magnitude=y)
    assert g["overall_coverage"] > 0.6, "fixture should look acceptable on average"
    assert g["passed"] is False
    assert g["n_failing_buckets"] >= 1
    assert "largest third" in g["reason_plain"]
    assert "marginal gate would have passed" in g["reason_plain"]


def test_gate_passes_a_band_that_holds_up_in_every_bucket():
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(1e8, 1e7, 150), rng.normal(9e8, 9e7, 150)])
    lo, hi = y - 3e8, y + 3e8
    g = conditional_coverage_gate(y, lo, hi, magnitude=y)
    assert g["passed"] is True
    assert g["n_failing_buckets"] == 0


def test_gate_reports_every_bucket_pass_or_fail():
    y, lo, hi = _band_that_misses_the_big_days()
    g = conditional_coverage_gate(y, lo, hi, magnitude=y,
                                  volatility=np.abs(np.gradient(y)))
    assert set(g["buckets"]) == {"magnitude", "volatility"}
    for axis in g["buckets"].values():
        assert len(axis) == 3, "all three buckets must be reported, not only failures"
        for st in axis.values():
            assert {"coverage", "n", "mean_width"} <= set(st)


def test_gate_scores_volatility_as_a_second_independent_axis():
    """GBQuantile's documented inverted response -- widest band at LOW volatility -- is a
    volatility failure that a magnitude-only gate cannot see."""
    n = 300
    rng = np.random.default_rng(5)
    vol = np.concatenate([np.full(200, 1.0), np.full(100, 50.0)])
    y = rng.normal(0, 1, n) * vol
    lo, hi = np.full(n, -3.0), np.full(n, 3.0)      # fine at low vol, hopeless at high
    g = conditional_coverage_gate(y, lo, hi, magnitude=np.abs(y), volatility=vol)
    vb = g["buckets"]["volatility"]
    assert vb[TERCILES[2]]["coverage"] < vb[TERCILES[0]]["coverage"]
    assert g["passed"] is False


def test_gate_returns_never_verified_rather_than_a_pass_on_thin_data():
    y = np.array([1.0, 2.0, 3.0])
    g = conditional_coverage_gate(y, y - 1, y + 1, magnitude=y, min_bucket_n=20)
    assert g["passed"] is None, "too little data must never read as a pass"
    assert "not verified" in g["reason_plain"]


def test_thin_buckets_are_excluded_from_the_verdict_not_failed():
    """A bucket with three rows should not decide a gate."""
    rng = np.random.default_rng(7)
    y = np.concatenate([rng.normal(0, 1, 200), np.array([1e9, 1.1e9, 1.2e9])])
    lo, hi = y - 3.0, y + 3.0
    lo[-3:], hi[-3:] = 0.0, 1.0                     # the three big rows all miss
    g = conditional_coverage_gate(y, lo, hi, magnitude=y, min_bucket_n=20)
    assert any("largest third" in t for t in g["thin_buckets"]) or g["passed"] is not None


# ── the selection rule ───────────────────────────────────────────────────────

def test_selection_prefers_a_passing_gate_over_a_lower_error():
    """An accurate point forecast with an unusable band is not the better product."""
    out = select_per_target([
        {"name": "sharp_bad_band", "dev_mae": 100.0,
         "conditional_gate": {"passed": False}},
        {"name": "ok_good_band", "dev_mae": 110.0,
         "conditional_gate": {"passed": True}},
    ])
    assert out["selected"] == "ok_good_band"


def test_selection_tie_break_keeps_l2_candidates_in_the_pool():
    """Expenditure's DEV-best is an L2 model, 1.0% ahead of the promoted L1 recipe. A strict
    argmin on one DEV fold would swap a recipe on noise; WS2 measured that a sub-1% DEV margin
    predicts nothing."""
    out = select_per_target([
        {"name": "HistGBDT_L2", "dev_mae": 51_088_706, "objective": "L2",
         "n_train_folds": 5, "sentinel": 1.0877,
         "conditional_gate": {"passed": True}},
        {"name": "LightGBM_L1_ws3", "dev_mae": 51_602_951, "objective": "L1",
         "n_train_folds": 5, "sentinel": 1.0882,
         "conditional_gate": {"passed": True}},
    ], mae_tie_pct=1.5)
    assert set(out["tied_within_pct"]) == {"HistGBDT_L2", "LightGBM_L1_ws3"}
    assert "L2" in [c["objective"] for c in out["ranked"]], "L2 was dropped from the pool"
    assert "predicts nothing" in out["reason"]


def test_selection_takes_the_argmin_when_nothing_is_tied():
    out = select_per_target([
        {"name": "a", "dev_mae": 100.0}, {"name": "b", "dev_mae": 200.0}],
        mae_tie_pct=1.0)
    assert out["selected"] == "a"
    assert "no other candidate is within" in out["reason"]


def test_selection_falls_back_and_states_the_defect_when_no_gate_passes():
    out = select_per_target([
        {"name": "a", "dev_mae": 100.0, "conditional_gate": {"passed": False}},
        {"name": "b", "dev_mae": 200.0, "conditional_gate": {"passed": False}}])
    assert out["selected"] == "a"
    assert "fell back to the full pool" in out["reason"]
    assert "band defect stated" in out["reason"]


def test_selection_with_no_candidates_returns_none():
    assert select_per_target([])["selected"] is None
