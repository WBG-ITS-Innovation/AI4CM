"""P0: the two defects that made published realized accuracy wrong.

Both were found by exercising the scorer against synthetic actuals
(docs/sessions/2026-08-13-refresh-retrain-score-diagnostic.md, item 3). Neither
threw, neither showed up in any existing test, and both changed the headline
number a client would be shown.

  1. **The wrong persistence ruler.** ``score_one`` recomputed the comparator as
     ``truth.iloc[pos - horizon_steps]`` with ``horizon_steps`` fixed at 5 for every
     row, while ``row["horizon"]`` sat unread two lines away. A published issue has
     ONE origin, so ``ŷ(t+h) = y(t)`` is the same number for all five horizons — and
     it is already in the artifact as ``origin_value``. The scorer used five different
     values and only h=5 was right: 12 of 15 realized skills and all three per-target
     aggregates were wrong. Revenues moved from -3.39% to +35.79% on re-score.

  2. **A missing actual scored as zero.** ``_truth_series`` routed the actuals through
     ``b_ml_pipeline.to_business_index``, which zero-fills missing flow days and
     forward-fills stocks. Both are correct for *modelling*, which needs a dense index.
     For *scoring* they mean ``TruthNotAvailable`` can never fire for a flow inside the
     data range: a genuinely absent day is scored as a real observation of 0.00 with a
     fabricated error in the tens of millions.

Each test below is paired with a mutation: the fix is reintroduced-in-reverse and the
test must fail. A regression test that passes against the bug is not a regression test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import published_forecasts as PF  # noqa: E402
from published_forecasts import (  # noqa: E402
    SCORECARD_COLUMNS,
    TruthNotAvailable,
    baseline_agrees,
    score_one,
    score_published,
)

REPO = BACKEND.parent
LIVE_ISSUE = REPO / "forecasts" / "published" / "2025-08-06"


# ─────────────────────────── fixtures ───────────────────────────

TARGETS = ("Revenues", "Expenditure", "State budget balance")


def _actuals_csv(path: Path, *, omit: str | None = None) -> Path:
    """A tiny canonical-shaped file: business days, three targets, optional gap."""
    days = pd.bdate_range("2025-06-02", "2025-08-13")
    rng = np.random.default_rng(20260813)
    frame = pd.DataFrame({"date": days})
    for i, t in enumerate(TARGETS):
        scale = 5.0e7 if t != "State budget balance" else 1.7e9
        frame[t] = scale * rng.normal(1.0, 0.15, size=len(days))
    if omit is not None:
        frame = frame[frame["date"] != pd.Timestamp(omit)]
    frame.to_csv(path, index=False)
    return path


def _published_issue(root: Path, origin: str, origin_value: float,
                     target_dates: list[str], target: str = "Revenues") -> Path:
    """A published issue in the real shape: one origin, five horizons, one origin_value."""
    d = root / origin
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"target": target, "horizon": h, "origin_date": origin,
             "origin_value": origin_value, "target_date": td,
             "p10": 1.0e7, "p50": 4.0e7, "p90": 9.0e7,
             "point_model": "LightGBM_L1", "target_transform": "ratio"}
            for h, td in enumerate(target_dates, start=1)]
    pd.DataFrame(rows).to_csv(d / "forecast.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({
        "issue_date": origin, "recipes": [{"target": target, "recipe_id": "rev-v1"}],
    }), encoding="utf-8")
    return d


# ─────────────── defect 1: the ruler is read, not recomputed ───────────────

def test_one_origin_means_one_ruler_for_every_horizon(tmp_path):
    """The arithmetic the diagnostic proved: five horizons, five identical baselines.

    Before the fix these were five different numbers — 82,103,400 / 104,560,400 /
    73,486,530 / 75,125,620 / 46,490,790 — of which only the last was right.
    """
    data = _actuals_csv(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", 46_490_793.48,
                     ["2025-08-07", "2025-08-08", "2025-08-11",
                      "2025-08-12", "2025-08-13"])

    out = score_published(data, published_root=root,
                          scorecard_path=tmp_path / "sc.csv")
    sc = pd.read_csv(tmp_path / "sc.csv")

    assert out["scored"] == 5
    assert sc["persistence_pred"].nunique() == 1, (
        f"one issue has one origin, so one ruler; got "
        f"{sorted(sc['persistence_pred'].unique())}")
    assert sc["persistence_pred"].iloc[0] == pytest.approx(46_490_793.48)
    assert (sc["persistence_source"] == "artifact: origin_value").all()


def test_the_scored_ruler_equals_the_artifacts_origin_value(tmp_path):
    """The one-implementation rule, on the artifact — as test_published_baseline_is_shared
    does for the DEV ruler. What ships must be the number that was published."""
    data = _actuals_csv(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", 46_490_793.48,
                     ["2025-08-07", "2025-08-08", "2025-08-11",
                      "2025-08-12", "2025-08-13"])
    score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")

    fc = pd.read_csv(root / "2025-08-06" / "forecast.csv")
    sc = pd.read_csv(tmp_path / "sc.csv")

    # The scorecard now records `origin_value` itself, so this no longer has to reach back
    # into the artifact to find the number -- which makes the check stricter, not merely
    # tidier. Two things must hold: what the scorecard recorded is what the artifact
    # published, and the ruler it scored against is that same number.
    published = fc.set_index(["target", "horizon"])["origin_value"]
    assert len(sc) == len(fc)
    for _, r in sc.iterrows():
        artifact = float(published.loc[(r["target"], int(r["horizon"]))])
        assert r["origin_value"] == pytest.approx(
            artifact, rel=PF.BASELINE_RTOL, abs=PF.BASELINE_ATOL), (
            f"h={r['horizon']}: scorecard recorded origin_value "
            f"{r['origin_value']:,.2f} but the artifact published {artifact:,.2f}")
        assert r["persistence_pred"] == pytest.approx(
            r["origin_value"], rel=PF.BASELINE_RTOL, abs=PF.BASELINE_ATOL), (
            f"h={r['horizon']}: scored against {r['persistence_pred']:,.2f} but the "
            f"row's own origin_value is {r['origin_value']:,.2f}")


def test_the_live_published_issue_carries_one_origin_value_per_target():
    """The premise the fix rests on, checked against the real committed artifact."""
    if not LIVE_ISSUE.exists():
        pytest.skip("no live published issue")
    fc = pd.read_csv(LIVE_ISSUE / "forecast.csv")
    for target, g in fc.groupby("target"):
        assert g["origin_value"].nunique() == 1, (
            f"{target}: {g['origin_value'].nunique()} origin values in one issue")
        assert g["origin_date"].nunique() == 1
        assert sorted(g["horizon"]) == [1, 2, 3, 4, 5]


def test_a_row_without_origin_value_falls_back_to_its_own_horizon(tmp_path):
    """The fallback must read the row's h, not a hardcoded 5.

    This is the mutation-sensitive one: with `horizon_steps` fixed at 5, an h=2 row
    is compared against the value 5 business days back instead of 2.
    """
    idx = pd.bdate_range("2025-01-01", periods=60)
    truth = pd.Series(np.arange(60, dtype=float) * 1e6, index=idx)
    td = idx[40]
    row = {"target": "Revenues", "target_date": str(td.date()), "horizon": 2,
           "p10": 0.0, "p50": 1.0, "p90": 1e12}          # no origin_value

    got = score_one(row, truth)
    assert got["persistence_pred"] == pytest.approx(float(truth.iloc[38]))
    assert got["persistence_pred"] != pytest.approx(float(truth.iloc[35]))
    assert got["persistence_source"].startswith("recomputed")


def test_reintroducing_the_fixed_horizon_fails_this_suite(tmp_path):
    """Mutation test: put the bug back, and the ruler assertions must break.

    Monkeypatched rather than edited, so the mutation cannot survive the test run.
    """
    idx = pd.bdate_range("2025-01-01", periods=60)
    truth = pd.Series(np.arange(60, dtype=float) * 1e6, index=idx)
    td = idx[40]
    row = {"target": "Revenues", "target_date": str(td.date()), "horizon": 2,
           "origin_value": float(truth.iloc[38]),
           "p10": 0.0, "p50": 1.0, "p90": 1e12}

    fixed = score_one(row, truth)["persistence_pred"]
    assert fixed == pytest.approx(float(truth.iloc[38]))

    def buggy_persistence(row, truth, td, horizon_steps):        # the pre-fix behaviour
        pos = truth.index.get_loc(td)
        return float(truth.iloc[pos - horizon_steps]), "recomputed: fixed 5", np.nan

    original = PF._persistence_for
    PF._persistence_for = buggy_persistence
    try:
        mutated = score_one(row, truth)["persistence_pred"]
    finally:
        PF._persistence_for = original

    assert mutated == pytest.approx(float(truth.iloc[35]))
    assert mutated != pytest.approx(fixed), (
        "the mutation produced the same ruler, so these tests would not catch it")


def test_a_disagreeing_origin_value_is_reported(tmp_path):
    """If the actuals move under an issued forecast, that must be visible.

    The row still scores — the published number is the comparator that was committed
    to — but the divergence is reported rather than averaged into a headline in silence.
    """
    data = _actuals_csv(tmp_path / "actuals.csv")
    truth = PF._truth_series(data, "Revenues")
    origin_actual = float(truth.loc[pd.Timestamp("2025-08-06")])

    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", origin_actual + 5.0e6,   # tampered
                     ["2025-08-07", "2025-08-08", "2025-08-11",
                      "2025-08-12", "2025-08-13"])
    out = score_published(data, published_root=root,
                          scorecard_path=tmp_path / "sc.csv")

    assert out["scored"] == 5
    assert len(out["baseline_disagreements"]) == 5
    first = out["baseline_disagreements"][0]
    assert first["delta"] == pytest.approx(5.0e6)
    assert not baseline_agrees(first["artifact_origin_value"],
                               first["recomputed_from_actuals"])


def test_agreement_tolerance_is_a_round_trip_not_a_margin():
    assert baseline_agrees(46_490_793.48, 46_490_793.48)
    assert baseline_agrees(46_490_793.48, 46_490_793.4800001)
    assert not baseline_agrees(46_490_793.48, 46_490_800.0)   # 6.52 apart
    assert baseline_agrees(1.0e8, np.nan)                     # nothing to compare


# ─────────────── defect 2: a missing actual is pending, not zero ───────────────

def test_a_missing_flow_day_is_pending_not_a_real_zero(tmp_path):
    """The defect verbatim: 2025-08-11 removed from the actuals.

    Before the fix this scored with y_true = 0.00 and an abs_error of 77.6M.
    """
    data = _actuals_csv(tmp_path / "actuals.csv", omit="2025-08-11")
    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", 46_490_793.48,
                     ["2025-08-07", "2025-08-08", "2025-08-11",
                      "2025-08-12", "2025-08-13"])

    out = score_published(data, published_root=root,
                          scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 4
    assert out["pending"] == 1
    assert ("Revenues", "2025-08-11") in out["pending_dates"]

    sc = pd.read_csv(tmp_path / "sc.csv")
    assert "2025-08-11" not in set(sc["target_date"].astype(str))
    assert not (sc["y_true"] == 0.0).any(), "a gap was scored as an observation of zero"


def test_a_missing_stock_day_is_not_forward_filled(tmp_path):
    """The stock half of the same defect.

    `to_business_index` ffills a stock, so a missing balance day was scored against
    yesterday's balance — a modelling assumption presented as an observation. In the
    diagnostic's gap run that produced a skill of -3318%.
    """
    data = _actuals_csv(tmp_path / "actuals.csv", omit="2025-08-11")
    truth = PF._truth_series(data, "State budget balance")
    assert np.isnan(truth.loc[pd.Timestamp("2025-08-11")]), (
        "a stock gap was forward-filled; scoring would use yesterday's value as truth")

    row = {"target": "State budget balance", "target_date": "2025-08-11", "horizon": 3,
           "origin_value": 1.7e9, "p10": 0.0, "p50": 1.7e9, "p90": 1e12}
    with pytest.raises(TruthNotAvailable):
        score_one(row, truth)


def test_the_scoring_series_is_not_the_modelling_series(tmp_path):
    """Both series exist on purpose; the scorer must use the un-filled one.

    Stated as a difference rather than as an implementation detail, so the test still
    means something if either function is rewritten.
    """
    data = _actuals_csv(tmp_path / "actuals.csv", omit="2025-08-11")
    from b_ml_pipeline import to_business_index

    modelling = to_business_index(pd.read_csv(data), "date", "Revenues")
    scoring = PF._truth_series(data, "Revenues")
    gap = pd.Timestamp("2025-08-11")

    assert modelling.loc[gap] == 0.0, "the modelling series is expected to zero-fill"
    assert np.isnan(scoring.loc[gap]), "the scoring series must leave the gap open"
    present = scoring.dropna().index
    assert modelling.index.equals(scoring.index)        # same calendar
    assert gap not in present and len(present) == len(modelling) - 1


def test_reintroducing_the_zero_fill_fails_this_suite(tmp_path):
    """Mutation test: route scoring back through the modelling series.

    With the zero-fill restored, the missing day is finite, TruthNotAvailable cannot
    fire, and the row is scored with y_true = 0.
    """
    data = _actuals_csv(tmp_path / "actuals.csv", omit="2025-08-11")
    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", 46_490_793.48,
                     ["2025-08-07", "2025-08-08", "2025-08-11",
                      "2025-08-12", "2025-08-13"])

    def zero_filled(data_path, target):                 # the pre-fix behaviour
        from b_ml_pipeline import to_business_index
        return to_business_index(pd.read_csv(data_path), "date", target)

    original = PF._truth_series
    PF._truth_series = zero_filled
    try:
        out = score_published(data, published_root=root,
                              scorecard_path=tmp_path / "sc.csv")
        sc = pd.read_csv(tmp_path / "sc.csv")
    finally:
        PF._truth_series = original

    assert out["pending"] == 0, "the mutation should make the gap invisible"
    assert (sc["y_true"] == 0.0).any(), (
        "the mutation did not reproduce the fabricated zero, so the regression test "
        "above would not catch it")


# ─────────────── the scorecard still describes itself ───────────────

def test_persistence_source_is_recorded_for_every_scored_row(tmp_path):
    """Which ruler was used is part of the record, not folklore."""
    data = _actuals_csv(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _published_issue(root, "2025-08-06", 46_490_793.48, ["2025-08-07", "2025-08-08"])
    score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")

    sc = pd.read_csv(tmp_path / "sc.csv")
    assert list(sc.columns) == list(SCORECARD_COLUMNS)
    assert sc["persistence_source"].notna().all()
