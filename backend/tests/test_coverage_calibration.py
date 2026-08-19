"""Interval coverage: reported per family, and calibrated without touching the holdout.

Two obligations are pinned here.

**1. Coverage is reported per family, against each family's own advertised level.** It used to
be computed in three places and gated in none: every recipe in ``registry/recipes.json`` shipped
``p10``/``p90`` while its coverage gate read *"This model reports no prediction intervals."*
``backend/coverage_report`` is the single reader, and these tests hold it to reporting a figure
for every family that publishes a band -- and to saying so plainly, rather than scoring zero,
for a family that does not.

**2. The calibration step never reads test-period data.** Split-conformal calibration is only
honest if the conformity scores come from targets already known when the band was issued. If a
single calibration row's target fell inside the evaluation window, the correction would be
measured partly against the answers it is meant to be judged on, and the band would look better
calibrated than it is. The last test drives the real pipeline over data that runs into the
sealed holdout and asserts that every row the calibration touched predates it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import e_quantile_daily_pipeline as eqp                                  # noqa: E402
from conformal import (DEFAULT_ALPHA, causal_calibration_split,          # noqa: E402
                       conformity_scores, conformal_width)
from coverage_report import (FAMILY_INTERVALS, CoverageMeasurement,      # noqa: E402
                            SelectionOnReportedCoverageError,
                            coverage_for_publication, measure_all,
                            score_band, to_frame)
from evaluation_windows import (DEV_START, TEST_END, TEST_START,        # noqa: E402
                                window_for)


# ── 1. coverage is reported per family ───────────────────────────────────────

@pytest.fixture(scope="module")
def measured():
    ms = measure_all()
    if not ms:
        pytest.skip("no forecast_runs artifacts on disk to score")
    return ms


def test_every_family_that_publishes_a_band_gets_a_coverage_figure(measured):
    """Per-family reporting is the requirement; silence for a band that exists is the defect."""
    scored = [m for m in measured if m.n > 0]
    assert scored, "artifacts exist but nothing was scored"
    families = {m.family for m in scored}
    assert families, "no family produced a scored band"
    for m in scored:
        assert m.overall is not None, f"{m.family}/{m.model} has rows but no coverage figure"
        assert 0.0 <= m.overall <= 1.0
        assert m.nominal is not None, (
            f"{m.family} publishes a band with no advertised level, so nothing can be scored "
            f"against it -- that must be fixed in FAMILY_INTERVALS, not papered over here")


def test_the_report_names_the_family_target_and_model_for_every_row(measured):
    """A coverage number without its owner cannot be acted on."""
    f = to_frame(measured)
    for col in ("family", "target", "model", "n", "nominal", "overall_coverage",
                "large_day_coverage", "windows", "magnitude_basis"):
        assert col in f.columns, col
    assert f["family"].notna().all() and (f["family"] != "").all()
    assert f["target"].notna().all() and (f["target"] != "").all()


def test_each_family_is_scored_against_its_own_advertised_level():
    """Scoring an 80% band against 90% manufactures a ten-point defect that is not there."""
    assert FAMILY_INTERVALS["e_quantile"]["nominal"] == pytest.approx(0.80)
    assert FAMILY_INTERVALS["b_ml"]["nominal"] == pytest.approx(0.90)
    assert FAMILY_INTERVALS["c_dl"]["nominal"] == pytest.approx(0.90)
    for fam, spec in FAMILY_INTERVALS.items():
        assert spec["nominal_source"], f"{fam} must say where its level comes from"


def test_a_family_with_empty_interval_columns_says_so_rather_than_scoring_zero():
    """A_STAT writes y_lo/y_hi and never fills them. That is 'no intervals', not '0% coverage'."""
    g = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_lo": [np.nan] * 3, "y_hi": [np.nan] * 3,
                      "y_pred": [1.0, 2.0, 3.0]})
    m = score_band(g, family="a_stat", target="T", model="ETS",
                   spec=FAMILY_INTERVALS["a_stat"])
    assert m.n == 0
    assert m.overall is None, "an absent band must never read as 0% coverage"
    assert "no intervals" in m.note or "published no intervals" in m.note


def test_large_day_coverage_is_bucketed_on_the_forecast_not_the_outcome():
    """The whole point of the metric fix, asserted on the reader that feeds the gate."""
    n = 400
    rng = np.random.default_rng(4)
    p50 = rng.normal(0, 100, n)
    sd = rng.uniform(10, 200, n)
    y = p50 + rng.normal(0, 1, n) * sd
    g = pd.DataFrame({"y_true": y, "yhat_p50": p50,
                      "yhat_p10": p50 - 1.2816 * sd,      # an EXACT 80% band
                      "yhat_p90": p50 + 1.2816 * sd})
    m = score_band(g, family="e_quantile", target="T", model="GBQuantile",
                   spec=FAMILY_INTERVALS["e_quantile"])
    assert m.overall == pytest.approx(0.80, abs=0.05)
    assert m.large_day == pytest.approx(0.80, abs=0.12), (
        "a band that is correct by construction must not be reported as failing on big days; "
        f"got {m.large_day}")
    assert "forecast" in m.basis


def test_publication_coverage_matches_the_recipe_s_own_interval_model():
    """Gating a recipe on another model's band reports a calibration the client never receives."""
    ms = [
        CoverageMeasurement(family="c_dl", target="T", model="DCNN", n=2000, nominal=0.90,
                            nominal_source="x", overall=0.90, windows=("dev", "test", "train")),
        CoverageMeasurement(family="e_quantile", target="T", model="GBQuantile", n=200,
                            nominal=0.80, nominal_source="x", overall=0.62, windows=("test",)),
    ]
    picked = coverage_for_publication(ms, "T", interval_model="GBQuantile")
    assert picked is not None and picked.model == "GBQuantile"
    assert picked.overall == pytest.approx(0.62), (
        "the recipe's own band measures 62%; picking the row with the most rows would have "
        "reported 90% from a model the recipe does not ship")
    assert coverage_for_publication(ms, "T", interval_model="NoSuchModel") is None, (
        "an absent measurement must be reported absent, never substituted")


def test_a_single_window_measurement_is_preferred_over_a_pooled_one():
    ms = [
        CoverageMeasurement(family="e_quantile", target="T", model="GBQuantile", n=3000,
                            nominal=0.80, nominal_source="x", overall=0.88,
                            windows=("dev", "test", "train")),
        CoverageMeasurement(family="e_quantile", target="T", model="GBQuantile", n=200,
                            nominal=0.80, nominal_source="x", overall=0.74, windows=("test",)),
    ]
    picked = coverage_for_publication(ms, "T", interval_model="GBQuantile")
    assert picked.windows == ("test",), (
        "a figure pooled across train/dev/test does not describe out-of-sample calibration")


def test_reported_coverage_may_not_be_used_to_choose():
    m = [CoverageMeasurement(family="e_quantile", target="T", model="GBQuantile", n=200,
                             nominal=0.80, nominal_source="x", overall=0.74, windows=("test",))]
    assert coverage_for_publication(m, "T", purpose="report") is not None
    with pytest.raises(SelectionOnReportedCoverageError):
        coverage_for_publication(m, "T", purpose="selection")


# ── 2. the calibration step never reads test-period data ─────────────────────

@pytest.mark.parametrize("h", [1, 5, 10, 21])
def test_the_calibration_slice_is_separated_from_the_fit_by_exactly_h(h):
    """Row t carries the target y(t+h), so without the gap the last h fit rows have answers
    inside the calibration slice."""
    fit_ix, cal_ix = causal_calibration_split(800, h)
    assert cal_ix[0] - fit_ix[-1] - 1 == h
    assert set(fit_ix).isdisjoint(cal_ix)


def test_conformalise_is_handed_no_row_dated_in_the_test_window(monkeypatch):
    """Unit-level: whatever the pipeline passes as the calibration source must predate TEST.

    Asserted on the *dates the model is fitted and scored on*, captured from inside
    ``_predict_quantiles``, rather than on the argument names -- a future refactor could rename
    the variables and keep the leak.
    """
    idx = pd.bdate_range("2015-01-05", "2023-12-29")     # entirely inside TRAIN
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = np.arange(len(idx), dtype=float)

    seen: list[pd.Timestamp] = []

    def spy(model_name, CONFIG, X_tr, y_tr, X_new, quantiles):
        seen.extend(list(X_tr.index) + list(X_new.index))
        return {q: np.zeros(len(X_new)) for q in quantiles}, 0

    monkeypatch.setattr(eqp, "_predict_quantiles", spy)
    cfg = eqp.Config(target="Revenues", cadence="Daily", horizon=5, data_path="unused")
    q_preds = {0.10: np.zeros(3), 0.50: np.zeros(3), 0.90: np.zeros(3)}
    eqp._conformalise("GBQuantile", cfg, X, y, q_preds, lo_q=0.10, hi_q=0.90,
                      alpha=DEFAULT_ALPHA)

    assert seen, "the calibration path fitted nothing, so this test proved nothing"
    offending = sorted({window_for(t) for t in seen} - {"train", "dev"})
    assert not offending, f"calibration touched report-only window(s): {offending}"
    assert max(seen) < pd.Timestamp(TEST_START)


def test_the_correction_is_computed_only_from_targets_known_at_the_origin():
    """The conformity scores must come from the calibration slice, never the evaluation rows."""
    rng = np.random.default_rng(2)
    n = 600
    y = rng.normal(0, 1, n)
    lo, hi = y - 5, y + 5                     # comically wide: any leak would shrink the width
    fit_ix, cal_ix = causal_calibration_split(n, 5)
    w_cal = conformal_width(conformity_scores(y[cal_ix], lo[cal_ix], hi[cal_ix]), DEFAULT_ALPHA)
    w_all = conformal_width(conformity_scores(y, lo, hi), DEFAULT_ALPHA)
    assert w_cal == pytest.approx(-5.0, abs=0.5) and w_all == pytest.approx(-5.0, abs=0.5)
    assert len(cal_ix) < n, "the calibration slice must be a strict subset"


def _synthetic_csv(tmp_path, seed=0):
    idx = pd.bdate_range("2015-01-05", TEST_END)
    rng = np.random.default_rng(seed)
    csv = tmp_path / "synthetic.csv"
    pd.DataFrame({"date": idx,
                  "Revenues": 1e8 + np.cumsum(rng.normal(0, 1e6, len(idx)))}).to_csv(csv,
                                                                                     index=False)
    return csv


def test_a_correction_that_would_invert_the_band_is_refused(monkeypatch):
    """Tightening is legitimate; tightening past zero width is not.

    A negative conformal width means the band was wider than it needed to be, and shrinking it is
    the correction working. But a p10 above its own p90 is not a narrow interval, it is a
    meaningless one, and this family treats crossing as reportable rather than sortable.
    """
    idx = pd.bdate_range("2015-01-05", "2023-12-29")
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = np.zeros(len(idx))

    # Calibration sees a band vastly wider than the actual spread, so the width is very negative.
    def spy(model_name, CONFIG, X_tr, y_tr, X_new, quantiles):
        return {0.10: np.full(len(X_new), -1e9), 0.90: np.full(len(X_new), 1e9)}, 0

    monkeypatch.setattr(eqp, "_predict_quantiles", spy)
    cfg = eqp.Config(target="Revenues", cadence="Daily", horizon=5, data_path="unused")
    narrow = {0.10: np.array([-1.0, -1.0]), 0.50: np.zeros(2), 0.90: np.array([1.0, 1.0])}
    out, note = eqp._conformalise("GBQuantile", cfg, X, y, narrow, lo_q=0.10, hi_q=0.90,
                                  alpha=DEFAULT_ALPHA)
    assert "invert" in note, note
    assert np.array_equal(out[0.10], narrow[0.10]), "the band must be returned untouched"
    assert np.all(out[0.10] <= out[0.90]), "a returned band must never be crossed"


def _trace_pipeline(tmp_path, monkeypatch, **cfg_kw):
    """Run the pipeline, recording every (fold, is_calibration, rows fitted, rows predicted).

    Returns the trace. Folds are recognised by order: each fold's main fit happens first, then
    ``_conformalise`` refits on a subset, so a calibration entry always follows the main entry
    for the same fold.
    """
    csv = _synthetic_csv(tmp_path, seed=cfg_kw.pop("seed", 0))
    trace = []
    real = eqp._predict_quantiles
    real_conformalise = eqp._conformalise
    in_calibration = {"flag": False}

    def wrapped_conformalise(*a, **kw):
        in_calibration["flag"] = True
        try:
            return real_conformalise(*a, **kw)
        finally:
            in_calibration["flag"] = False

    def spy(model_name, CONFIG, X_tr, y_tr, X_new, quantiles):
        trace.append({"calibration": in_calibration["flag"],
                      "fit": pd.DatetimeIndex(X_tr.index),
                      "predicted": pd.DatetimeIndex(X_new.index)})
        return real(model_name, CONFIG, X_tr, y_tr, X_new, quantiles)

    monkeypatch.setattr(eqp, "_predict_quantiles", spy)
    monkeypatch.setattr(eqp, "_conformalise", wrapped_conformalise)

    cfg = eqp.Config(target="Revenues", cadence="Daily", horizon=5, data_path=str(csv),
                     min_train_years=4, model_filter="GBQuantile",
                     out_root=str(tmp_path / "out"), cqr=True, **cfg_kw)
    return trace, cfg


def test_calibration_never_uses_a_row_dated_at_or_after_the_origin_it_serves(tmp_path,
                                                                            monkeypatch):
    """The invariant that makes the correction honest, asserted end to end on the real pipeline.

    Note carefully what is and is not required. ``evaluation_windows`` states that "TRAIN is a
    floor, not a cap ... a fold predicting 2025-06-01 legitimately trains on everything up to
    2025-05-31. What never happens is training on data at or after the origin it is predicting
    from." So when the pipeline rolls its origins through the holdout, a later fold's calibration
    slice legitimately contains holdout-dated rows -- they are in the *past* relative to that
    fold's origin. Asserting "no test-dated row ever" would forbid correct rolling-origin
    practice; the real requirement is causality, and that is what this asserts:

    every row the calibration fits or scores on predates its fold's first evaluation origin,
    by at least the horizon.

    ``test_calibration_stays_out_of_the_holdout_when_scoring_dev`` covers the window-purity case,
    where the two coincide.
    """
    from evaluation_windows import SelectionOnReportOnlyDataError

    trace, cfg = _trace_pipeline(tmp_path, monkeypatch, folds=2,
                                 eval_start=TEST_START, eval_end=TEST_END)
    # Crowning a best model from holdout rows is a selection, and the split refuses it. All the
    # calibration happens before that point, so one run pins both properties.
    with pytest.raises(SelectionOnReportOnlyDataError):
        eqp.run_pipeline(cfg)

    checked = 0
    current_origin = None
    for entry in trace:
        if not entry["calibration"]:
            current_origin = entry["predicted"].min()      # this fold's first evaluation origin
            continue
        assert current_origin is not None, "a calibration fit preceded any fold"
        touched = entry["fit"].union(entry["predicted"])
        assert touched.max() < current_origin, (
            f"calibration used {touched.max().date()}, at or after the origin "
            f"{current_origin.date()} it is meant to serve")
        # The real invariant: the TARGET of the latest calibration row must already have
        # happened by the origin. A calibration row at date d carries y(d + h business days).
        # The index is a pure business-day range, so BDay arithmetic is exact here.
        latest_target = touched.max() + pd.offsets.BDay(cfg.horizon)
        assert latest_target <= current_origin, (
            f"the latest calibration row is dated {touched.max().date()}, so its target falls on "
            f"{latest_target.date()} -- after the origin {current_origin.date()} whose band it "
            f"corrects. That target had not happened yet when the band was issued.")
        checked += 1

    assert checked >= 2, f"only {checked} calibration step(s) were verified"


def test_calibration_stays_out_of_the_holdout_when_scoring_dev(tmp_path, monkeypatch):
    """The window-purity case: scoring DEV, calibration must not reach into TEST or LIVE.

    Here causality and window purity coincide -- every origin precedes the holdout, so nothing
    the calibration may legitimately see is holdout-dated. This is the literal "calibration never
    reads test-period data" check, asserted where it is actually the right requirement.
    """
    trace, cfg = _trace_pipeline(tmp_path, monkeypatch, folds=2, seed=2,
                                 eval_start=DEV_START, eval_end="2024-12-31")
    eqp.run_pipeline(cfg)

    calib = [e for e in trace if e["calibration"]]
    assert calib, "CQR never ran, so nothing about calibration was verified"
    for entry in calib:
        touched = entry["fit"].union(entry["predicted"])
        offending = sorted({window_for(t) for t in touched} - {"train", "dev"})
        assert not offending, f"calibration reached into {offending}"
        assert touched.max() < pd.Timestamp(TEST_START), (
            f"calibration's latest row is {touched.max().date()}, at or after {TEST_START}")


def test_the_pipeline_logs_its_holdout_read(tmp_path, monkeypatch):
    """E_QUANTILE evaluated the holdout on every daily run and recorded nothing.

    ``require_test_access`` is now called on the reporting path, so the ledger has a factual
    answer to "how often was the holdout consulted" for this family too.
    """
    import evaluation_windows as ew
    from evaluation_windows import SelectionOnReportOnlyDataError

    csv = _synthetic_csv(tmp_path, seed=1)
    calls = []

    def spy(reason, caller=None, purpose=ew.PURPOSE_SELECTION):
        calls.append({"reason": reason, "caller": caller, "purpose": purpose})

    monkeypatch.setattr(ew, "require_test_access", spy)

    cfg = eqp.Config(target="Revenues", cadence="Daily", horizon=5, data_path=str(csv),
                     folds=1, min_train_years=4, eval_start=TEST_START, eval_end=TEST_END,
                     model_filter="GBQuantile", out_root=str(tmp_path / "out"))
    # Crowning on holdout rows is refused (see the test above); the ledger call happens at fold
    # construction, well before that, which is the point -- the read is announced when it starts.
    with pytest.raises(SelectionOnReportOnlyDataError):
        eqp.run_pipeline(cfg)

    assert calls, "the holdout was evaluated and the ledger was not told"
    assert any(c["purpose"] == ew.PURPOSE_REPORT for c in calls), (
        "a reporting read must be logged as a report, not as a selection")
    assert any("E_QUANTILE" in (c["reason"] or "") for c in calls)
    assert any(str(TEST_END) in (c["reason"] or "") or "holdout" in (c["reason"] or "")
               for c in calls), "the log entry must say which dates were read"
