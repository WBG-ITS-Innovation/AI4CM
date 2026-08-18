"""The scorecard schema — fixed before the first real row exists.

Why this file exists
--------------------
Every scorecard row is a permanent claim about what was forecast and what happened. A field
added *after* rows are written cannot be back-filled for those rows: once the data has moved
on there is nowhere to get the missing value from. So the field set and its order were settled
while ``forecasts/scorecard.csv`` still held a header and zero rows, and these tests hold the
schema to what it promises.

Before this, the schema existed as a 22-name tuple with two inline comments, and three tests
asserted only that the written columns equalled that tuple — a self-consistency check that
would have passed just as happily if a required field were missing. One was: **origin**. With
``issue_date`` now wall-clock (the publish-CLI fix), nothing in a scored row recorded the data
vintage the forecast was made from, so a row could not say whether it was a five-day-ahead call
or a backfill.
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

from published_forecasts import (  # noqa: E402
    SCORECARD_COLUMNS,
    SUPPORTED_QUANTILE_COLUMNS,
    UnsupportedIntervalShape,
    SCORECARD_SCHEMA_VERSION,
    interval_nominal_of,
    score_published,
)

LIVE_SCORECARD = REPO / "forecasts" / "scorecard.csv"


# ── fixtures: a published issue whose truth has arrived ───────────────────────

def _actuals(path: Path, start="2024-01-02", n=300) -> Path:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(11)
    pd.DataFrame({"date": idx,
                  "Revenues": 4.0e7 + 5.0e6 * rng.normal(0, 1, n)}).to_csv(path, index=False)
    return path


def _issue(root: Path, *, issue_date: str, origin: str, target_dates, quantiles=(10, 50, 90),
           with_interval_model=True, origin_value=4.6e7) -> Path:
    """A published issue in the real shape: one origin, one origin_value, N horizons."""
    d = root / issue_date
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, td in enumerate(target_dates, start=1):
        r = {"target": "Revenues", "horizon": h, "origin_date": origin,
             "origin_value": origin_value, "target_date": td,
             "point_model": "LightGBM_L1", "target_transform": "ratio"}
        for q, v in zip(quantiles, (1.0e7, 4.0e7, 9.0e7)):
            r[f"p{q}"] = v
        if with_interval_model:
            r["interval_model"] = "GBQuantile"
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "forecast.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({
        "issue_date": issue_date,
        "data_sha_at_issue": "sha_at_issue_aaa",
        "git_sha_at_issue": "gitsha_bbb",
        "recipes": [{"target": "Revenues", "recipe_id": "rev-v1"}],
    }), encoding="utf-8")
    return d


@pytest.fixture
def scored(tmp_path):
    """One issue, five horizons, all of whose truth is inside the actuals."""
    data = _actuals(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _issue(root, issue_date="2026-08-17", origin="2024-06-28",
           target_dates=["2024-07-01", "2024-07-02", "2024-07-03",
                         "2024-07-04", "2024-07-05"])
    out = score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 5, f"fixture should fully score; got {out}"
    return pd.read_csv(tmp_path / "sc.csv")


# ── the field set the schema must carry ──────────────────────────────────────

#: Every field a scored row is required to identify, and why. Kept as a literal list rather
#: than derived from SCORECARD_COLUMNS -- deriving it would make this test assert that the
#: schema equals itself, which is exactly the weakness of the checks it replaces.
REQUIRED = {
    "issue_date": "when the forecast was published",
    "target": "which Treasury line",
    "horizon": "how many business days ahead",
    "origin_date": "the last date whose data informed it",
    "origin_value": "the level at that origin",
    "p10": "lower edge", "p50": "central estimate", "p90": "upper edge",
    "y_true": "the actual",
    "abs_error": "absolute error against p50",
    "inside_interval": "did the actual fall inside the band",
    "recipe_id": "the champion in force at issue",
    "point_model": "which model produced p50",
    "data_sha_at_issue": "which dataset it was made from",
    "scored_at_data_sha": "which dataset it was scored against",
}


@pytest.mark.parametrize("field", sorted(REQUIRED))
def test_the_schema_carries_every_required_field(field):
    assert field in SCORECARD_COLUMNS, (
        f"a scored row cannot identify {REQUIRED[field]!r} without {field!r}")


def test_a_scored_row_populates_every_required_field(scored):
    """Present in the schema is not the same as filled in."""
    for field in sorted(REQUIRED):
        assert field in scored.columns, field
        assert scored[field].notna().all(), (
            f"{field!r} is in the schema but blank on a scored row")


def test_the_written_columns_are_the_schema_in_order(scored):
    assert list(scored.columns) == list(SCORECARD_COLUMNS)


def test_the_committed_scorecard_header_matches_the_schema():
    """The tracked file must not drift from the constant that defines it.

    ``forecasts/scorecard.csv`` is tracked, so a schema change that did not regenerate it would
    leave the repository shipping a header contradicting its own definition.
    """
    if not LIVE_SCORECARD.exists():
        pytest.skip("no scorecard on disk")
    header = LIVE_SCORECARD.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(SCORECARD_COLUMNS)


# ── schema versioning ────────────────────────────────────────────────────────

def test_every_row_records_the_schema_version(scored):
    """Rows accumulate across issues over months, so each says which field set it has."""
    assert "schema_version" in SCORECARD_COLUMNS
    assert (scored["schema_version"] == SCORECARD_SCHEMA_VERSION).all()
    assert SCORECARD_SCHEMA_VERSION >= 1


def test_the_schema_version_is_documented_where_it_is_defined():
    """A version number nobody can decode is not versioning."""
    src = (BACKEND / "published_forecasts.py").read_text(encoding="utf-8")
    i = src.index("SCORECARD_SCHEMA_VERSION")
    preamble = src[max(0, i - 1500):i]
    assert "1 --" in preamble or "1 -" in preamble, (
        "the current version must say what it comprises, so a bump is decodable")


# ── origin: the field that was missing ───────────────────────────────────────

def test_origin_date_is_recorded_and_precedes_the_target(scored):
    for _, r in scored.iterrows():
        origin = pd.Timestamp(r["origin_date"])
        target = pd.Timestamp(r["target_date"])
        assert origin < target, f"origin {origin.date()} is not before target {target.date()}"


def test_the_origin_to_target_gap_is_the_recorded_horizon(scored):
    """The horizon has to be checkable from the row, which is the point of carrying origin."""
    for _, r in scored.iterrows():
        expected = pd.Timestamp(r["origin_date"]) + pd.offsets.BDay(int(r["horizon"]))
        assert pd.Timestamp(r["target_date"]) == expected, (
            f"h={r['horizon']}: origin {r['origin_date']} + {r['horizon']} business days is "
            f"{expected.date()}, but target_date is {r['target_date']}")


def test_issue_date_and_origin_date_are_distinct_fields(scored):
    """The two were conflated until the publish CLI was fixed. They mean different things:
    issue_date is wall-clock, origin_date is the data vintage."""
    assert (scored["issue_date"] == "2026-08-17").all()
    assert (scored["origin_date"] == "2024-06-28").all()
    assert (scored["issue_date"] != scored["origin_date"]).all()


def test_the_recorded_ruler_is_the_recorded_origin_value(scored):
    """persistence_pred is derived from origin_value, so the row can now be audited alone."""
    for _, r in scored.iterrows():
        assert r["persistence_pred"] == pytest.approx(r["origin_value"], abs=0.01)


# ── the interval fields ──────────────────────────────────────────────────────

def test_interval_nominal_is_read_from_the_published_quantiles():
    """0.80 for p10-p90, and a different pair gives a different level -- not a constant."""
    assert interval_nominal_of(pd.DataFrame(columns=["p10", "p50", "p90"])) == pytest.approx(0.80)
    assert interval_nominal_of(pd.DataFrame(columns=["p5", "p50", "p95"])) == pytest.approx(0.90)
    assert interval_nominal_of(pd.DataFrame(columns=["p25", "p75"])) == pytest.approx(0.50)


def test_interval_nominal_is_none_rather_than_a_guess_when_no_pair_exists():
    assert interval_nominal_of(pd.DataFrame(columns=["p50"])) is None
    assert interval_nominal_of(pd.DataFrame(columns=["y_true"])) is None


def test_a_scored_row_states_the_level_its_hit_flag_was_measured_against(scored):
    """``inside_interval`` without ``interval_nominal`` is a hit rate against an unstated
    target. Keeping the level only in the column names is the defect the E_QUANTILE audit
    already had to fix once."""
    assert scored["interval_nominal"].tolist() == pytest.approx([0.80] * len(scored))


def test_a_non_default_quantile_pair_is_refused_rather_than_mislabelled(tmp_path):
    """The record can describe any pair; the arithmetic cannot yet score one.

    ``interval_nominal_of`` reads a p5-p95 band as 0.90 correctly (asserted above), but
    ``score_one`` and the row builder name ``p50`` and ``p10``/``p90`` directly, so such an
    issue cannot produce a row at all. Unreachable today -- ``forward_forecast.QUANTILES`` is
    fixed at (0.10, 0.50, 0.90) -- and deliberately left as a refusal rather than generalised,
    because making the arithmetic quantile-agnostic changes how every score is computed and
    belongs in a session that scopes it. What must NOT happen is the middle case: a row built
    against the wrong columns, or one labelled 80% when its band advertises 90%.
    """
    data = _actuals(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _issue(root, issue_date="2026-08-17", origin="2024-06-28",
           target_dates=["2024-07-01", "2024-07-02"], quantiles=(5, 50, 95))

    with pytest.raises(UnsupportedIntervalShape) as exc:
        score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")

    msg = str(exc.value)
    assert "p5" in msg and "p95" in msg, f"the refusal must name what it found: {msg}"
    assert "p10" in msg, f"...and what it supports: {msg}"
    assert not (tmp_path / "sc.csv").exists(), (
        "a refused issue must not leave a partially written scorecard")


def test_the_supported_quantile_shape_is_the_one_the_publisher_emits(tmp_path):
    """The refusal above is only acceptable while nothing real trips it."""
    import forward_forecast as ff

    emitted = {f"p{int(round(q * 100))}" for q in ff.QUANTILES}
    assert set(SUPPORTED_QUANTILE_COLUMNS) == emitted, (
        f"the publisher emits {sorted(emitted)} but the scorer supports "
        f"{sorted(SUPPORTED_QUANTILE_COLUMNS)} -- one of them moved, and real issues will now "
        f"be refused")


def test_the_interval_model_is_recorded_so_a_band_miss_can_be_attributed(scored):
    """Measured on the sealed window at the same 80% nominal, GBQuantile covered 78.0% of
    revenues outcomes against ResidualRF's 69.8% -- so which model made the band matters."""
    assert "interval_model" in SCORECARD_COLUMNS
    assert (scored["interval_model"] == "GBQuantile").all()
    assert (scored["point_model"] == "LightGBM_L1").all()


def test_an_issue_predating_the_interval_model_column_scores_with_it_blank(tmp_path):
    """An older artifact must score, and must not have a model name invented for it."""
    data = _actuals(tmp_path / "actuals.csv")
    root = tmp_path / "published"
    _issue(root, issue_date="2026-08-17", origin="2024-06-28",
           target_dates=["2024-07-01", "2024-07-02"], with_interval_model=False)
    out = score_published(data, published_root=root, scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 2, "a pre-schema issue must still score"

    sc = pd.read_csv(tmp_path / "sc.csv")
    blank = sc["interval_model"].isna() | (sc["interval_model"].astype(str) == "")
    assert blank.all(), (
        f"expected a blank interval_model, got {sc['interval_model'].tolist()} -- a guessed "
        f"model name would be a claim nobody made")
    # ...and the fields that DO exist are unaffected by the absence.
    assert sc["origin_value"].notna().all()
    assert sc["inside_interval"].notna().all()


# ── the three hashes answer three questions ──────────────────────────────────

def test_the_issue_and_scoring_hashes_are_both_recorded_and_can_differ(scored):
    """Actuals are normally revised or extended between issue and scoring, so these two
    differing is the ordinary case -- and the pair is what makes a reconciliation possible."""
    assert (scored["data_sha_at_issue"] == "sha_at_issue_aaa").all()
    assert (scored["git_sha_at_issue"] == "gitsha_bbb").all()
    assert scored["scored_at_data_sha"].notna().all()
    assert (scored["scored_at_data_sha"] != scored["data_sha_at_issue"]).all(), (
        "the fixture scores against different data than it was issued from; if these are "
        "equal the scoring hash is not being read from the scoring dataset")


# ── absence is recorded as absence ───────────────────────────────────────────

def test_optional_fields_read_absence_as_blank_never_as_a_substitute():
    """An issue predating a column must leave that cell empty, not plausible.

    These are the readers behind ``origin_date`` and ``origin_value``. A guessed value in a
    permanent record is a claim nobody made, and it would be indistinguishable from a real one.
    """
    from published_forecasts import _opt_date, _opt_float

    for absent in (None, np.nan, "", "not-a-date"):
        assert _opt_date(absent) is None, absent
    assert _opt_date("2025-08-06") == "2025-08-06"
    assert _opt_date(pd.Timestamp("2025-08-06")) == "2025-08-06"

    for absent in (None, np.nan, "", "abc", float("inf")):
        assert _opt_float(absent) is None, absent
    assert _opt_float("1.5") == pytest.approx(1.5)
    assert _opt_float(1.5) == pytest.approx(1.5)


def test_the_summary_reports_the_level_its_hit_rate_was_measured_against(scored, tmp_path):
    """The same defect one layer up: a hit rate printed beside a hardcoded level.

    ``summarize_scorecard`` used ``NOMINAL_COVERAGE`` (0.80) regardless of what the rows said,
    which is exactly what ``interval_nominal`` exists to stop -- it just moved the assumption
    into the summary a reader actually looks at.
    """
    from published_forecasts import summarize_scorecard

    s = summarize_scorecard(scored)["Revenues"]
    assert s["nominal_coverage"] == pytest.approx(0.80)
    assert 0.0 <= s["interval_hit_rate"] <= 1.0

    # A file mixing two band levels must say so rather than pick one.
    mixed = pd.concat([scored, scored.assign(interval_nominal=0.90)], ignore_index=True)
    assert summarize_scorecard(mixed)["Revenues"]["nominal_coverage"] == [0.80, 0.90]
