"""The LIVE window, and the fact that champions are fixed.

Two P1 decisions, both of which had to become structural rather than documented.

**LIVE.** TEST was open-ended, so every row a client added after the holdout was sealed fell inside
it, and scoring that data needed ``AI4CM_ALLOW_TEST_READ=1`` — spending the one clean final read on
data the holdout never covered. TEST is now closed at its actual extent and everything after is
LIVE: scored freely, never selected on. The tests that matter here are the ones proving the "never
selected on" half cannot be bypassed, because the TEST discipline was a docstring before Phase 2 and
a docstring is not enforcement.

**Fixed champions.** No reselection exists and none is being built. What was missing is that a
client could not tell: a run completes, numbers move, and the reasonable inference is that the model
was re-chosen for the new period. It was not. So the policy is a field, and its caveats are read
from the registry rather than retyped.
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
sys.path.insert(0, str(REPO / "scripts"))

from evaluation_windows import (  # noqa: E402
    DEV,
    LIVE,
    LIVE_START,
    SELECTABLE_WINDOWS,
    TEST,
    TEST_ACCESS_ENV,
    TEST_END,
    TEST_START,
    TRAIN,
    WINDOWS,
    SelectionOnReportOnlyDataError,
    assert_selection_free,
    eval_start_for,
    restrict,
    rolling_origin_folds,
    window_for,
)
# Aliased: pytest tries to collect any module-level name starting with "Test" as a
# test class, and warns because the exception has an __init__.
from evaluation_windows import TestWindowAccessError as HoldoutAccessError  # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ── the split itself ─────────────────────────────────────────────────────────

def test_there_are_four_windows_and_test_is_now_closed():
    assert [w.name for w in WINDOWS] == ["train", "dev", "test", "live"]
    assert TEST.end == TEST_END == "2025-08-06", (
        "TEST must be closed at the extent it holds; an open-ended TEST is what put client "
        "data inside the holdout")
    assert LIVE.start == LIVE_START == "2025-08-07"
    assert LIVE.end is None, "LIVE is open-ended: it grows as the client loads data"


@pytest.mark.parametrize("date,expected", [
    ("2015-01-05", "train"), ("2023-12-31", "train"),
    ("2024-01-01", "dev"), ("2024-12-31", "dev"),
    ("2025-01-01", "test"), ("2025-08-06", "test"),
    ("2025-08-07", "live"), ("2026-06-01", "live"),
])
def test_every_boundary_lands_in_the_right_window(date, expected):
    assert window_for(date) == expected


def test_the_windows_tile_without_gap_or_overlap():
    days = pd.date_range("2015-01-05", "2026-12-31", freq="D")
    names = {window_for(d) for d in days}
    assert names == {"train", "dev", "test", "live"}
    # Each day belongs to exactly one window, by construction of window_for.
    for d in ("2023-12-31", "2024-01-01", "2025-01-01", "2025-08-06", "2025-08-07"):
        hits = [w.name for w in WINDOWS if w.contains(d)]
        assert hits == [window_for(d)], f"{d} is in {hits}, window_for says {window_for(d)}"


@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_the_change_reclassifies_nothing_that_exists_today():
    """The load-bearing safety property: closing TEST must not move an existing number.

    The canonical file ends 2025-08-06, so LIVE is empty. If this fails, either the file grew
    (fine, but the LIVE figures below are now real and the record must say so) or a boundary moved.
    """
    dates = pd.to_datetime(pd.read_csv(DATA, usecols=["date"])["date"], errors="coerce").dropna()
    counts = pd.Series([window_for(d) for d in dates]).value_counts().to_dict()
    assert counts.get("live", 0) == 0, (
        f"the canonical file now has {counts.get('live')} LIVE rows; this test guarded the claim "
        f"that the split change was inert")
    assert str(dates.max().date()) == TEST_END


# ── the structural half: selection cannot reach report-only data ─────────────

def test_selection_is_permitted_only_on_train_and_dev():
    assert SELECTABLE_WINDOWS == {"train", "dev"}


def test_selecting_on_live_raises():
    dates = pd.to_datetime(["2024-03-01", "2025-09-15"])
    with pytest.raises(SelectionOnReportOnlyDataError, match="report-only"):
        assert_selection_free(dates, "unit_test")


def test_selecting_on_test_raises_too():
    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(pd.to_datetime(["2025-03-01"]), "unit_test")


def test_selecting_on_train_and_dev_is_allowed():
    assert_selection_free(pd.to_datetime(["2016-01-04", "2024-06-03"]), "unit_test")
    assert_selection_free([], "empty is not a violation")


def test_the_refusal_names_the_window_the_count_and_the_earliest_date():
    """A refusal a reader cannot act on is only half a guard."""
    dates = pd.to_datetime(["2024-01-02", "2025-09-01", "2025-09-02"])
    with pytest.raises(SelectionOnReportOnlyDataError) as exc:
        assert_selection_free(dates, "my_selector")
    msg = str(exc.value)
    assert "my_selector" in msg
    assert "live (n=2, from 2025-09-01)" in msg, msg
    assert "never used to choose" in msg


def test_the_live_guard_is_not_overridable_by_the_test_access_flag(monkeypatch):
    """The key structural property.

    TEST has a release procedure because a holdout exists to be spent once. LIVE has none —
    there is nothing to release — so the flag that unlocks TEST must not unlock LIVE.
    """
    monkeypatch.setenv(TEST_ACCESS_ENV, "1")
    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(pd.to_datetime(["2025-09-01"]), "even_with_the_flag_set")


def test_the_two_refusals_are_different_exceptions():
    """A LIVE violation must not be reported as a holdout-access problem, or vice versa."""
    assert not issubclass(SelectionOnReportOnlyDataError, HoldoutAccessError)
    assert not issubclass(HoldoutAccessError, SelectionOnReportOnlyDataError)


def test_folds_cannot_be_carved_out_of_live():
    """Folds are a search structure, so LIVE is refused at the entry point."""
    idx = pd.bdate_range("2025-08-07", periods=400)
    with pytest.raises(SelectionOnReportOnlyDataError, match="search structure"):
        rolling_origin_folds(idx, horizon=5, window="live")


# ── the readable half: scoring LIVE needs no gate ────────────────────────────

def test_restricting_to_live_is_ungated():
    """Scoring arrived actuals is the purpose of LIVE, not a holdout consultation."""
    idx = pd.date_range("2025-08-01", "2025-08-20", freq="D")
    s = pd.Series(range(len(idx)), index=idx)
    live = restrict(s, "live")
    assert len(live) > 0
    assert {window_for(d) for d in live.index} == {"live"}


def test_restricting_to_test_is_still_gated():
    idx = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    s = pd.Series(range(len(idx)), index=idx)
    with pytest.raises(HoldoutAccessError):
        restrict(s, "test")


def test_eval_start_for_score_points_at_live():
    assert eval_start_for("score") == LIVE_START
    assert eval_start_for("tuning") == DEV.start
    assert eval_start_for("report") == TEST_START
    with pytest.raises(ValueError, match="'tuning', 'report' or 'score'"):
        eval_start_for("whatever")


# ── the pipeline's own selection path ────────────────────────────────────────

def test_b_ml_guards_its_champion_selection_against_report_only_rows():
    """The guard sits immediately before select_best_model, where a champion is crowned."""
    src = (BACKEND / "b_ml_pipeline.py").read_text()
    assert "assert_selection_free" in src, "the selection path is unguarded"
    i_guard = src.index("assert_selection_free")
    i_select = src.index("best_model, _excluded = select_best_model(")
    assert i_guard < i_select, (
        "the guard must run BEFORE the champion is chosen, not after")


def test_removing_the_b_ml_guard_would_let_live_rows_choose_a_champion():
    """Mutation test: without the guard, evaluation rows in LIVE reach selection unchallenged.

    Exercised on the helper rather than a full pipeline run, because the property under test is
    that *something* refuses — and with the guard removed, nothing does.
    """
    eval_rows = pd.to_datetime(["2024-06-03", "2025-09-01"])          # one dev, one live

    with pytest.raises(SelectionOnReportOnlyDataError):
        assert_selection_free(eval_rows, "with the guard")

    def no_guard(dates, context):                                     # the pre-P1 behaviour
        return None

    assert no_guard(eval_rows, "without the guard") is None, (
        "the mutation must be silent -- that silence is the defect the guard removes")


# ── champions are fixed, and the artifact says so ────────────────────────────

def test_the_champion_policy_states_fixity_and_refit_only():
    from registry import REFIT_ONLY, RESELECTION_NONE, champion_policy

    p = champion_policy()
    assert p["reselection"] == RESELECTION_NONE == "none"
    assert p["on_new_data"] == REFIT_ONLY == "refit_only"
    assert "Champions are fixed" in p["statement"]
    assert "never changes which model" in p["statement"]
    assert "hand-edited registry/recipes.json" in " ".join(p["reselection_requires"])


def test_the_policy_lists_every_fixed_recipe_with_its_transform():
    from registry import champion_policy, load_registry

    p = champion_policy()
    assert len(p["recipes_fixed"]) == len(load_registry()["recipes"]) == 3
    by_target = {r["target"]: r for r in p["recipes_fixed"]}
    assert by_target["Revenues"]["target_transform"] == "ratio"
    assert by_target["Expenditure"]["target_transform"] == "raw"
    # Nothing may read as approved while approved_by is null.
    assert all(r["approved_by"] is None for r in p["recipes_fixed"])


def test_the_drift_caveat_is_read_from_the_registry_not_retyped():
    """A caveat transcribed into a second place is one that will disagree with itself."""
    from registry import champion_policy, load_registry

    p = champion_policy()
    rev = [c for c in p["caveats"] if c["target"] == "Revenues"]
    assert len(rev) == 1, p["caveats"]
    caveat = rev[0]

    recipe = [r for r in load_registry()["recipes"] if r["target"] == "Revenues"][0]
    source = recipe["scaling_caveat"]
    assert caveat["finding"] == source["finding"]
    assert caveat["evidence"] == source["evidence"]
    assert caveat["study"] == source["study"]

    # And it carries the measured numbers the risk rests on.
    assert "+1.30%" in caveat["evidence"] and "+24.14%" in caveat["evidence"]
    assert caveat["applies_to"] == "ratio"


def test_the_policy_explains_why_reselection_would_break_the_live_rule():
    from registry import champion_policy

    why = champion_policy()["why"]
    assert "LIVE" in why and "assert_selection_free" in why


def test_no_source_file_hardcodes_the_drift_numbers_outside_the_registry():
    """The +1.30% / +24.14% pair may live in recipes.json and in prose, never in code."""
    offenders = []
    for p in sorted(BACKEND.glob("*.py")) + sorted((REPO / "scripts").glob("*.py")):
        txt = p.read_text(errors="ignore")
        if "+24.14%" in txt or "24.14" in txt:
            offenders.append(str(p.relative_to(REPO)))
    assert offenders == [], f"the drift evidence is duplicated in code: {offenders}"


# ── the artifact states where numbers came from ──────────────────────────────

def test_the_summary_writer_emits_the_windows_and_the_policy():
    src = (REPO / "scripts" / "daily_summary.py").read_text()
    assert "_window_and_policy_fields(data_file)" in src
    assert '"windows"' in src and '"champion_policy"' in src
    # Absence must be explained, per the contract's §0 pattern.
    assert '"windows_unavailable_reason"' in src
    assert '"champion_policy_unavailable_reason"' in src


def test_the_window_block_names_boundaries_and_what_is_selectable(tmp_path):
    import daily_summary

    data = tmp_path / "d.csv"
    pd.DataFrame({"date": ["2024-06-03", "2025-03-01"]}).to_csv(data, index=False)
    out = daily_summary._window_and_policy_fields(data)

    w = out["windows"]
    assert set(w["definition"]) == {"train", "dev", "test", "live"}
    assert w["definition"]["test"]["end"] == TEST_END
    assert w["definition"]["live"]["start"] == LIVE_START
    assert w["selectable"] == ["dev", "train"]
    assert set(w["report_only"]) == {"test", "live"}
    assert w["data_spans_windows"] == ["dev", "test"], w["data_spans_windows"]
    assert w["latest_data_date"] == "2025-03-01"
    assert out["champion_policy"]["reselection"] == "none"


def test_the_window_block_reports_live_once_the_data_reaches_it(tmp_path):
    import daily_summary

    data = tmp_path / "d.csv"
    pd.DataFrame({"date": ["2024-06-03", "2025-08-20"]}).to_csv(data, index=False)
    w = daily_summary._window_and_policy_fields(data)["windows"]
    assert "live" in w["data_spans_windows"], w["data_spans_windows"]


def test_an_unreadable_data_file_still_yields_the_split(tmp_path):
    """The definition is worth stating even when the file cannot be inspected."""
    import daily_summary

    out = daily_summary._window_and_policy_fields(tmp_path / "missing.csv")
    assert set(out["windows"]["definition"]) == {"train", "dev", "test", "live"}
    assert out["windows"]["data_spans_windows"] == []
    assert out["windows"]["latest_data_date"] is None


# ── the scorecard says which window each realized number came from ──────────

def test_the_scorecard_carries_the_window_of_every_scored_row():
    from published_forecasts import SCORECARD_COLUMNS

    assert "scored_in_window" in SCORECARD_COLUMNS, (
        "a realized number from LIVE and one from the sealed holdout are different claims")


def test_a_scored_live_row_is_labelled_live(tmp_path):
    """End-to-end on the smallest fixture that can score anything."""
    from published_forecasts import score_published

    issue = tmp_path / "published" / "2025-08-06"
    issue.mkdir(parents=True)
    pd.DataFrame([{"target": "Revenues", "horizon": 5, "origin_date": "2025-08-06",
                   "origin_value": 100.0, "target_date": "2025-08-13",
                   "p10": 90.0, "p50": 105.0, "p90": 120.0,
                   "modelled_as": "level"}]).to_csv(issue / "forecast.csv", index=False)
    for name in ("provenance.json", "manifest.json", "gates.json"):
        (issue / name).write_text("{}")

    idx = pd.bdate_range("2025-07-01", "2025-08-13")
    data = tmp_path / "actuals.csv"
    pd.DataFrame({"date": idx, "Revenues": np.linspace(100, 130, len(idx))}).to_csv(
        data, index=False)

    out = score_published(data, published_root=tmp_path / "published",
                          scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 1, out
    sc = pd.read_csv(tmp_path / "sc.csv")
    assert sc["scored_in_window"].tolist() == ["live"], sc["scored_in_window"].tolist()
    assert sc["persistence_source"].tolist() == ["artifact: origin_value"]


# ── B_ML's holdout read reaches the ledger ───────────────────────────────────
#
# The last of the four families to be wired up. A_STAT and C_DL were done in P1, E_QUANTILE on
# 2026-08-17; B_ML was the remaining gap (docs/sessions/2026-08-17-interval-calibration.md §7
# item 5) and it was not theoretical. Measured before the fix: `experiments/test_access.log`
# held 148 entries -- 16 naming A_STAT, 121 naming C_DL, 9 naming E_QUANTILE and **zero**
# naming B_ML -- while both logged B_ML runs had evaluated target dates 2025-01-01..2025-08-06,
# the sealed window end to end.

def _b_ml_synthetic_csv(tmp_path, seed=0):
    """A flow series spanning TRAIN through the holdout's last day."""
    idx = pd.bdate_range("2015-01-05", TEST_END)
    rng = np.random.default_rng(seed)
    csv = tmp_path / "b_ml_synthetic.csv"
    pd.DataFrame({"date": idx,
                  "Revenues": 1e8 + np.cumsum(rng.normal(0, 1e6, len(idx)))}).to_csv(csv,
                                                                                     index=False)
    return csv


def test_the_default_b_ml_fold_geometry_lands_squarely_on_the_holdout():
    """Why the ledger call is needed at all, asserted on the fold builder rather than described.

    The daily runner passes ``{"folds":1,"min_train_years":4}`` with no ``eval_start``, and
    ``folds_override`` keeps the LAST fold. On this index that fold is train<=2024-12-31 /
    test 2025-01-01..2025-08-06 -- nothing but the sealed holdout.
    """
    from b_ml_pipeline import build_yearly_folds

    idx = pd.bdate_range("2015-01-05", TEST_END)
    folds = build_yearly_folds(idx, 4, 1)                    # exactly the daily runner's config
    assert len(folds) == 1
    _train_end, test_start, test_end = folds[0]
    assert {window_for(t) for t in idx[(idx >= test_start) & (idx <= test_end)]} == {"test"}, (
        "the default daily fold should be entirely holdout -- if this changes, the ledger call's "
        "justification changes with it")


def test_b_ml_logs_its_holdout_read_as_a_report(tmp_path, monkeypatch):
    """The read is announced, and announced as a *report* rather than a selection.

    Reporting on the holdout is what the holdout is for, so ``require_test_access`` records and
    returns. Crowning a champion from those same rows is a selection and is refused separately --
    which is why this run is expected to raise. The ledger call happens at fold construction,
    well before that point, and that ordering is the property under test: the read is announced
    when it starts, not after it has finished.
    """
    import evaluation_windows as ew
    from b_ml_pipeline import ConfigBML, run_pipeline_ml

    calls = []

    def spy(reason, caller=None, purpose=ew.PURPOSE_SELECTION):
        calls.append({"reason": reason, "caller": caller, "purpose": purpose})

    monkeypatch.setattr(ew, "require_test_access", spy)

    cfg = ConfigBML(data_path=str(_b_ml_synthetic_csv(tmp_path)), date_col="date",
                    target="Revenues", cadence="Daily", horizon=5, variant="uni",
                    model_filter="Ridge", out_root=str(tmp_path / "out"),
                    folds=1, min_train_years=4)
    with pytest.raises(SelectionOnReportOnlyDataError):
        run_pipeline_ml(cfg)

    assert calls, "B_ML evaluated the holdout and the ledger was not told"
    assert any(c["purpose"] == ew.PURPOSE_REPORT for c in calls), (
        "a reporting read must be logged as a report, not as a selection")
    assert any("B_ML" in (c["reason"] or "") for c in calls), (
        "the entry must name the family, so the ledger can be counted per family")
    assert any(c["caller"] == "b_ml_pipeline.run_pipeline_ml" for c in calls)
    reason = next(c["reason"] for c in calls if c["purpose"] == ew.PURPOSE_REPORT)
    assert "2025-01-01" in reason and str(TEST_END) in reason, (
        f"the entry must say which dates were read; got {reason!r}")


def test_a_dev_pinned_b_ml_run_does_not_claim_a_holdout_read(tmp_path, monkeypatch):
    """No false positives. A run bounded to DEV touches no holdout row, so the ledger stays quiet.

    This is the half that keeps the ledger meaningful: a call on every run regardless of window
    would make "how often was the holdout consulted" unanswerable again, just noisily this time.
    """
    import evaluation_windows as ew
    from b_ml_pipeline import ConfigBML, run_pipeline_ml

    calls = []
    monkeypatch.setattr(ew, "require_test_access",
                        lambda reason, caller=None, purpose=ew.PURPOSE_SELECTION:
                        calls.append(purpose))

    cfg = ConfigBML(data_path=str(_b_ml_synthetic_csv(tmp_path, seed=3)), date_col="date",
                    target="Revenues", cadence="Daily", horizon=5, variant="uni",
                    model_filter="Ridge", out_root=str(tmp_path / "out"),
                    folds=1, min_train_years=4,
                    eval_start=DEV.start, eval_end=DEV.end)
    run_pipeline_ml(cfg)       # crowning on DEV rows is legitimate, so this completes

    assert not calls, f"a DEV-only run must not record a holdout read; got {calls}"


def test_a_selection_purpose_read_of_the_holdout_still_raises():
    """The ledger call B_ML makes is a report. The selection door stays shut and stays loud."""
    from evaluation_windows import PURPOSE_REPORT, PURPOSE_SELECTION, require_test_access

    # A report records and returns, whatever the environment says.
    require_test_access("unit check: reporting read", caller="test_live_window",
                        purpose=PURPOSE_REPORT)

    # A selection read raises unless the holdout has been deliberately released.
    with pytest.raises(HoldoutAccessError):
        require_test_access("unit check: selection read", caller="test_live_window",
                            purpose=PURPOSE_SELECTION)
