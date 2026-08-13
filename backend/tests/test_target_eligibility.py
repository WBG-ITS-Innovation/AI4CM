"""Target eligibility, and the family capability matrix.

Two things are guarded here, both of the same shape: a verdict is worthless to a consumer without
the reason behind it, and a reason recorded only in prose goes stale without anything failing.

`targets_available` used to read `nrows=1` and return all 41 columns, so a length check was
impossible by construction. It now measures, and **every ineligibility carries both a
machine-readable code and prose a consumer may quote verbatim** — because a consumer told only "no"
has to invent an explanation, and an invented explanation shown to a treasury is worse than a blank.

The real canonical file is fully dense (every column has 3,867 usable rows against a 1,139
requirement), so the length check cannot be exercised against it. That is exactly why the rejection
cases below are built synthetically: a check that never fires looks like it works.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forecast_modes import (  # noqa: E402
    INELIGIBLE_ALL_NULL,
    INELIGIBLE_INSUFFICIENT_HISTORY,
    INELIGIBLE_NOT_NUMERIC,
    INELIGIBLE_UNUSABLE_DATE_INDEX,
    NON_TARGET_COLUMNS,
    VALIDATED_HORIZON,
    date_index_status,
    min_history_rows,
    target_eligibility,
    targets_available,
)

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ── the threshold is derived, not chosen ─────────────────────────────────────

def test_the_threshold_comes_from_the_evaluation_windows():
    """1008 + horizon + 126. If the fold sizing moves, this must move with it."""
    from evaluation_windows import DEFAULT_EVAL_BLOCK, DEFAULT_MIN_TRAIN

    assert min_history_rows(5) == DEFAULT_MIN_TRAIN + 5 + DEFAULT_EVAL_BLOCK == 1139
    assert min_history_rows(1) == DEFAULT_MIN_TRAIN + 1 + DEFAULT_EVAL_BLOCK
    assert min_history_rows(20) > min_history_rows(5), "a longer horizon needs more history"


# ── synthetic rejections: the cases the real file cannot exercise ────────────

def _frame(tmp_path: Path, cols: dict, periods: int = 300,
           dates=None, name: str = "s.csv") -> Path:
    idx = pd.bdate_range("2020-01-01", periods=periods) if dates is None else dates
    df = pd.DataFrame({"date": idx, "is_weekend": 0, "is_holiday": 0, **cols})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_a_short_numeric_column_is_ineligible_with_a_quotable_reason(tmp_path):
    p = _frame(tmp_path, {"ShortSeries": range(300)})
    e = target_eligibility(p)["ShortSeries"]
    assert e.eligible is False
    assert e.code == INELIGIBLE_INSUFFICIENT_HISTORY
    assert e.n_usable == 300 and e.n_required == 1139
    # The prose must carry the numbers, so a consumer never has to assemble them.
    assert "300 usable observations" in e.reason and "1,139 are needed" in e.reason
    assert "not a forecast" in e.reason


def test_a_non_numeric_column_is_ineligible(tmp_path):
    p = _frame(tmp_path, {"TextColumn": ["a"] * 300})
    e = target_eligibility(p)["TextColumn"]
    assert e.eligible is False
    assert e.code == INELIGIBLE_NOT_NUMERIC
    assert "cannot be forecast as a series" in e.reason


def test_an_all_null_column_is_ineligible_and_distinct_from_not_numeric(tmp_path):
    p = _frame(tmp_path, {"EmptyColumn": [None] * 300})
    e = target_eligibility(p)["EmptyColumn"]
    assert e.eligible is False
    assert e.code == INELIGIBLE_ALL_NULL, "an empty column is not the same defect as a text column"
    assert e.n_usable == 0


def test_a_numeric_column_with_gaps_is_judged_on_its_usable_rows(tmp_path):
    """Sparse-but-numeric is a length question, not a type question."""
    vals = [1.0 if i % 2 == 0 else None for i in range(1400)]
    p = _frame(tmp_path, {"Sparse": vals}, periods=1400)
    e = target_eligibility(p)["Sparse"]
    assert e.code == INELIGIBLE_INSUFFICIENT_HISTORY, e.as_dict()
    assert e.n_usable == 700, "nulls must not be counted as history"


def test_a_long_enough_numeric_column_is_eligible_and_carries_no_reason(tmp_path):
    p = _frame(tmp_path, {"Good": range(1200)}, periods=1200)
    e = target_eligibility(p)["Good"]
    assert e.eligible is True
    assert e.code is None and e.reason is None, "an eligible target must not carry a rejection"
    assert e.first_date == "2020-01-01"


def test_the_horizon_moves_the_threshold(tmp_path):
    """A column can be eligible at h=1 and not at h=20."""
    p = _frame(tmp_path, {"Borderline": range(1140)}, periods=1140)
    assert target_eligibility(p, horizon=1)["Borderline"].eligible is True
    e20 = target_eligibility(p, horizon=20)["Borderline"]
    assert e20.eligible is False and e20.code == INELIGIBLE_INSUFFICIENT_HISTORY


# ── the date index is a FILE-level property ──────────────────────────────────

def test_a_clean_date_index_passes():
    st = date_index_status(DATA)
    assert st["ok"] is True and st["reason"] is None
    assert st["n"] == st["n_unique"] == 3867
    assert st["n_unparseable"] == 0 and st["monotonic"] is True


def test_unparseable_dates_make_every_column_ineligible(tmp_path):
    idx = list(pd.bdate_range("2020-01-01", periods=1200).astype(str))
    idx[5] = "not a date"
    p = _frame(tmp_path, {"Good": range(1200)}, dates=idx, periods=1200)

    st = date_index_status(p)
    assert st["ok"] is False and "could not be parsed" in st["reason"]

    e = target_eligibility(p)["Good"]
    assert e.eligible is False
    assert e.code == INELIGIBLE_UNUSABLE_DATE_INDEX
    assert "no column in it can be forecast" in e.reason, (
        "a file-level failure must say it is file-level, not blame the column")


def test_duplicate_dates_make_every_column_ineligible(tmp_path):
    idx = list(pd.bdate_range("2020-01-01", periods=1200).astype(str))
    idx[7] = idx[6]
    p = _frame(tmp_path, {"Good": range(1200)}, dates=idx, periods=1200)

    st = date_index_status(p)
    assert st["ok"] is False and "duplicate date" in st["reason"]
    assert "ambiguous" in st["reason"], (
        "the reason must explain WHY duplicates matter: a horizon in index positions")
    assert target_eligibility(p)["Good"].code == INELIGIBLE_UNUSABLE_DATE_INDEX


def test_an_unsorted_index_is_reported_but_is_not_a_rejection(tmp_path):
    """Sortable is the requirement; already-sorted is not. Reversed input is fine."""
    idx = list(pd.bdate_range("2020-01-01", periods=1200).astype(str))[::-1]
    p = _frame(tmp_path, {"Good": range(1200)}, dates=idx, periods=1200)
    st = date_index_status(p)
    assert st["ok"] is True, st
    assert st["monotonic"] is False, "the state must still be reported"
    assert target_eligibility(p)["Good"].eligible is True


# ── the real file ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_every_column_in_the_canonical_file_is_eligible_today():
    """Documents the measured state: the file is dense, so nothing is rejected.

    If this ever fails, a column has become short or non-numeric and the Forecast page will stop
    offering it -- which is the intended behaviour, but it should be a deliberate discovery.
    """
    e = target_eligibility(DATA)
    assert len(e) == 41, f"the candidate set changed: {len(e)}"
    ineligible = {t: v.as_dict() for t, v in e.items() if not v.eligible}
    assert ineligible == {}, ineligible
    assert min(v.n_usable for v in e.values()) == 3867, (
        "the file is no longer fully dense; the length check now has real work to do")


@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_targets_available_returns_eligible_only_by_default():
    assert len(targets_available(DATA)) == 41
    assert len(targets_available(DATA, include_ineligible=True)) == 41
    assert not (NON_TARGET_COLUMNS & set(targets_available(DATA)))


def test_targets_available_can_include_the_rejects_for_a_page_to_grey_out(tmp_path):
    p = _frame(tmp_path, {"Good": range(1200), "Short": [1.0] * 300 + [None] * 900},
               periods=1200)
    assert targets_available(p) == ["Good"]
    assert sorted(targets_available(p, include_ineligible=True)) == ["Good", "Short"]


def test_the_calendar_flags_are_never_offered_as_targets(tmp_path):
    p = _frame(tmp_path, {"Good": range(1200)}, periods=1200)
    assert set(target_eligibility(p)) == {"Good"}
    assert "is_weekend" not in target_eligibility(p)
    assert "date" not in target_eligibility(p)


def test_eligibility_is_serialisable_for_an_artifact(tmp_path):
    import json
    p = _frame(tmp_path, {"Short": range(300)})
    payload = {t: v.as_dict() for t, v in target_eligibility(p).items()}
    assert json.loads(json.dumps(payload))["Short"]["code"] == INELIGIBLE_INSUFFICIENT_HISTORY


# ── family capability (the CHANGELOG correction) ──────────────────────────────

def test_e_quantile_really_does_have_a_stock_path():
    """The CHANGELOG said it had none. Re-measured here so the correction cannot silently rot.

    Checks the two structural markers of delta modelling rather than re-running the family: the
    ``y_lag_0`` feature, which is only added for a stock target, and a target whose magnitude is
    the change rather than the level.
    """
    import numpy as np

    from e_quantile_daily_pipeline import Config, _build_features, is_stock, to_business_index

    if not DATA.exists():
        pytest.skip("canonical data not present")
    assert is_stock("State budget balance") is True

    raw = pd.read_csv(DATA)
    raw["date"] = pd.to_datetime(raw["date"])
    cfg = Config(target="State budget balance", cadence="Daily", horizon=5, data_path=str(DATA))
    df = to_business_index(raw.set_index("date").sort_index(), "State budget balance")
    X, y_t, _od, ov = _build_features(df, cfg)

    assert "y_lag_0" in X.columns, "y_lag_0 is the delta-modelling marker and is stock-only"
    assert np.mean(np.abs(y_t)) < np.mean(np.abs(ov)) / 2, (
        "the target should be a CHANGE, an order of magnitude below the level")


def test_the_capability_matrix_says_e_quantile_supports_stock_but_publishes_nothing():
    from family_capabilities import family_supports_target

    r = family_supports_target("E_QUANTILE", "State budget balance")
    assert r["supported"] is True
    assert r["publishable"] is False
    assert r["code"] == "supported_but_not_published"
    assert "Supported is not the same as approved" in r["reason"]
    assert r["stock_method"] == "delta"


def test_b_ml_is_the_only_family_that_publishes_the_stock_target():
    from family_capabilities import FAMILY_CAPABILITIES, family_supports_target

    publishing = [f for f in FAMILY_CAPABILITIES
                  if family_supports_target(f, "State budget balance")["publishable"]]
    assert publishing == ["B_ML"], publishing


def test_a_stat_forecasts_the_level_not_the_delta():
    """A different method, recorded as such -- two stock forecasts are not comparable otherwise."""
    from family_capabilities import STOCK_DELTA, STOCK_LEVEL, FAMILY_CAPABILITIES

    assert FAMILY_CAPABILITIES["A_STAT"].stock_method == STOCK_LEVEL
    for fam in ("B_ML", "C_DL", "E_QUANTILE"):
        assert FAMILY_CAPABILITIES[fam].stock_method == STOCK_DELTA


def test_an_unknown_family_is_refused_by_name():
    from family_capabilities import family_supports_target

    r = family_supports_target("F_MAGIC", "Revenues")
    assert r["supported"] is False and r["code"] == "unknown_family"
    assert "not a model family" in r["reason"]


def test_the_is_stock_implementations_now_agree():
    """Was `..._disagree_and_it_is_recorded`. The divergence is closed, so this inverts.

    There were SEVEN copies of this question, not the four family pipelines alone: also
    `ensemble_postprocess`, the legacy `a_stat_models_pipeline`, and a `TARGET_STOCK` set in
    `make_weekly_from_daily_stat`. `run_a_stat` was the divergent one -- it alone treated "net"
    and "stock" as stock and "t0" as a flow, so a column named `t0` would have been modelled as a
    level there and as a delta everywhere else.

    All seven now import `target_kinds.is_stock`. This test stays as the regression guard: it
    probes the four family entry points independently, so a re-introduced local copy shows up
    here rather than being discovered by a client.
    """
    from family_capabilities import stock_alias_divergence

    d = stock_alias_divergence()
    assert d["agree"] is True, f"a divergence is back: {d['disputed']}"
    assert d["disputed"] == {}
    assert d["resolved"] is True
    assert d["one_definition"] == "backend/target_kinds.py"


def test_all_four_families_share_one_is_stock_object():
    """Not merely equal behaviour -- the same function, so they cannot drift."""
    from b_ml_pipeline import is_stock as b
    from c_dl_pipeline import is_stock as c
    from e_quantile_daily_pipeline import is_stock as e
    from run_a_stat import _is_stock as a
    from target_kinds import is_stock as canonical

    assert b is c is e is a is canonical


def test_the_alias_set_is_the_union_of_what_the_copies_held():
    """The union, because the two mistakes are unequal.

    Treating a stock as a flow zero-fills gaps in a level series and models the level directly
    where a delta was intended -- an order-of-magnitude error. Treating a flow as a stock is
    wrong but visible. When the mistakes are unequal, take the union.
    """
    from target_kinds import STOCK_ALIASES

    assert STOCK_ALIASES == {"state budget balance", "balance", "t0", "net", "stock"}
    # The three that only ever appeared in one family's set are all still stock.
    from target_kinds import is_stock
    for name in ("t0", "net", "stock"):
        assert is_stock(name) is True


def test_no_module_keeps_its_own_alias_set():
    """A literal alias set anywhere outside target_kinds is a copy waiting to drift."""
    offenders = []
    for p in sorted(BACKEND.glob("*.py")):
        if p.name in ("target_kinds.py", "family_capabilities.py"):
            continue          # the definition, and the regression reporter
        txt = p.read_text(errors="ignore")
        if '"state budget balance"' in txt or "'state budget balance'" in txt:
            offenders.append(p.name)
    assert offenders == [], f"these still hold their own alias set: {offenders}"


@pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
def test_unifying_reclassifies_nothing_that_exists_today():
    """The union widened the set; nothing in the canonical file moved because of it.

    Only "State budget balance" appears in the file. If a client ever loads a column named
    `t0`, `net` or `stock`, this fails and the reclassification becomes a deliberate discovery.
    """
    from target_kinds import STOCK_ALIASES, is_stock

    cols = {c.strip().lower() for c in pd.read_csv(DATA, nrows=1).columns}
    widened = STOCK_ALIASES - {"state budget balance", "balance"}
    assert not (widened & cols), (
        f"a newly-unioned alias is now a real column {widened & cols} -- its treatment just "
        f"changed. Confirm that is intended.")
    assert is_stock("State budget balance") is True
    assert sum(1 for c in cols if is_stock(c)) == 1, "exactly one stock column is expected"


def test_the_changelog_no_longer_asserts_the_stale_claim():
    """The root cause was a false claim nothing could fail on."""
    txt = (BACKEND.parent / "CHANGELOG.md").read_text()
    assert "No longer true — corrected 2026-08-12" in txt
    assert "family_capabilities.py" in txt, (
        "the correction must point at the machine-readable form")
