"""A forward forecast must predict unseen dates and must never read truth.

The failure this guards against is a forward artifact that quietly overlaps the data --
which on this project would specifically be an evaluation on the sealed 2025 holdout,
dressed up as a production run.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forward_forecast import (  # noqa: E402
    FORWARD_HORIZONS,
    QUANTILES,
    Champion,
    assert_forward_only,
    business_days_after,
    build_provenance,
    run_forward,
)
from preprocessing.fiscal_calendar import GROUP_A, GROUP_C  # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


# ── the guard that matters ────────────────────────────────────────────────────

def test_assert_forward_only_rejects_in_sample_dates():
    end = date(2025, 8, 6)
    assert_forward_only([date(2025, 8, 7), date(2025, 8, 8)], end)  # fine
    with pytest.raises(ValueError, match="at or before the data end"):
        assert_forward_only([date(2025, 8, 6)], end)
    with pytest.raises(ValueError, match="at or before the data end"):
        assert_forward_only([date(2025, 8, 7), date(2025, 7, 1)], end)


def test_business_days_skip_weekends_and_georgian_holidays():
    # 2025-08-28 is Mariamoba (Assumption), a Georgian public holiday on a Thursday.
    got = business_days_after(date(2025, 8, 25), 5)
    assert date(2025, 8, 28) not in got, f"forecast placed on a public holiday: {got}"
    assert all(d.weekday() < 5 for d in got)
    assert got == sorted(got) and len(set(got)) == 5


def test_business_days_after_is_strictly_after():
    got = business_days_after(date(2025, 8, 6), 3)
    assert all(d > date(2025, 8, 6) for d in got)
    assert got == [date(2025, 8, 7), date(2025, 8, 8), date(2025, 8, 11)]


# ── end-to-end on the real data ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_forward():
    """A cheap real run: one target, two horizons, a thin feature set."""
    if not DATA.exists():
        pytest.skip("canonical data not present")
    raw = pd.read_csv(DATA)
    champ = Champion(target="Revenues", point_model="LightGBM_L1",
                     fiscal_groups=(GROUP_A, GROUP_C), recipe_id="test")
    return raw, run_forward(raw, champ, horizons=(1, 2))


def test_every_target_date_is_beyond_the_data(small_forward):
    raw, df = small_forward
    data_end = pd.to_datetime(raw["date"]).max().normalize()
    assert (df["target_date"] > data_end).all(), (
        f"forward run emitted a date within the data (ends {data_end.date()})"
    )


def test_no_truth_column_exists(small_forward):
    """There must be nothing an accuracy metric could be computed from."""
    _, df = small_forward
    forbidden = {"y_true", "actual", "truth", "y", "error", "mae", "residual"}
    assert not (forbidden & set(df.columns)), sorted(forbidden & set(df.columns))


def test_intervals_are_ordered(small_forward):
    _, df = small_forward
    assert (df["p10"] <= df["p90"]).all()
    # Quantile crossing is an invalid interval, not merely a wide one.
    assert (df["p10"] <= df["p50_quantile_model"]).all()
    assert (df["p50_quantile_model"] <= df["p90"]).all()


def test_one_row_per_horizon_with_distinct_dates(small_forward):
    _, df = small_forward
    assert len(df) == 2
    assert df["horizon"].tolist() == [1, 2]
    assert df["target_date"].nunique() == 2, "each horizon must land on its own date"


def test_training_rows_shrink_as_horizon_grows(small_forward):
    """Horizon h can only train where y(t+h) is known, so deeper horizons have fewer rows.

    If this were flat, every horizon would be sharing one model and the per-day labels
    would be false.
    """
    _, df = small_forward
    n = df.sort_values("horizon")["n_train_rows"].tolist()
    assert n[0] > n[1], f"training rows did not shrink with horizon: {n}"


def test_provenance_records_the_sealed_window_and_the_scaling_decision():
    """WS4 has now run, so provenance must record WHICH transform was applied.

    This test previously asserted the string "WS4 pending". It failed the moment WS4 landed,
    which is the correct behaviour: the artifact must never describe a scaling decision that
    does not match the model that was actually fitted.
    """
    champ = Champion(target="Revenues", point_model="LightGBM_L1",
                     fiscal_groups=(GROUP_A,), recipe_id="r1", transform="ratio",
                     scaling="ratio-to-trailing-level (WS4 winner)")
    prov = build_provenance(str(DATA), [champ])
    assert prov["test_window_touched"] is False
    assert prov["run_kind"] == "forward_forecast"
    assert prov["data"]["sha256"]
    assert prov["calendar_version"]
    assert prov["recipes"][0]["target_transform"] == "ratio"
    assert "ratio" in prov["recipes"][0]["scaling"]
    assert any("no truth was read" in n for n in prov["notes"])


def test_forward_run_applies_the_registry_transform():
    """The published forecast must use the transform its DEV credentials were earned with.

    Publishing a raw fit under a recipe that won on `ratio` would make the quoted accuracy
    belong to a different model than the one that produced the numbers.
    """
    from registry import load_registry

    reg = load_registry()
    from run_forward_forecast import champions_from_registry

    champs = {c.target: c for c in champions_from_registry()}
    for r in reg["recipes"]:
        expected = r.get("params", {}).get("target_transform", "raw")
        assert champs[r["target"]].transform == expected, (
            f"{r['target']}: forward run would use {champs[r['target']].transform!r} "
            f"but the recipe specifies {expected!r}"
        )


def test_registry_champions_are_loadable_and_honest():
    from registry import load_registry, verify_against_log

    reg = load_registry()
    assert len(reg["recipes"]) == 3
    for r in reg["recipes"]:
        assert r["approved_by"] is None, "nothing is approved"
        assert "pending" in r["scaling"].lower() or "WS4" in r["scaling"]
        assert r["status"].startswith("candidate")
    out = verify_against_log()
    assert out["ok"], out["problems"]
    assert out["metrics_checked"] >= 9


def test_a_failing_gate_is_never_smoothed_into_a_pass():
    """The honesty property the demo turns on, stated independently of which target passes.

    Rewritten in P2. It previously asserted the specific verdicts of the day — both flows
    `withheld_as_forecast`, the stock target `publishable` — which made it a second copy of the
    registry rather than a check on it. All three of those verdicts changed when MASE became a
    gate and the sentinel threshold was calibrated, so the *property* is what belongs here:
    whatever the verdicts are, a failure is stated plainly and a fix is named. The verdicts
    themselves are pinned once, in `test_publication_gates.py`.
    """
    from registry import load_registry

    for r in load_registry()["recipes"]:
        pub = r["publication"]
        gates = r["dev_credentials"]["gates"]
        failing = [n for n, g in gates.items() if g.get("passed") is False]

        if pub["verdict"] == "publishable":
            assert not failing, (
                f"{r['target']} is publishable with failing gates {failing}")
            continue

        assert failing, f"{r['target']} is withheld but no gate failed"
        assert pub["reason_plain"], f"{r['target']} is withheld without a reason"
        assert pub["named_fix"], "a withheld model must name the fix"
        assert pub["decided_by"] in failing, (
            f"{r['target']}: decided_by {pub['decided_by']!r} is not among the failures")
        # Every failing gate must contribute its own words, not just the deciding one.
        for name in failing:
            assert gates[name]["reason_plain"], f"{r['target']}/{name} failed silently"


def test_the_reader_facing_reasons_carry_no_jargon():
    """These strings reach a Treasury reader, so they may not carry acronyms or model names.

    An earlier P2 draft wrote "MASE 1.104" into `reason_plain` and the existing prose tests in
    `test_insights.py` / `test_treasury_report.py` caught it. This asserts it at the source.
    """
    from registry import load_registry

    banned = ("MASE", "RMSE", "MAE", "sentinel", "LightGBM", "HistGBDT", "GBQuantile",
              "quantile", "P10", "P90", "DEV", "conformal")
    for r in load_registry()["recipes"]:
        texts = [r["publication"]["reason_plain"]]
        texts += [g["reason_plain"] for g in r["dev_credentials"]["gates"].values()
                  if g.get("passed") is False]
        for t in texts:
            found = [b for b in banned if b in t]
            assert not found, f"{r['target']}: jargon in reader-facing prose: {found} in {t[:90]!r}"
