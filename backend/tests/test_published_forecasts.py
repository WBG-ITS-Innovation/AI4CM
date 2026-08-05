"""Published forecasts must be retained immutably and scored only against arrived truth.

The load-bearing test is `test_scorer_refuses_a_date_whose_truth_has_not_arrived`. Without
it, "scoring published forecasts" could quietly become a way to read the sealed window.
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

from published_forecasts import (  # noqa: E402
    NOMINAL_COVERAGE,
    SCORECARD_COLUMNS,
    TruthNotAvailable,
    list_published,
    publish,
    score_one,
    score_published,
    summarize_scorecard,
)

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


def _truth(n=400, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(5)
    return pd.Series(1e8 + 2e7 * rng.normal(0, 1, n), index=idx)


def _forward_dir(tmp_path, target_dates, origin="2023-06-30", target="Revenues"):
    d = tmp_path / "fwd"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, td in enumerate(target_dates, start=1):
        rows.append({"target": target, "horizon": i, "origin_date": origin,
                     "origin_value": 1.0e8, "target_date": str(td),
                     "p10": 8.0e7, "p50": 1.0e8, "p90": 1.2e8,
                     "p50_quantile_model": 1.0e8, "point_model": "LightGBM_L1",
                     "interval_model": "GBQuantile", "n_train_rows": 500,
                     "n_features": 40, "modelled_as": "level",
                     "target_transform": "ratio"})
    pd.DataFrame(rows).to_csv(d / "forward_forecast.csv", index=False)
    (d / "forward_provenance.json").write_text(json.dumps({
        "data": {"sha256": "abc123", "latest_data_date": origin},
        "code": {"git_sha": "deadbeef"}, "calendar_version": "cal1",
        "test_window_touched": False,
        "recipes": [{"target": target, "recipe_id": "rev-v1",
                     "point_model": "LightGBM_L1", "target_transform": "ratio"}],
    }), encoding="utf-8")
    (d / "forward_gates.json").write_text(json.dumps({
        "rev-v1": {"target": target, "status": "candidate -- pre-tuning", "gates": {}}}),
        encoding="utf-8")
    return d


# ── the rule that keeps this from becoming a holdout read ──────────────────────

def test_scorer_refuses_a_date_whose_truth_has_not_arrived():
    """A published date is scored only once reality is in the canonical dataset.

    This is what separates scoring a committed forecast from evaluating a sealed window: the
    scorer cannot reach into data we do not have, so it cannot manufacture an accuracy number
    for the holdout.
    """
    truth = _truth()
    future = truth.index[-1] + pd.offsets.BDay(10)
    row = {"target": "Revenues", "target_date": str(future.date()),
           "p10": 8e7, "p50": 1e8, "p90": 1.2e8}
    with pytest.raises(TruthNotAvailable, match="not in the canonical dataset yet"):
        score_one(row, truth)


def test_score_published_reports_future_dates_as_pending_not_as_zero(tmp_path):
    """Pending must never be silently counted as scored, or as a perfect score."""
    truth_end = pd.Timestamp("2024-06-28")
    dates = [truth_end + pd.offsets.BDay(k) for k in (1, 2, 3)]
    fwd = _forward_dir(tmp_path, [d.date() for d in dates])
    root = tmp_path / "published"
    publish(fwd, issue_date="2024-06-28", published_root=root)

    out = score_published(DATA, published_root=root,
                          scorecard_path=tmp_path / "scorecard.csv")
    # Truth for July 2024 IS in the canonical data, so these should score.
    assert out["scored"] + out["pending"] == 3
    sc = pd.read_csv(tmp_path / "scorecard.csv")
    assert list(sc.columns) == list(SCORECARD_COLUMNS)


def test_the_live_published_run_is_entirely_pending(tmp_path):
    """Today's real published forecast must be 100% pending.

    Every one of its dates is beyond the data end by construction (the forward run asserts
    that), so any scored row here would mean the scorer had found truth that should not
    exist.
    """
    if not list_published():
        pytest.skip("nothing published yet")
    out = score_published(DATA, scorecard_path=tmp_path / "sc.csv")
    assert out["scored"] == 0, (
        f"{out['scored']} published rows scored, but every published date should still be "
        f"in the future"
    )
    assert out["pending"] > 0


# ── it must actually work once truth arrives ──────────────────────────────────

def test_scoring_works_on_a_backdated_issue(tmp_path):
    """Prove the machinery, not just the refusal.

    A forecast backdated into history is fully scoreable, so this exercises realized error,
    the persistence comparator and the interval hit in one pass.
    """
    truth = _truth()
    td = truth.index[100]
    y = float(truth.loc[td])
    row = {"target": "Revenues", "target_date": str(td.date()),
           "p10": y - 1e7, "p50": y + 5e6, "p90": y + 1e7}
    got = score_one(row, truth)
    assert got["y_true"] == pytest.approx(y)
    assert got["abs_error"] == pytest.approx(5e6)
    assert got["inside_interval"] is True
    # the comparator is y at (target_date - 5 business days), as everywhere else
    assert got["persistence_pred"] == pytest.approx(float(truth.iloc[95]))


def test_interval_miss_is_recorded_as_a_miss():
    truth = _truth()
    td = truth.index[120]
    y = float(truth.loc[td])
    row = {"target": "Revenues", "target_date": str(td.date()),
           "p10": y + 1e7, "p50": y + 2e7, "p90": y + 3e7}   # band entirely above truth
    assert score_one(row, truth)["inside_interval"] is False


def test_skill_is_negative_when_persistence_wins():
    truth = _truth()
    td = truth.index[150]
    y = float(truth.loc[td])
    pers = float(truth.iloc[145])
    row = {"target": "Revenues", "target_date": str(td.date()),
           "p10": 0.0, "p50": y + abs(y - pers) * 5 + 1e7, "p90": 1e12}
    got = score_one(row, truth)
    assert got["skill_vs_ruler_pct"] < 0


def test_summary_reports_hit_rate_against_the_nominal_coverage():
    df = pd.DataFrame({
        "target": ["Revenues"] * 4,
        "abs_error": [1.0, 2.0, 3.0, 4.0],
        "persistence_abs_error": [2.0, 4.0, 6.0, 8.0],
        "inside_interval": [True, True, True, False],
        "issue_date": ["2024-01-01"] * 4,
    })
    s = summarize_scorecard(df)["Revenues"]
    assert s["n"] == 4
    assert s["realized_mae"] == pytest.approx(2.5)
    assert s["skill_vs_ruler_pct"] == pytest.approx(50.0)
    assert s["interval_hit_rate"] == pytest.approx(0.75)
    assert s["nominal_coverage"] == NOMINAL_COVERAGE


def test_summary_of_an_empty_scorecard_is_empty_not_fabricated():
    assert summarize_scorecard(pd.DataFrame()) == {}


# ── retention is immutable ────────────────────────────────────────────────────

def test_publish_refuses_to_silently_overwrite(tmp_path):
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    root = tmp_path / "pub"
    publish(fwd, issue_date="2023-06-30", published_root=root)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        publish(fwd, issue_date="2023-06-30", published_root=root)
    publish(fwd, issue_date="2023-06-30", published_root=root, overwrite=True)


def test_published_dir_carries_provenance_and_recipe_id(tmp_path):
    fwd = _forward_dir(tmp_path, ["2023-07-03", "2023-07-04"])
    root = tmp_path / "pub"
    dest = publish(fwd, published_root=root)
    for f in ("forecast.csv", "provenance.json", "gates.json", "manifest.json"):
        assert (dest / f).exists(), f"missing {f}"
    man = json.loads((dest / "manifest.json").read_text())
    assert man["data_sha_at_issue"] == "abc123"
    assert man["git_sha_at_issue"] == "deadbeef"
    assert man["recipes"][0]["recipe_id"] == "rev-v1"
    assert man["test_window_touched"] is False
    assert man["issue_date"] == "2023-06-30"


def test_published_forecast_has_no_truth_column(tmp_path):
    """At issue time there is nothing to score against, and the file must reflect that."""
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    dest = publish(fwd, published_root=tmp_path / "pub")
    cols = set(pd.read_csv(dest / "forecast.csv").columns)
    assert not ({"y_true", "actual", "abs_error"} & cols)


def test_the_real_published_run_is_tracked_by_git():
    """Retention is pointless if the directory is gitignored -- which it was."""
    import subprocess

    if not list_published():
        pytest.skip("nothing published yet")
    d = list_published()[0] / "forecast.csv"
    out = subprocess.run(["git", "check-ignore", str(d)],
                         capture_output=True, text=True, cwd=str(BACKEND.parent))
    assert out.returncode != 0, f"{d} is gitignored; the published record would not survive"


# ── registry / published / forward must agree on the recipe ───────────────────

def test_registry_published_and_forward_agree_on_recipe_id_and_transform():
    """One recipe per target, and all three surfaces must name the same one.

    The failure this blocks is quiet and serious: a published forecast produced by one
    recipe while the registry advertises another means the DEV accuracy shown next to those
    numbers belongs to a different model. Checked across all three surfaces because they are
    written at different times by different code paths.
    """
    import sys as _s
    _s.path.insert(0, str(BACKEND))
    from registry import load_registry
    from run_forward_forecast import champions_from_registry

    reg = {r["target"]: r for r in load_registry()["recipes"]}
    champs = {c.target: c for c in champions_from_registry()}

    issues = list_published()
    if not issues:
        pytest.skip("nothing published yet")
    latest = issues[-1]
    man = json.loads((latest / "manifest.json").read_text())
    pub = {r["target"]: r for r in man["recipes"]}
    fc = pd.read_csv(latest / "forecast.csv")

    assert set(reg) == set(champs) == set(pub), (
        f"target sets differ: registry={sorted(reg)} forward={sorted(champs)} "
        f"published={sorted(pub)}"
    )
    for target, r in reg.items():
        want_id = r["id"]
        want_tf = r["params"].get("target_transform", "raw")
        assert champs[target].recipe_id == want_id
        assert champs[target].transform == want_tf
        assert pub[target]["recipe_id"] == want_id, (
            f"{target}: published under {pub[target]['recipe_id']} but the registry "
            f"advertises {want_id}"
        )
        assert pub[target].get("target_transform", "raw") == want_tf
        rows = fc[fc["target"] == target]
        assert (rows["target_transform"] == want_tf).all(), (
            f"{target}: published rows carry a transform other than {want_tf}"
        )


def test_no_duplicate_issue_dates():
    """Two issues on one date would double-count that forecast in the scorecard."""
    names = [p.name for p in list_published()]
    assert len(names) == len(set(names))
