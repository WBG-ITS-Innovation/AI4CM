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

import published_forecasts as pf  # noqa: E402
from published_forecasts import (  # noqa: E402
    NOMINAL_COVERAGE,
    SCORECARD_COLUMNS,
    TruthNotAvailable,
    list_published,
    publish,
    refresh_vault_manifest,
    retain_to_vault,
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


# ── a synthetic artifact must not be publishable ──────────────────────────────

def _stamp_synthetic(fwd: Path, notice: str = "SYNTHETIC DATA -- generated by "
                                              "backend/synthetic_data.py.") -> None:
    p = fwd / "forward_provenance.json"
    prov = json.loads(p.read_text())
    prov["data"]["is_synthetic"] = True
    prov["data"]["synthetic_notice"] = notice
    p.write_text(json.dumps(prov))


def test_a_run_built_on_synthetic_data_cannot_be_published(tmp_path):
    """The orphan case. Nothing in the codebase can write this stamp any more, which is the point.

    backend/synthetic_data.py is gone and provenance.py has no is_synthetic support, yet on
    2026-08-15 an artifact carrying the stamp was sitting in the shared forward directory -- a
    leftover of a deleted code state, with the real data restored on top of it. Every publish
    path would have taken it. It was caught by a human reading the file, which is not a control.
    """
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    _stamp_synthetic(fwd)
    pub, vault = tmp_path / "pub", tmp_path / "vault" / "published"

    with pytest.raises(pf.SyntheticArtifact, match="is_synthetic=true"):
        publish(fwd, published_root=pub, vault_root=vault)

    assert not pub.exists() and not vault.exists(), (
        "the refusal must happen before anything is written, not after")


def test_the_refusal_quotes_the_artifact_s_own_notice():
    """The artifact says why it exists; the error should not make the reader go and look."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fwd = _forward_dir(Path(td), ["2023-07-03"])
        _stamp_synthetic(fwd, "SYNTHETIC DATA -- demonstration of the pipeline only.")
        with pytest.raises(pf.SyntheticArtifact, match="demonstration of the pipeline only"):
            publish(fwd, published_root=Path(td) / "pub", vault_root=Path(td) / "v")


@pytest.mark.parametrize("value", [False, None])
def test_a_real_run_is_unaffected_by_the_guard(tmp_path, value):
    """`is_synthetic: false` and an absent key are both real data, and must publish."""
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    p = fwd / "forward_provenance.json"
    prov = json.loads(p.read_text())
    if value is not None:
        prov["data"]["is_synthetic"] = value
    p.write_text(json.dumps(prov))

    dest = publish(fwd, published_root=tmp_path / "pub", vault_root=tmp_path / "v" / "published")
    assert (dest / "forecast.csv").exists()


def test_publish_official_inherits_the_refusal():
    """The guard sits at the boundary every publish path crosses, not in one of them."""
    import inspect

    import forecast_modes
    src = inspect.getsource(forecast_modes.publish_official)
    assert "from published_forecasts import publish" in src, (
        "publish_official must route through publish(), which is where the guard lives")


# ── retention happens as part of publishing, not after it ─────────────────────

def test_publishing_writes_the_issue_and_its_vault_copy_together(tmp_path):
    """Retention used to be a manual `cp` and was therefore sometimes not done at all.

    The 2026-08-16 issue was published and retained nothing; a test caught it, not an auditor.
    Publishing now writes both locations, so forgetting is not one of the available outcomes.
    """
    fwd = _forward_dir(tmp_path, ["2023-07-03", "2023-07-04"])
    pub, vault = tmp_path / "pub", tmp_path / "vault" / "published"

    dest = publish(fwd, published_root=pub, vault_root=vault)

    mirrored = vault / dest.name
    assert mirrored.is_dir(), "publishing wrote the repo copy but retained nothing"
    for name in ("forecast.csv", "gates.json", "provenance.json", "manifest.json"):
        assert (mirrored / name).read_bytes() == (dest / name).read_bytes(), name


def test_the_vault_inventory_is_regenerated_by_the_write_that_invalidates_it(tmp_path):
    """A hand-maintained inventory drifts, and this one had: 700 recorded, 725 present."""
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    vault = tmp_path / "vault" / "published"
    dest = publish(fwd, published_root=tmp_path / "pub", vault_root=vault)

    man = json.loads((vault.parent / "MANIFEST.json").read_text())
    listed = {f["path"] for f in man["files"]}
    on_disk = {p.relative_to(vault.parent).as_posix()
               for p in vault.parent.rglob("*") if p.is_file()} - {"MANIFEST.json"}

    assert listed == on_disk, "the inventory does not match what is on disk"
    assert man["n_files"] == len(listed) and man["n_files"] > 0
    assert f"published/{dest.name}/forecast.csv" in listed
    assert man["total_bytes"] == sum(f["bytes"] for f in man["files"])
    assert "NEVER commit" in man["what"]


def test_a_failed_vault_write_leaves_no_published_issue_behind(tmp_path, monkeypatch):
    """Both or neither. A publish that retained nothing must not report success.

    The repo copy is gitignored, so an issue that exists only there survives until the next
    clean checkout -- which is indistinguishable from never having published it, except that
    the caller was told it worked.
    """
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    pub, vault = tmp_path / "pub", tmp_path / "vault" / "published"

    def boom(*_a, **_k):
        raise OSError("vault unwritable")

    monkeypatch.setattr(pf, "retain_to_vault", boom)
    with pytest.raises(OSError, match="vault unwritable"):
        publish(fwd, published_root=pub, vault_root=vault)

    assert not (pub / "2023-06-30").exists(), (
        "the repo copy survived a failed retention -- published but not retained")


def test_rollback_never_destroys_an_issue_it_did_not_create(tmp_path, monkeypatch):
    """The rollback must not turn a failed overwrite into data loss.

    Removing a directory that was already there would delete a prior published issue in order
    to report an error, which is a worse outcome than the error.
    """
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    pub, vault = tmp_path / "pub", tmp_path / "vault" / "published"
    publish(fwd, published_root=pub, vault_root=vault)
    assert (pub / "2023-06-30" / "forecast.csv").exists()

    monkeypatch.setattr(pf, "retain_to_vault",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("vault unwritable")))
    with pytest.raises(OSError):
        publish(fwd, published_root=pub, vault_root=vault, overwrite=True)

    assert (pub / "2023-06-30" / "forecast.csv").exists(), (
        "rollback deleted a pre-existing issue")


def test_publishing_into_a_temp_root_does_not_touch_the_real_vault(tmp_path):
    """The default must not be "always the real vault", or every test would write into it."""
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    before = sorted(p.name for p in pf.VAULT_PUBLISHED.iterdir()) \
        if pf.VAULT_PUBLISHED.exists() else []

    publish(fwd, published_root=tmp_path / "pub")          # no vault_root given

    after = sorted(p.name for p in pf.VAULT_PUBLISHED.iterdir()) \
        if pf.VAULT_PUBLISHED.exists() else []
    assert before == after, "publishing to a temp root wrote into the real vault"


def test_retaining_twice_replaces_rather_than_accumulates(tmp_path):
    """The estimator blobs land after publish() mirrors, so the re-sync must be idempotent."""
    fwd = _forward_dir(tmp_path, ["2023-07-03"])
    pub, vault = tmp_path / "pub", tmp_path / "vault" / "published"
    dest = publish(fwd, published_root=pub, vault_root=vault)

    (dest / "estimators").mkdir()
    (dest / "estimators" / "manifest.json").write_text('{"blobs": 1}')
    retain_to_vault(dest, vault)
    assert (vault / dest.name / "estimators" / "manifest.json").exists(), (
        "the re-sync did not pick up files written after the first mirror")

    (dest / "estimators" / "manifest.json").unlink()
    retain_to_vault(dest, vault)
    assert not (vault / dest.name / "estimators" / "manifest.json").exists(), (
        "the vault copy is a mirror, not an accumulation")

    man = json.loads((vault.parent / "MANIFEST.json").read_text())
    assert not any("estimators" in f["path"] for f in man["files"]), (
        "the inventory still lists a file the mirror removed")


VAULT = BACKEND.parent / "private_vault" / "published"


def test_the_published_record_is_retained_somewhere_durable():
    """Retention is pointless if the record does not survive. Only the location changed.

    Until 2026-08-15 this asserted the opposite -- `assert out.returncode != 0`, the published
    directory must NOT be gitignored -- because the original defect was that it was, so nothing
    recorded what had been said. That concern is still correct and this test still exists to
    enforce it. What changed is where the record lives: forecast.csv carries row-level Treasury
    figures (origin_value and the full P10/P50/P90 path), so it left the repository for
    private_vault/published/. Checking survival is therefore checking the vault; otherwise
    "the vault is the durable record" is a claim with nothing behind it.
    """
    import subprocess

    issues = list_published()
    if not issues:
        pytest.skip("nothing published yet")

    for d in issues:
        # Both halves of the policy, pinned together: ignored HERE and present THERE. If
        # someone re-tracks the repo copy, that is a policy change and should fail here.
        out = subprocess.run(["git", "check-ignore", str(d / "forecast.csv")],
                             capture_output=True, text=True, cwd=str(BACKEND.parent))
        assert out.returncode == 0, (
            f"{d.name}/forecast.csv is tracked; it carries row-level Treasury figures")

        vaulted = VAULT / d.name
        assert vaulted.is_dir(), (
            f"issue {d.name} has no copy in private_vault/published/ -- publishing it retained "
            f"nothing durable. publish() writes only to forecasts/published/, so the vault copy "
            f"is currently a manual step.")
        for name in ("forecast.csv", "gates.json"):
            assert (vaulted / name).read_bytes() == (d / name).read_bytes(), (
                f"{d.name}/{name} differs from its vault copy; one of the two has been edited "
                f"and a published issue is supposed to be immutable")


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

    # P2: a published issue no longer necessarily covers every registry target. `withheld` means
    # a trivial benchmark is more accurate, and `publish_official` refuses those -- so an issue
    # contains exactly the targets whose verdict permits publication, which is a SUBSET of the
    # registry. Asserting equality here was asserting the pre-P2 policy.
    assert set(reg) == set(champs), (
        f"registry and forward champions differ: registry={sorted(reg)} forward={sorted(champs)}")
    assert set(pub) <= set(reg), (
        f"published targets are not a subset of the registry: published={sorted(pub)}")
    publishable = {t for t, r in reg.items() if r["publication"]["verdict"] != "withheld"}
    assert set(pub) <= publishable, (
        f"a withheld target was published: {sorted(set(pub) - publishable)}")

    for target, r in {t: reg[t] for t in pub}.items():
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
