"""The Forecast page's reading tab shows the newest artifact per target, and says which.

Why this file exists
--------------------
The reading tab read one directory, ``backend/forecast_runs/forward/latest``, which only
``run_forward_forecast.py`` writes. A page-launched official run publishes into
``forecasts/published/<issue>/`` and never touches that directory, so after one the tab kept
showing whatever the runner last wrote: on this machine an artifact dated 2025-08-06
(inference-horizon map, §1.3, "The reading tab does not update from a page-launched run").

Decision of 2026-10-01: read-side only. No second writer for ``forward/latest``. The tab picks,
per target, the newest of the forward run and the published issues, by the artifact's own
``generated_at_utc``, never by directory name (the runner's issue is named by data date and
the page's by wall clock, so names do not order). Every row says where it came from and what
data it was built from, so staleness is visible rather than hidden.

Every fixture here is a temporary directory. The real forward directory and the real store
are never read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import insights as ins  # noqa: E402

TARGETS = ("Revenues", "Expenditure", "State budget balance")


def _rows(target: str, origin: str, p50: float) -> list:
    dates = pd.bdate_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=5)
    return [{"target": target, "horizon": h, "origin_date": origin, "origin_value": p50,
             "target_date": str(d.date()), "p10": p50 * 0.8, "p50": p50, "p90": p50 * 1.2,
             "point_model": "LightGBM_L1", "interval_model": "GBQuantile",
             "modelled_as": "level", "target_transform": "raw"}
            for h, d in enumerate(dates, start=1)]


def _provenance(generated: str, data_date: str) -> dict:
    return {"run_kind": "forward_forecast", "generated_at_utc": generated,
            "data": {"name": "d.csv", "sha256": "abc", "n_rows": 1, "latest_data_date": data_date},
            "code": {"git_sha": "deadbeef"}, "calendar_version": "cal1",
            "test_window_touched": False, "recipes": [], "notes": []}


def _forward(root: Path, targets, generated: str, origin: str, p50: float = 100.0) -> Path:
    """A runner artifact: forward_forecast.csv + forward_provenance.json."""
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for t in targets for r in _rows(t, origin, p50)]).to_csv(
        root / "forward_forecast.csv", index=False)
    (root / "forward_provenance.json").write_text(json.dumps(_provenance(generated, origin)))
    (root / "forward_gates.json").write_text("{}")
    return root


def _issue(store: Path, name: str, targets, generated: str, origin: str, p50: float = 200.0) -> Path:
    """A published issue: forecast.csv + provenance.json, as publish() writes them."""
    d = store / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for t in targets for r in _rows(t, origin, p50)]).to_csv(
        d / "forecast.csv", index=False)
    (d / "provenance.json").write_text(json.dumps(_provenance(generated, origin)))
    (d / "gates.json").write_text("{}")
    (d / "manifest.json").write_text(json.dumps({"issue_date": name}))
    return d


def _by_target(art: dict) -> dict:
    df = pd.DataFrame(art["forecasts"])
    return {t: g for t, g in df.groupby("target")}


# ── the pick ─────────────────────────────────────────────────────────────────

def test_a_newer_published_issue_wins_for_its_target_and_the_forward_run_keeps_the_rest(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    store = tmp_path / "published"
    _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    by = _by_target(art)
    assert by["Revenues"]["p50"].iloc[0] == 200.0, "the newer published Revenues did not win"
    assert by["Expenditure"]["p50"].iloc[0] == 100.0
    assert by["State budget balance"]["p50"].iloc[0] == 100.0
    assert art["sources"]["Revenues"]["kind"] == "published issue"
    assert art["sources"]["Expenditure"]["kind"] == "forward run"


def test_an_older_published_issue_does_not_displace_a_newer_forward_run(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-10-01T09:00:00+00:00", "2025-08-20")
    store = tmp_path / "published"
    _issue(store, "2026-09-30", ["Revenues"], "2026-09-30T08:00:00+00:00", "2025-08-06")

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    assert _by_target(art)["Revenues"]["p50"].iloc[0] == 100.0
    assert art["sources"]["Revenues"]["kind"] == "forward run"


def test_the_pick_is_by_generation_time_never_by_directory_name(tmp_path):
    """The runner names an issue by data date, the page by wall clock. Names do not order."""
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-10-01T09:00:00+00:00", "2025-08-20")
    store = tmp_path / "published"
    _issue(store, "2030-01-01", ["Revenues"], "2026-01-01T00:00:00+00:00", "2025-08-06")

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    assert art["sources"]["Revenues"]["kind"] == "forward run", (
        "a directory named far in the future displaced a newer artifact")


def test_a_tie_in_generation_time_prefers_the_published_copy(tmp_path):
    """publish() copies the runner's provenance verbatim, so a published runner issue ties with
    forward/latest. The published copy is the immutable record, so it is the one shown."""
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-10-01T09:00:00+00:00", "2025-08-20")
    store = tmp_path / "published"
    _issue(store, "2025-08-20", TARGETS, "2026-10-01T09:00:00+00:00", "2025-08-20")

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    assert all(s["kind"] == "published issue" for s in art["sources"].values())


# ── every row says where it came from ───────────────────────────────────────

def test_every_row_carries_its_source_and_data_date(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    store = tmp_path / "published"
    _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")

    df = pd.DataFrame(ins.load_newest_forecasts(forward_dir=fwd, published_root=store)["forecasts"])
    for col in ("source", "source_dir", "generated_at_utc", "data_through"):
        assert col in df.columns, f"rows do not carry {col!r}"
    rev = df[df["target"] == "Revenues"]
    assert set(rev["data_through"]) == {"2025-08-20"}
    assert set(rev["source"]) == {"published issue 2026-10-01"}
    exp = df[df["target"] == "Expenditure"]
    assert set(exp["data_through"]) == {"2025-08-06"}
    assert set(exp["source"]) == {"forward run"}


def test_the_sources_block_names_dir_generation_and_data_date_per_target(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    store = tmp_path / "published"
    d = _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")

    src = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)["sources"]
    assert src["Revenues"] == {"kind": "published issue", "label": "published issue 2026-10-01",
                               "dir": str(d), "generated_at_utc": "2026-10-01T09:00:00+00:00",
                               "data_through": "2025-08-20"}
    assert src["Expenditure"]["dir"] == str(fwd)


def test_the_provenance_returned_is_the_newest_chosen_artifacts_and_each_is_kept_per_target(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    store = tmp_path / "published"
    _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    assert art["provenance"]["generated_at_utc"] == "2026-10-01T09:00:00+00:00"
    assert art["provenance_by_target"]["Expenditure"]["generated_at_utc"] == "2026-08-16T00:57:51+00:00"


# ── degraded states ──────────────────────────────────────────────────────────

def test_with_no_published_store_the_forward_run_is_used_unchanged(tmp_path):
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=tmp_path / "absent")
    assert len(art["forecasts"]) == 15
    assert {s["kind"] for s in art["sources"].values()} == {"forward run"}
    assert art["dir"] == str(fwd)


def test_with_no_forward_run_the_published_store_alone_serves(tmp_path):
    store = tmp_path / "published"
    _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")
    art = ins.load_newest_forecasts(forward_dir=tmp_path / "absent", published_root=store)
    assert len(art["forecasts"]) == 5 and list(art["sources"]) == ["Revenues"]


def test_with_neither_the_loader_raises_the_same_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_forward_forecast.py"):
        ins.load_newest_forecasts(forward_dir=tmp_path / "a", published_root=tmp_path / "b")


def test_an_artifact_without_a_generation_time_never_beats_one_with(tmp_path):
    """A provenance that cannot say when it was made cannot claim to be newest."""
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    store = tmp_path / "published"
    d = _issue(store, "2026-10-01", ["Revenues"], "2026-10-01T09:00:00+00:00", "2025-08-20")
    prov = json.loads((d / "provenance.json").read_text())
    del prov["generated_at_utc"]
    (d / "provenance.json").write_text(json.dumps(prov))

    art = ins.load_newest_forecasts(forward_dir=fwd, published_root=store)
    assert art["sources"]["Revenues"]["kind"] == "forward run"


# ── the runner artifact loader is untouched ──────────────────────────────────

def test_the_runner_artifact_loader_still_reads_only_the_forward_directory(tmp_path):
    """The treasury report and the forward tests read the runner's own artifact; that path
    must not start consulting the published store behind their backs."""
    fwd = _forward(tmp_path / "forward", TARGETS, "2026-08-16T00:57:51+00:00", "2025-08-06")
    art = ins.load_forward_artifacts(out_dir=fwd)
    assert len(art["forecasts"]) == 15 and "sources" not in art
