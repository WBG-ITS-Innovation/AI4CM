"""Every page must import and render without raising.

Streamlit's ``AppTest`` executes the page script in-process, so an exception anywhere in the
page body fails the test — which is the coverage that matters here: these pages are long and
were untested, and an import error or a bad column reference is invisible until someone opens
the tab in front of an audience. This suite is what caught an import landing after its own use
while the visual pass was being applied.

**What this does and does not verify.** Every case renders against whatever run artifacts are
actually on disk. The fixtures below build synthetic run folders and set ``AI4CM_RUNS_DIR``,
but **the pages do not read that variable** — their runs directory is hard-coded — so the
three "states" currently exercise the same real artifacts rather than three different ones.
The fixtures are kept because making the pages honour an override is a content change, listed
in ``reports/ui_content_backlog.md``; until then, do not read these cases as covering the
empty-artifact path.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

# Streamlit lives only in the frontend venv. Skip cleanly rather than break collection, so a
# bare `pytest` from the repository root still runs the whole backend suite.
pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    FRONTEND / "Overview.py",
    FRONTEND / "pages" / "00_Data_Preprocessing.py",
    FRONTEND / "pages" / "00_Lab.py",
    FRONTEND / "pages" / "01_Dashboard.py",
    FRONTEND / "pages" / "02_History.py",
    FRONTEND / "pages" / "03_Models.py",
    FRONTEND / "pages" / "04_Compare.py",
    FRONTEND / "pages" / "05_Forecast.py",
]

TIMEOUT = 60


def _run(page: Path) -> AppTest:
    at = AppTest.from_file(str(page), default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_clean(at: AppTest, page: Path, state: str) -> None:
    if at.exception:
        msgs = "\n".join(str(e.value) for e in at.exception)
        pytest.fail(f"{page.name} raised in state '{state}':\n{msgs}")


# ── 1. every page renders with NO artifacts ───────────────────────────────────

@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_with_no_artifacts(page, tmp_path, monkeypatch):
    """Intended as the fresh-clone case. See the module docstring: the override is not yet
    honoured by the pages, so this currently duplicates the repository-artifacts case."""
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(tmp_path / "empty_runs"))
    (tmp_path / "empty_runs").mkdir(parents=True, exist_ok=True)
    at = _run(page)
    _assert_clean(at, page, "no artifacts")


# ── 2. every page renders against the real repository state ───────────────────

@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_with_repository_artifacts(page):
    """Whatever is actually committed -- registry, published forecast, scorecard."""
    at = _run(page)
    _assert_clean(at, page, "repository artifacts")


# ── 3. a withheld run must render and stay withheld ───────────────────────────

def _make_run(root: Path, *, withheld: bool) -> Path:
    """Minimal but schema-faithful run folder."""
    d = root / ("run_withheld" if withheld else "run_ok")
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    n = 60
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    y = 1e8 + 2e7 * rng.normal(0, 1, n)
    pred = pd.DataFrame({
        "date": idx, "target_date": idx, "origin_date": idx - pd.offsets.BDay(5),
        "origin_value": y * 0.98, "target": "Revenues", "horizon": 5,
        "model": "LightGBM_L1", "y_true": y, "y_pred": y * 1.01,
        "yhat_p10": y - 3e7, "yhat_p50": y * 1.01, "yhat_p90": y + 3e7,
    })
    pred.to_csv(d / "predictions_long.csv", index=False)
    pd.DataFrame({"model": ["LightGBM_L1"], "MAE": [2.0e7], "RMSE": [3.0e7],
                  "R2": [0.4], "target": ["Revenues"], "horizon": [5]}
                 ).to_csv(d / "leaderboard.csv", index=False)
    pd.DataFrame({"model": ["LightGBM_L1"], "target": ["Revenues"], "horizon": [5],
                  "fold": [1], "MAE": [2.0e7], "RMSE": [3.0e7], "MASE": [0.92],
                  "R2": [0.4]}).to_csv(d / "metrics_long.csv", index=False)
    integ = {
        "pipeline": "ML", "target": "Revenues", "horizon": 5,
        "run_status": "FAILED_QUALITY" if withheld else "SUCCESS",
        "quality_gate_passed": not withheld,
        "alignment_ok": True, "n_misaligned": 0, "misaligned_examples": [],
        "best_model": "LightGBM_L1", "mae_best": 2.0e7,
        "mae_persistence": 3.4e7, "skill_pct": 41.0,
        "shuffled_to_normal_ratio": 1.05 if withheld else 4.2,
        "signal_detected": not withheld,
        "signal_verdict": ("WEAK SIGNAL: shuffling the targets barely hurt "
                           "(ratio 1.05 < 1.50 required)") if withheld
                          else "signal present: shuffling made error 4.2x worse",
        "overfit_ratios": {"LightGBM_L1": 2.1}, "overfit_gate_ratio": 3.0,
        "overfit_excluded_models": [], "best_shift": 0,
        "shift_interpretation": "no shift", "probe": "ridge",
    }
    (d / "artifacts" / "integrity_report.json").write_text(json.dumps(integ, indent=2))
    return d


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_with_one_run(page, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _make_run(runs, withheld=False)
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(runs))
    at = _run(page)
    _assert_clean(at, page, "one run")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_with_a_withheld_run(page, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _make_run(runs, withheld=True)
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(runs))
    at = _run(page)
    _assert_clean(at, page, "withheld run")


# ── the withheld verdict must survive to the screen ──────────────────────────

def test_withheld_verdict_is_visible_on_the_forecast_page():
    """Revenues and Expenditure are `withheld_as_forecast` in the registry.

    If the page can render them without the word appearing, the honesty guarantee is only
    in the data and not in the product. This asserts existing behaviour; it does not add it.
    """
    at = _run(FRONTEND / "pages" / "05_Forecast.py")
    _assert_clean(at, FRONTEND / "pages" / "05_Forecast.py", "repository artifacts")
    blob = " ".join(
        str(getattr(el, "value", "") or getattr(el, "body", "") or "")
        for el in list(at.markdown) + list(at.error) + list(at.warning) + list(at.info)
    ).lower()
    from registry import load_registry
    held = [r for r in load_registry()["recipes"]
            if r["publication"]["verdict"] != "publishable"]
    if held:
        assert "withheld" in blob, (
            "the registry withholds a target but the Forecast page never says so"
        )


def test_no_sample_data_on_a_default_path():
    """sample_data must not be reachable without an explicit opt-in.

    Ground rule: no placeholder series in any default path. A demo path may exist, but it has
    to be chosen and labelled, never fallen back into.
    """
    offenders = []
    for page in PAGES:
        src = page.read_text()
        for line in src.splitlines():
            if "sample_data" not in line:
                continue
            low = line.strip().lower()
            if low.startswith("#") or "sample data" in low:
                continue
            # a reference is acceptable only if it is gated on an explicit user choice
            if not any(tok in line for tok in ("if ", "checkbox", "selectbox", "radio",
                                               "toggle", "SAMPLE", "demo_mode")):
                offenders.append(f"{page.name}: {line.strip()[:100]}")
    assert not offenders, "sample_data reachable without opt-in:\n" + "\n".join(offenders)
