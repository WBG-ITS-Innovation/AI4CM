"""The Scorecard must render honestly when there is nothing to show, and correctly when there is.

Why both states are tested
--------------------------
The true state of this project today is zero scored predictions and twenty-five pending
ones. A page that has only ever been seen in that state has a scored branch nobody has
ever rendered, and the first time it runs will be the day a client is looking at it.

So there are two fixtures. The empty one asserts the page says what is true rather than
rendering a blank table, and lists the pending dates, because a track record that shows
only the rows it has scored is one that can be made to look good by scoring selectively.
The scored one is deliberately synthetic, is labelled as such on screen, and exists to
prove the table, the chart and the summary metrics work.

The third group covers the upload. Its property is that the page decides nothing: it calls
``backend/ingest_actuals`` and renders what that returns, so the browser and the command
line cannot drift apart.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

from published_forecasts import SCORECARD_COLUMNS, SCORECARD_SCHEMA_VERSION  # noqa: E402

PAGE = FRONTEND / "pages" / "08_Scorecard.py"
SOURCE = PAGE.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_caches():
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _render() -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=180)
    at.run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    return at


def _text(at: AppTest) -> str:
    parts = []
    for collection in ("markdown", "caption", "info", "warning", "error", "success",
                       "subheader", "header"):
        for element in getattr(at, collection, []):
            parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


def _synthetic_scorecard(path: Path, n_days: int = 5) -> Path:
    """Clearly-synthetic scored rows. Every identifier says so."""
    rows = []
    for target, base in (("Revenues", 4.0e7), ("Expenditure", 5.0e7)):
        for i, day in enumerate(pd.bdate_range("2025-08-07", periods=n_days)):
            p50 = base * (1 + 0.02 * i)
            actual = base * (1 + 0.03 * i)
            persistence = base * (1 + 0.08 * i)
            ops = base * (1 + 0.10 * i)
            rows.append({
                "schema_version": SCORECARD_SCHEMA_VERSION,
                "issue_date": "2025-08-06", "target": target,
                "recipe_id": "SYNTHETIC-EXAMPLE", "horizon": i + 1,
                "origin_date": "2025-08-06", "origin_value": base,
                "target_date": str(day.date()),
                "p10": p50 * 0.9, "p50": p50, "p90": p50 * 1.1, "interval_nominal": 0.8,
                "y_true": actual, "abs_error": abs(actual - p50),
                "inside_interval": bool(p50 * 0.9 <= actual <= p50 * 1.1),
                "persistence_pred": persistence,
                "persistence_abs_error": abs(actual - persistence),
                "skill_vs_ruler_pct": 30.0, "persistence_source": "synthetic",
                "ops_pred": ops, "ops_abs_error": abs(actual - ops),
                "skill_vs_ops": 40.0, "ops_source": "synthetic",
                "scored_in_window": "live", "publication_verdict": "publishable",
                "point_model": "SYNTHETIC", "interval_model": "SYNTHETIC",
                "target_transform": "raw", "data_sha_at_issue": "0" * 8,
                "git_sha_at_issue": "0" * 8, "scored_at_data_sha": "0" * 8,
            })
    pd.DataFrame(rows, columns=list(SCORECARD_COLUMNS)).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Today's state: nothing scored, and it says so
# ---------------------------------------------------------------------------

def test_the_page_renders_with_nothing_scored():
    text = _text(_render())
    assert "Nothing has been scored yet, and that is the honest state." in text


def test_the_empty_state_lists_the_pending_dates():
    """A record showing only its scored rows can be made to look good by scoring selectively."""
    at = _render()
    assert "Days awaiting their actual figure" in _text(at)
    frames = [df.value for df in at.dataframe]
    assert frames, "the pending dates must be rendered, not merely counted"
    assert any("For the day" in list(f.columns) for f in frames)


def test_the_counts_are_the_real_ones():
    at = _render()
    metrics = {mt.label: mt.value for mt in at.metric}
    assert metrics["Predictions scored"] == "0"
    assert int(metrics["Still pending"]) > 0
    assert int(metrics["Forecast issues retained"]) > 0


def test_the_intro_explains_pending_without_jargon():
    text = _text(_render())
    assert "has not yet been reported" in text
    assert "scored against the actual figure once that figure arrives" in text


# ---------------------------------------------------------------------------
# 2. The scored branch, on a fixture that says it is a fixture
# ---------------------------------------------------------------------------

def test_the_scored_branch_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    at = _render()
    text = _text(at)
    assert "scored prediction(s)" in text
    assert "Nothing has been scored yet" not in text


def test_a_substituted_scorecard_says_so_on_screen(tmp_path, monkeypatch):
    """A synthetic table that does not announce itself is a synthetic table someone will quote."""
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    text = _text(_render())
    assert "You are looking at a substituted scorecard, not the real one." in text
    assert "Nothing on this screen describes a real published forecast." in text


def test_the_scored_table_carries_every_column_the_task_asks_for(tmp_path, monkeypatch):
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    at = _render()
    wanted = {"Issued", "For the day", "Central estimate", "Actual", "Absolute error",
              "Inside the range", "Better than the naive rule by",
              "Better than the current method by"}
    assert any(wanted <= set(df.value.columns) for df in at.dataframe), (
        "no rendered table carries the scored columns"
    )


def test_the_scored_branch_charts_forecast_against_actual(tmp_path, monkeypatch):
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    at = _render()
    assert len(at.get("plotly_chart")) >= 2, "one chart per scored target"


def test_the_scored_branch_summarises_per_target(tmp_path, monkeypatch):
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    labels = [mt.label for mt in _render().metric]
    assert labels.count("Actual fell inside the range") == 2
    assert labels.count("Better than the naive rule by") == 2


def test_a_corrupt_scorecard_falls_back_to_the_empty_state(tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\x00\x01\x02")
    monkeypatch.setenv("AI4CM_SCORECARD", str(bad))
    assert "Nothing has been scored yet" in _text(_render())


# ---------------------------------------------------------------------------
# 3. The upload decides nothing
# ---------------------------------------------------------------------------

def test_the_upload_calls_the_shared_ingest_and_reimplements_nothing():
    assert "from ingest_actuals import" in SOURCE
    assert "validate(candidate)" in SOURCE
    assert "install(candidate" in SOURCE
    for reimplementation in ("hashlib", "sha256", "shutil.copy",
                             "def _validate", "def _install"):
        assert reimplementation not in SOURCE, (
            f"the page carries its own ingest logic: {reimplementation}"
        )


def test_nothing_is_written_before_a_person_confirms():
    """The confirmation is not a formality: an install replaces the file every pipeline reads."""
    body = SOURCE.split("if uploaded is not None:")[1]
    checkbox_at = body.index("st.checkbox(")
    install_at = body.index("install(candidate")
    assert checkbox_at < install_at
    assert "disabled=not confirmed" in body


def test_the_install_button_is_gated_on_a_passing_check():
    body = SOURCE.split("if uploaded is not None:")[1]
    assert "if not check.ok:" in body
    refused = body.split("if not check.ok:")[1].split("else:")[0]
    assert "nothing on disk has changed" in refused
    assert "install(" not in refused


def test_the_result_summary_says_what_changed_on_disk():
    body = SOURCE.split("install(candidate")[1]
    for claim in ("new day(s) added", "value(s) revised", "previous file was kept at"):
        assert claim in body


def test_the_result_summary_says_how_many_were_scored_and_how_many_remain():
    body = SOURCE.split("install(candidate")[1]
    assert "rescored['scored']" in body
    assert "rescored['pending']" in body


def test_the_uploader_states_its_requirements_up_front():
    at = _render()
    helps = [str(getattr(u, "help", "")) for u in at.get("file_uploader")]
    assert any("must contain every column" in h for h in helps)


# ---------------------------------------------------------------------------
# 4. Sentence discipline on this page's own copy
# ---------------------------------------------------------------------------

def test_no_em_dashes_or_double_hyphens_in_visible_copy():
    for line in SOURCE.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for quoted in re.findall(r'"([^"]*)"', stripped):
            assert "—" not in quoted, f"em dash in user-visible copy: {quoted}"
            assert "--" not in quoted, f"double hyphen in user-visible copy: {quoted}"


def test_the_page_does_not_overclaim(tmp_path, monkeypatch):
    monkeypatch.setenv("AI4CM_SCORECARD", str(_synthetic_scorecard(tmp_path / "sc.csv")))
    text = _text(_render()).lower()
    for overclaim in ("proven in production", "guaranteed", "always accurate",
                      "state of the art"):
        assert overclaim not in text
