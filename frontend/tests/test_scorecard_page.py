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


def _synthetic_scorecard(path: Path, n_days: int = 5, skill: float = 30.0,
                         hit: bool = True) -> Path:
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
                "inside_interval": bool(hit and p50 * 0.9 <= actual <= p50 * 1.1),
                "persistence_pred": persistence,
                "persistence_abs_error": abs(actual - persistence),
                "skill_vs_ruler_pct": skill, "persistence_source": "synthetic",
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


def test_the_pending_rows_are_shown_in_full_and_not_merely_counted():
    """A record showing only its scored rows can be made to look good by scoring selectively.

    The pending list moved out of the results section's empty state and into "Published
    forecasts", where it is now one row per published prediction with its model and its
    published range, rather than a bare list of dates. So this asserts the property, which is
    that every waiting row is on screen, rather than the heading it used to sit under.
    """
    at = _render()
    text = _text(at)
    assert "Waiting for actual figures" in text

    frames = [df.value for df in at.dataframe]
    assert frames, "the pending rows must be rendered, not merely counted"
    pending = [f for f in frames if "For the day" in list(f.columns)]
    assert pending, f"no table of pending rows; columns were {[list(f.columns) for f in frames]}"

    # Every waiting row, not a sample of them.
    metrics = {mt.label: mt.value for mt in at.metric}
    assert len(pending[0]) == int(metrics["Still pending"]), (
        f"{len(pending[0])} rows shown against {metrics['Still pending']} pending")

    # And enough of each row to be useful: which line, which model, and the published range.
    for column in ("Treasury line", "Champion model", "Low (P10)", "Central (P50)",
                   "High (P90)"):
        assert column in list(pending[0].columns), f"{column} missing from the pending table"


def test_the_counts_are_the_real_ones():
    at = _render()
    metrics = {mt.label: mt.value for mt in at.metric}
    assert metrics["Predictions scored"] == "0"
    assert int(metrics["Still pending"]) > 0
    assert int(metrics["Forecast issues retained"]) > 0


def test_the_intro_explains_the_loop_without_jargon():
    """The top of the page must say what the page is and where the reader is in the cycle."""
    text = _text(_render())
    assert "Published forecasts are compared to what actually happened" in text
    assert "This is also where you upload them." in text
    # The loop, in order, in plain words.
    assert "written down before the day it describes" in text
    assert "you upload it here" in text
    assert "listed as pending" in text


def test_the_page_states_the_live_situation_rather_than_a_typed_figure():
    """"25 rows are waiting" must come from the scorer, or it becomes a lie on the first upload."""
    at = _render()
    text = _text(at)
    pending = int({mt.label: mt.value for mt in at.metric}["Still pending"])
    assert f"{pending} published forecast rows are waiting for actual figures" in text
    assert "Upload the Treasury's reported values to score them." in text

    source = PAGE.read_text(encoding="utf-8")
    assert "25 published" not in source, (
        "the count is hardcoded; it must be read from the scorer")


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


# ---------------------------------------------------------------------------
# 4. The published-forecasts inventory
#
# Read from forecasts/published/<issue_date>/forecast.csv, which is the record: written on the
# issue date and never edited. Nothing is copied or cached elsewhere, and whether a row counts
# as scored is decided by the scorer's own output rather than re-derived here.
# ---------------------------------------------------------------------------

def test_the_inventory_reads_the_publication_log_and_stores_nothing():
    source = PAGE.read_text(encoding="utf-8")
    assert "from published_forecasts import list_published" in source, (
        "the inventory must come from the publication log, not from a second copy")
    assert 'forecast.csv' in source
    for writer in (".to_csv(", "open(", "write_text(", "write_bytes("):
        # The uploader legitimately writes the candidate file; nothing else may write.
        occurrences = source.count(writer)
        if writer == "write_bytes(":
            assert occurrences <= 1, f"{writer} appears {occurrences} times"
        else:
            assert occurrences == 0, f"the page writes with {writer}, so it stores a copy"


def test_scored_versus_pending_is_decided_by_the_scorer_not_by_this_page():
    """Re-deriving "does this day have truth" would be a second copy of the scorer's rule."""
    source = PAGE.read_text(encoding="utf-8")
    fn = source.split("def load_published_rows")[1].split("\ndef ")[0]
    assert "scorecard" in fn.lower(), "scored status is not read from the scorecard"
    for reinvention in ("last_date", "Timestamp(", "bdate_range", "<= today", "TruthNotAvailable"):
        assert reinvention not in fn, (
            f"load_published_rows decides truth availability itself, via {reinvention!r}")


def test_the_inventory_degrades_to_a_message_when_nothing_is_published(monkeypatch, tmp_path):
    """A fresh clone has no forecasts/published/, and the page must still render."""
    import published_forecasts

    monkeypatch.setattr(published_forecasts, "PUBLISHED_ROOT", tmp_path / "absent")
    at = _render()
    text = _text(at)
    assert "No published forecast is held on this machine" in text
    assert not at.exception


def test_every_published_row_carries_its_model_and_its_range():
    at = _render()
    frames = [f for f in (df.value for df in at.dataframe) if "For the day" in list(f.columns)]
    assert frames, "no inventory table rendered"
    got = list(frames[0].columns)
    for column in ("For the day", "Treasury line", "Issued", "Champion model",
                   "Low (P10)", "Central (P50)", "High (P90)"):
        assert column in got, f"{column} missing; columns are {got}"


def test_the_inventory_explains_why_a_day_can_appear_twice():
    """Two issues forecast the same Revenues days from the same origin, so dates repeat.

    Without a sentence saying so, a duplicated date reads as a bug in the table.
    """
    assert "forecast in two issues from the" in _text(_render())


# ---------------------------------------------------------------------------
# 5. Retraining is explained, and is not a button
# ---------------------------------------------------------------------------

def test_the_page_offers_no_way_to_retrain_or_reselect():
    """Checked by mechanism, not by the word.

    The page legitimately says "retrain" several times, in the section heading and in the link
    to docs/REFRESH_AND_RETRAIN.md, because explaining retraining is the whole point of it. So
    this looks for the things that would actually do it.
    """
    source = PAGE.read_text(encoding="utf-8")
    for mechanism in ("official_run", "run_forward", "exploratory_run", "save_registry",
                      "forecast_modes", "subprocess", '"--mode"', "--publish"):
        assert mechanism not in source, f"the Scorecard page reaches for {mechanism!r}"

    buttons = [b.label for b in _render().get("button")]
    for b in buttons:
        low = b.lower()
        assert "retrain" not in low and "re-choose" not in low and "reselect" not in low, \
            f"a retrain button exists: {buttons}"


def test_the_retraining_explainer_renders_the_policy_rather_than_restating_it():
    """A hand-written copy of the policy is the copy that goes stale."""
    from registry import champion_policy

    policy = champion_policy()
    text = _text(_render())
    assert policy["statement"] in text, "the policy statement is not rendered verbatim"
    assert policy["reselection"] == "none" and policy["on_new_data"] == "refit_only"


def test_the_explainer_separates_refitting_from_re_choosing():
    """The whole point of the section: these are different acts and only one is automatic."""
    text = _text(_render())
    assert "Installing the file replaces the data the system reads" in text
    assert "with no button to press" in text
    assert "What never changes on its own." in text


def test_the_written_procedure_exists_and_is_linked():
    doc = REPO / "docs" / "REFRESH_AND_RETRAIN.md"
    assert doc.exists(), "docs/REFRESH_AND_RETRAIN.md is missing; the page links to it"
    body = doc.read_text(encoding="utf-8")
    # The distinction the doc exists to make, and the plain-words definitions it shares
    # with the app.
    # Compared against the unwrapped text: the doc is hard-wrapped, so a sentence a reader
    # sees as one line is two in the file.
    flat = " ".join(body.split())
    for required in ("**Refit.**", "**Re-choose.**",
                     "never saw while being chosen",
                     "nothing in this project writes that file",
                     "assert_selection_free"):
        assert required in flat, f"the procedure does not cover {required!r}"
    assert "—" not in body, "the procedure uses an em dash"
    # The label is on an expander, which _text does not collect.
    labels = [e.label for e in _render().get("expander")]
    assert "Read the refresh and retrain procedure" in labels, labels


# ---------------------------------------------------------------------------
# 6. The health verdict, all three branches
#
# It is derived only from columns the scorer already writes: skill_vs_ruler_pct and
# inside_interval. No new metric, no trend, no significance test. On a handful of scored days
# none of those would mean anything, and the third branch below is the one that says so.
# ---------------------------------------------------------------------------

def test_the_verdict_says_holding_up_when_it_beats_the_ruler_and_the_range_covers(
        monkeypatch, tmp_path):
    monkeypatch.setenv("AI4CM_SCORECARD",
                       str(_synthetic_scorecard(tmp_path / "sc.csv", n_days=8, skill=30.0)))
    text = _text(_render())
    assert "Holding up." in text
    assert "Degrading." not in text


def test_the_verdict_says_degrading_when_it_stops_beating_the_ruler(monkeypatch, tmp_path):
    monkeypatch.setenv("AI4CM_SCORECARD",
                       str(_synthetic_scorecard(tmp_path / "sc.csv", n_days=8, skill=-6.0)))
    text = _text(_render())
    assert "Degrading." in text
    assert "no longer more accurate than the simple rule of thumb" in text
    # And it must say what that means: a decision for a person, not an action for the page.
    assert "deliberate, recorded decision" in text


def test_the_verdict_says_degrading_when_the_published_range_stops_covering(
        monkeypatch, tmp_path):
    monkeypatch.setenv("AI4CM_SCORECARD",
                       str(_synthetic_scorecard(tmp_path / "sc.csv", n_days=8, skill=30.0,
                                                hit=False)))
    text = _text(_render())
    assert "Degrading." in text
    assert "covering fewer days than it claims" in text


def test_the_verdict_refuses_to_call_it_on_too_few_days(monkeypatch, tmp_path):
    """A confident verdict from four rows is worse than no verdict."""
    monkeypatch.setenv("AI4CM_SCORECARD",
                       str(_synthetic_scorecard(tmp_path / "sc.csv", n_days=3)))
    text = _text(_render())
    assert "Too few scored days to call it either way." in text
    assert "Holding up." not in text and "Degrading." not in text


def test_the_verdict_uses_no_metric_the_scorer_did_not_already_write():
    """"No new metrics" was the constraint. Checked at the source."""
    from published_forecasts import SCORECARD_COLUMNS

    import ast

    source = PAGE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn_node = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_health_verdict")

    fn = ast.get_source_segment(source, fn_node)
    used = {c for c in SCORECARD_COLUMNS if f'"{c}"' in fn}
    assert used <= {"skill_vs_ruler_pct", "inside_interval", "interval_nominal"}, used

    # Checked against the CODE, not the docstring. The docstring names the things the verdict
    # deliberately is not ("not a trend, a rolling window or a significance test"), so a
    # substring search over the whole function flags its own explanation.
    body = fn_node.body[1:] if ast.get_docstring(fn_node) else fn_node.body
    code = "\n".join(ast.get_source_segment(source, n) or "" for n in body)
    for invented in ("polyfit", "rolling", "ttest", "pvalue", "corr(", "std()", "ewm("):
        assert invented not in code, f"the verdict computes something new: {invented!r}"
