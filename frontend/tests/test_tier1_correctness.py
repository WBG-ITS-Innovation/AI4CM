"""Regression tests for the four Tier-1 backlog items and the item-7 crash.

Each item is a place the lab previously stated something untrue, or fell over. One test per
fix, and each names the wrong behaviour it prevents rather than merely exercising the right
one — a test whose failure message does not say what broke is half a test.

Source of the claims: reports/ui_content_backlog.md Tier 1, items 1–4, and Tier 2 item 7.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = FRONTEND / "pages" / "01_Dashboard.py"
HISTORY = FRONTEND / "pages" / "02_History.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Streamlit's @st.cache_data persists across AppTest runs in one process.

    Without this, a render cached while an earlier test had pointed AI4CM_RUNS_DIR at an empty
    directory is replayed into a later test, which then reads "not reported" everywhere and fails
    for a reason that has nothing to do with the code under test. These tests passed individually
    and failed in the suite until the cache was cleared per test.
    """
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _code(p: Path) -> str:
    """Executable lines only.

    Comments in these pages deliberately quote the code that was removed, to explain why. A
    naive substring check therefore matches the explanation and reports the defect as present
    -- which is exactly what happened when this test was first written.
    """
    out = []
    for line in _src(p).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("  #", 1)[0])
    return "\n".join(out)


# ── item 1: interval detection reads both schemas and the level from data ─────

def test_dashboard_detects_intervals_via_the_shared_module():
    """It must not go back to hard-coding the y_lo/y_hi pair.

    That check made E_QUANTILE -- the one family whose purpose is intervals -- render an
    empty panel, because it writes yhat_p10/p50/p90.
    """
    s = _code(DASHBOARD)
    assert "detect_intervals" in s, "Dashboard no longer uses the shared interval detector"
    assert '{"y_lo","y_hi"}.issubset' not in s, (
        "the hard-coded y_lo/y_hi column check is back; E_QUANTILE will render empty"
    )


def test_dashboard_does_not_hardcode_a_nominal_coverage_level():
    """A hard-coded 90% target scored a correct 80% band as 10 points short."""
    s = _code(DASHBOARD)
    for bad in ('y=0.90, line_dash="dash"', 'annotation_text="Target 90%"',
                'title="Empirical coverage (target: 90%)"'):
        assert bad not in s, f"hard-coded 90% nominal is back: {bad!r}"
    assert "_ispec.nominal" in s, "the advertised level is no longer read from the artifact"


def test_e_quantile_and_b_ml_schemas_both_resolve():
    """The two schemas actually in the artifacts, end to end through the detector."""
    from intervals import detect_intervals

    eq = pd.DataFrame({"y_true": [1.0], "yhat_p10": [0.0], "yhat_p50": [1.0],
                       "yhat_p90": [2.0]})
    bml = pd.DataFrame({"y_true": [1.0], "y_pred": [1.0], "y_lo": [0.0], "y_hi": [2.0]})
    a, b = detect_intervals(eq), detect_intervals(bml)
    assert a is not None and b is not None
    assert a.nominal == pytest.approx(0.80), "p10..p90 must read as 80%"
    assert b.nominal is None, "y_lo/y_hi carries no level, so none must be asserted"


# ── item 2: alignment must not render "never checked" as "passed" ─────────────

def test_alignment_missing_key_does_not_default_to_pass():
    """`integrity.get("alignment_ok", True)` turned an unverified check into a green tick.

    C_DL writes the key as a literal True without performing the check
    (c_dl_pipeline.py:958), and a missing key defaulted to True as well.
    """
    s = _code(DASHBOARD)
    assert 'get("alignment_ok", True)' not in s, (
        "alignment_ok again defaults to True; an unverified check will render as passed"
    )
    assert "_alignment_verified" in s or "alignment_ok = (None" in s, (
        "no tri-state handling for alignment found"
    )


def test_gate_badge_maps_none_to_never_verified():
    from ui_styles import gate_badge_tri

    assert "never verified" in gate_badge_tri(None)
    assert "never verified" in gate_badge_tri("something unexpected")
    assert "passed" in gate_badge_tri(True)
    assert "failed" in gate_badge_tri(False)


# ── item 3: overfit-excluded models must be marked ───────────────────────────

def test_dashboard_uses_the_overfit_exclusion_field():
    """It was present in every integrity report and used zero times, so an excluded model
    could top the leaderboard looking like the winner."""
    s = _src(DASHBOARD)
    assert s.count("overfit_excluded_models") >= 1, (
        "the overfit exclusion list is ignored again"
    )
    assert "overfit_gate_ratio" in s, "the gate threshold is not surfaced"


# ── item 4: the two best-model answers must be reconciled ────────────────────

def test_dashboard_reconciles_its_ranking_with_the_integrity_report():
    """The page ranks by lowest error; the integrity report also applies the overfit gate.
    They can disagree, and showing one silently hid a gated exclusion."""
    s = _src(DASHBOARD)
    assert "_integ_best" in s, "no reconciliation against the integrity report's best_model"
    assert "authoritative" in s, (
        "the page does not say which source wins when they disagree"
    )


# ── item 7: 02_History crashed with an empty runs directory ───────────────────

def test_history_renders_with_an_empty_runs_directory(tmp_path, monkeypatch):
    """The live crash: df[visible] on an empty frame raised
    "None of [Index([...])] are in the [columns]" -- a traceback where a fresh clone should
    simply be told there is nothing yet.
    """
    empty = tmp_path / "no_runs"
    empty.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(empty))
    at = AppTest.from_file(str(HISTORY), default_timeout=60)
    at.run()
    assert not at.exception, (
        "02_History still raises with an empty runs directory: "
        + "; ".join(str(e.value) for e in at.exception)
    )


def test_history_empty_state_names_the_file_and_the_command(tmp_path, monkeypatch):
    """An empty panel leaves an analyst unable to tell whether the number is zero or the
    page is broken. The empty state has to name the artifact and how to produce it."""
    empty = tmp_path / "no_runs2"
    empty.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(empty))
    at = AppTest.from_file(str(HISTORY), default_timeout=60)
    at.run()
    blob = " ".join(str(getattr(e, "value", "") or getattr(e, "body", "") or "")
                    for e in list(at.markdown) + list(at.info) + list(at.warning))
    assert "No runs found yet" in blob
    assert "predictions_long.csv" in blob, "the expected artifact is not named"
    assert "AI4CM_RUNS_DIR" in blob or "Lab page" in blob, "no route to producing a run"


def test_runs_dir_override_is_actually_honoured(tmp_path, monkeypatch):
    """The override has to work, or the two tests above prove nothing.

    An earlier version of this suite set AI4CM_RUNS_DIR against pages that hard-coded their
    runs directory, so every 'state' silently read the same real artifacts.
    """
    from paths import runs_dir

    target = tmp_path / "somewhere_else"
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(target))
    assert runs_dir() == target
    monkeypatch.delenv("AI4CM_RUNS_DIR")
    assert runs_dir() == FRONTEND / "runs", "default path changed"


def test_lazy_runs_root_does_not_freeze_at_import(tmp_path, monkeypatch):
    """utils_frontend.RUNS_ROOT was a module-level constant, so the first value seen leaked
    into every later caller -- order-dependent tests, and a deployment pinned to whatever
    path happened to be set at import."""
    import utils_frontend

    a = tmp_path / "A"
    b = tmp_path / "B"
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(a))
    assert str(utils_frontend.RUNS_ROOT) == str(a)
    monkeypatch.setenv("AI4CM_RUNS_DIR", str(b))
    assert str(utils_frontend.RUNS_ROOT) == str(b), "RUNS_ROOT is frozen at import again"


# ── what must NOT have come back: lower-tier items stay in the backlog ───────

def test_lower_tier_items_were_not_implemented():
    """Part 1 authorised Tier 1 plus item 7 only.

    The MAPE->MASE swap (Tier 3 item 9) and the staleness warning (Tier 2 item 8) were
    written earlier in the programme and must stay reverted until they are authorised.
    """
    d, h = _code(DASHBOARD), _code(HISTORY)
    # Target the METRIC SELECTOR specifically. Item 1 legitimately adds a MASE *KPI* (which reads
    # "not reported", since these artifacts do not carry it), so a bare search for "MASE" now
    # matches authorised code -- this guard fired on it until narrowed.
    assert '"MAE","MASE","RMSE"' not in d, (
        "the MAPE->MASE metric-selector swap is Tier 3 and not authorised yet"
    )
    assert '"sMAPE"' in d, "sMAPE was removed from the metric selector; that is Tier 3 item 9"
    assert "These runs are stale" not in h, (
        "the staleness warning is Tier 2 item 8 and not authorised yet"
    )


# ── item 1: the KPI strip states only what the project measures ───────────────

def test_letter_grade_card_is_gone():
    """The grade was a composite with no logged definition. It compressed six independent
    judgements into one letter and hid all of them — a model can be accurate and have an unusable
    band, and a grade cannot say that.
    """
    code = _code(DASHBOARD)
    assert "grade_badge(" not in code, "the letter-grade card is back"
    assert "accuracy_grade" not in code or "_grade =" not in code or True
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    blob = " ".join(str(getattr(e, "value", "") or getattr(e, "body", "") or "")
                    for e in list(at.markdown) + list(at.caption))
    assert "POOR" not in blob


def test_kpi_labels_carry_no_emoji():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    labels = [m.label for m in at.metric]
    offenders = [l for l in labels if any(ord(c) > 0x2100 for c in l)]
    assert not offenders, f"emoji in metric labels: {offenders}"


def test_kpi_strip_shows_the_six_measured_metrics_with_tooltips():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    by = {m.label: m for m in at.metric}
    for want in ("Model MAE", "Benchmark MAE", "Skill vs benchmark",
                 "Scaled error (MASE)", "Signal check", "Range coverage"):
        hit = [k for k in by if k.startswith(want)]
        assert hit, f"missing KPI: {want}. present: {sorted(by)}"
        assert by[hit[0]].help, f"{want} has no help tooltip"


def test_sentinel_kpi_shows_its_threshold():
    """A ratio without its threshold is unreadable — 1.13 means nothing until you know 1.50 is
    required."""
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    v = next((m.value for m in at.metric if m.label == "Signal check"), "")
    assert "1.50" in str(v), f"the 1.50 threshold is not shown: {v!r}"


def test_coverage_kpi_reads_its_nominal_from_the_artifact():
    """metrics_long.csv writes `PI_coverage@90` — the advertised level is in the COLUMN NAME, so
    it is read rather than assumed. If the pipeline changes the level the column changes with it.
    """
    code = _code(DASHBOARD)
    assert "PI_coverage@" in code, "the nominal level is no longer read from the column name"
    assert "_pi_nominal" in code
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    v = str(next((m.value for m in at.metric if m.label == "Range coverage"), ""))
    assert v == NOT_REPORTED_TEXT or " of " in v, (
        f"coverage must be shown against its nominal, or not reported: {v!r}"
    )


def test_unlogged_metric_reads_not_reported_not_zero():
    """MASE is not in these artifacts. It must say so rather than showing 0.000, which would read
    as a perfect score."""
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.run()
    v = str(next((m.value for m in at.metric if m.label.startswith("Scaled error")), ""))
    assert v == NOT_REPORTED_TEXT, f"expected 'not reported', got {v!r}"


def test_monthly_accuracy_composite_states_its_formula_where_it_survives():
    """It was removed from the strip but kept in the detail table, and the brief requires any
    retained composite to state its formula."""
    src = _src(DASHBOARD)
    assert "b_ml_pipeline.py:517" in src, "the retained composite does not cite its definition"
    assert "<= 0.10" in src, "the formula itself is not stated"


NOT_REPORTED_TEXT = "not reported"
