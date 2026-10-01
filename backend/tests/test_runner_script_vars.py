"""Guard against unbound-variable bugs in scripts/run_daily_forecast.sh.

Context: the backtest wiring shipped with `--run-dir "$OUT_DIR"` where the
script's variable is actually named RUN_DIR.  `bash -n` passed, because -n
only checks *syntax* — it never evaluates the script, so an undefined
variable is invisible to it.  The failure only appeared at the very end of a
multi-minute pipeline run, after every model had already been fitted.

This test does the cheap static check `bash -n` cannot: every uppercase
variable the script *reads* must either be assigned inside the script or be
a documented external input (env var / exported pipeline setting).
"""
from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_daily_forecast.sh"

# Variables legitimately supplied from outside the script: documented env
# overrides in its header, plus the TG_* settings it exports to the pipelines.
EXTERNAL = {
    "FAMILIES", "MODE", "RUN_DATE", "STALE_DAYS", "STAT_MODEL",
    "TG_CADENCE", "TG_DATA_PATH", "TG_DATE_COL", "TG_HORIZON", "TG_TARGET",
    "TG_FAMILY", "TG_MODEL_FILTER", "TG_OUT_ROOT", "TG_PARAM_OVERRIDES",
    # Item 1e: optional pin. When set, the script and every runner refuse to
    # proceed unless the input file's SHA-256 matches, so "the same run" means the
    # same bytes rather than the same filename. Documented in the script header.
    "AI4CM_EXPECTED_DATA_SHA256",
    "HOME", "PATH", "PWD", "IFS", "OSTYPE", "BASH_SOURCE",
}

READ_RE = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")
ASSIGN_RE = re.compile(r"^\s*(?:export\s+|local\s+)?([A-Z_][A-Z0-9_]*)=", re.MULTILINE)


def test_every_referenced_variable_is_assigned_or_external():
    text = SCRIPT.read_text()
    assigned = set(ASSIGN_RE.findall(text))
    referenced = set(READ_RE.findall(text))
    unbound = referenced - assigned - EXTERNAL
    assert not unbound, (
        f"scripts/run_daily_forecast.sh reads variable(s) that are never "
        f"assigned: {sorted(unbound)}. Either assign them or add them to "
        f"EXTERNAL in this test if they are documented env inputs."
    )


def test_backtest_block_uses_the_run_directory():
    """The backtest report must be generated into the run folder."""
    text = SCRIPT.read_text()
    assert 'backtest_report.py" --run-dir "$RUN_DIR"' in text


# ---------------------------------------------------------------------------
# Selection must stay inside the selectable window, and one family's abort must
# not take another family down with it.
# ---------------------------------------------------------------------------

def _overrides_for(family: str) -> str:
    """The `overrides=` literal from a family's branch of the case statement."""
    text = SCRIPT.read_text()
    branch = re.search(rf"^\s*{re.escape(family)}\)(.*?)^\s*;;", text,
                       re.MULTILINE | re.DOTALL)
    assert branch, f"{family} has no branch in run_daily_forecast.sh"
    found = re.search(r"overrides='([^']*)'", branch.group(1))
    assert found, f"{family}'s branch sets no overrides"
    return found.group(1)


def test_b_ml_bounds_evaluation_to_the_selectable_window():
    """B_ML crowns a champion, so its folds must not reach the sealed holdout.

    With no bound, ``build_yearly_folds`` folds to the end of the data. Every row past
    2024-12-31 is then TEST or LIVE, and ``assert_selection_free`` -- which sits immediately
    before ``select_best_model`` -- refuses the whole run:

        SelectionOnReportOnlyDataError: b_ml_pipeline.select_best_model('Revenues', h=5):
        refusing to select on report-only data. Rows fall in test (n=3276, from 2025-01-01).

    The bound is compared against the real window constant rather than a literal, so moving
    the seal moves this test with it.
    """
    import json
    import sys
    sys.path.insert(0, str(SCRIPT.resolve().parents[1] / "backend"))
    from evaluation_windows import TEST_START

    overrides = json.loads(_overrides_for("B_ML"))
    assert "eval_end" in overrides, (
        "B_ML sets no eval_end, so its folds run to the end of the data and "
        "select_best_model refuses the run")
    assert overrides["eval_end"] < TEST_START, (
        f"B_ML evaluates to {overrides['eval_end']}, which is not before the sealed "
        f"holdout at {TEST_START}")


def test_e_quantile_selects_only_inside_the_dev_window():
    """E_QUANTILE chooses a best model, so its evaluation window must sit inside DEV.

    Its override used to be ``{"eval_start": "2025-01-01"}`` with no end: a window aimed into
    the sealed holdout, which ``assert_selection_free`` refused on every daily run
    (2026-09-29 audit, open decision; resolved 2026-10-01 as Option A). The window is now
    ``DEV.start .. DEV.end``, which on the business-day index gives 262 evaluation points in
    53 five-day blocks, clearing the ~150 the script's comment asks for. Both bounds are
    compared against the window constants, so moving the split moves this test with it.
    """
    import json
    import sys
    sys.path.insert(0, str(SCRIPT.resolve().parents[1] / "backend"))
    from evaluation_windows import DEV, TEST_START

    overrides = json.loads(_overrides_for("E_QUANTILE"))
    assert "eval_start" in overrides and "eval_end" in overrides, (
        "E_QUANTILE must bound its evaluation at both ends; a missing end runs to the end of "
        "the data, a missing start lets the tiler fold wherever it likes")
    assert overrides["eval_start"] >= DEV.start, (
        f"E_QUANTILE starts evaluating at {overrides['eval_start']}, before DEV ({DEV.start})")
    assert overrides["eval_end"] <= DEV.end < TEST_START, (
        f"E_QUANTILE evaluates to {overrides['eval_end']}, which reaches past DEV ({DEV.end}) "
        f"toward the sealed holdout at {TEST_START}")


def test_a_later_family_is_not_blocked_by_an_earlier_abort():
    """The script runs under ``set -e``, so family order decides what a failure costs.

    E_QUANTILE aborted on every default run until its window was bounded to DEV
    (``test_e_quantile_selects_only_inside_the_dev_window``). While it sat ahead of C_DL in
    the default order, that abort also cost the C_DL run, which is unrelated to it. The order
    is kept: any family's abort should cost only the families after it, and the one with an
    open history of aborting belongs last.
    """
    text = SCRIPT.read_text()
    default = re.search(r'FAMILIES="\$\{FAMILIES:-([^}"]*)\}"', text)
    assert default, "the FAMILIES default is no longer readable"
    order = default.group(1).split()
    assert order.index("C_DL") < order.index("E_QUANTILE"), (
        f"default FAMILIES order is {order}: an E_QUANTILE abort under `set -e` stops "
        f"C_DL from running at all")
