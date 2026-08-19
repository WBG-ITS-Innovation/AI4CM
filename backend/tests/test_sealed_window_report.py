"""The sealed-window champion harness: causal, logged, selection-free, and honest about itself.

Why a new harness needed testing at all
---------------------------------------
Nothing could measure a champion on the holdout. ``b_ml_pipeline`` has the window but not
``target_transform`` — so its ``LightGBM_L1`` is not the Revenues champion, which is *defined* by
ratio-to-trailing-level scaling. ``ws2_tune.design`` has the recipe but its ``make_folds`` is a
selection path that refuses TEST. This harness borrows the recipe and rewrites the fold
construction as reporting.

Three properties carry the weight: the training data for a fold must predate that fold's first
origin (the borrowed geometry does NOT embargo, and leaks 5 rows), the holdout read must reach the
ledger, and nothing may be selected.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

from evaluation_windows import DEV, TEST_END, TEST_START, window_for      # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="canonical data not present")
TARGETS = ["Revenues", "Expenditure", "State budget balance"]


@pytest.fixture(scope="module")
def folds_revenues():
    import sealed_window_report as swr
    return swr.sealed_folds("Revenues", log=False)


# ── causality: the embargo the borrowed geometry lacks ───────────────────────

@needs_data
def test_no_training_target_lands_inside_the_evaluation_block(folds_revenues):
    """The property the harness exists to guarantee.

    ``build_yearly_folds`` puts ``train_end`` at the last business day before the block, and
    ``design()`` shifts targets by -H over the whole series, so the last H training origins carry
    answers from inside the block. Training on them fits the model to the window it is scored on.
    """
    import sealed_window_report as swr
    from ws2_tune import design

    folds, _ = folds_revenues
    assert folds, "no folds built"
    s, *_ = design("Revenues")
    tmap = swr._target_date_map(s.index, swr.HORIZON)
    for f in folds:
        latest = max(tmap[d] for d in f.X_tr.index if d in tmap)
        assert latest < f.origin_dates.min(), (
            f"a training target ({latest.date()}) lands at or after the first evaluation origin "
            f"({f.origin_dates.min().date()})")


@needs_data
def test_the_embargo_actually_removes_rows(folds_revenues):
    """If it removed none, either the geometry changed or the embargo stopped working."""
    folds, meta = folds_revenues
    assert meta["n_embargoed"] > 0, (
        "the borrowed geometry leaks the last H training origins; an embargo that drops nothing "
        "is not embargoing")
    assert meta["n_embargoed"] <= 2 * swr_horizon(), (
        "far more rows were dropped than the horizon explains — check the fold geometry")


def swr_horizon():
    import sealed_window_report as swr
    return swr.HORIZON


@needs_data
def test_the_borrowed_selection_path_still_lacks_the_embargo():
    """Recorded as a finding, not silently inherited.

    ``ws2_tune.make_folds`` is the tuner's own fold builder and it does not embargo, so its DEV
    folds train on 5 rows whose targets fall inside the DEV block. Changing a selection path is
    out of scope for a reporting harness; this test makes the gap visible so it cannot be
    forgotten, and will fail loudly if someone fixes it (at which point delete this test).
    """
    import sealed_window_report as swr
    from ws2_tune import design, make_folds

    s, X, y_t, *_ = design("Revenues")
    tmap = swr._target_date_map(s.index, swr.HORIZON)
    dev_folds, _ = make_folds("Revenues", "dev")
    f = dev_folds[0]
    latest = max(tmap[d] for d in f.X_tr.index if d in tmap)
    first = f.X_te.index.min()
    assert latest >= first, (
        "ws2_tune.make_folds now embargoes — good. Delete this test and the note in "
        "sealed_window_report's docstring.")


# ── the evaluation window is the sealed one, and it is logged ────────────────

@needs_data
def test_the_evaluated_rows_are_holdout_rows(folds_revenues):
    folds, _ = folds_revenues
    tds = pd.DatetimeIndex([d for f in folds for d in f.target_dates])
    assert len(tds) > 100
    assert {window_for(d) for d in tds} == {"test"}, (
        f"expected only holdout target dates, got {sorted({window_for(d) for d in tds})}")
    assert tds.min() >= pd.Timestamp(TEST_START)
    assert tds.max() <= pd.Timestamp(TEST_END)


@needs_data
def test_the_holdout_read_is_logged_as_a_report(monkeypatch):
    import evaluation_windows as ew
    import sealed_window_report as swr

    calls = []
    # Patched on `swr`, not on `evaluation_windows`: the module does
    # `from evaluation_windows import require_test_access`, so the name is bound in ITS namespace
    # and patching the source module would not be seen.
    monkeypatch.setattr(swr, "require_test_access",
                        lambda reason, caller=None, purpose=ew.PURPOSE_SELECTION:
                        calls.append({"reason": reason, "caller": caller, "purpose": purpose}))
    swr.sealed_folds("Revenues", log=True)

    assert calls, "the holdout was evaluated and the ledger was not told"
    assert any(c["purpose"] == ew.PURPOSE_REPORT for c in calls)
    assert any("Champion sealed-window evaluation" in (c["reason"] or "") for c in calls)
    assert any(c["caller"] == "sealed_window_report.sealed_folds" for c in calls)


@needs_data
def test_a_dev_scoped_call_no_longer_reads_holdout_targets(monkeypatch):
    """This test is the history of the finding, kept because the history is the point.

    It first asserted "a DEV-scoped call logs no holdout read", and FAILING was the finding: folds
    are origin-bounded while truth is read at ``origin + H``, so 4 DEV origins in late December
    2024 were scored against early-January 2025 — inside the sealed holdout. It was then rewritten
    to assert the leak's existence. Now that both harnesses require the target date to be in an
    allowed window, it asserts the fixed state: no holdout target, and nothing for the ledger to
    record.
    """
    import evaluation_windows as ew
    import sealed_window_report as swr

    calls = []
    monkeypatch.setattr(swr, "require_test_access",
                        lambda *a, **k: calls.append(k.get("purpose")))
    folds, _ = swr.sealed_folds("Revenues", eval_start=DEV.start, eval_end=DEV.end, log=True)

    tds = pd.DatetimeIndex([d for f in folds for d in f.target_dates])
    assert len(tds) > 200, "the DEV fold should still be substantial after the fix"
    assert {window_for(d) for d in tds} == {"dev"}, (
        f"a DEV-scoped call must read DEV truth only; got "
        f"{sorted({window_for(d) for d in tds})}")
    assert not calls, f"nothing holdout was read, so the ledger must stay quiet; got {calls}"


# The pin `test_the_tuners_dev_fold_is_scored_against_holdout_rows` lived here and has been
# DELETED, per its own assertion message, because the leak it recorded is fixed: ws2_tune's folds
# now require an evaluation row's TARGET date to be in an allowed window, not just its origin. The
# permanent replacement is `test_no_fold_reads_holdout_truth.py`, which asserts the property for
# every fold builder rather than documenting one instance of its absence.


# ── recipe fidelity: it must be the champion, not a lookalike ───────────────

@needs_data
@pytest.mark.parametrize("target", TARGETS)
def test_the_estimator_matches_the_recipes_declared_parameters(target):
    """The specific error this module exists to avoid: a different model under the champion's name."""
    import sealed_window_report as swr
    name, est, rec = swr.champion_estimator(target)
    assert name == rec["point_model"]
    declared = {k: v for k, v in (rec.get("params") or {}).items() if v is not None}
    actual = est.get_params()
    for k, v in declared.items():
        if k in actual:
            assert actual[k] == v, f"{target}: {k} pipeline={actual[k]!r} registry={v!r}"


@needs_data
def test_the_ratio_transform_is_applied_for_the_recipe_that_declares_it():
    """Revenues is credentialed WITH ratio scaling; measuring it without is a different model."""
    import sealed_window_report as swr
    _, meta_rev = swr.sealed_folds("Revenues", log=False)
    _, meta_exp = swr.sealed_folds("Expenditure", log=False)
    assert meta_rev["transform"] == "ratio"
    assert meta_exp["transform"] in ("raw", None)

    folds, _ = swr.sealed_folds("Revenues", log=False)
    assert folds[0].inverse is not None, (
        "a ratio recipe must carry an inverse, or predictions stay in ratio space and the MAE is "
        "off by the divisor's order of magnitude")


@needs_data
def test_predictions_are_returned_in_original_units():
    """A ratio-space MAE and a lari MAE differ by ~7 orders of magnitude."""
    import sealed_window_report as swr
    preds, _ = swr.evaluate_champion("Revenues", log=False)
    assert len(preds) > 100
    assert preds["y_pred"].abs().median() > 1e6, (
        "predictions look like ratios, not currency — the inverse was not applied")
    mae = float(np.mean(np.abs(preds["y_true"] - preds["y_pred"])))
    assert 1e6 < mae < 1e9, f"MAE {mae:,.0f} is not on the scale of this series"


# ── it reports; it never selects ────────────────────────────────────────────

def test_the_module_never_selects():
    """Checked on EXECUTABLE code, with docstrings stripped.

    The first version scanned raw text and tripped on its own prose ("never crowns anything"),
    which is the classic way a source-text assertion becomes noise. Stripping docstrings keeps it
    checking behaviour instead of vocabulary.
    """
    import ast

    src = (BACKEND / "sealed_window_report.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for token in ("select_best_model", "argmin", "argmax", "idxmin", "idxmax",
                  "assert_selection_free"):
        assert token not in code, (
            f"a reporting harness must not call {token}; selection belongs to train/dev paths")


@needs_data
def test_it_states_the_gap_to_the_logged_credential():
    """A sealed-window figure must not be presented as continuous with credentials it cannot
    reproduce -- the producing script is absent from the repository."""
    import sealed_window_report as swr
    d = swr.dev_reconstruction("Revenues")
    assert d["recomputed_dev_mae"] is not None
    assert d["logged_dev_mae"] > 0
    assert d["delta_pct"] > 0, "if this is now zero the credential became reproducible; say so"
    assert "not reproducible" in d["note"] or "reconstruction" in d["note"]
