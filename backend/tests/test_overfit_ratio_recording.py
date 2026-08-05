"""M-4 (producer side): the pipeline itself must record memorisation as fatal.

test_b_ml_overfitting.py covers ``select_best_model`` by handing it a ratio
dictionary directly — including ``float("inf")``.  That tests the *consumer*.
Nothing tests the *producer*: the pipeline code that decides a ratio is
infinite in the first place, at b_ml_pipeline.py ~751-756::

    if not np.isnan(ratio):
        overfit_ratios[model_name] = ratio if prev is None else max(prev, ratio)
    if train_mae_fold == 0.0:
        # Zero training error is exact memorisation, not a good fit.
        overfit_ratios[model_name] = float("inf")

That second branch matters because ``ratio = val/train`` is ``NaN`` when
``train_mae_fold == 0`` — the division is guarded — so without the explicit
zero check a perfectly memorising model would record *no ratio at all* and sail
through the gate as "no evidence of overfitting".  ExtraTrees hit exactly this
on the real Revenues run with ``train_MAE=0.00``.

The capacity floors added by M-4 mean no shipped model can reach zero training
error any more, so this test injects a deliberate memoriser
(1-nearest-neighbour, which reproduces its training targets exactly) to keep
the safety net itself under test.  It also checks the evidence is *published*,
because a gate whose reasoning never reaches the artifact cannot be audited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import b_ml_pipeline  # noqa: E402
from b_ml_pipeline import (  # noqa: E402
    OVERFIT_GATE_RATIO,
    ConfigBML,
    available_models,
    run_pipeline_ml,
)

HORIZON = 5
MEMORISER = "Memoriser"


def _flow_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-01", "2023-08-06")
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 5.0)
    values = 7.0e7 * weekly * rng.normal(1.0, 0.12, size=len(dates))
    pd.DataFrame({"date": dates, "Revenues": np.maximum(values, 0.0)}).to_csv(
        path, index=False
    )


@pytest.fixture(scope="module")
def report_with_memoriser(tmp_path_factory):
    """Run B_ML with a perfect memoriser alongside an honest model."""
    tmp_path = tmp_path_factory.mktemp("memoriser")
    csv = tmp_path / "s.csv"
    _flow_csv(csv)
    out = tmp_path / "out"

    honest = available_models()["Ridge"]
    original = b_ml_pipeline.available_models
    # 1-NN predicts its own training targets exactly -> train_MAE == 0.0
    b_ml_pipeline.available_models = lambda: {
        MEMORISER: KNeighborsRegressor(n_neighbors=1),
        "Ridge": honest,
    }
    try:
        run_pipeline_ml(ConfigBML(
            data_path=str(csv), date_col="date", target="Revenues", cadence="Daily",
            horizon=HORIZON, variant="uni", model_filter=None,
            out_root=str(out), folds=1, min_train_years=4,
        ))
    finally:
        b_ml_pipeline.available_models = original

    return json.loads((out / "artifacts" / "integrity_report.json").read_text())


def test_overfit_evidence_is_published(report_with_memoriser):
    """A gate whose reasoning never reaches the artifact cannot be audited."""
    r = report_with_memoriser
    assert "overfit_ratios" in r, "the report carries no generalisation evidence"
    assert "overfit_excluded_models" in r
    assert r.get("overfit_gate_ratio") == OVERFIT_GATE_RATIO
    assert set(r["overfit_ratios"]) == {MEMORISER, "Ridge"}, r["overfit_ratios"]


def test_zero_train_mae_is_recorded_as_infinite_not_missing(report_with_memoriser):
    """The zero-error branch must fire; a missing ratio would read as innocent.

    ``float('inf')`` is not JSON-representable, so the pipeline serialises it as
    ``null`` (b_ml_pipeline.py ~1003-1006).  ``null`` here means "infinite",
    which is why the *absence* of the key would be the dangerous outcome.
    """
    ratios = report_with_memoriser["overfit_ratios"]
    assert MEMORISER in ratios, (
        "a model with train_MAE == 0 recorded no ratio at all — the explicit "
        "zero-error branch is gone and memorisation now looks like clean data"
    )
    assert ratios[MEMORISER] is None, (
        f"expected the memoriser's ratio to be infinite (serialised as null), "
        f"got {ratios[MEMORISER]!r}"
    )


def test_honest_model_keeps_a_finite_ratio(report_with_memoriser):
    """The zero check must not smear across models."""
    ridge = report_with_memoriser["overfit_ratios"]["Ridge"]
    assert ridge is not None and np.isfinite(ridge), (
        f"Ridge should have a finite val/train ratio, got {ridge!r}"
    )
    assert ridge > 0


def test_memoriser_is_excluded_from_selection(report_with_memoriser):
    r = report_with_memoriser
    assert MEMORISER in r["overfit_excluded_models"], (
        f"the memoriser was not excluded: {r['overfit_excluded_models']}"
    )


def test_memoriser_is_never_crowned_best(report_with_memoriser):
    """The whole point: exact memorisation must not win on test MAE."""
    r = report_with_memoriser
    assert r["best_model"] != MEMORISER, (
        "a model that memorised every training row was crowned best"
    )
    assert r["best_model"] == "Ridge", r["best_model"]


def test_published_best_model_is_not_an_excluded_model(report_with_memoriser):
    """Invariant that must hold on every run, not just this one."""
    r = report_with_memoriser
    assert r["best_model"] not in r["overfit_excluded_models"], (
        f"best_model {r['best_model']!r} is also in overfit_excluded_models "
        f"{r['overfit_excluded_models']!r} — the gate was computed but not applied"
    )
