"""C-2 (publication): the persistence number that ships must be the shared one.

test_unified_baseline.py proves each family *imports*
``forecast_integrity.compute_persistence_baseline`` and no longer inlines its
own copy.  That is necessary but not sufficient, because it checks source text
rather than the artifact.

Two live implementations of h-step persistence exist:

  * ``forecast_integrity.compute_persistence_baseline``            (the shared ruler)
  * ``preprocessing.integrity.compute_persistence_baseline_from_origin``
    (reached via ``compute_integrity_report``)

and in ``b_ml_pipeline`` the second one wins.  The pipeline computes the shared
value, then does::

    legacy_report = compute_integrity_report(...)
    integrity_report.update(legacy_report)      # <-- overwrites mae_persistence

so the number that reaches ``artifacts/integrity_report.json`` — and therefore
the Dashboard, the daily summary and the backtest report — comes from the
duplicate.  Today the two agree to the cent, which is exactly why a divergence
would go unnoticed.

These tests assert the published artifact equals the shared function's value,
and pin the two implementations to each other, so any future drift or any
reordering of that merge fails loudly instead of silently changing the ruler
every downstream number is measured against.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import compute_persistence_baseline  # noqa: E402

HORIZON = 5


def _flow_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-01", "2023-08-06")
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 5.0)
    values = 7.0e7 * weekly * rng.normal(1.0, 0.12, size=len(dates))
    values[rng.random(len(dates)) < 0.03] = 0.0        # true holiday zeros
    pd.DataFrame({"date": dates, "Revenues": np.maximum(values, 0.0)}).to_csv(
        path, index=False
    )


def _shared_value(preds: pd.DataFrame, model: str | None) -> float:
    if model and "model" in preds.columns and model in set(preds["model"]):
        preds = preds[preds["model"] == model]
    valid = preds.dropna(subset=["origin_value", "y_true"])
    return compute_persistence_baseline(valid)["mae_persistence"]


# ── there must be exactly ONE implementation ───────────────────────────────
#
# This file used to assert that the two copies of the persistence baseline returned
# the same number. Item 1c retired the duplicate, so the property to guard is no
# longer "they agree" but "there is only one". The tests below fail if a second
# implementation reappears -- which is how the original defect arose.

def test_the_duplicate_persistence_implementation_is_gone():
    """`preprocessing.integrity` must not resurrect its own copy."""
    import preprocessing.integrity as legacy

    for name in ("compute_persistence_baseline_from_origin",
                 "compute_baselines",
                 "compute_baseline_maes",
                 "shift_sanity_check",
                 "compute_integrity_report"):
        assert not hasattr(legacy, name), (
            f"preprocessing.integrity.{name} is back. It was retired in item 1c "
            f"because its output overwrote the shared module's via "
            f"integrity_report.update(legacy_report); use forecast_integrity instead."
        )


def test_the_deprecated_module_still_re_exports_the_sentinel():
    """Retiring the duplicates must not break the sentinel's existing importers."""
    import preprocessing.integrity as legacy
    import forecast_integrity as shared

    for name in ("signal_sentinel", "leakage_sentinel", "MIN_SIGNAL_RATIO"):
        assert hasattr(legacy, name), f"{name} vanished from the deprecated shim"
        assert getattr(legacy, name) is getattr(shared, name), (
            f"{name} is a COPY in the shim rather than a re-export -- that is a "
            f"second implementation by another route"
        )


def test_b_ml_no_longer_merges_a_legacy_report_over_the_shared_one():
    """The specific line that made the duplicate authoritative.

    `integrity_report.update(legacy_report)` overwrote mae_persistence, skill_pct,
    best_shift and the alignment fields with the duplicate's values, and the two
    disagreed on is_lag0_issue -- so the published report could assert both
    is_lag0_issue and is_persistence_like simultaneously (review §1.2, §1.4).
    """
    src = (BACKEND_DIR / "b_ml_pipeline.py").read_text()
    offending = [ln.strip() for ln in src.splitlines()
                 if "update(legacy_report)" in ln and not ln.strip().startswith("#")]
    assert not offending, f"the legacy merge is back: {offending}"
    assert "compute_integrity_report" not in src.replace(
        "# These used to come from preprocessing.integrity.compute_integrity_report,", ""
    ), "b_ml_pipeline calls the retired compute_integrity_report again"


# ── B_ML: the artifact must carry the shared value ────────────────────────

def test_b_ml_published_baseline_equals_the_shared_function(tmp_path):
    """Survives the compute_integrity_report merge that overwrites the field."""
    from b_ml_pipeline import ConfigBML, run_pipeline_ml

    csv = tmp_path / "s.csv"
    _flow_csv(csv)
    out = tmp_path / "out"
    run_pipeline_ml(ConfigBML(
        data_path=str(csv), date_col="date", target="Revenues", cadence="Daily",
        horizon=HORIZON, variant="uni", model_filter="Ridge",
        out_root=str(out), folds=1, min_train_years=4,
    ))

    report = json.loads((out / "artifacts" / "integrity_report.json").read_text())
    preds = pd.read_csv(out / "predictions_long.csv")
    expected = _shared_value(preds, report.get("best_model") or report.get("model"))

    assert "mae_persistence" in report, "the published report has no baseline at all"
    assert report["mae_persistence"] == expected, (
        f"published mae_persistence {report['mae_persistence']!r} != shared "
        f"function's {expected!r} — something between the shared call and the "
        f"artifact is substituting a different ruler"
    )


def test_b_ml_leaderboard_baseline_row_equals_the_shared_function(tmp_path):
    """The leaderboard reference line must use the same ruler as the report."""
    from b_ml_pipeline import ConfigBML, run_pipeline_ml

    csv = tmp_path / "s.csv"
    _flow_csv(csv)
    out = tmp_path / "out"
    run_pipeline_ml(ConfigBML(
        data_path=str(csv), date_col="date", target="Revenues", cadence="Daily",
        horizon=HORIZON, variant="uni", model_filter="Ridge",
        out_root=str(out), folds=1, min_train_years=4,
    ))

    lb = pd.read_csv(out / "leaderboard.csv")
    row = lb[lb["model"].astype(str).str.contains("ersistence", na=False)]
    assert len(row) == 1, f"expected exactly one persistence row, got {len(row)}"

    preds = pd.read_csv(out / "predictions_long.csv")
    valid = preds.dropna(subset=["origin_value", "y_true"])
    expected = compute_persistence_baseline(valid)["mae_persistence"]
    assert float(row["MAE"].iloc[0]) == expected, (
        f"leaderboard baseline {float(row['MAE'].iloc[0])!r} != shared "
        f"function's {expected!r}"
    )


# ── A_STAT: same guarantee on the other family that publishes it ──────────

def test_a_stat_published_baseline_equals_the_shared_function(tmp_path):
    csv = tmp_path / "s.csv"
    _flow_csv(csv)
    out = tmp_path / "a_stat"
    env = {
        "TG_MODEL_FILTER": "NAIVE", "TG_TARGET": "Revenues", "TG_CADENCE": "Daily",
        "TG_HORIZON": str(HORIZON), "TG_DATE_COL": "date",
        "TG_DATA_PATH": str(csv), "TG_OUT_ROOT": str(out),
        "TG_PARAM_OVERRIDES": '{"folds":1,"min_train_years":4}',
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        import run_a_stat
        importlib.reload(run_a_stat)
        run_a_stat.main()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    report = json.loads((out / "artifacts" / "integrity_report.json").read_text())
    preds = pd.read_csv(out / "predictions_long.csv")
    expected = _shared_value(preds, None)
    assert report["mae_persistence"] == expected, (
        f"A_STAT published {report['mae_persistence']!r} != shared function's "
        f"{expected!r}"
    )
