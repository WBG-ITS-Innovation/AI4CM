"""Regression test for model-audit finding C-1: C_DL target scaling.

Before the fix, the C_DL neural nets trained on a raw revenue target (~1e8)
while their inputs were standardized to ~unit scale. The networks stayed pinned
near their ~0 initialization, so predictions collapsed to ~hundreds against
actuals of ~1e8 — i.e. every model was far worse than a do-nothing baseline.

This test drives the pipeline exactly the way the runner does (a flow target,
no explicit target_transform, so the "auto" default decides) on a synthetic,
spiky, nonnegative series that contains true zeros. It asserts the predictions
come back on the right order of magnitude.

It FAILS on the old behavior (default "none" -> identity -> predictions ~0) and
PASSES with the fix (default "auto" -> log1p + train-only standardization).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import c_dl_pipeline as pipe
from c_dl_pipeline import ConfigDL


def _make_spiky_flow_series(csv_path: Path) -> float:
    """Write a synthetic daily flow series and return its median level.

    The series is nonnegative, has a weekly rhythm, occasional true zeros
    (holidays), and large end-of-month spikes — the same shape that broke the
    real Revenues run.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", "2022-12-31", freq="B")
    base = 7.0e7  # ~70 million, same order of magnitude as real revenue
    dow = dates.dayofweek.to_numpy()
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dow / 5.0)
    noise = rng.normal(1.0, 0.12, size=len(dates))
    values = base * weekly * noise
    # End-of-month spikes (revenue lumps on the last business day of a month).
    is_eom = pd.Series(dates).dt.is_month_end.to_numpy()
    values[is_eom] *= 2.2
    # True zeros on ~3% of days (public holidays with no receipts).
    zero_mask = rng.random(len(dates)) < 0.03
    values[zero_mask] = 0.0
    values = np.maximum(values, 0.0)

    pd.DataFrame({"date": dates, "Revenues": values}).to_csv(csv_path, index=False)
    return float(np.median(values))


def _build_config(data_path: Path, out_root: Path, target_transform=None) -> ConfigDL:
    kwargs = dict(
        data_path=str(data_path),
        date_col="date",
        holidays_csv=None,
        out_root_uni=str(out_root),
        targets=["Revenues"],          # a flow target (is_stock == False)
        cadences=["daily"],
        seq_len_daily=16,
        epochs=3,
        batch_size=64,
        valid_frac=0.1,
        conformal_calib_frac=0.2,
        min_train_years=2,
        device="cpu",                  # deterministic, no GPU dependency
        quick_mode=True,
    )
    # When target_transform is None we leave the ConfigDL default in place, so
    # the test also locks in that the default routes flows to scaling.
    if target_transform is not None:
        kwargs["target_transform"] = target_transform
    cfg = ConfigDL(**kwargs)
    cfg.horizons_daily = [5]
    cfg.models_univariate = ["mlp"]    # one cheap model is enough to expose the bug
    return cfg


def _run_and_get_pred_median(cfg: ConfigDL, out_root: Path) -> float:
    pipe.run_pipeline(config=cfg, run_univariate=True, run_multivariate=False)
    pred_csv = out_root / "daily" / "predictions_long.csv"
    assert pred_csv.exists(), f"pipeline produced no predictions at {pred_csv}"
    preds = pd.read_csv(pred_csv)
    assert len(preds) > 0, "predictions_long.csv is empty"
    return float(np.median(np.abs(preds["y_pred"].values)))


def test_c_dl_predictions_are_on_target_scale(tmp_path):
    """Default (auto) path: prediction magnitude must be sane vs the target."""
    data_path = tmp_path / "synthetic_flow.csv"
    target_median = _make_spiky_flow_series(data_path)

    out_root = tmp_path / "out_default"
    # No target_transform passed -> ConfigDL default ("auto") applies, exactly as
    # the runner behaves. On the OLD default ("none") this collapses and fails.
    cfg = _build_config(data_path, out_root)
    pred_median = _run_and_get_pred_median(cfg, out_root)

    lo, hi = 0.1 * target_median, 10.0 * target_median
    assert lo <= pred_median <= hi, (
        f"C_DL predictions are off-scale: median |y_pred|={pred_median:,.0f} "
        f"is outside [{lo:,.0f}, {hi:,.0f}] around target median "
        f"{target_median:,.0f}. This is the audit C-1 collapse."
    )


def test_c_dl_none_transform_still_collapses(tmp_path):
    """Guardrail: with target_transform='none' the old collapse still happens.

    This documents the bug and proves the assertion above is meaningful — the
    same magnitude check fails when scaling is disabled.
    """
    data_path = tmp_path / "synthetic_flow.csv"
    target_median = _make_spiky_flow_series(data_path)

    out_root = tmp_path / "out_none"
    cfg = _build_config(data_path, out_root, target_transform="none")
    pred_median = _run_and_get_pred_median(cfg, out_root)

    # Raw ~1e8 target -> predictions stay near ~0, orders of magnitude too small.
    assert pred_median < 0.1 * target_median, (
        f"expected the unscaled path to collapse, but median |y_pred|="
        f"{pred_median:,.0f} was not << target median {target_median:,.0f}"
    )
