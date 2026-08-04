"""M-5 (wiring): the shuffled-target sentinel must be scored on held-out rows.

The sentinel's *arithmetic* is covered by test_signal_sentinel_semantics.py.
What was never covered is how ``b_ml_pipeline`` **calls** it — and that call site
is where the original defect lived.

The pre-M-5 pipeline built the sentinel's slices as::

    X_train_leak = f_train_sample.iloc[:500]     # first 500 rows
    X_test_leak  = f_train_sample.iloc[-100:]    # last 100 rows of the SAME frame

so the "held-out" evaluation rows sat adjacent to — and on short series
overlapped — the sentinel's own training rows.  The ratio it produced was
therefore not a held-out measurement at all, which is half of why the real
Revenues run reported a nonsense 0.83.

The fix (b_ml_pipeline.py ~1054-1075) splits by position and inserts a
horizon-sized embargo, so no training row's target can reach into the
evaluation slice.

This test spies on the real call rather than inspecting source text, so it
survives refactors but still fails if the embargo or the disjointness is ever
removed.

Both a long and a short history are exercised, and the short case is the one
that matters: with a few thousand training rows the old ``iloc[:500]`` /
``iloc[-100:]`` slices happen to be disjoint, so a long-history fixture alone
cannot detect the original bug.  It only bites when the training frame is
smaller than 600 rows — which is exactly the Demo/Balanced profile a user
reaches for first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import forecast_integrity as integrity_mod  # noqa: E402
from b_ml_pipeline import ConfigBML, run_pipeline_ml  # noqa: E402

HORIZON = 5

# (label, first_date, last_date, min_train_years) — "short" keeps the sentinel's
# training frame under 600 rows, the regime where the pre-M-5 slices overlapped.
HISTORIES = [
    ("long", "2016-01-01", "2023-08-06", 4),
    ("short", "2021-01-01", "2023-06-30", 1),
]


def _flow_csv(path: Path, first: str, last: str) -> None:
    """Business-day flow data with a mild weekly shape."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(first, last)
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 5.0)
    values = 7.0e7 * weekly * rng.normal(1.0, 0.12, size=len(dates))
    pd.DataFrame({"date": dates, "Revenues": np.maximum(values, 0.0)}).to_csv(
        path, index=False
    )


def _run_capturing_sentinel(tmp_path: Path, history: tuple) -> list[dict]:
    """Run B_ML with the sentinel replaced by a spy that records its inputs."""
    label, first, last, min_years = history
    csv = tmp_path / f"{label}.csv"
    _flow_csv(csv, first, last)

    # Patch on forecast_integrity: item 1c moved signal_sentinel there and
    # b_ml_pipeline imports it from that module, so patching the deprecated
    # preprocessing.integrity shim would no longer intercept the call.
    calls: list[dict] = []
    real = integrity_mod.signal_sentinel

    def spy(X_train, y_train, X_test, y_test, horizon):
        calls.append({
            "label": label,
            "train_idx": list(X_train.index),
            "eval_idx": list(X_test.index),
            "horizon": horizon,
            "n_train": len(X_train),
            "n_eval": len(X_test),
        })
        return real(X_train, y_train, X_test, y_test, horizon)

    integrity_mod.signal_sentinel = spy
    try:
        run_pipeline_ml(ConfigBML(
            data_path=str(csv), date_col="date", target="Revenues",
            cadence="Daily", horizon=HORIZON, variant="uni", model_filter="Ridge",
            out_root=str(tmp_path / "out"), folds=1, min_train_years=min_years,
        ))
    finally:
        integrity_mod.signal_sentinel = real
    return calls


@pytest.fixture(params=HISTORIES, ids=[h[0] for h in HISTORIES])
def sentinel_calls(request, tmp_path):
    return _run_capturing_sentinel(tmp_path, request.param)


def test_sentinel_is_actually_invoked(sentinel_calls):
    """A silently skipped sentinel is indistinguishable from a passing one."""
    assert sentinel_calls, "signal_sentinel was never called — the check did not run"


def test_sentinel_train_and_eval_rows_are_disjoint(sentinel_calls):
    """The evaluation slice must not contain any row the sentinel trained on."""
    for call in sentinel_calls:
        overlap = set(call["train_idx"]) & set(call["eval_idx"])
        assert not overlap, (
            f"[{call['label']}] {len(overlap)} row(s) appear in both the "
            f"sentinel's training and evaluation slices "
            f"(n_train={call['n_train']}, n_eval={call['n_eval']}) — "
            f"the ratio is not a held-out measurement"
        )


def test_sentinel_split_has_a_horizon_sized_embargo(sentinel_calls):
    """A gap of >= horizon keeps training targets out of the evaluation slice.

    Without it, the last (h-1) training rows have targets dated at or after the
    first evaluation row, so the sentinel is partly scored on data it was
    fitted through.
    """
    for call in sentinel_calls:
        h = int(call["horizon"])
        last_train = max(call["train_idx"])
        first_eval = min(call["eval_idx"])
        gap_steps = len(pd.bdate_range(last_train, first_eval)) - 1
        assert gap_steps >= h, (
            f"[{call['label']}] only {gap_steps} step(s) between the sentinel's "
            f"last training row ({last_train.date()}) and its first evaluation "
            f"row ({first_eval.date()}); need >= horizon ({h})"
        )


def test_sentinel_evaluation_slice_is_the_recent_tail(sentinel_calls):
    """Scoring must use the most recent rows, not an arbitrary interior block.

    Signal decays; measuring it on 2016 rows while forecasting 2023 would be
    the same class of mistake in a different disguise.
    """
    for call in sentinel_calls:
        assert min(call["eval_idx"]) > max(call["train_idx"]), (
            f"[{call['label']}] the sentinel's evaluation slice must come "
            f"strictly after its training slice in time"
        )
        assert call["n_eval"] >= 5, (
            f"[{call['label']}] evaluation slice too small ({call['n_eval']})"
        )
