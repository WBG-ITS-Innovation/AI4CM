"""The sentinel's probe is swappable, and swapping it must not change the default.

Written for the probe study (reports/sentinel_probe_study.md). The point of these tests is
narrow: the instrument can be varied for study purposes, the default is unchanged, and the
probe that produced a number is always recorded alongside it -- so a ratio measured with one
instrument can never be silently compared against a threshold calibrated for another.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import (  # noqa: E402
    DEFAULT_PROBE,
    MIN_SIGNAL_RATIO,
    PROBE_FOREST,
    PROBE_RIDGE,
    PROBE_TREE,
    signal_sentinel,
)

N, P = 900, 8


def _split(X, y):
    k = int(N * 0.8)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def _null():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(0, 1, (N, P)), columns=[f"f{i}" for i in range(P)])
    return X, pd.Series(rng.normal(0, 1, N))


def _interaction_only():
    """Signal carried purely by an interaction, so a linear probe cannot see it."""
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(0, 1, (N, P)), columns=[f"f{i}" for i in range(P)])
    y = 3.0 * ((X["f0"] > 0).astype(float) * (X["f1"] > 0).astype(float)) \
        + rng.normal(0, 0.3, N)
    return X, pd.Series(y)


def test_default_probe_is_still_ridge():
    """The 1.50 threshold is calibrated to the ridge probe.

    Changing the default would silently change what every historical ratio meant, so it is
    pinned here. Any future change must be a deliberate decision with its own threshold
    review, not a side effect.
    """
    assert DEFAULT_PROBE == PROBE_RIDGE


def test_omitting_probe_matches_explicit_ridge():
    X, y = _interaction_only()
    a = signal_sentinel(*_split(X, y), horizon=5)
    b = signal_sentinel(*_split(X, y), horizon=5, probe=PROBE_RIDGE)
    assert a["shuffled_to_normal_ratio"] == b["shuffled_to_normal_ratio"]


@pytest.mark.parametrize("probe", [PROBE_RIDGE, PROBE_TREE, PROBE_FOREST])
def test_every_probe_reports_which_one_it_was(probe):
    X, y = _null()
    out = signal_sentinel(*_split(X, y), horizon=5, probe=probe)
    assert out["probe"] == probe


def test_unknown_probe_is_rejected():
    X, y = _null()
    with pytest.raises(ValueError, match="unknown sentinel probe"):
        signal_sentinel(*_split(X, y), horizon=5, probe="magic")


@pytest.mark.parametrize("probe", [PROBE_RIDGE, PROBE_TREE, PROBE_FOREST])
def test_no_probe_finds_signal_in_noise(probe):
    """The null control. A probe that fires here is useless as an instrument."""
    X, y = _null()
    out = signal_sentinel(*_split(X, y), horizon=5, probe=probe)
    assert out["shuffled_to_normal_ratio"] < MIN_SIGNAL_RATIO
    assert out["signal_detected"] is False


def test_ridge_misses_interaction_only_signal_and_trees_do_not():
    """The study's central finding, pinned as a regression test.

    A pure two-way interaction carries strong signal and zero linear signal. The ridge
    probe misses it; both tree probes catch it. This is a real blind spot in the
    instrument, and it is the reason the probe study exists.
    """
    X, y = _interaction_only()
    parts = _split(X, y)
    ridge = signal_sentinel(*parts, horizon=5, probe=PROBE_RIDGE)
    tree = signal_sentinel(*parts, horizon=5, probe=PROBE_TREE)
    forest = signal_sentinel(*parts, horizon=5, probe=PROBE_FOREST)

    assert ridge["signal_detected"] is False, (
        f"ridge unexpectedly detected interaction signal "
        f"(ratio {ridge['shuffled_to_normal_ratio']:.3f})"
    )
    assert tree["signal_detected"] is True
    assert forest["signal_detected"] is True
    assert tree["shuffled_to_normal_ratio"] > ridge["shuffled_to_normal_ratio"]


def test_probes_are_deterministic():
    """A study comparing instruments needs each instrument to be reproducible."""
    X, y = _interaction_only()
    parts = _split(X, y)
    for probe in (PROBE_RIDGE, PROBE_TREE, PROBE_FOREST):
        a = signal_sentinel(*parts, horizon=5, probe=probe)["shuffled_to_normal_ratio"]
        b = signal_sentinel(*parts, horizon=5, probe=probe)["shuffled_to_normal_ratio"]
        assert a == b, f"{probe} is not deterministic"
