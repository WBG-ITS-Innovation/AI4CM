"""M-5 (semantics): leakage, no-signal and persistence-mimicry must stay distinct.

Three different things can make a forecast unusable, and they call for three
different responses:

  * **leakage** — the model can see the future.  Its error is implausibly
    SMALL, so the shuffled/real ratio goes UP.  Response: fix the features.
  * **no signal** — the features never predicted the target.  Shuffling the
    labels barely hurts, so the ratio sits near (or below) 1.  Response: find
    better features, or accept that the horizon is not forecastable.
  * **persistence-mimicry** — the model replays a recent actual.  It can post
    good skill and real signal while being useless for planning.  Response:
    compare against the persistence baseline and the shift diagnostic.

The pre-M-5 code collapsed the first two: a LOW ratio was reported as
``leakage_warning=true``.  That is the inversion this file exists to prevent.
On the real Revenues run it produced "Best model: WITHHELD — leakage flag
raised" at ratio 0.83, when 0.83 means the shuffled model *beat* the real one.
The withholding was right; the stated reason was the opposite of the truth, and
a wrong reason costs as much trust as a wrong number.

These tests pin the *direction* of the ratio and the *separation* of the three
gate reasons, so neither can be quietly re-conflated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preprocessing.integrity import MIN_SIGNAL_RATIO, signal_sentinel  # noqa: E402
from daily_summary import gate_reasons  # noqa: E402

HORIZON = 5
N = 400


def _series(seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 1e8)   # treasury-scale level
    return y, rng


def _split(X, y):
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    k = int(len(X) * 0.7)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


# ── direction of the ratio: the inversion must not come back ──────────────

def test_leakage_drives_the_ratio_UP_and_is_never_called_no_signal():
    """An oracle feature (literally the label) must not read as 'no signal'."""
    y, _ = _series()
    label = y.shift(-HORIZON)
    X = pd.DataFrame({"lag_1": y.shift(1), "oracle": label})
    r = signal_sentinel(*_split(X, label), horizon=HORIZON)

    assert r["shuffled_to_normal_ratio"] > MIN_SIGNAL_RATIO, (
        f"leakage must push the ratio well above {MIN_SIGNAL_RATIO}, "
        f"got {r['shuffled_to_normal_ratio']:.2f}"
    )
    assert r["signal_detected"] is True
    assert "NO SIGNAL" not in r["signal_verdict"].upper()


def test_no_signal_drives_the_ratio_DOWN_and_is_never_called_leakage():
    """Pure-noise features must not read as leakage — this is the M-5 inversion."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, N), "b": rng.normal(0, 1, N)})
    y = pd.Series(rng.normal(0, 1, N))
    r = signal_sentinel(*_split(X, y), horizon=HORIZON)

    assert r["shuffled_to_normal_ratio"] < MIN_SIGNAL_RATIO
    assert r["signal_detected"] is False
    assert r["leakage_warning"] is False, (
        "a low ratio means the features carry no signal; reporting it as "
        "leakage is the pre-M-5 inversion"
    )
    assert "SIGNAL" in r["signal_verdict"].upper()


def test_the_sentinel_never_asserts_leakage_at_any_ratio():
    """This check measures signal.  Leakage is owned by the other checks."""
    y, rng = _series(2)
    label = y.shift(-HORIZON)
    cases = {
        "oracle": pd.DataFrame({"lag_1": y.shift(1), "oracle": label}),
        "legit": pd.DataFrame({"lag_1": y.shift(1), "rmean_7":
                               y.rolling(7, min_periods=1).mean().shift(1)}),
        "noise": pd.DataFrame({"a": pd.Series(rng.normal(0, 1, N)),
                               "b": pd.Series(rng.normal(0, 1, N))}),
    }
    for name, X in cases.items():
        r = signal_sentinel(*_split(X, label), horizon=HORIZON)
        assert r["leakage_warning"] is False, f"sentinel asserted leakage on '{name}'"


def test_persistence_mimicry_is_invisible_to_the_sentinel():
    """A lagged-copy forecast is a *shift* finding, not a signal finding.

    Persistence features carry genuine signal on an autocorrelated series, so
    the sentinel correctly reports signal present.  That is precisely why
    persistence-mimicry needs its own detector and must not be folded into the
    sentinel's verdict.
    """
    y, _ = _series(3)
    label = y.shift(-HORIZON)
    X = pd.DataFrame({"lag_h": y.shift(HORIZON)})     # a persistence anchor
    r = signal_sentinel(*_split(X, label), horizon=HORIZON)
    assert r["leakage_warning"] is False
    assert r["signal_detected"] is True, (
        "persistence features do carry signal — so the sentinel cannot be the "
        "thing that catches persistence-mimicry"
    )


# ── separation of the three gate reasons ──────────────────────────────────

def _reasons(report=None, leak=False, shift=False):
    return gate_reasons(report or {}, leak, shift)


def _joined(rs):
    return " | ".join(rs).lower()


def test_no_signal_reason_does_not_mention_leakage():
    rs = _reasons({"signal_detected": False, "shuffled_to_normal_ratio": 0.83})
    assert any("no signal" in r.lower() for r in rs), rs
    assert not any("leak" in r.lower() for r in rs), (
        f"the no-signal reason must not be phrased as leakage: {rs}"
    )
    assert "0.83" in _joined(rs), "the reason should quote the measured ratio"


def test_leakage_reason_does_not_mention_no_signal():
    rs = _reasons(leak=True)
    assert any("leak" in r.lower() for r in rs), rs
    assert not any("no signal" in r.lower() for r in rs), rs


def test_persistence_reason_is_neither_leakage_nor_no_signal():
    rs = _reasons({"signal_detected": True}, shift=True)
    assert any("persistence-like" in r.lower() for r in rs), rs
    assert not any("leak" in r.lower() for r in rs), rs
    assert not any("no signal" in r.lower() for r in rs), rs


def test_all_three_conditions_yield_three_separate_reasons():
    """Co-occurrence must not collapse into a single verdict."""
    rs = _reasons({"signal_detected": False, "shuffled_to_normal_ratio": 0.83,
                   "run_status": "SUCCESS"}, leak=True, shift=True)
    joined = _joined(rs)
    assert "leak" in joined and "no signal" in joined and "persistence-like" in joined, rs
    assert len(rs) == len(set(rs)) >= 3, f"expected 3+ distinct reasons, got {rs}"


def test_signal_present_and_clean_shift_produces_no_reason():
    """The gate must stay quiet when nothing is wrong."""
    assert _reasons({"signal_detected": True, "run_status": "SUCCESS"}) == []


def test_absent_signal_field_is_not_treated_as_a_failure():
    """Families that do not run the sentinel (A_STAT) must not be blamed for it."""
    rs = _reasons({"run_status": "SUCCESS"})
    assert not any("no signal" in r.lower() for r in rs), (
        f"a missing signal_detected field means 'not measured', not 'failed': {rs}"
    )
