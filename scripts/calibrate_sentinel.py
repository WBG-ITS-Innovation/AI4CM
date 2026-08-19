#!/usr/bin/env python3
"""Estimate the null distribution of the shuffled-target sentinel, and its false-positive rate.

    ./backend/.venv/bin/python scripts/calibrate_sentinel.py [--draws 120] [--json out.json]

Why this exists
---------------
``MIN_SIGNAL_RATIO`` was a bare constant with a one-line comment: *"shuffled MAE must be at
least this multiple of real MAE"*. The value 1.50 was rationalised in prose — "the error must
get at least half again worse… the margin above 1.00 exists so that noise cannot pass" — but
nobody had ever measured what a signal-free feature set actually produces, so the phrase
"noise cannot pass" had no number behind it. A threshold deciding publication on that basis
is a convention, not a measurement.

The null
--------
H0: the features carry no information about the target.

Two constructions, because they fail differently:

* ``permuted`` — take the REAL feature matrix and permute its rows. The feature-target
  pairing is destroyed while every feature's marginal distribution, scale and
  cross-correlation is preserved. This is the more realistic null and the wider one, so it
  is the one the threshold is set from.
* ``noise`` — replace features with standard normal draws of the same shape. Cleaner but
  optimistic: real features have heavier tails and mutual correlation, which a probe can
  latch onto.

The target is left exactly as the pipeline builds it (h-step ahead, delta for a stock
target), and the split is the real TRAIN -> DEV. Only the features are nulled, so any
apparent signal is an artifact of the instrument rather than of the data.

Reading the output
------------------
The false-positive rate at a threshold is the fraction of null draws that reach it. A
threshold whose FPR is 0.00% over a few hundred draws is safe against noise; a threshold
far above that point is not *safer*, it merely also rejects real readings.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

CANDIDATE_THRESHOLDS = (1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.50)
HORIZON = 5


def _design(target: str, raw: pd.DataFrame):
    from b_ml_pipeline import is_stock
    from forward_forecast import Champion, _build_design
    from registry import load_registry

    r = [x for x in load_registry()["recipes"] if x["target"] == target][0]
    champ = Champion(target=target, point_model=r["point_model"],
                     fiscal_groups=tuple(r["feature_groups"]), exog_blocks=(),
                     recipe_id=r["id"], scaling=r["scaling"], transform="raw")
    X, s = _build_design(raw, champ)
    y = s.shift(-HORIZON)
    if is_stock(target):
        y = y - s
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def calibrate(draws: int = 120) -> Dict:
    from evaluation_windows import DEV, TRAIN
    from forecast_integrity import signal_sentinel
    from registry import load_registry

    data = REPO / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
    raw = pd.read_csv(data)
    targets = [r["target"] for r in load_registry()["recipes"]]

    per_target: Dict[str, Dict] = {}
    pooled: Dict[str, List[float]] = {"permuted": [], "noise": []}

    for t in targets:
        X, y = _design(t, raw)
        tr = X.index <= pd.Timestamp(TRAIN.end)
        dv = (X.index >= pd.Timestamp(DEV.start)) & (X.index <= pd.Timestamp(DEV.end))
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[dv], y[dv]

        real = signal_sentinel(Xtr, ytr, Xte, yte, horizon=HORIZON)["shuffled_to_normal_ratio"]
        got: Dict[str, List[float]] = {"permuted": [], "noise": []}
        for k in range(draws):
            g = np.random.default_rng(1000 + k)

            ptr = Xtr.sample(frac=1.0, random_state=1000 + k); ptr.index = Xtr.index
            pte = Xte.sample(frac=1.0, random_state=2000 + k); pte.index = Xte.index
            v = signal_sentinel(ptr, ytr, pte, yte, horizon=HORIZON)["shuffled_to_normal_ratio"]
            if np.isfinite(v):
                got["permuted"].append(float(v))

            ntr = pd.DataFrame(g.normal(size=Xtr.shape), index=Xtr.index, columns=Xtr.columns)
            nte = pd.DataFrame(g.normal(size=Xte.shape), index=Xte.index, columns=Xte.columns)
            v = signal_sentinel(ntr, ytr, nte, yte, horizon=HORIZON)["shuffled_to_normal_ratio"]
            if np.isfinite(v):
                got["noise"].append(float(v))

        for name, vals in got.items():
            pooled[name].extend(vals)
        per_target[t] = {
            "real_reading": round(float(real), 6),
            "n_features": int(X.shape[1]),
            **{name: _describe(np.array(vals)) for name, vals in got.items()},
        }

    out = {
        "horizon": HORIZON,
        "draws_per_target_per_null": draws,
        "targets": targets,
        "per_target": per_target,
        "pooled": {name: _describe(np.array(v)) for name, v in pooled.items()},
        "fpr_by_threshold": {
            name: {f"{th:.2f}": round(float((np.array(v) >= th).mean()), 6)
                   for th in CANDIDATE_THRESHOLDS}
            for name, v in pooled.items()
        },
    }
    return out


def _describe(a: np.ndarray) -> Dict:
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "p99_9": round(float(np.percentile(a, 99.9)), 6),
        "max": round(float(a.max()), 6),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args()

    res = calibrate(a.draws)
    print(f"NULL CALIBRATION — {res['draws_per_target_per_null']} draws per target per null, "
          f"h={res['horizon']}\n")
    for t, v in res["per_target"].items():
        print(f"{t}  (real reading {v['real_reading']:.4f}, {v['n_features']} features)")
        for name in ("permuted", "noise"):
            d = v[name]
            print(f"   null[{name:8s}] n={d['n']:3d}  median={d['median']:.4f}  "
                  f"p95={d['p95']:.4f}  p99={d['p99']:.4f}  max={d['max']:.4f}")
    print("\nPOOLED:")
    for name in ("permuted", "noise"):
        d = res["pooled"][name]
        print(f"   {name:8s} n={d['n']}  median={d['median']:.4f}  p95={d['p95']:.4f}  "
              f"p99={d['p99']:.4f}  p99.9={d['p99_9']:.4f}  max={d['max']:.4f}")
    print("\nFALSE-POSITIVE RATE (pooled permuted null — the one the threshold is set from):")
    for th, fpr in res["fpr_by_threshold"]["permuted"].items():
        print(f"   threshold {th}  ->  FPR {fpr:.4%}")

    from publication_gates import SENTINEL_MIN
    print(f"\nSENTINEL_MIN in use: {SENTINEL_MIN}  "
          f"(FPR {res['fpr_by_threshold']['permuted'].get(f'{SENTINEL_MIN:.2f}', 'n/a')})")

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
