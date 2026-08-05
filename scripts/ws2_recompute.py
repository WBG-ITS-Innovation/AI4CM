#!/usr/bin/env python
"""Recompute the WS2 metrics correctly from the EXISTING search results.

The 100-trial Optuna search is NOT re-run. Its winning configuration is read back from
``experiments/runs/<run_id>.json`` (``params_full.best_params``) and refitted once per target,
which is what "recompute the metrics" means here.

Two defects in the original harness made two of its four published numbers unusable:

1. **Non-canonical ruler.** The harness built target dates as ``origin + BDay(h)`` -- calendar
   business days, which ignore Georgian public holidays and can land off the series -- and
   filtered rows by ``X.notna()``. Both changed which ``(y_true, origin_value)`` pairs entered
   the baseline, so it produced 90,800,654 for Revenues against the canonical 88,317,355 and
   76,722,514 for Expenditure against 73,117,667. Every ``skill_vs_ruler`` value it logged is
   therefore incomparable with every other workstream's. Here the ruler is computed from the
   target series on the DEV window with a plain h-step shift, which reproduces the canonical
   value the registry and the WS4/WS5 rows carry.

2. **Sentinel units mismatch.** The probe was handed the *ratio-transformed* training target
   alongside *original-scale* test truth, so both the real and shuffled errors were dominated by
   the same units gap and their ratio collapsed to exactly 1.0000 -- a number that measures
   nothing. Here both sides are in original units.

The ``dev_mae`` values from the original runs were always sound; they are reproduced as a check.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluation_windows import mase, seasonal_naive_scale                      # noqa: E402
from experiment_log import log_run, read_log                                   # noqa: E402
from forecast_integrity import (PROBE_FOREST, PROBE_RIDGE, PROBE_TREE,         # noqa: E402
                                compute_persistence_baseline, signal_sentinel)
from preprocessing.fiscal_calendar import calendar_version                     # noqa: E402
from provenance import describe_code, describe_input                           # noqa: E402
from tuning import fit_quantiles                                               # noqa: E402
from ws2_tune import H, design, make_folds                                     # noqa: E402

DATA = str(REPO / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv")


def canonical_dev_ruler(target: str) -> float:
    """The DEV ruler as every other workstream computes it: y(t) vs y(t-h) on the DEV window."""
    from b_ml_pipeline import to_business_index
    s = to_business_index(pd.read_csv(DATA), "date", target)
    idx = s.index[(s.index >= "2024-01-01") & (s.index <= "2024-12-31")]
    df = pd.DataFrame({"target_date": idx, "y_true": s.reindex(idx).to_numpy(),
                       "origin_value": s.shift(H).reindex(idx).to_numpy()}).dropna()
    return float(compute_persistence_baseline(df)["mae_persistence"])


def main() -> int:
    di, code, calv = describe_input(DATA), describe_code(), calendar_version()
    raw = pd.read_csv(DATA)
    rawi = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    bidx = pd.date_range(rawi.index.min().normalize(), rawi.index.max().normalize(), freq="B")

    prior = {r["target"]: r for r in read_log() if "WS2" in (r.get("note") or "")}
    out_rows = []

    for target, row in prior.items():
        detail = json.loads((REPO / "experiments" / "runs" / f"{row['run_id']}.json").read_text())
        pf = detail["params_full"]
        model, best_params = pf["model"], pf["best_params"]
        tf = pf.get("target_transform", "raw")

        folds, ctx = make_folds(target, "dev")
        assert len(folds) == 1, f"{target}: expected one DEV fold, got {len(folds)}"
        f = folds[0]
        s, X, y_t, y_true_s, tf_ctx, lvl, stock = ctx

        qp, ncross = fit_quantiles(model, f.X_tr, f.y_tr, f.X_te, best_params, H)
        if f.inverse is not None:
            qp = {q: np.asarray(f.inverse(v), dtype=float) for q, v in qp.items()}
        y = f.y_te
        p50 = qp[0.50]
        dev_mae = float(np.mean(np.abs(y - p50)))

        # ── defect 1 fixed: the canonical ruler ──────────────────────────────────
        ruler = canonical_dev_ruler(target)
        skill = (ruler - dev_mae) / ruler * 100.0

        # ── defect 2 fixed: both probe sides in ORIGINAL units ───────────────────
        ite = f.X_te.index
        y_tr_orig = y_true_s.reindex(f.X_tr.index)
        probes = {}
        for name in (PROBE_RIDGE, PROBE_TREE, PROBE_FOREST):
            r = signal_sentinel(f.X_tr, y_tr_orig, f.X_te,
                                pd.Series(y, index=ite), horizon=H, probe=name)
            probes[name] = r["shuffled_to_normal_ratio"]

        cov = float(np.mean((y >= qp[0.10]) & (y <= qp[0.90])))
        q = pd.qcut(pd.Series(np.abs(y)), 3, labels=["low", "mid", "high"])
        terc = {l: float(np.mean((y[(q == l).to_numpy()] >= qp[0.10][(q == l).to_numpy()]) &
                                (y[(q == l).to_numpy()] <= qp[0.90][(q == l).to_numpy()])))
                for l in ("low", "mid", "high")}
        ser = (rawi[target].reindex(bidx).ffill() if stock
               else rawi[target].reindex(bidx).fillna(0.0))
        m = mase(pd.Series(y), pd.Series(p50),
                 seasonal_naive_scale(ser, season=5, window="train"))

        d = log_run(
            target=target, model=f"{model}_tuned_recomputed",
            git_sha=code["git_sha"], data_sha=di["sha256"],
            feature_names=[f"ws2recompute:{row['run_id']}"],
            params={"study": "ws2_tuning_recomputed", "model": model,
                    "best_params": best_params, "target_transform": tf, "horizon": H,
                    "supersedes_run_id": row["run_id"],
                    "ruler_source": "canonical DEV h-step persistence",
                    "sentinel_probes": {k: v for k, v in probes.items()}},
            seed=0,
            fold_scheme="DEV confirmation refitted from the stored best_params (search not re-run)",
            dev_mae=dev_mae, mase=m, skill_vs_ruler=skill,
            sentinel_ratio=probes[PROBE_RIDGE], coverage=terc, ruler=ruler,
            calendar_version=calv,
            note=(f"WS2 RECOMPUTED: canonical ruler and units-consistent sentinel; "
                  f"supersedes {row['run_id']}"))

        out_rows.append(dict(
            target=target, model=model, dev_mae=dev_mae,
            dev_mae_original=float(row["dev_mae"]),
            ruler_canonical=ruler, ruler_harness=float(detail["ruler"]),
            skill=skill, skill_bogus=float(row["skill_vs_ruler"]),
            ridge=probes[PROBE_RIDGE], tree=probes[PROBE_TREE], forest=probes[PROBE_FOREST],
            sentinel_original=float(row["sentinel_ratio"]),
            cov=cov, cov_low=terc["low"], cov_mid=terc["mid"], cov_high=terc["high"],
            mase=m, crossings=ncross, run_id=d["run_id"]))
        print(f"[{target}] dev_mae={dev_mae:,.0f} (was {float(row['dev_mae']):,.0f}) "
              f"ruler {float(detail['ruler']):,.0f} -> {ruler:,.0f} "
              f"skill {float(row['skill_vs_ruler']):.2f}% -> {skill:.2f}% "
              f"sentinel {float(row['sentinel_ratio']):.4f} -> ridge {probes[PROBE_RIDGE]:.4f} "
              f"tree {probes[PROBE_TREE]:.4f}", flush=True)

    pd.DataFrame(out_rows).to_csv(os.environ.get("OUT", "/tmp/ws2_recomputed.csv"), index=False)
    print("\nWROTE", os.environ.get("OUT", "/tmp/ws2_recomputed.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
