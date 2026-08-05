#!/usr/bin/env python
"""Part 4: apply CQR to the quantile models and gate on conditional coverage.

Uses each target's WS2 winning configuration (read back from the logged run, not re-searched) as
the band producer, then calibrates conformally on a causal slice of the TRAIN window and scores
the DEV window before and after.

Information-set justification: the calibration slice is the LAST rows of the training window with
an h-row gap, so every conformity score is computed from a target already known when the band is
issued. No calibration row's target falls at or after a DEV origin.
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

from conformal import (DEFAULT_ALPHA, TERCILES, assign_terciles_by_edges,   # noqa: E402
                       causal_calibration_split, conditional_coverage_gate,
                       coverage_by_bucket, cqr_calibrate, select_per_target,
                       tercile_edges)
from experiment_log import log_run, read_log                                # noqa: E402
from preprocessing.fiscal_calendar import calendar_version                  # noqa: E402
from provenance import describe_code, describe_input                        # noqa: E402
from tuning import fit_quantiles                                            # noqa: E402
from ws2_tune import H, make_folds                                          # noqa: E402

DATA = str(REPO / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv")


def trailing_vol(s: pd.Series, window: int = 21) -> pd.Series:
    """Realised volatility over the preceding window, shifted so it is known at the origin."""
    return s.pct_change().rolling(window, min_periods=5).std().shift(1)


def main() -> int:
    di, code, calv = describe_input(DATA), describe_code(), calendar_version()
    prior = {r["target"]: r for r in read_log() if "WS2" in (r.get("note") or "")
             and "RECOMPUTED" not in (r.get("note") or "")}
    rows = []

    for target, prow in prior.items():
        detail = json.loads((REPO / "experiments" / "runs" / f"{prow['run_id']}.json").read_text())
        pf = detail["params_full"]
        model, best = pf["model"], pf["best_params"]

        folds, ctx = make_folds(target, "dev")
        f = folds[0]
        s = ctx[0]

        # ── BEFORE: the band as the model produces it ────────────────────────
        qp, ncross = fit_quantiles(model, f.X_tr, f.y_tr, f.X_te, best, H)
        if f.inverse is not None:
            qp = {q: np.asarray(f.inverse(v), dtype=float) for q, v in qp.items()}
        y = np.asarray(f.y_te, dtype=float)
        lo0, hi0 = qp[0.10], qp[0.90]

        # ── calibrate on a causal slice of the TRAIN window ──────────────────
        fit_ix, cal_ix = causal_calibration_split(len(f.X_tr), H)
        qc, _ = fit_quantiles(model, f.X_tr.iloc[fit_ix], f.y_tr[fit_ix],
                              f.X_tr.iloc[cal_ix], best, H)
        y_cal_model = f.y_tr[cal_ix]
        lo_c, hi_c = qc[0.10], qc[0.90]
        # calibrate in the space the model works in, then invert for evaluation
        cal_g = cqr_calibrate(y_cal_model, lo_c, hi_c, alpha=DEFAULT_ALPHA, grouped=False)
        cal_t = cqr_calibrate(y_cal_model, lo_c, hi_c, alpha=DEFAULT_ALPHA, grouped=True)

        def widen(cal):
            """Apply the correction, bucketing by the same observable used to calibrate."""
            lo_m, hi_m = qp[0.10], qp[0.90]
            basis = (lo_m + hi_m) / 2.0
            return cal.apply(lo_m, hi_m, magnitude=basis if cal.grouped else None)

        lo1, hi1 = widen(cal_g)
        lo2, hi2 = widen(cal_t)

        vol = trailing_vol(s).reindex(f.X_te.index).to_numpy(dtype=float)
        edges = tercile_edges(y)
        mb = assign_terciles_by_edges(y, edges)

        res = {"target": target, "model": model, "crossings": ncross,
               "n": int(len(y)), "cal_n": int(len(cal_ix)),
               "width_global": cal_g.width, "cal_note": cal_t.note}
        for tag, (lo, hi) in (("before", (lo0, hi0)), ("cqr_global", (lo1, hi1)),
                              ("cqr_grouped", (lo2, hi2))):
            g = conditional_coverage_gate(y, lo, hi, magnitude=y, volatility=vol)
            cb = coverage_by_bucket(y, lo, hi, mb)
            res[f"{tag}_overall"] = g["overall_coverage"]
            res[f"{tag}_gate"] = g["passed"]
            res[f"{tag}_reason"] = g["reason_plain"]
            for t in TERCILES:
                res[f"{tag}_{t.split()[0]}"] = cb.get(t, {}).get("coverage", np.nan)
            res[f"{tag}_mean_width"] = float(np.mean(np.asarray(hi) - np.asarray(lo)))
            if tag != "before":
                log_run(target=target, model=f"{model}+{tag}",
                        git_sha=code["git_sha"], data_sha=di["sha256"],
                        feature_names=[f"ws7:{tag}"],
                        params={"study": "ws7_cqr", "model": model, "cqr": tag,
                                "alpha": DEFAULT_ALPHA, "horizon": H,
                                "calibration_rows": int(len(cal_ix)),
                                "conditional_floor": g["floor"],
                                "gate_passed": g["passed"]},
                        seed=0,
                        fold_scheme="CQR calibrated on a causal TRAIN slice; DEV confirmation",
                        dev_mae=float(np.mean(np.abs(y - qp[0.50]))), mase=None,
                        skill_vs_ruler=None, sentinel_ratio=None,
                        coverage={"low": res[f"{tag}_smallest"],
                                  "mid": res[f"{tag}_middle"],
                                  "high": res[f"{tag}_largest"]},
                        ruler=None, calendar_version=calv,
                        note=f"WS7 CQR {tag} on {target}, DEV n={len(y)}")
        rows.append(res)
        print(f"[{target}] before {res['before_overall']:.1%} "
              f"(largest {res['before_largest']:.1%}, gate {res['before_gate']}) -> "
              f"global {res['cqr_global_overall']:.1%} (largest {res['cqr_global_largest']:.1%}, "
              f"gate {res['cqr_global_gate']}) -> grouped {res['cqr_grouped_overall']:.1%} "
              f"(largest {res['cqr_grouped_largest']:.1%}, gate {res['cqr_grouped_gate']})",
              flush=True)

    pd.DataFrame(rows).to_csv(os.environ.get("OUT", "/tmp/ws7.csv"), index=False)
    print("\nWROTE", os.environ.get("OUT", "/tmp/ws7.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
