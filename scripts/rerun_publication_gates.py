#!/usr/bin/env python
"""Re-run the publication gates with interval coverage actually fed in, and diff the verdicts.

Why this exists
---------------
Every recipe in ``registry/recipes.json`` carries ``interval_model: GBQuantile`` and ships
``p10``/``p90`` in ``forecast.csv``, while every one of its coverage gates reads *"This model
reports no prediction intervals."* The gate was never fed a number, and ``publication_gates``
excluded coverage from ``unmeasured_gates``, so the omission was invisible rather than merely
unfixed.

This script closes the loop: it measures coverage from the materialised run artifacts
(``backend/coverage_report``), feeds it to ``publication_gates.decide`` alongside each recipe's
logged DEV credentials, and prints the verdict **before and after** so a calibration change can
never move a verdict without someone seeing why.

Windows and discipline
----------------------
It reads only rows a pipeline already wrote to disk, fits nothing and chooses nothing. Coverage
is passed as a *report* -- ``coverage_for_publication(..., purpose="report")`` -- because the
holdout may be reported on and never selected on. The verdict logic itself is untouched; only
the inputs are more complete than before.

    ./backend/.venv/bin/python scripts/rerun_publication_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from coverage_report import (coverage_for_publication, measure_all,   # noqa: E402
                             to_frame)
from publication_gates import Measured, decide                        # noqa: E402
from registry import load_registry                                    # noqa: E402


def main() -> int:
    log = pd.read_csv(REPO / "experiments" / "log.csv")
    measurements = measure_all()

    print("=" * 100)
    print("MEASURED INTERVAL COVERAGE (from materialised run artifacts; nothing refitted)")
    print("=" * 100)
    f = to_frame(measurements)
    f = f[f["n"] > 0].copy()
    f["overall_%"] = (f["overall_coverage"] * 100).round(1)
    f["large_day_%"] = (f["large_day_coverage"] * 100).round(1)
    f["nominal_%"] = (f["nominal"] * 100)
    pd.set_option("display.width", 250)
    print(f[["family", "target", "model", "n", "nominal_%", "overall_%", "large_day_%",
             "n_large_day", "windows"]].to_string(index=False))
    print("\nDay size is grouped by what was known at the origin, never by the actual. "
          "See backend/conformal.py for why that distinction changes the numbers.")

    rows = []
    print("\n" + "=" * 100)
    print("PUBLICATION VERDICTS -- before (coverage absent) vs after (coverage fed in)")
    print("=" * 100)
    for r in load_registry()["recipes"]:
        target = r["target"]
        row = log[log["run_id"] == r["dev_credentials"]["run_id"]].iloc[0]
        base = dict(target=target, mase=row["mase"], sentinel_ratio=row["sentinel_ratio"],
                    skill_vs_ruler_pct=row["skill_vs_ruler"],
                    leakage_detected=False, persistence_mimicry=False)

        before = decide(Measured(**base))

        # Gate the recipe on the band the recipe actually ships, not on whichever family
        # happens to have the most scored rows for this target.
        cm = coverage_for_publication(measurements, target,
                                      interval_model=r.get("interval_model"),
                                      purpose="report")
        after = decide(Measured(**base,
                                has_intervals=True,   # every recipe ships p10/p90
                                coverage=(cm.overall if cm else None),
                                coverage_nominal=(cm.nominal if cm else None)))

        cov_gate = after["gates"]["coverage"]
        rows.append({
            "target": target,
            "registry_verdict": r["publication"]["verdict"],
            "verdict_before": before["verdict"],
            "verdict_after": after["verdict"],
            "changed": before["verdict"] != after["verdict"],
            "coverage_measured": (None if cm is None else round(cm.overall, 4)),
            "coverage_n": (0 if cm is None else cm.n),
            "coverage_source": ("no interval artifact exists for this target" if cm is None
                                else f"{cm.family}/{cm.model} over {cm.n} rows "
                                     f"({','.join(cm.windows)})"),
            "coverage_gate": cov_gate["passed"],
            "decided_by_before": before["decided_by"],
            "decided_by_after": after["decided_by"],
        })
        print(f"\n--- {target}")
        print(f"    registry says      : {r['publication']['verdict']}")
        print(f"    before             : {before['verdict']} "
              f"(decided by {before['decided_by']}), unmeasured={before['unmeasured_gates']}")
        print(f"    after              : {after['verdict']} "
              f"(decided by {after['decided_by']}), unmeasured={after['unmeasured_gates']}")
        print(f"    coverage gate      : passed={cov_gate['passed']} "
              f"measured={cov_gate['measured']} threshold={cov_gate.get('threshold')}")
        print(f"    coverage source    : {rows[-1]['coverage_source']}")
        print(f"    coverage says      : {cov_gate['reason_plain']}")

    d = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(d[["target", "registry_verdict", "verdict_before", "verdict_after", "changed",
             "coverage_measured", "coverage_gate"]].to_string(index=False))

    changed = d[d["changed"]]
    if changed.empty:
        print("\nNo verdict changes. Feeding coverage in did not move any target's decision.")
    else:
        print(f"\n{len(changed)} VERDICT CHANGE(S) -- each needs review before publishing:")
        for _, c in changed.iterrows():
            print(f"  {c['target']}: {c['verdict_before']} -> {c['verdict_after']} "
                  f"(coverage {c['coverage_measured']})")

    out = REPO / "reports" / "coverage_gate_rerun.json"
    out.write_text(json.dumps({"verdicts": rows,
                               "measurements": [m.as_dict() for m in measurements]},
                              indent=2, default=str))
    print(f"\nWROTE {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
