"""Runner: production forward forecast for the three Phase-2 champion targets.

Emits the next five Georgian business days after the end of the data, with P10/P50/P90,
full provenance, and the DEV gate verdicts attached by recipe_id.

    ./backend/.venv/bin/python backend/run_forward_forecast.py

TEST (2025) is not touched: every target date is beyond the data end, so there is no truth
to read. Gate verdicts come from the DEV credentials run, never from forward dates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from forward_forecast import (DEFAULT_OUT, Champion, build_provenance, run_forward,  # noqa: E402
                             write_artifacts)
from registry import load_registry  # noqa: E402

DATA = str(BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv")


def champions_from_registry() -> list:
    reg = load_registry()
    out = []
    for r in reg["recipes"]:
        out.append(Champion(
            target=r["target"],
            point_model=r["point_model"],
            fiscal_groups=tuple(r["feature_groups"]),
            exog_blocks=tuple(r.get("exog_blocks") or ()),
            recipe_id=r["id"],
            scaling=r["scaling"],
            transform=r.get("params", {}).get("target_transform", "raw"),
        ))
    return out


def main(publish: bool = False) -> int:
    champs = champions_from_registry()
    raw = pd.read_csv(DATA)
    print(f"[forward] data through {pd.to_datetime(raw['date']).max().date()}, "
          f"{len(raw)} rows")

    frames = []
    sink: list = []
    for c in champs:
        print(f"[forward] {c.target}: {c.point_model} + GBQuantile, "
              f"groups={sorted(c.fiscal_groups)}, exog={sorted(c.exog_blocks) or 'none'}")
        df = run_forward(raw, c, estimator_sink=sink)
        frames.append(df)
        for _, r in df.iterrows():
            print(f"           {r['target_date'].date()}  h={r['horizon']}  "
                  f"P50={r['p50']:>18,.0f}  [{r['p10']:>18,.0f} .. {r['p90']:>18,.0f}]")

    forecasts = pd.concat(frames, ignore_index=True)
    prov = build_provenance(DATA, champs)

    # Gate verdicts are DEV credentials, carried by recipe_id -- never recomputed on
    # forward dates, which have no truth.
    reg = load_registry()
    gates = {r["id"]: {"target": r["target"],
                       "gates": r.get("dev_credentials", {}).get("gates", {}),
                       "status": r["status"]}
             for r in reg["recipes"]}

    paths = write_artifacts(DEFAULT_OUT, forecasts, prov, gates)
    print("\n[forward] artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"[forward] test_window_touched = {prov['test_window_touched']}")

    if publish:
        dest = _publish_and_retain(forecasts, prov, sink)
        print(f"[forward] published to {dest}")
    return 0


def _publish_and_retain(forecasts: pd.DataFrame, prov: dict, sink: list) -> Path:
    """Publish the issue and retain the estimators behind it, then prune old blobs.

    Retention is opt-in on this runner because publishing is: an accidental publish is not
    reversible, since a published forecast is the only record of what was said.
    """
    from estimator_store import (DEFAULT_KEEP_LAST, issues_with_unscored_horizons,
                                 prune_estimators, save_estimators)
    from published_forecasts import PUBLISHED_ROOT, publish

    dest = publish(DEFAULT_OUT)
    origin = pd.DatetimeIndex(pd.to_datetime(forecasts["origin_date"]).unique())
    mpath = save_estimators(dest, sink, keep_index=origin, provenance=prov)
    total = json.loads(mpath.read_text())["total_bytes"]
    print(f"[forward] retained {len(sink)} estimators, {total / 1024 / 1024:.2f} MB "
          f"(blobs gitignored; manifest tracked -- see backend/estimator_store.py)")

    protect = issues_with_unscored_horizons(PUBLISHED_ROOT)
    for act in prune_estimators(PUBLISHED_ROOT, keep_last=DEFAULT_KEEP_LAST, protect=protect):
        print(f"[forward] pruned {act['n_blobs']} blobs from {act['issue_date']}, "
              f"freed {act['bytes_freed'] / 1024 / 1024:.2f} MB")
    return dest


if __name__ == "__main__":
    raise SystemExit(main(publish="--publish" in sys.argv))
