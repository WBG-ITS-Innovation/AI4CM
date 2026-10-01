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
from typing import Optional

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
        out = _publish_and_retain(forecasts, prov, gates, sink)
        for target, reason in out["refused"]:
            print(f"[forward] REFUSED {target}: {reason}")
        if out["dest"] is None:
            print("[forward] nothing published: every target was refused")
        else:
            print(f"[forward] published {', '.join(out['published'])} to {out['dest']}")
    return 0


def _publish_and_retain(forecasts: pd.DataFrame, prov: dict, gates: dict, sink: list, *,
                        published_root: Optional[Path] = None,
                        registry: Optional[dict] = None) -> dict:
    """Publish the targets the verdict allows, retain their estimators, prune old blobs.

    Retention is opt-in on this runner because publishing is: an accidental publish is not
    reversible, since a published forecast is the only record of what was said.

    THE VERDICT GUARD IS THE PAGE'S. ``forecast_modes.refuse_withheld`` is applied per target,
    exactly as ``publish_official`` applies it, so a target the Forecast page refuses is refused
    here with the same words. Until 2026-10-01 this function published the whole forward
    directory in one call with no verdict check, so the runner put `withheld` targets into the
    record that the page would have refused (scoring-loop audit, finding F3). The shared forward
    directory (``DEFAULT_OUT``) still holds every target, because the Forecast page's reading tab
    and the treasury report read it; only the published copy is filtered.

    Returns ``{"dest", "published", "refused"}``: the issue directory (None when nothing
    published), the targets published, and ``(target, reason)`` for each refusal.

    ``published_root`` redirects the record, and with it switches vault retention off, as
    ``published_forecasts.publish`` does -- a test publishing into a temporary directory must not
    write into the real vault. ``registry`` is passed through to the guard.
    """
    import shutil

    from estimator_store import (DEFAULT_KEEP_LAST, issues_with_unscored_horizons,
                                 prune_estimators, save_estimators)
    from forecast_modes import NotOfficial, refuse_withheld
    from published_forecasts import PUBLISHED_ROOT, publish, retain_to_vault

    refused: list = []
    kept: list = []
    for target in list(dict.fromkeys(forecasts["target"])):
        try:
            refuse_withheld(target, registry)
        except NotOfficial as exc:
            refused.append((target, str(exc)))
            continue
        kept.append(target)

    if not kept:
        return {"dest": None, "published": [], "refused": refused}

    allowed = forecasts[forecasts["target"].isin(kept)].reset_index(drop=True)
    prov_pub = dict(prov)
    prov_pub["recipes"] = [r for r in prov.get("recipes", []) if r.get("target") in kept]
    gates_pub = {rid: g for rid, g in (gates or {}).items() if g.get("target") in kept}
    sink_pub = [e for e in sink if e.target in kept]

    # Staged beside the shared forward directory, never in it: the staged copy is the filtered
    # record, the shared directory keeps every target for the surfaces that read it.
    issue_date = str(pd.to_datetime(allowed["origin_date"]).max().date())
    staged = DEFAULT_OUT.parent / "staging" / f"{issue_date}--runner"
    if staged.exists():
        shutil.rmtree(staged)
    write_artifacts(staged, allowed, prov_pub, gates_pub)
    dest = publish(staged, published_root=published_root)
    shutil.rmtree(staged, ignore_errors=True)

    root = Path(published_root) if published_root is not None else PUBLISHED_ROOT
    origin = pd.DatetimeIndex(pd.to_datetime(allowed["origin_date"]).unique())
    mpath = save_estimators(dest, sink_pub, keep_index=origin, provenance=prov_pub)
    if published_root is None:
        retain_to_vault(dest)      # blobs landed after publish() mirrored; idempotent re-sync
    total = json.loads(mpath.read_text())["total_bytes"]
    print(f"[forward] retained {len(sink_pub)} estimators, {total / 1024 / 1024:.2f} MB "
          f"(blobs gitignored; manifest tracked -- see backend/estimator_store.py)")

    protect = issues_with_unscored_horizons(root)
    for act in prune_estimators(root, keep_last=DEFAULT_KEEP_LAST, protect=protect):
        print(f"[forward] pruned {act['n_blobs']} blobs from {act['issue_date']}, "
              f"freed {act['bytes_freed'] / 1024 / 1024:.2f} MB")
    return {"dest": dest, "published": kept, "refused": refused}


if __name__ == "__main__":
    raise SystemExit(main(publish="--publish" in sys.argv))
