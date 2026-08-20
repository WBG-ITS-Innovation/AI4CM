"""Lab runner for the pretrained zero-shot forecasters. Exploratory only.

Reads the same ``TG_*`` environment contract every other family's runner reads, so the Lab
launches this exactly as it launches A_STAT or B_ML, and writes the same
``predictions_long.csv`` shape, so the Dashboard, the overlay chart and the download buttons all
work without knowing this family exists.

WHAT IT DOES, AND WHY IT IS A BACKTEST RATHER THAN ONE FORECAST
--------------------------------------------------------------
There is no training step, so "run this model" could have meant one forward forecast off the end
of the data. It does not, because a forward forecast has no ``y_true`` beside it and therefore
cannot be compared to anything. Instead it walks the evaluation window: at each origin it hands
the model the history up to that origin, asks for ``horizon`` steps, and keeps the last one. That
produces exactly the rows every other family produces, so these models can later be measured on
the same footing rather than needing a separate path built for them.

Nothing is fitted at any origin. The same fixed weights answer every time, which is what makes
the walk cheap: measured on Revenues, roughly 0.1s per origin once the checkpoint is loaded.

CAUSALITY
---------
Each origin sees only data up to and including that origin. The forecast for a target date is
made from an origin ``horizon`` business days earlier, which is the same convention the other
families use, so a row here means the same thing as a row there.

BOUNDED TO TRAIN AND DEV
------------------------
``eval_end`` arrives from the Lab through ``TG_PARAM_OVERRIDES`` and is honoured. The Lab sets it
from ``frontend/exploratory.py``, which pins it to the last dev date, so a run launched from the
interface cannot reach the sealed window. This runner does not widen it and does not default it
open: with no ``eval_end`` given it stops at the last dev date it was told about, and with none at
all it refuses rather than running to the end of the file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import foundation_models as fm                                            # noqa: E402
from target_kinds import is_stock                                         # noqa: E402


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _overrides() -> dict:
    raw = _env("TG_PARAM_OVERRIDES", "{}")
    try:
        return json.loads(raw) or {}
    except ValueError:
        return {}


def main() -> int:
    model = _env("TG_MODEL_FILTER")
    target = _env("TG_TARGET")
    horizon = int(_env("TG_HORIZON", "5") or 5)
    data_path = Path(_env("TG_DATA_PATH"))
    date_col = _env("TG_DATE_COL", "date")
    out_root = Path(_env("TG_OUT_ROOT", str(BACKEND.parent / "runs" / "foundation")))
    ov = _overrides()

    spec = fm.spec_for(model)
    if spec is None:
        print(f"[foundation] {model!r} is not a registered foundation model. "
              f"Known: {', '.join(fm.names())}")
        return 2
    if not fm.installed(spec):
        print(f"[foundation] {spec.requires} is {fm.REASON_NOT_INSTALLED}")
        return 3
    if not data_path.exists():
        print(f"[foundation] data file not found: {data_path}")
        return 4

    frame = pd.read_csv(data_path)
    if target not in frame.columns:
        print(f"[foundation] {target!r} is not a column in {data_path.name}")
        return 5

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    series = (frame.dropna(subset=[date_col])
                   .set_index(date_col)[target]
                   .astype(float).dropna().sort_index())
    series = fm.business_days_only(series)
    if is_stock(target):
        print(f"[foundation] note: {target!r} is a balance. The model is given the level as it "
              f"stands; no change-from-origin reconstruction is applied, unlike B_ML.")

    # The evaluation window. `eval_end` is honoured and never widened -- see the module docstring.
    eval_end = ov.get("eval_end")
    if not eval_end:
        print("[foundation] refusing to run: no eval_end was given, and defaulting it open would "
              "let an exploratory run reach the sealed holdout. The Lab always sets it.")
        return 6
    eval_end = pd.Timestamp(eval_end)
    eval_start = ov.get("eval_start")
    evaluable = series.loc[:eval_end]

    origins = evaluable.index[-(len(evaluable)):]
    if eval_start:
        origins = origins[origins >= pd.Timestamp(eval_start)]
    # An origin needs `horizon` further business days inside the window to have a truth to score
    # against, and enough history behind it to be worth asking about.
    min_context = 64
    usable = [o for i, o in enumerate(evaluable.index)
              if i >= min_context and i + horizon < len(evaluable)]
    if eval_start:
        usable = [o for o in usable if o >= pd.Timestamp(eval_start)]
    if not usable:
        print(f"[foundation] no origin has both {min_context} business days of history behind it "
              f"and {horizon} inside the window. Nothing to evaluate.")
        return 7

    print(f"[foundation] {spec.name} on {target!r}, h={horizon}, "
          f"{len(usable)} origins from {usable[0].date()} to {usable[-1].date()}")
    print(f"[foundation] checkpoint {spec.repo} @ {spec.revision}")
    print(f"[foundation] zero-shot: no fitting at any origin, the same fixed weights answer each")

    positions = {ts: i for i, ts in enumerate(evaluable.index)}
    rows, t0, failures = [], time.time(), 0
    for n, origin in enumerate(usable, start=1):
        i = positions[origin]
        history = evaluable.iloc[: i + 1]
        target_date = evaluable.index[i + horizon]
        result, why = fm.forecast(spec.name, history, horizon=horizon)
        if result is None:
            failures += 1
            if failures <= 3:
                print(f"[foundation] origin {origin.date()} failed: {why}")
            continue
        rows.append({
            "date": target_date, "target_date": target_date, "origin_date": origin,
            "origin_value": float(evaluable.iloc[i]),
            "y_true": float(evaluable.iloc[i + horizon]),
            "y_pred": result["p50"][-1],
            "y_lo": result["p10"][-1], "y_hi": result["p90"][-1],
            "yhat_p10": result["p10"][-1], "yhat_p50": result["p50"][-1],
            "yhat_p90": result["p90"][-1],
            "model": spec.name, "horizon": horizon, "target": target,
        })
        if n % 25 == 0 or n == len(usable):
            print(f"[foundation] {n}/{len(usable)} origins, {time.time() - t0:.1f}s", flush=True)

    if not rows:
        print("[foundation] every origin failed; nothing written")
        return 8

    out_root.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(rows)
    preds.to_csv(out_root / "predictions_long.csv", index=False)

    err = (preds["y_true"] - preds["y_pred"]).abs()
    mae = float(err.mean())
    inside = float(((preds["y_true"] >= preds["yhat_p10"]) &
                    (preds["y_true"] <= preds["yhat_p90"])).mean())
    pd.DataFrame([{
        "target": target, "horizon": horizon, "model": spec.name,
        "MAE": mae, "PI_coverage@80": inside, "n_origins": len(preds),
    }]).to_csv(out_root / "metrics_long.csv", index=False)
    pd.DataFrame([{"target": target, "horizon": horizon, "model": spec.name,
                   "MAE": mae, "rank": 1}]).to_csv(out_root / "leaderboard.csv", index=False)

    artifacts = out_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "config.json").write_text(json.dumps({
        "family": fm.FAMILY_FOUNDATION, "model": spec.name, "target": target,
        "horizon": horizon, "data_path": str(data_path), "date_col": date_col,
        "eval_start": str(eval_start) if eval_start else None, "eval_end": str(eval_end),
        "repo": spec.repo, "revision": spec.revision,
        "zero_shot": True, "fitted_here": False,
        "context_business_days": fm.DEFAULT_CONTEXT,
        "n_origins": len(preds), "failures": failures,
        "note": ("Exploratory. No training step, weights pinned by commit hash, and this model "
                 "is not eligible to become a champion: the registry's champion pool is the "
                 "machine-learning family alone."),
    }, indent=2), encoding="utf-8")

    print(f"[foundation] wrote {len(preds)} rows to {out_root}")
    print(f"[foundation] MAE {mae:,.0f}   80% band covered {inside:.1%} of outcomes"
          f"{f'   ({failures} origin(s) failed)' if failures else ''}")
    print("[foundation] EXPLORATORY: not published, not scored, cannot become champion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
