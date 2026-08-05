"""Retain the current forward run, then score whatever truth has arrived.

    ./backend/.venv/bin/python backend/run_publish_and_score.py

Publishing is idempotent per issue date: re-running on the same day is a no-op unless
--overwrite is passed, because a published forecast is the only record of what was said.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from forward_forecast import DEFAULT_OUT  # noqa: E402
from published_forecasts import list_published, publish, score_published  # noqa: E402

DATA = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an already-published issue date")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if not args.score_only:
        try:
            dest = publish(DEFAULT_OUT, overwrite=args.overwrite)
            print(f"[publish] retained -> {dest}")
        except FileExistsError as exc:
            print(f"[publish] skipped: {exc}")
        except FileNotFoundError as exc:
            print(f"[publish] no forward run to retain: {exc}")
            return 1

    print(f"[score] published issues: {len(list_published())}")
    out = score_published(DATA)
    print(f"[score] scored={out['scored']}  pending={out['pending']}  -> {out['scorecard']}")
    if out["summary"]:
        print(f"\n{'target':<24}{'n':>4}{'realized MAE':>16}{'ruler MAE':>16}"
              f"{'skill':>8}{'in range':>10}")
        for t, s in out["summary"].items():
            print(f"{t:<24}{s['n']:>4}{s['realized_mae']:>16,.0f}"
                  f"{s['persistence_mae']:>16,.0f}{s['skill_vs_ruler_pct']:>7.2f}%"
                  f"{s['interval_hit_rate']:>9.0%}")
    else:
        print("[score] nothing scoreable yet -- every published date is still in the future.")
        print("        This is the expected state for a forecast issued today; the scorer")
        print("        refuses to evaluate a date whose truth is not in the data.")
    if out["pending_dates"]:
        print(f"[score] awaiting truth for {len(out['pending_dates'])} target-dates, "
              f"earliest {out['pending_dates'][0][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
