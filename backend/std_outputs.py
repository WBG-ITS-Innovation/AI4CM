# std_outputs.py — drop-in helper to standardize outputs across pipelines
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional, List
import json
import pandas as pd

def ensure_out_root(out_root: Path) -> None:
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "artifacts").mkdir(parents=True, exist_ok=True)

def write_predictions_long(out_root: Path, df: pd.DataFrame) -> Path:
    # Require columns: ['date','y_true','y_pred','target','cadence','family','model','fold','horizon']
    req = {'date','y_pred','target','cadence','family','model','horizon'}
    miss = req - set(map(str.lower, df.columns))
    # best-effort: rename common variations
    cols = {c.lower(): c for c in df.columns}
    def col(name): 
        return cols.get(name, name)
    if 'date' not in cols:
        raise ValueError("predictions_long must include a 'date' column")
    path = out_root / "predictions_long.csv"
    df.to_csv(path, index=False)
    return path

def write_metrics_long(out_root: Path, df: pd.DataFrame) -> Path:
    # Expect columns: ['model','metric','value','target','cadence','family','horizon','fold']
    path = out_root / "metrics_long.csv"
    df.to_csv(path, index=False)
    return path

def write_leaderboard(out_root: Path, df: pd.DataFrame) -> Path:
    # Expect wide-ish: model + metric columns, already aggregated across folds
    path = out_root / "leaderboard.csv"
    df.to_csv(path, index=False)
    return path

def write_config(out_root: Path, config: Mapping) -> Path:
    p = out_root / "artifacts" / "config.json"
    p.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return p
