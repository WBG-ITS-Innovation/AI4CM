# -*- coding: utf-8 -*-
"""
G · Ensembling & Model Selection (post-process)
-----------------------------------------------
• Inputs: a list of folders each containing a predictions_long.csv
  (same schema as A/B/C families: date,target,horizon,model,y_true,y_pred,y_lo,y_hi[,cadence],...).
• Builds ensembles per target × horizon × cadence × date:
    - ens_median       : median of all available models' predictions
    - ens_mean_topK    : mean of top-K models by global MAE (default K=3)
    - ens_invMAE       : inverse-MAE weighted average (global weights)
• Outputs (per cadence):
    outputs/ensemble/<cadence>/
      predictions_ensemble.csv
      metrics_ensemble.csv
      leaderboard_ensemble.csv
      plots/
        <Target>_h<H>_overlay_top_ensemble.png
        <Target>_h<H>_leaderboard_mae_ensemble.png
        <Target>_h<H>_monthly_bars_top_vs_ops_ensemble.png
• Fast: no training; reads&combines existing predictions. Compares against Treasury Ops baseline if present.

Author: Georgia project
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

# ------------------ CONFIG (edit as needed) ------------------
# Source folders (each must contain predictions_long.csv; cadence inferred from path or overridden by CADENCE).
SOURCE_DIRS = [
    r".\outputs\statistical\daily",
    r".\outputs\ml_univariate\daily",
    r".\outputs\ml_multivariate\daily",
    #r".\outputs\dl_univariate\daily",
    r".\outputs\dl_multivariate\daily",
]
OUT_ROOT = r".\outputs\ensemble"
CADENCE  = "daily"                  # "daily" | "weekly" | "monthly"
TOP_K    = 3                        # K for ens_mean_topK
# -------------------------------------------------------------

def ensure_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "plots"), exist_ok=True)

def is_stock(target: str) -> bool:
    return target.strip().lower() in {"state budget balance","balance","t0"}

def coverage_width(y_true, lo, hi):
    if lo is None or hi is None: return (np.nan, np.nan)
    m = np.isfinite(lo) & np.isfinite(hi)
    if not m.any(): return (np.nan, np.nan)
    cover = float(np.mean((y_true[m]>=lo[m]) & (y_true[m]<=hi[m])))
    width = float(np.mean(hi[m]-lo[m]))
    return cover, width

def safe_mape(y_true, y_pred):
    eps = 1e-9
    t = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(np.abs(t-p)/np.maximum(np.abs(t), eps)))

def smape(y_true, y_pred):
    eps = 1e-9
    t = np.asarray(y_true); p = np.asarray(y_pred)
    return float(np.mean(2*np.abs(p-t)/(np.abs(t)+np.abs(p)+eps)))

def evaluate_block(df_pred: pd.DataFrame, target: str, horizon: int, ops_pred: Optional[pd.Series]) -> pd.DataFrame:
    rows = []
    for model, g in df_pred.groupby("model", sort=False):
        g = g.sort_values("date")
        yt = g["y_true"].to_numpy(); yp = g["y_pred"].to_numpy()
        mae  = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp)**2)))
        r2   = np.nan
        ss_tot = np.sum((yt - np.mean(yt))**2)
        if ss_tot>0: r2 = 1 - np.sum((yt - yp)**2)/ss_tot

        if is_stock(target):
            mape = np.nan; smp = np.nan
            m_true = pd.Series(yt, index=pd.to_datetime(g["date"])).resample("ME").last()
            m_pred = pd.Series(yp, index=pd.to_datetime(g["date"])).resample("ME").last()
        else:
            mape = safe_mape(yt, yp); smp = smape(yt, yp)
            m_true = pd.Series(yt, index=pd.to_datetime(g["date"])).resample("ME").sum()
            m_pred = pd.Series(yp, index=pd.to_datetime(g["date"])).resample("ME").sum()

        tol10 = float(np.mean((np.abs(m_true - m_pred)/np.maximum(np.abs(m_true),1e-9)) <= 0.10)) if len(m_true) else np.nan
        cover, width = coverage_width(yt, g["y_lo"].to_numpy(), g["y_hi"].to_numpy())

        mae_skill = np.nan
        if (ops_pred is not None) and (not is_stock(target)):
            aligned = g.set_index("date").join(ops_pred.rename("ops_pred"), how="left")
            if aligned["ops_pred"].notna().any():
                mae_ops = float(np.mean(np.abs(aligned["y_true"] - aligned["ops_pred"])))
                mae_skill = (1.0 - mae/mae_ops) if mae_ops>0 else np.nan

        rows.append({"target":target,"horizon":horizon,"cadence":CADENCE,"model":model,
                     "MAE":mae,"RMSE":rmse,"sMAPE":smp,"MAPE":mape,"R2":r2,
                     "PI_coverage@90":cover,"PI_width@90":width,
                     "Monthly_TOL10_Accuracy":tol10,"MAE_skill_vs_Ops":mae_skill})
    return pd.DataFrame(rows)

def plot_overlay_top(df_slice, target, h, best_model, ops_series, out_png):
    fig, ax = plt.subplots(figsize=(12,4))
    g = df_slice[df_slice["model"]==best_model].sort_values("date")
    ax.plot(g["date"], g["y_pred"], label=best_model, linewidth=2)
    if "y_lo" in g.columns and "y_hi" in g.columns and g["y_lo"].notna().any() and g["y_hi"].notna().any():
        ax.fill_between(g["date"], g["y_lo"], g["y_hi"], alpha=0.2, label="PI 90%")
    ax.plot(g["date"], g["y_true"], label="Actual", linewidth=2)
    if ops_series is not None:
        ops_aligned = ops_series.reindex(pd.to_datetime(g["date"]))
        ax.plot(g["date"], ops_aligned.values, label="Ops baseline", linestyle="--")
    ax.set_title(f"{target} | {CADENCE} | h=+{h} | ENSEMBLE vs Ops")
    ax.legend(loc="best"); ax.grid(True); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_leaderboard_bar(mdf, target, h, out_png):
    g = mdf.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(g["model"], g["MAE"]); ax.invert_yaxis()
    ax.set_title(f"{target} | {CADENCE} | h=+{h} | Ensemble leaderboard by MAE")
    ax.set_xlabel("MAE (lower is better)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_monthly_bars(df_slice, target, h, ops_series, out_png):
    idx = pd.to_datetime(df_slice["date"])
    s_true = pd.Series(df_slice["y_true"].to_numpy(), index=idx)
    s_pred = pd.Series(df_slice["y_pred"].to_numpy(), index=idx)
    if is_stock(target):
        m_true = s_true.resample("ME").last(); m_pred = s_pred.resample("ME").last()
    else:
        m_true = s_true.resample("ME").sum();  m_pred = s_pred.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12,4))
    x = m_true.index
    ax.bar(x - pd.Timedelta(days=3), m_true.values, width=6, label="Actual")
    ax.bar(x, m_pred.values, width=6, label="Ensemble", alpha=0.7)
    if ops_series is not None and (not is_stock(target)):
        ops_m = ops_series.reindex(idx).resample("ME").sum()
        ax.bar(x + pd.Timedelta(days=3), ops_m.values, width=6, label="Ops baseline", alpha=0.7)
    ax.set_title(f"{target} | {CADENCE} | h=+{h} | Monthly aggregates (Ensemble vs Ops)")
    ax.legend(loc="best"); ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ------------------ LOAD ------------------

def load_all_predictions(source_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for d in source_dirs:
        f = os.path.join(d, "predictions_long.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, parse_dates=["date"])
            if "cadence" not in df.columns:
                # infer cadence from path (best-effort)
                if "\\daily" in d or "/daily" in d:  df["cadence"]= "daily"
                elif "\\weekly" in d or "/weekly" in d: df["cadence"]="weekly"
                elif "\\monthly" in d or "/monthly" in d: df["cadence"]="monthly"
                else: df["cadence"]= CADENCE
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No predictions_long.csv found in SOURCE_DIRS.")
    P = pd.concat(frames, ignore_index=True)
    # keep only selected cadence
    P = P[P["cadence"].str.lower()==CADENCE.lower()].copy()
    # sanitize models
    P["model"] = P["model"].astype(str)
    return P

def load_ops_by_target(source_dirs: List[str]) -> Dict[str, pd.Series]:
    ops: Dict[str, pd.Series] = {}
    for d in source_dirs:
        for t in ["Revenues","Expenditure","State budget balance"]:
            f = os.path.join(d, f"{t}_ops_baseline_daily.csv")
            if os.path.exists(f) and t not in ops:
                s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["forecast"]
                if CADENCE=="daily":
                    ops[t] = s
                elif CADENCE=="weekly":
                    ops[t] = s.resample("W-FRI").sum()
                else:
                    ops[t] = s.resample("ME").sum()
    return ops

# ------------------ ENSEMBLING ------------------

def global_mae_by_model(P: pd.DataFrame) -> pd.Series:
    """Global MAE per model across all targets/horizons/dates (for ranking)."""
    err = (P["y_true"] - P["y_pred"]).abs()
    g = P.assign(err=err).groupby("model")["err"].mean().sort_values()
    return g  # index=model, values=MAE (ascending)

def build_ensembles(P: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    # 1) Median of all models available for a given target×horizon×date
    med = (P.groupby(["target","horizon","cadence","date"], as_index=False)
             .agg(y_true=("y_true","first"),
                  y_pred=("y_pred","median"),
                  y_lo=("y_lo","median"),
                  y_hi=("y_hi","median")))
    med["model"] = "ens_median"
    med["split_id"] = "ensemble"
    med["defn_variant"] = "Ensemble"

    # 2) Mean of top-K models by global MAE
    mae_rank = global_mae_by_model(P)  # ascending -> best first
    top_models = list(mae_rank.index[:max(1, top_k)])
    P_top = P[P["model"].isin(top_models)]
    mean_top = (P_top.groupby(["target","horizon","cadence","date"], as_index=False)
                   .agg(y_true=("y_true","first"),
                        y_pred=("y_pred","mean"),
                        y_lo=("y_lo","mean"),
                        y_hi=("y_hi","mean")))
    mean_top["model"]="ens_mean_topK"; mean_top["split_id"]="ensemble"; mean_top["defn_variant"]="Ensemble"

    # 3) Inverse-MAE weighted (global)
    inv_w = 1.0 / mae_rank
    inv_w = inv_w / inv_w.sum()
    W = P.merge(inv_w.rename("w").reset_index(), on="model", how="left")
    def _weighted_row(df):
        w = df["w"].to_numpy()
        y_true = df["y_true"].iloc[0]
        y_pred = np.average(df["y_pred"], weights=w)
        y_lo   = np.average(df["y_lo"],   weights=w) if df["y_lo"].notna().any() else np.nan
        y_hi   = np.average(df["y_hi"],   weights=w) if df["y_hi"].notna().any() else np.nan
        return pd.Series({"y_true":y_true, "y_pred":y_pred, "y_lo":y_lo, "y_hi":y_hi})
    inv_mean = (W.groupby(["target","horizon","cadence","date"], as_index=False)
                  .apply(_weighted_row, include_groups=False).reset_index(drop=True))
    inv_mean["model"]="ens_invMAE"; inv_mean["split_id"]="ensemble"; inv_mean["defn_variant"]="Ensemble"

    return pd.concat([med, mean_top, inv_mean], ignore_index=True)

# ------------------ MAIN ------------------

def main():
    out_root = os.path.join(OUT_ROOT, CADENCE)
    ensure_dirs(out_root)

    # Load predictions & Ops baseline
    P = load_all_predictions(SOURCE_DIRS)
    OPS = load_ops_by_target(SOURCE_DIRS)

    # Build ensembles per target×horizon×date
    ENS = build_ensembles(P, TOP_K)

    # Evaluate & plot per target×horizon
    metrics_rows = []
    for (t,h), df_th in ENS.groupby(["target","horizon"]):
        ops_series = OPS.get(t)
        metr = evaluate_block(df_th, t, h, ops_series)
        metrics_rows.append(metr)

        # Leaderboard for ensembles (per target×horizon)
        lb = metr.sort_values("MAE").reset_index(drop=True)
        lb.assign(target=t, horizon=h).to_csv(
            os.path.join(out_root, f"leaderboard_{t.replace(' ','_')}_h{h}.csv"), index=False
        )

        # Plots for the best ensemble
        best = lb.iloc[0]["model"]
        plot_overlay_top(df_th, t, h, best, ops_series,
                         os.path.join(out_root, f"{t.replace(' ','_')}_h{h}_overlay_top_ensemble.png"))
        plot_leaderboard_bar(metr, t, h,
                         os.path.join(out_root, f"{t.replace(' ','_')}_h{h}_leaderboard_mae_ensemble.png"))
        plot_monthly_bars(df_th[df_th["model"]==best], t, h, ops_series,
                         os.path.join(out_root, f"{t.replace(' ','_')}_h{h}_monthly_bars_top_vs_ops_ensemble.png"))

    METR = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    METR.to_csv(os.path.join(out_root, "metrics_ensemble.csv"), index=False)

    # Save predictions (all three ensembles)
    ENS.sort_values(["target","horizon","date","model"]).to_csv(
        os.path.join(out_root, "predictions_ensemble.csv"), index=False
    )

    # Global ensemble leaderboard (mean MAE across folds)
    rows=[]
    for (t,h), g in METR.groupby(["target","horizon"]):
        g2 = g.groupby("model", as_index=False)["MAE"].mean().sort_values("MAE")
        for rank,row in enumerate(g2.itertuples(index=False), start=1):
            rows.append({"target":t,"horizon":h,"model":row.model,"MAE":row.MAE,"rank":rank})
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "leaderboard_ensemble.csv"), index=False)

    print(f"[OK] Ensemble outputs → {out_root}")

if __name__ == "__main__":
    main()
