# make_weekly_from_daily_stat.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
RUN_DIR = r".\outputs\statistical"        # <-- point this to the folder that contains predictions_long.csv
WEEKLY_OUT = os.path.join(RUN_DIR, "weekly")
TARGET_STOCK = {"state budget balance","balance","t0"}  # case-insensitive

os.makedirs(os.path.join(WEEKLY_OUT,"plots"), exist_ok=True)

def is_stock(target:str)->bool:
    return target.strip().lower() in TARGET_STOCK

def resample_weekly(df_pred_tmh):
    """Resample daily predictions to W-FRI. For flows: sum; for stocks: last."""
    idx = pd.to_datetime(df_pred_tmh["date"])
    agg = "last" if is_stock(df_pred_tmh["target"].iloc[0]) else "sum"
    g = df_pred_tmh.assign(date=pd.to_datetime(df_pred_tmh["date"]))
    y_true_w = g.set_index("date")["y_true"].resample("W-FRI").agg(agg)
    y_pred_w = g.set_index("date")["y_pred"].resample("W-FRI").agg(agg)
    y_lo_w   = g.set_index("date")["y_lo"].resample("W-FRI").agg(agg)
    y_hi_w   = g.set_index("date")["y_hi"].resample("W-FRI").agg(agg)
    out = pd.DataFrame({
        "date": y_pred_w.index,
        "y_true": y_true_w.values,
        "y_pred": y_pred_w.values,
        "y_lo": y_lo_w.values,
        "y_hi": y_hi_w.values,
    })
    return out

def safe_mape(y_true, y_pred):
    eps=1e-9
    y_true = np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(np.mean(np.abs(y_true-y_pred)/np.maximum(np.abs(y_true),eps)))

def smape(y_true, y_pred):
    eps=1e-9
    y_true = np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+eps)))

def evaluate_block(df_pred, target, horizon, ops_weekly=None):
    rows=[]
    for model, g in df_pred.groupby("model"):
        g = g.sort_values("date")
        y_true = g["y_true"].to_numpy(); y_pred = g["y_pred"].to_numpy()
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        r2   = np.nan
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot>0: r2 = 1 - np.sum((y_true - y_pred)**2)/ss_tot
        mape = np.nan if is_stock(target) else safe_mape(y_true,y_pred)
        smp  = np.nan if is_stock(target) else smape(y_true,y_pred)
        cover = np.nan; width = np.nan
        if g["y_lo"].notna().any() and g["y_hi"].notna().any():
            m = np.isfinite(g["y_lo"]) & np.isfinite(g["y_hi"])
            if m.any():
                cover = float(np.mean((g["y_true"][m]>=g["y_lo"][m]) & (g["y_true"][m]<=g["y_hi"][m])))
                width = float(np.mean(g["y_hi"][m]-g["y_lo"][m]))
        mae_skill = np.nan
        if (ops_weekly is not None) and (not is_stock(target)):
            aligned = g.set_index("date").join(ops_weekly.rename("ops_pred"), how="left")
            if aligned["ops_pred"].notna().any():
                mae_ops = float(np.mean(np.abs(aligned["y_true"] - aligned["ops_pred"])))
                mae_skill = (1.0 - mae/mae_ops) if mae_ops>0 else np.nan
        rows.append({"target":target,"horizon":horizon,"model":model,"MAE":mae,"RMSE":rmse,"sMAPE":smp,"MAPE":mape,
                     "R2":r2,"PI_coverage@90":cover,"PI_width@90":width,"MAE_skill_vs_Ops":mae_skill})
    return pd.DataFrame(rows)

# Load daily predictions
pred_path = os.path.join(RUN_DIR, "predictions_long.csv")
pred = pd.read_csv(pred_path, parse_dates=["date"])
pred["target"] = pred["target"].astype(str)
pred["model"]  = pred["model"].astype(str)

# Optional: weekly Ops baseline by summing daily Ops files (flows only)
ops_weekly = {}
for t in pred["target"].unique():
    if is_stock(t): 
        ops_weekly[t] = None
        continue
    f = os.path.join(RUN_DIR, f"{t}_ops_baseline_daily.csv")
    if os.path.exists(f):
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")["forecast"]
        ops_weekly[t] = d.resample("W-FRI").sum()

# Build weekly predictions
weekly_preds = []
weekly_metrics = []

for (t,h,m), g in pred.groupby(["target","horizon","model"]):
    w = resample_weekly(g)
    w["target"]=t; w["horizon"]=h; w["model"]=m
    weekly_preds.append(w)

# Combine and evaluate
if weekly_preds:
    W = pd.concat(weekly_preds, ignore_index=True)
    W = W.sort_values(["target","horizon","date","model"])
    W.to_csv(os.path.join(WEEKLY_OUT, "predictions_long_weekly.csv"), index=False)

    for (t,h), g in W.groupby(["target","horizon"]):
        ops_w = ops_weekly.get(t)
        metr = evaluate_block(g, t, h, ops_weekly=ops_w)
        weekly_metrics.append(metr)

    M = pd.concat(weekly_metrics, ignore_index=True)
    M.to_csv(os.path.join(WEEKLY_OUT, "metrics_long_weekly.csv"), index=False)

    # Leaderboard and plots
    rows=[]
    for (t,h), gg in M.groupby(["target","horizon"]):
        lb = gg.sort_values("MAE").reset_index(drop=True)
        for r,row in enumerate(lb.itertuples(index=False), start=1):
            rows.append({"target":t,"horizon":h,"model":row.model,"MAE":row.MAE,"rank":r})

        # plots: top vs actual vs ops
        best = lb.iloc[0]["model"]
        p = W[(W["target"]==t) & (W["horizon"]==h) & (W["model"]==best)].sort_values("date")

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(p["date"], p["y_pred"], label=f"{best} (top)", linewidth=2)
        if p["y_lo"].notna().any() and p["y_hi"].notna().any():
            ax.fill_between(p["date"], p["y_lo"], p["y_hi"], alpha=0.2, label="PI 90%")
        ax.plot(p["date"], p["y_true"], label="Actual", linewidth=2)
        if ops_weekly.get(t) is not None:
            ops_al = ops_weekly[t].reindex(p["date"])
            ax.plot(p["date"], ops_al.values, label="Ops baseline (weekly sum)", linestyle="--")
        ax.set_title(f"{t} | Weekly | h=+{h} | TOP vs Actual vs Ops")
        ax.legend(); ax.grid(True); fig.tight_layout()
        fig.savefig(os.path.join(WEEKLY_OUT,"plots", f"{t.replace(' ','_')}_h{h}_weekly_top.png")); plt.close(fig)

        # leaderboard bar
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(lb["model"], lb["MAE"]); ax.invert_yaxis()
        ax.set_title(f"{t} | Weekly | h=+{h} | Leaderboard by MAE"); ax.set_xlabel("MAE (lower is better)")
        fig.tight_layout(); fig.savefig(os.path.join(WEEKLY_OUT,"plots", f"{t.replace(' ','_')}_h{h}_weekly_leaderboard.png")); plt.close(fig)

    pd.DataFrame(rows).to_csv(os.path.join(WEEKLY_OUT, "leaderboard_weekly.csv"), index=False)
    print(f"[OK] Weekly outputs written to: {WEEKLY_OUT}")
else:
    print("[WARN] No weekly predictions created.")
