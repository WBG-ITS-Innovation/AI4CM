# make_ml_heatmaps.py
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAMILIES = ["ml_univariate","ml_multivariate"]
CADENCES = ["daily","weekly","monthly"]

def _heatmap(ax, data, row_labels, col_labels, title, vmin=None, vmax=None, fmt="{:.2f}"):
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title); ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i,j]
            s = "" if (np.isnan(val) or np.isinf(val)) else (fmt.format(val))
            ax.text(j, i, s, ha="center", va="center", fontsize=8)
    return im

def _save_heatmap(df, target, cadence, metric, out_dir, higher_is_better=False, decimals=2):
    piv = df.pivot_table(index="model", columns="horizon", values=metric, aggfunc="mean")
    if piv.empty: return
    models = list(piv.index)
    horizons = list(piv.columns)
    data = piv.to_numpy()
    vals = data[np.isfinite(data)]
    if vals.size == 0: return
    vmin, vmax = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
    fig, ax = plt.subplots(figsize=(1.2*len(horizons)+3, 0.5*len(models)+3))
    im = _heatmap(ax, data, models, horizons,
                  title=f"{target} | {cadence} | {metric}",
                  vmin=vmin if not math.isinf(vmin) else None,
                  vmax=vmax if not math.isinf(vmax) else None,
                  fmt="{:." + str(decimals) + "f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{target.replace(' ','_')}_heatmap_{metric.replace('@','_').replace('%','p')}.png"))
    plt.close(fig)

for fam in FAMILIES:
    for cad in CADENCES:
        mpath = os.path.join("outputs", fam, cad, "metrics_long.csv")
        if not os.path.exists(mpath):
            print(f"[WARN] Missing metrics: {mpath}")
            continue
        m = pd.read_csv(mpath)
        if m.empty:
            continue
        for t, g in m.groupby("target"):
            out_dir = os.path.join("outputs", fam, cad, "plots")
            for metric in ["MAE","RMSE","sMAPE","MAPE"]:               # lower is better
                if metric in g.columns: _save_heatmap(g, t, cad, metric, out_dir, False, 2)
            for metric in ["R2","PI_coverage@90","Monthly_TOL10_Accuracy","MAE_skill_vs_Ops"]:  # higher is better
                if metric in g.columns: _save_heatmap(g, t, cad, metric, out_dir, True, 2)
        print(f"[OK] Heatmaps → outputs/{fam}/{cad}/plots")
