#!/usr/bin/env python
"""Workstream 2 tuning driver — Optuna search per target, then one DEV confirmation.

    TARGETS="['Revenues','Expenditure','State budget balance']" TRIALS=100 \
      SC=/tmp/ws2 ./backend/.venv/bin/python scripts/ws2_tune.py

Lives in the repository rather than a scratch directory because a ~45-minute search whose
driver is not version-controlled cannot be reproduced or resumed. Per-target results are
appended to experiments/log.csv as each target completes, so an interrupted run keeps
whatever it finished; the summary CSV under $SC is a convenience, not the record.

Searches TRAIN-internal rolling-origin folds only. DEV is touched once per target, after
that target's search closes. TEST is never touched.
"""

import sys, os, json, time, ast
sys.path.insert(0,"backend")
import numpy as np, pandas as pd
from b_ml_pipeline import (ConfigBML, to_business_index, calendar_exog, choose_recipe,
                           lag_window_features, is_stock, build_yearly_folds)
from forecast_integrity import compute_persistence_baseline, signal_sentinel
from evaluation_windows import seasonal_naive_scale, mase
from provenance import describe_input, describe_code
from experiment_log import log_run
from preprocessing.fiscal_calendar import calendar_version
from preprocessing.multivariate import build_exog_features
from target_scaling import trailing_level
from tuning import FoldData, fit_point, fit_quantiles, objective_mae, run_study, QUANTILES
from registry import load_registry

SC=os.environ["SC"]; DATA="backend/data/processed/master_daily_clean_treasury.csv"
di=describe_input(DATA); code=describe_code(); CALV=calendar_version()
raw=pd.read_csv(DATA); H=5
rawi=pd.read_csv(DATA,parse_dates=["date"]).set_index("date")
bidx=pd.date_range(rawi.index.min().normalize(),rawi.index.max().normalize(),freq="B")
REG={r["target"]:r for r in load_registry()["recipes"]}

def design(target):
    rec=REG[target]; s=to_business_index(raw,"date",target)
    cfg=ConfigBML(target=target,cadence="Daily",horizon=H,data_path="",date_col="date",
                  model_filter=None,variant="univariate",out_root="",
                  fiscal_groups=tuple(rec["feature_groups"]))
    lags,wins=choose_recipe(cfg)
    cal=calendar_exog(s.index,y=s,fiscal_groups=tuple(rec["feature_groups"]))
    if rec.get("exog_blocks"):
        cal=pd.concat([cal,build_exog_features(raw,target,list(rec["exog_blocks"]),s.index,
                                               date_col="date")],axis=1)
    L=list(lags)
    if is_stock(target) and 0 not in L: L=[0]+L
    X=lag_window_features(s,L,wins).join(cal)
    stock=is_stock(target)
    y_t=(s.shift(-H)-s) if stock else s.shift(-H)     # modelling target
    y_true=s.shift(-H)                                 # published truth (level for stock)
    tf=rec["params"].get("target_transform","raw")
    lvl=trailing_level(s.diff(H) if stock else s) if tf=="ratio" else None
    return s,X,y_t,y_true,tf,lvl,stock

def make_folds(target, window):
    s,X,y_t,y_true,tf,lvl,stock=design(target)
    lo,hi=(None,"2023-12-31") if window=="train" else ("2024-01-01","2024-12-31")
    folds=build_yearly_folds(s.index,4,None,eval_start=lo,eval_end=hi)
    out=[]
    for (tr_end,te_start,te_end) in folds:
        mtr=(X.index<=tr_end); mte=(X.index>=te_start)&(X.index<=te_end)
        ok=X.notna().all(axis=1)&y_t.notna()
        itr=X.index[mtr&ok]; ite=X.index[mte&ok]
        if len(itr)<200 or len(ite)==0: continue
        ytr=y_t.loc[itr].to_numpy(float); yte_true=y_true.loc[ite].to_numpy(float)
        origin=s.loc[ite].to_numpy(float)
        inv=None
        if tf=="ratio":
            ltr=lvl.reindex(itr).to_numpy(float); lte=lvl.reindex(ite).to_numpy(float)
            ytr=ytr/ltr
            if stock: inv=(lambda p,l=lte,o=origin: p*l+o)
            else:     inv=(lambda p,l=lte: p*l)
        elif stock:
            inv=(lambda p,o=origin: p+o)
        out.append(FoldData(X_tr=X.loc[itr],y_tr=ytr,X_te=X.loc[ite],
                            y_te=yte_true,inverse=inv))
    return out,(s,X,y_t,y_true,tf,lvl,stock)



def full_study(target, n_trials):
    rec=REG[target]
    folds,ctx=make_folds(target,"train")
    s,X,y_t,y_true,tf,lvl,stock=ctx
    t0=time.time()
    res=run_study(folds,["LGBMQuantile","CatBoostQuantile"],H,n_trials,seed=0,
                  progress=lambda n,v,m: print(f"  trial {n:>3} {m:<18}{v:>15,.0f}",flush=True)
                            if n%10==0 else None)
    res["seconds"]=round(time.time()-t0,1)
    print(f"[{target}] best={res['best_model']} obj={res['best_value']:,.0f} "
          f"trials={res['n_trials_completed']}/{res['n_trials_requested']} "
          f"failed={res['n_trials_failed']} in {res['seconds']}s",flush=True)

    # DEV confirmation, once
    dfolds,_=make_folds(target,"dev")
    assert len(dfolds)==1, f"expected one DEV fold, got {len(dfolds)}"
    f=dfolds[0]
    qp,ncross=fit_quantiles(res["best_model"],f.X_tr,f.y_tr,f.X_te,res["best_params"],H)
    if f.inverse is not None:
        qp={q:np.asarray(f.inverse(v),dtype=float) for q,v in qp.items()}
    y=f.y_te; p50=qp[0.50]
    mae=float(np.mean(np.abs(y-p50)))
    ite=f.X_te.index
    origin=s.reindex(ite).to_numpy(float)
    pdf=pd.DataFrame({"target_date":[d+pd.offsets.BDay(H) for d in ite],
                      "y_true":y,"origin_value":origin,"y_pred":p50})
    ruler=compute_persistence_baseline(pdf.dropna())["mae_persistence"]
    cov=float(np.mean((y>=qp[0.10])&(y<=qp[0.90])))
    q=pd.qcut(pd.Series(np.abs(y)),3,labels=["low","mid","high"])
    terc={l: float(np.mean((y[(q==l).to_numpy()]>=qp[0.10][(q==l).to_numpy()])&
                           (y[(q==l).to_numpy()]<=qp[0.90][(q==l).to_numpy()])))
          for l in ("low","mid","high")}
    sent=signal_sentinel(f.X_tr,pd.Series(f.y_tr),f.X_te,pd.Series(f.y_tr[:len(f.X_te)]),H) \
         if False else None
    # sentinel on the same design, true targets, TRAIN->DEV
    sent=signal_sentinel(f.X_tr,pd.Series(f.y_tr,index=f.X_tr.index),
                         f.X_te,pd.Series(y,index=f.X_te.index),H)
    ser=(rawi[target].reindex(bidx).ffill() if stock else rawi[target].reindex(bidx).fillna(0.0))
    d=log_run(target=target,model=f"{res['best_model']}_tuned",git_sha=code["git_sha"],
        data_sha=di["sha256"],feature_names=[f"ws2:{rec['id']}"],
        params={"study":"ws2_tuning","model":res["best_model"],
                "best_params":res["best_params"],"target_transform":tf,
                "n_trials_completed":res["n_trials_completed"],"horizon":H,
                "early_stopping":"h-gapped","quantiles":list(QUANTILES)},
        seed=0,fold_scheme="Optuna on TRAIN-internal yearly folds 2019-2023; one DEV fold",
        dev_mae=mae,mase=mase(pdf["y_true"],pdf["y_pred"],
                              seasonal_naive_scale(ser,season=5,window="train")),
        skill_vs_ruler=(ruler-mae)/ruler*100,
        sentinel_ratio=sent["shuffled_to_normal_ratio"],coverage=terc,ruler=ruler,
        calendar_version=CALV,
        note=f"WS2 tuned {res['best_model']}, DEV n={len(y)}, crossings={ncross}")
    out=dict(target=target,best_model=res["best_model"],train_obj=res["best_value"],
             trials=res["n_trials_completed"],failed=res["n_trials_failed"],
             seconds=res["seconds"],dev_mae=mae,ruler=ruler,
             skill=(ruler-mae)/ruler*100,coverage=cov,
             cov_low=terc["low"],cov_mid=terc["mid"],cov_high=terc["high"],
             sentinel=sent["shuffled_to_normal_ratio"],crossings=ncross,
             best_params=json.dumps(res["best_params"]),run_id=d["run_id"])
    print(f"[{target}] DEV MAE={mae:,.0f} skill={(ruler-mae)/ruler*100:.2f}% "
          f"cov={cov:.1%} sentinel={sent['shuffled_to_normal_ratio']:.4f} "
          f"crossings={ncross}",flush=True)
    return out


if __name__=="__main__":
    n_trials=int(os.environ.get("TRIALS","100"))
    RES=os.path.join(SC,"ws2_results.csv")
    rows=pd.read_csv(RES).to_dict("records") if os.path.exists(RES) else []
    done={r["target"] for r in rows}
    for target in ast.literal_eval(os.environ["TARGETS"]):
        if target in done:
            print(f"[{target}] already done, skipping",flush=True); continue
        rows.append(full_study(target,n_trials))
        pd.DataFrame(rows).to_csv(RES,index=False)
    print("WROTE",RES,flush=True)
