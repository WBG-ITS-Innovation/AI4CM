
# A. Statistical (Baselines & Calendar-Aware) — Runner

This script trains and evaluates the **A-cluster** statistical models:
- **Baselines:** naive last, weekday mean, 7-day moving average
- **Calendar-aware statistical:** SARIMAX (with exogenous calendar), ETS (Holt-Winters), STL + ARIMA

It generates standardized outputs that integrate with our comparison framework.

## Targets
- **Revenues** (flow) — daily amounts
- **Expenditure** (flow) — daily amounts
- **State budget balance** (stock) — end-of-day level

> For A-cluster, we forecast **totals** above. Component-level models are **not required** here and are covered in the **C-cluster** (multivariate & hierarchical). If desired, you can add `Taxes` as an extra total target.

## Features (exogenous calendar for SARIMAX)
- Day-of-week one-hot (drop one)
- `is_weekend`
- Month number, day-of-year
- `is_holiday` (from data if available)
- `is_eom` (end-of-month), `is_eoq` (end-of-quarter)

ETS and STL+ARIMA are **univariate** (no exog). Baselines use the training window only.

## Horizons & Folds
- Horizons: **D+1, D+5, D+20** (configurable)
- Folds: two rolling-origin folds
  - Train ≤ 2023-12-31 → Test 2024-01-01…2024-12-31
  - Train ≤ 2024-12-31 → Test 2025-01-01…2025-08-06

## Ops Baseline (always included for flows)
Implements the Treasury baseline:
- **Monthly**: 3-year average of the same month
- **Daily**: distribute monthly forecast using 3-year **within-month daily profile** (or switch to `daily_avg`)

Emits:
- `ops_baseline/<Target>_ops_baseline_monthly.csv`
- `ops_baseline/<Target>_ops_baseline_daily.csv`

## Outputs
- `predictions/predictions_long.csv` — one row per date×model×target×horizon×fold (with 90% PI when available)
- `metrics/metrics_long.csv` — MAE, RMSE, sMAPE/MAPE (flows), R², PI coverage/width, Monthly_TOL10_Accuracy, Skill vs Ops
- `leaderboards/leaderboard.csv` — rank by MAE (per target×horizon)
- `plots/*overlay*` — ALL vs Ops and TOP vs Ops overlays for each target×horizon×fold
- `artifacts/RUN.md` and `artifacts/config.json`

## Usage
```python
from a_stat_models_pipeline import run_pipeline, CONFIG
# (Optional) Adjust CONFIG targets, folds, models, horizons, seasonal_period, etc.
CONFIG.targets = ["Revenues","Expenditure","State budget balance"]
CONFIG.horizons = [1,5,20]
CONFIG.run_models = ["naive_last","weekday_mean","movavg7","sarimax","ets","stl_arima"]
run_dir = run_pipeline(CONFIG)
print("Outputs:", run_dir)
```

## Notes
- **Stocks vs flows**: Balance is **not** summed; monthly metrics use **EOM last** for Balance, **sum** for flows.
- **Intervals**: SARIMAX/ETS/STL+ARIMA produce 90% PIs; baselines omit PIs by default.
- **Performance**: SARIMAX & STL+ARIMA refit every origin (rolling) — accurate but can be slow; increase `refit_every` or switch to less frequent refits in future iterations for speed.

