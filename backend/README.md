
# Cashflow Experiments UI

A demo-ready Dash app with **experiment memory**, **async model runs**, and interactive Plotly charts.

## Features
- Upload datasets (CSV/Excel) → normalized to daily (`date`, `balance` and/or `net_flow`).
- Run models asynchronously (Diskcache long callbacks), so the UI stays responsive.
- **Experiment history** persists in SQLite (`data/experiments.db`), with metrics + artifacts on disk.
- Interactive forecast chart, confidence band, range slider; heatmap of net flows.
- Click any past run to reopen its forecast and metrics.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.app
# open http://127.0.0.1:8060
```

## Add models
Create a file in `models/` with:
```python
def predict(df, horizon:int, params:dict) -> dict:
    return {"forecast": DataFrame[date,yhat,yhat_lower,yhat_upper],
            "metrics": {"mae":..., "rmse":..., "mape":...},
            "details": {...}}
```
Register it in `models/registry.py`.

## Persisted memory
- SQLite DB: `data/experiments.db` (tables: datasets, runs).
- Artifacts per run: `artifacts/<run_id>/forecast.csv`, `metrics.json`.
- Replace SQLite with Postgres later without changing the UI (DB functions are isolated in `core/db.py`).

## Next steps
- Swap Diskcache for Celery/RQ worker when runs take longer; add FastAPI for `/forecast` and `/experiments` endpoints.
- Add leaderboards (multi-model bakeoffs) and scenario sliders (AR/AP timing).

