# Georgia Treasury • Forecast Lab

A production-grade sandbox to compare multiple forecasting families on Treasury time series.

- **Families**
  - **A · Statistical** (ETS, SARIMAX, STL-ARIMA, Theta, baselines)
  - **B · Machine Learning** (Ridge/Lasso/ElasticNet, Random Forest, Extra Trees, HistGBDT, XGBoost, LightGBM)
  - **C · Deep Learning** (LSTM / GRU / TCN / Transformer-style models with conformal intervals)
  - **E · Quantile** (gradient-boosted quantile models for P10/P50/P90)
- **Workflow**
  1. **Data Pre-processing** — standardize raw Excel/CSV into a “master daily” dataset.
  2. **Lab** — configure and run models (single family/target/horizon) with full control and live logs.
  3. **Dashboard** — explore Actual vs. baseline vs. predictions + metrics + residuals.
  4. **History** — browse and download artifacts from past runs.
  5. **Models** — documentation on every family and key parameters.

---

## 1. Repository layout

```text
georgia-treasury-lab/
  backend/                  # model pipelines and runners (no venv; see scripts)
    a_stat_models_pipeline.py
    b_ml_pipeline.py
    c_dl_pipeline.py
    e_quantile_daily_pipeline.py
    preprocess_data.py
    run_a_stat.py
    run_b_ml_univariate.py
    run_b_ml_multivariate.py
    run_c_dl_univariate.py
    run_c_dl_multivariate.py
    run_c_dl_quick_univariate.py
    run_c_dl_quick_multivariate.py
    run_e_quantile_daily_univariate.py
    run_e_quantile_daily_multivariate.py
    run_preprocess.py
    requirements.txt
    utils/...
  frontend/                 # Streamlit application
    Overview.py             # landing page, quick links, backend paths, recent runs
    app.py                  # (if present) entry alias
    utils_frontend.py       # paths, run folder management
    backend_bridge.py       # launches backend runners with live logs
    pages/
      00_Data_Preprocessing.py
      00_Lab.py
      01_Dashboard.py
      02_History.py
      03_Models.py
    runs/                   # GENERATED: all run folders (ignored by Git)
    runs_uploads/           # GENERATED: uploaded files (ignored by Git)
    .tg_paths.json          # GENERATED: backend python + dir (ignored by Git)
  scripts/                  # helper scripts for setup and running
    setup_windows.bat
    run_app_windows.bat
    setup_unix.sh
    run_app_unix.sh
    verify_backend_env.py   # optional env self-test
  data_preprocessed/        # GENERATED: pre-processed daily datasets (ignored by Git)
  .gitignore
  README.md
