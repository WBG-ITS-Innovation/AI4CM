# Ai4CM Forecast Lab

A local sandbox for testing multiple forecasting model families on Treasury time-series, including Statistical, Machine Learning, Deep Learning, and Quantile models.  
Built to explore modernized cash-flow forecasting, model comparison, and capacity building.

## Repository Overview
```
ai4cm/
│
├── backend/                   # Model pipelines, runners, preprocessing
│   ├── run_a_stat.py          # A · Statistical
│   ├── run_b_ml_univariate.py # B · ML (uni)
│   ├── run_c_dl_univariate.py # C · DL (uni)
│   ├── run_e_quantile_*       # E · Quantile
│   ├── preprocess_data.py     # Excel → Daily data converter
│   ├── requirements.txt       # Backend dependencies
│   └── ...                    # Pipelines, utilities, data templates
│
├── frontend/                  # Streamlit application
│   ├── Overview.py            # Landing page
│   ├── pages/                 # Lab, Dashboard, History, Preprocessing, Models
│   ├── utils_frontend.py      # Run folders, backend linking
│   ├── backend_bridge.py      # Launch backend processes
│   ├── runs/                  # GENERATED: each experiment is saved here
│   ├── runs_uploads/          # GENERATED: uploaded data files
│   ├── requirements.txt       # Frontend dependencies
│   └── .tg_paths.json         # GENERATED: backend python + directory
│
├── scripts/                   # Setup & run scripts
│   ├── setup_windows.bat
│   ├── run_app_windows.bat
│   ├── setup_unix.sh
│   ├── run_app_unix.sh
│   └── verify_backend_env.py
│
├── data_preprocessed/         # GENERATED: cleaned daily datasets
├── .gitignore
└── README.md
```

> Important: Directories like `.venv/`, `frontend/runs/`, `frontend/runs_uploads/`, `data_preprocessed/`, and `frontend/.tg_paths.json` are generated automatically and should **NOT** be committed to Git.

---

## Model Families Supported

| Family | Description |
|-------|-------------|
| **A · Statistical** | ETS, SARIMAX, STL-ARIMA, Theta, moving-average/weekday baselines |
| **B · Machine Learning** | Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, HistGBDT, XGBoost, LightGBM |
| **C · Deep Learning** | LSTM, GRU, TCN, Transformer-style networks (with conformal prediction intervals) |
| **E · Quantile** | Quantile boosting models outputting P10 / P50 / P90 |

Each experiment produces:
- ✔ `predictions_long.csv`
- ✔ `metrics_long.csv`
- ✔ `leaderboard.csv`
- ✔ Plots in `plots/`
- ✔ Config in `artifacts/`

---

## 1. Requirements 
Treasury staff do **not** need Python knowledge. The setup scripts handle everything.

### **Option A — Use Git (recommended)**  
**Install Git (Windows)**  
Download: https://git-scm.com/download/win  
Verify:
```
git --version
```

**Install Python (Windows)**  
Download Python 3.11+:  
https://www.python.org/downloads/windows/  
Check “Add Python to PATH”.  
Verify:
```
python --version
```

### **Option B — No Git required**
Download ZIP:  
https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype → *Download ZIP*  
Extract to:
```
C:\Users\<name>\Documents\georgia-treasury-prototype
```
Python is still required.

---

## 2. Installation (Windows)

This installation is fully automatic.

### **If using Git**
```
cd C:\Users\<your_name>\Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git
cd georgia-treasury-prototype
scripts\setup_windows.bat
scripts\run_app_windows.bat
```

### **If using ZIP**
```
cd C:\Users\<your_name>\Documents
cd georgia-treasury-prototype
scripts\setup_windows.bat
scripts\run_app_windows.bat
```

App opens at:  
http://localhost:8501

---

## 3. Installation (macOS / Linux)

```
cd ~/Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git
cd georgia-treasury-prototype
./scripts/setup_unix.sh
./scripts/run_app_unix.sh
```

---

## 4. Using the Streamlit App

### **4.1 Overview Page**
- Quick links  
- Backend Python auto-detection  
- Recent runs + log preview + ZIP downloads  

### **4.2 Data Pre-processing**
Convert raw Excel → standardized daily dataset.

Steps:
1. Upload `Balance_by_Day_2015-2025.xlsx`
2. Enter column names (optional)
3. Choose variant: `raw / clean_conservative / clean_treasury`
4. Run preprocessing  
Outputs saved in `data_preprocessed/<variant>/`

### **4.3 Lab – Running Forecast Models**
Select:
- Target  
- Cadence (Daily/Weekly/Monthly)  
- Horizon  
- Family (A/B/C/E)  
- Variant (uni/multi)  
- Run profile (Demo/Balanced/Thorough)

Outputs saved in:  
`frontend/runs/run_<family>_<model>_<target>_<cadence>_h<h>_<timestamp>/`

### **4.4 Dashboard**
- View Actual vs Baseline vs Models  
- Metrics, leaderboards  
- Residuals  
- Prediction intervals  

### **4.5 History**
- All runs  
- Log tail  
- Download predictions, metrics, artifacts ZIP  

---

## 5. Example Experiments (UI)

### **Example 1 — A · Statistical (ETS)**
Target: Revenues  
Cadence: Monthly  
Horizon: 6  
Model: ETS  

Run folder:
```
frontend/runs/run_A_uni_ETS_Revenues_Monthly_h6_YYYYMMDD/
```

### **Example 2 — B · Machine Learning (Ridge)**
Target: State budget balance  
Cadence: Daily  
Horizon: 1  
Model: Ridge  
Lags: `[1,3,7,14]`  

### **Example 3 — C · Deep Learning (LSTM)**
Lookback: 48–90  
Epochs: 3–20  

---

## 6. Running Pipelines Directly (Backend CLI)

### Activate backend environment
```
cd backend
.\.venv\Scripts\Activate.ps1
```
(macOS/Linux: `source .venv/bin/activate`)

### **Example A — ETS**
```
python run_a_stat.py
```

### **Example B — Ridge**
```
python run_b_ml_univariate.py
```

### **Example C — LSTM**
```
python run_c_dl_univariate.py
```

Outputs appear in `frontend/runs/`.

---

## 7. Accessing Run Outputs

Each run contains:
```
backend_run.log
outputs/
   predictions_long.csv
   metrics_long.csv
   leaderboard.csv
   plots/
   artifacts/
```

---

## 8. Troubleshooting

### Backend Python missing  
Re-run `scripts/setup_windows.bat`.

### Missing CSV in Dashboard  
Check `backend_run.log`.

### Excel errors  
Ensure correct date/balance columns.

### Torch install problems  
Use CPU wheel from: https://pytorch.org/get-started/locally/

---

## Summary
This prototype provides:

- Fully automated installation  
- Multi-model forecasting (Stat, ML, DL, Quantile)  
- Streamlit UI for Treasury analysts  
- Backend CLI for developers  
- Reproducibility: logs, predictions, metrics, plots  

A complete forecasting laboratory for Georgia Treasury.
