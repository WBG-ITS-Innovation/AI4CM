# Georgia Treasury • Forecast Lab

A production-grade sandbox for testing multiple forecasting model families on Treasury time-series, including Statistical, Machine Learning, Deep Learning, and Quantile models.
Built for Georgia Treasury operations to support modernized cash-flow forecasting, model comparison, and capacity building.

📁 Repository Overview
georgia-treasury-prototype/
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

🧩 Model Families Supported

| Family                   | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **A · Statistical**      | ETS, SARIMAX, STL-ARIMA, Theta, moving-average/weekday baselines.                  |
| **B · Machine Learning** | Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, HistGBDT, XGBoost, LightGBM. |
| **C · Deep Learning**    | LSTM, GRU, TCN, Transformer-style networks (with conformal prediction intervals).  |
| **E · Quantile**         | Boosted quantile models outputting P10 / P50 / P90 forecasts.                      |

Each experiment produces:
✔ predictions_long.csv
✔ metrics_long.csv
✔ leaderboard.csv
✔ Plots in plots/
✔ Config dumps in artifacts/

1. Requirements

Option A — Use Git (recommended)
Install Git (Windows)

Download and install Git:
https://git-scm.com/download/win

Verify:
git --version

Install Python (Windows)

Download Python 3.11+ (3.13 supported):
https://www.python.org/downloads/windows/

During install → Check “Add Python to PATH”.

Verify: python --version

Option B — Do NOT install Git

You can download the ZIP directly:

Go to the repo page:
https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype

Click Code → Download ZIP

Extract it anywhere, e.g.:
C:\Users\<name>\Documents\georgia-treasury-prototype

You still need Python installed (see above).

⚙️ 2. Installation (Windows)

This installation is fully automatic.

If using Git:
cd C:\Users\<your_name>\Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git
cd georgia-treasury-prototype
scripts\setup_windows.bat
scripts\run_app_windows.bat

If using the ZIP download:
cd C:\Users\<your_name>\Documents
# (extract ZIP here)
cd georgia-treasury-prototype
scripts\setup_windows.bat
scripts\run_app_windows.bat


After running run_app_windows.bat, your browser opens at:

http://localhost:8501

🧪 3. Installation (macOS / Linux)
cd ~/Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git
cd georgia-treasury-prototype

./scripts/setup_unix.sh
./scripts/run_app_unix.sh


App opens at:

http://localhost:8501

