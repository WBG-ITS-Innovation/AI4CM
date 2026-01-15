# AI4CM Forecast Lab

AI4CM is a local forecasting sandbox for Treasury time-series: upload a dataset, run multiple model families (Statistical, Machine Learning, Deep Learning, Quantile), and compare results in a Streamlit UI.  
It is designed for hands-on exploration, model comparison, and capacity building.

---

## What this repo does (in one minute)

1. **You bring a CSV** (or generate one via the Data Pre-processing tab).
2. The **Streamlit frontend** helps you pick:
   - the **target series** to forecast (e.g., revenue, balance, inflows)
   - the **cadence** (daily/weekly/monthly)
   - the **forecast horizon** (how far ahead)
   - the **model family** (A/B/C/E)
3. The UI launches the selected **backend runner** in a separate Python environment.
4. Each run produces:
   - forecasts (`predictions_long.csv`)
   - evaluation metrics (`metrics_long.csv`, `leaderboard.csv`)
   - plots and artifacts
   - a log file (`backend_run.log`)
5. You browse and download results from **Dashboard** and **History**.

---

## Repository Overview

```
AI4CM/
│
├── backend/                   # Model pipelines, runners, preprocessing
│   ├── run_a_stat.py          # A · Statistical
│   ├── run_b_ml_univariate.py # B · ML (uni)
│   ├── run_c_dl_univariate.py # C · DL (uni)
│   ├── run_e_quantile_*       # E · Quantile
│   ├── preprocess_data.py     # Excel → Daily data converter (used by UI)
│   ├── requirements.txt       # Backend dependencies
│   └── ...                    # Pipelines, utilities, templates
│
├── frontend/                  # Streamlit application
│   ├── Overview.py            # Landing page (start here)
│   ├── pages/                 # Lab, Dashboard, History, Preprocessing, Models
│   ├── utils_frontend.py      # Run folders + backend linking
│   ├── backend_bridge.py      # Launch backend processes
│   ├── runs/                  # GENERATED: each experiment saved here
│   ├── runs_uploads/          # GENERATED: uploaded datasets
│   ├── requirements.txt       # Frontend dependencies
│   └── .tg_paths.json         # GENERATED: backend python + directory
│
├── scripts/                   # Setup & run scripts (fastest path)
│   ├── setup_windows.bat
│   ├── run_app_windows.bat
│   ├── setup_unix.sh
│   ├── run_app_unix.sh
│   └── verify_backend_env.py
│
├── data_preprocessed/         # GENERATED: cleaned daily datasets from Excel
├── .gitignore
└── README.md
```

> Important: directories like `.venv/`, `frontend/runs/`, `frontend/runs_uploads/`, `data_preprocessed/`, and `frontend/.tg_paths.json` are generated automatically and should **NOT** be committed to Git.

---

## Model Families Supported

| Family | What it is | Typical use |
|---|---|---|
| **A · Statistical** | ETS, SARIMAX, STL-ARIMA, Theta, simple baselines | Strong, explainable baselines; good first pass |
| **B · Machine Learning** | Ridge/Lasso/ElasticNet, RandomForest, ExtraTrees, HistGBDT, XGBoost, LightGBM | Uses lag/rolling features; good when signals are nonlinear |
| **C · Deep Learning** | LSTM, GRU, TCN, Transformer-like nets | Sequence models; useful when enough history exists |
| **E · Quantile** | Quantile boosting models (P10/P50/P90) | Risk-aware ranges rather than a single point |

Each experiment produces:
- `predictions_long.csv`
- `metrics_long.csv`
- `leaderboard.csv`
- plots under `plots/`
- config + artifacts under `artifacts/`

---

## 1) Prerequisites (install once)

### 1.1 Install Python (macOS + Windows)

**Recommended version:** Python **3.11** (3.12 also works).  
Avoid Python 3.13 unless you know you need it.

**Windows**
1. Download Python from python.org
2. During installation, check **“Add Python to PATH”**
3. Verify in PowerShell:
   ```powershell
   python --version
   ```
   or
   ```powershell
   py --version
   ```

**macOS**
You have 3 good options:
- **Option A (easiest):** Install Python 3.11 from python.org
- **Option B:** Install Homebrew, then `brew install python@3.11`
- **Option C:** Use conda (Miniconda/Anaconda)

Verify in Terminal:
```bash
python3 --version
```

### 1.2 Install Git (optional but recommended)

You only need Git if you want to **clone** and later **pull updates**.  
If you prefer, you can download a ZIP instead (see below).

**Windows**
- Download Git for Windows, install with defaults.
- Verify:
  ```powershell
  git --version
  ```

**macOS**
- Option A: Install Xcode Command Line Tools (includes git):
  ```bash
  xcode-select --install
  ```
- Verify:
  ```bash
  git --version
  ```

### 1.3 What is “Terminal” / “PowerShell”?

Some steps below require running commands.

**Windows**
- Use **PowerShell**:
  - Start menu → type **PowerShell** → open
- You can also use “Terminal” (Windows 11), but PowerShell is fine.

**macOS**
- Use **Terminal**:
  - Applications → Utilities → **Terminal**
  - Or Spotlight search: “Terminal”

---

## 2) Get the code

### Option A — Clone with Git (recommended)

**Windows (PowerShell)**
```powershell
cd $HOME\Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git AI4CM
cd AI4CM
```

**macOS (Terminal)**
```bash
cd ~/Documents
git clone https://github.com/WBG-ITS-Innovation/georgia-treasury-prototype.git AI4CM
cd AI4CM
```

### Option B — Download ZIP (no Git required)

1. In GitHub: click **Code → Download ZIP**
2. Unzip to a simple path, for example:
   - **Windows:** `C:\Users\<you>\Documents\AI4CM`
   - **macOS:** `~/Documents/AI4CM`

> Make sure you end up with folders like `backend/`, `frontend/`, `scripts/` directly under `AI4CM/`.

---

## 3) Install & Run (recommended: use scripts)

This is the easiest and most reliable option. It creates both virtual environments and wires the frontend to the backend automatically.

### Windows (PowerShell)

**Step 1 — Go to the repo**
```powershell
cd $HOME\Documents\AI4CM
```

**Step 2 — Setup (one time)**
```powershell
scripts\setup_windows.bat
```

**Step 3 — Run the app**
```powershell
scripts\run_app_windows.bat
```

Open:
- http://localhost:8501

### macOS / Linux (Terminal)

**Step 1 — Go to the repo**
```bash
cd ~/Documents/AI4CM
```

**Step 2 — Setup (one time)**
```bash
chmod +x scripts/setup_unix.sh scripts/run_app_unix.sh
./scripts/setup_unix.sh
```

**Step 3 — Run the app**
```bash
./scripts/run_app_unix.sh
```

Open:
- http://localhost:8501

---

## 4) Optional: verify backend environment

If you want to confirm the backend venv has everything installed:

### Windows (PowerShell)
```powershell
.\backend\.venv\Scripts\python.exe scripts\verify_backend_env.py
```

### macOS / Linux (Terminal)
```bash
./backend/.venv/bin/python scripts/verify_backend_env.py
```

---

## 5) Manual installation (advanced / no scripts)

Use this only if you cannot run the scripts.

### 5.1 Backend venv

**macOS / Linux**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

**Windows (PowerShell)**
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

### 5.2 Frontend venv + run Streamlit

**macOS / Linux**
```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run Overview.py
```

**Windows (PowerShell)**
```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run Overview.py
```

### 5.3 Wire the frontend to the backend

The UI reads `frontend/.tg_paths.json` to locate the backend environment.

Create it manually from the repo root.

**macOS / Linux**
```bash
python3 - <<'PY'
import json, pathlib
repo = pathlib.Path.cwd()
frontend = repo / "frontend" / ".tg_paths.json"
data = {
  "backend_python": str((repo/"backend"/".venv"/"bin"/"python").resolve()),
  "backend_dir": str((repo/"backend").resolve())
}
frontend.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Wrote", frontend)
PY
```

**Windows (PowerShell)**
```powershell
python - <<'PY'
import json, pathlib
repo = pathlib.Path.cwd()
frontend = repo / "frontend" / ".tg_paths.json"
data = {
  "backend_python": str((repo/"backend"/".venv"/"Scripts"/"python.exe").resolve()),
  "backend_dir": str((repo/"backend").resolve())
}
frontend.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Wrote", frontend)
PY
```

---

## 6) Using the Streamlit app (what each tab is for)

### Overview
- confirms backend connection (auto-detected from `.tg_paths.json`)
- shows recent runs
- quick access to run logs and downloads

### Data Pre-processing
Converts **Balance_by_Day_*.xlsx** into a standardized daily dataset.

Typical steps:
1. Upload an Excel file
2. Choose variant: `raw / clean_conservative / clean_treasury`
3. Run preprocessing  
Outputs go to `data_preprocessed/<variant>/` and are available for the Lab.

### Lab
The main “experiment runner”:
- Choose **target**, **cadence**, **horizon**
- Choose **family** (A/B/C/E) + **model**
- Optionally set hyperparameters / overrides
- Run and watch the log live
Outputs go to:
`frontend/runs/run_<family>_<model>_<target>_<cadence>_h<h>_<timestamp>/`

### Dashboard
Visual analysis:
- actual vs predicted overlays
- metrics + leaderboard
- residual diagnostics
- intervals (if produced)

### History
Archive of all runs:
- open a run folder
- view logs
- download CSVs and artifacts

### Models
Reference page for model families + parameter definitions.

---

## 7) Running pipelines directly (Backend CLI)

This is for developers / advanced usage.

### Activate backend environment

**Windows (PowerShell)**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
cd backend
source .venv/bin/activate
```

### Examples
```bash
python run_a_stat.py
python run_b_ml_univariate.py
python run_c_dl_univariate.py
```

Outputs appear under `frontend/runs/`.

---

## 8) Troubleshooting

### “Backend Python missing/invalid”
- Run the setup script again (`scripts/setup_*`)
- Confirm `frontend/.tg_paths.json` exists
- Confirm `backend/.venv` exists

### “File does not exist: frontend/Overview.py”
You are running Streamlit from the wrong folder.  
Run from the repo root:
```bash
cd ~/Documents/AI4CM
python -m streamlit run frontend/Overview.py
```

### Excel pre-processing errors
- confirm the date column is parseable as dates
- confirm the balance column is numeric
- try providing explicit column names in the UI

### PyTorch install issues
The scripts install CPU wheels. If you need GPU (rare on Mac), follow PyTorch install instructions.

---

## Summary

AI4CM provides:
- a local, reproducible forecasting laboratory
- a Streamlit UI for running + comparing model families
- structured outputs (predictions, metrics, plots, logs)
- setup scripts for Windows/macOS so others can run it easily
