# AI4CM Forecast Lab

A local sandbox to preprocess Treasury cash/balance time-series, run forecasting experiments (Stat/ML/DL/Quantile), and review results in a Streamlit UI.

---

## What this repo does (in plain language)

You can use this repo to:

- turn a raw balance Excel/CSV into a clean daily dataset
- run forecasting experiments (different model “families”)
- compare models visually (charts + metrics + leaderboards)
- keep a full record of each run (log + config + outputs)

The project has two parts:

- **backend/** → preprocessing + forecasting code
- **frontend/** → Streamlit web app (the UI)

They run in separate Python environments so installs are cleaner and more stable.

---


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
- `predictions_long.csv`
- `metrics_long.csv`
- `leaderboard.csv`
- Plots in `plots/`
- Config in `artifacts/`

---

# Setup overview (3 steps)

1) Install Python (+ Git if you want the easy “clone” option)  
2) Get the code (clone or download zip)  
3) Create 2 environments (backend + frontend) and run Streamlit

---

# Step 0 — Install the basics

## 0.1 Install Python

### Windows
1. Go to: https://www.python.org/downloads/windows/  
2. Download **Python 3.11.x**  
3. Run the installer and **check**:
   - ✅ **Add python.exe to PATH**
4. Finish install

Quick check:
- Open **Start Menu** → type **PowerShell** → open it
- Paste:
```powershell
python --version
pip --version
```

If `python` is “not recognized”, you likely didn’t check “Add to PATH”. Re-run the installer.

### macOS
1. Go to: https://www.python.org/downloads/macos/  
2. Download **Python 3.11.x** for macOS  
3. Install it
4. Quick check:
   - Open **Applications → Utilities → Terminal**
   - Paste:
```bash
python3 --version
pip3 --version
```

If your Mac asks to install developer tools, accept. (That installs command-line helpers.)

---

## 0.2 Install Git (recommended, but optional)

Git makes it easy to download updates, but you can also use ZIP.

### Windows
1. Install Git: https://git-scm.com/download/win
2. Check in PowerShell:
```powershell
git --version
```

### macOS
Git is usually installed. If not:
```bash
xcode-select --install
git --version
```

---

## 0.3 Optional: Install VS Code (recommended)
VS Code is handy for editing files and reading logs.  
Download: https://code.visualstudio.com/

---

# Step 1 — Get the code

Choose ONE option.

## Option A (recommended): Clone with Git

### Windows
1. Open **PowerShell**:
   - Start Menu → type **PowerShell** → open
2. Paste:
```powershell
cd $HOME\Documents
git clone https://github.com/WBG-ITS-Innovation/AI4CM.git
cd AI4CM
```

### macOS
1. Open **Terminal**:
   - Applications → Utilities → Terminal
2. Paste:
```bash
cd ~/Documents
git clone https://github.com/WBG-ITS-Innovation/AI4CM.git
cd AI4CM
```

## Option B: Download ZIP (no Git)

1. On GitHub: **Code → Download ZIP**
2. Unzip it into:
   - Windows: `C:\Users\<you>\Documents\AI4CM`
   - macOS: `~/Documents/AI4CM`
3. Make sure the folder contains `backend/` and `frontend/`.

Tip: if possible, avoid working from a OneDrive-synced folder (it can sometimes lock files).

---

# Step 2 — Open a terminal in the project folder

Here's how to do it:

## Windows
- Option 1: **PowerShell**
  1. Start Menu → PowerShell
  2. Then paste:
     ```powershell
     cd $HOME\Documents\AI4CM
     ```
- Option 2: **File Explorer**
  1. Open the AI4CM folder in File Explorer
  2. Click the address bar, type `powershell`, press Enter

## macOS
- Option 1: **Terminal**
  1. Applications → Utilities → Terminal
  2. Then paste:
     ```bash
     cd ~/Documents/AI4CM
     ```
- Option 2: Finder shortcut
  1. Finder → open the AI4CM folder
  2. Right-click inside folder → **New Terminal at Folder** (if enabled)

From here onward, all commands assume your terminal is in the AI4CM folder.

---

# Step 3 — Create the backend environment (required)

The backend environment contains the forecasting + preprocessing dependencies.

## Windows (PowerShell)
If you ever see a script execution warning when activating, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then run:
```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

## macOS (Terminal)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

Quick sanity check:

- Windows:
```powershell
.\backend\.venv\Scripts\python.exe --version
```

- macOS:
```bash
./backend/.venv/bin/python --version
```

---

# Step 4 — Create the frontend environment + run the app

The frontend environment contains Streamlit (the UI).

## Windows (PowerShell)
```powershell
cd .\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run Overview.py
```

## macOS (Terminal)
```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run Overview.py
```

Your browser should open automatically. If not, go to:
- http://localhost:8501

Leave this terminal running while you use the app.

---

# One-time setup inside the app (backend path)

The UI needs to know where the backend Python is so it can launch jobs.

In the Streamlit **Overview** page:
1. Set **Backend directory** to:
   - `<repo_root>/backend`
2. Set **Backend python** to:
   - macOS: `<repo_root>/backend/.venv/bin/python`
   - Windows: `<repo_root>\backend\.venv\Scripts\python.exe`
3. Click **Save**

This writes a local file: `frontend/.tg_paths.json` (not committed).

---

# Using the app

## Data Pre-processing
Use this page to convert raw balance files into standardized daily data.

Typical flow:
1. Upload your Excel/CSV (e.g., `Balance_by_Day_2015-2025.xlsx`)
2. If needed, specify date and balance columns
3. Choose a variant:
   - `raw`
   - `clean_conservative`
   - `clean_treasury`
4. Run preprocessing

Outputs are written under:
- `data_preprocessed/<variant>/...`

Uploads are stored under:
- `frontend/runs_uploads/`

## Lab
The Lab page runs forecasting experiments.

You choose:
- dataset (CSV)
- target column
- cadence (Daily / Weekly / Monthly)
- horizon
- family + model
- optional advanced overrides

Each run is saved under:
- `frontend/runs/run_<...timestamp...>/`

## Dashboard + History
- Dashboard: plots, metrics, leaderboards
- History: browse runs, inspect logs, download output bundles

---

# Outputs (what to expect)

Each run folder typically contains:
```
backend_run.log
outputs/
  predictions_long.csv
  metrics_long.csv
  leaderboard.csv
  plots/
  artifacts/
```

Start with `backend_run.log` if anything fails.

---

# Common issues (and fast fixes)

## “File does not exist: frontend/Overview.py”
You ran Streamlit from the wrong folder.
From the repo root, you can always run:
```bash
python -m streamlit run frontend/Overview.py
```

## “Backend Python missing/invalid” (in Lab or Pre-processing)
- Confirm `backend/.venv` exists
- In Overview, re-save backend dir + backend python
- Restart Streamlit

## macOS Streamlit feels slow
Install watchdog in the frontend venv:
```bash
pip install watchdog
```

## Torch install issues
Use PyTorch’s official selector (CPU is fine for most runs):
https://pytorch.org/get-started/locally/

---

## Notes for contributors

- Prefer Python 3.11 for compatibility
- Keep generated folders out of git (see `.gitignore`)
- If you rename backend runner scripts, update the Streamlit pages that call them
