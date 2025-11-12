@echo off
setlocal enabledelayedexpansion

REM This script assumes the repo contains:
REM   backend\   (from TreasuryGeorgiaBackEnd, without .venv/outputs)
REM   frontend\  (from TreasuryGeorgiaFrontEnd, without .venv/runs)

set "SCRIPTS_DIR=%~dp0"
for %%I in ("%SCRIPTS_DIR%\..") do set "REPO_DIR=%%~fI"
set "BACKEND_DIR=%REPO_DIR%\backend"
set "FRONTEND_DIR=%REPO_DIR%\frontend"

echo === Georgia Treasury Lab — Windows setup ===
echo Repo:      %REPO_DIR%
echo Backend:   %BACKEND_DIR%
echo Frontend:  %FRONTEND_DIR%
echo.

REM Check Python launcher
where py >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] Python launcher 'py' not found. Install Python 3.x and retry.
  exit /b 1
)

REM ---------- Backend venv ----------
if not exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  echo [Backend] Creating venv...
  py -3 -m venv "%BACKEND_DIR%\.venv" || goto :venv_fail
) else (
  echo [Backend] venv already exists.
)
echo [Backend] Upgrading pip...
"%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade pip

echo [Backend] Installing requirements...
"%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install -r "%BACKEND_DIR%\requirements.txt"

echo [Backend] Ensuring PyTorch CPU wheels (optional)...
"%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM ---------- Frontend venv ----------
if not exist "%FRONTEND_DIR%\.venv\Scripts\python.exe" (
  echo [Frontend] Creating venv...
  py -3 -m venv "%FRONTEND_DIR%\.venv" || goto :venv_fail
) else (
  echo [Frontend] venv already exists.
)
echo [Frontend] Upgrading pip...
"%FRONTEND_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade pip

echo [Frontend] Installing Streamlit & plotting deps...
"%FRONTEND_DIR%\.venv\Scripts\python.exe" -m pip install streamlit pandas plotly

REM ---------- Wire frontend -> backend via .tg_paths.json ----------
echo [Link] Writing frontend\.tg_paths.json...
"%BACKEND_DIR%\.venv\Scripts\python.exe" -c "import json; f=r'%FRONTEND_DIR%\\.tg_paths.json'; bp=r'%BACKEND_DIR%\\.venv\\Scripts\\python.exe'; bd=r'%BACKEND_DIR%'; open(f,'w',encoding='utf-8').write(json.dumps({'backend_python':bp,'backend_dir':bd},indent=2))"

echo.
echo ✅ Setup complete.
echo Next step: scripts\run_app_windows.bat
exit /b 0

:venv_fail
echo [ERROR] Could not create virtual environment. Ensure Python 3.x is installed.
exit /b 1
