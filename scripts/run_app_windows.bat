@echo off
setlocal
set "SCRIPTS_DIR=%~dp0"
for %%I in ("%SCRIPTS_DIR%\..") do set "REPO_DIR=%%~fI"
set "FRONTEND_DIR=%REPO_DIR%\frontend"

if not exist "%FRONTEND_DIR%\.venv\Scripts\python.exe" (
  echo [ERROR] Frontend venv not found. Run scripts\setup_windows.bat first.
  exit /b 1
)

echo Starting Streamlit...
"%FRONTEND_DIR%\.venv\Scripts\python.exe" -m streamlit run "%FRONTEND_DIR%\Overview.py"
