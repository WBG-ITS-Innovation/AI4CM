param(
  [switch]$InstallBoosters = $true
)

Write-Host "==> Creating venv with Python 3.11 if available..."
$py311 = (& py -0p | Select-String "3.11").ToString()
if ($py311) { & py -3.11 -m venv .venv } else { & py -m venv .venv }

Write-Host "==> Activating venv ..."
& .\.venv\Scripts\Activate.ps1

# Clear common corporate TLS overrides for this session
if (Test-Path Env:PIP_CERT) { Remove-Item Env:PIP_CERT }
if (Test-Path Env:REQUESTS_CA_BUNDLE) { Remove-Item Env:REQUESTS_CA_BUNDLE }

Write-Host "==> Upgrading pip ..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "==> Installing base requirements ..."
$base = @(
  "numpy==2.3.3",
  "pandas==2.3.3",
  "matplotlib==3.10.7",
  "scipy==1.16.2",
  "scikit-learn==1.5.2"
)
foreach ($pkg in $base) {
  .\.venv\Scripts\python.exe -m pip install $pkg
}

if ($InstallBoosters) {
  Write-Host "==> Installing optional boosters (best-effort) ..."
  $boosters = @("xgboost==2.1.1","lightgbm==4.5.0","catboost==1.2.7")
  foreach ($pkg in $boosters) {
    try { .\.venv\Scripts\python.exe -m pip install $pkg }
    catch { Write-Warning "Skipped $pkg ($($_.Exception.Message))" }
  }
}

Write-Host "==> Validating imports ..."
.\.venv\Scripts\python.exe -c "import numpy,pandas,sklearn; print('OK ->', numpy.__version__, pandas.__version__, sklearn.__version__)"

Write-Host "==> Done. Use:"
Write-Host "   .\.venv\Scripts\python.exe run_b_ml_univariate.py"
Write-Host "   .\.venv\Scripts\python.exe run_b_ml_multivariate.py"
