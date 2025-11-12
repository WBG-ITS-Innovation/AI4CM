# Georgia Treasury Forecast Lab — Clean Integration

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If behind corporate TLS, set PIP_CERT/REQUESTS_CA_BUNDLE to certs\corp_bundle.pem first.

pip install -U pip wheel
pip install -r requirements.txt
