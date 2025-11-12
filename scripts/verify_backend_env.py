# scripts/verify_backend_env.py
import importlib, sys, json
mods = [
  "numpy","pandas","scipy","statsmodels",
  "sklearn","matplotlib",
  "xgboost","lightgbm",
  "torch","torchvision","torchaudio",
  "openpyxl"
]
results = {}
for m in mods:
  try:
    importlib.import_module(m)
    results[m] = "OK"
  except Exception as e:
    results[m] = f"ERROR: {e.__class__.__name__}: {e}"
print(json.dumps(results, indent=2))
if any(v.startswith("ERROR") for v in results.values()):
  sys.exit(1)
