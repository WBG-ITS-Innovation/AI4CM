#!/usr/bin/env python
# Georgia — C·DL quick (smoke) multivariate
from __future__ import annotations
import os, sys, json, subprocess

def _ensure_torch():
    try:
        import torch  # noqa
    except Exception:
        print("[setup] installing torch ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.1,<3"])

def main():
    _ensure_torch()
    ovr = {"quick_mode": True, "max_epochs": 3}
    try:
        ui = json.loads(os.environ.get("TG_PARAM_OVERRIDES", "{}") or "{}")
        ovr.update(ui)
    except Exception:
        pass
    os.environ["TG_PARAM_OVERRIDES"] = json.dumps(ovr)
    from run_c_dl_multivariate import main as _run
    _run()

if __name__ == "__main__":
    main()
