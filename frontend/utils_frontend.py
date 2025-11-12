# frontend/utils_frontend.py
from __future__ import annotations
import io, json, os, re
from pathlib import Path
import zipfile

APPROOT      = Path(__file__).resolve().parent      # frontend/
RUNS_ROOT    = APPROOT / "runs"
UPLOADS_ROOT = APPROOT / "runs_uploads"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)

PATHS_FILE = APPROOT / ".tg_paths.json"

RUNNER_SENTINELS = {
    "run_a_stat.py",
    "run_b_ml_univariate.py", "run_b_ml_multivariate.py",
    "run_c_dl_univariate.py", "run_c_dl_multivariate.py",
    "run_e_quantile_daily_univariate.py", "run_e_quantile_daily_multivariate.py",
    "run_preprocess.py",
}

def _is_backend_dir(d: Path) -> bool:
    try:
        if not d.exists() or not d.is_dir():
            return False
        names = {p.name for p in d.iterdir() if p.is_file()}
        return any(x in names for x in RUNNER_SENTINELS)
    except Exception:
        return False

def _auto_guess_backend_dir() -> str:
    # Prefer monorepo sibling "backend"
    sib = APPROOT.parent / "backend"
    if _is_backend_dir(sib):
        return str(sib.resolve())
    # Back-compat: sibling "TreasuryGeorgiaBackEnd"
    legacy = APPROOT.parent / "TreasuryGeorgiaBackEnd"
    if _is_backend_dir(legacy):
        return str(legacy.resolve())
    return ""

def _auto_guess_python(backend_dir: str) -> str:
    if backend_dir:
        b = Path(backend_dir)
        for c in (b/".venv"/"Scripts"/"python.exe", b/".venv"/"bin"/"python"):
            if c.exists():
                return str(c.resolve())
    return ""  # keep empty so UI can show a clear error

def load_paths() -> dict:
    # 1) Prefer saved JSON
    bp = ""; bd = ""
    if PATHS_FILE.exists():
        try:
            data = json.loads(PATHS_FILE.read_text(encoding="utf-8"))
            bp = data.get("backend_python","")
            bd = data.get("backend_dir","")
        except Exception:
            pass
    # 2) Autodetect if missing/invalid
    if not bd or not _is_backend_dir(Path(bd)):
        bd = _auto_guess_backend_dir()
    if not bp or not Path(bp).exists():
        bp = _auto_guess_python(bd)
    return {"backend_python": bp, "backend_dir": bd}

def save_paths(backend_python: str, backend_dir: str) -> None:
    PATHS_FILE.write_text(
        json.dumps({"backend_python": backend_python, "backend_dir": backend_dir}, indent=2),
        encoding="utf-8"
    )

def new_run_folders(run_name: str | None = None):
    from uuid import uuid4
    if not run_name:
        run_id = f"run_{uuid4().hex[:8]}"
    else:
        run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)[:160] or f"run_{uuid4().hex[:8]}"
    run_dir = RUNS_ROOT / run_id
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, out_dir

def list_runs():
    return sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)

def collect_output_files(out_dir: Path):
    return sorted([p for p in Path(out_dir).rglob("*") if p.is_file()])

def zip_outputs(out_dir: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in collect_output_files(out_dir):
            zf.write(p, arcname=str(p.relative_to(out_dir)))
    bio.seek(0)
    return bio.read()
