# utils_frontend.py — shared helpers (paths, runs, uploads, zipping) • WINDOWS-SAFE
from __future__ import annotations
import io, json, os, re
from pathlib import Path
import zipfile

# IMPORTANT: utils_frontend.py sits in the frontend root.
# Use .parent (NOT parents[1]) so RUNS/UPLOADS live under the frontend folder.
APPROOT   = Path(__file__).resolve().parent         # …/TreasuryGeorgiaFrontEnd
RUNS_ROOT = APPROOT / "runs"
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

def _fix_venv_backslash(p: str) -> str:
    """Windows convenience: turn ...BackEnd.venv... into ...BackEnd\.venv..."""
    if not p:
        return p
    fixed = re.sub(r"(BackEnd)(\.venv)", r"\1\\\.venv", p)
    return fixed.replace("\\\\", "\\")

def _is_backend_dir(d: Path) -> bool:
    try:
        if not d.exists() or not d.is_dir():
            return False
        names = {p.name for p in d.iterdir() if p.is_file()}
        return any(x in names for x in RUNNER_SENTINELS)
    except Exception:
        return False

def _auto_guess_backend_dir() -> str:
    """Prefer …/TreasuryGeorgiaBackEnd alongside this frontend; else shallow scan."""
    sib = APPROOT.parent / "TreasuryGeorgiaBackEnd"
    if _is_backend_dir(sib):
        return str(sib.resolve())
    # shallow scan 2 levels up for any folder that looks like the backend
    for root in [APPROOT.parent, APPROOT.parent.parent]:
        for cand in root.glob("**/TreasuryGeorgiaBackEnd"):
            if _is_backend_dir(cand):
                return str(cand.resolve())
    return ""

def _auto_guess_python(backend_dir: str) -> str:
    """Prefer the backend .venv python. Never silently fall back to the frontend venv."""
    if backend_dir:
        b = Path(backend_dir)
        for c in (b/".venv"/"Scripts"/"python.exe", b/".venv"/"bin"/"python"):
            try:
                if c.exists():
                    return str(c.resolve())
            except Exception:
                pass
    # If no backend venv, return empty and let the UI show a clear error
    return ""

def load_paths() -> dict:
    """Load saved paths (if any). If invalid or missing, AUTODETECT the backend dir + its .venv python."""
    data = {"backend_python": "", "backend_dir": ""}
    if PATHS_FILE.exists():
        try:
            data.update(json.loads(PATHS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass

    bd = data.get("backend_dir", "")
    if not bd or not _is_backend_dir(Path(bd)):
        bd = _auto_guess_backend_dir()

    bp = _fix_venv_backslash(data.get("backend_python", ""))
    if not bp or not Path(bp).exists():
        bp = _auto_guess_python(bd)

    return {"backend_python": bp, "backend_dir": bd}

def save_paths(backend_python: str, backend_dir: str) -> None:
    PATHS_FILE.write_text(
        json.dumps({"backend_python": backend_python, "backend_dir": backend_dir}, indent=2),
        encoding="utf-8"
    )

def new_run_folders(run_name: str | None = None):
    """Create runs under the FRONTEND repo. If a name is given, use it; else generate a slug."""
    from uuid import uuid4
    if not run_name:
        run_id = f"run_{uuid4().hex[:8]}"
    else:
        # sanitize for Windows
        run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)[:160]
        if not run_id:
            run_id = f"run_{uuid4().hex[:8]}"
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
