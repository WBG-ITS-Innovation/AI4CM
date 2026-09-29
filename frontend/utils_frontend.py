# frontend/utils_frontend.py
from __future__ import annotations
import io, json, os, re
from pathlib import Path
import zipfile

APPROOT      = Path(__file__).resolve().parent      # frontend/
from paths import runs_dir
# Resolved LAZILY. A module-level constant is imported once and cached, so the first caller's
# AI4CM_RUNS_DIR value would leak into every later one -- which made the page tests
# order-dependent -- and in a deployment would pin the lab to whatever path was set when this
# module first loaded. Default resolves to APPROOT/"runs" exactly as before.
class _LazyRunsRoot:
    def _p(self):
        return runs_dir()

    def __truediv__(self, other):
        return self._p() / other

    def __getattr__(self, name):
        return getattr(self._p(), name)

    def __fspath__(self):
        return str(self._p())

    def __str__(self):
        return str(self._p())


RUNS_ROOT    = _LazyRunsRoot()
UPLOADS_ROOT = APPROOT / "runs_uploads"
runs_dir().mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Cross-run comparison helpers
# ---------------------------------------------------------------------------

def best_metric_row(metr, target=None, horizon=None):
    """The row for a run's best model, or ``(None, reason)`` saying why there is not one.

    Returns ``(row, None)`` on success and ``(None, reason)`` otherwise, where ``reason`` is
    a sentence fit to show a reader. Nothing here raises: a run that cannot be compared is a
    normal state of the world, and the caller needs to say so rather than fall over.

    **Why a long-format run is skipped rather than pivoted.** Every E_QUANTILE run writes
    ``metrics_long.csv`` with ``model,fold,metric,quantile,value``, which has a ``model``
    column and no ``MAE``. Indexing ``MAE`` behind a guard that only checked for ``model``
    is what raised ``KeyError: 'MAE'``. Pivoting it would not help: that file carries
    ``pinball`` and ``coverage_p10_p90`` only, so none of the columns this table ranks by
    (MAE, RMSE, sMAPE, R2) exist in it at all, and a pivot would contribute a row of blanks
    implying a comparison that never happened.
    """
    if metr is None or getattr(metr, "empty", True):
        return None, "No metrics file was written for this run."

    m = metr.copy()
    if target is not None and "target" in m.columns:
        m = m[m["target"] == target]
    if horizon is not None and "horizon" in m.columns:
        m = m[m["horizon"] == horizon]
    if m.empty:
        return None, "This run has no metrics for the selected target and horizon."

    # Without a model column there is nothing to rank, and the first row is the run's only
    # row -- the behaviour this helper was extracted from, kept deliberately.
    if "model" not in m.columns:
        return m.iloc[0], None

    if "MAE" not in m.columns:
        return None, ("This run reports metrics in long format, with no MAE column to rank "
                      "models by, so it is not in the table.")
    if not m["MAE"].notna().any():
        return None, "Every MAE in this run is blank, so its models cannot be ranked."

    return m.loc[m["MAE"].idxmin()], None


def load_run_outputs(run_dir: Path) -> dict:
    """Load standard outputs (predictions, metrics, leaderboard, config) from a run.

    Returns a dict with keys:
        pred   : pd.DataFrame | None
        metr   : pd.DataFrame | None
        lb     : pd.DataFrame | None
        config : dict
        run_id : str
    """
    import pandas as _pd

    out = run_dir / "outputs"
    if not out.exists():
        return {"pred": None, "metr": None, "lb": None, "config": {}, "run_id": run_dir.name}

    def _find(name: str):
        for base in [out, out / "daily", out / "weekly", out / "monthly"]:
            p = base / name
            if p.exists():
                return p
        return None

    pred = metr = lb = None
    pp = _find("predictions_long.csv")
    mp = _find("metrics_long.csv")
    lp = _find("leaderboard.csv")
    if pp:
        pred = _pd.read_csv(pp)
        if "date" in pred.columns:
            pred["date"] = _pd.to_datetime(pred["date"], errors="coerce")
            pred = pred.dropna(subset=["date"]).sort_values("date")
    if mp:
        metr = _pd.read_csv(mp)
    if lp:
        lb = _pd.read_csv(lp)

    cfg = {}
    cfg_p = _find("artifacts/config.json")
    if cfg_p:
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {"pred": pred, "metr": metr, "lb": lb, "config": cfg, "run_id": run_dir.name}
