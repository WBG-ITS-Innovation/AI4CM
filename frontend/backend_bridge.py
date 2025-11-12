# backend_bridge.py — robust runner launcher with progress callback
from __future__ import annotations
import os, subprocess, time
from pathlib import Path
from typing import Callable, Dict, Tuple

def launch_backend(
    backend_py: str,
    runner_script: str,
    backend_dir: str,
    env_vars: Dict[str, str],
    run_dir: Path,
    on_progress: Callable[[str, float], None] | None = None,
) -> Tuple[int, float, Path, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(env_vars.get("TG_OUT_ROOT", str(run_dir / "outputs"))).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "backend_run.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("="*80 + "\n")
        lf.write(f"[bridge] python = {backend_py or '(empty)'}\n")
        lf.write(f"[bridge] runner = {runner_script or '(empty)'}\n")
        lf.write(f"[bridge] cwd    = {backend_dir or '(empty)'}\n")
        lf.write(f"[bridge] out    = {out_root}\n")
        lf.write("="*80 + "\n")

    bp = Path(backend_py) if backend_py else None
    if not bp or not bp.exists():
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("[bridge][ERROR] backend_py is missing or invalid\n")
        return (1, 0.0, log_path, str(out_root))

    rs = Path(runner_script)
    if not rs.exists():
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[bridge][ERROR] runner not found: {rs}\n")
        return (1, 0.0, log_path, str(out_root))

    cwd = Path(backend_dir) if backend_dir else run_dir
    if not cwd.exists():
        cwd = run_dir  # safe fallback for Windows

    env = os.environ.copy()
    for k, v in env_vars.items():
        env[k] = str(v)

    t0 = time.time()
    proc = subprocess.Popen(
        [str(bp), str(rs)],
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    tail = []
    with open(log_path, "a", encoding="utf-8") as lf:
        for line in proc.stdout:
            lf.write(line)
            tail.append(line)
            if on_progress:
                on_progress("".join(tail[-400:]), time.time() - t0)

    rc = proc.wait()
    elapsed = time.time() - t0
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[bridge] rc={rc}, elapsed={elapsed:.1f}s\n")
    return (rc, elapsed, log_path, str(out_root))
