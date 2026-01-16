#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"
FRONTEND_DIR="$REPO_DIR/frontend"

echo "=== Georgia Treasury Lab — Unix setup ==="
echo "Repo:     $REPO_DIR"
echo "Backend:  $BACKEND_DIR"
echo "Frontend: $FRONTEND_DIR"
echo

# ---- Backend venv ----
if [ ! -x "$BACKEND_DIR/.venv/bin/python" ]; then
  echo "[Backend] Creating venv..."
  python3 -m venv "$BACKEND_DIR/.venv"
else
  echo "[Backend] venv already exists."
fi
"$BACKEND_DIR/.venv/bin/python" -m pip install --upgrade pip
"$BACKEND_DIR/.venv/bin/python" -m pip install -r "$BACKEND_DIR/requirements.txt" || true
# Torch CPU wheels
"$BACKEND_DIR/.venv/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || true

# ---- Frontend venv ----
if [ ! -x "$FRONTEND_DIR/.venv/bin/python" ]; then
  echo "[Frontend] Creating venv..."
  python3 -m venv "$FRONTEND_DIR/.venv"
else
  echo "[Frontend] venv already exists."
fi
"$FRONTEND_DIR/.venv/bin/python" -m pip install --upgrade pip
"$FRONTEND_DIR/.venv/bin/python" -m pip install -r "$FRONTEND_DIR/requirements.txt"

# ---- Wire frontend -> backend
python3 - <<PY
import json, pathlib
frontend = pathlib.Path(r"$FRONTEND_DIR") / ".tg_paths.json"
data = {
  "backend_python": r"$BACKEND_DIR/.venv/bin/python",
  "backend_dir": r"$BACKEND_DIR"
}
frontend.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Wrote", frontend)
PY

echo
echo "✅ Setup complete."
echo "Next: scripts/run_app_unix.sh"
