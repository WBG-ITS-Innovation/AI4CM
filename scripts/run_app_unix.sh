#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$REPO_DIR/frontend"

if [ ! -x "$FRONTEND_DIR/.venv/bin/python" ]; then
  echo "[ERROR] Frontend venv not found. Run scripts/setup_unix.sh first."
  exit 1
fi

exec "$FRONTEND_DIR/.venv/bin/python" -m streamlit run "$FRONTEND_DIR/Overview.py"
