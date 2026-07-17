#!/usr/bin/env bash
#
# run_daily_forecast.sh — run the full AI4CM forecast pipeline end to end and
# write outputs plus a plain-text summary into a dated folder.
#
# - Activates backend/.venv.
# - Runs each pipeline family (A stat, B ML, E quantile, C DL) on the latest
#   data, each into its own subfolder under backend/forecast_runs/<date>/.
# - Writes SUMMARY.txt (date, models run, key metrics, leakage/shift flags,
#   data-freshness warning).
# - Exits non-zero on any failure.
# - Runnable from the repo root (or anywhere) and idempotent: re-running on the
#   same day recreates the same dated folder from scratch.
#
# Overridable via environment variables (defaults in parentheses):
#   FAMILIES        which families to run   (A_STAT B_ML E_QUANTILE C_DL)
#   TG_TARGET       target series           (Revenues)
#   TG_CADENCE      Daily/Weekly/Monthly    (Daily)
#   TG_HORIZON      forecast horizon        (5)
#   TG_DATE_COL     date column name        (date)
#   TG_DATA_PATH    input CSV               (newest master_daily_clean_*.csv)
#   STAT_MODEL      A_STAT model to run     (ETS)
#   STALE_DAYS      freshness threshold     (3)

set -euo pipefail

# ── Resolve paths so the script works from any working directory ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"

# ── Activate the backend virtual environment ──
VENV_ACTIVATE="$BACKEND_DIR/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "[ERROR] Backend venv not found at $VENV_ACTIVATE. Run scripts/setup_unix.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# ── Configuration (all overridable via the environment) ──
FAMILIES="${FAMILIES:-A_STAT B_ML E_QUANTILE C_DL}"
TG_TARGET="${TG_TARGET:-Revenues}"
TG_CADENCE="${TG_CADENCE:-Daily}"
TG_HORIZON="${TG_HORIZON:-5}"
TG_DATE_COL="${TG_DATE_COL:-date}"
STAT_MODEL="${STAT_MODEL:-ETS}"
STALE_DAYS="${STALE_DAYS:-3}"
RUN_DATE="$(date +%F)"

# ── Choose the latest data file (newest by modification time) unless given ──
if [ -z "${TG_DATA_PATH:-}" ]; then
  PROCESSED_DIR="$BACKEND_DIR/data/processed"
  # -t sorts by mtime (newest first); head -1 takes the newest.
  TG_DATA_PATH="$(ls -t "$PROCESSED_DIR"/master_daily_clean_*.csv 2>/dev/null | head -1 || true)"
  if [ -z "$TG_DATA_PATH" ]; then
    echo "[ERROR] No data file found in $PROCESSED_DIR (master_daily_clean_*.csv)." >&2
    exit 1
  fi
fi
echo "[daily] Using data file: $TG_DATA_PATH"

# ── Prepare the dated output folder (idempotent: wipe and recreate) ──
RUNS_ROOT="$BACKEND_DIR/forecast_runs"
RUN_DIR="$RUNS_ROOT/$RUN_DATE"
# Safety guard: only ever remove a path that lives under forecast_runs/.
case "$RUN_DIR" in
  "$RUNS_ROOT"/*) rm -rf "$RUN_DIR" ;;
  *) echo "[ERROR] Refusing to remove unexpected path: $RUN_DIR" >&2; exit 1 ;;
esac
mkdir -p "$RUN_DIR"
echo "[daily] Output folder: $RUN_DIR"

# ── Shared pipeline settings ──
export TG_TARGET TG_CADENCE TG_HORIZON TG_DATE_COL TG_DATA_PATH
export TG_MODEL_FILTER=""   # empty = run all models the family offers

# Map a family name to (runner script, param overrides). The quantile pipeline
# needs folds>=2 with a 1-year minimum train window; stat/ML are happy with a
# single fold and a 4-year window. These were verified against the real data.
run_family() {
  local family="$1"
  local out_dir="$RUN_DIR/$(echo "$family" | tr '[:upper:]' '[:lower:]')"
  mkdir -p "$out_dir"
  echo "[daily] === Running $family -> $out_dir ==="

  local runner overrides model_filter=""
  case "$family" in
    A_STAT)
      runner="run_a_stat.py"
      overrides='{"folds":1,"min_train_years":4}'
      model_filter="$STAT_MODEL"   # stat runs one model per invocation
      ;;
    B_ML)
      runner="run_b_ml_univariate.py"
      overrides='{"folds":1,"min_train_years":4}'
      ;;
    E_QUANTILE)
      runner="run_e_quantile_daily_univariate.py"
      overrides='{"folds":2,"min_train_years":1}'
      ;;
    C_DL)
      runner="run_c_dl_quick_univariate.py"
      overrides='{}'
      ;;
    *)
      echo "[ERROR] Unknown family: $family" >&2
      return 1
      ;;
  esac

  # Run the family. `set -e` plus this explicit check means any pipeline
  # failure aborts the whole script with a non-zero exit code.
  TG_FAMILY="$family" \
  TG_MODEL_FILTER="$model_filter" \
  TG_OUT_ROOT="$out_dir" \
  TG_PARAM_OVERRIDES="$overrides" \
    python "$BACKEND_DIR/$runner"
}

# ── Run every requested family ──
for family in $FAMILIES; do
  run_family "$family"
done

# ── Build the plain-text summary ──
echo "[daily] === Writing summary ==="
python "$SCRIPT_DIR/daily_summary.py" \
  --run-dir "$RUN_DIR" \
  --data-file "$TG_DATA_PATH" \
  --date-col "$TG_DATE_COL" \
  --target "$TG_TARGET" \
  --cadence "$TG_CADENCE" \
  --horizon "$TG_HORIZON" \
  --run-date "$RUN_DATE" \
  --families "$FAMILIES" \
  --stale-days "$STALE_DAYS"

echo "[daily] DONE. Summary: $RUN_DIR/SUMMARY.txt"
