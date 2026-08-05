"""Guard against unbound-variable bugs in scripts/run_daily_forecast.sh.

Context: the backtest wiring shipped with `--run-dir "$OUT_DIR"` where the
script's variable is actually named RUN_DIR.  `bash -n` passed, because -n
only checks *syntax* — it never evaluates the script, so an undefined
variable is invisible to it.  The failure only appeared at the very end of a
multi-minute pipeline run, after every model had already been fitted.

This test does the cheap static check `bash -n` cannot: every uppercase
variable the script *reads* must either be assigned inside the script or be
a documented external input (env var / exported pipeline setting).
"""
from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_daily_forecast.sh"

# Variables legitimately supplied from outside the script: documented env
# overrides in its header, plus the TG_* settings it exports to the pipelines.
EXTERNAL = {
    "FAMILIES", "MODE", "RUN_DATE", "STALE_DAYS", "STAT_MODEL",
    "TG_CADENCE", "TG_DATA_PATH", "TG_DATE_COL", "TG_HORIZON", "TG_TARGET",
    "TG_FAMILY", "TG_MODEL_FILTER", "TG_OUT_ROOT", "TG_PARAM_OVERRIDES",
    # Item 1e: optional pin. When set, the script and every runner refuse to
    # proceed unless the input file's SHA-256 matches, so "the same run" means the
    # same bytes rather than the same filename. Documented in the script header.
    "AI4CM_EXPECTED_DATA_SHA256",
    "HOME", "PATH", "PWD", "IFS", "OSTYPE", "BASH_SOURCE",
}

READ_RE = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")
ASSIGN_RE = re.compile(r"^\s*(?:export\s+|local\s+)?([A-Z_][A-Z0-9_]*)=", re.MULTILINE)


def test_every_referenced_variable_is_assigned_or_external():
    text = SCRIPT.read_text()
    assigned = set(ASSIGN_RE.findall(text))
    referenced = set(READ_RE.findall(text))
    unbound = referenced - assigned - EXTERNAL
    assert not unbound, (
        f"scripts/run_daily_forecast.sh reads variable(s) that are never "
        f"assigned: {sorted(unbound)}. Either assign them or add them to "
        f"EXTERNAL in this test if they are documented env inputs."
    )


def test_backtest_block_uses_the_run_directory():
    """The backtest report must be generated into the run folder."""
    text = SCRIPT.read_text()
    assert 'backtest_report.py" --run-dir "$RUN_DIR"' in text
