"""Every runner's TG_HORIZON default is the validated horizon.

Why this file exists
--------------------
Each family runner reads ``TG_HORIZON`` from the environment with a literal fallback. Seven of
them fell back to 5. ``run_a_stat.py`` fell back to 6 (inference-horizon map, §2.5, "the odd
one out"). It was harmless only because every caller sets ``TG_HORIZON``; a runner invoked by
hand without it would have evaluated at a horizon nothing in the project is measured at, under
a leaderboard that looks like every other.

The defaults are literals in eight files and one shell script, so this reads them as text.
That is deliberate: importing a runner executes nothing useful (the default lives inside
``main()``), and a regex that finds nothing must fail rather than pass, which the coverage
assertion below enforces.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from forecast_modes import VALIDATED_HORIZON  # noqa: E402

#: Every file that reads TG_HORIZON with a fallback. A runner added later must appear here or
#: the coverage test fails, so a new runner cannot ship an unpinned default by omission.
RUNNERS = sorted(p for p in BACKEND.glob("run_*.py") if "TG_HORIZON" in p.read_text())
SCRIPT = REPO / "scripts" / "run_daily_forecast.sh"

#: A line that reads the variable AND supplies a fallback. Lines that merely list the name
#: (for printing, or as a key list) carry no default and are not matched.
PY_DEFAULT = re.compile(r'TG_HORIZON"\]?\s*(?:,\s*"?(\d+)"?\s*\)|\s*\)?\s*or\s*(\d+))')
SH_DEFAULT = re.compile(r'\$\{TG_HORIZON:-(\d+)\}')


def _defaults_in(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "TG_HORIZON" not in line:
            continue
        for m in PY_DEFAULT.finditer(line):
            out.extend(int(g) for g in m.groups() if g)
        for m in SH_DEFAULT.finditer(line):
            out.append(int(m.group(1)))
    return out


def test_the_runners_with_a_default_are_the_ones_expected():
    """The regex must be finding the real fallbacks, not matching nothing and passing."""
    with_default = {p.name for p in RUNNERS if _defaults_in(p)}
    assert with_default >= {
        "run_a_stat.py", "run_b_ml_univariate.py", "run_b_ml_multivariate.py",
        "run_c_dl_univariate.py", "run_c_dl_multivariate.py",
        "run_e_quantile_daily_univariate.py", "run_e_quantile_daily_multivariate.py",
        "run_foundation.py",
    }, with_default
    assert _defaults_in(SCRIPT), "the daily script's TG_HORIZON fallback was not found"


@pytest.mark.parametrize("path", RUNNERS + [SCRIPT], ids=lambda p: p.name)
def test_every_horizon_default_is_the_validated_horizon(path):
    found = _defaults_in(path)
    if not found:
        pytest.skip(f"{path.name} reads TG_HORIZON without a literal fallback")
    assert all(h == VALIDATED_HORIZON for h in found), (
        f"{path.name} falls back to horizon {found}; everything official is measured at "
        f"{VALIDATED_HORIZON}")
