"""The two VALIDATED_HORIZON literals agree, and there is no third.

Why this file exists
--------------------
``VALIDATED_HORIZON = 5`` is written twice: ``backend/forecast_modes.py``, which refuses an
official run at any other horizon, and ``frontend/pages/07_Forecast.py``, which drives the
page's caption, the exploratory slider's default and its warning. Nothing compared them, so a
change to one would leave the page describing a horizon the backend no longer accepts
(inference-horizon map, §2.5 and §2.7).

Decision of 2026-10-01: a drift test, not a shared module. The two interpreters keep their
separate import stacks, and this test is what fails when the literals differ. It reads the
page as text rather than importing it, because the page is a Streamlit script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
FRONTEND = REPO / "frontend"
sys.path.insert(0, str(BACKEND))

from forecast_modes import VALIDATED_HORIZON  # noqa: E402

PAGE = FRONTEND / "pages" / "07_Forecast.py"
LITERAL = re.compile(r"^VALIDATED_HORIZON\s*=\s*(\d+)\s*$", re.MULTILINE)


def page_literal(text: str) -> int:
    """The page's module-level literal. Exactly one must exist, or the check means nothing."""
    found = LITERAL.findall(text)
    assert len(found) == 1, f"expected one VALIDATED_HORIZON literal on the page, found {found}"
    return int(found[0])


def test_the_page_literal_equals_the_backend_constant():
    assert page_literal(PAGE.read_text(encoding="utf-8")) == VALIDATED_HORIZON, (
        f"frontend/pages/07_Forecast.py says VALIDATED_HORIZON = "
        f"{page_literal(PAGE.read_text(encoding='utf-8'))}, backend/forecast_modes.py says "
        f"{VALIDATED_HORIZON}; the page would describe a horizon the backend refuses")


def test_the_check_fails_on_a_drifted_literal():
    """The comparison itself, proven against a doctored copy of the page's text."""
    drifted = LITERAL.sub(f"VALIDATED_HORIZON = {VALIDATED_HORIZON + 5}",
                          PAGE.read_text(encoding="utf-8"), count=1)
    assert page_literal(drifted) != VALIDATED_HORIZON


def test_no_third_copy_of_the_literal_exists():
    """Two literals are checked here; a third would be unchecked and could drift unseen."""
    copies = []
    for p in sorted(list(FRONTEND.glob("*.py")) + list((FRONTEND / "pages").glob("*.py"))
                    + list(BACKEND.glob("*.py")) + list((REPO / "scripts").glob("*.py"))):
        if LITERAL.search(p.read_text(encoding="utf-8")):
            copies.append(p.relative_to(REPO).as_posix())
    assert copies == ["backend/forecast_modes.py", "frontend/pages/07_Forecast.py"], copies
