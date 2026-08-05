"""One place that formats Georgian lari for display.

Three pages were each rounding and separating money their own way, so the same figure
could read as 94,751,609 on one page and 94.75M on another. A Treasury reader comparing
two screens should never have to work out whether they are looking at the same number.

Millions with one decimal is the ministry's own convention for daily flows.
"""
from __future__ import annotations

from typing import Optional


def gel_millions(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format a raw GEL amount as millions. Returns an em dash for missing values."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v / 1_000_000:,.{decimals}f}{suffix}"


def gel_millions_signed(value: Optional[float], decimals: int = 1) -> str:
    """As above, with an explicit sign -- for changes and differences."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v / 1_000_000:+,.{decimals}f}"


UNIT_LABEL = "million lari"
UNIT_SHORT = "M GEL"
