"""Display formatting for Georgian lari, and the missing-value contract.

Two rules the whole lab depends on.

**Millions, always.** Treasury flows run to 1e8-1e9 raw. Printing those digits to an analyst
is a dump, not communication; millions with one decimal is the unit the ministry speaks in.

**A missing value reads as missing.** ``NOT_REPORTED`` is the single rendering for absent
data. It is deliberately a *phrase*, not ``0`` and not an em dash: both of those read as
numbers on a dashboard, and "coverage 0" is a catastrophic model while "coverage not
reported" is an unmeasured one. Confusing the two is the single most misleading thing a
lab like this can do, so it is centralised here rather than left to each page.
"""
from __future__ import annotations

import math
from typing import Any, Optional

#: The one rendering for absent data anywhere in the lab.
NOT_REPORTED = "not reported"

#: Used where a value is present but was never independently checked -- distinct from both a
#: real value and a missing one. See the tri-state gate badge in ui_styles.
NOT_VERIFIED = "not independently verified"

UNIT_LABEL = "million lari"
UNIT_SHORT = "M GEL"


def is_missing(value: Any) -> bool:
    """True for None, NaN, empty string, and the strings pandas writes for nulls."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def gel_millions(value: Any, decimals: int = 1, suffix: str = "") -> str:
    """Format a raw GEL amount as millions. Missing renders as ``NOT_REPORTED``."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NOT_REPORTED
    if math.isinf(v):
        return NOT_REPORTED
    return f"{v / 1_000_000:,.{decimals}f}{suffix}"


def gel_millions_signed(value: Any, decimals: int = 1) -> str:
    """As above with an explicit sign -- for changes and differences."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NOT_REPORTED
    return f"{v / 1_000_000:+,.{decimals}f}"


def pct(value: Any, decimals: int = 1, signed: bool = False) -> str:
    """Percentage from a fraction (0.83 -> '83.0%'). Missing renders as ``NOT_REPORTED``."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        v = float(value) * 100.0
    except (TypeError, ValueError):
        return NOT_REPORTED
    return f"{v:{'+' if signed else ''}.{decimals}f}%"


def pct_points(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Percentage from a value already in percent (55.92 -> '55.92%')."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NOT_REPORTED
    return f"{v:{'+' if signed else ''}.{decimals}f}%"


def ratio(value: Any, decimals: int = 2) -> str:
    """A bare ratio such as the sentinel (1.2255 -> 'x1.23')."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        return f"x{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return NOT_REPORTED


def count(value: Any) -> str:
    if is_missing(value):
        return NOT_REPORTED
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return NOT_REPORTED


def number(value: Any, decimals: int = 3) -> str:
    """A plain number such as MASE. Missing renders as ``NOT_REPORTED``."""
    if is_missing(value):
        return NOT_REPORTED
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return NOT_REPORTED
