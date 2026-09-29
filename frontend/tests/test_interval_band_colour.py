"""The interval band must take its colour from whatever the palette actually hands it.

The bug this holds down
-----------------------
The Dashboard drew the prediction-interval band by slicing the series colour as if it
were always ``#RRGGBB``::

    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"

The palette is ``px.colors.qualitative.Set2``, and every one of its eight entries is an
``rgb()`` string, not hex. So ``color[1:3]`` was ``'gb'`` and the call raised
``ValueError: invalid literal for int() with base 16: 'gb'``.

This was not an edge case reachable by an unusual palette: with the interval toggle on,
interval data present and a single model selected, it raised every time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
px = pytest.importorskip("plotly.express", reason="plotly is installed in frontend/.venv only")

import sys  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

DASHBOARD = FRONTEND / "pages" / "04_Dashboard.py"
SOURCE = DASHBOARD.read_text(encoding="utf-8")


def _helper():
    try:
        from ui_styles import color_with_alpha
    except ImportError:
        pytest.fail("ui_styles has no `color_with_alpha`: the band colour is still being "
                    "parsed by slicing hex out of the palette entry")
    return color_with_alpha


def test_every_entry_of_the_palette_the_page_uses_converts():
    """The regression, stated against the real palette rather than a sample of it."""
    convert = _helper()
    for entry in px.colors.qualitative.Set2:
        out = convert(entry, 0.15)
        assert out.startswith("rgba("), entry
        assert out.endswith(",0.15)"), entry


def test_an_rgb_string_keeps_its_channels():
    convert = _helper()
    assert convert("rgb(102,194,165)", 0.15) == "rgba(102,194,165,0.15)"


def test_hex_still_works_because_other_callers_pass_it():
    convert = _helper()
    assert convert("#155860", 0.15) == "rgba(21,88,96,0.15)"


def test_spacing_and_case_do_not_matter():
    convert = _helper()
    assert convert("RGB( 102 , 194 , 165 )", 0.2) == "rgba(102,194,165,0.2)"
    assert convert("#FCFBF9", 0.2) == convert("#fcfbf9", 0.2)


def test_an_unparseable_colour_raises_rather_than_drawing_something_wrong():
    """Silently substituting a colour would hide a palette change; a band is not worth that."""
    convert = _helper()
    with pytest.raises(ValueError):
        convert("chartreuse-ish", 0.15)


def test_the_dashboard_no_longer_slices_hex_out_of_the_palette():
    assert "int(color[1:3],16)" not in SOURCE.replace(" ", ""), (
        "04_Dashboard.py is slicing hex out of a palette entry again; an rgb() entry "
        "raises ValueError")
