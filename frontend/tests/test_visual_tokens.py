"""Part 2: palette, logo and chart chrome.

Presentation only, but two things here are worth asserting rather than eyeballing:

* **AA contrast is re-measured, not quoted.** The ratios are computed from the token hex values
  in this test, so a future palette edit that breaks a threshold fails here rather than in a
  client review.
* **The logo is passed through verbatim.** The instruction is to use the client's official
  Treasury emblem as provided — not redrawn, recoloured, cropped or regenerated. A test is the
  only thing that stops a well-meaning "tidy-up" from altering it.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))

# ui_styles imports streamlit, which lives only in frontend/.venv. Skip cleanly rather than
# break collection, so a bare `pytest` from the repository root still runs the backend suite.
pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")

import ui_styles as u  # noqa: E402

PAGES = [FRONTEND / "Overview.py"] + sorted((FRONTEND / "pages").glob("*.py"))


# ── contrast, measured here ───────────────────────────────────────────────────

def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lum(hexs: str) -> float:
    h = hexs.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("name,fg,bg,threshold", [
    ("text on background", "#1A1C1E", "#FCFBF9", 4.5),
    ("text on secondary background", "#1A1C1E", "#F5F3F0", 4.5),
    ("primary on background", "#155860", "#FCFBF9", 4.5),
    ("primary on secondary background", "#155860", "#F5F3F0", 4.5),
    ("white on primary", "#FFFFFF", "#155860", 4.5),
])
def test_streamlit_theme_pairings_meet_aa(name, fg, bg, threshold):
    r = contrast(fg, bg)
    assert r >= threshold, f"{name}: {r:.2f}:1 is below {threshold}:1"


@pytest.mark.parametrize("role,ink,tint,threshold", [
    ("pass", "pass_ink", "pass_tint", 4.5),
    ("warn", "warn_ink", "warn_tint", 4.5),
    ("stop", "stop_ink", "stop_tint", 4.5),
    ("muted text", "muted", "bg2", 4.5),
    ("faint text", "faint", "bg2", 4.5),
    ("control border", "control", "bg", 3.0),
])
def test_status_tokens_meet_aa_on_their_own_tint(role, ink, tint, threshold):
    """Status colours are ink on a soft tint of themselves, never a light-on-light fill —
    which is what keeps them legible inside a badge or a table cell."""
    r = contrast(u.TOK[ink], u.TOK[tint])
    assert r >= threshold, f"{role}: {r:.2f}:1 is below {threshold}:1"


def test_config_uses_only_the_six_supported_theme_keys():
    """Streamlit 1.40.1 accepts exactly six. Anything else is silently ignored, which would
    make the config look like it carried tokens it does not."""
    cfg = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())
    allowed = {"base", "primaryColor", "backgroundColor", "secondaryBackgroundColor",
               "textColor", "font"}
    assert set(cfg["theme"]) <= allowed, set(cfg["theme"]) - allowed


def test_config_carries_the_console_tokens_not_the_old_lab_palette():
    cfg = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())["theme"]
    assert cfg["primaryColor"].upper() == "#155860", "primary is not the console slate-teal"
    assert cfg["backgroundColor"].upper() == "#FCFBF9", "background is not the warm paper"
    assert cfg["primaryColor"].upper() != "#1D4ED8", "the old lab blue is back"


def test_greyscale_separation_of_accent_from_background():
    """A printer renders both as luma. Under about 60 of 255 the accent muddies into the page."""
    def luma(hexs):
        h = hexs.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return 0.299 * r + 0.587 * g + 0.114 * b
    sep = abs(luma(u.TOK["bg"]) - luma(u.ACCENT))
    assert sep >= 60, f"accent/background greyscale separation is only {sep:.0f} of 255"


# ── the logo is used as provided ──────────────────────────────────────────────

def test_logo_file_exists_and_is_svg():
    p = FRONTEND / "assets" / "logo.svg"
    assert p.exists(), "frontend/assets/logo.svg is missing; the header cannot show the emblem"
    assert "<svg" in p.read_text(encoding="utf-8")[:400]


def test_logo_is_inlined_verbatim_not_redrawn_or_recoloured():
    """The emblem is the client's official mark. The header may size it; it may not alter it.

    Compared path-by-path against the file on disk: every `d=` geometry and every fill/stroke
    must survive into the rendered header unchanged.
    """
    raw = (FRONTEND / "assets" / "logo.svg").read_text(encoding="utf-8")
    rendered = u.app_header()

    paths = re.findall(r'\sd="([^"]{20,})"', raw)
    assert paths, "fixture assumption: the logo has path geometry"
    for d in paths:
        assert d in rendered, "a path was altered or dropped when inlining the logo"

    for attr in re.findall(r'\s(?:fill|stroke)="([^"]+)"', raw):
        if attr.lower() == "none":
            continue
        assert attr in rendered, f"colour {attr!r} was changed when inlining the logo"

    vb = re.search(r'viewBox="([^"]+)"', raw)
    if vb:
        assert vb.group(1) in rendered, "viewBox changed — the logo was cropped or rescaled"


def test_header_survives_a_missing_logo(monkeypatch, tmp_path):
    """A missing mark must not take the page down; name and subtitle still render."""
    monkeypatch.setattr(u, "_LOGO_PATH", tmp_path / "absent.svg")
    h = u.app_header("A page", "A subtitle")
    assert "<svg" not in h
    assert "A page" in h and "A subtitle" in h


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_renders_the_app_header(page):
    src = page.read_text(encoding="utf-8")
    assert "render_app_header(" in src, f"{page.name} has no app header"


# ── chart chrome ──────────────────────────────────────────────────────────────

def test_chrome_applies_the_template_and_leaves_data_untouched():
    import plotly.graph_objects as go

    fig = go.Figure([go.Scatter(x=[1, 2, 3], y=[10.0, 20.0, 30.0], mode="lines+markers",
                                name="P50")])
    before_x = list(fig.data[0].x)
    before_y = list(fig.data[0].y)
    u.plotly_chrome(fig)
    assert list(fig.data[0].x) == before_x, "chrome altered x data"
    assert list(fig.data[0].y) == before_y, "chrome altered y data"
    assert fig.layout.template is not None


def test_chrome_gives_traces_a_non_colour_encoding():
    """DESIGN_TOKENS §3: a chart legible only in colour stops working when printed."""
    import plotly.graph_objects as go

    fig = go.Figure([
        go.Scatter(x=[1, 2], y=[1, 2], mode="lines", name="P50"),
        go.Scatter(x=[1, 2], y=[2, 3], mode="lines", name="Upper"),
        go.Scatter(x=[1, 2], y=[0, 1], mode="lines", name="Lower"),
    ])
    u.plotly_chrome(fig, kinds={"P50": "p50", "Upper": "upper", "Lower": "lower"})
    dashes = [t.line.dash for t in fig.data]
    assert dashes == ["solid", "dash", "dot"], dashes
    assert len(set(dashes)) == 3, "traces are not distinguishable without colour"


def test_band_pattern_is_a_hatch_not_a_flat_tint():
    """A light tint disappears in greyscale; a hatch does not."""
    assert u.BAND_PATTERN["shape"] == "/"
