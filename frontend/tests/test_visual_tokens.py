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


def test_the_emblem_file_still_holds_its_own_geometry_and_colours():
    """The emblem is the client's official mark and is used exactly as delivered.

    The mark moved out of the page header and into the sidebar brand, where it is handed to
    ``st.logo`` as a file path. Streamlit reads an ``.svg`` path and serves that file's own
    text, adding only an ``xmlns`` attribute if one is missing, and ours has one. So passing
    the path IS passing the file, and what remains to protect is the file itself: nobody may
    tidy, recolour, crop or regenerate the asset.
    """
    raw = (FRONTEND / "assets" / "logo.svg").read_text(encoding="utf-8")

    paths = re.findall(r'\sd="([^"]{20,})"', raw)
    assert len(paths) >= 10, f"the emblem has lost path geometry: {len(paths)} paths left"

    fills = set(re.findall(r'\sfill="(#[0-9a-fA-F]{3,6})"', raw))
    assert {"#ffbe00", "#003159"} <= {f.lower() for f in fills}, (
        f"the emblem's gold or navy has been changed: {sorted(fills)}")

    assert re.search(r'viewBox="0 0 77\.199 78\.92"', raw), \
        "viewBox changed: the emblem was cropped or rescaled"
    assert "xmlns" in raw, "the emblem lost its xmlns and will not render in an img tag"


def test_the_brand_hands_streamlit_the_file_and_not_a_copy_of_it():
    """Our code must pass the path through, never a transformed or embedded version."""
    import inspect

    src = inspect.getsource(u.render_brand)
    assert "st.logo(str(_LOGO_PATH)" in src, (
        "render_brand no longer hands st.logo the emblem's own path")
    assert u._LOGO_PATH.name == "logo.svg" and u._LOGO_PATH.parent.name == "assets"
    assert u._LOGO_PATH.exists(), "the emblem the brand points at is missing"

    # And the stylesheet must not carry a second, embedded copy that could drift from the file.
    css = u.brand_css("Some Wordmark")
    assert "<svg" not in css and "base64" not in css, (
        "the brand stylesheet embeds an image; the emblem has two sources now")


def test_the_brand_is_one_light_plaque_because_the_sidebar_is_dark():
    """The emblem is gold and navy. The sidebar is navy. So the plaque behind it is light.

    Recolouring the mark to suit the sidebar is not available, so this is the alternative and
    it is asserted rather than left to a future tidy-up that "simplifies" the background away.
    """
    css = u.brand_css("Some Wordmark")
    assert 'data-testid="stSidebarHeader"' in css, "the seal has no plaque behind it"
    assert 'data-testid="stSidebarNav"]::before' in css, "the wordmark is not above the nav"
    assert css.count(u.TOK["bg"]) >= 2, (
        "the seal's plaque and the wordmark's band are not the same light colour")


def test_the_wordmark_is_one_string_and_reaches_the_stylesheet():
    """One string, so changing the name is one edit plus its translation."""
    assert isinstance(u.WORDMARK, str) and u.WORDMARK.strip()
    assert "\u2014" not in u.WORDMARK and "--" not in u.WORDMARK, \
        "the wordmark carries punctuation the house style forbids"
    assert u.WORDMARK in u.brand_css(u.WORDMARK)


def test_brand_survives_a_missing_emblem(monkeypatch, tmp_path):
    """A missing mark must not take the sidebar down; the wordmark still renders."""
    monkeypatch.setattr(u, "_LOGO_PATH", tmp_path / "absent.svg")
    assert u.WORDMARK in u.brand_css(u.WORDMARK)


def test_the_page_header_no_longer_carries_the_emblem():
    """It used to, which put a Treasury seal beside the word "Scorecard" on every page.

    The mark belongs to the app, so it is shown once in the sidebar. A page header is text.
    """
    h = u.app_header("A page", "A subtitle")
    assert "<svg" not in h and "ds-logo" not in h and "ds-appbar" not in h
    assert "A page" in h and "A subtitle" in h
    assert h.startswith("<h1"), f"a page title should be an h1: {h[:60]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_shows_the_brand_exactly_once(page):
    src = page.read_text(encoding="utf-8")
    assert src.count("render_brand()") == 1, f"{page.name} does not call render_brand() once"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_page_draws_a_second_title_of_its_own(page):
    """Every page used to draw render_app_header AND a bigger page_header saying the same.

    One title per page. The intro sentence below it is page_intro's job.
    """
    src = page.read_text(encoding="utf-8")
    assert 'page_header("' not in src, (
        f"{page.name} draws a second page title; render_app_header is the only one")


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


# ── radio pills: the empty pill beside the Language selector ──────────────────
#
# Symptom: an empty bordered pill sat next to the "Language / ენა" label in the sidebar.
#
# Cause, read out of the installed Streamlit build rather than guessed at. Streamlit renders
# THREE different <label> elements inside [data-testid="stRadio"]:
#
#   1. the widget's own label, label[data-testid="stWidgetLabel"], holding the label text
#      (main.js module 78286, the styled element exported as `Yv`);
#   2. the help-tooltip wrapper, also a <label>, holding the "?" icon AND NO TEXT
#      (same module, exported as `Cl`), present only when the widget is given help=;
#   3. one <label> per option, inside div[role="radiogroup"] (the radio chunk contains
#      exactly one styled "label", the option Root, so there is one per option and no nesting).
#
# The rule was written as `[data-testid="stRadio"] label`, which borders all three. Number 2
# is the empty pill: a bordered box with the rule's 6px/16px padding around an icon and no
# words. It was never specific to the language toggle. Five of the six radios in the app pass
# help= and all five had it; the language one is simply the one on every page.
#
# Note for anyone editing _GLOBAL_CSS: its contents, comments included, are injected into the
# page inside a <style> block, so they reach the browser and the page body. A CSS comment here
# quoting the Georgian label text is what test_english_mode_renders_no_georgian_body_text
# caught on all ten pages. Long explanations belong in a test, which is why this one is here.

def test_radio_pills_are_scoped_to_the_options():
    """The pill border must reach the options and nothing else in the widget."""
    rules = re.findall(r'([^{}]*?)\{[^{}]*?border\s*:[^{}]*?\}', u._GLOBAL_CSS, re.S)
    radio_label_rules = [r.strip() for r in rules
                         if 'stRadio' in r and 'label' in r]
    assert radio_label_rules, "the radio pill rule has gone; this test guards nothing"
    for selector in radio_label_rules:
        assert '[role="radiogroup"]' in selector, (
            "this borders every <label> in the widget, including the help-tooltip wrapper, "
            f"which has no text and renders as an empty pill: {selector!r}")


def test_no_rule_targets_every_label_inside_a_radio():
    """The unscoped selector must not come back by another route."""
    offenders = re.findall(r'\[data-testid="stRadio"\]\s+label', u._GLOBAL_CSS)
    assert not offenders, (
        "an unscoped `[data-testid=\"stRadio\"] label` selector is back; scope it to "
        '`[data-testid="stRadio"] [role="radiogroup"] label`')


@pytest.mark.parametrize("language", ["en", "ka"])
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_language_radio_still_renders_both_options(page, language):
    """The fix is presentational, so every radio must survive it in both languages.

    Asserted on the language toggle because it is the widget the bug was reported against and
    the only one present on every page. Its options are the formatted names, so this also
    confirms the format_func still runs.
    """
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(REPO / "backend"))
    from i18n import LANGUAGES, SESSION_KEY

    at = AppTest.from_file(str(page), default_timeout=180)
    at.session_state[SESSION_KEY] = language
    at.run()
    assert not at.exception, f"{page.name} raised in {language}"

    radios = [r for r in at.get("radio") if r.options == list(LANGUAGES.values())]
    assert len(radios) == 1, (
        f"{page.name} in {language}: expected exactly one language radio offering "
        f"{list(LANGUAGES.values())}, found {[r.options for r in at.get('radio')]}")
    assert radios[0].label.strip(), "the language radio lost its label"


def test_every_radio_in_the_app_keeps_its_options():
    """No other radio regressed: every one still offers options and none of them is blank.

    Deliberately not "two or more". The Dashboard's cadence radio is built from the cadence
    folders the selected run actually wrote, so a daily-only run offers exactly one and that
    is correct. A blank option is the thing an empty pill would look like from here, so that
    is what this checks.
    """
    from streamlit.testing.v1 import AppTest

    for page in PAGES:
        at = AppTest.from_file(str(page), default_timeout=180)
        at.run()
        assert not at.exception, f"{page.name} raised"
        for r in at.get("radio"):
            assert r.options, f"{page.name}: radio {r.label!r} renders no options at all"
            assert all(str(o).strip() for o in r.options), (
                f"{page.name}: radio {r.label!r} has a blank option: {r.options}")
