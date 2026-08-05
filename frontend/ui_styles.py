# frontend/ui_styles.py — Shared CSS and UI component helpers
"""
Centralised styling for the AI4CM Streamlit interface.

Design principles:
    - Modern, professional analytics aesthetic (internal tool, not flashy)
    - Strong information hierarchy via typography and spacing
    - Color used intentionally: trust/success = teal, caution = amber, fail = rose
    - Consistent card containers with subtle depth
    - Readable numbers (monospace for metrics, proper comma formatting)
    - Local / lightweight — all CSS, no external dependencies
"""
from __future__ import annotations

import streamlit as st

# ── Brand palette ───────────────────────────────────────────────────────
# These are semantic colours rather than arbitrary brand colours.
COLORS = {
    "trust":       "#0d9488",   # teal-600
    "trust_bg":    "#f0fdfa",   # teal-50
    "trust_bdr":   "#99f6e4",   # teal-200
    "caution":     "#d97706",   # amber-600
    "caution_bg":  "#fffbeb",   # amber-50
    "caution_bdr": "#fde68a",   # amber-200
    "fail":        "#e11d48",   # rose-600
    "fail_bg":     "#fff1f2",   # rose-50
    "fail_bdr":    "#fecdd3",   # rose-200
    "info":        "#2563eb",   # blue-600
    "info_bg":     "#eff6ff",   # blue-50
    "info_bdr":    "#bfdbfe",   # blue-200
    "neutral":     "#64748b",   # slate-500
    "neutral_bg":  "#f8fafc",   # slate-50
    "neutral_bdr": "#e2e8f0",   # slate-200
    "text":        "#1e293b",   # slate-800
    "muted":       "#94a3b8",   # slate-400
    "bg":          "#ffffff",
    "surface":     "#f8fafc",   # slate-50
}


def inject_global_css():
    """Inject the global stylesheet once per session."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ── Re-usable component builders ───────────────────────────────────────

def metric_card(label: str, value: str, *, delta: str = "", icon: str = "",
                status: str = "neutral") -> str:
    """Return HTML for a styled metric card.

    status: "trust" | "caution" | "fail" | "info" | "neutral"
    """
    c = COLORS
    bg     = c.get(f"{status}_bg", c["neutral_bg"])
    border = c.get(f"{status}_bdr", c["neutral_bdr"])
    accent = c.get(status, c["neutral"])

    icon_html = f'<span class="mc-icon">{icon}</span>' if icon else ""
    delta_html = (
        f'<span class="mc-delta" style="color:{accent}">{delta}</span>'
        if delta else ""
    )

    return f"""
    <div class="metric-card" style="background:{bg}; border:1.5px solid {border};">
        <div class="mc-label">{icon_html}{label}</div>
        <div class="mc-value" style="color:{accent}">{value}</div>
        {delta_html}
    </div>
    """


def status_badge(text: str, status: str = "neutral") -> str:
    """Return HTML for an inline status badge/pill."""
    c = COLORS
    bg     = c.get(f"{status}_bg", c["neutral_bg"])
    border = c.get(f"{status}_bdr", c["neutral_bdr"])
    color  = c.get(status, c["neutral"])
    return (
        f'<span style="display:inline-block; padding:4px 14px; border-radius:20px; '
        f'font-size:13px; font-weight:600; background:{bg}; color:{color}; '
        f'border:1.5px solid {border}; letter-spacing:0.02em;">{text}</span>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Return HTML for a styled section header with optional subtitle."""
    sub = f'<p class="sh-sub">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="section-header">
        <h3 class="sh-title">{title}</h3>
        {sub}
    </div>
    """


def callout_box(message: str, status: str = "info", *, icon: str = "") -> str:
    """Return HTML for a callout/alert box."""
    c = COLORS
    bg     = c.get(f"{status}_bg", c["info_bg"])
    border = c.get(f"{status}_bdr", c["info_bdr"])
    color  = c.get(status, c["info"])
    icon_html = f'<span style="margin-right:8px;">{icon}</span>' if icon else ""
    return (
        f'<div class="callout-box" style="background:{bg}; border-left:4px solid {color}; '
        f'border-top:1px solid {border}; border-right:1px solid {border}; border-bottom:1px solid {border};">'
        f'{icon_html}<span style="color:{color};">{message}</span></div>'
    )


def page_header(title: str, subtitle: str = "") -> str:
    """Return HTML for a styled page header."""
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    return f'<h1 class="page-title">{title}</h1>{sub}'


def info_tip(text: str) -> str:
    """Return HTML for an inline info tip box."""
    return f'<div class="info-tip">💡 {text}</div>'


def glossary_table(rows: list) -> str:
    """Return HTML for a styled glossary table.

    rows: list of (term, definition) tuples
    """
    body = ""
    for term, defn in rows:
        body += f"<tr><td><b>{term}</b></td><td>{defn}</td></tr>"
    return (
        '<table class="glossary-table">'
        "<thead><tr><th>Term</th><th>Definition</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def grade_badge(grade: str) -> str:
    """Return a large, visually distinct grade badge (A/B/C/D/F)."""
    grade_map = {
        "A": ("trust",   "Excellent"),
        "B": ("trust",   "Good"),
        "C": ("caution", "Fair"),
        "D": ("caution", "Weak"),
        "F": ("fail",    "Poor"),
    }
    status, label = grade_map.get(grade, ("neutral", "N/A"))
    c = COLORS
    bg     = c.get(f"{status}_bg", c["neutral_bg"])
    border = c.get(f"{status}_bdr", c["neutral_bdr"])
    accent = c.get(status, c["neutral"])

    return f"""
    <div class="grade-badge" style="background:{bg}; border:2px solid {border};">
        <span class="gb-letter" style="color:{accent};">{grade}</span>
        <span class="gb-label" style="color:{accent};">{label}</span>
    </div>
    """


# ── Global CSS ─────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ── Typography ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Page-level layout ──────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 120px);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0f172a !important;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    border-radius: 8px;
    padding: 8px 12px;
    margin: 2px 8px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: rgba(13,148,136,0.2) !important;
    color: #5eead4 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"] * {
    color: #5eead4 !important;
}

/* Header area */
header[data-testid="stHeader"] {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #e2e8f0;
}

/* Main content padding */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(13,148,136,0.25) !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1.5px solid #e2e8f0 !important;
}

/* ── Select boxes & inputs ──────────────────────────────── */
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
.mc-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 8px;
}
.mc-icon {
    margin-right: 5px;
}
.mc-value {
    font-size: 26px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.mc-delta {
    font-size: 11px;
    font-weight: 600;
    margin-top: 6px;
    opacity: 0.8;
}

/* ── Grade badge ────────────────────────────────────────── */
.grade-badge {
    border-radius: 16px;
    padding: 18px 22px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 110px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}
.grade-badge:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
.gb-letter {
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
}
.gb-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

/* ── Section headers ────────────────────────────────────── */
.section-header {
    margin: 32px 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}
.sh-title {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 3px 0;
    letter-spacing: -0.01em;
}
.sh-sub {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}

/* ── Callout box ────────────────────────────────────────── */
.callout-box {
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0;
}

/* ── Data tables ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
[data-testid="stDataFrame"] table {
    font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] th {
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 10px !important;
    letter-spacing: 0.08em;
    color: #64748b !important;
    background: #f8fafc !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] td {
    font-size: 13px !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    gap: 0px;
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
}
[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    border-radius: 8px !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* ── Streamlit metric tweaks ────────────────────────────── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    transition: all 0.15s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-weight: 800;
}

/* ── Expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1.5px solid #e2e8f0;
    background: white;
    transition: all 0.15s ease;
}
[data-testid="stExpander"]:hover {
    border-color: #cbd5e1;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
}

/* ── Plotly chart containers ────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    border: 1.5px solid #e2e8f0;
    padding: 8px;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Trust verdict banner ───────────────────────────────── */
.trust-verdict {
    border-radius: 14px;
    padding: 22px 28px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.tv-icon {
    font-size: 36px;
    flex-shrink: 0;
}
.tv-text {
    flex: 1;
}
.tv-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
}
.tv-detail {
    font-size: 13px;
    opacity: 0.85;
    line-height: 1.5;
}

/* ── Model comparison cards ─────────────────────────────── */
.model-card {
    border-radius: 12px;
    padding: 18px;
    border: 1.5px solid #e2e8f0;
    background: white;
    margin-bottom: 10px;
    transition: all 0.15s ease;
}
.model-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.model-card.winner {
    border-color: #99f6e4;
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
}
.model-card.poor {
    border-color: #fecdd3;
    background: #fff1f2;
    opacity: 0.85;
}

/* ── Scrollable terminal ────────────────────────────────── */
.scroll-term {
    height: 360px;
    overflow: auto;
    background: #0f172a;
    color: #e2e8f0;
    padding: 16px 18px;
    border-radius: 12px;
    font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace !important;
    font-size: 12.5px;
    white-space: pre;
    border: 1px solid #1e293b;
    line-height: 1.6;
}

/* ── Info tooltip ───────────────────────────────────────── */
.info-tip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    font-size: 12px;
    color: #1e40af;
    line-height: 1.4;
    margin: 4px 0;
}

/* ── Page title ─────────────────────────────────────────── */
.page-title {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-top: 0;
    margin-bottom: 24px;
}

/* ── Glossary table ─────────────────────────────────────── */
.glossary-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.glossary-table th {
    background: #f1f5f9;
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    border-bottom: 2px solid #e2e8f0;
}
.glossary-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
}
.glossary-table tr:last-child td {
    border-bottom: none;
}
.glossary-table tr:hover td {
    background: #f8fafc;
}

/* ── Spacer helper ──────────────────────────────────────── */
.spacer-sm { height: 8px; }
.spacer-md { height: 16px; }
.spacer-lg { height: 28px; }
.spacer-xl { height: 48px; }

/* ── Download buttons ───────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #0d9488 !important;
}

/* ── Radio pills ────────────────────────────────────────── */
[data-testid="stRadio"] > div {
    gap: 6px;
}
[data-testid="stRadio"] label {
    border-radius: 8px;
    border: 1.5px solid #e2e8f0;
    padding: 6px 16px;
    font-weight: 500;
    font-size: 13px;
    transition: all 0.15s ease;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM (frontend-quality pass)
#
# One type scale, one spacing scale, one accent. Added rather than replacing the
# existing helpers so pages can migrate incrementally without a flag day.
#
# Two components below are load-bearing rather than decorative:
#   * gate_badge_tri  -- passed / failed / NEVER VERIFIED must be visually distinct in
#     colour AND word. "Never verified" reading as a pass is the worst failure this lab
#     can commit, because it converts an unmeasured thing into a reassuring one.
#   * reading_this_chart -- every non-obvious chart carries a plain-language caption. An
#     analyst who cannot tell what a chart claims will substitute their own guess.
# ══════════════════════════════════════════════════════════════════════════════

#: Accent, matched to reports/treasury_report.html so app and report read as one product.
# ── Console design tokens (DESIGN_TOKENS.md), adopted as the lab's look ──────────
#
# The operator explicitly overrode that document's own "not a request to restyle" line.
# Contrast independently re-measured (WCAG 2.x, sRGB); every pairing passes and none needed
# adjusting — the figures are stated in .streamlit/config.toml and reports/phase14_visual.md.
#
# Streamlit 1.40.1 carries only six theme keys, so everything below is what config cannot
# express and must be applied per element.
ACCENT = "#155860"          # slate-teal, primaryColor
ACCENT_INK = "#0E3E44"      # accent as ink on its own tint  (9.89:1)
ACCENT_TINT = "#E2EEEF"

TOK = {
    "bg": "#FCFBF9",            # warm paper
    "bg2": "#F5F3F0",
    "ink": "#1A1C1E",
    "muted": "#585E64",         # 5.93:1 on bg2
    "faint": "#696F76",         # 4.58:1 on bg2 — the tightest pairing in the set
    "hairline": "#E2DED8",      # decorative only; exempt under WCAG 1.4.11
    "control": "#96918A",       # >=3:1 boundary token — form edges, chart axes (3.13:1)
    "pass_ink": "#1B593B", "pass_tint": "#E2F0E7",     # 7.03:1
    "warn_ink": "#7C4A08", "warn_tint": "#F9EEDB",     # 6.44:1
    "stop_ink": "#8D2828", "stop_tint": "#F9E7E7",     # 7.14:1
}

#: Publication type scale — few deliberate steps, not a fluid ramp. (px, line-height px)
TYPE = {
    "micro":   (11, 16), "tiny": (12, 18), "small": (13, 20), "base": (15, 24),
    "lede":    (17, 26.4), "title": (22, 28), "display": (32, 36), "figure": (44, 46),
}

#: Max measure for prose. A verdict sentence spanning a projector is unreadable from the back.
MEASURE_CH = 72

FONT_STACK = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"

# ── Greyscale-survivable series encodings (DESIGN_TOKENS.md §3) ──────────────────
#
# These pages get printed and circulated, so a chart legible only in colour stops working the
# moment it leaves the screen. Every series carries a NON-COLOUR encoding as well.
SERIES_STYLE = {
    "p50":      dict(dash="solid", width=2.6, marker_symbol="circle"),
    "upper":    dict(dash="dash",  width=1.4, marker_symbol="triangle-up-open"),
    "lower":    dict(dash="dot",   width=1.4, marker_symbol="triangle-down-open"),
    "observed": dict(dash="dot",   width=1.6, marker_symbol="square-open"),
    "model":    dict(dash="solid", width=2.0, marker_symbol="circle-open"),
}
#: Band fill: a diagonal hatch, not a flat tint — a light tint disappears in greyscale.
BAND_PATTERN = dict(shape="/", size=6, solidity=0.12)


TYPE_SCALE = {"display": 27, "title": 20, "section": 17, "body": 15, "caption": 12.5}
SPACE = {"xs": 4, "sm": 8, "md": 14, "lg": 22, "xl": 34}

#: Tri-state gate vocabulary. The third state is the point of the whole thing.
GATE_PASSED = "passed"
GATE_FAILED = "failed"
GATE_UNVERIFIED = "unverified"

# Status inks on their own tint, per DESIGN_TOKENS §1.2 — ink on a soft tint of itself, never
# a light-on-light fill, which is what keeps them >=4.5:1 inside a badge or a table cell.
# The glyph is a second, non-colour encoding so the badge survives greyscale printing.
_GATE_STYLE = {
    GATE_PASSED: ("✓", "passed", TOK["pass_ink"], TOK["pass_tint"], TOK["pass_ink"]),
    GATE_FAILED: ("✕", "failed", TOK["stop_ink"], TOK["stop_tint"], TOK["stop_ink"]),
    GATE_UNVERIFIED: ("—", "never verified", TOK["faint"], TOK["bg2"], TOK["control"]),
}

DESIGN_CSS = f"""
<style>
:root {{
  --accent: {ACCENT}; --accent-ink: {ACCENT_INK}; --accent-tint: {ACCENT_TINT};
  --ink: {TOK['ink']}; --muted: {TOK['muted']}; --faint: {TOK['faint']};
  --hairline: {TOK['hairline']}; --control: {TOK['control']};
  --bg: {TOK['bg']}; --bg2: {TOK['bg2']};
}}
/* Tabular figures everywhere. GEL columns must align on the decimal, and a figure that changes
   width as it updates reads as unstable. Streamlit 1.40.1 cannot set this. */
html, body, [class*="st-"], .stMarkdown, .stMetric, .stDataFrame,
[data-testid="stMetricValue"], table td, table th {{
  font-variant-numeric: tabular-nums; font-feature-settings: 'tnum' 1;
}}
/* A readable measure for prose. Also cannot be set via config. */
.stMarkdown p {{ max-width: {MEASURE_CH}ch; }}
/* Metric labels as eyebrow text, matching the console. */
[data-testid="stMetricLabel"] {{
  font-size: {TYPE['micro'][0]}px; letter-spacing: 0.06em; text-transform: uppercase;
  color: {TOK['muted']};
}}
[data-testid="stMetricValue"] {{ font-size: {TYPE['title'][0]}px; letter-spacing: -0.011em; }}
h1 {{ font-size: {TYPE['display'][0]}px; line-height: {TYPE['display'][1]}px;
      letter-spacing: -0.02em; }}
h2 {{ font-size: {TYPE['title'][0]}px; line-height: {TYPE['title'][1]}px;
      letter-spacing: -0.011em; }}
h3 {{ font-size: {TYPE['lede'][0]}px; line-height: {TYPE['lede'][1]}px; }}

/* ── app header: logo + name + subtitle ─────────────────────────────────────── */
.ds-appbar {{ display:flex; align-items:center; gap:14px; padding:2px 0 12px;
  border-bottom:1px solid var(--hairline); margin-bottom:16px; }}
.ds-appbar .ds-logo {{ flex:0 0 auto; width:44px; height:44px; display:block; }}
.ds-appbar .ds-logo svg {{ width:100%; height:100%; display:block; }}
.ds-appbar .ds-name {{ font-size:{TYPE['lede'][0]}px; font-weight:650; color:var(--ink);
  line-height:1.2; }}
.ds-appbar .ds-sub {{ font-size:{TYPE['tiny'][0]}px; color:var(--muted); line-height:1.35;
  max-width:{MEASURE_CH}ch; }}

.ds-card {{ background:#fff; border:1px solid var(--hairline); border-radius:10px;
  padding:14px; margin-bottom:8px; }}
.ds-card .ds-label {{ font-size:{TYPE['micro'][0]}px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; margin:0 0 2px; }}
.ds-card .ds-value {{ font-size:{TYPE['title'][0]}px; font-weight:650; margin:0;
  font-variant-numeric: tabular-nums; }}
.ds-card .ds-sub {{ font-size:{TYPE['tiny'][0]}px; color:var(--muted); margin:2px 0 0; }}
.ds-card.ds-missing .ds-value {{ font-size:{TYPE['base'][0]}px; font-weight:500;
  color:var(--faint); font-style:italic; }}
.ds-gate {{ display:inline-flex; align-items:center; gap:6px;
  font-size:{TYPE['micro'][0]}px; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase; padding:3px 10px; border-radius:999px; border:1px solid; }}
.ds-reading {{ background:var(--bg2); border-left:3px solid var(--accent);
  padding:8px 12px; margin:6px 0 14px; font-size:{TYPE['tiny'][0]}px; color:var(--muted);
  border-radius:0 6px 6px 0; max-width:{MEASURE_CH}ch; }}
.ds-reading b {{ color:var(--ink); }}
.ds-empty {{ background:var(--bg2); border:1px dashed var(--control); border-radius:10px;
  padding:22px; color:var(--muted); font-size:{TYPE['base'][0]}px; }}
.ds-empty .ds-empty-t {{ font-weight:650; color:var(--ink); margin-bottom:6px; }}
.ds-empty code {{ background:#fff; border:1px solid var(--hairline); padding:1px 5px;
  border-radius:4px; font-size:{TYPE['tiny'][0]}px; }}
.ds-sample {{ background:{TOK['warn_tint']}; border:1px solid {TOK['warn_ink']};
  color:{TOK['warn_ink']}; font-weight:700; letter-spacing:.06em;
  font-size:{TYPE['tiny'][0]}px; padding:3px 10px; border-radius:6px;
  display:inline-block; }}
/* Tables: hairline rules and an eyebrow header row. */
[data-testid="stDataFrame"] thead th {{ font-size:{TYPE['micro'][0]}px;
  letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}

@media print {{
  [data-testid="stToolbar"], [data-testid="stSidebar"] {{ display:none !important; }}
  .stApp {{ background:#fff !important; }}
  .ds-card, .ds-empty, [data-testid="stMetric"] {{ break-inside: avoid; }}
}}
</style>
"""


def inject_design_system() -> None:
    """Call after inject_global_css() on every page."""
    import streamlit as st
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def ds_metric(label: str, value: str, *, sub: str = "", missing: bool = False) -> str:
    """Metric card. ``missing`` renders the value in the missing style, not as a number."""
    from format_gel import NOT_REPORTED
    if missing or value in (NOT_REPORTED, None, ""):
        value = value or NOT_REPORTED
        missing = True
    cls = "ds-card ds-missing" if missing else "ds-card"
    subhtml = f'<p class="ds-sub">{sub}</p>' if sub else ""
    return (f'<div class="{cls}"><p class="ds-label">{label}</p>'
            f'<p class="ds-value">{value}</p>{subhtml}</div>')


def gate_badge_tri(state: str, *, label: str = "") -> str:
    """Tri-state gate badge: passed / failed / never verified.

    ``state`` accepts the constants above, or ``True``/``False``/``None`` from an artifact.
    **None maps to "never verified", never to a pass** -- that mapping is the reason this
    component exists.
    """
    if state is True:
        state = GATE_PASSED
    elif state is False:
        state = GATE_FAILED
    elif state is None or state not in _GATE_STYLE:
        state = GATE_UNVERIFIED
    icon, word, fg, bg, bdr = _GATE_STYLE[state]
    text = f"{label} — {word}" if label else word
    return (f'<span class="ds-gate" style="color:{fg};background:{bg};border-color:{bdr}">'
            f'{icon} {text}</span>')


def reading_this_chart(text: str) -> str:
    """The plain-language caption that says what a chart claims."""
    return f'<div class="ds-reading"><b>Reading this chart.</b> {text}</div>'


def empty_state(what: str, *, filename: str, looked_in: str, command: str) -> str:
    """Honest empty state: what is missing, where we looked, and how to produce it.

    A blank panel makes an analyst wonder whether the number is zero or the page is broken.
    Naming the file and the command removes that question.
    """
    return (f'<div class="ds-empty"><div class="ds-empty-t">{what}</div>'
            f'<div>Expected file: <code>{filename}</code></div>'
            f'<div>Looked in: <code>{looked_in}</code></div>'
            f'<div style="margin-top:8px">Produce it with:</div>'
            f'<div><code>{command}</code></div></div>')


def sample_data_badge() -> str:
    """Header marker for any page showing demonstration rather than real data."""
    return '<span class="ds-sample">SAMPLE DATA</span>'


# ── plain-language tooltips, written once and reused ──────────────────────────
#
# Every non-obvious metric gets the same wording everywhere. Divergent explanations of the
# same number are how two analysts end up with two different beliefs about it.
HELP = {
    "skill": (
        "How much smaller this model's typical error is than the shared benchmark of "
        "'assume the value from five working days ago repeats'. 40% means errors are 40% "
        "smaller than that benchmark. Every model in the lab is measured against the same "
        "benchmark, so these numbers are comparable across model families."
    ),
    "sentinel": (
        "A self-test for whether the inputs actually inform the target. We shuffle the "
        "historical answers, refit, and see how much worse the model gets. If the inputs "
        "carry real information, destroying the link should hurt badly. We require the error "
        "to get at least 1.50x worse; below that we treat the model as tracking a typical "
        "level rather than anticipating individual days. 1.50 is a deliberate margin above "
        "1.00 (where shuffling changed nothing at all) so that noise cannot pass."
    ),
    "mase": (
        "Error divided by the error of a simple seasonal repeat, measured on the training "
        "period. Below 1.00 means better than that simple rule; above 1.00 means worse. "
        "Unlike a percentage error it stays meaningful on days when the actual value is near "
        "zero, which is why it replaces MAPE on the daily flow targets."
    ),
    "coverage": (
        "The share of actual values that landed inside the predicted range. If a range is "
        "advertised as covering 8 days in 10, coverage should be close to 80%. Well below "
        "that means the range is too narrow and understates risk."
    ),
    "tercile_coverage": (
        "Coverage split by how large the day is: the smallest third, middle third and "
        "largest third of days by magnitude. This matters more than the overall figure "
        "because a range can look well calibrated on average while missing most of the "
        "biggest days — and the biggest days are the ones a cash buffer exists for."
    ),
    "withheld": (
        "The model's numbers are shown, but we are not calling them a forecast. It passed "
        "its accuracy checks and failed the signal self-test, which means it tracks the "
        "typical level rather than anticipating individual days. The numbers are useful as a "
        "guide to the normal range; they should not be relied on to anticipate an unusual day."
    ),
    "ruler": (
        "The single shared benchmark: predict that the value five working days ago repeats. "
        "One implementation is used by every model family so that skill numbers are "
        "comparable. It is deliberately simple — beating it is a floor, not an achievement."
    ),
    "nominal": (
        "The range's advertised coverage — how often the actual value is supposed to land "
        "inside it. Read from the run's own artifact. Where an artifact does not record it, "
        "this lab reports it as not reported rather than assuming a level, because scoring a "
        "range against the wrong advertised level produces a verdict about nothing."
    ),
    "overfit_ratio": (
        "Validation error divided by training error. A model far better on data it trained "
        "on than on data it did not has memorised rather than learned. Above the gate "
        "threshold the model is excluded from best-model selection."
    ),
    "alignment": (
        "Whether each prediction was checked against the actual value for the date it "
        "claims to predict. Some model families record this as a fixed value rather than "
        "performing the check, in which case this lab shows 'never verified' rather than a "
        "pass."
    ),
}


def plotly_layout(fig, *, height: int = 380, ytitle: str = "", xtitle: str = "",
                  legend_bottom: bool = True):
    """One layout for every chart in the lab, so charts stop looking like different products."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                  size=TYPE_SCALE["caption"] + 0.5, color="#0f172a"),
        yaxis_title=ytitle or None,
        xaxis_title=xtitle or None,
        hoverlabel=dict(font_size=12.5, font_family="Inter, sans-serif"),
    )
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#e2e8f0")
    return fig


#: Hover templates. Values arrive already divided into millions by the caller.
HOVER_SERIES = ("<b>%{x|%a %d %b %Y}</b><br>%{fullData.name}: "
                "%{y:,.1f} M GEL<extra></extra>")
HOVER_BAND = ("<b>%{x|%a %d %b %Y}</b><br>"
              "P90 (upper): %{customdata[2]:,.1f} M GEL<br>"
              "P50 (central): %{customdata[1]:,.1f} M GEL<br>"
              "P10 (lower): %{customdata[0]:,.1f} M GEL<extra></extra>")
HOVER_BAR_PCT = "<b>%{x}</b><br>%{fullData.name}: %{y:.1%}<extra></extra>"


def plotly_chrome(fig, *, showlegend: bool = True, yaxis_tickformat: str = "",
                  height: int | None = None, kinds=None, greyscale: bool = True):
    """Apply the lab's chart chrome to an EXISTING figure.

    Presentation only: it never touches traces' x/y data or hover *content*. It applies the token
    template (surfaces, font, axis colours, colourway, hover label) and, unless switched off,
    gives each line trace a dash pattern and marker symbol so the chart survives greyscale
    printing — DESIGN_TOKENS §3.

    Same signature as before plus two optional arguments, so existing call sites upgrade with no
    edit.
    """
    fig.update_layout(
        template=plotly_template(),
        showlegend=showlegend,
        margin=dict(l=8, r=8, t=40, b=8),
    )
    if not showlegend:
        fig.update_layout(legend=None)
    if height:
        fig.update_layout(height=height)
    if yaxis_tickformat:
        fig.update_yaxes(tickformat=yaxis_tickformat)
    if greyscale:
        greyscale_safe(fig, kinds=kinds)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APP HEADER  (Part 2c)
#
# Renders frontend/assets/logo.svg AS PROVIDED — inlined verbatim, sized by CSS only. The file
# is the client's official Treasury emblem: it is not redrawn, recoloured, cropped or
# regenerated, and no fill/stroke is overridden. If the file is absent the header still renders
# with name and subtitle, because a missing mark must not take the page down.
# ══════════════════════════════════════════════════════════════════════════════

APP_NAME = "Treasury Forecast Lab"
APP_SUBTITLE = ("Daily cash-flow forecasting for the Georgian Treasury — research and "
                "evaluation workbench")

from pathlib import Path as _Path

_LOGO_PATH = _Path(__file__).resolve().parent / "assets" / "logo.svg"


def _logo_svg() -> str:
    """Inline the logo verbatim. Returns "" when the file is absent."""
    try:
        raw = _LOGO_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    # Strip only an XML prolog/doctype, which cannot appear mid-document. The <svg> element and
    # everything inside it — paths, fills, viewBox — is passed through untouched.
    import re as _re
    raw = _re.sub(r"<\?xml[^>]*\?>", "", raw)
    raw = _re.sub(r"<!DOCTYPE[^>]*>", "", raw)
    return raw.strip()


def app_header(page_title: str = "", page_subtitle: str = "") -> str:
    """The header for every page: logo, app name, one-line subtitle.

    ``page_title``/``page_subtitle`` override the app-level strings for a specific page.
    """
    logo = _logo_svg()
    logo_html = f'<span class="ds-logo">{logo}</span>' if logo else ""
    name = page_title or APP_NAME
    sub = page_subtitle or APP_SUBTITLE
    return (f'<div class="ds-appbar">{logo_html}'
            f'<span><span class="ds-name">{name}</span><br>'
            f'<span class="ds-sub">{sub}</span></span></div>')


def render_app_header(page_title: str = "", page_subtitle: str = "") -> None:
    import streamlit as st
    st.markdown(app_header(page_title, page_subtitle), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE  (Part 2b)
# ══════════════════════════════════════════════════════════════════════════════

def plotly_template():
    """A Plotly template matching the tokens. Registered once, applied per figure.

    Axes use the ``control`` border token (>=3:1) rather than the decorative hairline, because a
    chart axis carries information — DESIGN_TOKENS §1.2 makes that distinction explicitly.
    """
    import plotly.graph_objects as go
    return go.layout.Template(layout=go.Layout(
        paper_bgcolor=TOK["bg"], plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_STACK, size=TYPE["small"][0], color=TOK["ink"]),
        title=dict(font=dict(size=TYPE["lede"][0], color=TOK["ink"])),
        xaxis=dict(showgrid=False, linecolor=TOK["control"], ticks="outside",
                   tickcolor=TOK["control"], tickfont=dict(size=TYPE["tiny"][0],
                                                           color=TOK["muted"])),
        yaxis=dict(gridcolor=TOK["hairline"], zerolinecolor=TOK["control"],
                   linecolor=TOK["control"],
                   tickfont=dict(size=TYPE["tiny"][0], color=TOK["muted"])),
        legend=dict(orientation="h", y=-0.18, x=0,
                    font=dict(size=TYPE["tiny"][0], color=TOK["muted"])),
        hoverlabel=dict(font_size=TYPE["tiny"][0], font_family=FONT_STACK,
                        bgcolor="#FFFFFF", bordercolor=TOK["control"]),
        colorway=[ACCENT, TOK["stop_ink"], TOK["warn_ink"], TOK["pass_ink"],
                  TOK["muted"], ACCENT_INK],
        margin=dict(l=8, r=8, t=40, b=8),
    ))


def greyscale_safe(fig, *, kinds=None):
    """Give every line trace a non-colour encoding as well as colour.

    DESIGN_TOKENS §3: these pages get printed, and a chart legible only in colour stops working
    the moment it leaves the screen. Dash pattern and marker symbol are assigned by role where
    the caller names one, otherwise cycled so no two traces share both.

    Appearance only — no trace's x/y data is touched.
    """
    kinds = kinds or {}
    cycle = ["p50", "upper", "lower", "observed", "model"]
    i = 0
    for tr in fig.data:
        if getattr(tr, "mode", None) is None and tr.type != "scatter":
            continue
        role = kinds.get(getattr(tr, "name", "") or "", None)
        if role is None:
            role = cycle[i % len(cycle)]
            i += 1
        st_ = SERIES_STYLE.get(role, SERIES_STYLE["model"])
        try:
            tr.update(line=dict(dash=st_["dash"], width=st_["width"]))
        except Exception:
            pass
        try:
            if "markers" in (getattr(tr, "mode", "") or ""):
                tr.update(marker=dict(symbol=st_["marker_symbol"]))
        except Exception:
            pass
    return fig
