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
