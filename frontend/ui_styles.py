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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: box-shadow 0.15s ease;
}
.metric-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.mc-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 6px;
}
.mc-icon {
    margin-right: 4px;
}
.mc-value {
    font-size: 28px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.mc-delta {
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
}

/* ── Grade badge ────────────────────────────────────────── */
.grade-badge {
    border-radius: 16px;
    padding: 16px 20px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100px;
}
.gb-letter {
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.gb-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Section headers ────────────────────────────────────── */
.section-header {
    margin: 28px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}
.sh-title {
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 2px 0;
}
.sh-sub {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}

/* ── Callout box ────────────────────────────────────────── */
.callout-box {
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.5;
    margin: 8px 0;
}

/* ── Data tables ────────────────────────────────────────── */
[data-testid="stDataFrame"] table {
    font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] th {
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 11px !important;
    letter-spacing: 0.06em;
    color: #64748b !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {
    gap: 0px;
}
[data-testid="stHorizontalBlock"] [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 14px;
    padding: 8px 18px;
}

/* ── Streamlit metric tweaks ────────────────────────────── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
}

/* ── Expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

/* ── Plotly chart containers ────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 4px;
    background: white;
}

/* ── Trust verdict banner ───────────────────────────────── */
.trust-verdict {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.tv-icon {
    font-size: 32px;
    flex-shrink: 0;
}
.tv-text {
    flex: 1;
}
.tv-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 4px;
}
.tv-detail {
    font-size: 13px;
    opacity: 0.85;
}

/* ── Model comparison cards ─────────────────────────────── */
.model-card {
    border-radius: 10px;
    padding: 16px;
    border: 1.5px solid #e2e8f0;
    background: white;
    margin-bottom: 8px;
}
.model-card.winner {
    border-color: #99f6e4;
    background: #f0fdfa;
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
    background: #0b0b0b;
    color: #e6e6e6;
    padding: 12px 14px;
    border-radius: 10px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 13px;
    white-space: pre;
    border: 1px solid #334155;
}

/* ── Spacer helper ──────────────────────────────────────── */
.spacer-sm { height: 8px; }
.spacer-md { height: 16px; }
.spacer-lg { height: 28px; }
</style>
"""
