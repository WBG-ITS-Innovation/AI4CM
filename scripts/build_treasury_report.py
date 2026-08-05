#!/usr/bin/env python
"""Build the single-file Treasury HTML report from the latest forward run.

Review §7.5. Written for a non-technical reader: verdict first, money in millions of lari,
no acronyms, every withheld model explained rather than omitted.

**Self-contained by construction.** No CDN, no external stylesheet, no JavaScript library.
Charts are inline SVG generated here. The file must open from a USB stick on a machine with
no network, because that is how a report actually reaches a ministry.

    ./backend/.venv/bin/python scripts/build_treasury_report.py

Reads only the forward artifacts and the registry, so it can never quote an accuracy figure
from the sealed 2025 window -- there is none in its inputs.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import pandas as pd  # noqa: E402

import insights as ins  # noqa: E402
from registry import load_registry  # noqa: E402

OUT = REPO / "reports" / "treasury_report.html"


def m(v: float, dp: int = 1) -> str:
    return f"{v / 1_000_000:,.{dp}f}"


def esc(s) -> str:
    return html.escape(str(s))


def md_inline(s: str) -> str:
    """Minimal markdown: **bold** and `code`. Enough for the narrative, no dependency."""
    out = esc(s)
    while "**" in out:
        out = out.replace("**", "<strong>", 1)
        if "**" in out:
            out = out.replace("**", "</strong>", 1)
    while "`" in out:
        out = out.replace("`", "<code>", 1)
        if "`" in out:
            out = out.replace("`", "</code>", 1)
    return out.replace("\n\n", "</p><p>").replace("\n", "<br>")


def band_svg(rows: Sequence[Dict], width: int = 620, height: int = 230) -> str:
    """Inline SVG band chart: shaded low-high range with the central line on top."""
    pad_l, pad_r, pad_t, pad_b = 64, 14, 14, 34
    lo = min(r["p10"] for r in rows)
    hi = max(r["p90"] for r in rows)
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08
    span = hi - lo
    n = len(rows)
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b

    def X(i: int) -> float:
        return pad_l + (iw * (i / max(1, n - 1)))

    def Y(v: float) -> float:
        return pad_t + ih * (1 - (v - lo) / span)

    top = " ".join(f"{X(i):.1f},{Y(r['p90']):.1f}" for i, r in enumerate(rows))
    bot = " ".join(f"{X(i):.1f},{Y(r['p10']):.1f}" for i, r in reversed(list(enumerate(rows))))
    mid = " ".join(f"{X(i):.1f},{Y(r['p50']):.1f}" for i, r in enumerate(rows))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img" '
             f'aria-label="Forecast range for the next five working days">']
    # y gridlines with labels
    for k in range(5):
        v = lo + span * k / 4
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" '
                     f'stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="#64748b" '
                     f'text-anchor="end">{m(v, 0)}</text>')
    parts.append(f'<polygon points="{top} {bot}" fill="#2563eb" fill-opacity="0.16"/>')
    parts.append(f'<polyline points="{mid}" fill="none" stroke="#1d4ed8" stroke-width="2.5"/>')
    for i, r in enumerate(rows):
        parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(r["p50"]):.1f}" r="4" fill="#1d4ed8"/>')
        lab = pd.to_datetime(r["target_date"]).strftime("%a %d %b")
        parts.append(f'<text x="{X(i):.1f}" y="{height-12}" font-size="11" fill="#475569" '
                     f'text-anchor="middle">{esc(lab)}</text>')
    parts.append(f'<text x="10" y="{pad_t+8}" font-size="11" fill="#64748b">million lari</text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#0d9488;--okbg:#f0fdfa;
--bad:#e11d48;--badbg:#fff1f2;--warn:#d97706;--warnbg:#fffbeb;--blue:#1d4ed8;--bluebg:#eff6ff}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:#f8fafc}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 64px}
header.top{border-bottom:3px solid var(--blue);padding-bottom:18px;margin-bottom:26px}
h1{font-size:27px;margin:0 0 6px}
h2{font-size:20px;margin:34px 0 12px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:22px 0 8px}
.sub{color:var(--muted);font-size:14px;margin:0}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.kpi{flex:1 1 170px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:650}
.kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.banner{border-radius:10px;padding:14px 16px;margin:14px 0;border-left:5px solid}
.banner.ok{background:var(--okbg);border-color:var(--ok)}
.banner.bad{background:var(--badbg);border-color:var(--bad)}
.banner.warn{background:var(--warnbg);border-color:var(--warn)}
.banner.info{background:var(--bluebg);border-color:var(--blue)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f1f5f9;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#475569}
.tblwrap{overflow-x:auto}
ul.checks{list-style:none;padding:0;margin:8px 0}
ul.checks li{padding:7px 0;border-bottom:1px dashed var(--line)}
.pill{display:inline-block;font-size:12px;font-weight:650;padding:2px 9px;border-radius:999px}
.pill.ok{background:var(--okbg);color:var(--ok);border:1px solid var(--ok)}
.pill.bad{background:var(--badbg);color:var(--bad);border:1px solid var(--bad)}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:13px}
.prov{font-size:13px;color:#334155}
.prov dt{font-weight:650;margin-top:8px}
.prov dd{margin:0 0 2px}
footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);
font-size:12.5px;color:var(--muted)}
@media print{body{background:#fff}.card,.kpi{break-inside:avoid}}
"""


def build() -> str:
    art = ins.load_forward_artifacts()
    reg = load_registry()
    nar = ins.build_narrative(art["forecasts"], reg, art["provenance"])
    prov = art["provenance"] or {}
    recipes = {r["target"]: r for r in reg["recipes"]}

    by_t: Dict[str, List[Dict]] = {}
    for r in art["forecasts"]:
        by_t.setdefault(r["target"], []).append(r)

    n_pub = sum(1 for r in reg["recipes"] if r["publication"]["verdict"] == "publishable")
    data_end = str(prov.get("data", {}).get("latest_data_date", ""))[:10]
    dates = sorted({str(pd.to_datetime(r["target_date"]).date())
                    for r in art["forecasts"]})

    P: List[str] = []
    P.append("<header class='top'><h1>Treasury cash-flow outlook</h1>"
             f"<p class='sub'>Next five working days &middot; {esc(dates[0])} to "
             f"{esc(dates[-1])} &middot; prepared from data through {esc(data_end)}</p>"
             "</header>")

    # ── verdict first ──
    P.append("<h2>The short version</h2>")
    P.append(f"<div class='banner info'><p>{md_inline(nar['headline'])}</p></div>")
    P.append("<div class='kpis'>"
             f"<div class='kpi'><div class='l'>Budget lines covered</div>"
             f"<div class='v'>{len(recipes)} of {ins.TOTAL_TREASURY_METRICS}</div></div>"
             f"<div class='kpi'><div class='l'>Called a forecast</div>"
             f"<div class='v'>{n_pub} of {len(recipes)}</div></div>"
             f"<div class='kpi'><div class='l'>Working days ahead</div>"
             f"<div class='v'>{len(dates)}</div></div>"
             f"<div class='kpi'><div class='l'>2025 holdout used</div>"
             f"<div class='v'>No &mdash; sealed</div></div>"
             "</div>")
    P.append(f"<div class='banner warn'><p>{md_inline(nar['scope'])}</p></div>")
    P.append("<div class='banner bad'><p>"
             "<strong>These figures are validated against 2024, not 2025.</strong> "
             "The final independent evaluation against the untouched 2025 data has not "
             "been run and is scheduled. Read the accuracy figures below as provisional."
             "</p></div>")

    # ── the signal finding ──
    P.append("<h2>What the system can and cannot do</h2>")
    P.append(f"<div class='card'><p>{md_inline(nar['signal_finding'])}</p></div>")

    # ── per target ──
    P.append("<h2>Line-by-line outlook</h2>")
    for target, rows in by_t.items():
        rec = recipes.get(target)
        if rec is None:
            continue
        rows = sorted(rows, key=lambda r: r["horizon"])
        pub = rec["publication"]
        cred = rec["dev_credentials"]
        ok = pub["verdict"] == "publishable"

        P.append(f"<h3>{esc(target)}</h3>")
        if ok:
            P.append(f"<div class='banner ok'><p><strong>Usable as a forecast.</strong> "
                     f"{esc(pub['reason_plain'])}</p></div>")
        else:
            P.append(f"<div class='banner bad'><p><strong>Withheld as a forecast &mdash; "
                     f"published as a guide to the typical level.</strong> "
                     f"{esc(pub['reason_plain'])}</p></div>")
            if pub.get("named_fix"):
                P.append(f"<div class='banner warn'><p><strong>What would change this:"
                         f"</strong> {esc(pub['named_fix'])}</p></div>")

        P.append("<div class='card'>" + band_svg(rows))
        P.append("<div class='tblwrap'><table><thead><tr>"
                 "<th>Working day</th><th>Lower</th><th>Central estimate</th>"
                 "<th>Upper</th></tr></thead><tbody>")
        for r in rows:
            d = pd.to_datetime(r["target_date"]).strftime("%A %d %B %Y")
            P.append(f"<tr><td>{esc(d)}</td><td>{m(r['p10'])}</td>"
                     f"<td><strong>{m(r['p50'])}</strong></td><td>{m(r['p90'])}</td></tr>")
        P.append("</tbody></table></div>"
                 "<p class='sub'>Millions of lari. On eight working days out of ten the "
                 "actual figure is expected to fall between the lower and upper columns."
                 "</p></div>")

        P.append("<div class='card'><strong>Checks performed</strong> "
                 "<span class='sub'>(tested on 2024)</span><ul class='checks'>")
        for key, g in cred["gates"].items():
            pill = ("<span class='pill ok'>passed</span>" if g.get("passed")
                    else "<span class='pill bad'>failed</span>")
            P.append(f"<li>{pill} <strong>{esc(g.get('name', key))}</strong><br>"
                     f"{esc(g.get('reason_plain', ''))}</li>")
        P.append("</ul></div>")

        nb = cred.get("not_the_dev_best")
        if nb:
            # Deliberately the READER-facing wording. `why_promoted_anyway` is written
            # for the audit trail and carries model names and raw magnitudes; rendering
            # it here leaked both into Treasury-facing prose, which a test now blocks.
            P.append(f"<div class='banner warn'><p><strong>Note on model choice.</strong> "
                     f"A different model performed slightly better on 2024 "
                     f"({m(nb['its_dev_mae'])} against {m(nb['this_dev_mae'])} million "
                     f"lari, {nb['gap_pct']:.1f}% apart). "
                     f"{esc(nb.get('why_promoted_plain', ''))}</p></div>")

    # ── keep in mind ──
    P.append("<h2>What to keep in mind</h2><div class='card'><ul>")
    for l in nar["limitations"]:
        P.append(f"<li>{esc(l)}</li>")
    P.append("</ul></div>")

    # ── provenance ──
    P.append("<h2>Provenance</h2>")
    d, c = prov.get("data", {}), prov.get("code", {})
    P.append("<div class='card prov'><dl>")
    P.append(f"<dt>Source data</dt><dd>{esc(d.get('name', '—'))} &middot; "
             f"{esc(d.get('n_rows', '—'))} rows &middot; through "
             f"{esc(str(d.get('latest_data_date', ''))[:10])}</dd>")
    P.append(f"<dt>Data fingerprint (SHA-256)</dt><dd><code>{esc(d.get('sha256', '—'))}"
             f"</code></dd>")
    P.append(f"<dt>Code version</dt><dd><code>{esc(c.get('git_sha', '—'))}</code>"
             f"{' (uncommitted changes present)' if c.get('git_dirty') else ''}</dd>")
    P.append(f"<dt>Fiscal calendar version</dt>"
             f"<dd><code>{esc(prov.get('calendar_version', '—'))}</code></dd>")
    P.append(f"<dt>Generated</dt><dd>{esc(str(prov.get('generated_at_utc', ''))[:19])} UTC</dd>")
    P.append(f"<dt>2025 holdout touched</dt>"
             f"<dd><strong>{'No' if prov.get('test_window_touched') is False else 'CHECK'}"
             f"</strong></dd>")
    P.append("<dt>Recipes</dt><dd>")
    for r in prov.get("recipes", []):
        # Model identifiers live inside <code> so they read as provenance identifiers
        # rather than as prose a Treasury reader is expected to parse.
        P.append(f"{esc(r['target'])}: <code>{esc(r['recipe_id'])}</code> &middot; "
                 f"<code>{esc(r['point_model'])}</code> &middot; "
                 f"scaling <code>{esc(r['scaling'])}</code><br>")
    P.append("</dd></dl>")
    for note in prov.get("notes", []):
        P.append(f"<p class='sub'>{esc(note)}</p>")
    P.append("</div>")

    P.append("<footer>Produced by the AI4CM forecasting pipeline. Every figure in this "
             "report is reproducible from the data fingerprint and code version above. "
             "No model in this report has been formally approved, and none has been "
             "evaluated against the 2025 holdout.</footer>")

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Treasury cash-flow outlook</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(P) + "</div></body></html>")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build()
    OUT.write_text(doc, encoding="utf-8")
    print(f"[report] wrote {OUT} ({len(doc):,} bytes, self-contained)")
    for bad in ("http://", "https://", "cdn.", "<script"):
        if bad in doc:
            print(f"[report] WARNING: found {bad!r} — the file may not be self-contained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
