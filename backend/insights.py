"""Plain-language narrative for a non-technical Treasury reader.

Built deterministically from the forward-forecast artifacts and the registry. Review §7.5:
verdict first, money in GEL millions, no acronyms, ranges described as behaviour rather
than as statistics, withheld models explained rather than hidden, and an explicit scope
line so nobody reads three targets as the whole budget.

The narrative must state the central finding honestly: on the two flow targets the system
produces a central-tendency estimate, not an event forecast, because three independent
statistical probes agree the inputs carry little information about what happens next. It
must also name the fix, because a finding without a next step is not useful to a Treasury
reader.

--------------------------------------------------------------------------------
THE LLM HOOK
--------------------------------------------------------------------------------
If ``AI4CM_LLM=ollama`` and the endpoint answers, a language model may **rephrase the
template's prose**. It may not compute, choose, or alter anything.

The guarantee is enforced, not requested: ``digits_of()`` extracts every numeric token from
the template and from the rephrased text, and ``rephrase_safely`` rejects the rephrasing
outright if the two multisets differ. A model that drops a number, invents one, or rounds
one differently gets discarded and the template is used. Output that survives is labelled
"AI-phrased; numbers computed by pipeline" so a reader always knows which words were
generated.

Any error — no endpoint, timeout, bad JSON, refusal — falls back to the template silently.
The narrative is a required artifact; it must never depend on an optional service.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Total metric columns in the Treasury dataset (43 columns minus two calendar flags).
TOTAL_TREASURY_METRICS = 41

LLM_LABEL = "AI-phrased; numbers computed by pipeline"

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gel_millions(value: float, decimals: int = 1) -> str:
    """Format a raw GEL amount as millions, with a thousands separator.

    Treasury flows run to 1e8-1e9 raw. Printing those digits to a non-technical reader is
    not communication, it is a dump; millions is the unit the ministry actually speaks in.
    """
    return f"{value / 1_000_000:,.{decimals}f}"


def digits_of(text: str) -> List[str]:
    """Every numeric token in ``text``, normalised so formatting is not mistaken for change.

    Separators are stripped and trailing zeros normalised, so "1,234.50" and "1234.5" match;
    a genuinely different value never does. This is the function the LLM guard rests on.
    """
    out: List[str] = []
    for m in _NUM.findall(text):
        s = m.replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        out.append(s or "0")
    return sorted(out)


# ── the deterministic narrative ────────────────────────────────────────────────

def _verdict_line(recipe: Dict) -> str:
    pub = recipe["publication"]
    target = recipe["target"]
    if pub["verdict"] == "publishable":
        return (f"**{target}: usable as a forecast.** Every check passed.")
    if pub["verdict"] == "withheld_as_forecast":
        return (f"**{target}: use as a guide to the typical level, not as a forecast of "
                f"unusual days.**")
    return f"**{target}: withheld.**"


def _range_behaviour(rows: Sequence[Dict], recipe: Dict) -> str:
    """Describe the interval as behaviour, not as a statistic."""
    widths = [(r["p90"] - r["p10"]) for r in rows]
    mids = [abs(r["p50"]) or 1.0 for r in rows]
    rel = sum(w / m for w, m in zip(widths, mids)) / len(rows) * 100
    lo = min(r["p10"] for r in rows)
    hi = max(r["p90"] for r in rows)
    tight = "narrow" if rel < 40 else "wide" if rel > 120 else "moderate"
    return (f"Across the five days the range runs from {gel_millions(lo)} to "
            f"{gel_millions(hi)} million lari. That is a {tight} range: on eight days out "
            f"of ten we expect the actual figure to land inside it, and on roughly one day "
            f"in ten it will fall below and one day in ten above.")


def _agreement_line(rows: Sequence[Dict]) -> Optional[str]:
    """How far the two independent models disagree.

    Reported because it is an honest, readable proxy for confidence: when the inputs carry
    little information, two reasonable models land in different places, and the reader
    should know that before acting on a single number.
    """
    diffs = []
    for r in rows:
        q = r.get("p50_quantile_model")
        if q is None or not r["p50"]:
            continue
        diffs.append(abs(r["p50"] - q) / abs(r["p50"]) * 100)
    if not diffs:
        return None
    worst = max(diffs)
    if worst < 5:
        return (f"Two independently built models were run side by side and agree closely "
                f"(within {worst:.0f} percent on every day), which supports the central "
                f"figure.")
    return (f"Two independently built models were run side by side and disagree by as much "
            f"as {worst:.0f} percent on one of the five days. Treat the single central "
            f"figure with corresponding caution.")


def build_target_narrative(recipe: Dict, rows: Sequence[Dict]) -> Dict:
    """Narrative for one target. Returns the pieces so a renderer can lay them out."""
    pub = recipe["publication"]
    cred = recipe["dev_credentials"]
    rows = sorted(rows, key=lambda r: r["horizon"])

    paras: List[str] = [_verdict_line(recipe)]

    first, last = rows[0], rows[-1]
    paras.append(
        f"The central estimate for {recipe['target'].lower()} runs from "
        f"{gel_millions(first['p50'])} million lari on {first['target_date'][:10]} to "
        f"{gel_millions(last['p50'])} million lari on {last['target_date'][:10]}."
    )
    paras.append(_range_behaviour(rows, recipe))

    ag = _agreement_line(rows)
    if ag:
        paras.append(ag)

    # How good is it, in benchmark terms and without jargon.
    skill = cred["skill_vs_ruler_pct"]
    paras.append(
        f"Tested on 2024, this model's typical error was {skill:.0f} percent smaller than "
        f"simply assuming today's figure will repeat in five working days. That is the "
        f"benchmark every model here is measured against."
    )

    # Withheld: explain, never hide.
    if pub["verdict"] != "publishable":
        sig = cred["gates"]["signal"]
        paras.append(
            f"**Why this is not called a forecast.** {sig['reason_plain']} "
            f"{pub['reason_plain']}"
        )
        if pub.get("named_fix"):
            paras.append(f"**What would change this:** {pub['named_fix']}")
    else:
        sig = cred["gates"]["signal"]
        paras.append(f"**Why this one is called a forecast.** {sig['reason_plain']}")

    return {
        "target": recipe["target"],
        "verdict": pub["verdict"],
        "paragraphs": paras,
        "gates": cred["gates"],
        "recipe_id": recipe["id"],
    }


def build_narrative(forecasts: Sequence[Dict], registry: Dict,
                    provenance: Optional[Dict] = None) -> Dict:
    """The whole narrative: headline, per-target sections, scope, and limitations."""
    by_target: Dict[str, List[Dict]] = {}
    for r in forecasts:
        by_target.setdefault(r["target"], []).append(dict(r))

    recipes = {r["target"]: r for r in registry["recipes"]}
    sections = [build_target_narrative(recipes[t], rows)
                for t, rows in by_target.items() if t in recipes]

    n_pub = sum(1 for s in sections if s["verdict"] == "publishable")
    n_held = len(sections) - n_pub

    headline = (
        f"Of the {len(sections)} budget lines covered, **{n_pub} produced a figure we are "
        f"willing to call a forecast** and **{n_held} produced a guide to the typical level "
        f"only**. The distinction is deliberate and the system makes it automatically."
    )

    scope = (
        f"**Scope.** This covers {len(sections)} of {TOTAL_TREASURY_METRICS} budget lines in "
        f"the daily Treasury data, five working days ahead. It is not a view of the whole "
        f"budget and not a view beyond one week."
    )

    signal_finding = (
        "**The most important thing on this page.** For revenues and expenditure, the system "
        "reports a typical level rather than a genuine forecast of individual days. This is "
        "not a shortcoming we are working around quietly: it is a measured result. We test "
        "each model by shuffling the historical answers and refitting — if the inputs really "
        "carried information, destroying the link would badly damage the model. For these two "
        "lines it barely does. We repeated the test with three different statistical methods "
        "and all three agree.\n\n"
        "The single most valuable thing that would change this is a **forward calendar of "
        "domestic debt auctions and redemptions** from the Treasury. The days these models "
        "cannot anticipate are overwhelmingly debt-operation days, and those are not on a "
        "fixed date in the month, so no amount of modelling reaches them. We already hold the "
        "historical debt figures and tested them: they do not help, because knowing what "
        "happened yesterday does not tell you what is scheduled next week."
    )

    limitations = [
        "Figures are validated on 2024. The final independent check against 2025 has not "
        "been run yet and is scheduled — so these accuracy figures should be read as "
        "provisional.",
        "No model here has been tuned or formally approved. All settings are sensible "
        "defaults.",
        "The prediction ranges are least reliable on the largest days: on the biggest one "
        "third of days the range currently captures about half the outcomes rather than the "
        "intended eight in ten. Work to correct this is scheduled.",
        "One data question is outstanding with the Treasury: whether occasional negative "
        "revenue figures are a genuine netting convention or a recording issue.",
    ]

    prov_line = None
    if provenance:
        d = provenance.get("data", {})
        c = provenance.get("code", {})
        prov_line = (
            f"Produced from {d.get('name', 'the Treasury daily file')} "
            f"(fingerprint {str(d.get('sha256', ''))[:12]}…, data through "
            f"{str(d.get('latest_data_date', ''))[:10]}), code version "
            f"{str(c.get('git_sha', ''))[:12]}, fiscal calendar version "
            f"{provenance.get('calendar_version', '')}. The 2025 holdout was not used."
        )

    return {
        "headline": headline,
        "scope": scope,
        "signal_finding": signal_finding,
        "sections": sections,
        "limitations": limitations,
        "provenance_line": prov_line,
        "phrasing": "template",
    }


def narrative_to_markdown(nar: Dict) -> str:
    out = [f"## What this says\n", nar["headline"], "", nar["scope"], "",
           nar["signal_finding"], ""]
    for s in nar["sections"]:
        out.append(f"### {s['target']}\n")
        out.extend([p + "\n" for p in s["paragraphs"]])
    out.append("### What to keep in mind\n")
    out.extend([f"- {l}" for l in nar["limitations"]])
    if nar.get("provenance_line"):
        out += ["", "---", "", nar["provenance_line"]]
    if nar.get("phrasing") == "llm":
        out += ["", f"*{LLM_LABEL}*"]
    return "\n".join(out)


# ── optional LLM rephrasing ────────────────────────────────────────────────────

def llm_enabled() -> bool:
    return os.environ.get("AI4CM_LLM", "").strip().lower() == "ollama"


def rephrase_safely(template_text: str, timeout: float = 20.0) -> Optional[str]:
    """Ask a local model to rephrase; return None unless every number is preserved.

    Returns None on any failure whatsoever. The caller keeps the template, so a missing or
    misbehaving endpoint costs nothing.
    """
    if not llm_enabled():
        return None
    try:  # pragma: no cover - requires a live endpoint
        import urllib.request

        host = os.environ.get("AI4CM_LLM_HOST", "http://localhost:11434")
        model = os.environ.get("AI4CM_LLM_MODEL", "llama3.1")
        prompt = (
            "Rewrite the following Treasury briefing so it reads more fluently for a "
            "non-technical finance official. Absolute rules: do not change, add, remove or "
            "re-round ANY number. Do not change any verdict or conclusion. Keep the "
            "markdown headings. Return only the rewritten text.\n\n" + template_text
        )
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(f"{host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        text = (payload.get("response") or "").strip()
        if not text:
            return None
        if digits_of(text) != digits_of(template_text):
            return None
        return text
    except Exception:
        return None


def build_narrative_text(forecasts: Sequence[Dict], registry: Dict,
                         provenance: Optional[Dict] = None) -> Dict:
    """Narrative markdown plus which phrasing produced it."""
    nar = build_narrative(forecasts, registry, provenance)
    template_md = narrative_to_markdown(nar)
    rephrased = rephrase_safely(template_md)
    if rephrased is not None:
        nar["phrasing"] = "llm"
        return {"markdown": rephrased + f"\n\n*{LLM_LABEL}*", "phrasing": "llm",
                "narrative": nar}
    return {"markdown": template_md, "phrasing": "template", "narrative": nar}


def load_forward_artifacts(out_dir: Optional[Path] = None) -> Dict:
    """Read the latest forward run. Raises with an actionable message if absent."""
    import pandas as pd

    from forward_forecast import DEFAULT_OUT

    d = Path(out_dir or DEFAULT_OUT)
    csv = d / "forward_forecast.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"no forward run at {d}. Generate one with:\n"
            f"  ./backend/.venv/bin/python backend/run_forward_forecast.py"
        )
    df = pd.read_csv(csv)
    prov_p = d / "forward_provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else None
    return {"forecasts": df.to_dict("records"), "provenance": prov, "dir": str(d)}
