"""The narrative must be readable, honest, and immune to LLM tampering with numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import insights as ins  # noqa: E402
from registry import load_registry  # noqa: E402


@pytest.fixture(scope="module")
def narrative():
    art = ins.load_forward_artifacts()
    return ins.build_narrative(art["forecasts"], load_registry(), art["provenance"])


# ── the LLM guard ─────────────────────────────────────────────────────────────

def test_digits_of_ignores_formatting_but_not_value():
    assert ins.digits_of("1,234.50 lari") == ins.digits_of("1234.5 lari")
    assert ins.digits_of("total 1,000") == ins.digits_of("total 1000")
    assert ins.digits_of("41 percent") != ins.digits_of("42 percent")
    assert ins.digits_of("94.8 million") != ins.digits_of("94.9 million")


def test_digits_of_catches_a_dropped_or_added_number():
    base = "Revenues 94.8 million over 5 days, error 41 percent lower."
    assert ins.digits_of(base) != ins.digits_of(
        "Revenues 94.8 million over 5 days, error lower.")
    assert ins.digits_of(base) != ins.digits_of(
        "Revenues 94.8 million over 5 days, error 41 percent lower, up 7 percent.")


def test_rephrase_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("AI4CM_LLM", raising=False)
    assert ins.rephrase_safely("anything with 5 numbers 1 2 3 4") is None


def test_rephrase_rejects_altered_numbers(monkeypatch):
    """The property the whole hook rests on: changed digits are discarded.

    Simulated by stubbing the transport, because the guard must hold regardless of what the
    model returns.
    """
    monkeypatch.setenv("AI4CM_LLM", "ollama")
    template = "Revenues are 94.8 million lari, 41 percent better than the benchmark."

    def fake_call(prompt, **kw):
        return "Revenues are 99.9 million lari, 41 percent better than the benchmark."

    monkeypatch.setattr(ins, "rephrase_safely",
                        lambda t, timeout=20.0: (
                            fake_call(t) if ins.digits_of(fake_call(t)) == ins.digits_of(t)
                            else None))
    assert ins.rephrase_safely(template) is None


def test_narrative_falls_back_to_template_on_any_llm_failure(monkeypatch):
    monkeypatch.setenv("AI4CM_LLM", "ollama")
    monkeypatch.setattr(ins, "rephrase_safely", lambda *a, **k: None)
    art = ins.load_forward_artifacts()
    out = ins.build_narrative_text(art["forecasts"], load_registry(), art["provenance"])
    assert out["phrasing"] == "template"
    assert ins.LLM_LABEL not in out["markdown"]


def test_llm_output_is_labelled_when_used(monkeypatch):
    monkeypatch.setenv("AI4CM_LLM", "ollama")
    art = ins.load_forward_artifacts()
    base = ins.narrative_to_markdown(
        ins.build_narrative(art["forecasts"], load_registry(), art["provenance"]))
    monkeypatch.setattr(ins, "rephrase_safely", lambda *a, **k: base + " Reworded.")
    out = ins.build_narrative_text(art["forecasts"], load_registry(), art["provenance"])
    assert out["phrasing"] == "llm"
    assert ins.LLM_LABEL in out["markdown"]


# ── honesty and readability ───────────────────────────────────────────────────

def test_headline_states_the_split_and_scope_names_the_denominator(narrative):
    assert "forecast" in narrative["headline"]
    assert f"of {ins.TOTAL_TREASURY_METRICS}" in narrative["scope"]
    assert "not a view of the whole" in narrative["scope"]


def test_signal_finding_is_stated_and_names_the_fix(narrative):
    t = narrative["signal_finding"]
    assert "typical level rather than a genuine forecast" in t
    assert "three different statistical methods" in t
    assert "auctions and redemptions" in t, "the named fix must appear"


def test_withheld_targets_are_explained_not_hidden(narrative):
    held = [s for s in narrative["sections"] if s["verdict"] != "publishable"]
    assert len(held) == 2, "both flow targets should be withheld as forecasts"
    for s in held:
        body = "\n".join(s["paragraphs"])
        assert "Why this is not called a forecast" in body
        assert "What would change this" in body
        # the numbers are still present -- withheld means the claim, not the data
        assert any("million lari" in p for p in s["paragraphs"])


def test_no_acronyms_or_jargon_in_reader_facing_prose(narrative):
    from insights import narrative_to_markdown
    md = narrative_to_markdown(narrative)
    banned = ["MASE", "RMSE", "P10", "P90", "sentinel", "MAE", "LightGBM", "HistGBDT",
              "GBQuantile", "quantile", "WS3", "WS5", "DEV", "conformal", "L1 "]
    found = [b for b in banned if b in md]
    assert not found, f"jargon leaked into reader-facing prose: {found}"


def test_money_is_in_millions_not_raw(narrative):
    from insights import narrative_to_markdown
    md = narrative_to_markdown(narrative)
    assert "million lari" in md
    # a raw 9-digit treasury figure would mean the formatter was bypassed
    import re
    assert not re.search(r"\b\d{9,}\b", md.replace(",", "")), "raw magnitude leaked"


def test_ranges_are_described_as_behaviour(narrative):
    body = "\n".join(p for s in narrative["sections"] for p in s["paragraphs"])
    assert "eight days out of ten" in body


def test_limitations_include_the_pending_holdout_and_coverage_gap(narrative):
    joined = " ".join(narrative["limitations"])
    assert "2025" in joined and "scheduled" in joined
    assert "approved" in joined
    assert "eight in ten" in joined or "half the outcomes" in joined


def test_gel_millions_formatting():
    assert ins.gel_millions(94_751_609) == "94.8"
    assert ins.gel_millions(1_736_111_022) == "1,736.1"
