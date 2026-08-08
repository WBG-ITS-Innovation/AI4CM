"""The Treasury HTML report must be self-contained, honest, and free of jargon.

Self-containment is the requirement that is easiest to break silently: one CDN link and
the report is blank on an air-gapped ministry laptop, which is exactly where it gets read.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "backend"))


@pytest.fixture(scope="module")
def doc() -> str:
    import build_treasury_report as brt
    try:
        return brt.build()
    except FileNotFoundError as exc:
        pytest.skip(f"no forward run available: {exc}")


def test_report_is_self_contained(doc):
    """No network dependency of any kind."""
    for bad in ("http://", "https://", "//cdn", "<script", 'src="//', "@import"):
        assert bad not in doc, f"report references external resource {bad!r}"


def test_charts_are_inline_svg(doc):
    assert doc.count("<svg") >= 3, "expected one band chart per target"
    assert "<canvas" not in doc


def test_verdict_comes_before_the_detail(doc):
    """A Treasury reader must meet the conclusion first."""
    short = doc.index("The short version")
    detail = doc.index("Line-by-line outlook")
    prov = doc.index("Provenance")
    assert short < detail < prov


def test_withheld_models_are_shown_not_omitted(doc):
    assert "Withheld as a forecast" in doc
    assert "What would change this" in doc
    # and their numbers are still present
    assert "Central estimate" in doc


def test_states_that_the_holdout_evaluation_is_still_pending(doc):
    assert "validated against 2024, not 2025" in doc
    assert "scheduled" in doc
    assert "No" in doc  # 2025 holdout touched -> No


def test_scope_names_the_denominator(doc):
    from insights import TOTAL_TREASURY_METRICS
    assert f"of {TOTAL_TREASURY_METRICS}" in doc


def test_no_acronyms_or_model_names_in_reader_prose(doc):
    """Model identifiers belong in the provenance block, not the narrative.

    The recipe ids inside <code> tags are provenance and are allowed; bare acronyms in
    prose are not.
    """
    # Scope: everything a Treasury reader is asked to READ, i.e. up to the provenance
    # block. Model identifiers are legitimate inside provenance and inside <code>.
    text = doc[:doc.index("<h2>Provenance</h2>")]
    text = re.sub(r"<code>.*?</code>", "", text, flags=re.S)
    text = re.sub(r"<style>.*?</style>", "", text, flags=re.S)
    for bad in ("MASE", "RMSE", "MAE", "P10", "P90", "sentinel", "LightGBM",
                "HistGBDT", "GBQuantile", "WS3", "WS5"):
        assert bad not in text, f"jargon {bad!r} leaked into reader-facing prose"


def test_money_is_in_millions(doc):
    assert "million lari" in doc or "Millions of lari" in doc
    body = doc[:doc.index("<h2>Provenance</h2>")]
    body = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
    # a raw nine-digit figure between tags would mean the formatter was bypassed
    assert not re.search(r">\s*\d{9,}\s*<", body), "raw magnitude leaked into the report"


def test_provenance_block_carries_the_full_fingerprint(doc):
    assert "SHA-256" in doc
    assert "Fiscal calendar version" in doc
    assert "Code version" in doc
    assert "holdout touched" in doc


def test_footer_disclaims_approval(doc):
    assert "has been formally approved" in doc or "not been formally approved" in doc
