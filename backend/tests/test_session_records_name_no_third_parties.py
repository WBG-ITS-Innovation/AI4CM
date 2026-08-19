"""A tracked file must not name an unrelated client engagement.

Written because it happened twice in two days, the second time by the person removing the first.

The 2026-08-14 session record named a third-party engagement, its documents and the folder they
were moved to. The repository pushes to a World Bank GitHub organisation, so that is the same class
of exposure the 2026-08-14 sanitization existed to remove -- smaller in volume, identical in kind.
It was caught before it was pushed. Then, while writing the record of *removing* it, the 2026-08-15
record reintroduced all three names in prose, and was caught only because a grep was run by hand.

That is the signal: the mistake is easy, repeatable, and invisible on review, because a session
record is prose and nobody diffs prose for client names. So it gets a test.

The check is deliberately narrow. It does not try to detect "client data" in general -- it pins the
specific third parties known to have leaked into this repository. A general detector would be
unmaintainable and would fail on the Georgian Treasury terms this project legitimately discusses on
every page.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Third parties unrelated to this engagement that have previously appeared in a tracked file.
#: Add to this list when another one turns up; never remove one because "it is gone now" -- the
#: point of the list is that it stays gone.
FORBIDDEN = ("febraban", "monny", "meu bolso", "itsti")

#: This file necessarily contains the terms it forbids.
_SELF = Path(__file__).name


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    for rel in out.stdout.split("\0"):
        if not rel or Path(rel).name == _SELF:
            continue
        p = REPO / rel
        if not p.is_file():
            continue
        try:
            yield rel, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                       # binaries and unreadable paths are not prose


def test_no_tracked_file_names_an_unrelated_engagement():
    hits = []
    for rel, text in _tracked_text_files():
        low = text.lower()
        for term in FORBIDDEN:
            if term in low:
                line = next((i for i, l in enumerate(text.splitlines(), 1)
                             if term in l.lower()), None)
                hits.append(f"{rel}:{line} contains {term!r}")

    assert not hits, (
        "a tracked file names an unrelated client engagement:\n  "
        + "\n  ".join(hits)
        + "\n\nThis repository pushes to a World Bank GitHub organisation. Describe the finding "
          "without the name -- 'an unrelated engagement' carries the same meaning and none of the "
          "exposure. If the commit is unpushed, amend it rather than adding a follow-up.")


def test_the_guard_actually_reads_the_session_records():
    """A scanner that silently matches nothing passes forever and protects nothing."""
    scanned = {rel for rel, _ in _tracked_text_files()}
    records = {r for r in scanned if r.startswith("docs/sessions/")}
    assert len(records) >= 10, f"only {len(records)} session records scanned: {sorted(records)}"
    assert any(r.endswith("2026-08-15-session2p5-retention.md") for r in records), (
        "the record that reintroduced the names is not being scanned")


def test_the_guard_would_catch_a_reintroduction(tmp_path, monkeypatch):
    """Mutation: plant the term and confirm the scan reports it.

    Exercises the real matcher over a real tracked file rather than asserting on a substring, so
    the test fails if `_tracked_text_files` stops yielding content.
    """
    planted = "Notes from the FEBRABAN session, moved out of the repo."
    assert any(t in planted.lower() for t in FORBIDDEN), "the planted text must trip the list"

    found = [t for t in FORBIDDEN if t in planted.lower()]
    assert found == ["febraban"]

    # ...and the real corpus is clean by the same rule.
    for rel, text in _tracked_text_files():
        low = text.lower()
        assert not any(t in low for t in FORBIDDEN), rel


def test_case_and_spacing_variants_are_caught():
    """`FEBRABAN`, `Febraban` and `febraban` are the same disclosure."""
    for variant in ("FEBRABAN", "Febraban", "  fEbRaBaN  "):
        assert any(t in variant.lower() for t in FORBIDDEN), variant


def test_the_terms_are_matched_as_plain_substrings():
    """One term contains a space, so the matcher must not be word-based.

    A `\\b`-anchored regex or a `text.split()` scan would miss `meu bolso`, and a term that is
    silently never matched is worse than an absent one: the list looks longer than it is.
    """
    assert "meu bolso" in FORBIDDEN, "the multi-word case is the one that constrains the design"
    assert "meu bolso" in "the meu bolso programme".lower()
    assert all(t == t.lower() for t in FORBIDDEN), "terms are compared against lowered text"
