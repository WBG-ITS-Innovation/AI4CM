"""The README is a deliverable, so its claims are checked rather than reviewed.

Three kinds of claim are checked here, and each of them was wrong at some point while the
document was being written:

* **Commands.** Every command a reader is told to type either runs, or names a file that
  exists. The step that writes ``frontend/.tg_paths.json`` used ``Path.resolve()`` in its
  first draft, and inside a virtual environment ``bin/python`` is a symlink to the
  interpreter it was built from, so it recorded the system Python instead of the
  environment holding the packages. Running it is how that was found.
* **Definitions.** The house style says a term is defined once and then reused everywhere.
  A README that paraphrases the app's glossary breaks that quietly, because both texts
  read fine on their own.
* **Figures.** Row counts, test counts and file paths go stale silently.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
README = REPO / "README.md"
sys.path.insert(0, str(FRONTEND))

pytest.importorskip("streamlit", reason="ui_styles imports streamlit")
from ui_styles import GLOSSARY  # noqa: E402

BODY = README.read_text(encoding="utf-8")
FLAT = " ".join(BODY.split())


def _code_blocks() -> list[str]:
    return re.findall(r"```\n(.*?)```", BODY, re.S)


# ── the definitions are the app's, word for word ──────────────────────────────

@pytest.mark.parametrize("term", sorted(GLOSSARY), ids=lambda t: t)
def test_the_readme_uses_the_app_wording_for_every_term(term):
    """Not a paraphrase. Same sentence, so a reader meets one definition of each word."""
    definition = " ".join(GLOSSARY[term].split())
    assert definition in FLAT, (
        f"the README's {term!r} is not the app's wording.\n  app: {definition[:120]}")


# ── every path the README names exists ────────────────────────────────────────

@pytest.mark.parametrize("relative", [
    "backend/requirements.txt",
    "backend/requirements-foundation.txt",
    "frontend/requirements.txt",
    "frontend/Overview.py",
    "frontend/pages",
    "pytest.ini",
    "docs/ADDING_A_MODEL.md",
    "docs/REFRESH_AND_RETRAIN.md",
    "docs/DATA_SEMANTICS.md",
    "docs/AGENT_ARTIFACT_CONTRACT.md",
    "docs/SIGNAL_FINDING.md",
    "docs/sessions",
    "scripts/setup_unix.sh",
    "scripts/setup_windows.bat",
    "forecasts/scorecard.csv",
    "registry/recipes.json",
    "experiments/log.csv",
])
def test_paths_the_readme_names_exist(relative):
    assert relative in BODY, f"fixture drift: the README no longer names {relative}"
    assert (REPO / relative).exists(), f"the README points at {relative}, which is missing"


# ── the two-venv claim is the one people get wrong, so it is asserted ──────────

def test_streamlit_is_only_in_the_frontend_environment():
    """The README's central setup claim, and the cause of its commonest error."""
    frontend_py = FRONTEND / ".venv" / "bin" / "python"
    backend_py = REPO / "backend" / ".venv" / "bin" / "python"
    if not (frontend_py.exists() and backend_py.exists()):
        pytest.skip("both environments must exist to compare them")

    import subprocess

    def has(interpreter, module):
        return subprocess.run([str(interpreter), "-c", f"import {module}"],
                              capture_output=True).returncode == 0

    assert has(frontend_py, "streamlit"), "streamlit is missing from frontend/.venv"
    assert not has(backend_py, "streamlit"), (
        "streamlit is in backend/.venv; the README says it is not, and the two-venv "
        "explanation rests on that")
    assert has(backend_py, "sklearn"), "the modelling stack is missing from backend/.venv"
    assert not has(frontend_py, "sklearn"), (
        "the modelling stack is in frontend/.venv; the README says it is not")


def test_the_paths_command_records_the_environment_and_not_the_system_python():
    """resolve() follows the venv symlink to the base interpreter. absolute() does not."""
    command = next((b for b in _code_blocks() if "tg_paths.json" in b), "")
    assert command, "the README no longer shows the step that writes .tg_paths.json"
    assert ".absolute()" in command
    assert ".resolve()" not in command, (
        "resolve() follows bin/python to the interpreter the environment was built from, "
        "so it records the system Python rather than the environment with the packages")


# ── figures ───────────────────────────────────────────────────────────────────

def test_the_page_map_lists_every_page_in_nav_order():
    names = ["Overview", "Start here", "Data Preprocessing", "Lab", "Dashboard",
             "Compare", "History", "Forecast", "Scorecard", "Documentation"]
    # Split on a real horizontal rule, not on "---": the table's own separator row is
    # |---|---| and splitting on that cut the table off at its first line.
    table = BODY.split("## The pages, in the order the sidebar shows them")[1]
    table = table.split("\n---\n")[0]
    positions = []
    for name in names:
        marker = f"**{name}**"
        assert marker in table, f"the page map omits {name}"
        positions.append(table.index(marker))
    assert positions == sorted(positions), "the page map is not in sidebar order"

    pages = sorted((FRONTEND / "pages").glob("*.py"))
    assert len(pages) + 1 == len(names), (
        f"{len(pages)} pages plus Overview against {len(names)} rows in the map")


def test_the_expected_test_counts_are_stated_for_both_commands():
    """Both, and labelled, because the two numbers are not comparable."""
    assert "./backend/.venv/bin/python -m pytest -q" in BODY
    assert "./frontend/.venv/bin/python -m pytest frontend/tests -q" in BODY
    counts = re.findall(r"Expect ([\d,]+) passed, (\d+) skipped", BODY)
    assert len(counts) == 2, f"expected two stated counts, found {counts}"
    assert BODY.count("pytest.ini") >= 1, (
        "the README must explain why the root command collects both suites")


def test_the_data_file_figures_match_the_file_when_it_is_present():
    data = REPO / "backend" / "data" / "processed" / "master_daily_clean_treasury.csv"
    if not data.exists():
        pytest.skip("the Treasury data file is not on this machine")
    import pandas as pd

    frame = pd.read_csv(data)
    lines = [c for c in frame.columns if c not in ("date", "is_weekend", "is_holiday")]
    assert f"{len(frame):,}" in BODY, f"the README's row count is not {len(frame):,}"
    assert str(len(lines)) in BODY, f"the README's line count is not {len(lines)}"
    assert str(frame["date"].min()) in BODY and str(frame["date"].max()) in BODY


def test_the_readme_says_plainly_that_nothing_auto_publishes():
    """The first thing a Treasury reader needs to know, and it must not be buried."""
    opening = BODY[:2000]
    assert "Nothing here publishes to the Treasury" in opening
    assert "No figure leaves this machine" in opening


# ── house style ───────────────────────────────────────────────────────────────

def test_no_em_or_en_dashes():
    offenders = [(i, l.strip()[:70]) for i, l in enumerate(BODY.splitlines(), 1)
                 if "—" in l or "–" in l]
    assert not offenders, f"em or en dash in the README: {offenders}"


def test_no_filler_words():
    banned = ("seamlessly", "leverage", "robust", "delve", "empower", "cutting-edge",
              "state-of-the-art", "unlock", "elevate", "harness", "utilise", "utilize")
    lowered = BODY.lower()
    found = [w for w in banned if w in lowered]
    assert not found, f"filler words in the README: {found}"


def test_no_double_hyphen_as_punctuation_outside_code():
    offenders, in_code = [], False
    for i, line in enumerate(BODY.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if re.search(r"(?<![\w-])--(?![\w-])", line):
            offenders.append((i, line.strip()[:70]))
    assert not offenders, f"double hyphen as punctuation: {offenders}"
