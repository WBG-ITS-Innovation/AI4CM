"""The Forecast page's reading tab labels every line with the source and data date it shows.

Why this file exists
--------------------
The tab used to read one directory and say nothing about where its numbers came from, so a
reader could not tell a fresh page-launched run from the runner's artifact of weeks before
(inference-horizon map, §1.3). It now shows the newest artifact per target and says, under
each line's heading, which artifact that was and what data it was built from. The test renders
the real page against whatever artifacts this machine holds, and skips when it holds none,
because a fresh clone has neither a forward run nor a published issue.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
REPO = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, str(REPO / "backend"))

pytest.importorskip("streamlit", reason="streamlit is installed in frontend/.venv only")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = FRONTEND / "pages" / "07_Forecast.py"
SOURCE_LINE = re.compile(r"Source: (forward run|published issue \S+), generated \d{4}-\d{2}-\d{2}, "
                         r"data through \d{4}-\d{2}-\d{2}\.")


@pytest.fixture(scope="module")
def rendered():
    import insights as ins

    try:
        art = ins.load_newest_forecasts()
    except (FileNotFoundError, AttributeError) as exc:
        pytest.skip(f"no forecast artifact on this machine: {exc}")
    at = AppTest.from_file(str(PAGE), default_timeout=240).run()
    if at.exception:
        pytest.fail("\n".join(str(e.value) for e in at.exception))
    return at, art


def _captions(at) -> list:
    return [str(c.value) for c in at.caption]


def test_every_line_shown_says_its_source_and_data_date(rendered):
    at, art = rendered
    lines = [c for c in _captions(at) if SOURCE_LINE.search(c)]
    assert len(lines) >= len(art["sources"]), (
        f"{len(art['sources'])} line(s) are shown but only {len(lines)} source caption(s) "
        f"were rendered: {lines}")


def test_the_source_captions_match_the_loaders_choice(rendered):
    at, art = rendered
    text = "\n".join(_captions(at))
    for target, src in art["sources"].items():
        expected = (f"Source: {src['label']}, generated {src['generated_at_utc'][:10]}, "
                    f"data through {src['data_through']}.")
        assert expected in text, f"{target}: expected caption {expected!r} not rendered"


def test_the_header_says_where_the_numbers_come_from(rendered):
    at, _ = rendered
    text = "\n".join(_captions(at))
    assert "newest artifact available for each line" in text
