"""Where the lab looks for run artifacts.

One resolver, honouring ``AI4CM_RUNS_DIR``. Two reasons this exists rather than each page
hard-coding ``APPROOT / "runs"``:

* **Testability.** The page smoke tests need to render each page against an empty directory,
  a single ordinary run, and a withheld run. With a hard-coded path all three "states"
  silently read the developer's real runs folder, so the tests passed while testing nothing.
  That is exactly what happened before this module existed.
* **Deployability.** A server that mounts artifacts elsewhere should not need a code change.
"""
from __future__ import annotations

import os
from pathlib import Path

APPROOT = Path(__file__).resolve().parent
ENV_VAR = "AI4CM_RUNS_DIR"


def runs_dir() -> Path:
    """The run-artifacts root. ``AI4CM_RUNS_DIR`` overrides the default when set."""
    override = os.environ.get(ENV_VAR, "").strip()
    return Path(override).expanduser() if override else APPROOT / "runs"


#: Overrides the scorecard the Scorecard page READS.
#:
#: Only the read path. Scoring always writes the real ``forecasts/scorecard.csv``, because a
#: scoring run that could be redirected by an environment variable would let a test, or a
#: mistyped shell, quietly replace the track record.
#:
#: This exists for one reason: the honest state of this project today is zero scored rows, and
#: a page that has only ever been seen empty is a page whose scored branch has never rendered.
#: The override lets that branch be rendered against a clearly-labelled synthetic file, and the
#: page says on screen that it is reading one.
SCORECARD_ENV_VAR = "AI4CM_SCORECARD"


def scorecard_path() -> Path:
    """The scorecard to display. ``AI4CM_SCORECARD`` overrides the default when set."""
    override = os.environ.get(SCORECARD_ENV_VAR, "").strip()
    return Path(override).expanduser() if override else APPROOT.parent / "forecasts" / "scorecard.csv"


def scorecard_is_overridden() -> bool:
    """True when the page is reading a substituted scorecard rather than the real one."""
    return bool(os.environ.get(SCORECARD_ENV_VAR, "").strip())
