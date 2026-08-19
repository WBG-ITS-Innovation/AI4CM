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
