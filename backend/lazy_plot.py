"""Lazy matplotlib.pyplot, so importing a forecasting pipeline does not require a plotting stack.

`backend/b_ml_pipeline.py` imported `matplotlib.pyplot` at module level. That made pyplot a hard
import-time dependency of the *forecasting* code path, and the Streamlit frontend — whose venv has
no matplotlib, because it renders with Plotly — crashed with ModuleNotFoundError the moment it
imported the pipeline to run a forecast.

Installing matplotlib into the frontend venv would have hidden the problem rather than fixed it:
the frontend does not plot with matplotlib, and a forecast has no business requiring a plotting
library to be importable. The dependency is real but it is only needed by the chart-writing
functions, so it is deferred to first use.

`plt` below is a proxy. Attribute access imports pyplot on demand and raises a message naming the
cause if it is genuinely absent, so a missing plotting stack fails where plotting is attempted
rather than at import.
"""
from __future__ import annotations

from typing import Any


class _LazyPyplot:
    """Imports ``matplotlib.pyplot`` on first attribute access, with a headless backend."""

    _mod: Any = None

    def _load(self):
        if self._mod is None:
            try:
                import matplotlib
                # Agg: these are file-writing charts on a server with no display. Selecting it
                # before pyplot loads avoids a backend that needs a GUI.
                matplotlib.use("Agg", force=False)
                import matplotlib.pyplot as _plt
            except Exception as exc:  # pragma: no cover - depends on the environment
                raise ModuleNotFoundError(
                    "matplotlib is required to write the diagnostic charts, but it is not "
                    "installed in this interpreter. Forecasting and evaluation do not need it — "
                    "only the chart-writing steps do. Install matplotlib, or call the pipeline "
                    "with plotting disabled."
                ) from exc
            type(self)._mod = _plt
        return self._mod

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


plt = _LazyPyplot()
