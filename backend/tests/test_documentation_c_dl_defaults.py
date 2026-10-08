"""The Documentation page's "Defaults by cadence" table for C_DL states the code's defaults.

Why this file exists
--------------------
``frontend/pages/09_Documentation.py`` carries a hand-typed table of C_DL defaults. Its daily
row said horizon 14; neither the pipeline (``ConfigDL.horizons_daily = [1, 5, 20]``) nor the
runner (``TG_HORIZON`` fallback 5) says that (inference-horizon map, §2.5). Measured while
writing this: the other three columns were wrong too, in every row. A table of defaults that
disagrees with the defaults is the kind of documentation a reader trusts most and should least.

Both sides are read as source. The page is not rendered, so no Streamlit is needed, and the
pipeline is not imported, so no torch is loaded: ``c_dl_pipeline`` pulls in torch at import,
and loading it beside LightGBM in one test process is where this machine's intermittent
segmentation fault appears. Reading the dataclass defaults from the file is exact and cheap.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from forecast_modes import VALIDATED_HORIZON  # noqa: E402

PAGE = REPO / "frontend" / "pages" / "09_Documentation.py"
PIPELINE = BACKEND / "c_dl_pipeline.py"
RUNNER = BACKEND / "run_c_dl_univariate.py"


def _page_defaults_c() -> dict:
    """The dict ``defaults_c()`` returns, read from the source without executing the page."""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "defaults_c")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    return ast.literal_eval(ret.value)


def _config_dl_defaults() -> dict:
    """``ConfigDL``'s field defaults and the lists ``__post_init__`` fills in, from the source."""
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ConfigDL")
    out: dict = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    for node in ast.walk(post):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and t.attr.startswith("horizons_"):
                out[t.attr] = ast.literal_eval(node.value)
    for key in ("seq_len_daily", "seq_len_weekly", "seq_len_monthly", "epochs", "batch_size",
                "horizons_daily", "horizons_weekly", "horizons_monthly"):
        assert key in out, f"ConfigDL.{key} not found in {PIPELINE.name}; the parser needs updating"
    return out


def _code_defaults() -> dict:
    """What the pipeline and the runner actually default to, per cadence.

    Lookback, epochs and batch size are ``ConfigDL``'s own defaults, and the runner's
    ``ov.get(...)`` fallbacks repeat the same numbers. The horizon is the runner's: it always
    replaces the pipeline's list with ``[TG_HORIZON]`` for the active cadence, and its fallback
    is the validated horizon.
    """
    cfg = _config_dl_defaults()
    return {
        "daily": {"lookback": cfg["seq_len_daily"], "horizon": VALIDATED_HORIZON,
                  "epochs": cfg["epochs"], "batch_size": cfg["batch_size"]},
        "weekly": {"lookback": cfg["seq_len_weekly"], "horizon": VALIDATED_HORIZON,
                   "epochs": cfg["epochs"], "batch_size": cfg["batch_size"]},
        "monthly": {"lookback": cfg["seq_len_monthly"], "horizon": VALIDATED_HORIZON,
                    "epochs": cfg["epochs"], "batch_size": cfg["batch_size"]},
    }


def test_the_runner_repeats_the_pipelines_defaults():
    """The table quotes one set of numbers; this is what makes 'one set' true."""
    cfg = _config_dl_defaults()
    src = RUNNER.read_text(encoding="utf-8")
    assert f'ov.get("lookback", {cfg["seq_len_daily"]})' in src
    assert f'ov.get("lookback", {cfg["seq_len_weekly"]})' in src
    assert f'ov.get("lookback", {cfg["seq_len_monthly"]})' in src
    assert f'ov.get("max_epochs", {cfg["epochs"]})' in src
    assert f'ov.get("batch_size", {cfg["batch_size"]})' in src
    assert f'int(env["TG_HORIZON"] or {VALIDATED_HORIZON})' in src


def test_the_daily_horizon_row_is_the_validated_horizon():
    assert _page_defaults_c()["daily"]["horizon"] == VALIDATED_HORIZON


@pytest.mark.parametrize("cadence", ("daily", "weekly", "monthly"))
def test_every_cell_of_the_defaults_table_matches_the_code(cadence):
    page, code = _page_defaults_c()[cadence], _code_defaults()[cadence]
    assert page == code, f"{cadence}: page says {page}, the code defaults to {code}"


def test_the_pipelines_own_horizon_lists_are_stated_beside_the_table():
    """The runner always passes one horizon, so the lists are reached only when nothing is
    passed at all. A reader should still be told they exist, in the caption, not the cells."""
    src = PAGE.read_text(encoding="utf-8")
    cfg = _config_dl_defaults()
    for lst in (cfg["horizons_daily"], cfg["horizons_weekly"], cfg["horizons_monthly"]):
        assert ", ".join(str(h) for h in lst) in src, f"the page does not state {lst}"
