"""Every runner must be able to be told where its evaluation ends.

Why this is a test and not a docstring
--------------------------------------
Before this, ``eval_end`` reached the pipelines unevenly:

* B_ML had it and used it.
* E_QUANTILE's ``Config`` had it and both runners silently ignored it, so a caller could
  set it and be no better off.
* C_DL did not have it at all, and its runners *default* ``eval_start`` to the holdout.
* A_STAT had neither, and folded over every full year in the file.

The result was that "evaluate on train and dev only" was expressible in exactly one of
four families, and the Lab consequently expressed it in none. Each test below fails if a
family loses the ability to be bounded, which is the property the exploratory path rests
on -- ``frontend/tests/test_exploratory_window.py`` asserts the consequence, and this
asserts the mechanism.

None of this changes an unbounded run. Every bound defaults to ``None``, and the tests
that assert the old behaviour still holds are here for that reason.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from evaluation_windows import DEV, window_for  # noqa: E402

IDX = pd.bdate_range("2015-01-05", "2025-08-06")
BOUND = DEV.end


def _windows(folds) -> set:
    names = set()
    for _train_end, test_start, test_end in folds:
        for d in IDX[(IDX >= test_start) & (IDX <= test_end)]:
            names.add(window_for(d))
    return names


# ---------------------------------------------------------------------------
# A_STAT
# ---------------------------------------------------------------------------

def test_a_stat_yearly_folds_accept_bounds():
    import run_a_stat

    bounded = run_a_stat._yearly_folds(IDX, 4, None, eval_end=BOUND)
    assert bounded
    assert _windows(bounded) <= {"train", "dev"}


def test_a_stat_yearly_folds_unbounded_are_unchanged():
    """The reporting run this module was written for must be byte-identical."""
    import run_a_stat

    folds = run_a_stat._yearly_folds(IDX, 4, 3)
    assert len(folds) == 3
    assert folds[-1][2] == IDX[-1], "the last unbounded fold still ends at the last row"


def test_a_stat_yearly_folds_trim_a_straddling_block():
    import run_a_stat

    folds = run_a_stat._yearly_folds(IDX, 4, None, eval_start="2024-06-03", eval_end=BOUND)
    assert folds[-1][1] == pd.Timestamp("2024-06-03")
    assert folds[-1][2] <= pd.Timestamp(BOUND)


def test_a_stat_fallback_fold_cannot_reach_past_the_bound():
    import run_a_stat

    assert _windows(run_a_stat._fallback_fold(IDX, 6, eval_end=BOUND)) <= {"train", "dev"}


def test_a_stat_fallback_fold_returns_nothing_when_the_bound_leaves_nothing():
    import run_a_stat

    assert run_a_stat._fallback_fold(IDX, 6, eval_end="2015-01-06") == []


# ---------------------------------------------------------------------------
# C_DL
# ---------------------------------------------------------------------------

def test_c_dl_config_carries_an_eval_end():
    pytest.importorskip("torch", reason="c_dl_pipeline imports torch")
    from c_dl_pipeline import ConfigDL

    assert "eval_end" in ConfigDL.__dataclass_fields__
    assert ConfigDL.__dataclass_fields__["eval_end"].default is None


def test_c_dl_folds_accept_a_ceiling():
    pytest.importorskip("torch", reason="c_dl_pipeline imports torch")
    from c_dl_pipeline import build_yearly_folds

    bounded = build_yearly_folds(IDX, 4, eval_start=None, eval_end=BOUND)
    assert bounded
    assert _windows(bounded) <= {"train", "dev"}


def test_c_dl_folds_unbounded_are_unchanged():
    pytest.importorskip("torch", reason="c_dl_pipeline imports torch")
    from c_dl_pipeline import build_yearly_folds

    assert build_yearly_folds(IDX, 4) == build_yearly_folds(IDX, 4, eval_end=None)


def test_c_dl_ceiling_drops_a_block_that_starts_past_it():
    pytest.importorskip("torch", reason="c_dl_pipeline imports torch")
    from c_dl_pipeline import build_yearly_folds

    for _train_end, test_start, _test_end in build_yearly_folds(IDX, 4, eval_end=BOUND):
        assert test_start <= pd.Timestamp(BOUND)


# ---------------------------------------------------------------------------
# The runners themselves: a config field nobody reads is not a capability
# ---------------------------------------------------------------------------

RUNNERS_READING_EVAL_END = (
    "run_b_ml_univariate.py",
    "run_b_ml_multivariate.py",
    "run_c_dl_univariate.py",
    "run_c_dl_multivariate.py",
    "run_e_quantile_daily_univariate.py",
    "run_e_quantile_daily_multivariate.py",
    "run_a_stat.py",
)


@pytest.mark.parametrize("name", RUNNERS_READING_EVAL_END)
def test_runner_reads_eval_end_from_overrides(name):
    """E_QUANTILE's config had ``eval_end`` for months while its runners ignored it.

    That is the failure mode this checks for: a bound that exists in the dataclass, is
    honoured by the fold builder, and is never actually passed from the caller.
    """
    src = (BACKEND_DIR / name).read_text(encoding="utf-8")
    if name.startswith("run_b_ml"):
        # B_ML applies every override by setattr onto the config, so it needs no literal.
        assert "setattr(cfg, k, v)" in src
        return
    assert "eval_end" in src, f"{name} cannot be told where its evaluation ends"


@pytest.mark.parametrize("name", RUNNERS_READING_EVAL_END)
def test_runner_is_syntactically_valid(name):
    ast.parse((BACKEND_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# One error-report shape across all four families
# ---------------------------------------------------------------------------

ALL_RUNNERS = RUNNERS_READING_EVAL_END


@pytest.mark.parametrize("name", ALL_RUNNERS)
def test_every_runner_writes_a_failure_report(name):
    """The Lab can only explain a failure it can read.

    Only B_ML wrote ``artifacts/error.json``, so a failed A_STAT, C_DL or E_QUANTILE run
    left nothing but a log tail and the UI could do no better than print the traceback.
    """
    src = (BACKEND_DIR / name).read_text(encoding="utf-8")
    assert "write_error_report" in src, f"{name} fails silently as far as the UI is concerned"


def test_write_error_report_records_type_message_and_traceback(tmp_path):
    from runner_errors import error_report_path, write_error_report

    try:
        raise ValueError("window 'train' has 3 rows: too few for 5 folds")
    except ValueError as exc:
        path = write_error_report(tmp_path, exc, context="unit test")

    assert path == error_report_path(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["error_type"] == "ValueError"
    assert "too few for 5 folds" in payload["error"]
    assert "Traceback" in payload["traceback"]
    assert payload["context"] == "unit test"


def test_write_error_report_never_raises_on_an_unwritable_root(tmp_path):
    """A failure while reporting a failure would replace one bug with a worse one."""
    blocked = tmp_path / "a-file-not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    from runner_errors import write_error_report

    write_error_report(blocked, RuntimeError("boom"))   # must not raise
