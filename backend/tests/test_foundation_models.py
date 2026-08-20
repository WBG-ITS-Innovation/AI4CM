"""Pretrained zero-shot forecasters: optional, pinned, and never publishable.

Three properties matter more than anything else here, and each is asserted rather than described.

**Optional.** The extras are not in the core requirements. With them absent, the app, the
registry and this whole suite must still work, and the models must report themselves as not
installed rather than raising. A fresh clone following the core README must be unaffected. That
is tested by simulating the absence rather than by trusting that the imports look lazy.

**Pinned.** Every checkpoint is pinned to an exact commit hash, never a tag and never ``main``. A
forecast whose weights can change underneath it is not a record of anything.

**Never publishable.** These carry no measured result and cannot become the model behind an
official forecast. The lock is structural and pre-existing, not something added for them, and the
tests below name the three independent mechanisms.
"""
from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import foundation_models as fm                                            # noqa: E402

#: True when the optional extras are actually installed in this interpreter.
EXTRAS_PRESENT = all(find_spec(spec.requires) is not None for spec in fm.MODELS)


# ---------------------------------------------------------------------------
# Optional: absent extras must degrade, never break
# ---------------------------------------------------------------------------

def test_the_module_imports_without_any_extra_installed():
    """Nothing heavy at module level, so the registry is readable anywhere.

    Checked on the source rather than by importing, because this test process may well have the
    extras installed and would then pass while a fresh clone failed.
    """
    header = fm.__file__ and Path(fm.__file__).read_text(encoding="utf-8").split("MODELS:")[0]
    for heavy in ("import torch", "import chronos", "import timesfm", "import transformers",
                  "from chronos", "from timesfm", "import numpy", "import pandas"):
        assert f"\n{heavy}" not in header, (
            f"{heavy!r} at module level would make this module unimportable without the extras, "
            f"and the registry unreadable from the Streamlit interpreter")


def test_availability_reports_rather_than_raises_when_a_package_is_absent(monkeypatch):
    """The absence is simulated, so this holds whether or not the extras are installed here."""
    monkeypatch.setattr(fm, "installed", lambda spec: False)
    table = fm.availability()

    assert table, "the registry must still enumerate its models with nothing installed"
    for name, info in table.items():
        assert info["installed"] is False
        assert info["reason"], f"{name} must say WHY it is unavailable"
        assert "requirements-foundation.txt" in info["reason"], (
            "the reason must name the file that fixes it")
        assert info["revision"], "the pin is a property of the registry, not of the install"


def test_a_forecast_with_the_package_absent_returns_a_reason_and_does_not_raise(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(fm, "installed", lambda spec: False)
    series = pd.Series(range(400), index=pd.bdate_range("2020-01-01", periods=400), dtype=float)

    result, why = fm.forecast(fm.names()[0], series, horizon=5)
    assert result is None
    assert why and "requirements-foundation.txt" in why


def test_an_unknown_model_name_is_refused_by_name():
    import pandas as pd

    series = pd.Series(range(100), index=pd.bdate_range("2020-01-01", periods=100), dtype=float)
    result, why = fm.forecast("NoSuchModel", series, horizon=5)
    assert result is None
    assert "NoSuchModel" in why


def test_the_extras_are_not_in_the_core_requirements():
    """The rail that keeps a fresh clone unaffected by this whole feature."""
    core = BACKEND / "requirements.txt"
    if not core.exists():
        pytest.skip("no core requirements file in this checkout")
    text = core.read_text(encoding="utf-8").lower()
    for package in ("chronos", "timesfm", "transformers", "accelerate"):
        assert package not in text, (
            f"{package} belongs in requirements-foundation.txt, not the core install")


def test_the_optional_requirements_file_exists_and_pins_every_version():
    optional = BACKEND / "requirements-foundation.txt"
    assert optional.exists(), "the optional extras need a file of their own"

    pins = [l.strip() for l in optional.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    assert pins, "the file must actually pin something"
    for line in pins:
        assert "==" in line, f"{line!r} is not an exact pin"
        assert not any(c in line for c in "<>~"), f"{line!r} is a range, not a pin"


# ---------------------------------------------------------------------------
# Pinned: an exact commit, never a moving reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", fm.MODELS, ids=lambda s: s.name)
def test_every_checkpoint_is_pinned_to_a_commit_hash(spec):
    """40 hex characters. A tag or a branch name can be repointed at different bytes."""
    assert len(spec.revision) == 40, f"{spec.name}: {spec.revision!r} is not a full commit hash"
    assert all(c in "0123456789abcdef" for c in spec.revision.lower()), spec.revision
    assert spec.revision not in ("main", "master"), "a branch is not a pin"


@pytest.mark.parametrize("spec", fm.MODELS, ids=lambda s: s.name)
def test_every_checkpoint_names_its_repo_and_package(spec):
    assert "/" in spec.repo, f"{spec.name}: {spec.repo!r} is not a Hugging Face repo id"
    assert spec.requires, "a model that names no package cannot be reported as missing"
    assert spec.kind in ("chronos", "timesfm"), spec.kind


@pytest.mark.parametrize("spec", fm.MODELS, ids=lambda s: s.name)
def test_every_summary_says_it_is_unmeasured(spec):
    """A reader skimming summaries must not have to cross-reference a badge to learn this."""
    assert "not yet measured" in spec.summary
    assert len(spec.summary.split()) >= 25, "one clause is not an explanation"


# ---------------------------------------------------------------------------
# Never publishable: the three independent locks
# ---------------------------------------------------------------------------

def test_no_foundation_model_is_in_the_champion_pool():
    """Lock 1, and the strongest: the pool is the machine-learning family alone.

    A registry recipe may only promote a `point_model` from `CHAMPION_POOL_CATEGORY`, and
    `composition()` fails outright if a promoted model falls outside it. F_FOUNDATION is a
    different category, so this holds by construction with nothing added to enforce it.
    """
    import model_reference as mr

    pool = mr.model_pool()
    comp = mr.composition(pool)

    foundation = {n for n, v in pool.items() if v["pipeline"] == fm.FAMILY_FOUNDATION}
    assert foundation, "the family should be enumerable, or these tests prove nothing"
    assert comp["champion_pool_category"] == "machine-learning models"
    assert not (foundation & set(comp["champion_pool"])), (
        f"a foundation model reached the champion pool: {foundation & set(comp['champion_pool'])}")


def test_no_foundation_model_counts_as_competing():
    """Not a lock, but the same honesty: they are not ranked against the measured models.

    They produce a point forecast, so they COULD be counted. They are not, because they were
    never put through this project's evaluation protocol, and adding them would inflate the
    competing figure with entries carrying no measurement.
    """
    import model_reference as mr

    pool = mr.model_pool()
    comp = mr.composition(pool)
    assert "pretrained zero-shot forecasters" not in mr.COMPETING_CATEGORIES
    assert comp["counts"]["pretrained zero-shot forecasters"] == len(fm.MODELS)


def test_every_foundation_model_is_untested_in_the_ledger():
    """Lock 2: status is derived from experiments/log.csv, never declared.

    An untested model has no measurement the publication gates can read and no run id a recipe
    could cite, which is lock 3.
    """
    from model_catalog import status_of, UNTESTED

    for spec in fm.MODELS:
        assert status_of(spec.name) == UNTESTED, (
            f"{spec.name} has a ledger row. That is not forbidden, but it means these models are "
            f"no longer exploratory-by-absence and the claim needs restating.")


def test_the_family_is_described_for_a_reader():
    """A new family that appears in a count but has no description is a number with no meaning."""
    import model_reference as mr

    for spec in fm.MODELS:
        assert spec.name in mr.DESCRIPTIONS, f"{spec.name} has no description"
        assert mr.DESCRIPTIONS[spec.name]["summary"]

    assert fm.FAMILY_FOUNDATION in mr._CATEGORY_BY_PIPELINE, (
        "client_category() raises on an unknown pipeline by design; the family must be mapped")
    assert mr._CATEGORY_BY_PIPELINE[fm.FAMILY_FOUNDATION] in mr.CATEGORY_ORDER


def test_the_runner_refuses_to_default_the_evaluation_window_open():
    """The bound is about which dates a RESULT may be measured on, not what a model trained on.

    A zero-shot model has trained on nothing here, which makes it tempting to treat the sealed
    window as harmless for it. It is not: a result scored over the holdout has spent the holdout.
    """
    source = (BACKEND / "run_foundation.py").read_text(encoding="utf-8")
    assert 'if not eval_end:' in source
    assert "refusing to run" in source
    assert "F_FOUNDATION" in (BACKEND.parent / "frontend" / "exploratory.py").read_text(
        encoding="utf-8"), "the Lab must apply its exploratory bound to this family too"


# ---------------------------------------------------------------------------
# Behaviour, when the extras are actually here
# ---------------------------------------------------------------------------

needs_extras = pytest.mark.skipif(
    not EXTRAS_PRESENT,
    reason="optional extras absent; install backend/requirements-foundation.txt")


def test_weekends_are_dropped_before_a_model_sees_the_series():
    """Measured, not stylistic. See the module docstring in foundation_models.

    The Treasury table carries a zero for every weekend. Handing those over spends the context
    window teaching a weekly zero pattern, and Chronos then forecast INTO it: two of five median
    steps came back negative. Business days only, like every other family here.
    """
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=28, freq="D")
    series = pd.Series(1.0, index=index)
    series[series.index.dayofweek >= 5] = 0.0

    kept = fm.business_days_only(series)
    assert len(kept) == 20
    assert (kept.index.dayofweek < 5).all()
    assert not (kept == 0).any(), "the weekend zeros are what this removes"


def test_a_short_series_is_refused_with_a_reason_rather_than_forecast():
    import pandas as pd

    series = pd.Series(range(10), index=pd.bdate_range("2024-01-01", periods=10), dtype=float)
    result, why = fm.forecast(fm.names()[0], series, horizon=5)
    assert result is None
    assert "too short" in why


@needs_extras
@pytest.mark.parametrize("name", fm.names())
def test_a_real_forecast_is_ordered_and_labelled(name):
    """p10 <= p50 <= p90 on every step, and the result carries what reproduces it."""
    import pandas as pd

    data = BACKEND / "data" / "processed" / "master_daily_clean_treasury.csv"
    if not data.exists():
        pytest.skip("no Treasury table in this checkout")
    frame = pd.read_csv(data)
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["Revenues"].astype(float).dropna()

    result, why = fm.forecast(name, series, horizon=5)
    assert result is not None, why

    for i in range(5):
        assert result["p10"][i] <= result["p50"][i] <= result["p90"][i], f"crossed at step {i}"
    assert result["zero_shot"] is True
    assert result["revision"] == fm.spec_for(name).revision, (
        "the result must record the exact weights that produced it")
    assert len(result["target_dates"]) == 5
    assert all(pd.Timestamp(d).dayofweek < 5 for d in result["target_dates"]), (
        "a forecast for a Saturday is not a forecast anybody asked for")


@needs_extras
def test_the_checkpoint_is_loaded_once_per_process_not_once_per_forecast():
    """The defect that made a backtest impossible, kept as a test.

    The first version loaded inside the predict call. Harmless for one forecast; for the 145-origin
    walk `run_foundation.py` does, it meant TimesFM's 925 MB checkpoint was read and recompiled at
    every origin, and the run did not finish inside ten minutes. With the load cached it takes
    11.3s.
    """
    import pandas as pd

    fm._loaded.cache_clear()
    series = pd.Series(range(400), index=pd.bdate_range("2020-01-01", periods=400), dtype=float)
    name = fm.names()[0]

    fm.forecast(name, series, horizon=3)
    after_first = fm._loaded.cache_info()
    fm.forecast(name, series, horizon=3)
    after_second = fm._loaded.cache_info()

    assert after_second.hits > after_first.hits, (
        "the second forecast reloaded the checkpoint instead of reusing it")
    assert after_second.misses == after_first.misses, "it loaded a second copy"
