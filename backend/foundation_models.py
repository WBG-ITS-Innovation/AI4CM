"""Pretrained zero-shot forecasters, registered as exploratory candidates and nothing more.

What these are
--------------
A foundation forecaster is a model trained once, by somebody else, on a large collection of other
people's time series, and then asked to forecast *this* series without ever being fitted to it.
There is no training step here. The series is handed over as context and the model answers.

Which makes them a genuinely different kind of thing from everything else on this shelf, and the
reason they get their own family tag rather than being folded into C_DL: every C_DL model was
trained on Treasury data, by this pipeline, on train and dev only, and its numbers mean "this
architecture learned this series". A Chronos number means "a model that has never seen Georgian
Treasury data guessed this". Counting the two together would make "deep-learning models: 5" mean
two different things in a sentence a client reads.

EXPLORATORY ONLY, AND WHY THAT IS STRUCTURAL RATHER THAN A PROMISE
-----------------------------------------------------------------
None of these can become the model behind an official forecast, and that holds without any new
code. Three independent reasons, each verifiable:

1. ``model_reference.CHAMPION_POOL_CATEGORY`` is ``"machine-learning models"``, which is B_ML
   alone. A registry recipe may only promote a ``point_model`` from that category, and
   ``composition()`` fails if a promoted model falls outside it. F_FOUNDATION is not B_ML.
2. Status is derived from ``experiments/log.csv``, so these are UNTESTED until a measured run is
   entered in the ledger. ``publication_gates`` cannot pass a gate it has no measurement for.
3. A recipe cites a ledger ``run_id`` and ``registry.verify_against_log`` checks the quoted
   numbers against it. A model with no ledger row has nothing a recipe could cite.

So the lock is the same one every unmeasured model is behind. Nothing was added to enforce it.

NOTHING DOWNLOADS AT RUNTIME, AFTER THE FIRST FETCH
---------------------------------------------------
Weights are fetched once and cached under ``~/.cache/huggingface/hub``. Every load here pins an
exact ``revision`` -- a git commit hash on the Hugging Face repo, not a tag and not ``main`` --
so the bytes are fixed. A pinned revision that is already cached is loaded from disk without a
network call, which is what keeps the pipeline auditable: a rerun uses the same weights as the
original run, and an offline machine with a warm cache still works.

If the cache is cold and the network is unavailable, the wrapper reports that it cannot load
rather than raising, and the model is reported as unavailable.

WHY THE CONTEXT IS BUSINESS DAYS
--------------------------------
Measured, not assumed. The Treasury table carries a row for every calendar day, and the 1104
weekend rows are zeros. Handing those to Chronos spends its context window teaching it a weekly
zero pattern, and it then forecasts *into* that pattern: on Revenues at h=5, two of the five
median steps came back NEGATIVE (-4.2M and -1.1M). On the business-day series the same call
returns 74.9M to 78.9M, none of them negative, against a recent business-day mean of 95.3M. Every
other family here models business days for the same reason.

INSTALLING
----------
Optional extras, in ``requirements-foundation.txt``, deliberately NOT in the core requirements.
Every import in this module is lazy, so with the extras absent the app, the registry and the full
test suite all work and these models report themselves as not installed. A fresh clone following
the core README setup is unaffected by this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

#: The family tag. Additive: see ``model_reference._CATEGORY_BY_PIPELINE``, where it maps to a
#: client-facing category, and ``COMPETING_CATEGORIES``, which it is deliberately NOT in.
FAMILY_FOUNDATION = "F_FOUNDATION"

#: Quantiles reported when a model produces them natively. Matches the E_QUANTILE family and the
#: published interval, so a band from here is comparable to a band from there.
QUANTILE_LEVELS: Tuple[float, float, float] = (0.1, 0.5, 0.9)

#: How much history is handed to a model that does not state its own limit.
DEFAULT_CONTEXT = 512


@dataclass(frozen=True)
class FoundationSpec:
    """One pretrained forecaster, and everything needed to reproduce a run of it."""

    name: str
    #: Import name that must be present. One string, because these each come from one package.
    requires: str
    #: Hugging Face repo id.
    repo: str
    #: EXACT commit hash on that repo. Never a tag, never ``main``: both move, and a forecast
    #: whose weights can change is not reproducible.
    revision: str
    #: Whether the model produces quantiles itself. Where it does not, only a point forecast is
    #: reported -- an interval invented here would be this wrapper's uncertainty, not the model's.
    native_quantiles: bool
    summary: str
    #: Which loader/predictor pair to use. Each pretrained family has its own API and they are
    #: not compatible: Chronos wants a torch tensor and returns (quantiles, mean); TimesFM wants
    #: a list of numpy arrays and returns (point, quantiles) with the deciles in one block.
    kind: str = "chronos"
    added: str = "2026-08-19"
    #: Rough download, for a reader deciding whether to fetch it on a phone connection.
    download_mb: Optional[float] = None
    #: Measured wall-clock on CPU, filled in from the smoke run rather than estimated.
    notes: Dict[str, str] = field(default_factory=dict)


#: The registry. One entry per checkpoint.
#:
#: Only models that were actually installed and actually produced a forecast appear here. A name
#: with no working install behind it would be a claim, and this project's whole point is that a
#: registered model is a runnable one. What was attempted and failed is recorded in the session
#: record, not registered here.
MODELS: Tuple[FoundationSpec, ...] = (
    FoundationSpec(
        name="Chronos_Bolt_Small",
        requires="chronos",
        repo="amazon/chronos-bolt-small",
        revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
        native_quantiles=True,
        summary="A pretrained forecaster that has never seen this data. It was trained once on a "
                "large collection of other people's time series, and is asked to forecast this "
                "one from its recent history alone, with no fitting step. Useful as a reading on "
                "how much of this series is predictable from shape alone. Registered as an "
                "exploratory candidate and not yet measured.",
    ),
    FoundationSpec(
        name="TimesFM_2p5_200M",
        requires="timesfm",
        repo="google/timesfm-2.5-200m-pytorch",
        revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        native_quantiles=True,
        kind="timesfm",
        download_mb=925.2,
        summary="A second pretrained forecaster, larger than the one above and from a different "
                "research group, asked the same question in the same way: forecast this series "
                "from its recent history, having never been trained on it. Two independent "
                "zero-shot readings are worth more than one, because agreement between them says "
                "something a single model cannot. Registered as an exploratory candidate and not "
                "yet measured.",
    ),
)


def spec_for(name: str) -> Optional[FoundationSpec]:
    return next((s for s in MODELS if s.name == name), None)


def names() -> List[str]:
    return [s.name for s in MODELS]


# ---------------------------------------------------------------------------
# Availability, reported rather than raised
# ---------------------------------------------------------------------------

REASON_NOT_INSTALLED = (
    "not installed. These are optional extras: install them with "
    "`./backend/.venv/bin/python -m pip install -r backend/requirements-foundation.txt`. "
    "The rest of the app is unaffected without them."
)


def installed(spec: FoundationSpec) -> bool:
    """Is this model's package importable here? Never raises, never imports the package."""
    from importlib.util import find_spec as _find

    try:
        return _find(spec.requires) is not None
    except (ImportError, ValueError):
        return False


def availability() -> Dict[str, Dict]:
    """``{name: {installed, reason, repo, revision, ...}}`` for every registered checkpoint.

    Safe to call from the Streamlit interpreter: it inspects import metadata and never loads a
    model or touches the network.
    """
    out: Dict[str, Dict] = {}
    for spec in MODELS:
        ok = installed(spec)
        out[spec.name] = {
            "name": spec.name,
            "family": FAMILY_FOUNDATION,
            "installed": ok,
            "reason": "" if ok else f"{spec.requires} is {REASON_NOT_INSTALLED}",
            "requires": spec.requires,
            "repo": spec.repo,
            "revision": spec.revision,
            "native_quantiles": spec.native_quantiles,
            "summary": spec.summary,
            "added": spec.added,
        }
    return out


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def business_days_only(series):
    """Drop weekend rows. See the module docstring for the measured reason this matters."""
    return series[series.index.dayofweek < 5]


# ══════════════════════════════════════════════════════════════════════════════
# LOADED ONCE PER PROCESS, NOT ONCE PER FORECAST
#
# The first version loaded the checkpoint inside the predict call. That is harmless for a single
# forecast and ruinous for a backtest: `run_foundation.py` walks a few hundred origins, and
# reloading the weights at each one meant TimesFM's 925 MB checkpoint was read and recompiled 145
# times. Chronos survived it (0.1s from a warm cache); TimesFM did not finish inside ten minutes.
#
# Keyed on (kind, repo, revision) rather than on the model name, so two entries pointing at the
# same pinned checkpoint share one load and a revision change is a different cache entry rather
# than a stale hit.
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4)
def _loaded(kind: str, repo: str, revision: str):
    """The pinned checkpoint, loaded once. Imports inside, never at module level."""
    if kind == "chronos":
        import torch
        from chronos import BaseChronosPipeline

        return BaseChronosPipeline.from_pretrained(
            repo, revision=revision, device_map="cpu", torch_dtype=torch.float32)

    if kind == "timesfm":
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=revision)
        # `compile` fixes the context and horizon budget, so it belongs with the load rather than
        # the call. max_horizon is generous so one compiled model serves every horizon the Lab
        # offers without a recompile per step count.
        model.compile(timesfm.ForecastConfig(
            max_context=DEFAULT_CONTEXT, max_horizon=64,
            normalize_inputs=True, infer_is_positive=True, fix_quantile_crossing=True))
        return model

    raise ValueError(f"no loader for kind {kind!r}")


def _predict_chronos(spec: FoundationSpec, window, horizon: int):
    """``(p10, p50, p90)`` from a pinned Chronos checkpoint.

    Chronos returns ``(quantiles, mean)`` with the requested levels as the last axis, already
    ordered, so the three columns map straight across.
    """
    import torch

    pipe = _loaded(spec.kind, spec.repo, spec.revision)
    tensor = torch.tensor(window, dtype=torch.float32)
    quantiles, _mean = pipe.predict_quantiles(
        tensor, prediction_length=int(horizon), quantile_levels=list(QUANTILE_LEVELS))
    block = quantiles[0].numpy()
    return block[:, 0], block[:, 1], block[:, 2]


def _predict_timesfm(spec: FoundationSpec, window, horizon: int):
    """``(p10, p50, p90)`` from a pinned TimesFM checkpoint.

    The quantile block has TEN columns, and which is which was checked rather than assumed:
    column 0 is the mean and columns 1 to 9 are the deciles q10 through q90. Verified on real
    data -- column 5 came back byte-identical to the returned point forecast (60,286,384) and
    columns 1 to 9 were sorted. So p10 is column 1, p50 column 5, p90 column 9.

    ``fix_quantile_crossing`` and ``infer_is_positive`` are asked for at compile time: the first
    because independently produced quantiles can cross, and the second because a negative
    Treasury revenue figure is not a defensible forecast.
    """
    import numpy as np

    model = _loaded(spec.kind, spec.repo, spec.revision)
    _point, quantiles = model.forecast(horizon=int(horizon),
                                       inputs=[np.asarray(window, dtype="float32")])
    block = np.asarray(quantiles)[0]
    return block[:, 1], block[:, 5], block[:, 9]


#: Which predictor each family uses. A dict rather than a chain of ifs so a new family is one
#: entry and one function, the same shape as the rest of this project's registries.
_PREDICTORS = {"chronos": _predict_chronos, "timesfm": _predict_timesfm}


def forecast(name: str, series, horizon: int,
             context: int = DEFAULT_CONTEXT) -> Tuple[Optional[Dict], Optional[str]]:
    """``(result, None)`` or ``(None, reason)``. Never raises.

    ``series`` is a pandas Series indexed by date. Weekends are dropped here rather than by the
    caller, so every caller gets the same treatment and none has to remember why.

    The result carries ``p10``/``p50``/``p90`` when the model produces quantiles itself, and only
    ``p50`` when it does not. An interval this wrapper invented would describe this wrapper's
    uncertainty rather than the model's, which is worse than having none.
    """
    spec = spec_for(name)
    if spec is None:
        return None, f"{name!r} is not a registered foundation model."
    if not installed(spec):
        return None, f"{spec.requires} is {REASON_NOT_INSTALLED}"

    daily = business_days_only(series.dropna())
    if len(daily) < 32:
        return None, (f"The context is too short: {len(daily)} business days, and at least 32 are "
                      f"needed before a zero-shot forecast means anything.")

    predictor = _PREDICTORS.get(spec.kind)
    if predictor is None:
        return None, f"{spec.name} has no predictor registered for kind {spec.kind!r}."

    try:
        p10, p50, p90 = predictor(spec, daily.values[-context:], int(horizon))
    except Exception as exc:                           # noqa: BLE001 - reported, never raised
        return None, f"{spec.name} could not produce a forecast: {type(exc).__name__}: {exc}"

    import pandas as pd

    future = pd.bdate_range(daily.index[-1] + pd.offsets.BDay(1), periods=int(horizon))
    return {
        "model": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "target_dates": [d.date().isoformat() for d in future],
        "origin_date": daily.index[-1].date().isoformat(),
        "context_used": int(min(len(daily), context)),
        "p10": [float(v) for v in p10],
        "p50": [float(v) for v in p50],
        "p90": [float(v) for v in p90],
        "native_quantiles": spec.native_quantiles,
        "zero_shot": True,
    }, None
