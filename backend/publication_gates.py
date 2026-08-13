"""The publication decision: which gates a recipe must pass, and why those thresholds.

Until P2 this decision had no code. ``registry/recipes.json`` carried hand-written gate
verdicts and a hand-written ``publication.verdict``, so the reasoning lived in a commit
message and the numbers lived in a file, with nothing tying them together. This module is
the reasoning; the registry becomes its output, and ``test_publication_gates.py`` asserts
the two agree.

--------------------------------------------------------------------------------
WHAT WAS WRONG
--------------------------------------------------------------------------------
Measured on the three live recipes:

    target                MASE    skill_vs_ruler   sentinel   verdict
    Revenues             0.758       55.92%         1.2255    withheld_as_forecast
    Expenditure          1.104       29.42%         1.0882    withheld_as_forecast
    State budget balance 1.578       20.01%         7.0058    publishable

MASE is the ratio of the model's error to a TRAIN-only seasonal-naive benchmark, so
**MASE < 1 means "better than the naive forecast" and MASE > 1 means "worse"**. It was
computed on every run and gated on nothing. The consequence: the project withheld its
only model that beats the naive benchmark (Revenues, 0.758) and published the one that
loses to it by 58% (the stock target, 1.578).

--------------------------------------------------------------------------------
THE THREE THRESHOLD DECISIONS
--------------------------------------------------------------------------------
**1. MASE < 1.0 becomes a binding gate.**
The threshold is not chosen, it is definitional: MASE is a ratio against a benchmark, so
1.0 is the break-even where the model and the benchmark are equally good. Anything above
means a documented trivial alternative is better, and there is no reading under which such
a model's numbers are the best available estimate. Deliberately no safety margin — adding
one (0.9, 0.95) would reintroduce exactly the uncalibrated-constant problem this module
exists to fix. The margin is *reported* instead, so a model at 0.99 is visibly marginal.

**2. `vs_ruler` is demoted from gate to reported diagnostic, and MASE replaces it.**
Its threshold was ``> 0%`` — beat h-step persistence by any margin at all. Measured, that
is close to vacuous on a flow: a featureless constant (the TRAIN mean, no features, no
model) scores **38.58%** on Revenues and **35.69%** on Expenditure against the same ruler,
because persistence is a poor benchmark for a spiky series. So ``> 0%`` passes models far
worse than a constant.

It cannot simply be raised, because the constant's score is period- and target-dependent:
the same constant scores **-299.43%** on the stock target. Any fixed percentage would be
meaningful for one target and arbitrary for the others. MASE does the same job properly —
scale-free, comparable across targets, and normalised by a benchmark computed on TRAIN
only. ``skill_vs_ruler`` is still reported, because it is the project's shared ruler and
the Forecast page plots it, but it no longer decides publication.

**3. The signal sentinel threshold is calibrated: 1.50 -> 1.15.**
1.50 was a bare constant with a comment. Its null distribution had never been estimated,
so its false-positive rate was unknown. Measured here (see ``reports/sentinel_calibration.md``):
360 null draws, features row-permuted to destroy the feature-target pairing while
preserving their marginals, across all three targets.

    pooled permuted null (n=360):
      median 1.0023   p95 1.0532   p99 1.0943   p99.9 1.1124   max 1.1164

    false-positive rate by threshold:
      1.05 -> 6.94%      1.10 -> 0.56%      1.12 -> 0.00%
      1.15 -> 0.00%      1.50 -> 0.00%

**1.50 was not a noise margin.** The null never comes close to it: its FPR is 0.00%, and so
is 1.12's. A threshold eight-plus null standard deviations out does not buy protection that
1.15 lacks; it only rejects real readings. The best flow reading anywhere (1.421, tree
probe on Revenues) sat 0.08 below a line that no null draw approached.

So the threshold is set to **1.15** — the smallest round value with a *measured* 0.00%
false-positive rate over 360 draws, sitting above the observed null maximum of 1.1164. The
next lower round value, 1.10, has a measured FPR of 0.56%.

This changes verdicts, and it should: at 1.15, Revenues' 1.2255 clears. Expenditure's
1.0882 does not — and now for a stated reason rather than an assumed one, because it sits
*below* the pooled null's 99th percentile of 1.0943. Its reading is not merely under a
threshold; it is inside the distribution of readings that signal-free features produce.

--------------------------------------------------------------------------------
THE FOUR VERDICTS STAY FOUR
--------------------------------------------------------------------------------
Leakage, no-signal, persistence-mimicry and coverage remain four independent conditions
with four separate reasons, exactly as ``test_failure_mode_distinctness.py`` requires. The
accuracy gate added here is a fifth condition, not a merge of any of them: a model can beat
the naive benchmark while replaying persistence, and can fail the benchmark while carrying
real signal. Nothing here collapses one into another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# ── thresholds, each with its justification above ────────────────────────────

#: MASE break-even. A ratio against a benchmark: 1.0 is definitional, not chosen.
MASE_MAX = 1.0

#: Calibrated from a measured null distribution. FPR 0.00% over 360 draws.
SENTINEL_MIN = 1.15

#: Superseded. Kept as a named constant so the change is greppable and the old value is
#: not silently forgotten; nothing reads it as a gate any more.
SENTINEL_MIN_UNCALIBRATED = 1.50

#: Train/DEV MAE ratio above which a model is excluded from crowning (unchanged).
OVERFIT_MAX = 3.0

#: The null study behind SENTINEL_MIN.
SENTINEL_NULL = {
    "n_draws": 360,
    "construction": "feature rows permuted, marginals preserved, per target, TRAIN->DEV",
    "median": 1.0023,
    "p95": 1.0532,
    "p99": 1.0943,
    "p99_9": 1.1124,
    "max": 1.1164,
    "fpr_at_threshold": {"1.05": 0.0694, "1.10": 0.0056, "1.12": 0.0,
                         "1.15": 0.0, "1.50": 0.0},
    "study": "reports/sentinel_calibration.md",
}

#: Verdicts, ordered by severity. The first that applies wins.
WITHHELD = "withheld"
WITHHELD_AS_FORECAST = "withheld_as_forecast"
PUBLISHABLE = "publishable"


@dataclass
class Measured:
    """What a run measured. ``None`` means "not measured", never "passed"."""

    target: str
    mase: Optional[float] = None
    sentinel_ratio: Optional[float] = None
    skill_vs_ruler_pct: Optional[float] = None
    overfit_ratio: Optional[float] = None
    leakage_detected: Optional[bool] = None
    persistence_mimicry: Optional[bool] = None
    coverage: Optional[float] = None
    coverage_nominal: Optional[float] = None
    coverage_band: Optional[List[float]] = None


def _num(x) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def evaluate_gates(m: Measured) -> Dict[str, Dict]:
    """Every gate's verdict, measurement, threshold and plain-language reason.

    ``passed`` is tri-state: ``True`` / ``False`` / ``None`` = not measured. A gate that
    could not be evaluated must never read as a pass.
    """
    gates: Dict[str, Dict] = {}

    # ── accuracy: the gate that was missing ──────────────────────────────────
    mase = _num(m.mase)
    if mase is None:
        gates["accuracy_vs_naive"] = {
            "passed": None, "name": "accuracy vs repeating the same weekday last week",
            # The acronym lives here, not in `name`: the HTML treasury report renders
            # `name` to readers and test_treasury_report.py forbids jargon there.
            "metric": "MASE",
            "measured": None, "threshold": MASE_MAX,
            "reason_plain": ("MASE was not measured for this run, so we cannot say whether the "
                            "model beats a naive seasonal forecast. Not measured is not a pass."),
        }
    else:
        passed = mase < MASE_MAX
        pct = abs(1.0 - mase) * 100.0
        gates["accuracy_vs_naive"] = {
            "passed": bool(passed),
            "name": "accuracy vs repeating the same weekday last week",
            # The acronym lives here, not in `name`: the HTML treasury report renders
            # `name` to readers and test_treasury_report.py forbids jargon there.
            "metric": "MASE",
            "measured": round(mase, 6), "threshold": MASE_MAX,
            "margin_pct": round(pct, 2),
            # Reader-facing: no acronyms, no model names, no raw ratios. The number lives in
            # `measured`; this sentence has to survive a Treasury reader with no notes.
            # `test_insights.py` and `test_treasury_report.py` enforce that, and caught an
            # earlier draft of this string that said "MASE".
            "reason_plain": (
                f"This model is {pct:.1f}% more accurate than simply repeating what happened "
                f"on the same weekday last week, so it is the better of the two estimates."
                if passed else
                f"Simply repeating what happened on the same weekday last week is {pct:.1f}% "
                f"more accurate than this model. There is no reading under which this model's "
                f"numbers are the best available estimate, so they are not published as a "
                f"forecast."),
        }

    # ── signal: calibrated, and one input among several ──────────────────────
    sen = _num(m.sentinel_ratio)
    if sen is None:
        gates["signal"] = {
            "passed": None, "name": "signal (shuffled-target control)",
            "measured": None, "threshold": SENTINEL_MIN,
            "reason_plain": "The shuffled-target control was not run, so signal is unmeasured.",
        }
    else:
        passed = sen >= SENTINEL_MIN
        inside_null = sen < SENTINEL_NULL["p99"]
        gates["signal"] = {
            "passed": bool(passed),
            "name": "signal (shuffled-target control)",
            "measured": round(sen, 6), "threshold": SENTINEL_MIN,
            "threshold_source": (
                f"calibrated against a measured null distribution "
                f"(n={SENTINEL_NULL['n_draws']}, false-positive rate 0.00%); "
                f"see {SENTINEL_NULL['study']}"),
            "null_p99": SENTINEL_NULL["p99"],
            "inside_null_distribution": bool(inside_null),
            "reason_plain": (
                f"Shuffling the historical answers made this model's error {sen:.2f} times "
                f"worse (we require {SENTINEL_MIN:.2f} times). The inputs carry real "
                f"information about what happens next."
                if passed else
                f"Shuffling the historical answers barely worsened this model's error "
                f"({sen:.2f} times, where we require {SENTINEL_MIN:.2f} times). "
                + ("That reading sits INSIDE the range that deliberately uninformative inputs "
                   "produce on this data, so it is not distinguishable from no information at "
                   "all. "
                   if inside_null else
                   "That is below the required level, though above the range that deliberately "
                   "uninformative inputs produce. ")
                + "The model is tracking the typical level rather than anticipating events."),
        }

    # ── leakage: a distinct verdict, unchanged ───────────────────────────────
    if m.leakage_detected is None:
        gates["leakage"] = {"passed": None, "name": "leakage", "measured": None,
                            "threshold": "no future information in features",
                            "reason_plain": "Leakage was not checked for this run."}
    else:
        gates["leakage"] = {
            "passed": not bool(m.leakage_detected), "name": "leakage",
            "measured": bool(m.leakage_detected),
            "threshold": "no future information in features",
            "reason_plain": ("No feature was found to carry information from after the forecast "
                             "origin." if not m.leakage_detected else
                             "A feature carries information from after the forecast origin, so "
                             "the model's accuracy is not achievable in production."),
        }

    # ── persistence-mimicry: a distinct verdict, unchanged ───────────────────
    if m.persistence_mimicry is None:
        gates["persistence_mimicry"] = {
            "passed": None, "name": "persistence-mimicry (shift diagnostic)",
            "measured": None, "threshold": "predictions must not be a lagged copy",
            "reason_plain": "The shift diagnostic was not run for this run."}
    else:
        gates["persistence_mimicry"] = {
            "passed": not bool(m.persistence_mimicry),
            "name": "persistence-mimicry (shift diagnostic)",
            "measured": bool(m.persistence_mimicry),
            "threshold": "predictions must not be a lagged copy",
            "reason_plain": ("The forecast does not simply replay a recent actual."
                             if not m.persistence_mimicry else
                             "The forecast is essentially a lagged copy of the series — it "
                             "replays a recent actual rather than anticipating anything."),
        }

    # ── coverage: a distinct verdict, level read as data ─────────────────────
    cov = _num(m.coverage)
    if cov is None:
        gates["coverage"] = {
            "passed": None, "name": "interval coverage", "measured": None,
            "threshold": None,
            "reason_plain": ("This model reports no prediction intervals, so their calibration "
                             "was not measured. Point forecasts are unaffected."),
        }
    else:
        nominal = _num(m.coverage_nominal) or 0.80
        band = m.coverage_band or [round(max(0.0, nominal - 0.10), 10),
                                   round(min(1.0, nominal + 0.10), 10)]
        passed = band[0] <= cov <= band[1]
        gates["coverage"] = {
            "passed": bool(passed), "name": "interval coverage",
            "measured": round(cov, 6), "threshold": list(band),
            "nominal": nominal,
            "reason_plain": (
                f"The {nominal:.0%} interval contained {cov:.1%} of outcomes, within the "
                f"accepted band."
                if passed else
                f"The interval is advertised as {nominal:.0%} but contained {cov:.1%} of "
                f"outcomes, outside the accepted [{band[0]:.0%}, {band[1]:.0%}]. The point "
                f"forecast may still be usable; the interval is not."),
        }

    # ── overfitting: unchanged ───────────────────────────────────────────────
    ratio = _num(m.overfit_ratio)
    gates["overfitting"] = {
        "passed": None if ratio is None else bool(ratio <= OVERFIT_MAX),
        "name": "overfitting (DEV/TRAIN error ratio)",
        "measured": None if ratio is None else round(ratio, 4),
        "threshold": OVERFIT_MAX,
        "reason_plain": ("The overfit ratio was not recorded." if ratio is None else
                         f"DEV error is {ratio:.2f}x TRAIN error, within the {OVERFIT_MAX:.1f}x "
                         f"limit." if ratio <= OVERFIT_MAX else
                         f"DEV error is {ratio:.2f}x TRAIN error, above the {OVERFIT_MAX:.1f}x "
                         f"limit: the model memorised rather than generalised."),
    }

    return gates


#: Which failures produce which verdict. Severity order, first match wins.
#:
#: `withheld` is for failures where a documented alternative is strictly better, so showing
#: the numbers at all would invite a worse decision than not showing them: leakage (the
#: accuracy is unachievable) and MASE >= 1 (a naive forecast beats it).
#:
#: `withheld_as_forecast` is for failures where the numbers remain the best central-tendency
#: estimate available but a specific claim cannot be made: no signal, persistence-mimicry,
#: miscalibrated intervals. The numbers are shown; the claim is withheld.
_SEVERITY = (
    ("leakage", WITHHELD),
    ("accuracy_vs_naive", WITHHELD),
    ("signal", WITHHELD_AS_FORECAST),
    ("persistence_mimicry", WITHHELD_AS_FORECAST),
    ("coverage", WITHHELD_AS_FORECAST),
)


def publication_verdict(gates: Dict[str, Dict]) -> Dict:
    """The publication decision, with every failing reason kept separate.

    All failures are reported, not just the one that set the verdict — a model that both
    loses to the naive benchmark and shows no signal has two problems, and a reader fixing
    one should know about the other.
    """
    failed = [(name, gates[name]) for name, _ in _SEVERITY
              if name in gates and gates[name].get("passed") is False]

    verdict = PUBLISHABLE
    for name, v in _SEVERITY:
        if name in gates and gates[name].get("passed") is False:
            verdict = v
            break

    unmeasured = sorted(n for n, g in gates.items()
                        if g.get("passed") is None and n != "coverage")

    return {
        "verdict": verdict,
        "failing_gates": [n for n, _ in failed],
        "reasons": [g["reason_plain"] for _, g in failed],
        "reason_plain": (" ".join(g["reason_plain"] for _, g in failed) if failed else
                         "Every measured gate passed."),
        "unmeasured_gates": unmeasured,
        "decided_by": ([n for n, _ in failed][0] if failed else None),
    }


def decide(m: Measured) -> Dict:
    """Convenience: gates plus verdict for one target."""
    gates = evaluate_gates(m)
    out = publication_verdict(gates)
    out["gates"] = gates
    out["target"] = m.target
    return out
