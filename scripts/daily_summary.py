#!/usr/bin/env python3
"""
Write a short plain-text summary of a daily forecast run.

Called by scripts/run_daily_forecast.sh after every pipeline family has run.
For each family it reports the models that ran, the best model's error, the
skill vs. the persistence baseline, and any leakage or shift flags.

Two sources of flags are combined, and both are shown:
  1. Warnings the *pipeline itself* recorded in artifacts/integrity_report.json
     (leakage_warning, shift_interpretation, and the "model ≈ naive baseline"
     guard).  These are surfaced verbatim — not re-thresholded here.
  2. Two independent checks this summary computes from the predictions
     (origin_date >= target_date, and detect_lagged_copy).

Every field is always printed; where a family did not produce a value
(e.g. C_DL quick mode writes no integrity report), the line reads
"n/a (not produced)" rather than being silently dropped.

The exit code matters: a non-zero exit tells the shell script the run should
be treated as failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The leakage/shift diagnostics live in the backend package.  Add backend/ to
# the import path based on this file's location (scripts/ is a sibling of
# backend/), so the helper works no matter what directory it is called from.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from forecast_integrity import detect_lagged_copy  # noqa: E402  (import after sys.path edit)

NA = "n/a (not produced)"


def _find_one(family_dir: Path, filename: str) -> Path | None:
    """Return the first matching file under family_dir, or None.

    We search recursively because the DL pipeline nests its outputs one level
    deeper (in a `daily/` subfolder) than the other families.
    """
    direct = family_dir / filename
    if direct.exists():
        return direct
    matches = sorted(family_dir.rglob(filename))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict:
    """Read a JSON file into a dict, tolerating a missing file."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_integrity_report(family_dir: Path) -> Path | None:
    """Locate a family's integrity report.

    Most families write `integrity_report.json`.  The DL pipeline instead
    writes `integrity_<Target>_h<H>.json` (e.g. integrity_Revenues_h5.json),
    nested under daily/artifacts/.  Try the standard name first (unchanged
    behaviour for the other families), then fall back to any integrity_*.json,
    sorted so the choice is deterministic when several match.
    """
    standard = _find_one(family_dir, "integrity_report.json")
    if standard is not None:
        return standard
    matches = sorted(family_dir.rglob("integrity_*.json"))
    return matches[0] if matches else None


def gate_reasons(report: dict, leakage_flag: bool, shift_flag: bool = False) -> list[str]:
    """Reasons a family fails the quality gate (empty list = no failure).

    A family fails when:
      * the pipeline recorded run_status=FAILED_QUALITY (or, in older
        artifacts, quality_gate_passed=false);
      * a leakage flag was raised — a leaky model is not usable no matter how
        good it looks;
      * the shuffled-target control found no usable signal (M-5).  Note this
        is reported as "no signal", not "leakage": a low shuffled/real ratio
        means the features never predicted the target, which is the opposite
        of the model seeing the future;
      * a shift diagnostic says the forecast is essentially a lagged copy of
        the series.  Skill above the threshold and "reproduces persistence"
        can both be true at once — measured against different baselines — and
        a model that only replays yesterday is not usable for planning even
        when it clears the skill bar;
      * the intervals are miscalibrated (item 5).  This is the FOURTH distinct
        verdict, and it used to be invisible here: a coverage failure arrived
        only as `run_status=FAILED_QUALITY` or the generic "quality gate
        failed", so an E_QUANTILE family withheld for broken intervals was
        indistinguishable from one withheld for poor skill.  A model whose P50
        is excellent and whose 80% band covers 43% of outcomes needs a
        different fix from one that simply cannot forecast, and the reason a
        family was withheld is what a treasury reader acts on.

    The four verdicts — leakage, no signal, persistence-mimicry, coverage —
    are deliberately independent conditions, each with its own phrasing, and
    ``test_failure_mode_distinctness.py`` pins the separation.
    """
    reasons: list[str] = []
    coverage_reasons = _coverage_failure_reasons(report or {})
    if report:
        status = str(report.get("run_status", "")).strip().upper()
        generic = None
        if status == "FAILED_QUALITY":
            generic = "run_status=FAILED_QUALITY"
        else:
            # X9: read through the single canonical reader. This function previously checked
            # only `quality_gate_passed`, so a B_ML run -- which wrote the INVERTED
            # `quality_gate_failed` -- was reported as PASSING its gate when it had failed.
            from forecast_integrity import read_gate
            if read_gate(report) is False:
                generic = "quality gate failed (per the family's integrity report)"
        # Suppress the generic line only when the specific cause is already stated below.
        # Dropping it unconditionally would hide failures that have no named cause.
        if generic and not coverage_reasons:
            reasons.append(generic)
    if leakage_flag:
        reasons.append("leakage flag raised")
    if report.get("signal_detected") is False:
        ratio = report.get("shuffled_to_normal_ratio")
        ratio_s = f"{float(ratio):.2f}" if _is_number(ratio) else "n/a"
        reasons.append(f"no signal beyond shuffled targets (ratio {ratio_s})")
    if shift_flag:
        reasons.append("forecast is persistence-like (shift diagnostic)")
    reasons.extend(coverage_reasons)
    return reasons


def _composition_fields() -> dict:
    """The model-composition block for SUMMARY.json, or an explicit reason it is absent.

    Derived from ``model_reference.model_pool()``, which imports every family's registry. That
    import can fail on a machine missing a modelling library, and a summary must not fail because
    the *catalogue* could not be read -- the run itself is unaffected.

    So absence follows the pattern documented in AGENT_ARTIFACT_CONTRACT.md §0: never a bare
    missing key, always a companion field naming the reason. A consumer that finds
    ``client_framing`` absent and ``client_framing_unavailable_reason`` present knows the
    composition was not derivable here, as opposed to a writer that forgot to emit it.
    """
    try:
        sys.path.insert(0, str(BACKEND_DIR))
        from model_reference import client_framing, composition

        comp = composition()
        return {
            "client_framing": client_framing(),
            "model_composition": {
                "counts": comp["counts"],
                "members": comp["members"],
                # The two different things "champion" means here. Conflating them is how a true
                # sentence becomes a wrong one: `champion_pool` is what a registry recipe may
                # promote, `daily_best_model_families` is every family this file writes a
                # per-family `best_model` for -- and the Agent ranks across the latter.
                "champion_pool": comp.get("champion_pool"),
                "champion_pool_category": comp.get("champion_pool_category"),
                "daily_best_model_families": comp.get("daily_best_model_families"),
                # Integrity cross-check, carried so a consumer does not have to trust the
                # sentence: any model a recipe actually promotes that is NOT in champion_pool.
                # Non-empty means the eligible pool a client was told about is wrong.
                "promoted_by_registry": comp.get("promoted_by_registry"),
                "promoted_outside_champion_pool": comp.get("promoted_outside_champion_pool"),
            },
        }
    except Exception as exc:                       # noqa: BLE001 - a catalogue, not the run
        return {"client_framing_unavailable_reason":
                f"could not derive the model composition: {type(exc).__name__}: {exc}"}


def _coverage_failure_reasons(report: dict) -> list[str]:
    """Interval miscalibration, phrased as its own verdict.

    The nominal level is read as data (``coverage_nominal``) and only falls back to the key name
    when a run predates that field — the level is a property of the fitted alphas, not of the
    string that carries the number (audit field #2).

    Deliberately says nothing about skill, leakage or persistence: a family whose intervals are
    broken but whose median is excellent must be told exactly that.
    """
    out: list[str] = []
    # 1. The family's own gate already named coverage. Trust it and quote it.
    for r in report.get("quality_gate_reasons") or []:
        if "coverage" in str(r).lower():
            out.append(f"intervals miscalibrated: {r}")
    if out:
        return out

    # 2. No named reason, but the numbers are present and outside the band.
    measured = None
    for key in ("coverage_p10_p90", "coverage"):
        if _is_number(report.get(key)):
            measured = float(report[key])
            break
    if measured is None:
        for key, val in report.items():
            if str(key).startswith("coverage_p") and _is_number(val):
                measured = float(val)
                break
    if measured is None:
        return out

    nominal = report.get("coverage_nominal")
    nominal = float(nominal) if _is_number(nominal) else 0.80
    band = report.get("coverage_band")
    if isinstance(band, (list, tuple)) and len(band) == 2 and all(_is_number(b) for b in band):
        lo, hi = float(band[0]), float(band[1])
    else:
        lo, hi = max(0.0, nominal - 0.10), min(1.0, nominal + 0.10)
    if not (lo <= measured <= hi):
        out.append(f"intervals miscalibrated: coverage {measured:.1%} outside "
                   f"[{lo:.0%}, {hi:.0%}] (nominal {nominal:.0%})")
    return out


def _fmt_money(x) -> str:
    """Format a large number with thousands separators (or 'n/a')."""
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def pipeline_leakage(report: dict) -> tuple[str, bool]:
    """Leakage warning the pipeline itself recorded (surfaced verbatim).

    Returns (text, is_flag).  If the field was never produced, text is the
    "n/a" marker and is_flag is False.
    """
    if "leakage_warning" not in report:
        return (NA, False)
    if report.get("leakage_warning") is True:
        ratio = report.get("shuffled_to_normal_ratio")
        ratio_s = f"{float(ratio):.2f}" if _is_number(ratio) else "n/a"
        return (f"leakage_warning=true (shuffled_to_normal_ratio={ratio_s})", True)
    return ("none", False)


def pipeline_shift(report: dict) -> tuple[list[str], bool]:
    """Shift warnings the pipeline itself recorded (surfaced verbatim).

    Returns (lines, is_flag).  Combines the pipeline's own shift_interpretation
    string with its "model ≈ naive baseline" guard.  The guard reproduces the
    pipeline's exact comparison (b_ml_pipeline.py GUARD B: model MAE within 10%
    of the shift=-h MAE) using the numbers the pipeline stored, so the numbers
    shown are the pipeline's own.
    """
    has_fields = ("shift_interpretation" in report) or ("best_shift" in report)
    if not has_fields:
        return ([NA], False)

    lines: list[str] = []

    interp = report.get("shift_interpretation")
    if isinstance(interp, str) and not interp.strip().upper().startswith("OK"):
        lines.append(interp)  # verbatim

    if report.get("is_critical_timestamping_bug") is True:
        lines.append("is_critical_timestamping_bug=true")

    # GUARD B: model performance ≈ naive (shift=-h) baseline.
    mae_model = report.get("mae_model")
    naive_mae = report.get("mae_shift_minus_h")
    if _is_number(mae_model) and _is_number(naive_mae):
        denom = max(mae_model, naive_mae, 1.0)
        if abs(mae_model - naive_mae) / denom < 0.1:
            lines.append(
                f"model performance ≈ naive baseline (shift=-h): "
                f"Model MAE={_fmt_money(mae_model)} vs Naive MAE={_fmt_money(naive_mae)}"
            )

    if not lines:
        return (["none"], False)
    return (lines, True)


def summary_leakage_check(preds: pd.DataFrame) -> tuple[str, bool]:
    """This summary's own leakage check: any prediction that peeks ahead."""
    if not {"origin_date", "target_date"}.issubset(preds.columns):
        return (NA, False)
    origin = pd.to_datetime(preds["origin_date"], errors="coerce")
    target = pd.to_datetime(preds["target_date"], errors="coerce")
    violations = int((origin >= target).sum())
    if violations > 0:
        return (f"{violations} row(s) with origin_date >= target_date", True)
    return ("none", False)


def summary_shift_check(preds: pd.DataFrame) -> tuple[str, bool]:
    """This summary's own shift check via detect_lagged_copy."""
    if not {"y_true", "y_pred"}.issubset(preds.columns):
        return (NA, False)
    result = detect_lagged_copy(preds)
    flagged = [m["model"] for m in result.get("per_model", []) if m.get("flagged")]
    if flagged:
        return ("; ".join(result.get("details", [])), True)
    return ("none", False)


def summarize_family(name: str, family_dir: Path) -> dict:
    """Collect the summary facts for one pipeline family."""
    info: dict = {
        "name": name,
        "ok": False,
        "models": NA,
        "best_model": NA,
        "skill_pct": NA,
        "run_status": NA,
        "quality": NA,
        "gate_passed": None, "gate_reasons": [],
        "integrity_found": False,
        "pipe_leak": NA, "pipe_leak_flag": False,
        "pipe_shift": [NA], "pipe_shift_flag": False,
        "chk_leak": NA, "chk_leak_flag": False,
        "chk_shift": NA, "chk_shift_flag": False,
        "notes": [],
    }

    pred_path = _find_one(family_dir, "predictions_long.csv")
    if pred_path is None:
        info["notes"].append("no predictions_long.csv found")
        return info

    preds = pd.read_csv(pred_path)
    if preds.empty:
        info["notes"].append("predictions_long.csv is empty")
        return info
    info["ok"] = True

    # ── Models run + best model, from the leaderboard if present ──
    lb_path = _find_one(family_dir, "leaderboard.csv")
    if lb_path is not None:
        lb = pd.read_csv(lb_path)
        if "model" in lb.columns:
            info["models"] = ", ".join(str(m) for m in lb["model"].tolist())
        if "model" in lb.columns and "MAE" in lb.columns:
            real = lb[~lb["model"].astype(str).str.contains("baseline", case=False, na=False)]
            real = real.dropna(subset=["MAE"])
            if not real.empty:
                best = real.loc[real["MAE"].idxmin()]
                info["best_model"] = f"{best['model']} (MAE {_fmt_money(best['MAE'])})"
    elif "model" in preds.columns:
        info["models"] = ", ".join(str(m) for m in preds["model"].unique().tolist())

    # ── Skill vs persistence + run status, from the integrity report ──
    report = _read_json(_find_integrity_report(family_dir))
    info["integrity_found"] = bool(report)

    # X11: a persistence baseline computed over zero prediction rows is not a baseline. It
    # previously appeared in the leaderboard as an ordinary row, indistinguishable from one
    # measured on real predictions, so a reader could not tell "computed on nothing" from
    # "computed and equal to zero".
    _n_pred = int(len(preds)) if preds is not None else 0
    info["n_prediction_rows"] = _n_pred
    _has_baseline = bool(report) and report.get("mae_persistence") is not None
    info["baseline_without_predictions"] = bool(_has_baseline and _n_pred == 0)
    if info["baseline_without_predictions"]:
        info["gate_reasons"] = list(info.get("gate_reasons", [])) + [
            "persistence baseline reported with zero prediction rows"
        ]
    if _is_number(report.get("skill_pct")):
        info["skill_pct"] = f"{float(report['skill_pct']):.2f}%"
    if report.get("run_status"):
        info["run_status"] = str(report["run_status"])

    # ✅ Prefer the family's OWN verified best model over the leaderboard's
    # lowest MAE.  A family that evaluates several models (E_QUANTILE) selects
    # its best among those that PASS the quality gate, so the lowest-MAE model
    # may be one it rejected.  Taking the name from the leaderboard while the
    # skill/coverage numbers come from the integrity report produced a report
    # that named one model and described another.
    declared_best = report.get("best_model")
    if isinstance(declared_best, str) and declared_best.strip():
        mae = None
        detail = report.get("models")
        if isinstance(detail, dict) and declared_best in detail:
            mae = detail[declared_best].get("mae_p50")
        if mae is None:
            mae = report.get("mae_p50") or report.get("mae_model")
        info["best_model"] = (f"{declared_best} (MAE {_fmt_money(mae)})"
                              if mae is not None else declared_best)

    # ── Flags: pipeline-recorded (verbatim) and this summary's own checks ──
    info["pipe_leak"], info["pipe_leak_flag"] = pipeline_leakage(report)
    info["pipe_shift"], info["pipe_shift_flag"] = pipeline_shift(report)
    info["chk_leak"], info["chk_leak_flag"] = summary_leakage_check(preds)
    info["chk_shift"], info["chk_shift_flag"] = summary_shift_check(preds)

    # ── Quality gate: computed last, because leakage flags feed into it ──
    info["gate_reasons"] = gate_reasons(
        report, family_leakage_flag(info), family_shift_flag(info)
    )
    # X10: ONE PUBLISHER, and this is not it. The family's integrity_report.json publishes the
    # verdict; SUMMARY.json derives from it and from the summary-side flags. Both computing it
    # independently is what let SUMMARY.json contradict integrity_report.json on the
    # 2026-08-04 C_DL run. The derivation is recorded alongside the value so a reader can see
    # which source produced it.
    from forecast_integrity import read_gate
    _published = read_gate(report) if info["integrity_found"] else None
    if info["gate_reasons"]:
        info["gate_passed"] = False
        info["gate_source"] = "derived: summary-side flags raised"
    elif _published is None:
        info["gate_passed"] = None  # never verified — must not look like a pass
        info["gate_source"] = ("never verified: no integrity report, or it records no gate "
                               "verdict")
    else:
        info["gate_passed"] = _published
        info["gate_source"] = "published by the family's integrity_report.json"
    info["gate_published"] = _published

    return info


def family_leakage_flag(s: dict) -> bool:
    return bool(s["pipe_leak_flag"] or s["chk_leak_flag"])


def family_shift_flag(s: dict) -> bool:
    return bool(s["pipe_shift_flag"] or s["chk_shift_flag"])


def check_freshness(data_file: Path, date_col: str, run_date: str, stale_days: int) -> tuple[str, bool]:
    """Return (message, is_stale) describing how current the data file is."""
    try:
        dates = pd.read_csv(data_file, usecols=[date_col])[date_col]
        latest = pd.to_datetime(dates, errors="coerce").max()
    except Exception as exc:
        return (f"Could not read latest data date from {data_file.name}: {exc}", True)

    if pd.isna(latest):
        return (f"No valid dates found in {data_file.name}", True)

    run_ts = pd.to_datetime(run_date)
    gap_days = (run_ts.normalize() - latest.normalize()).days
    line = f"Latest data date: {latest.date()} ({gap_days} day(s) before run date {run_ts.date()})"
    return (line, gap_days > stale_days)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a daily forecast run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target", required=True)
    parser.add_argument("--cadence", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--families", required=True, help="space-separated family names")
    parser.add_argument("--stale-days", type=int, default=3)
    parser.add_argument("--mode", choices=["production", "backtest"], default="production",
                        help="production: warn when data is stale. backtest: the data "
                             "deliberately ends in the past; label the run instead of warning.")
    # The contract gate. On by default: an artifact that fails the contract is one the Agent
    # would read wrongly, so publishing it is the harm. --no-validate exists for diagnosing a
    # broken run, not for getting past the gate.
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the artifact contract check (diagnosis only)")
    parser.add_argument("--strict-validate", action="store_true",
                        help="promote contract warnings to errors; what a NEW run should meet")
    parser.add_argument("--published-root", default=None, type=Path,
                        help="also validate the published forecast issues under this directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_file = Path(args.data_file)
    families = args.families.split()

    fresh_line, is_stale = check_freshness(data_file, args.date_col, args.run_date, args.stale_days)
    summaries = [summarize_family(fam, run_dir / fam.lower()) for fam in families]

    n_ok = sum(1 for s in summaries if s["ok"])
    n_leak = sum(1 for s in summaries if family_leakage_flag(s))
    n_shift = sum(1 for s in summaries if family_shift_flag(s))
    n_quality = sum(1 for s in summaries if s["gate_passed"] is False)

    lines: list[str] = []
    lines.append("AI4CM Daily Forecast Summary")
    lines.append("=" * 40)
    lines.append(f"Run date:   {args.run_date}")
    lines.append(f"Data file:  {data_file.name}")
    lines.append(fresh_line)
    if args.mode == "backtest":
        # A backtest is *supposed* to end in the past: models are scored on days
        # they never saw.  Calling that "stale" would report a working historical
        # evaluation as a production failure, so label it instead.
        lines.append("MODE: BACKTEST — historical evaluation on held-out days. "
                     "Models never saw the evaluation window; errors below are out-of-sample.")
    elif is_stale:
        lines.append(f"WARNING: data appears STALE (older than {args.stale_days} day(s)) — "
                     f"forecasts may be based on out-of-date inputs.")
    lines.append(f"Target: {args.target} | Cadence: {args.cadence} | Horizon: {args.horizon}")
    lines.append(f"Families requested: {', '.join(families)}")
    lines.append("")

    for s in summaries:
        lines.append(f"[{s['name']}]")
        if not s["ok"]:
            lines.append(f"  STATUS: no usable output ({'; '.join(s['notes']) or 'unknown'})")
            lines.append("")
            continue
        lines.append(f"  Models run: {s['models']}")
        if s["gate_passed"] is False:
            reasons = "; ".join(s["gate_reasons"])
            best_display = (f"WITHHELD — {reasons}, not usable; "
                            f"{s['best_model']} for diagnosis only")
        elif s["gate_passed"] is None:
            best_display = f"{s['best_model']} (integrity not verified)"
        else:
            best_display = s["best_model"]
        s["best_model_display"] = best_display
        lines.append(f"  Best model: {best_display}")
        lines.append(f"  Skill vs persistence: {s['skill_pct']}")
        lines.append(f"  Run status: {s['run_status']}")
        if s["gate_passed"] is True:
            lines.append("  Quality gate: PASSED")
        elif s["gate_passed"] is False:
            lines.append(f"  Quality gate: FAILED ({'; '.join(s['gate_reasons'])})")
        else:
            lines.append(f"  Quality gate: {NA}")

        leak_flag = "YES" if family_leakage_flag(s) else "none"
        lines.append(f"  Leakage flag: {leak_flag}")
        lines.append(f"    - pipeline: {s['pipe_leak']}")
        lines.append(f"    - summary check (origin_date >= target_date): {s['chk_leak']}")

        shift_flag = "YES" if family_shift_flag(s) else "none"
        lines.append(f"  Shift flag: {shift_flag}")
        for i, w in enumerate(s["pipe_shift"]):
            label = "pipeline" if i == 0 else "pipeline (cont.)"
            lines.append(f"    - {label}: {w}")
        lines.append(f"    - summary check (detect_lagged_copy): {s['chk_shift']}")
        lines.append("")

    lines.append("-" * 40)
    lines.append(f"Overall: {n_ok}/{len(families)} families produced output.")
    lines.append(f"Flags raised: {n_leak} leakage, {n_shift} shift, "
                 f"{n_quality} quality, "
                 f"data {'backtest window' if args.mode == 'backtest' else ('STALE' if is_stale else 'fresh')}.")

    report = "\n".join(lines) + "\n"
    (run_dir / "SUMMARY.txt").write_text(report)

    # Machine-readable twin of the text report, for downstream tooling.
    payload = {
        # run_id: SUMMARY.json previously carried no identifier of its own run, so a consumer
        # holding the file could not say which run produced it or join it to anything.
        "run_id": run_dir.name,
        "schema_version": 2,
        "run_date": args.run_date,
        "target": args.target,
        "cadence": args.cadence,
        "horizon": args.horizon,
        # data_file: review C1. SUMMARY.txt has printed `Data file: <name>` since it was
        # written; the JSON twin did not carry it, so two artifacts of the same run
        # disagreed about whether the input was knowable, and a consumer that reached for
        # `data_file` got None and rendered it. The name only -- the digest, row count and
        # date range belong to provenance.json (contract 7), and duplicating them here
        # would create a second place for them to drift.
        "data_file": data_file.name,
        # client_framing / model_composition: `model_reference.client_framing()` and
        # `composition()` existed and were tested, and nothing ever wrote them -- so no artifact
        # carried the composition and the Agent correctly reported "composition not recorded" on
        # every run. A derived sentence nobody publishes is not a contract field.
        #
        # Written as BOTH the prose sentence a client reads and the counts it was derived from, so
        # a consumer can requote the sentence or recompute from the numbers without re-deriving
        # the categories itself. Never a single headline count: the entries are not one kind of
        # thing (see reports/gate_audit.md §4).
        **_composition_fields(),
        "families": [
            {
                "name": s["name"],
                "ok": s["ok"],
                "models": s["models"],
                "best_model": s["best_model"],
                "best_model_display": s.get("best_model_display", s["best_model"]),
                "skill_pct": s["skill_pct"],
                "run_status": s["run_status"],
                "integrity_verified": s["integrity_found"],
                "gate_passed": s["gate_passed"],
                "gate_reasons": s["gate_reasons"],
                # X10: which source produced the verdict above, and what the family itself
                # published, so the two can never silently disagree again.
                "gate_source": s.get("gate_source"),
                "gate_published_by_family": s.get("gate_published"),
                # X11: a persistence baseline without prediction rows is not a result. Recorded
                # so a consumer can tell "baseline computed on nothing" from "baseline 0".
                "n_prediction_rows": s.get("n_prediction_rows"),
                "baseline_without_predictions": s.get("baseline_without_predictions"),
                "leakage_flag": family_leakage_flag(s),
                "shift_flag": family_shift_flag(s),
            }
            for s in summaries
        ],
        "overall": {
            "families_requested": len(families),
            "families_ok": n_ok,
            "families_gate_passed": sum(
                1 for s in summaries if s["gate_passed"] is True
            ),
            "leakage_flags": n_leak,
            "shift_flags": n_shift,
            "quality_gate_failures": n_quality,
        },
        "mode": args.mode,
        "freshness": {"line": fresh_line, "stale": bool(is_stale),
                      "backtest": args.mode == "backtest"},
    }
    (run_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2))

    print(report)

    # ── the contract gate ────────────────────────────────────────────────────────────────────
    # SUMMARY.json and the per-family tables are a published interface, read by the AI4CM Agent.
    # Validate the files that were just written, before anything downstream consumes them. This
    # reads the ARTIFACTS -- the existing contract tests grep the writer's source, which is why no
    # SUMMARY.json on disk carries schema_version even though every such test passes.
    if not args.no_validate:
        from artifact_validation import validate_run
        vrep = validate_run(run_dir, strict=args.strict_validate,
                            published_root=args.published_root)
        print("\n" + "-" * 40)
        print("ARTIFACT CONTRACT")
        print(vrep.summary())
        if not vrep.ok:
            print("\nERROR: artifacts failed the contract and must not be published. "
                  "Fix the writer, not the validator.", file=sys.stderr)
            return 2

    if n_ok < len(families):
        print("ERROR: one or more requested families produced no output.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())