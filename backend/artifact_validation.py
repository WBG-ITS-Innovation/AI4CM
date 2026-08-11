"""Validate the Agent-facing artifacts before they are published. Fail loudly, never silently.

`SUMMARY.json`, the per-family `leaderboard.csv` / `predictions_long.csv` / `metrics_long.csv`, and
`forecasts/published/<issue>/` are a **published interface**: the AI4CM Agent reads them, and a
malformed field there is not an internal detail, it is a wrong answer given to someone.

--------------------------------------------------------------------------------
THREE KINDS OF DEFECT, DELIBERATELY SEPARATED
--------------------------------------------------------------------------------
**MALFORMED** — the artifact cannot be parsed as what it claims to be: a missing file, an
unreadable CSV, a required column absent, a number that is a string.

**INCOMPLETE** — parseable, but a field that should carry a value does not, and nothing says why.
This is the one that matters most on this project, because *absence is meaningful here*: a point
model legitimately reports no coverage, and a family that never ran a check legitimately reports no
verdict. So "incomplete" is reserved for absence that is **not** accounted for by an explicit
"not applicable" marker.

**INCONSISTENT** — every field parses and is present, and they contradict each other. A coverage
value with no nominal level; a skill figure that does not reconcile with its own logged MAEs; a
champion naming a model no pool contains. These are the dangerous ones: each individual number
looks fine, so no per-field check finds them.

--------------------------------------------------------------------------------
WHY THIS RUNS AGAINST ARTIFACTS AND NOT AGAINST SOURCE
--------------------------------------------------------------------------------
`backend/tests/test_artifact_contract.py` asserts contract clauses by grepping the *writer's
source* -- e.g. that ``'"schema_version"'`` appears in `daily_summary.py`. Every such test passes
today, and yet **no `SUMMARY.json` on disk carries `schema_version` or `run_id`**, because the
committed runs predate the writer that emits them. A source test cannot notice that. This module
reads the files.

--------------------------------------------------------------------------------
SEVERITY
--------------------------------------------------------------------------------
``ERROR`` blocks publication. ``WARNING`` does not, and is used for defects that are real but
historical -- an old artifact missing a field added later is a fact about that artifact, not a
reason to refuse to read it. ``validate_run(..., strict=True)`` promotes warnings to errors, which
is what a *new* run should be held to.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

ERROR = "ERROR"
WARNING = "WARNING"

#: Absence that is accounted for. A consumer seeing one of these knows the field was *not
#: applicable*, as opposed to a field that simply failed to appear.
NOT_APPLICABLE_MARKERS: Sequence[str] = (
    "n/a (not produced)",       # daily_summary.NA
    "not reported",             # frontend/format_gel.NOT_REPORTED
)

#: `predictions_long.csv` must let a consumer place every row in time and attribute it to a model.
PREDICTIONS_REQUIRED: Sequence[str] = ("origin_date", "target_date", "y_true", "model")

#: A leaderboard must at minimum name a model and score it.
LEADERBOARD_REQUIRED: Sequence[str] = ("model",)

#: Tolerance for reconciling a published skill percentage against its own logged MAEs.
#: Generous: the point is to catch a figure computed against a *different* baseline, which is off
#: by percentage points, not by rounding.
SKILL_RECONCILE_TOLERANCE_PCT = 0.75


@dataclass
class Finding:
    severity: str
    artifact: str
    kind: str                    # "malformed" | "incomplete" | "inconsistent"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.artifact} ({self.kind}): {self.message}"


@dataclass
class ValidationReport:
    findings: List[Finding] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)

    def add(self, severity: str, artifact: str, kind: str, message: str) -> None:
        self.findings.append(Finding(severity, artifact, kind, message))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_kind(self, kind: str) -> List[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def summary(self) -> str:
        if not self.findings:
            return f"OK -- {len(self.checked)} artifact(s) validated, no findings."
        head = (f"{len(self.errors)} error(s), {len(self.warnings)} warning(s) across "
                f"{len(self.checked)} artifact(s):")
        return "\n".join([head] + [f"  {f}" for f in self.findings])

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ArtifactContractError(self.summary())


class ArtifactContractError(RuntimeError):
    """One or more artifacts are unfit to publish."""


def _is_number(x) -> bool:
    if isinstance(x, bool) or x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _is_not_applicable(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in {m.lower()
                                                        for m in NOT_APPLICABLE_MARKERS}


def _pct(x) -> Optional[float]:
    """Read a skill figure that may be a number or a formatted string like ``"27.51%"``."""
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip().rstrip("%").strip()
        if _is_number(s):
            return float(s)
    return None


# ── SUMMARY.json ──────────────────────────────────────────────────────────────

def validate_summary(path: Path, rep: ValidationReport, strict: bool = False) -> None:
    art = "SUMMARY.json"
    sev = ERROR if strict else WARNING
    path = Path(path)
    if not path.exists():
        rep.add(ERROR, art, "malformed", f"missing at {path}")
        return
    rep.checked.append(art)
    try:
        d = json.loads(path.read_text())
    except Exception as exc:                       # noqa: BLE001
        rep.add(ERROR, art, "malformed", f"not valid JSON: {exc}")
        return
    if not isinstance(d, dict):
        rep.add(ERROR, art, "malformed", f"top level is {type(d).__name__}, expected object")
        return

    for key in ("run_date", "target", "cadence", "horizon", "families"):
        if key not in d:
            rep.add(ERROR, art, "malformed", f"required key {key!r} absent")

    # Identity. Without these a consumer holding the file cannot say which run produced it, so
    # their absence is a real defect -- but on a historical artifact it is a fact, not a blocker.
    for key in ("run_id", "schema_version"):
        if key not in d:
            rep.add(sev, art, "incomplete",
                    f"{key!r} absent -- a consumer cannot identify which run this is. Written by "
                    f"the current daily_summary.py, so this artifact predates it.")

    fams = d.get("families")
    if not isinstance(fams, list):
        if "families" in d:
            rep.add(ERROR, art, "malformed",
                    f"'families' is {type(fams).__name__}, expected list")
        return
    if not fams:
        rep.add(ERROR, art, "incomplete", "'families' is empty -- a run with no family is not a run")

    names: List[str] = []
    for i, f in enumerate(fams):
        where = f"families[{i}]"
        if not isinstance(f, dict):
            rep.add(ERROR, art, "malformed", f"{where} is {type(f).__name__}, expected object")
            continue
        name = f.get("name") or where
        names.append(str(name))
        for key in ("name", "ok", "run_status", "gate_passed", "gate_reasons"):
            if key not in f:
                rep.add(ERROR, art, "malformed", f"{name}: required key {key!r} absent")

        # gate_passed is TRI-state and the None case must survive the round trip: "never
        # verified" must never be indistinguishable from "failed".
        gp = f.get("gate_passed", "MISSING")
        if gp not in (True, False, None, "MISSING"):
            rep.add(ERROR, art, "malformed",
                    f"{name}: gate_passed is {gp!r}; must be true, false or null (tri-state)")

        # A failure with no reason is unactionable.
        if gp is False and not (f.get("gate_reasons") or []):
            rep.add(ERROR, art, "inconsistent",
                    f"{name}: gate_passed is false with an empty gate_reasons -- a withheld "
                    f"family must say why")
        if gp is True and (f.get("gate_reasons") or []):
            rep.add(ERROR, art, "inconsistent",
                    f"{name}: gate_passed is true but gate_reasons is non-empty "
                    f"({f['gate_reasons']}) -- the verdict and its reasons disagree")

        # skill_pct is either a number, a formatted percentage, or an explicit not-applicable.
        sp = f.get("skill_pct", "MISSING")
        if sp == "MISSING":
            rep.add(sev, art, "incomplete", f"{name}: skill_pct absent")
        elif _pct(sp) is None and not _is_not_applicable(sp):
            rep.add(ERROR, art, "malformed",
                    f"{name}: skill_pct is {sp!r} -- neither a number, a percentage string, nor "
                    f"one of the not-applicable markers {list(NOT_APPLICABLE_MARKERS)}")

        # X11: a baseline computed over zero rows is not a baseline.
        if f.get("baseline_without_predictions") is True and _pct(sp) is not None:
            rep.add(ERROR, art, "inconsistent",
                    f"{name}: reports skill {sp!r} while baseline_without_predictions is true -- "
                    f"a skill figure over zero prediction rows is not a measurement")

        npr = f.get("n_prediction_rows")
        if _is_number(npr) and float(npr) == 0 and _pct(sp) is not None:
            rep.add(ERROR, art, "inconsistent",
                    f"{name}: reports skill {sp!r} with n_prediction_rows=0")

    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        rep.add(ERROR, art, "inconsistent",
                f"duplicate family names {sorted(dupes)} -- a consumer keying by name loses rows")

    _cross_check_overall(d, rep, art)


def _cross_check_overall(d: Dict, rep: ValidationReport, art: str) -> None:
    """`overall` is a derived block, so it can contradict what it derives from."""
    ov = d.get("overall")
    if not isinstance(ov, dict):
        return
    fams = [f for f in d.get("families", []) if isinstance(f, dict)]
    checks = (
        ("families_ok", sum(1 for f in fams if f.get("ok") is True)),
        ("families_gate_passed", sum(1 for f in fams if f.get("gate_passed") is True)),
        ("leakage_flags", sum(1 for f in fams if f.get("leakage_flag") is True)),
        ("shift_flags", sum(1 for f in fams if f.get("shift_flag") is True)),
        ("quality_gate_failures", sum(1 for f in fams if f.get("gate_passed") is False)),
    )
    for key, recomputed in checks:
        if key in ov and _is_number(ov[key]) and int(ov[key]) != recomputed:
            rep.add(ERROR, art, "inconsistent",
                    f"overall.{key} is {ov[key]} but the families list gives {recomputed}")
    if ("families_requested" in ov and _is_number(ov["families_requested"])
            and int(ov["families_requested"]) < len(fams)):
        rep.add(ERROR, art, "inconsistent",
                f"overall.families_requested is {ov['families_requested']} but "
                f"{len(fams)} families are listed")


# ── the tabular artifacts ─────────────────────────────────────────────────────

def _read_csv(path: Path, art: str, rep: ValidationReport) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as exc:                       # noqa: BLE001
        rep.add(ERROR, art, "malformed", f"unreadable CSV: {exc}")
        return None


def validate_family_tables(family_dir: Path, rep: ValidationReport,
                           strict: bool = False) -> None:
    """`leaderboard.csv`, `predictions_long.csv`, `metrics_long.csv` for one family."""
    family_dir = Path(family_dir)
    fam = family_dir.name
    sev = ERROR if strict else WARNING

    lb_path = family_dir / "leaderboard.csv"
    pr_path = family_dir / "predictions_long.csv"

    lb = pr = None
    if lb_path.exists():
        art = f"{fam}/leaderboard.csv"
        rep.checked.append(art)
        lb = _read_csv(lb_path, art, rep)
        if lb is not None:
            if lb.empty:
                rep.add(ERROR, art, "incomplete", "no rows -- a leaderboard with no model")
            for col in LEADERBOARD_REQUIRED:
                if col not in lb.columns:
                    rep.add(ERROR, art, "malformed", f"required column {col!r} absent")
            if "model" in lb.columns:
                if lb["model"].isna().any():
                    rep.add(ERROR, art, "malformed",
                            f"{int(lb['model'].isna().sum())} row(s) with no model name")
                # Identity columns present on SOME rows only. A_STAT's leaderboard does this:
                # target/horizon/cadence are populated on the baseline row and blank on the
                # winner, so "which model won for target X" is unanswerable from the file.
                for col in ("target", "horizon", "cadence"):
                    if col in lb.columns:
                        n = int(lb[col].notna().sum())
                        if 0 < n < len(lb):
                            rep.add(ERROR, art, "inconsistent",
                                    f"{col!r} is populated on {n} of {len(lb)} rows -- a "
                                    f"partially-identified leaderboard cannot be keyed by "
                                    f"{col}; rows: "
                                    f"{lb.loc[lb[col].isna(), 'model'].tolist()[:4]}")
                for col in lb.columns:
                    if col != "model" and lb[col].notna().sum() == 0:
                        rep.add(sev, art, "incomplete",
                                f"column {col!r} is entirely empty -- absent-because-not-computed "
                                f"is indistinguishable from absent-because-failed")

    if pr_path.exists():
        art = f"{fam}/predictions_long.csv"
        rep.checked.append(art)
        pr = _read_csv(pr_path, art, rep)
        if pr is not None:
            if pr.empty:
                rep.add(ERROR, art, "incomplete", "no prediction rows")
            for col in PREDICTIONS_REQUIRED:
                if col not in pr.columns:
                    rep.add(ERROR, art, "malformed", f"required column {col!r} absent")
            if {"origin_date", "target_date"}.issubset(pr.columns):
                od = pd.to_datetime(pr["origin_date"], errors="coerce")
                td = pd.to_datetime(pr["target_date"], errors="coerce")
                bad = int((od >= td).sum())
                if bad:
                    rep.add(ERROR, art, "inconsistent",
                            f"{bad} row(s) with origin_date >= target_date -- the model would "
                            f"have been predicting a date it could already see")
            if "y_pred" in pr.columns and pr["y_pred"].notna().sum() == 0:
                rep.add(ERROR, art, "incomplete", "y_pred is entirely empty")

    # Cross-artifact: every leaderboard model must be joinable to the predictions.
    if lb is not None and pr is not None and "model" in lb.columns and "model" in pr.columns:
        art = f"{fam}/leaderboard.csv+predictions_long.csv"
        rep.checked.append(art)
        missing = sorted(set(lb["model"].dropna()) - set(pr["model"].dropna()))
        if missing:
            rep.add(sev, art, "inconsistent",
                    f"model(s) scored in the leaderboard with no rows in predictions_long: "
                    f"{missing} -- a consumer joining the two silently loses them")
        non_ascii = sorted(m for m in set(lb["model"].dropna()) if not str(m).isascii())
        if non_ascii:
            rep.add(sev, art, "inconsistent",
                    f"decorated model name(s) used as a join key: {non_ascii} -- a display "
                    f"character in an identifier breaks joins against artifacts written plainly")

    mx_path = family_dir / "metrics_long.csv"
    if mx_path.exists():
        art = f"{fam}/metrics_long.csv"
        rep.checked.append(art)
        mx = _read_csv(mx_path, art, rep)
        if mx is not None:
            if mx.empty:
                rep.add(ERROR, art, "incomplete", "no metric rows")
            elif {"metric", "value"}.issubset(mx.columns):
                _validate_long_metrics(mx, art, rep, sev)
            else:
                _validate_wide_metrics(mx, art, rep, sev)


def _validate_long_metrics(mx: pd.DataFrame, art: str, rep: ValidationReport,
                           sev: str) -> None:
    """The key/value shape (E_QUANTILE). Coverage rows must carry their nominal level."""
    if "model" not in mx.columns:
        rep.add(ERROR, art, "malformed", "long-form metrics with no 'model' column")
    cov = mx[mx["metric"].astype(str).str.startswith("coverage")]
    if len(cov) and "quantile" in mx.columns and cov["quantile"].isna().all():
        rep.add(sev, art, "inconsistent",
                f"{len(cov)} coverage row(s) carry no nominal level (the 'quantile' column is "
                f"blank on every one), so the level is only inferable from the metric NAME -- "
                f"see e_quantile_daily_pipeline.emit_coverage")


def _validate_wide_metrics(mx: pd.DataFrame, art: str, rep: ValidationReport,
                           sev: str) -> None:
    """The one-column-per-metric shape (A_STAT, B_ML)."""
    if "model" not in mx.columns:
        rep.add(ERROR, art, "malformed", "wide metrics with no 'model' column")
    for col in mx.columns:
        if col in ("target", "horizon", "cadence", "model"):
            continue
        if mx[col].notna().sum() == 0:
            rep.add(sev, art, "incomplete",
                    f"metric column {col!r} is entirely empty -- a consumer cannot tell "
                    f"'not applicable to these models' from 'computation failed'")
    # The interval's advertised level lives in a column NAME here (PI_coverage@90), the same
    # defect audit field #2 fixed for E_QUANTILE.
    named = [c for c in mx.columns if "coverage@" in str(c) or "coverage_p" in str(c)]
    if named and not any(str(c).startswith("coverage_nominal") for c in mx.columns):
        rep.add(sev, art, "inconsistent",
                f"coverage column(s) {named} carry their nominal level in the column NAME with no "
                f"'coverage_nominal' field beside them -- nothing ties the name to the level "
                f"actually used")


# ── coverage / skill consistency, wherever they appear ────────────────────────

def validate_coverage_block(obj: Dict, artifact: str, rep: ValidationReport) -> None:
    """A coverage number must travel with the level it describes.

    Reads the level as data. A `coverage_p*` value with no `coverage_nominal` beside it is exactly
    audit field #2: the consumer is left inferring the nominal level from a key name that nothing
    guarantees still matches the fitted alphas.
    """
    cov_keys = [k for k in obj if str(k).startswith("coverage_p")]
    has_value = any(_is_number(obj[k]) for k in cov_keys)
    if not has_value:
        return
    if not _is_number(obj.get("coverage_nominal")):
        rep.add(ERROR, artifact, "inconsistent",
                f"coverage value present ({', '.join(cov_keys)}) with no numeric "
                f"'coverage_nominal' -- the level is only inferable from the key name")
        return

    nominal = float(obj["coverage_nominal"])
    lo, hi = obj.get("coverage_lower_quantile"), obj.get("coverage_upper_quantile")
    if _is_number(lo) and _is_number(hi):
        implied = round(float(hi) - float(lo), 10)
        if abs(implied - nominal) > 1e-9:
            rep.add(ERROR, artifact, "inconsistent",
                    f"coverage_nominal is {nominal} but the recorded quantiles {lo}/{hi} imply "
                    f"{implied}")
        for k in cov_keys:
            expected = f"coverage_p{int(round(float(lo) * 100))}_p{int(round(float(hi) * 100))}"
            if k != expected and k != "coverage_p10_p90":
                rep.add(ERROR, artifact, "inconsistent",
                        f"coverage key {k!r} does not name the recorded quantiles {lo}/{hi} "
                        f"(which would be {expected!r})")
    for k in cov_keys:
        v = obj[k]
        if _is_number(v) and not (0.0 <= float(v) <= 1.0):
            rep.add(ERROR, artifact, "malformed",
                    f"{k} is {v} -- a coverage proportion must lie in [0, 1]")


def validate_skill_reconciles(obj: Dict, artifact: str, rep: ValidationReport,
                              tolerance_pct: float = SKILL_RECONCILE_TOLERANCE_PCT) -> None:
    """A published skill figure must follow from the MAEs published beside it.

    ``skill = (mae_persistence - mae_model) / mae_persistence * 100``. This is the check that
    would have caught the WS2 harness bug, where a non-canonical ruler inflated every logged skill
    while both MAEs sat in the same record disagreeing with it.
    """
    skill = _pct(obj.get("skill_pct"))
    if skill is None:
        return
    base = obj.get("mae_persistence")
    model = None
    for key in ("mae_model", "mae_p50", "mae", "MAE"):
        if _is_number(obj.get(key)):
            model = float(obj[key])
            break
    if not _is_number(base) or model is None:
        return
    base = float(base)
    if base <= 0:
        rep.add(ERROR, artifact, "inconsistent",
                f"mae_persistence is {base} -- a skill figure cannot be derived from a "
                f"non-positive baseline")
        return
    implied = (base - model) / base * 100.0
    if abs(implied - skill) > tolerance_pct:
        rep.add(ERROR, artifact, "inconsistent",
                f"skill_pct is {skill:.2f}% but the MAEs beside it "
                f"(model {model:,.2f} vs persistence {base:,.2f}) imply {implied:.2f}% -- "
                f"a gap of {abs(implied - skill):.2f}pp means the figure was computed against a "
                f"different baseline")


def validate_champion_is_in_the_pool(name: Optional[str], artifact: str,
                                     rep: ValidationReport,
                                     pool: Optional[Sequence[str]] = None) -> None:
    """A champion naming a model no pipeline offers cannot be reproduced or explained."""
    if not name:
        return
    clean = str(name).split(" (")[0].strip()
    clean = "".join(ch for ch in clean if ch.isascii()).strip()
    if not clean or "baseline" in clean.lower() or "persistence" in clean.lower():
        return
    if pool is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from model_reference import model_pool
            pool = list(model_pool())
        except Exception as exc:                   # noqa: BLE001
            rep.add(WARNING, artifact, "incomplete",
                    f"could not load the model pool to validate champion {clean!r}: {exc}")
            return
    if clean not in set(pool):
        rep.add(ERROR, artifact, "inconsistent",
                f"champion {clean!r} is not in the model pool ({len(pool)} models). A consumer "
                f"cannot look up what it is, and it cannot be refitted.")


# ── published forecast directory ──────────────────────────────────────────────

def validate_published_issue(issue_dir: Path, rep: ValidationReport,
                             strict: bool = False) -> None:
    """`forecasts/published/<issue_date>/` — the immutable record the Agent reads."""
    issue_dir = Path(issue_dir)
    art = f"published/{issue_dir.name}"
    sev = ERROR if strict else WARNING
    rep.checked.append(art)

    fc_path = issue_dir / "forecast.csv"
    if not fc_path.exists():
        rep.add(ERROR, art, "malformed", "forecast.csv absent -- this is not a published issue")
        return
    fc = _read_csv(fc_path, art, rep)
    if fc is None:
        return
    for col in ("target", "horizon", "target_date", "origin_date", "p10", "p50", "p90"):
        if col not in fc.columns:
            rep.add(ERROR, art, "malformed", f"forecast.csv: required column {col!r} absent")
    if "y_true" in fc.columns:
        rep.add(ERROR, art, "inconsistent",
                "forecast.csv carries a y_true column -- a forecast issued before its truth "
                "existed cannot have one")
    if {"p10", "p50", "p90"}.issubset(fc.columns):
        crossed = int(((fc["p10"] > fc["p50"]) | (fc["p50"] > fc["p90"])).sum())
        if crossed:
            rep.add(ERROR, art, "inconsistent",
                    f"{crossed} row(s) with crossed quantiles (p10 > p50 or p50 > p90)")

    for name in ("provenance.json", "manifest.json", "gates.json"):
        p = issue_dir / name
        if not p.exists():
            rep.add(ERROR, art, "incomplete", f"{name} absent")
            continue
        try:
            json.loads(p.read_text())
        except Exception as exc:                   # noqa: BLE001
            rep.add(ERROR, art, "malformed", f"{name} is not valid JSON: {exc}")

    # Estimators: absent is legitimate (an issue published before retention existed, or one
    # pruned), but it must be DISTINGUISHABLE from a manifest that lists blobs which are gone.
    est_manifest = issue_dir / "estimators" / "manifest.json"
    if not est_manifest.exists():
        rep.add(sev, art, "incomplete",
                "no estimators/manifest.json -- this issue cannot be re-derived. Legitimate for "
                "an issue published before retention existed; a consumer sees EstimatorMissing.")
    else:
        try:
            m = json.loads(est_manifest.read_text())
        except Exception as exc:                   # noqa: BLE001
            rep.add(ERROR, art, "malformed", f"estimators/manifest.json invalid: {exc}")
            return
        pruned = bool((m.get("retention") or {}).get("pruned"))
        gone = [e["file"] for e in m.get("estimators", [])
                if not (issue_dir / "estimators" / e["file"]).exists()]
        if gone and not pruned:
            rep.add(ERROR, art, "inconsistent",
                    f"{len(gone)} estimator blob(s) listed in the manifest are missing from disk "
                    f"and retention.pruned is not set -- absence is unexplained (e.g. {gone[:3]})")
        for e in m.get("estimators", []):
            if not e.get("sha256"):
                rep.add(ERROR, art, "incomplete",
                        f"estimator {e.get('fit_id')} has no sha256 -- it cannot be verified")


# ── the entry point ───────────────────────────────────────────────────────────

def validate_run(run_dir: Path, strict: bool = False,
                 published_root: Optional[Path] = None) -> ValidationReport:
    """Validate everything the Agent reads for one run. Returns the report; never raises.

    Call ``.raise_if_invalid()`` to block publication, which is what the pipeline does.
    """
    run_dir = Path(run_dir)
    rep = ValidationReport()

    validate_summary(run_dir / "SUMMARY.json", rep, strict=strict)

    summary_path = run_dir / "SUMMARY.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            for f in d.get("families", []):
                if isinstance(f, dict):
                    label = f"SUMMARY.json:{f.get('name')}"
                    validate_coverage_block(f, label, rep)
                    validate_skill_reconciles(f, label, rep)
                    validate_champion_is_in_the_pool(f.get("best_model"), label, rep)
        except Exception:                          # noqa: BLE001 - already reported above
            pass

    for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if (child / "leaderboard.csv").exists() or (child / "predictions_long.csv").exists():
            validate_family_tables(child, rep, strict=strict)

    if published_root is not None:
        root = Path(published_root)
        if root.exists():
            for issue in sorted(p for p in root.iterdir() if p.is_dir()):
                if (issue / "forecast.csv").exists():
                    validate_published_issue(issue, rep, strict=strict)

    return rep


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate the Agent-facing artifacts of a run.")
    ap.add_argument("run_dir")
    ap.add_argument("--strict", action="store_true",
                    help="promote warnings to errors; what a NEW run should be held to")
    ap.add_argument("--published-root", default=None)
    a = ap.parse_args()

    rep = validate_run(Path(a.run_dir), strict=a.strict,
                       published_root=Path(a.published_root) if a.published_root else None)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
