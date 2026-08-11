"""Persist the fitted estimators behind a published forecast, so an issue can be re-derived.

This exists to answer one question: *"what exactly produced the number you published on this
date?"* Re-running the pipeline today answers a different question, because the data has grown and
the code has moved. Official runs still refit on current data — nothing here serves a stale model.

--------------------------------------------------------------------------------
WHY THE BLOBS ARE NOT TRACKED IN GIT, UNLIKE EVERY OTHER PUBLISHED FILE
--------------------------------------------------------------------------------
``forecasts/published/`` is deliberately un-ignored in ``.gitignore``: a published forecast is the
only record of what was actually said, so it must survive a clone. **This directory is the one
exception**, and the reason is not disk.

A fitted estimator embeds client data:

* ``ScaledRegressor.y_train_ref_`` is the full training target -- measured on the Revenues
  champion, 2406 raw Treasury figures in GEL;
* ``ScaledRegressor.level`` is the full trailing-level series -- 2763 points;
* and every tree ensemble's **split thresholds and leaf values are functions of the training
  data**, which cannot be scrubbed at all without destroying the model.

The first two are removed here (see ``scrub_for_persistence``). The third is irreducible. So a
persisted model is *not* a data-free object, and committing one would put raw Treasury figures
permanently into every clone of a World Bank repository -- unremovable without rewriting history.

The resolution keeps the audit trail without the data: **the manifest is tracked, the blobs are
not.** ``manifest.json`` carries a SHA-256 per blob, so a clone can still prove which estimator
produced a published number and detect a substituted one. What a clone cannot do is *run* it. That
is the intended trade: the claim stays auditable forever, the client data stays local.

A pruned or never-retained estimator therefore reads as an explicit **"not available"**
(``EstimatorMissing``) rather than as silence.

--------------------------------------------------------------------------------
FORMAT, AND WHY PICKLE IS ACCEPTABLE HERE
--------------------------------------------------------------------------------
joblib at ``compress=3``. It is the only format that round-trips every estimator in the 13-model
pool *plus* this project's own ``ScaledRegressor`` wrapper, exactly and without changing the
numbers. ONNX has no converter for the wrapper and computes in float32, which would defeat a
reproduce-to-tolerance test; LightGBM's native dump covers the booster only, cannot represent
``HistGBDT_L1`` at all, and measured on a comparable fit is not even smaller (109 KB gzipped
against 116 KB for the pickle); ``skops`` has no lightgbm or catboost support.

Unpickling executes code, so **the digest is verified before the bytes are ever handed to
joblib**. That reduces the exposure from "arbitrary code from any file on disk" to "arbitrary code
only if the attacker can already write to our git history".
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Subdirectory of a published issue holding the estimators.
ESTIMATOR_DIR = "estimators"
MANIFEST_NAME = "manifest.json"

SCHEMA_VERSION = 1

#: Packages whose version changes can change a prediction. Checked at load time against the
#: manifest, because sklearn itself only *warns* on a version mismatch (see ``load_estimator``).
PINNED_PACKAGES: Sequence[str] = ("scikit-learn", "lightgbm", "numpy", "joblib")

#: How many issues keep their blobs. Older ones are pruned, and say so in their manifest.
#: Any issue with unscored horizons is protected regardless of age -- see ``prune_estimators``.
DEFAULT_KEEP_LAST = 8

RETENTION_NOTE = (
    "Blobs are NOT tracked in git; this manifest is. A fitted estimator embeds training data "
    "(and tree split thresholds, which cannot be scrubbed), so committing one would place raw "
    "Treasury figures permanently in every clone. The tracked SHA-256 still proves which "
    "estimator produced a published number and detects substitution. This deliberately diverges "
    "from the retention rule for forecast.csv, which IS the claim and must survive a clone; an "
    "estimator only re-derives it."
)


class EstimatorMissing(FileNotFoundError):
    """The estimator was never retained, or was pruned. Distinct from corrupt."""


class EstimatorDigestMismatch(RuntimeError):
    """The blob on disk is not the blob that was published. Never overridable."""


class ReproductionUnavailable(RuntimeError):
    """The estimator loaded, but the published number cannot honestly be re-derived.

    Almost always because the canonical data file has moved on: a feature the model was fitted on
    can no longer be built, or the origin row now has gaps. Predicting anyway would return a
    number that looks like the published one and is not.
    """


class EstimatorVersionMismatch(RuntimeError):
    """A pinned library differs from the one that wrote the blob.

    Raised by this project rather than left to the library, because scikit-learn only emits an
    ``InconsistentVersionWarning`` and loads anyway -- measured: a state stamped 1.3.2 loads under
    1.8.0 and predicts happily. A warning is invisible in a CLI or Streamlit run, so a
    wrong-version load would silently return different numbers under a published label.
    """


# ── scrubbing ─────────────────────────────────────────────────────────────────

def scrub_for_persistence(est, keep_index: pd.Index):
    """Return a copy of ``est`` with training data removed, without changing what it predicts.

    ``keep_index`` is the set of rows the persisted model must still be able to predict -- for a
    published issue, the single forecast origin.

    Two attributes of ``ScaledRegressor`` hold data that the prediction path does not need in
    full:

    ``y_train_ref_``
        The full training target. It is consumed only by ``sanity_check_prediction_scale``, which
        reduces it to ``nanmedian(abs(y))``. Replacing the array with a one-element array holding
        exactly that median leaves the check arithmetically identical. It is *replaced* rather
        than deleted because ``predict`` references the attribute unconditionally.

    ``level``
        The trailing-level divisor. ``_rows_level`` only ever does ``level.reindex(X.index)``, so
        restricting it to ``keep_index`` is equivalent for those rows -- and for any other row it
        raises (``strict`` defaults to True) instead of silently forward-filling, which is the
        correct behaviour for a replay artifact.

    Equivalence is proved by test, not asserted here: see ``test_estimator_store.py``.
    """
    out = copy.deepcopy(est)
    disclosure: Dict[str, str] = {}

    ref = getattr(out, "y_train_ref_", None)
    if ref is not None:
        arr = np.asarray(ref, dtype=float)
        med = float(np.nanmedian(np.abs(arr)))
        out.y_train_ref_ = np.array([med], dtype=float)
        disclosure["y_train_ref_"] = (
            f"full training target ({arr.size} values) replaced by a single number, "
            f"median(|y|) = {med:.6g}, which is all the scale check consumes")

    lvl = getattr(out, "level", None)
    if isinstance(lvl, pd.Series):
        out.level = lvl.reindex(keep_index)
        disclosure["level"] = (
            f"trailing-level series reduced from {len(lvl)} points to {len(out.level)} "
            f"(only the rows this artifact must predict)")

    return out, disclosure


def _classes(est) -> Dict[str, Optional[str]]:
    """Outer and inner estimator class, so the manifest names what is inside the blob."""
    inner = getattr(est, "estimator_", None) or getattr(est, "base", None)
    cls = type(est)
    return {
        "estimator_class": f"{cls.__module__}.{cls.__qualname__}",
        "inner_class": (f"{type(inner).__module__}.{type(inner).__qualname__}"
                        if inner is not None else None),
        "library": (type(inner) if inner is not None else cls).__module__.split(".")[0],
    }


# ── writing ───────────────────────────────────────────────────────────────────

@dataclass
class FittedEstimator:
    """One estimator, with everything needed to identify the fit that produced it."""

    target: str
    horizon: int
    kind: str                    # "point" | "q10" | "q50" | "q90"
    estimator: object
    feature_names: Sequence[str]
    n_train_rows: int
    origin_date: str
    target_transform: str = "raw"
    recipe_id: str = ""
    selection_run_id: Optional[str] = None
    #: The recipe's feature groups, recorded so the design matrix can be rebuilt at load time
    #: without trusting the registry not to have moved on.
    fiscal_groups: Sequence[str] = ()
    exog_blocks: Sequence[str] = ()

    @property
    def slug(self) -> str:
        return _slug(self.target)

    def fit_id(self, issue_date: str) -> str:
        """Identity of *this* fit.

        Keyed on the issue date rather than the origin, because a same-day re-issue takes a
        suffixed date (``-r2``) and the two issues are different fits that must not share an id.

        Deliberately not a logged ``run_id``: the forward path writes no row to
        ``experiments/log.csv``, and the ``run_id`` on a recipe identifies the DEV selection fit
        (TRAIN <=2023, scored on DEV 2024) -- a different fit on a different window. That id
        travels as ``selection_run_id`` so it cannot be misread as this one.
        """
        return f"{issue_date}/{self.slug}/h{int(self.horizon)}/{self.kind}"

    @property
    def filename(self) -> str:
        return f"{self.slug}/h{int(self.horizon)}_{self.kind}.joblib"


def _slug(target: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in target.strip().lower()).strip("_")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_estimators(issue_dir: Path, fitted: Sequence[FittedEstimator],
                    keep_index: pd.Index,
                    provenance: Optional[Dict] = None,
                    compress: int = 3) -> Path:
    """Scrub, write and digest every estimator behind one published issue.

    Returns the manifest path. The manifest is the tracked artifact; the blobs are not.
    """
    import joblib

    from provenance import describe_environment

    est_dir = Path(issue_dir) / ESTIMATOR_DIR
    est_dir.mkdir(parents=True, exist_ok=True)
    issue_date = Path(issue_dir).name

    entries: List[Dict] = []
    for f in fitted:
        scrubbed, disclosure = scrub_for_persistence(f.estimator, keep_index)
        dest = est_dir / f.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scrubbed, dest, compress=compress)
        entry = {
            "fit_id": f.fit_id(issue_date),
            "target": f.target,
            "horizon": int(f.horizon),
            "kind": f.kind,
            "file": f.filename,
            "sha256": _sha256(dest),
            "bytes": dest.stat().st_size,
            "recipe_id": f.recipe_id,
            "selection_run_id": f.selection_run_id,
            "selection_run_id_note": (
                "The run that SELECTED this recipe on DEV 2024, not the fit in this blob. The "
                "published fit uses all history through the issue date and has no logged run_id."),
            "target_transform": f.target_transform,
            "n_train_rows": int(f.n_train_rows),
            "n_features": len(f.feature_names),
            "feature_names": list(f.feature_names),
            "fiscal_groups": list(f.fiscal_groups),
            "exog_blocks": list(f.exog_blocks),
            "origin_date": f.origin_date,
            "scrubbed": disclosure,
        }
        entry.update(_classes(scrubbed))
        entries.append(entry)

    env = describe_environment()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue_date": Path(issue_dir).name,
        "environment": env,
        "pinned_packages": list(PINNED_PACKAGES),
        "retention": {
            "blobs_tracked_in_git": False,
            "keep_last_issues": DEFAULT_KEEP_LAST,
            "pruned": False,
            "note": RETENTION_NOTE,
        },
        "total_bytes": sum(e["bytes"] for e in entries),
        "estimators": entries,
        "data_sha256": (provenance or {}).get("data", {}).get("sha256"),
        "git_sha": (provenance or {}).get("code", {}).get("git_sha"),
        "note": ("Fitted estimators behind this issue, for re-deriving what was published. "
                 "Official runs refit on current data; nothing here serves a stale model."),
    }
    mpath = est_dir / MANIFEST_NAME
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return mpath


# ── reading ───────────────────────────────────────────────────────────────────

def read_manifest(issue_dir: Path) -> Dict:
    p = Path(issue_dir) / ESTIMATOR_DIR / MANIFEST_NAME
    if not p.exists():
        raise EstimatorMissing(
            f"no estimator manifest at {p}. This issue was published without retaining its "
            f"estimators, so it cannot be re-derived.")
    return json.loads(p.read_text())


def _find(manifest: Dict, target: str, horizon: int, kind: str) -> Dict:
    for e in manifest.get("estimators", []):
        if (e["target"] == target and int(e["horizon"]) == int(horizon)
                and e["kind"] == kind):
            return e
    have = sorted({(e["target"], e["horizon"], e["kind"])
                   for e in manifest.get("estimators", [])})
    raise EstimatorMissing(
        f"no estimator for target={target!r} horizon={horizon} kind={kind!r}. "
        f"This issue retained {len(have)} estimators.")


def check_versions(manifest: Dict) -> Dict[str, Dict[str, Optional[str]]]:
    """Compare the current interpreter against the one that wrote the blobs.

    Returns only the differences, keyed by package. Python is compared at major.minor: a patch
    bump does not change numerics, a minor bump can change pickle compatibility.
    """
    from provenance import describe_environment

    now = describe_environment()
    then = manifest.get("environment", {})
    diffs: Dict[str, Dict[str, Optional[str]]] = {}

    def mm(v: Optional[str]) -> Optional[str]:
        return ".".join(str(v).split(".")[:2]) if v else None

    if mm(then.get("python")) != mm(now.get("python")):
        diffs["python"] = {"published_with": then.get("python"), "running": now.get("python")}
    for name in manifest.get("pinned_packages", PINNED_PACKAGES):
        a = (then.get("packages") or {}).get(name)
        b = (now.get("packages") or {}).get(name)
        if a != b:
            diffs[name] = {"published_with": a, "running": b}
    return diffs


def load_estimator(issue_dir: Path, target: str, horizon: int, kind: str = "point",
                   allow_version_mismatch: bool = False):
    """Load one persisted estimator, refusing rather than guessing.

    Order matters: **the digest is checked before the bytes reach joblib**, because unpickling
    executes code. Integrity is not overridable; version compatibility is, explicitly and
    loudly.
    """
    import joblib

    manifest = read_manifest(issue_dir)
    entry = _find(manifest, target, horizon, kind)
    path = Path(issue_dir) / ESTIMATOR_DIR / entry["file"]

    if not path.exists():
        pruned = manifest.get("retention", {}).get("pruned")
        raise EstimatorMissing(
            f"{entry['fit_id']} is recorded in the manifest but its blob is not on disk"
            + (f" -- it was pruned on {manifest['retention'].get('pruned_at')}. "
               f"Blobs are not tracked in git, so a clone never has them; the manifest digest "
               f"remains as proof of what was published." if pruned else
               f" at {path}. Expected SHA-256 {entry['sha256'][:12]}."))

    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise EstimatorDigestMismatch(
            f"{entry['fit_id']}: the blob on disk is not the one that was published. "
            f"Manifest records SHA-256 {entry['sha256']}, file is {actual}. Refusing to unpickle "
            f"-- loading would execute code from a file we cannot vouch for, and any prediction "
            f"from it would carry a published label it did not earn.")

    diffs = check_versions(manifest)
    if diffs and not allow_version_mismatch:
        detail = "; ".join(f"{k}: published with {v['published_with']}, running {v['running']}"
                           for k, v in sorted(diffs.items()))
        raise EstimatorVersionMismatch(
            f"{entry['fit_id']} was fitted under a different environment ({detail}). "
            f"scikit-learn only warns about this and loads anyway, which would silently return "
            f"different numbers under a published label -- so this refuses instead. Pass "
            f"allow_version_mismatch=True to load it as UNVERIFIED.")

    est = joblib.load(path)
    return est, {**entry, "version_diffs": diffs,
                 "verified": not diffs,
                 "manifest_environment": manifest.get("environment", {})}


def reproduce_prediction(issue_dir: Path, data_path: Path, target: str, horizon: int,
                         kind: str = "point", allow_version_mismatch: bool = False) -> Dict:
    """Re-derive one published number from its persisted estimator.

    The design matrix is rebuilt from the canonical data file rather than stored alongside the
    blob: it keeps one less copy of Treasury data at rest, and it ties reproduction to the exact
    input, whose SHA-256 the issue's provenance already records. A data file that has moved on is
    reported rather than silently used.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_forecast import Champion, _build_design
    from provenance import describe_input

    est, meta = load_estimator(issue_dir, target, horizon, kind,
                               allow_version_mismatch=allow_version_mismatch)
    manifest = read_manifest(issue_dir)

    data_sha = describe_input(str(data_path)).get("sha256")
    sha_at_issue = manifest.get("data_sha256")

    # The design must be rebuilt with the SAME feature groups the estimator was fitted on.
    # Building a different set and reindexing to the stored names would fill the difference with
    # NaN and predict on it -- which produces a plausible-looking number that is simply wrong
    # (measured while writing this: 106,407,564 against a published 58,319,423). Hence the groups
    # travel in the manifest, and any shortfall raises below rather than being reindexed away.
    champ = Champion(target=target, point_model="",
                     fiscal_groups=tuple(meta.get("fiscal_groups") or ()),
                     exog_blocks=tuple(meta.get("exog_blocks") or ()),
                     recipe_id=meta.get("recipe_id", ""), scaling="",
                     transform=meta.get("target_transform", "raw"))
    X, _ = _build_design(pd.read_csv(data_path), champ)

    want = list(meta["feature_names"])
    missing = [c for c in want if c not in X.columns]
    if missing:
        raise ReproductionUnavailable(
            f"{meta['fit_id']}: {len(missing)} of {len(want)} features could not be rebuilt from "
            f"{Path(data_path).name} ({missing[:5]}{'...' if len(missing) > 5 else ''}). Refusing "
            f"to substitute NaN -- the estimator would return a plausible number that means "
            f"nothing.")
    X = X.reindex(columns=want)

    origin = pd.Timestamp(meta["origin_date"])
    if origin not in X.index:
        raise ReproductionUnavailable(
            f"the issue origin {origin.date()} is not in {Path(data_path).name}, so the published "
            f"prediction cannot be re-derived from this file.")
    row = X.loc[[origin]]
    nan_cols = [c for c in row.columns if bool(pd.isna(row[c]).any())]
    if nan_cols:
        raise ReproductionUnavailable(
            f"{meta['fit_id']}: the rebuilt origin row is missing values for {len(nan_cols)} "
            f"features ({nan_cols[:5]}). The data file has changed shape since the issue; "
            f"refusing to predict on gaps.")
    pred = float(np.asarray(est.predict(row)).ravel()[0])

    fc = pd.read_csv(Path(issue_dir) / "forecast.csv")
    m = fc[(fc["target"] == target) & (fc["horizon"].astype(int) == int(horizon))]
    published_raw = float(m["p50"].iloc[0]) if len(m) else float("nan")
    # For a stock target the artifact stores the reconstructed level; the estimator predicts the
    # delta it was fitted on. Comparing those directly would be an order-of-magnitude error.
    base = float(m["origin_value"].iloc[0]) if len(m) else 0.0
    modelled = str(m["modelled_as"].iloc[0]) if len(m) else "level"
    rebuilt = pred + (base if modelled.startswith("delta") else 0.0)

    return {
        "fit_id": meta["fit_id"],
        "reproduced": rebuilt,
        "published": published_raw,
        "abs_diff": abs(rebuilt - published_raw),
        "rel_diff": (abs(rebuilt - published_raw) / abs(published_raw)
                     if published_raw else float("nan")),
        "verified_environment": meta["verified"],
        "version_diffs": meta["version_diffs"],
        "data_sha_matches_issue": (data_sha == sha_at_issue),
        "data_sha_now": data_sha,
        "data_sha_at_issue": sha_at_issue,
    }


# ── retention ─────────────────────────────────────────────────────────────────

def issues_with_unscored_horizons(published_root: Path,
                                  scorecard: Optional[Path] = None) -> List[str]:
    """Issues that still have a horizon without a realized score.

    These are protected from pruning however old they are: an unscored horizon is exactly the
    one whose prediction is still going to be argued about, so it is the one that most needs to
    be re-derivable.
    """
    root = Path(published_root)
    if not root.exists():
        return []
    scored: Dict[str, int] = {}
    sc = Path(scorecard) if scorecard else (root.parent / "scorecard.csv")
    if sc.exists():
        try:
            df = pd.read_csv(sc)
            if len(df) and {"issue_date", "y_true"} <= set(df.columns):
                done = df[df["y_true"].notna()]
                scored = done.groupby(done["issue_date"].astype(str)).size().to_dict()
        except Exception:               # pragma: no cover - a broken scorecard protects all
            return [d.name for d in root.iterdir() if d.is_dir()]

    out: List[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        fc = d / "forecast.csv"
        if not fc.exists():
            continue
        try:
            n = len(pd.read_csv(fc))
        except Exception:               # pragma: no cover
            continue
        if scored.get(d.name, 0) < n:
            out.append(d.name)
    return out


def prune_estimators(published_root: Path, keep_last: int = DEFAULT_KEEP_LAST,
                     protect: Sequence[str] = (), now: Optional[str] = None) -> List[Dict]:
    """Delete blobs for old issues, recording the deletion in each manifest.

    An issue in ``protect`` -- in practice, one with unscored horizons -- keeps its blobs however
    old it is: those are exactly the issues whose predictions are still going to be argued about.

    The manifest is never deleted. A pruned issue therefore still proves what it published and
    reads as an explicit "not available" on load, rather than as an issue that never had
    estimators.
    """
    root = Path(published_root)
    if not root.exists():
        return []
    issues = sorted(d for d in root.iterdir()
                    if d.is_dir() and (d / ESTIMATOR_DIR / MANIFEST_NAME).exists())
    keep = set(str(p.name) for p in issues[-int(keep_last):]) | set(protect)
    stamp = now or pd.Timestamp.now("UTC").isoformat()

    actions: List[Dict] = []
    for issue in issues:
        if issue.name in keep:
            continue
        est_dir = issue / ESTIMATOR_DIR
        manifest = json.loads((est_dir / MANIFEST_NAME).read_text())
        if manifest.get("retention", {}).get("pruned"):
            continue
        freed = 0
        for e in manifest.get("estimators", []):
            p = est_dir / e["file"]
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
        for sub in sorted((d for d in est_dir.rglob("*") if d.is_dir()), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
        manifest.setdefault("retention", {}).update(
            {"pruned": True, "pruned_at": stamp, "bytes_freed": freed})
        (est_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str),
                                            encoding="utf-8")
        actions.append({"issue_date": issue.name, "bytes_freed": freed,
                        "n_blobs": len(manifest.get("estimators", []))})
    return actions
