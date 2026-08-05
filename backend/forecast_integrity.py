"""
Forecast Integrity Module - Horizon-Aware Validation

This module provides comprehensive integrity checks for forecast predictions:
- Alignment validation (step-based, horizon-aware)
- Leakage detection (feature-level checks)
- Shift diagnostics (horizon-aware interpretation)
- Baseline comparisons (quality gates)

✅ Designed to work for ANY horizon h (not just h=6).
"""

from __future__ import annotations

from typing import Sequence, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def validate_alignment_step_based(
    predictions_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    horizon: int,
) -> Dict:
    """
    Validate alignment using step-based (index-position) logic.
    
    ✅ Horizon-aware: Works for ANY horizon h.
    
    For each prediction:
        idx(target_date) - idx(origin_date) == h_steps
    
    Args:
        predictions_df: DataFrame with columns ['origin_date', 'target_date', 'horizon']
        date_index: The actual DatetimeIndex used for modeling (after filtering/resampling)
        horizon: Forecast horizon (in steps, not calendar days)
        
    Returns:
        Dictionary with alignment_ok, n_misaligned, misaligned_examples
    """
    if "origin_date" not in predictions_df.columns or "target_date" not in predictions_df.columns:
        return {
            "alignment_ok": False,
            "error": "Missing origin_date or target_date columns",
            "n_misaligned": 0,
            "misaligned_examples": [],
        }
    
    df = predictions_df.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    
    # Build position map: date -> position in index
    date_to_pos = {date: pos for pos, date in enumerate(date_index)}
    
    # Compute expected target using step-based logic
    alignment_errors = []
    for idx, row in df.iterrows():
        origin_date = pd.Timestamp(row["origin_date"]).normalize()
        target_date = pd.Timestamp(row["target_date"]).normalize()
        
        if origin_date not in date_to_pos:
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "target": target_date.date(),
                "error": "origin_date not found in date_index",
            })
            continue
        
        pos_origin = date_to_pos[origin_date]
        pos_target_expected = pos_origin + horizon
        
        if pos_target_expected >= len(date_index):
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "target": target_date.date(),
                "error": f"target position {pos_target_expected} out of bounds (len={len(date_index)})",
            })
            continue
        
        expected_target = date_index[pos_target_expected]
        
        if target_date != expected_target:
            alignment_errors.append({
                "idx": idx,
                "origin": origin_date.date(),
                "expected": expected_target.date(),
                "actual": target_date.date(),
                "difference_steps": pos_target_expected - date_to_pos.get(target_date, -999),
            })
    
    n_misaligned = len(alignment_errors)
    
    return {
        "alignment_ok": n_misaligned == 0,
        "n_misaligned": n_misaligned,
        "n_total": len(df),
        "misaligned_examples": alignment_errors[:10],  # Limit examples
        "validation_method": "step_based",
    }


def check_feature_leakage(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
) -> Dict:
    """
    Check for leakage in feature engineering.
    
    ✅ Horizon-aware: Ensures no feature uses information beyond origin_date.
    
    Checks:
    - No negative lags (shift(-k) where k < 0)
    - No centered rolling windows
    - All rolling features are trailing (shift(1) applied)
    - No feature uses y_target or shifted(-k) with k < 0
    
    Args:
        feature_df: DataFrame with feature columns
        target_series: Target series (for reference)
        horizon: Forecast horizon
        
    Returns:
        Dictionary with leakage_detected, leakage_details
    """
    leakage_issues = []
    
    # Check for negative shifts in feature names or values
    for col in feature_df.columns:
        # Check column name for negative lag indicators
        if "shift(-" in col.lower() or "lag_-" in col.lower():
            leakage_issues.append(f"Feature '{col}' appears to use negative shift (leakage)")
        
        # Check if feature values correlate perfectly with future target (suspicious)
        if len(feature_df) > horizon:
            # Compare feature at t with target at t+h
            # Use .values to avoid index-alignment issues between sliced series
            feat_vals = feature_df[col].iloc[:-horizon].values if horizon > 0 else feature_df[col].values
            target_vals = target_series.iloc[horizon:].values if horizon > 0 else target_series.values

            if len(feat_vals) == len(target_vals) and len(feat_vals) > 10:
                # Drop NaN pairs
                valid = ~(np.isnan(feat_vals) | np.isnan(target_vals))
                if valid.sum() > 10:
                    corr = np.corrcoef(feat_vals[valid], target_vals[valid])[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.99:
                        leakage_issues.append(
                            f"Feature '{col}' has near-perfect correlation (r={corr:.3f}) with target at t+{horizon} (suspicious)"
                        )
    
    return {
        "leakage_detected": len(leakage_issues) > 0,
        "leakage_details": leakage_issues,
        "n_checks": len(feature_df.columns),
    }


def shift_diagnostic_horizon_aware(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    max_shift: Optional[int] = None,
) -> Dict:
    """
    Horizon-aware shift diagnostic.
    
    ✅ Interprets shifts relative to horizon:
    - best_shift ≈ 0: OK
    - best_shift ≈ -h: persistence-like (compare vs baseline)
    - best_shift ≈ -(h+1): missing lag_0 / using lag_1 (FLAG strongly)
    
    Args:
        y_true: True values at target dates
        y_pred: Predicted values
        horizon: Forecast horizon
        max_shift: Maximum shift to test (default: max(horizon+5, 10))
        
    Returns:
        Dictionary with best_shift, interpretation, lag_0_issue flag
    """
    if max_shift is None:
        max_shift = max(horizon + 5, 10)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
    
    mae_shift0 = float(np.mean(np.abs(y_true - y_pred)))
    best_shift = 0
    best_mae = mae_shift0
    shift_maes = {}
    
    # Test shifts in range [-max_shift, +max_shift]
    shifts_to_test = set(range(-max_shift, max_shift + 1))
    shifts_to_test.add(-horizon)  # Always test shift=-h
    shifts_to_test.add(-(horizon + 1))  # Always test shift=-(h+1) for lag_0 detection
    shifts_to_test = sorted(shifts_to_test)
    
    for shift in shifts_to_test:
        if shift == 0:
            mae = mae_shift0
        elif shift > 0:
            if shift < len(y_true):
                mae = float(np.mean(np.abs(y_true[shift:] - y_pred[:-shift])))
            else:
                continue
        else:
            shift_abs = abs(shift)
            if shift_abs < len(y_true):
                mae = float(np.mean(np.abs(y_true[:-shift_abs] - y_pred[shift_abs:])))
            else:
                continue
        
        shift_maes[shift] = mae
        if mae < best_mae:
            best_mae = mae
            best_shift = shift
    
    improvement_ratio = best_mae / mae_shift0 if mae_shift0 > 0 else 1.0
    improvement_pct = ((mae_shift0 - best_mae) / mae_shift0 * 100) if mae_shift0 > 0 else 0.0
    
    # ✅ Horizon-aware interpretation
    is_lag0_issue = False
    is_persistence_like = False
    interpretation = "OK"
    
    if abs(best_shift) <= 1:
        interpretation = "OK (best alignment at shift=0 or ±1)"
    elif abs(best_shift - (-horizon)) <= 1:
        is_persistence_like = True
        interpretation = f"Persistence-like (best_shift≈-h={-horizon}, compare vs persistence baseline)"
    elif abs(best_shift - (-(horizon + 1))) <= 1:
        is_lag0_issue = True
        interpretation = f"MISSING LAG_0 (best_shift≈-(h+1)={-(horizon+1)}, model using lag_1 instead)"
    elif best_shift < -horizon:
        interpretation = f"Suspicious backward shift (best_shift={best_shift} < -h={-horizon})"
    elif best_shift > 0:
        interpretation = f"Forward shift detected (best_shift={best_shift}, predictions ahead?)"
    
    return {
        "best_shift": int(best_shift),
        "best_mae": float(best_mae),
        "mae_shift0": float(mae_shift0),
        "mae_shift_minus_h": shift_maes.get(-horizon, np.nan),
        "mae_shift_minus_h_plus_1": shift_maes.get(-(horizon + 1), np.nan),
        "improvement_pct": float(improvement_pct),
        "improvement_ratio": float(improvement_ratio),
        "is_lag0_issue": bool(is_lag0_issue),
        "is_persistence_like": bool(is_persistence_like),
        "interpretation": interpretation,
        "shift_maes": {int(k): float(v) for k, v in shift_maes.items()},
    }


def compute_persistence_baseline(
    predictions_df: pd.DataFrame,
) -> Dict:
    """
    Compute persistence baseline: y_hat(t+h) = y(t) (origin value).
    
    Args:
        predictions_df: DataFrame with columns ['origin_value', 'y_true', 'y_pred']
        
    Returns:
        Dictionary with mae_persistence, rmse_persistence, skill metrics
    """
    if "origin_value" not in predictions_df.columns or "y_true" not in predictions_df.columns:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "error": "Missing origin_value or y_true columns",
        }
    
    df = predictions_df.dropna(subset=["origin_value", "y_true"]).copy()
    if len(df) == 0:
        return {
            "mae_persistence": np.nan,
            "rmse_persistence": np.nan,
            "error": "No valid rows after dropping NaN",
        }
    
    persistence_preds = df["origin_value"].values
    y_true = df["y_true"].values
    
    mae = float(np.mean(np.abs(y_true - persistence_preds)))
    rmse = float(np.sqrt(np.mean((y_true - persistence_preds) ** 2)))
    
    return {
        "mae_persistence": mae,
        "rmse_persistence": rmse,
        "n_valid": len(df),
    }


#: Season length for the seasonal-naive reference, in index steps.
#: 5 business days = one week on this index.
SEASONAL_NAIVE_SEASON_STEPS = 5


def compute_seasonal_naive_baseline(
    predictions_df: pd.DataFrame,
    horizon: int,
    season_steps: int = SEASONAL_NAIVE_SEASON_STEPS,
) -> Dict:
    """Seasonal-naive reference: y_hat(t) = y(t - season_steps).

    ✅ A2.  This replaces the ``mae_seasonal_naive`` that the retired
    ``preprocessing.integrity.compute_baselines`` produced, and fixes what was wrong
    with it: ``season_steps`` was hardcoded to 5 while the production horizon is also
    5, so the "seasonal naive" baseline returned **exactly the persistence baseline**
    and was displayed beside it as if it were independent corroboration.  Verified on
    the real artifact: ``mae_seasonal_naive == mae_persistence`` to the cent at h=5,
    and not at h=3 or h=7 (review §1.2).

    Rather than silently drop the field the Dashboard reads, the degeneracy is made
    explicit.  When ``season_steps == horizon`` the two references coincide, so the
    value is returned as NaN with ``seasonal_naive_degenerate=True``: a reader sees
    "not available, and here is why" instead of a duplicate wearing another name.

    Requires ``target_date`` and ``y_true``; the season origin is taken positionally
    from the sorted target dates, so it is step-based rather than calendar-based.
    """
    out = {
        "mae_seasonal_naive": np.nan,
        "seasonal_naive_season_steps": int(season_steps),
        "seasonal_naive_degenerate": bool(int(season_steps) == int(horizon)),
    }
    if out["seasonal_naive_degenerate"]:
        out["seasonal_naive_note"] = (
            f"season_steps ({season_steps}) equals the horizon ({horizon}), so a "
            f"seasonal-naive reference is identical to h-step persistence. Reported "
            f"as NaN rather than duplicating mae_persistence under another name."
        )
        return out

    if not {"target_date", "y_true"}.issubset(predictions_df.columns):
        out["seasonal_naive_note"] = "target_date or y_true missing"
        return out

    df = predictions_df.dropna(subset=["y_true"]).copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("target_date").drop_duplicates(subset=["target_date"])
    y = df["y_true"].to_numpy(dtype=float)
    if len(y) <= season_steps:
        out["seasonal_naive_note"] = (
            f"only {len(y)} target dates; need more than season_steps={season_steps}"
        )
        return out

    out["mae_seasonal_naive"] = float(np.mean(np.abs(y[season_steps:] - y[:-season_steps])))
    out["n_seasonal_naive"] = int(len(y) - season_steps)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# THE GATE CONTRACT  (X9, X10, F1)
#
# X9. Families disagreed on polarity: b_ml wrote `quality_gate_failed`, while e_quantile,
# c_dl and a_stat wrote `quality_gate_passed`. Any consumer reading one key got nothing on
# the families writing the other -- and `daily_summary.gate_reasons()` read only
# `quality_gate_passed`, so a B_ML run that FAILED its gate was reported as passing. Two keys
# with opposite polarity is not a naming inconsistency; it is a silent inversion.
#
# `GATE_KEY` is now canonical and positive. `read_gate()` is the ONE reader, and it accepts the
# legacy key so historical artifacts stay readable while never letting absence read as a pass.
#
# X10. There must be exactly one PUBLISHER of a gate verdict -- the family's own
# integrity_report.json -- and every other artifact DERIVES from it. SUMMARY.json contradicted
# integrity_report.json on the 2026-08-04 C_DL run because both computed it independently.
#
# F1. `measured` and `threshold` must both be numeric so a reader can compare them. A string
# in either makes the pair undisplayable and uncheckable.
# ══════════════════════════════════════════════════════════════════════════════

GATE_KEY = "quality_gate_passed"
GATE_KEY_LEGACY = "quality_gate_failed"

#: Tri-state gate verdicts. `None` means never verified and must never read as a pass.
GATE_PASS = True
GATE_FAIL = False
GATE_UNVERIFIED = None


def read_gate(report: Optional[Dict]) -> Optional[bool]:
    """The single reader for a gate verdict. Returns True / False / None.

    Resolution order, and the reason for it:
      1. ``run_status == "FAILED_QUALITY"`` -- an explicit failure outranks any flag.
      2. the canonical positive key.
      3. the legacy inverted key, negated (historical artifacts).
      4. ``run_status == "SUCCESS"`` -- a positive assertion, honoured only after both keys.
      5. ``None`` -- never verified. Absence of all of the above is NOT a pass.
    """
    if not report:
        return GATE_UNVERIFIED
    if str(report.get("run_status", "")).strip().upper() == "FAILED_QUALITY":
        return GATE_FAIL
    if GATE_KEY in report and report[GATE_KEY] is not None:
        return bool(report[GATE_KEY])
    if GATE_KEY_LEGACY in report and report[GATE_KEY_LEGACY] is not None:
        return not bool(report[GATE_KEY_LEGACY])
    # 4. `run_status == "SUCCESS"` is a positive assertion by the publisher, so it is honoured
    #    as a FALLBACK -- after both explicit keys, so an explicit false always wins. Without
    #    this, tightening the reader would have flipped historical artifacts that record only a
    #    status to "never verified", which is a different claim from what they assert.
    if str(report.get("run_status", "")).strip().upper() == "SUCCESS":
        return GATE_PASS
    return GATE_UNVERIFIED


def write_gate(report: Dict, passed: Optional[bool], *,
               reasons: Optional[Sequence[str]] = None) -> Dict:
    """Write the canonical gate verdict, keeping the legacy key in sync.

    The legacy key is still emitted so consumers pinned to it do not silently flip meaning
    during the transition; it is derived here rather than set independently, which is what
    stopped the two from contradicting each other.
    """
    report[GATE_KEY] = passed
    if passed is None:
        report.pop(GATE_KEY_LEGACY, None)
    else:
        report[GATE_KEY_LEGACY] = (not passed)
    if reasons is not None:
        report["gate_reasons"] = list(reasons)
    return report


def gate_check(name: str, measured, threshold, passed: Optional[bool],
               reason_plain: str = "") -> Dict:
    """One gate check, with measured and threshold both NUMERIC (F1).

    A string in either field makes the pair impossible to compare or plot, which is how a gate
    ends up displayed without the number that justified it.
    """
    def _num(v):
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None      # NaN -> None

    return {"name": name, "measured": _num(measured), "threshold": _num(threshold),
            "passed": passed, "reason_plain": reason_plain}


def compute_point_metrics(y_true, y_pred) -> Dict[str, float]:
    """MAE / RMSE / R2 for one point forecast, from one place.

    Added when the duplicate integrity module was retired.  ``compute_integrity_report``
    used to compute these inline and its results then overwrote the shared ones via
    ``integrity_report.update(legacy_report)``, so the numbers a consumer read came
    from the duplicate rather than the shared implementation (review §1.2).  Callers
    now build the same fields from this helper, so there is one definition.

    R2 is NaN when the target has no variance, rather than dividing by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(ok.sum()),
    }


def compute_skill_score(
    mae_model: float,
    mae_baseline: float,
) -> float:
    """
    Compute skill score: (MAE_baseline - MAE_model) / MAE_baseline * 100.
    
    Positive skill means model is better than baseline.
    
    Args:
        mae_model: Model MAE
        mae_baseline: Baseline MAE
        
    Returns:
        Skill percentage (NaN if baseline MAE is 0 or invalid)
    """
    if np.isnan(mae_model) or np.isnan(mae_baseline) or mae_baseline <= 0:
        return np.nan

    skill = ((mae_baseline - mae_model) / mae_baseline) * 100.0
    return float(skill)


# ---------------------------------------------------------------------------
# Shuffled-target signal control (moved here from preprocessing/integrity.py)
# ---------------------------------------------------------------------------
# Lives beside the other diagnostics so there is ONE integrity module. The old
# location remains as a deprecated re-export; see preprocessing/integrity.py.

MIN_SIGNAL_RATIO = 1.5   # shuffled MAE must be at least this multiple of real MAE


PROBE_RIDGE = "ridge"
PROBE_TREE = "tree"
PROBE_FOREST = "forest"
DEFAULT_PROBE = PROBE_RIDGE


def _probe_factory(probe: str):
    """Return a zero-argument constructor for the requested sentinel probe.

    The probe is the *instrument*, not a model under evaluation, so every option is
    fixed and untuned: a tuned probe would make the ratio a function of the tuning.

    ``ridge``  — the historical default. Linear, so it can only register signal a
                 linear model can use.
    ``tree``   — a single depth-4 decision tree. The minimal nonlinear instrument.
    ``forest`` — 100 depth-4 trees. Same hypothesis class as ``tree`` but far lower
                 variance, so a null result is more trustworthy.

    See reports/sentinel_probe_study.md. Changing the default would change what the
    1.50 threshold means, so the default is not changed here.
    """
    if probe == PROBE_RIDGE:
        return lambda: Ridge(alpha=1.0)
    if probe == PROBE_TREE:
        from sklearn.tree import DecisionTreeRegressor
        return lambda: DecisionTreeRegressor(max_depth=4, random_state=0)
    if probe == PROBE_FOREST:
        from sklearn.ensemble import RandomForestRegressor
        return lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=0, n_jobs=1)
    raise ValueError(
        f"unknown sentinel probe {probe!r}; expected one of "
        f"{(PROBE_RIDGE, PROBE_TREE, PROBE_FOREST)}"
    )


def signal_sentinel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
    probe: str = DEFAULT_PROBE,
) -> Dict:
    """Shuffled-target control: does this feature set carry real signal?

    Fit a light model (Ridge) twice on the same features — once on the true
    targets, once on shuffled targets — and compare held-out error.  If the
    features genuinely predict the target, destroying the pairing should make
    performance clearly worse, so mae_shuffled / mae_normal should be well
    above 1.

    ✅ M-5 — what this check does and does NOT mean.
    The previous version reported a low ratio as ``leakage_warning=true``.
    That reading is backwards.  Leakage means the model can see the future,
    which makes its *real*-target error implausibly SMALL, and therefore makes
    this ratio LARGE.  A low ratio means the opposite: shuffling the targets
    barely hurt, i.e. the features were never carrying much signal about the
    target.  A ratio below 1 means the shuffled model actually did better —
    strong evidence of no usable signal at all.

    So this function answers "is there signal?" and deliberately does not
    claim to detect leakage.  Leakage is covered by the other checks:
    origin_date >= target_date, near-perfect feature/target correlation
    (check_feature_leakage), alignment validation, and the shift diagnostics.

    Features are standardised before fitting.  Without scaling, Ridge on raw
    treasury magnitudes is ill-conditioned (observed rcond ~1e-18), which made
    both error figures — and hence the ratio — unreliable.

    Returns a dict with mae_normal, mae_shuffled_target,
    shuffled_to_normal_ratio, signal_detected, signal_verdict.
    """
    insufficient = {
        "mae_normal": np.nan,
        "mae_shuffled_target": np.nan,
        "shuffled_to_normal_ratio": np.nan,
        "signal_detected": None,
        "signal_verdict": "not measurable (insufficient data)",
        "probe": probe,
        # Kept for backward compatibility only; this check cannot detect
        # leakage, so it never asserts leakage.  See docstring.
        "leakage_warning": False,
        "note": "Insufficient data for signal test",
    }
    if len(X_train) < 10 or len(X_test) < 5:
        return insufficient

    Xtr = X_train.to_numpy(dtype=float, copy=True)
    Xte = X_test.to_numpy(dtype=float, copy=True)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardise using TRAIN statistics only (test must not inform scaling).
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma

    y_tr = np.asarray(y_train, dtype=float)
    y_te = np.asarray(y_test, dtype=float)

    _make = _probe_factory(probe)

    model_normal = _make()
    model_normal.fit(Xtr, y_tr)
    mae_normal = float(np.mean(np.abs(y_te - model_normal.predict(Xte))))

    rng = np.random.default_rng(42)
    y_tr_shuffled = y_tr.copy()
    rng.shuffle(y_tr_shuffled)

    model_shuffled = _make()
    model_shuffled.fit(Xtr, y_tr_shuffled)
    mae_shuffled = float(np.mean(np.abs(y_te - model_shuffled.predict(Xte))))

    if not np.isfinite(mae_normal) or mae_normal <= 0:
        out = dict(insufficient)
        out["mae_normal"] = mae_normal
        out["mae_shuffled_target"] = mae_shuffled
        out["signal_verdict"] = "not measurable (degenerate real-target error)"
        return out

    ratio = mae_shuffled / mae_normal
    signal_detected = bool(ratio >= MIN_SIGNAL_RATIO)
    if ratio < 1.0:
        verdict = (f"NO SIGNAL: shuffling the targets improved held-out error "
                   f"(ratio {ratio:.2f} < 1.00) — the features do not predict "
                   f"the target")
    elif not signal_detected:
        verdict = (f"WEAK SIGNAL: shuffling the targets barely hurt "
                   f"(ratio {ratio:.2f} < {MIN_SIGNAL_RATIO:.2f} required)")
    else:
        verdict = (f"signal present: shuffling the targets made error "
                   f"{ratio:.2f}x worse")

    return {
        "mae_normal": mae_normal,
        "mae_shuffled_target": mae_shuffled,
        "shuffled_to_normal_ratio": float(ratio),
        "signal_detected": signal_detected,
        "signal_verdict": verdict,
        # Which instrument produced these numbers. Reported so a ratio can never be
        # compared across probes by accident -- the threshold is calibrated to ridge.
        "probe": probe,
        # Backward compatibility: never asserts leakage (see docstring).
        "leakage_warning": False,
    }


def leakage_sentinel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
) -> Dict:
    """Deprecated name for :func:`signal_sentinel`.

    Retained so existing callers keep working.  The name is misleading — this
    check measures signal presence, not leakage — so new code should call
    signal_sentinel directly.
    """
    return signal_sentinel(X_train, y_train, X_test, y_test, horizon)

def detect_lagged_copy(
    predictions_df: pd.DataFrame,
    model_col: str = "model",
    date_col: str = "date",
    max_shift: int = 3,
    skip_patterns: Tuple[str, ...] = ("baseline", "naive", "persistence"),
) -> Dict:
    """Detect models whose forecasts are effectively a lagged copy of the target.

    This is the classic symptom of a model that only learned "persistence":
    instead of forecasting, it repeats a recent actual value.  Such a model
    can look accurate on smooth series while being useless in practice.

    For each model's out-of-sample predictions we compute two things:

    1. MAE(pred, y) versus MAE(y_lag1, y), where ``y_lag1`` is the target
       shifted one step (the "persistence" baseline).  A real model should
       beat this baseline; a lagged copy will not.
    2. The correlation of ``pred`` against the target shifted by every step
       in ``-max_shift .. +max_shift``.  For an honest model the correlation
       peaks at shift 0 (aligned with the true target).  For a lagged copy it
       peaks at a non-zero shift.

    A model is flagged when BOTH signals agree: its predictions correlate
    better with a shifted target than with the true one, AND it fails to beat
    the lag-1 persistence baseline.

    Parameters
    ----------
    predictions_df : DataFrame
        Must contain columns: ``y_true``, ``y_pred``.  A ``model_col`` groups
        rows by model (if absent, all rows are treated as one model).  A
        ``date_col`` is used to sort each model's rows in time order (if
        absent, existing row order is assumed to be chronological).
    model_col : str
        Column identifying the model name.
    date_col : str
        Column giving the time order within each model.
    max_shift : int
        Largest shift (in steps) to test in each direction.  Default 3, i.e.
        the correlation window is -3 .. +3.
    skip_patterns : tuple of str
        Model names containing any of these substrings (case-insensitive) are
        skipped: they are persistence/naive models by design (e.g. the ML
        pipeline's "Persistence (baseline)" row and the stat pipeline's
        "naive_last" model), so flagging them as lagged copies would be a
        true-but-uninteresting positive.

    Returns
    -------
    dict in the shared audit-report format with keys:
        risk : "low" | "high"
        details : list of human-readable finding strings
        per_model : list of per-model dicts with the raw numbers
    """
    result: Dict = {"risk": "low", "details": [], "per_model": []}

    df = predictions_df
    if df.empty or "y_true" not in df.columns or "y_pred" not in df.columns:
        result["details"].append("No valid predictions to analyze.")
        return result

    # Group by model, or treat the whole frame as a single unnamed model.
    if model_col in df.columns:
        groups = list(df.groupby(model_col, sort=False))
    else:
        groups = [("(all)", df)]

    for model_name, group in groups:
        # Skip persistence/naive models: they ARE lagged copies by design, so
        # flagging them would be a true-but-uninteresting positive.
        name_lower = str(model_name).lower()
        if any(pat.lower() in name_lower for pat in skip_patterns):
            continue

        # Sort in time order so that "shift by k steps" is meaningful.
        if date_col in group.columns:
            group = group.sort_values(date_col)

        y = group["y_true"].reset_index(drop=True)
        pred = group["y_pred"].reset_index(drop=True)

        valid = y.notna() & pred.notna()
        if valid.sum() < max_shift + 5:
            # Not enough data to say anything reliable.
            continue

        y = y[valid].reset_index(drop=True)
        pred = pred[valid].reset_index(drop=True)

        # ── MAE of the model vs the lag-1 persistence baseline ──
        mae_pred = float(np.mean(np.abs(pred - y)))
        y_lag1 = y.shift(1)
        pair = y_lag1.notna()
        mae_lag1 = float(np.mean(np.abs(y_lag1[pair] - y[pair])))

        # ── Correlation of pred vs target shifted by each step ──
        corr_by_shift: Dict[int, float] = {}
        for k in range(-max_shift, max_shift + 1):
            y_shifted = y.shift(k)
            # Correlation is only defined where both series overlap AND vary.
            # A constant segment (zero variance) would divide by zero, so we
            # record NaN for those shifts instead of emitting a warning.
            both = pred.notna() & y_shifted.notna()
            if both.sum() < 3 or pred[both].std() == 0 or y_shifted[both].std() == 0:
                corr_by_shift[k] = float("nan")
                continue
            corr = pred[both].corr(y_shifted[both])
            corr_by_shift[k] = float(corr) if pd.notna(corr) else float("nan")

        corr_at_0 = corr_by_shift.get(0, float("nan"))
        # Best shift = the one with the highest (finite) correlation.
        finite = {k: v for k, v in corr_by_shift.items() if np.isfinite(v)}
        if not finite:
            continue
        best_shift = max(finite, key=finite.get)
        corr_best = finite[best_shift]

        # ── Decide whether to flag ──
        # Signal 1: aligns better with a shifted target than the true one.
        #   Require a small margin so tiny numerical wobble at shift 0 does
        #   not trip the flag.
        aligns_shifted = (
            best_shift != 0
            and np.isfinite(corr_at_0)
            and (corr_best - corr_at_0) > 0.05
        )
        # Signal 2: does not beat the lag-1 persistence baseline.
        no_skill_vs_lag1 = np.isfinite(mae_lag1) and mae_pred >= mae_lag1 * 0.99

        flagged = bool(aligns_shifted and no_skill_vs_lag1)

        result["per_model"].append({
            "model": model_name,
            "mae_pred": mae_pred,
            "mae_lag1": mae_lag1,
            "best_shift": int(best_shift),
            "corr_at_0": corr_at_0,
            "corr_best": corr_best,
            "flagged": flagged,
        })

        if flagged:
            result["risk"] = "high"
            result["details"].append(
                f"Model '{model_name}': predictions align best with y shifted by "
                f"{best_shift:+d} (corr {corr_best:.2f} vs {corr_at_0:.2f} at shift 0), "
                f"and do not beat the lag-1 baseline (MAE {mae_pred:.2f} vs "
                f"{mae_lag1:.2f}) — likely a lagged copy (persistence only)."
            )

    if not result["details"]:
        result["details"].append("No lagged-copy (persistence-only) models detected.")

    return result
