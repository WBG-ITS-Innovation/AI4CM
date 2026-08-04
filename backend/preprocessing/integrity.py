"""DEPRECATED re-export shim. Use ``forecast_integrity`` instead.

This module used to be a second, parallel integrity implementation. It carried its
own copies of the h-step persistence baseline, the shift diagnostic and the
alignment check, and ``b_ml_pipeline`` merged its output over the shared module's
via ``integrity_report.update(legacy_report)`` -- so the number that reached
``integrity_report.json``, the Dashboard, the daily summary and the backtest report
came from the duplicate, not from the function the tests guarded (review §1.2).

Retired in Phase 2 item 1c (decision D7):

======================================  ==========================================
removed                                 use instead
======================================  ==========================================
``compute_persistence_baseline_from_origin``  ``forecast_integrity.compute_persistence_baseline``
``compute_baselines``                   ``forecast_integrity.compute_persistence_baseline``
                                        (its ``seasonal_naive`` output was degenerate:
                                        ``season_steps`` was hardcoded to 5, so at the
                                        production horizon h=5 it returned exactly the
                                        persistence baseline while presenting itself as
                                        independent corroboration)
``shift_sanity_check``                  ``forecast_integrity.shift_diagnostic_horizon_aware``
``validate_alignment``                  ``forecast_integrity.validate_alignment_step_based``
``compute_baseline_maes``               -- (only ever fed ``compute_baselines``)
``compute_integrity_report``            build the report from the shared helpers; see
                                        ``b_ml_pipeline`` for the reference assembly
======================================  ==========================================

``signal_sentinel``, ``leakage_sentinel`` and ``MIN_SIGNAL_RATIO`` moved to
``forecast_integrity`` unchanged and are re-exported here so existing imports keep
working. New code should import them from ``forecast_integrity``.
"""
from __future__ import annotations

from forecast_integrity import (  # noqa: F401
    MIN_SIGNAL_RATIO,
    check_feature_leakage,
    compute_persistence_baseline,
    compute_point_metrics,
    compute_skill_score,
    detect_lagged_copy,
    leakage_sentinel,
    shift_diagnostic_horizon_aware,
    signal_sentinel,
    validate_alignment_step_based,
)

__all__ = [
    "MIN_SIGNAL_RATIO",
    "check_feature_leakage",
    "compute_persistence_baseline",
    "compute_point_metrics",
    "compute_skill_score",
    "detect_lagged_copy",
    "leakage_sentinel",
    "shift_diagnostic_horizon_aware",
    "signal_sentinel",
    "validate_alignment_step_based",
]
