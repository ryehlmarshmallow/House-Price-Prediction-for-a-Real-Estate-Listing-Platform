"""Alerting rules and retraining policy (Module 7).

Monitoring is only useful if it changes a decision. The policy below converts
the drift and performance statistics into two classes of signal:

* **warning** - an anomaly that a human should inspect, but which does not by
  itself justify retraining;
* **critical** - a sustained breach that automatically opens a retraining
  ticket.

Sustained breaches are required for the accuracy and coverage rules because a
single small window can move an error metric by more than 20 percent through
sampling noise alone. The thresholds live in the central configuration so that
the policy can be audited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revaluation.config import (
    COVERAGE_DROP_THRESHOLD,
    DRIFT_MIN_FEATURES,
    DRIFT_PSI_THRESHOLD,
    PERFORMANCE_CONSECUTIVE_WINDOWS,
    PERFORMANCE_DEGRADATION_THRESHOLD,
)

RETRAIN_ACTIONS = {
    "retrain_now": (
        "Open a retraining ticket: refit on the most recent labelled data, "
        "recalibrate the conformal intervals, and re-run the shadow evaluation."
    ),
    "investigate_and_schedule": (
        "Keep the current model in production, open an investigation task and "
        "schedule retraining for the next planned release."
    ),
    "continue_monitoring": "No action beyond the standard monthly review.",
}


def _consecutive(flags: list[bool], required: int) -> bool:
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        if run >= required:
            return True
    return False


def evaluate_alerts(
    performance: pd.DataFrame,
    drift_summary: pd.DataFrame,
    baseline: dict,
) -> dict:
    """Apply the monitoring policy to one scenario.

    Parameters
    ----------
    performance:
        Per-window table produced by ``performance_table``.
    drift_summary:
        Per-window aggregation with ``n_drifted``, ``max_psi`` and
        ``prediction_psi`` columns.
    baseline:
        Reference out-of-fold metrics (used for the degradation ratio).
    """
    performance = performance.sort_values("window").reset_index(drop=True)
    drift_lookup = drift_summary.set_index("window")
    alerts: list[dict] = []
    windows = performance["window"].tolist()

    rmse_ratios = performance["rmse_ratio_vs_baseline"].tolist()
    coverage = performance["90_coverage"].tolist()
    bias = performance["bias_pct"].abs().tolist()

    for index, window in enumerate(windows):
        row = performance.iloc[index]
        drift_row = drift_lookup.loc[window]

        if drift_row["n_drifted"] >= DRIFT_MIN_FEATURES:
            alerts.append(
                {
                    "window": window,
                    "rule": "R1 input drift",
                    "severity": "warning",
                    "value": int(drift_row["n_drifted"]),
                    "threshold": f">= {DRIFT_MIN_FEATURES} features with PSI >= {DRIFT_PSI_THRESHOLD}",
                    "detail": ", ".join(drift_row["top_drifted"][:5]),
                    "action": "Review the drifting inputs for data-pipeline or market-structure causes.",
                }
            )
        if drift_row["prediction_psi"] >= DRIFT_PSI_THRESHOLD:
            alerts.append(
                {
                    "window": window,
                    "rule": "R2 prediction drift",
                    "severity": "warning",
                    "value": round(float(drift_row["prediction_psi"]), 3),
                    "threshold": f"PSI >= {DRIFT_PSI_THRESHOLD}",
                    "detail": "Score distribution moved relative to the reference vintage.",
                    "action": "Check whether the shift is explained by a known market move.",
                }
            )
        if row["rmse"] > baseline["rmse"] * (1 + PERFORMANCE_DEGRADATION_THRESHOLD):
            alerts.append(
                {
                    "window": window,
                    "rule": "R3 accuracy breach",
                    "severity": "critical",
                    "value": round(float(row["rmse_ratio_vs_baseline"]), 3),
                    "threshold": f"RMSE ratio > {PERFORMANCE_DEGRADATION_THRESHOLD}",
                    "detail": f"RMSE {row['rmse']:.0f} versus baseline {baseline['rmse']:.0f}.",
                    "action": "Inspect the worst errors and prepare a retraining candidate.",
                }
            )
        if row["90_coverage"] < (0.90 - COVERAGE_DROP_THRESHOLD):
            alerts.append(
                {
                    "window": window,
                    "rule": "R4 interval coverage",
                    "severity": "warning",
                    "value": round(float(row["90_coverage"]), 3),
                    "threshold": f"coverage < {0.90 - COVERAGE_DROP_THRESHOLD:.2f}",
                    "detail": "The 90% conformal interval under-covers the realised sales.",
                    "action": "Recalibrate the conformal quantiles on recent residuals.",
                }
            )
        if abs(row["bias_pct"]) > 5.0:
            alerts.append(
                {
                    "window": window,
                    "rule": "R5 systematic bias",
                    "severity": "warning",
                    "value": round(float(row["bias_pct"]), 2),
                    "threshold": "|bias| > 5 percent",
                    "detail": "Predictions are systematically shifted relative to realised prices.",
                    "action": "Verify the macro-context feed and the recorded sale dates.",
                }
            )

    # Sustained-breach escalation -----------------------------------------
    sustained_accuracy = _consecutive(
        [ratio > 1 + PERFORMANCE_DEGRADATION_THRESHOLD for ratio in rmse_ratios],
        PERFORMANCE_CONSECUTIVE_WINDOWS,
    )
    sustained_coverage = _consecutive(
        [value < (0.90 - COVERAGE_DROP_THRESHOLD) for value in coverage],
        PERFORMANCE_CONSECUTIVE_WINDOWS,
    )
    sustained_bias = _consecutive(
        [value > 5.0 for value in bias], PERFORMANCE_CONSECUTIVE_WINDOWS
    )
    sustained_drift = _consecutive(
        [
            int(drift_lookup.loc[window]["n_drifted"]) >= DRIFT_MIN_FEATURES
            for window in windows
        ],
        PERFORMANCE_CONSECUTIVE_WINDOWS,
    )

    if sustained_accuracy:
        alerts.append(
            {
                "window": "policy",
                "rule": "R6 sustained accuracy degradation",
                "severity": "critical",
                "value": PERFORMANCE_DEGRADATION_THRESHOLD,
                "threshold": f"RMSE ratio > {PERFORMANCE_DEGRADATION_THRESHOLD} for {PERFORMANCE_CONSECUTIVE_WINDOWS} consecutive windows",
                "detail": "Accuracy has not recovered within one monitoring window.",
                "action": "Retrain the model on the most recent labelled data.",
            }
        )
    if sustained_coverage:
        alerts.append(
            {
                "window": "policy",
                "rule": "R7 sustained coverage erosion",
                "severity": "critical",
                "value": round(float(np.mean(coverage)), 3),
                "threshold": f"coverage < {0.90 - COVERAGE_DROP_THRESHOLD:.2f} for {PERFORMANCE_CONSECUTIVE_WINDOWS} consecutive windows",
                "detail": "The calibrated uncertainty no longer matches the realised error distribution.",
                "action": "Recalibrate conformal quantiles and retrain if the erosion persists.",
            }
        )
    if sustained_bias:
        alerts.append(
            {
                "window": "policy",
                "rule": "R8 sustained bias",
                "severity": "critical",
                "value": round(float(np.mean(bias)), 2),
                "threshold": f"|bias| > 5 percent for {PERFORMANCE_CONSECUTIVE_WINDOWS} consecutive windows",
                "detail": "A directional error persists across windows.",
                "action": "Retrain with recent data and re-check the macro features.",
            }
        )
    if sustained_drift and not (sustained_accuracy or sustained_coverage):
        alerts.append(
            {
                "window": "policy",
                "rule": "R9 sustained input drift",
                "severity": "warning",
                "value": int(drift_summary["n_drifted"].max()),
                "threshold": f"drift for {PERFORMANCE_CONSECUTIVE_WINDOWS} consecutive windows",
                "detail": "The input population has moved away from the training vintage.",
                "action": "Schedule retraining for the next release cycle.",
            }
        )

    critical = [alert for alert in alerts if alert["severity"] == "critical"]
    warnings = [alert for alert in alerts if alert["severity"] == "warning"]
    if critical:
        decision = "retrain_now"
    elif warnings:
        decision = "investigate_and_schedule"
    else:
        decision = "continue_monitoring"

    return {
        "alerts": alerts,
        "decision": decision,
        "action": RETRAIN_ACTIONS[decision],
        "n_critical": len(critical),
        "n_warnings": len(warnings),
        "rules": {
            "R1": "input drift: at least three features with PSI >= 0.20",
            "R2": "prediction drift: PSI of the score distribution >= 0.20",
            "R3": "accuracy breach: window RMSE > 1.2 x reference RMSE",
            "R4": "coverage: 90% interval coverage < 0.85",
            "R5": "bias: absolute mean signed error > 5% of price",
            "R6": "sustained accuracy degradation over two windows",
            "R7": "sustained coverage erosion over two windows",
            "R8": "sustained directional bias over two windows",
            "R9": "sustained input drift over two windows",
        },
    }
