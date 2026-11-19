"""Step 06 - simulate the post-deployment stream and build the monitoring report.

Usage: uv run python scripts/06_run_monitoring.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from revaluation.config import MONITORING_DIR
from revaluation.data.features import model_matrix
from revaluation.monitoring.dashboard import build_figures, render_report, write_summary
from revaluation.monitoring.performance import evaluate_alerts
from revaluation.monitoring.stream import build_stream, drift_table, performance_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("monitoring")


def summarise_drift(drift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, group in drift.groupby("window"):
        features = group[group["feature"] != "__prediction__"]
        prediction = group[group["feature"] == "__prediction__"]
        drifted = features[features["psi"] >= 0.20]
        rows.append(
            {
                "window": window,
                "n_features": len(features),
                "n_drifted": len(drifted),
                "share_drifted": float(len(drifted) / max(len(features), 1)),
                "max_psi": float(features["psi"].max()),
                "mean_psi": float(features["psi"].mean()),
                "top_drifted": drifted.sort_values("psi", ascending=False)["feature"].head(6).tolist(),
                "prediction_psi": float(prediction["psi"].iloc[0]) if len(prediction) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def main() -> None:
    data = build_stream()
    observed = data["scenarios"]["observed"]
    stress = data["scenarios"]["stress"]

    observed_performance = performance_table(
        observed, data["stream"], data["baseline"], data["quantiles"]
    )
    observed_drift = drift_table(
        data["reference"], observed, data["stream"], data["model"], data["quantiles"]
    )
    observed_summary = summarise_drift(observed_drift)
    observed_alerting = evaluate_alerts(
        observed_performance, observed_summary, data["baseline"]
    )

    stress_performance = performance_table(
        stress, data["stream"], data["baseline"], data["quantiles"]
    )
    stress_drift = drift_table(
        data["reference"], stress, data["stream"], data["model"], data["quantiles"]
    )
    stress_summary = summarise_drift(stress_drift)
    stress_alerting = evaluate_alerts(stress_performance, stress_summary, data["baseline"])

    observed_performance.to_csv(MONITORING_DIR / "performance_by_window.csv", index=False)
    stress_performance.to_csv(
        MONITORING_DIR / "performance_by_window_stress.csv", index=False
    )
    observed_drift.to_csv(MONITORING_DIR / "drift_by_window.csv", index=False)
    stress_drift.to_csv(MONITORING_DIR / "drift_by_window_stress.csv", index=False)
    observed_summary.to_csv(MONITORING_DIR / "drift_summary.csv", index=False)

    reference_predictions = np.expm1(
        data["model"].predict(model_matrix(data["reference"])[0])
    )
    latest_mask = (data["stream"]["window"] == data["stream"]["window"].max()).to_numpy()
    figures = build_figures(
        observed_performance,
        observed_drift,
        reference_predictions,
        observed.prediction[latest_mask],
        data["baseline"]["rmse"],
    )

    reference = {
        "n": len(data["reference"]),
        "period": f"{data['reference']['sale_date'].min():%Y-%m} to {data['reference']['sale_date'].max():%Y-%m}",
        "baseline": data["baseline"],
        "conformal": data["quantiles"],
        "n_stream": len(data["stream"]),
        "stream_period": f"{data['stream']['sale_date'].min():%Y-%m} to {data['stream']['sale_date'].max():%Y-%m}",
    }
    summary = write_summary(
        reference=reference,
        performance=observed_performance.to_dict(orient="records"),
        drift=observed_summary.to_dict(orient="records"),
        alerting=observed_alerting,
        stress_performance=stress_performance.to_dict(orient="records"),
        stress_alerting=stress_alerting,
        quantiles=data["quantiles"],
    )
    render_report(summary, figures)

    print(
        json.dumps(
            {
                "observation_windows": observed_summary["window"].tolist(),
                "n_drifted_features": observed_summary["n_drifted"].tolist(),
                "latest_mae": round(float(observed_performance["mae"].iloc[-1]), 2),
                "latest_coverage_90": round(float(observed_performance["90_coverage"].iloc[-1]), 3),
                "decision_observed": observed_alerting["decision"],
                "decision_stress": stress_alerting["decision"],
                "alerts_observed": observed_alerting["n_critical"] + observed_alerting["n_warnings"],
            },
            indent=2,
        )
    )
    print(f"HTML report: {MONITORING_DIR / 'monitoring_report.html'}")


if __name__ == "__main__":
    main()
