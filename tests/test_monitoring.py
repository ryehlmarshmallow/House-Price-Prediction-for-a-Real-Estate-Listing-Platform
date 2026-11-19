"""Tests for drift statistics and the retraining policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revaluation.models.train import conformal_quantiles
from revaluation.monitoring.drift import (
    categorical_psi,
    feature_drift,
    numeric_psi,
    population_stability,
)
from revaluation.monitoring.performance import evaluate_alerts


def test_psi_is_zero_for_identical_distributions() -> None:
    reference = pd.Series(np.random.default_rng(0).normal(0, 1, 2000))
    assert numeric_psi(reference, reference) == 0.0


def test_psi_detects_a_clear_shift() -> None:
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(3, 1, 2000))
    assert numeric_psi(reference, shifted) > 0.2


def test_categorical_psi_detects_mix_change() -> None:
    reference = pd.Series(["a"] * 800 + ["b"] * 200)
    current = pd.Series(["a"] * 300 + ["b"] * 700)
    assert categorical_psi(reference, current) > 0.2
    assert categorical_psi(reference, reference) == 0.0


def test_feature_drift_reports_severity() -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame(
        {"stable": rng.normal(0, 1, 500), "unstable": rng.normal(0, 1, 500), "kind": ["x"] * 500}
    )
    current = pd.DataFrame(
        {"stable": rng.normal(0, 1, 500), "unstable": rng.normal(4, 1, 500), "kind": ["y"] * 500}
    )
    table = feature_drift(reference, current, ["stable", "unstable", "kind"]).set_index("feature")
    assert table.loc["stable", "severity"] == "stable"
    assert table.loc["unstable", "psi"] > 0.2
    assert table.loc["kind", "psi"] > 0.2


def test_prediction_psi() -> None:
    rng = np.random.default_rng(3)
    reference = pd.Series(rng.normal(200_000, 40_000, 1000))
    current = pd.Series(rng.normal(200_000, 40_000, 1000))
    assert population_stability(reference, current) < 0.05


def test_conformal_quantiles_are_ordered() -> None:
    y_log = np.log1p(np.random.default_rng(4).normal(200_000, 30_000, 1000))
    prediction = y_log + np.random.default_rng(5).normal(0, 0.1, 1000)
    quantiles = conformal_quantiles(y_log, prediction, levels=(0.80, 0.90))
    assert quantiles["0.80"] < quantiles["0.90"]
    assert 0.05 < quantiles["0.80"] < 0.5


def _performance(**overrides) -> pd.DataFrame:
    rows = [
        {
            "window": "W1",
            "n": 100,
            "rmse": 21_000,
            "mae": 15_000,
            "mape": 9.0,
            "bias_pct": -1.0,
            "90_coverage": 0.90,
            "rmse_ratio_vs_baseline": 1.05,
        },
        {
            "window": "W2",
            "n": 100,
            "rmse": 21_500,
            "mae": 15_400,
            "mape": 9.1,
            "bias_pct": -1.2,
            "90_coverage": 0.89,
            "rmse_ratio_vs_baseline": 1.07,
        },
    ]
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _drift(n_drifted: list[int], prediction_psi: float = 0.05) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window": ["W1", "W2"],
            "n_drifted": n_drifted,
            "max_psi": [0.25, 0.25],
            "mean_psi": [0.08, 0.08],
            "top_drifted": [["mortgage_rate_pct"], ["mortgage_rate_pct"]],
            "prediction_psi": [prediction_psi, prediction_psi],
        }
    )


def test_policy_stays_calm_when_metrics_are_healthy() -> None:
    baseline = {"rmse": 20_000}
    result = evaluate_alerts(_performance(), _drift([1, 1]), baseline)
    assert result["decision"] == "continue_monitoring"
    assert result["n_critical"] == 0


def test_policy_recommends_retraining_for_sustained_coverage_erosion() -> None:
    baseline = {"rmse": 20_000}
    performance = _performance(**{"90_coverage": [0.82, 0.80]})
    result = evaluate_alerts(performance, _drift([5, 6]), baseline)
    assert result["decision"] == "retrain_now"
    assert result["n_critical"] >= 1
    assert any("coverage" in alert["rule"].lower() for alert in result["alerts"])


def test_policy_flags_input_drift_as_warning() -> None:
    baseline = {"rmse": 20_000}
    result = evaluate_alerts(_performance(), _drift([4, 5]), baseline)
    assert result["decision"] == "investigate_and_schedule"
    assert any("input drift" in alert["rule"].lower() for alert in result["alerts"])
