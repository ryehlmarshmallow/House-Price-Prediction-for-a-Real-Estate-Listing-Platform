"""Regression metrics and business-oriented KPIs (Module 5).

Prices are modelled on the natural-log scale to stabilise variance and to make
the multiplicative error structure explicit. All reported metrics are computed
on the original dollar scale after back-transformation, because that is the
scale on which agents and sellers act.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from revaluation.config import (
    COST_OVER_VALUATION,
    COST_UNDER_VALUATION,
    ERROR_TOLERANCE_PCT,
    ERROR_TOLERANCE_USD,
)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE and R2 on the dollar scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_pred <= 0):
        y_pred = np.clip(y_pred, 1.0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}mape": mape,
        f"{prefix}r2": r2,
        f"{prefix}median_ae": float(np.median(np.abs(y_true - y_pred))),
        f"{prefix}rmse_log": float(
            np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
        ),
        f"{prefix}n": int(mask.sum()),
    }


def business_kpis(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    """Business-facing KPIs derived from the over/under-valuation loss model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    absolute = np.abs(error)
    over = np.clip(error, 0, None)
    under = np.clip(-error, 0, None)
    asymmetric_cost = COST_OVER_VALUATION * over + COST_UNDER_VALUATION * under
    relative = absolute / np.clip(y_true, 1.0, None)
    return {
        f"{prefix}within_tolerance_usd": float(
            np.mean(absolute <= ERROR_TOLERANCE_USD) * 100.0
        ),
        f"{prefix}within_tolerance_pct": float(
            np.mean(relative * 100.0 <= ERROR_TOLERANCE_PCT) * 100.0
        ),
        f"{prefix}over_valuation_rate": float(np.mean(error > 0) * 100.0),
        f"{prefix}mean_asymmetric_cost": float(np.mean(asymmetric_cost)),
        f"{prefix}median_asymmetric_cost": float(np.median(asymmetric_cost)),
        f"{prefix}total_asymmetric_cost": float(np.sum(asymmetric_cost)),
    }


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
    prefix: str = "",
) -> dict[str, float]:
    """Empirical coverage and sharpness of a prediction interval."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return {
        f"{prefix}coverage": float(inside.mean()),
        f"{prefix}coverage_gap": float(inside.mean() - coverage),
        f"{prefix}mean_width": float(width.mean()),
        f"{prefix}median_width": float(np.median(width)),
        f"{prefix}mean_width_pct": float(
            np.mean(width / np.clip(y_true, 1.0, None)) * 100.0
        ),
    }


def metrics_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten a list of metric dictionaries into a tidy comparison table."""
    return pd.DataFrame(records)
