"""Replay of a post-deployment listing stream (Module 7).

Because the competition test set is unlabelled, monitoring is simulated with a
temporal backtest that respects the direction of time:

* the **reference model** is refitted on sales up to 31 December 2008, the
  vintage that would have been in production at the start of 2009;
* the **stream** consists of the 2009-2010 sales in half-year windows, which the
  reference model has never seen;
* observed sale prices are revealed as delayed labels, exactly as a platform
  would receive recorded deeds some weeks after listing.

Two scenarios are scored: the observed replay, and an adverse stress scenario
in which the contextual distributions are regenerated under a tighter credit
market and neighbourhood gentrification. The stress scenario is used only to
demonstrate that the alerting rules fire; it is not an estimate of future
accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

from revaluation.config import N_FOLDS, RANDOM_STATE, TARGET
from revaluation.data.features import MODEL_FEATURES, load_modeling_table, model_matrix
from revaluation.data.synthetic import DriftParameters, generate_context
from revaluation.models.metrics import (
    business_kpis,
    interval_metrics,
    regression_metrics,
)
from revaluation.models.train import conformal_quantiles

logger = logging.getLogger(__name__)

CONTEXT_COLUMNS = (
    "renovation_flag",
    "renovation_scope",
    "neigh_lat",
    "neigh_lon",
    "dist_school_km",
    "dist_hospital_km",
    "dist_transit_km",
    "dist_park_km",
    "school_rating",
    "walk_score",
    "crime_index",
    "amenity_density_1_5km",
    "park_area_ha",
    "amenity_score",
    "mortgage_rate_pct",
    "hpi_index",
    "hpi_yoy_pct",
    "months_supply",
    "local_unemployment_pct",
)

#: Features monitored for covariate drift. Numeric attributes plus the
#: categorical fields whose level mix is business-relevant.
MONITORED_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *[f for f in MODEL_FEATURES if f in (
                "lot_frontage",
                "lot_area",
                "overall_qual",
                "overall_cond",
                "year_built",
                "gr_liv_area",
                "total_sf",
                "total_bsmt_sf",
                "first_flr_sf",
                "second_flr_sf",
                "garage_area",
                "garage_cars",
                "fireplaces",
                "full_bath",
                "tot_rms_abv_grd",
                "age_at_sale",
                "remod_age",
                "amenity_score",
                "dist_school_km",
                "dist_hospital_km",
                "dist_transit_km",
                "dist_park_km",
                "school_rating",
                "walk_score",
                "crime_index",
                "amenity_density_1_5km",
                "park_area_ha",
                "mortgage_rate_pct",
                "hpi_index",
                "hpi_yoy_pct",
                "months_supply",
                "local_unemployment_pct",
                "liv_area_ratio",
                "quality_condition_index",
            )],
            "neighborhood",
            "ms_zoning",
            "house_style",
            "exter_qual",
            "kitchen_qual",
            "central_air",
            "garage_type",
            "foundation",
            "bldg_type",
        ]
    )
)


@dataclass
class MonitoringScenario:
    name: str
    label: str
    frame: pd.DataFrame
    prediction: np.ndarray
    lower80: np.ndarray
    upper80: np.ndarray
    lower90: np.ndarray
    upper90: np.ndarray


def assign_windows(frame: pd.DataFrame, n_windows: int = 4) -> pd.DataFrame:
    """Assign equal-volume chronological monitoring windows.

    Equal-volume windows keep the sampling error of every PSI and RMSE estimate
    comparable, which matters because the 2010 sales are concentrated in the
    first half of the year.
    """
    out = frame.copy()
    ranks = out["sale_date"].rank(method="first")
    codes = pd.qcut(ranks, n_windows, labels=False, duplicates="drop").astype(int)
    labels = {}
    for code in sorted(codes.unique()):
        subset = out.loc[codes == code]
        start = subset["sale_date"].min().strftime("%Y-%m")
        end = subset["sale_date"].max().strftime("%Y-%m")
        labels[code] = f"W{code + 1} {start} to {end}"
    out["window"] = codes.map(labels)
    return out


def build_reference_model(artifact: dict, reference: pd.DataFrame):
    """Refit the selected estimator on the pre-2009 vintage."""
    X_reference, y_reference = model_matrix(reference)
    y_log = np.log1p(y_reference.astype(float))
    pipeline = clone(artifact["pipeline"])
    pipeline.fit(X_reference, y_log)

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_log = cross_val_predict(pipeline, X_reference, y_log, cv=folds, n_jobs=1)
    quantiles = conformal_quantiles(y_log.to_numpy(), oof_log)
    baseline = regression_metrics(y_reference.to_numpy(), np.expm1(oof_log))
    baseline.update(business_kpis(y_reference.to_numpy(), np.expm1(oof_log)))
    return pipeline, quantiles, baseline


def score_frame(model, quantiles: dict, frame: pd.DataFrame) -> MonitoringScenario:
    X, _ = model_matrix(frame)
    log_prediction = model.predict(X)
    prediction = np.expm1(log_prediction)
    q80 = quantiles["0.80"]
    q90 = quantiles["0.90"]
    return MonitoringScenario(
        name="",
        label="",
        frame=frame,
        prediction=prediction,
        lower80=np.expm1(log_prediction - q80),
        upper80=np.expm1(log_prediction + q80),
        lower90=np.expm1(log_prediction - q90),
        upper90=np.expm1(log_prediction + q90),
    )


def build_stream() -> dict:
    """Construct reference model, windows and both monitoring scenarios."""
    import joblib

    from revaluation.config import MODELS_DIR

    artifact = joblib.load(MODELS_DIR / "final_model.joblib")
    frame = load_modeling_table()
    labeled = frame[frame[TARGET].notna()].copy()
    labeled["sale_date"] = pd.to_datetime(
        {"year": labeled["yr_sold"].astype(int), "month": labeled["mo_sold"].astype(int), "day": 15}
    )
    reference = labeled[labeled["sale_date"] < "2009-01-01"].copy()
    stream = labeled[labeled["sale_date"] >= "2009-01-01"].copy()
    if len(stream) < 200:
        raise ValueError("Insufficient post-2008 sales to simulate a monitoring stream.")

    model, quantiles, baseline = build_reference_model(artifact, reference)
    logger.info(
        "Reference model fitted on %s sales; baseline OOF RMSE=%.0f",
        len(reference),
        baseline["rmse"],
    )

    observed = score_frame(model, quantiles, stream)
    observed.name = "observed"
    observed.label = "Observed replay"

    drift = DriftParameters(rate_shift=1.5, supply_shift=1.0, amenity_improvement=0.6)
    properties = pd.DataFrame(
        {
            "Id": stream["id"].to_numpy(),
            "Neighborhood": stream["neighborhood"].to_numpy(),
            "YrSold": stream["yr_sold"].to_numpy(),
            "MoSold": stream["mo_sold"].to_numpy(),
            "YearBuilt": stream["year_built"].to_numpy(),
            "YearRemodAdd": stream["year_remod_add"].to_numpy(),
            "OverallQual": stream["overall_qual"].to_numpy(),
            "OverallCond": stream["overall_cond"].to_numpy(),
        }
    )
    context = generate_context(properties, seed=RANDOM_STATE, drift=drift)
    context = context.rename(columns={"Id": "id"})
    stress_frame = stream.drop(columns=list(CONTEXT_COLUMNS)).merge(
        context, on="id", how="left", validate="one_to_one"
    )
    stress_frame["sale_date"] = pd.to_datetime(
        {
            "year": stress_frame["yr_sold"].astype(int),
            "month": stress_frame["mo_sold"].astype(int),
            "day": 15,
        }
    )
    stress = score_frame(model, quantiles, stress_frame)
    stress.name = "stress"
    stress.label = "Adverse stress scenario"

    stream = assign_windows(stream)
    reference = reference.assign(window="reference")
    return {
        "artifact": artifact,
        "reference": reference,
        "stream": stream,
        "windows": sorted(stream["window"].unique()),
        "model": model,
        "quantiles": quantiles,
        "baseline": baseline,
        "scenarios": {
            "observed": observed,
            "stress": stress,
        },
    }


def window_slice(scenario: MonitoringScenario, stream: pd.DataFrame) -> dict[str, MonitoringScenario]:
    """Split a scored scenario into window-level slices."""
    slices = {}
    for window in sorted(stream["window"].unique()):
        mask = (stream["window"] == window).to_numpy()
        slices[window] = MonitoringScenario(
            name=scenario.name,
            label=scenario.label,
            frame=scenario.frame.loc[mask].reset_index(drop=True),
            prediction=scenario.prediction[mask],
            lower80=scenario.lower80[mask],
            upper80=scenario.upper80[mask],
            lower90=scenario.lower90[mask],
            upper90=scenario.upper90[mask],
        )
    return slices


def performance_table(
    scenario: MonitoringScenario,
    stream: pd.DataFrame,
    baseline: dict,
    quantiles: dict,
) -> pd.DataFrame:
    """Per-window accuracy, bias, interval coverage and business KPIs."""
    rows = []
    for window, slice_ in window_slice(scenario, stream).items():
        actual = slice_.frame[TARGET].to_numpy()
        prediction = slice_.prediction
        metrics = regression_metrics(actual, prediction)
        metrics.update(business_kpis(actual, prediction))
        metrics.update(
            interval_metrics(actual, slice_.lower80, slice_.upper80, 0.80, prefix="80_")
        )
        metrics.update(
            interval_metrics(actual, slice_.lower90, slice_.upper90, 0.90, prefix="90_")
        )
        error = prediction - actual
        metrics.update(
            {
                "window": window,
                "n": len(actual),
                "bias_pct": float(np.mean(error / actual) * 100),
                "rmse_ratio_vs_baseline": float(metrics["rmse"] / baseline["rmse"]),
                "mae_ratio_vs_baseline": float(metrics["mae"] / baseline["mae"]),
                "month_start": str(slice_.frame["sale_date"].min().date()),
                "month_end": str(slice_.frame["sale_date"].max().date()),
            }
        )
        rows.append(metrics)
    table = pd.DataFrame(rows)
    return table


def drift_table(
    reference: pd.DataFrame, scenario: MonitoringScenario, stream: pd.DataFrame, model, quantiles
) -> pd.DataFrame:
    """Feature and prediction drift for every window."""
    from revaluation.monitoring.drift import feature_drift, population_stability

    reference_predictions = np.expm1(model.predict(model_matrix(reference)[0]))
    rows = []
    for window, slice_ in window_slice(scenario, stream).items():
        table = feature_drift(reference, slice_.frame, list(MONITORED_FEATURES))
        table["window"] = window
        prediction_psi = population_stability(
            pd.Series(reference_predictions), pd.Series(slice_.prediction)
        )
        table.loc[len(table)] = {
            "feature": "__prediction__",
            "dtype": "numeric",
            "psi": round(float(prediction_psi), 4),
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "total_variation": np.nan,
            "severity": (
                "significant" if prediction_psi >= 0.20 else "moderate" if prediction_psi >= 0.10 else "stable"
            ),
            "reference_mean": float(np.mean(reference_predictions)),
            "current_mean": float(np.mean(slice_.prediction)),
            "reference_mode": None,
            "current_mode": None,
            "window": window,
        }
        rows.append(table)
    return pd.concat(rows, ignore_index=True)
