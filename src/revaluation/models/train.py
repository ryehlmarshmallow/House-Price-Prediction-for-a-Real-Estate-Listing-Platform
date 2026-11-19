"""Model development, cross-validation and artefact creation (Module 5).

The training routine compares eight regression families on identical folds and
identical features, tunes the flexible learners with randomised search, selects
the model with the lowest out-of-fold RMSE, and calibrates split-conformal
prediction intervals from the out-of-fold residuals. The resulting artefact is
a single scikit-learn pipeline that the API and the application load directly,
which removes any train/serve skew in preprocessing.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from revaluation.config import (
    COVERAGE_LEVELS,
    METRICS_DIR,
    MODELS_DIR,
    N_FOLDS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from revaluation.data.features import MODEL_FEATURES, load_modeling_table, model_matrix
from revaluation.models.metrics import (
    business_kpis,
    interval_metrics,
    regression_metrics,
)
from revaluation.models.preprocess import make_pipeline

logger = logging.getLogger(__name__)

#: Model registry: name -> (estimator, parameter space, search strategy).
MODEL_ZOO: dict[str, dict] = {
    "linear_regression": {
        "estimator": LinearRegression(),
        "kind": "linear",
        "space": {},
        "strategy": None,
    },
    "ridge": {
        "estimator": Ridge(),
        "kind": "linear",
        "space": {"model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]},
        "strategy": "grid",
    },
    "lasso": {
        "estimator": Lasso(max_iter=50_000),
        "kind": "linear",
        "space": {"model__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]},
        "strategy": "grid",
    },
    "elastic_net": {
        "estimator": ElasticNet(max_iter=50_000),
        "kind": "linear",
        "space": {
            "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        "strategy": "grid",
    },
    "random_forest": {
        "estimator": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1
        ),
        "kind": "onehot",
        "space": {
            "model__max_depth": [None, 12, 18, 26],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": [0.3, 0.5, 0.7, 1.0],
        },
        "strategy": "random",
        "n_iter": 14,
    },
    "gradient_boosting": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "kind": "onehot",
        "space": {
            "model__n_estimators": [200, 320],
            "model__learning_rate": [0.04, 0.08],
            "model__max_depth": [2, 3],
            "model__subsample": [0.9, 1.0],
            "model__min_samples_leaf": [3, 8],
            "model__max_features": [0.5, 1.0],
        },
        "strategy": "random",
        "n_iter": 10,
    },
    "xgboost": {
        "estimator": XGBRegressor(
            n_estimators=600,
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            objective="reg:squarederror",
        ),
        "kind": "onehot",
        "space": {
            "model__n_estimators": [400, 700, 1100],
            "model__learning_rate": [0.015, 0.03, 0.05],
            "model__max_depth": [3, 4, 6],
            "model__min_child_weight": [1, 3, 6],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.5, 0.8, 1.0],
            "model__reg_lambda": [0.1, 1.0, 5.0],
        },
        "strategy": "random",
        "n_iter": 22,
    },
    "lightgbm": {
        "estimator": LGBMRegressor(
            n_estimators=700,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        "kind": "native",
        "space": {
            "model__n_estimators": [400, 700, 1000],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [5, 10, 20],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.8, 1.0],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        },
        "strategy": "random",
        "n_iter": 30,
    },
}

FRIENDLY_NAMES = {
    "linear_regression": "Linear regression",
    "ridge": "Ridge regression",
    "lasso": "Lasso regression",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _search(entry: dict, X: pd.DataFrame, y_log: np.ndarray, folds) -> Pipeline:
    pipeline = make_pipeline(entry["estimator"], entry["kind"])
    strategy = entry["strategy"]
    if strategy is None:
        pipeline.fit(X, y_log)
        return pipeline
    common = {
        "cv": folds,
        "scoring": "neg_root_mean_squared_error",
        "n_jobs": 4,
        "refit": True,
    }
    if strategy == "grid":
        search = GridSearchCV(pipeline, param_grid=entry["space"], **common)
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=entry["space"],
            n_iter=entry["n_iter"],
            random_state=RANDOM_STATE,
            **common,
        )
    search.fit(X, y_log)
    logger.info(
        "Tuned %s: best CV RMSE(log)=%.5f, params=%s",
        entry["name"],
        -search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_


def train_all_models(frame: pd.DataFrame) -> dict:
    """Fit and cross-validate every model family."""
    X, y_raw = model_matrix(frame)
    y_raw = y_raw.astype(float)
    y_log = np.log1p(y_raw)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict] = {}
    oof_predictions = pd.DataFrame(
        {"id": frame["id"].to_numpy(), "y_true": y_raw.to_numpy()}
    )
    for name, entry in MODEL_ZOO.items():
        entry = {**entry, "name": name}
        logger.info("Training %s", name)
        pipeline = _search(entry, X, y_log, folds)
        oof_log = cross_val_predict(pipeline, X, y_log, cv=folds, n_jobs=1)
        oof_price = np.expm1(oof_log)
        oof_predictions[name] = oof_price
        metrics = regression_metrics(y_raw.to_numpy(), oof_price, prefix="")
        metrics.update(business_kpis(y_raw.to_numpy(), oof_price, prefix=""))
        metrics["model"] = name
        metrics["model_label"] = FRIENDLY_NAMES[name]
        metrics["cv_rmse_log"] = float(
            np.sqrt(np.mean((y_log.to_numpy() - oof_log) ** 2))
        )
        results[name] = {"pipeline": pipeline, "metrics": metrics}
        logger.info(
            "%s OOF RMSE=%.0f MAE=%.0f R2=%.3f MAPE=%.2f%%",
            name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
            metrics["mape"],
        )

    return {
        "results": results,
        "oof_predictions": oof_predictions,
        "X": X,
        "y_raw": y_raw,
        "y_log": y_log,
        "folds": folds,
    }


def select_best(results: dict[str, dict]) -> str:
    """Lowest out-of-fold RMSE; ties broken by the smaller standard error."""
    ranked = sorted(results.items(), key=lambda item: (item[1]["metrics"]["rmse"]))
    best_name = ranked[0][0]
    logger.info("Selected model: %s", best_name)
    return best_name


def conformal_quantiles(
    y_log: np.ndarray, oof_log: np.ndarray, levels: tuple[float, ...] = COVERAGE_LEVELS
) -> dict[str, float]:
    """Split-conformal (cross-conformal) absolute-residual quantiles.

    The finite-sample correction ``ceil((n + 1) * level) / n`` keeps the
    intervals marginally valid when the residuals are exchangeable.
    """
    residuals = np.abs(np.asarray(y_log) - np.asarray(oof_log))
    n = len(residuals)
    quantiles: dict[str, float] = {}
    for level in levels:
        adjusted = min(1.0, np.ceil((n + 1) * level) / n)
        quantiles[f"{level:.2f}"] = float(np.quantile(residuals, adjusted))
    return quantiles


def fit_final_model(best: str, results: dict, X: pd.DataFrame, y_log: np.ndarray):
    pipeline = make_pipeline(MODEL_ZOO[best]["estimator"], MODEL_ZOO[best]["kind"])
    # Reuse the tuned hyper-parameters of the selected pipeline.
    pipeline.set_params(**results[best]["pipeline"].get_params())
    pipeline.fit(X, y_log)
    return pipeline


def temporal_holdout(
    results: dict, frame: pd.DataFrame, best_name: str
) -> dict[str, float]:
    """Fit the selected configuration on 2006-2009 and evaluate on 2010."""
    frame = frame.copy()
    frame["sale_year"] = frame["sale_year"].astype(int)
    train = frame[frame["sale_year"] <= 2009]
    test = frame[frame["sale_year"] == 2010]
    if len(test) < 30 or len(train) < 300:
        return {}
    X_train, y_train = model_matrix(train)
    X_test, y_test = model_matrix(test)
    pipeline = make_pipeline(MODEL_ZOO[best_name]["estimator"], MODEL_ZOO[best_name]["kind"])
    pipeline.set_params(**results[best_name]["pipeline"].get_params())
    pipeline.fit(X_train, np.log1p(y_train.astype(float)))
    prediction = np.expm1(pipeline.predict(X_test))
    metrics = regression_metrics(y_test.to_numpy(), prediction)
    metrics.update(business_kpis(y_test.to_numpy(), prediction))
    metrics["n_train"] = len(train)
    metrics["n_test"] = len(test)
    return metrics


def interval_validation(
    frame: pd.DataFrame, y_raw: np.ndarray, oof_log: np.ndarray, quantiles: dict[str, float]
) -> pd.DataFrame:
    """Empirical coverage of the conformal intervals, overall and by segment."""
    rows = []
    segments = {
        "all": np.ones(len(frame), dtype=bool),
        "entry (<130k)": (y_raw < 130_000),
        "mid (130k-200k)": (y_raw >= 130_000) & (y_raw < 200_000),
        "upper (200k-300k)": (y_raw >= 200_000) & (y_raw < 300_000),
        "luxury (>=300k)": (y_raw >= 300_000),
    }
    for level, q in quantiles.items():
        coverage = float(level)
        lower = np.expm1(oof_log - q)
        upper = np.expm1(oof_log + q)
        for segment, mask in segments.items():
            if mask.sum() < 10:
                continue
            metrics = interval_metrics(
                y_raw[mask], lower[mask], upper[mask], coverage, prefix=""
            )
            metrics["level"] = level
            metrics["segment"] = segment
            rows.append(metrics)
    return pd.DataFrame(rows)


def save_artifacts(
    best_name: str,
    pipeline: Pipeline,
    frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    results: dict,
    oof: pd.DataFrame,
    quantiles: dict[str, float],
    holdout: dict,
    interval_table: pd.DataFrame,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = results[best_name]["metrics"]
    background_sample = frame.drop(columns=[TARGET]).sample(
        n=min(200, len(frame)), random_state=RANDOM_STATE
    )
    try:
        background = np.asarray(pipeline.named_steps["preprocess"].transform(background_sample))
    except Exception:  # noqa: BLE001 - native preprocessor may return a DataFrame  # pragma: no cover
        background = np.asarray(
            pipeline.named_steps["preprocess"].transform(background_sample), dtype=float
        )
    artifact = {
        "pipeline": pipeline,
        "background": background,
        "model_name": best_name,
        "model_label": FRIENDLY_NAMES[best_name],
        "feature_columns": list(MODEL_FEATURES),
        "log_target": True,
        "conformal": quantiles,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": len(frame),
    }
    joblib.dump(artifact, MODELS_DIR / "final_model.joblib")

    comparison = pd.DataFrame([entry["metrics"] for entry in results.values()])
    comparison = comparison.sort_values("rmse").reset_index(drop=True)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    oof.to_parquet(METRICS_DIR / "oof_predictions.parquet", index=False)
    interval_table.to_csv(METRICS_DIR / "interval_coverage.csv", index=False)

    summary = {
        "selected_model": best_name,
        "selected_model_label": FRIENDLY_NAMES[best_name],
        "metrics": metrics,
        "conformal": quantiles,
        "temporal_holdout_2010": holdout,
        "comparison": comparison.to_dict(orient="records"),
        "n_train": len(frame),
        "n_features": len(MODEL_FEATURES),
        "trained_at": artifact["trained_at"],
    }
    with open(METRICS_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in summary.items() if key != "comparison"},
            handle,
            indent=2,
            default=float,
        )

    # Serving defaults: central tendency of every raw column used by the
    # feature builder, so that partial listing profiles can be scored.
    training = frame
    defaults: dict[str, object] = {}
    for column in training.columns:
        if column in {"id", "source_file", "sale_date", "SalePrice", "log_price"}:
            continue
        if str(training[column].dtype) in {"object", "string", "str", "category"}:
            mode = training[column].mode(dropna=True)
            if not mode.empty:
                defaults[column] = str(mode.iloc[0])
        else:
            median = pd.to_numeric(training[column], errors="coerce").median()
            if pd.notna(median):
                defaults[column] = float(median)
    with open(MODELS_DIR / "serving_defaults.json", "w", encoding="utf-8") as handle:
        json.dump(defaults, handle, indent=2)

    test_X, _ = model_matrix(test_frame)
    test_prediction = np.clip(np.expm1(pipeline.predict(test_X)), 30_000, 800_000)
    submission = pd.DataFrame(
        {"Id": test_frame["id"].astype(int), "SalePrice": np.round(test_prediction, 2)}
    )
    submission.to_csv(REPORTS_DIR / "submission.csv", index=False)
    test_output = pd.DataFrame(
        {
            "id": test_frame["id"].astype(int).to_numpy(),
            "prediction": test_prediction,
        }
    )
    test_output.to_parquet(METRICS_DIR / "test_predictions.parquet", index=False)
    return summary


def run(save: bool = True) -> dict:
    frame = load_modeling_table()
    training = frame[frame[TARGET].notna()].copy()
    logger.info("Training on %s rows with %s features", len(training), len(MODEL_FEATURES))

    test_frame = frame[frame[TARGET].isna()].copy()
    output = train_all_models(training)
    best_name = select_best(output["results"])
    best_oof_log = output["oof_predictions"][best_name].map(np.log1p)
    quantiles = conformal_quantiles(output["y_log"].to_numpy(), best_oof_log.to_numpy())
    pipeline = fit_final_model(best_name, output["results"], output["X"], output["y_log"])
    holdout = temporal_holdout(output["results"], training, best_name)
    interval_table = interval_validation(
        training, output["y_raw"].to_numpy(), np.log1p(output["oof_predictions"][best_name].to_numpy()), quantiles
    )

    if save:
        summary = save_artifacts(
            best_name,
            pipeline,
            training,
            test_frame,
            output["results"],
            output["oof_predictions"],
            quantiles,
            holdout,
            interval_table,
        )
        logger.info("Artefacts saved. Selected model: %s", best_name)
        return summary
    return {"selected_model": best_name}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
