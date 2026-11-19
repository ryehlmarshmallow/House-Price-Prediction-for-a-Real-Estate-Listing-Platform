"""Model explanation utilities based on SHAP (Module 5, residual analysis).

Tree ensembles are explained with TreeSHAP; the linear benchmarks are explained
with their standardised coefficients. The same code path serves the offline
report and the local explanations shown in the application, which keeps the
narrative consistent between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from revaluation.data.features import MODEL_FEATURES, feature_group_table


def transformed_matrix(pipeline: Pipeline, X: pd.DataFrame):
    """Return the model-ready matrix and its column names."""
    preprocess = pipeline.named_steps["preprocess"]
    matrix = preprocess.transform(X)
    names = list(preprocess.get_feature_names_out())
    if isinstance(matrix, pd.DataFrame):
        return matrix, list(matrix.columns)
    return np.asarray(matrix), names


def _explainer(pipeline: Pipeline, background=None):
    model = pipeline.named_steps["model"]
    kind = type(model).__name__
    if kind in {"LGBMRegressor", "XGBRegressor", "RandomForestRegressor", "GradientBoostingRegressor"}:
        return shap.TreeExplainer(model), "tree"
    if hasattr(model, "coef_"):
        if background is None:
            raise ValueError("Linear explanations require a background sample.")
        return shap.LinearExplainer(model, background), "linear"
    raise TypeError(f"No explanation backend available for {kind}")


def global_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 300,
    seed: int = 42,
    background=None,
) -> pd.DataFrame:
    """Mean absolute SHAP value per raw feature, aggregated to business groups."""
    sample = X.sample(n=min(max_samples, len(X)), random_state=seed)
    matrix, names = transformed_matrix(pipeline, sample)
    masker = background if background is not None else matrix
    explainer, _ = _explainer(pipeline, background=masker if kind_is_linear(pipeline) else None)
    values = explainer.shap_values(matrix)
    values = np.asarray(values)
    if values.ndim == 3:  # some explainers return (n, features, outputs)
        values = values[:, :, 0]
    importance = pd.DataFrame(
        {"encoded_feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    )

    # Map one-hot encoded columns back to their raw feature.
    def raw_feature(encoded: str) -> str:
        if encoded in MODEL_FEATURES:
            return encoded
        for candidate in sorted(MODEL_FEATURES, key=len, reverse=True):
            if encoded.startswith(f"{candidate}_"):
                return candidate
        return encoded

    importance["feature"] = importance["encoded_feature"].map(raw_feature)
    aggregated = (
        importance.groupby("feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    groups = feature_group_table().set_index("feature")["group"]
    aggregated["group"] = aggregated["feature"].map(groups).fillna("Other")
    total = aggregated["mean_abs_shap"].sum()
    aggregated["share_pct"] = 100.0 * aggregated["mean_abs_shap"] / max(total, 1e-12)
    return aggregated.reset_index(drop=True)


def kind_is_linear(pipeline: Pipeline) -> bool:
    return hasattr(pipeline.named_steps["model"], "coef_")


def shap_matrix(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 400,
    seed: int = 42,
    background=None,
):
    """Return (shap_values, feature_frame) for beeswarm visualisation."""
    sample = X.sample(n=min(max_samples, len(X)), random_state=seed)
    matrix, names = transformed_matrix(pipeline, sample)
    masker = background if background is not None else matrix
    explainer, _ = _explainer(pipeline, background=masker if kind_is_linear(pipeline) else None)
    values = np.asarray(explainer.shap_values(matrix))
    if values.ndim == 3:
        values = values[:, :, 0]
    raw_names = []
    for encoded in names:
        raw = encoded
        for candidate in sorted(MODEL_FEATURES, key=len, reverse=True):
            if encoded.startswith(f"{candidate}_"):
                raw = candidate
                break
        raw_names.append(raw)
    frame = pd.DataFrame(values, columns=[f"{name}_{i}" for i, name in enumerate(raw_names)])
    frame.columns = [
        name if list(frame.columns).count(name) == 1 else f"{name}__{i}"
        for i, name in enumerate(raw_names)
    ]
    return values, pd.DataFrame(matrix), names


def local_explanation(
    pipeline: Pipeline,
    row: pd.DataFrame,
    top_k: int = 5,
    background=None,
) -> list[dict]:
    """Per-feature contributions for a single listing, in log-price units."""
    matrix, names = transformed_matrix(pipeline, row)
    arr = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    explainer, _ = _explainer(pipeline, background=background)
    values = np.asarray(explainer.shap_values(matrix))
    if values.ndim > 1:
        values = values[0]
    records = []
    for index, encoded in enumerate(names):
        raw = encoded
        for candidate in sorted(MODEL_FEATURES, key=len, reverse=True):
            if encoded.startswith(f"{candidate}_"):
                raw = candidate
                break
        records.append(
            {
                "feature": raw,
                "encoded_feature": encoded,
                "contribution_log": float(values[index]),
                "value": float(arr[0, index]) if index < arr.shape[1] else None,
            }
        )
    frame = pd.DataFrame(records)
    aggregated = frame.groupby("feature", as_index=False)["contribution_log"].sum()
    aggregated["abs"] = aggregated["contribution_log"].abs()
    aggregated = aggregated.sort_values("abs", ascending=False).head(top_k)
    return [
        {
            "feature": record["feature"],
            "contribution_log": float(record["contribution_log"]),
            "direction": "increases" if record["contribution_log"] > 0 else "decreases",
            "relative_effect_pct": float(np.expm1(abs(record["contribution_log"])) * 100),
        }
        for _, record in aggregated.iterrows()
    ]
