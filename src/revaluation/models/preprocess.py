"""Preprocessing pipelines shared by training and serving (Module 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from revaluation.data.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

NUMERIC_USED: tuple[str, ...] = tuple(c for c in NUMERIC_FEATURES if c in MODEL_FEATURES)
CATEGORICAL_USED: tuple[str, ...] = tuple(
    c for c in CATEGORICAL_FEATURES if c in MODEL_FEATURES
)


class NativePreprocessor(BaseEstimator, TransformerMixin):
    """Median imputation plus native categorical dtype handling.

    Kept outside a ColumnTransformer because the latter densifies the output
    and would strip the pandas category dtype that LightGBM relies on.
    """

    def fit(self, X: pd.DataFrame, y=None) -> NativePreprocessor:
        self.numeric_medians_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        for column in NUMERIC_USED:
            if column in X.columns:
                values = pd.to_numeric(X[column], errors="coerce")
                median = float(values.median())
                self.numeric_medians_[column] = (
                    median if np.isfinite(median) else 0.0
                )
        for column in CATEGORICAL_USED:
            if column in X.columns:
                values = X[column].astype("object").dropna().astype(str)
                self.categories_[column] = sorted(values.unique().tolist())
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for column in NUMERIC_USED:
            if column in X.columns:
                values = pd.to_numeric(X[column], errors="coerce").fillna(
                    self.numeric_medians_.get(column, 0.0)
                )
                out[column] = values.astype(float)
        for column in CATEGORICAL_USED:
            if column in X.columns:
                values = X[column].astype("object")
                out[column] = pd.Categorical(
                    values.astype(str).where(values.notna(), None),
                    categories=self.categories_.get(column, []),
                )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            [c for c in (*NUMERIC_USED, *CATEGORICAL_USED) if c in self.feature_names_in_],
            dtype=object,
        )


def make_preprocessor(kind: str) -> ColumnTransformer:
    """Build a preprocessing transformer.

    ``kind`` is one of ``linear`` (impute and standardise, one-hot encode),
    ``onehot`` (impute, one-hot encode, no scaling) or ``native`` (impute only,
    categorical columns passed through as pandas categories).
    """
    if kind == "native":
        return NativePreprocessor()
    if kind not in {"linear", "onehot"}:
        raise ValueError(f"Unknown preprocessor kind: {kind}")

    numeric_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if kind == "linear":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_USED)),
            ("cat", categorical, list(CATEGORICAL_USED)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, kind: str) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(kind)), ("model", estimator)])
