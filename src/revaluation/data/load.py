"""Loading and validation of the Kaggle Ames housing data set.

The raw competition files are treated as immutable inputs. Every downstream
artifact is produced by the pipeline from these files together with the
synthetic contextual data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from revaluation.config import ID_COLUMN, TARGET, TEST_CSV, TRAIN_CSV

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_COLUMNS = 81
EXPECTED_TEST_COLUMNS = 80


def _read_csv(path, expected_columns: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the competition file at {path}. "
            "Place train.csv and test.csv in the project data directory."
        )
    frame = pd.read_csv(path, keep_default_na=True)
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"{path.name} has {frame.shape[1]} columns, expected {expected_columns}."
        )
    return frame


def load_train() -> pd.DataFrame:
    """Return the labelled competition training frame (1,460 rows)."""
    frame = _read_csv(TRAIN_CSV, EXPECTED_TRAIN_COLUMNS)
    if TARGET not in frame.columns:
        raise ValueError(f"Training frame is missing the target column '{TARGET}'.")
    logger.info("Loaded training data: %s rows x %s columns", *frame.shape)
    return frame


def load_test() -> pd.DataFrame:
    """Return the unlabelled competition test frame (1,459 rows)."""
    frame = _read_csv(TEST_CSV, EXPECTED_TEST_COLUMNS)
    logger.info("Loaded test data: %s rows x %s columns", *frame.shape)
    return frame


def load_combined() -> pd.DataFrame:
    """Concatenate train and test, preserving the file of origin."""
    train = load_train().assign(source_file="train", log_price=lambda d: np.log1p(d[TARGET]))
    test = load_test().assign(source_file="test", log_price=float("nan"))
    combined = pd.concat([train, test], ignore_index=True)
    if combined[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate property identifiers found across train and test.")
    logger.info("Combined frame: %s rows", len(combined))
    return combined


def dataset_profile() -> dict:
    """Small structural summary used by the data dictionary and the report."""
    train = load_train()
    test = load_test()
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": int(train.shape[1]),
        "numeric_columns": int(train.select_dtypes("number").shape[1]),
        "categorical_columns": int(train.select_dtypes(include=["object", "str"]).shape[1]),
        "target_min": float(train[TARGET].min()),
        "target_max": float(train[TARGET].max()),
        "target_mean": float(train[TARGET].mean()),
        "target_median": float(train[TARGET].median()),
        "sale_year_min": int(train["YrSold"].min()),
        "sale_year_max": int(train["YrSold"].max()),
        "neighbourhoods": int(train["Neighborhood"].nunique()),
        "source": "Kaggle House Prices - Advanced Regression Techniques (Ames, Iowa)",
    }
