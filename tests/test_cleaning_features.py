"""Tests for cleaning, label normalisation and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revaluation.data.clean import (
    CleaningState,
    apply_cleaning,
    build_clean_dataset,
    normalise_labels,
)
from revaluation.data.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    load_modeling_table,
    model_matrix,
)


@pytest.fixture(scope="module")
def cleaned():
    frame, payload = build_clean_dataset(save=False)
    return frame, payload


def test_cleaning_preserves_expected_rows(cleaned) -> None:
    _, payload = cleaned
    assert payload["rows"]["train_clean"] == 1458
    assert payload["rows"]["test_clean"] == 1459
    assert payload["report"]["outliers_removed"] == 2


def test_no_missing_values_remain(cleaned) -> None:
    frame, _ = cleaned
    # The target and the target-derived analysis fields are legitimately absent
    # for the unlabelled test portion; every model input must be complete.
    engineered = engineer_features(frame)
    X, _ = model_matrix(engineered)
    assert X.isna().sum().sum() == 0


def test_labels_are_normalised() -> None:
    frame = pd.DataFrame({"Neighborhood": [" collgcr ", "COLLGCR", "NoRidge", "NewPlace"]})
    vocabulary = {"Neighborhood": {"collgcr": "CollgCr", "noridge": "NoRidge"}}
    normalised, fixed = normalise_labels(frame, vocabulary)
    assert normalised["Neighborhood"].tolist() == [
        "CollgCr",
        "CollgCr",
        "NoRidge",
        "NewPlace",
    ]
    assert fixed == 2


def test_structural_absence_semantics() -> None:
    row = {
        "id": 1,
        "neighborhood": "NAmes",
        "ms_sub_class": 20,
        "lot_frontage": np.nan,
        "lot_area": 8000.0,
        "yr_sold": 2010,
        "year_built": 1990,
        "garage_area": 0.0,
        "garage_cars": 0.0,
        "garage_yr_blt": np.nan,
        "pool_area": 0.0,
        "alley": np.nan,
        "bsmt_qual": np.nan,
        "bsmt_exposure": np.nan,
        "mas_vnr_area": np.nan,
        "total_bsmt_sf": np.nan,
    }
    frame = pd.DataFrame([row])
    state = CleaningState(lot_frontage_global_median=70.0)
    for column in ("alley", "bsmt_qual", "bsmt_exposure"):
        state.mode_by_column[column] = "TA"
    cleaned, _ = apply_cleaning(frame, state, is_training=False)
    assert cleaned.loc[0, "alley"] == "None"
    assert cleaned.loc[0, "bsmt_qual"] == "None"
    assert cleaned.loc[0, "mas_vnr_area"] == 0.0
    assert cleaned.loc[0, "lot_frontage"] == 70.0
    assert cleaned.loc[0, "garage_yr_blt"] == 1990


def test_engineered_formulas() -> None:
    frame = pd.DataFrame(
        {
            "total_bsmt_sf": [1000.0],
            "first_flr_sf": [1200.0],
            "second_flr_sf": [800.0],
            "full_bath": [2.0],
            "half_bath": [1.0],
            "bsmt_full_bath": [1.0],
            "bsmt_half_bath": [0.0],
            "wood_deck_sf": [100.0],
            "open_porch_sf": [50.0],
            "enclosed_porch": [0.0],
            "three_ssn_porch": [0.0],
            "screen_porch": [0.0],
            "yr_sold": [2010],
            "year_built": [1990],
            "year_remod_add": [2005],
            "pool_area": [0.0],
            "garage_area": [500.0],
            "garage_cars": [2.0],
            "fireplaces": [1.0],
            "gr_liv_area": [2000.0],
            "lot_area": [10000.0],
            "tot_rms_abv_grd": [8.0],
            "overall_qual": [7.0],
            "overall_cond": [5.0],
            "renovation_scope": ["Moderate"],
        }
    )
    engineered = engineer_features(frame)
    assert engineered.loc[0, "total_sf"] == 3000.0
    assert engineered.loc[0, "total_bath"] == 3.5
    assert engineered.loc[0, "age_at_sale"] == 20
    assert engineered.loc[0, "is_renovated"] == 1
    assert engineered.loc[0, "renovation_scope_code"] == 2
    assert engineered.loc[0, "quality_condition_index"] == 35.0
    assert engineered.loc[0, "has_garage"] == 1


def test_model_matrix_contains_no_leakage_columns() -> None:
    frame = load_modeling_table()
    X, y = model_matrix(frame)
    forbidden = {
        "SalePrice",
        "log_price",
        "price_per_sqm",
        "days_on_market",
        "sale_type",
        "sale_condition",
        "yr_sold",
        "mo_sold",
        "sale_date",
    }
    assert forbidden.isdisjoint(X.columns)
    assert y is not None and len(X) == len(y)
    assert set(X.columns) == set(MODEL_FEATURES)
    assert len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES) > len(MODEL_FEATURES)
