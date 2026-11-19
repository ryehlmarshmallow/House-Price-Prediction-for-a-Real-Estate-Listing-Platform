"""Feature engineering and selection (Module 4).

Derived features follow three principles:

1. **Temporal validity.** Only information observable at listing time is used.
   Post-transaction fields (sale condition, sale type, days on market) are kept
   in the analytical table but excluded from the model inputs.
2. **Interpretability.** Every engineered variable has a business meaning that
   an agent can explain to a seller.
3. **Collinearity awareness.** Several raw columns are algebraic combinations
   of others (total basement area, first/second floor area). Composites are
   built deliberately and their variance inflation is measured before the
   linear models are fitted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from revaluation.config import (
    CONTRACT,
    DATA_PROCESSED,
    NEIGHBOURHOOD_SEGMENT,
    TARGET,
)
from revaluation.data.clean import load_cleaned

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "ms_sub_class",
    "ms_zoning",
    "street",
    "alley",
    "lot_shape",
    "land_contour",
    "utilities",
    "lot_config",
    "land_slope",
    "neighborhood",
    "condition1",
    "condition2",
    "bldg_type",
    "house_style",
    "roof_style",
    "roof_matl",
    "exterior1st",
    "exterior2nd",
    "mas_vnr_type",
    "exter_qual",
    "exter_cond",
    "foundation",
    "bsmt_qual",
    "bsmt_cond",
    "bsmt_exposure",
    "bsmt_fin_type1",
    "bsmt_fin_type2",
    "heating",
    "heating_qc",
    "central_air",
    "electrical",
    "kitchen_qual",
    "functional",
    "fireplace_qu",
    "garage_type",
    "garage_finish",
    "garage_qual",
    "garage_cond",
    "paved_drive",
    "pool_qc",
    "fence",
    "misc_feature",
    "renovation_scope",
)

NUMERIC_FEATURES: tuple[str, ...] = CONTRACT.numeric

#: Columns that are constant or almost constant in the Ames sample and carry no
#: generalisable information; they are retained in the data set but excluded
#: from modelling to avoid degenerate one-hot columns.
UNINFORMATIVE_FEATURES: tuple[str, ...] = (
    "utilities",
    "street",
    "pool_qc",
    # Exact duplicate of the engineered is_renovated flag.
    "renovation_flag",
)

MODEL_FEATURES: tuple[str, ...] = tuple(
    column
    for column in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    if column not in UNINFORMATIVE_FEATURES
)

RENOVATION_SCOPE_CODES = {"None": 0, "Minor": 1, "Moderate": 2, "Major": 3}

#: Exact algebraic composites of other numeric columns. Including them and
#: their components in the same linear design would be rank-deficient by
#: construction, so VIF is computed on the identifiable core and the
#: composites are documented separately.
VIF_EXCLUDED_COMPOSITES: tuple[str, ...] = (
    "total_sf",
    "total_bath",
    "total_porch_sf",
    "gr_liv_area",
    "total_bsmt_sf",
    # Ages are algebraically linked to the construction years; the age
    # variables are retained because they carry the business meaning.
    "year_built",
    "year_remod_add",
)
VIF_FEATURES: tuple[str, ...] = tuple(
    column
    for column in NUMERIC_FEATURES
    if column not in VIF_EXCLUDED_COMPOSITES and column in MODEL_FEATURES
)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables; deterministic and row-wise (safe for serving)."""
    out = frame.copy()

    total_bsmt = out["total_bsmt_sf"].fillna(0)
    out["total_sf"] = (
        total_bsmt + out["first_flr_sf"].fillna(0) + out["second_flr_sf"].fillna(0)
    )
    out["total_bath"] = (
        out["full_bath"].fillna(0)
        + 0.5 * out["half_bath"].fillna(0)
        + out["bsmt_full_bath"].fillna(0)
        + 0.5 * out["bsmt_half_bath"].fillna(0)
    )
    out["total_porch_sf"] = (
        out["wood_deck_sf"].fillna(0)
        + out["open_porch_sf"].fillna(0)
        + out["enclosed_porch"].fillna(0)
        + out["three_ssn_porch"].fillna(0)
        + out["screen_porch"].fillna(0)
    )
    out["age_at_sale"] = out["yr_sold"] - out["year_built"]
    out["remod_age"] = out["yr_sold"] - out["year_remod_add"]
    out["is_renovated"] = (out["year_remod_add"] > out["year_built"]).astype(int)
    out["has_pool"] = (out["pool_area"].fillna(0) > 0).astype(int)
    out["has_garage"] = (
        (out["garage_area"].fillna(0) > 0) | (out["garage_cars"].fillna(0) > 0)
    ).astype(int)
    out["has_fireplace"] = (out["fireplaces"].fillna(0) > 0).astype(int)
    out["has_second_floor"] = (out["second_flr_sf"].fillna(0) > 0).astype(int)
    out["liv_area_ratio"] = out["gr_liv_area"] / out["lot_area"].replace(0, np.nan)
    out["rooms_per_1000sf"] = out["tot_rms_abv_grd"] / (out["total_sf"] / 1000).replace(
        0, np.nan
    )
    out["avg_room_size"] = out["gr_liv_area"] / out["tot_rms_abv_grd"].replace(0, np.nan)
    out["garage_area_ratio"] = out["garage_area"].fillna(0) / out["lot_area"].replace(
        0, np.nan
    )
    out["quality_condition_index"] = out["overall_qual"] * out["overall_cond"]
    out["renovation_scope_code"] = (
        out["renovation_scope"].map(RENOVATION_SCOPE_CODES).fillna(0).astype(int)
    )

    # Ratios can be undefined for pathological lots; use the training median.
    for column in ("liv_area_ratio", "rooms_per_1000sf", "avg_room_size", "garage_area_ratio"):
        median = out[column].median()
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).fillna(median)

    return out


def model_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return the model input frame and the (optional) target vector."""
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    X = frame.loc[:, list(MODEL_FEATURES)].copy()
    # scikit-learn's missing-value handling cannot compare pandas nullable
    # strings (NA) inside object arrays, so categorical columns are cast to
    # object with numpy NaN before they reach any transformer.
    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            values = X[column].astype("object")
            X[column] = values.where(values.notna(), np.nan)
    y = frame[TARGET].copy() if TARGET in frame.columns else None
    return X, y


def build_modeling_table(save: bool = True) -> pd.DataFrame:
    """Engineer features on the cleaned data and persist the analytical table."""
    cleaned = load_cleaned()
    engineered = engineer_features(cleaned)
    engineered["price_segment"] = pd.cut(
        engineered[TARGET],
        bins=[0, 130_000, 200_000, 300_000, np.inf],
        labels=["entry", "mid", "upper", "luxury"],
    ).astype("string")
    engineered["neighbourhood_segment"] = (
        engineered["neighborhood"].map(NEIGHBOURHOOD_SEGMENT).fillna("established")
    )
    engineered["sale_year"] = engineered["yr_sold"].astype(int)
    engineered["sale_month"] = engineered["mo_sold"].astype(int)
    engineered["sale_condition"] = engineered["sale_condition"].astype("string")
    engineered["sale_type"] = engineered["sale_type"].astype("string")

    if save:
        engineered.to_parquet(DATA_PROCESSED / "modeling_table.parquet", index=False)
    return engineered


def load_modeling_table() -> pd.DataFrame:
    path = DATA_PROCESSED / "modeling_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Modeling table not found. Run `uv run python scripts/03_build_features.py` first."
        )
    return pd.read_parquet(path)


def training_frame(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = load_modeling_table() if frame is None else frame
    train = frame.loc[frame[TARGET].notna()].copy()
    if len(train) < 100:
        raise ValueError("Training frame is unexpectedly small.")
    return train


def _vif_values(data: pd.DataFrame) -> pd.Series:
    values = data.to_numpy(dtype=float)
    with_constant = np.column_stack([np.ones(len(values)), values])
    scores = [
        variance_inflation_factor(with_constant, i + 1) for i in range(values.shape[1])
    ]
    return pd.Series(scores, index=data.columns)


def vif_table(
    frame: pd.DataFrame, columns: list[str] | None = None, max_iterations: int = 40
) -> pd.DataFrame:
    """Variance inflation factors with iterative removal of VIF > 10.

    The table reports the initial VIF, the VIF in the final retained set and
    whether the feature was retained, so the report can document exactly which
    variables were dropped to obtain an identifiable linear specification.
    """
    columns = list(columns or VIF_FEATURES)
    data = frame.loc[:, [c for c in columns if c in frame.columns]].astype(float)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median())
    data = data.loc[:, data.std() > 1e-9]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initial = _vif_values(data)
        remaining = list(data.columns)
        dropped: dict[str, float] = {}
        for _ in range(max_iterations):
            current = _vif_values(data[remaining])
            worst = current.idxmax()
            if current[worst] <= 10.0 or len(remaining) <= 1:
                break
            dropped[worst] = float(current[worst])
            remaining.remove(worst)
        final = (
            _vif_values(data[remaining])
            if len(remaining) > 1
            else pd.Series(dtype=float)
        )

    records = []
    for column in data.columns:
        records.append(
            {
                "feature": column,
                "vif_initial": float(initial[column]),
                "vif_final": float(final.get(column, np.nan)) if column in remaining else np.nan,
                "retained": column in remaining,
                "drop_reason": (
                    None
                    if column in remaining
                    else f"Removed at VIF={dropped.get(column, float('nan')):.1f}"
                ),
            }
        )
    return pd.DataFrame(records).sort_values("vif_initial", ascending=False)


def feature_group_table() -> pd.DataFrame:
    """Mapping from feature to business group, used in the report appendix."""
    groups = []
    for column in MODEL_FEATURES:
        if column in CATEGORICAL_FEATURES:
            group = "Categorical"
        elif column.startswith(("dist_", "school_", "walk_", "crime_", "amenity_", "park_")):
            group = "Neighbourhood amenities"
        elif column.startswith(("mortgage_", "hpi_", "months_", "local_")):
            group = "Macro-financial"
        elif column.startswith(("total_", "has_", "gr_liv", "first_", "second_", "low_qual")):
            group = "Structural composites"
        elif column in ("age_at_sale", "remod_age", "is_renovated", "renovation_scope_code"):
            group = "Age and renovation"
        elif column in ("neigh_lat", "neigh_lon"):
            group = "Location"
        else:
            group = "Structural"
        groups.append({"feature": column, "group": group})
    return pd.DataFrame(groups)


#: Documentation of every engineered feature. The data dictionary exporter and
#: the report appendix read this table, so the documentation cannot drift away
#: from the implementation.
FEATURE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "feature": "total_sf",
        "formula": "total_bsmt_sf + first_flr_sf + second_flr_sf",
        "unit": "sqft",
        "rationale": "Single measure of usable floor area across basement and living levels.",
    },
    {
        "feature": "total_bath",
        "formula": "full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath",
        "unit": "bathroom equivalents",
        "rationale": "Weights half bathrooms at one half, the conventional listing convention.",
    },
    {
        "feature": "total_porch_sf",
        "formula": "sum of deck, open, enclosed, three-season and screened porch areas",
        "unit": "sqft",
        "rationale": "Aggregates outdoor amenity space into one interpretable quantity.",
    },
    {
        "feature": "age_at_sale",
        "formula": "yr_sold - year_built",
        "unit": "years",
        "rationale": "Depreciation and vintage effect; replaces the raw construction year in modelling.",
    },
    {
        "feature": "remod_age",
        "formula": "yr_sold - year_remod_add",
        "unit": "years",
        "rationale": "Time since the last permitted remodelling.",
    },
    {
        "feature": "is_renovated",
        "formula": "1 if year_remod_add > year_built else 0",
        "unit": "flag",
        "rationale": "Identifies properties with a recorded renovation.",
    },
    {
        "feature": "renovation_scope_code",
        "formula": "ordinal mapping of renovation_scope: None=0, Minor=1, Moderate=2, Major=3",
        "unit": "ordinal",
        "rationale": "Captures renovation depth for models that cannot use the categorical label directly.",
    },
    {
        "feature": "has_pool",
        "formula": "1 if pool_area > 0 else 0",
        "unit": "flag",
        "rationale": "Rare amenity with a non-linear price effect.",
    },
    {
        "feature": "has_garage",
        "formula": "1 if garage_area > 0 or garage_cars > 0 else 0",
        "unit": "flag",
        "rationale": "Separates absence of a garage from a small garage.",
    },
    {
        "feature": "has_fireplace",
        "formula": "1 if fireplaces > 0 else 0",
        "unit": "flag",
        "rationale": "Separates absence of a fireplace from one fireplace.",
    },
    {
        "feature": "has_second_floor",
        "formula": "1 if second_flr_sf > 0 else 0",
        "unit": "flag",
        "rationale": "Distinguishes single-storey from two-storey stock.",
    },
    {
        "feature": "liv_area_ratio",
        "formula": "gr_liv_area / lot_area",
        "unit": "ratio",
        "rationale": "Development intensity of the parcel.",
    },
    {
        "feature": "rooms_per_1000sf",
        "formula": "tot_rms_abv_grd / (total_sf / 1000)",
        "unit": "rooms per 1,000 sqft",
        "rationale": "Room density, a proxy for compartmentalisation.",
    },
    {
        "feature": "avg_room_size",
        "formula": "gr_liv_area / tot_rms_abv_grd",
        "unit": "sqft per room",
        "rationale": "Spaciousness of the layout.",
    },
    {
        "feature": "garage_area_ratio",
        "formula": "garage_area / lot_area",
        "unit": "ratio",
        "rationale": "Share of the parcel devoted to the garage.",
    },
    {
        "feature": "quality_condition_index",
        "formula": "overall_qual * overall_cond",
        "unit": "index 1-100",
        "rationale": "Joint material quality and maintenance condition.",
    },
    {
        "feature": "price_per_sqm",
        "formula": "SalePrice / (gr_liv_area * 0.092903)",
        "unit": "USD per sqm",
        "rationale": "Analysis and reporting only; excluded from model inputs because it uses the target.",
    },
    {
        "feature": "log_price",
        "formula": "ln(1 + SalePrice)",
        "unit": "log USD",
        "rationale": "Modelling target; reduces skewness from 1.88 to 0.12.",
    },
)
