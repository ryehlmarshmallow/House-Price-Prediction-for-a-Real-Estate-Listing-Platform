"""Data cleaning and quality control (Module 3).

The Ames files encode missing information in two different ways: structural
absence (a house without a basement has no basement quality) and genuine
absence (lot frontage unknown). Treating both as null destroys signal, so the
pipeline distinguishes them explicitly, records every decision with before and
after counts, and produces a machine-readable cleaning log that feeds the
report and the monitoring baseline.

All cleaning parameters estimated from data (medians, modes, category
vocabularies) are stored in a reusable ``CleaningState`` so that the same
transformation is applied at serving time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from revaluation.config import (
    CLEAN_ID,
    DATA_PROCESSED,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    METRICS_DIR,
    TARGET,
)
from revaluation.data.load import load_combined
from revaluation.data.synthetic import load_context

# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------
_SPECIAL_NAMES = {
    "SalePrice": "SalePrice",
    "1stFlrSF": "first_flr_sf",
    "2ndFlrSF": "second_flr_sf",
    "3SsnPorch": "three_ssn_porch",
    "MSSubClass": "ms_sub_class",
    "MSZoning": "ms_zoning",
}

CASE_INSENSITIVE_CATEGORY_ALIASES = {
    "c(all)": "C (all)",
    "c (all)": "C (all)",
    "2story": "2Story",
    "2 story": "2Story",
    "1.5story": "1.5Story",
    "1.5 story": "1.5Story",
    "1story": "1Story",
    "1 story": "1Story",
    "sfoyer": "SFoyer",
    "slvl": "SLvl",
}

#: Columns where a missing value means the property does not have the feature.
ABSENCE_TO_NONE = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
)

#: Columns where a missing value plausibly means a zero quantity.
ABSENCE_TO_ZERO = (
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
)

#: Columns where missingness is unstructured and is imputed with the mode.
MODE_IMPUTED = (
    "Electrical",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
    "Functional",
    "Utilities",
    "MSZoning",
)

#: Materials that should always be present for a habitable dwelling.
MANDATORY = (
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "PavedDrive",
)

OUTLIER_RULES: dict[str, str] = {
    "implausible_area_price": "GrLivArea > 4000 and SalePrice < 300000",
    "extreme_lot": "LotArea > 100000",
    "implausible_price_per_sqft": "SalePrice / GrLivArea outside [25, 500]",
    "invalid_garage_year": "GarageYrBlt > YrSold + 1",
    "invalid_remodel_year": "YearRemodAdd < YearBuilt",
}


def snake_case(name: str) -> str:
    if name in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[name]
    name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    name = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", name)
    return name.lower()


def rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {column: snake_case(column) for column in frame.columns}
    return frame.rename(columns=mapping)


def category_key(value: str) -> str:
    """Case- and punctuation-insensitive key used to canonicalise labels."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


@dataclass
class CleaningState:
    """Parameters estimated from training data and reused at serving time."""

    lot_frontage_median_by_neighbourhood: dict[str, float] = field(default_factory=dict)
    lot_frontage_global_median: float = 70.0
    mode_by_column: dict[str, str] = field(default_factory=dict)
    garage_year_fallback_ratio: float = 0.0
    category_vocabularies: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lot_frontage_median_by_neighbourhood": {
                str(k): float(v) for k, v in self.lot_frontage_median_by_neighbourhood.items()
            },
            "lot_frontage_global_median": float(self.lot_frontage_global_median),
            "mode_by_column": dict(self.mode_by_column),
            "garage_year_fallback_ratio": float(self.garage_year_fallback_ratio),
            "category_vocabularies": {
                column: dict(levels)
                for column, levels in self.category_vocabularies.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CleaningState:
        return cls(
            lot_frontage_median_by_neighbourhood={
                str(k): float(v)
                for k, v in payload.get(
                    "lot_frontage_median_by_neighbourhood", {}
                ).items()
            },
            lot_frontage_global_median=float(
                payload.get("lot_frontage_global_median", 70.0)
            ),
            mode_by_column={
                str(k): str(v) for k, v in payload.get("mode_by_column", {}).items()
            },
            garage_year_fallback_ratio=float(
                payload.get("garage_year_fallback_ratio", 0.0)
            ),
            category_vocabularies={
                str(column): {str(k): str(v) for k, v in levels.items()}
                for column, levels in payload.get("category_vocabularies", {}).items()
            },
        )


@dataclass
class CleaningReport:
    initial_rows: int = 0
    final_rows: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    missing_after: dict[str, int] = field(default_factory=dict)
    duplicate_rows_removed: int = 0
    outlier_flags: dict[str, int] = field(default_factory=dict)
    outliers_removed: int = 0
    label_variants_fixed: int = 0
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "outlier_flags": self.outlier_flags,
            "outliers_removed": self.outliers_removed,
            "label_variants_fixed": self.label_variants_fixed,
            "decisions": self.decisions,
        }


def normalise_labels(
    frame: pd.DataFrame,
    vocabularies: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Trim, collapse whitespace and canonicalise spelling and case variants.

    When a column vocabulary estimated from the training data is supplied, any
    value whose case- and punctuation-insensitive key matches a known level is
    replaced by the canonical spelling. Unknown levels are preserved, which is
    the correct behaviour for a new zoning code or a new neighbourhood.
    """
    frame = frame.copy()
    vocabularies = vocabularies or {}
    fixed = 0
    for column in frame.select_dtypes(include=["object", "str"]).columns:
        original = frame[column]
        cleaned = (
            original.astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA})
        lowercase = cleaned.str.lower()
        mapped = lowercase.map(CASE_INSENSITIVE_CATEGORY_ALIASES)
        cleaned = cleaned.where(mapped.isna(), mapped)
        vocabulary = vocabularies.get(column, {})
        if vocabulary:
            keys = cleaned.map(lambda value: category_key(value) if pd.notna(value) else None)
            canonical = keys.map(vocabulary)
            cleaned = cleaned.where(canonical.isna(), canonical)
        fixed += int((cleaned.fillna("__na__") != original.fillna("__na__")).sum())
        frame[column] = cleaned
    return frame, fixed


def map_ms_subclass(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the numeric building-type code with a readable category."""
    mapping = {
        20: "1Fam_1Story_1946+",
        30: "1Fam_1Story_1945-",
        40: "1Fam_1Story_Attic",
        45: "1Fam_1.5Story_Unfinished",
        50: "1Fam_1.5Story_Finished",
        60: "1Fam_2Story_1946+",
        70: "1Fam_2Story_1945-",
        75: "1Fam_2.5Story",
        80: "1Fam_MultiLevel",
        85: "1Fam_SplitLevel",
        90: "Duplex_AllStyles",
        120: "Twnhs_1Story_1946+",
        150: "Twnhs_1.5Story",
        160: "Twnhs_2Story_1946+",
        180: "Twnhs_1Story_PUD",
        190: "2Fam_Conversion",
    }
    frame = frame.copy()
    frame["ms_sub_class"] = frame["ms_sub_class"].map(mapping).fillna("Other")
    return frame


def fit_cleaning_state(frame: pd.DataFrame) -> CleaningState:
    """Estimate imputation parameters from the labelled training frame."""
    state = CleaningState()
    lot_frontage = frame["lot_frontage"]
    neighbourhood = frame["neighborhood"].astype(str)
    medians = lot_frontage.groupby(neighbourhood).median()
    state.lot_frontage_median_by_neighbourhood = {
        str(k): float(v) for k, v in medians.items() if pd.notna(v)
    }
    state.lot_frontage_global_median = float(lot_frontage.median())
    for column in MODE_IMPUTED:
        target_column = snake_case(column)
        if target_column in frame.columns:
            mode = frame[target_column].mode(dropna=True)
            if not mode.empty:
                state.mode_by_column[target_column] = str(mode.iloc[0])
    vocabularies: dict[str, dict[str, str]] = {}
    for column in frame.select_dtypes(include=["object", "str"]).columns:
        levels = frame[column].dropna().astype(str).unique()
        mapping: dict[str, str] = {}
        for level in levels:
            mapping.setdefault(category_key(level), level)
        vocabularies[column] = mapping
    state.category_vocabularies = vocabularies

    garage = frame["garage_yr_blt"]
    built = frame["year_built"]
    mask = garage.notna() & built.notna()
    state.garage_year_fallback_ratio = float((garage[mask] - built[mask]).median())
    return state


def apply_cleaning(
    frame: pd.DataFrame,
    state: CleaningState,
    *,
    is_training: bool,
    report: CleaningReport | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented cleaning rules and record the impact."""
    report = report or CleaningReport()
    report.initial_rows = int(report.initial_rows or len(frame))
    out = frame.copy()

    # 1. Column names and categorical vocabulary ---------------------------
    out, fixed = normalise_labels(out, state.category_vocabularies)
    report.label_variants_fixed += fixed
    out = map_ms_subclass(out)

    # 2. Structural absence ------------------------------------------------
    for column in ABSENCE_TO_NONE:
        target_column = snake_case(column)
        if target_column in out.columns:
            out[target_column] = out[target_column].astype("object").fillna("None")
    for column in ABSENCE_TO_ZERO:
        target_column = snake_case(column)
        if target_column in out.columns:
            out[target_column] = out[target_column].fillna(0.0)

    # 3. Unstructured missingness (test rows carry a handful of gaps) ------
    for column, mode in state.mode_by_column.items():
        if column in out.columns:
            out[column] = out[column].fillna(mode)
    for column in MANDATORY:
        target_column = snake_case(column)
        if target_column in out.columns and out[target_column].isna().any():
            fallback = out[target_column].mode(dropna=True)
            if not fallback.empty:
                out[target_column] = out[target_column].fillna(fallback.iloc[0])

    # 4. Lot frontage: neighbourhood median (fallback: global median) ------
    neighbourhood = out["neighborhood"].astype(str)
    mapped_median = neighbourhood.map(state.lot_frontage_median_by_neighbourhood)
    out["lot_frontage"] = out["lot_frontage"].fillna(mapped_median)
    out["lot_frontage"] = out["lot_frontage"].fillna(state.lot_frontage_global_median)

    # 5. Garage year: when the garage existed before construction, use the
    #    construction year times the median lag observed in training data.
    garage_mask = out["garage_yr_blt"].isna()
    if garage_mask.any():
        out.loc[garage_mask, "garage_yr_blt"] = (
            out.loc[garage_mask, "year_built"] + state.garage_year_fallback_ratio
        )
    # A garage year in the future of the sale is a data-entry error.
    future = out["garage_yr_blt"] > (out["yr_sold"] + 1)
    out.loc[future, "garage_yr_blt"] = out.loc[future, "year_built"]
    # No garage implies no garage year.
    no_garage = (out["garage_area"].fillna(0) == 0) & (out["garage_cars"].fillna(0) == 0)
    out.loc[no_garage, "garage_yr_blt"] = out.loc[no_garage, "year_built"]

    # 6. Derived sanity columns -------------------------------------------
    out["log_price"] = np.log(out[TARGET]) if TARGET in out.columns else np.nan
    out["price_per_sqm"] = (
        out[TARGET] / (out["gr_liv_area"] * 0.092903) if TARGET in out.columns else np.nan
    )

    # 7. Outlier detection -------------------------------------------------
    flags = outlier_flags(out)
    report.outlier_flags = {name: int(mask.sum()) for name, mask in flags.items()}
    any_flag = np.logical_or.reduce([mask.to_numpy() for mask in flags.values()])
    if is_training:
        # Only physically implausible records are dropped: two very large
        # houses sold far below their structural value (partial sales).
        drop_mask = flags["implausible_area_price"]
        report.outliers_removed = int(drop_mask.sum())
        report.decisions.append(
            f"Removed {int(drop_mask.sum())} implausible area-price records from the "
            "training data (GrLivArea > 4000 sqft sold below 300k)."
        )
        out = out.loc[~drop_mask].copy()
    else:
        report.decisions.append(
            f"Flagged {int(any_flag.sum())} test records for review without removal."
        )

    # 8. Duplicate detection ----------------------------------------------
    feature_columns = [
        c for c in out.columns if c not in {CLEAN_ID, ID_COLUMN, *LEAKAGE_COLUMNS}
    ]
    duplicates = out.duplicated(subset=feature_columns, keep="first")
    if is_training and duplicates.any():
        report.duplicate_rows_removed = int(duplicates.sum())
        report.decisions.append(
            f"Removed {int(duplicates.sum())} exact duplicate listings on all structural features."
        )
        out = out.loc[~duplicates].copy()

    report.final_rows = len(out)
    report.missing_before = {
        str(k): int(v) for k, v in frame.isna().sum().items() if int(v) > 0
    }
    report.missing_after = {
        str(k): int(v) for k, v in out.isna().sum().items() if int(v) > 0
    }
    if not report.decisions:
        report.decisions.append("Standard cleaning pipeline applied.")
    return out, report


def outlier_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the documented implausibility rules.

    Rules whose source columns are unavailable are reported as all-false, which
    keeps partial serving payloads and unit tests usable.
    """
    empty = pd.Series(False, index=frame.index)
    flags: dict[str, pd.Series] = {}

    if TARGET in frame.columns and "gr_liv_area" in frame.columns:
        flags["implausible_area_price"] = (frame["gr_liv_area"] > 4000) & (
            frame[TARGET] < 300_000
        )
        ppsf = frame[TARGET] / frame["gr_liv_area"].replace(0, np.nan)
        flags["implausible_price_per_sqft"] = (ppsf < 25) | (ppsf > 500)
    else:
        flags["implausible_area_price"] = empty.copy()
        flags["implausible_price_per_sqft"] = empty.copy()

    flags["extreme_lot"] = (
        frame["lot_area"] > 100_000 if "lot_area" in frame.columns else empty.copy()
    )
    flags["invalid_garage_year"] = (
        frame["garage_yr_blt"] > (frame["yr_sold"] + 1)
        if {"garage_yr_blt", "yr_sold"}.issubset(frame.columns)
        else empty.copy()
    )
    flags["invalid_remodel_year"] = (
        frame["year_remod_add"] < frame["year_built"]
        if {"year_remod_add", "year_built"}.issubset(frame.columns)
        else empty.copy()
    )
    return flags


def build_clean_dataset(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """End-to-end cleaning: load, merge synthetic context, clean, persist."""
    combined = load_combined()
    context = load_context()
    merged = combined.merge(context, on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["amenity_score"].isna().any():
        raise ValueError("Synthetic context is incomplete; regenerate it before cleaning.")

    merged = rename_columns(merged)
    train_mask = merged["source_file"] == "train"

    report = CleaningReport()
    report.initial_rows = int(train_mask.sum())
    state = fit_cleaning_state(merged.loc[train_mask])
    train_clean, train_report = apply_cleaning(
        merged.loc[train_mask], state, is_training=True, report=report
    )
    test_clean, _ = apply_cleaning(
        merged.loc[~train_mask], state, is_training=False, report=CleaningReport()
    )

    cleaned = pd.concat([train_clean, test_clean], ignore_index=True)
    # Preserve the chronological order for temporal diagnostics.
    cleaned["sale_date"] = pd.to_datetime(
        {"year": cleaned["yr_sold"], "month": cleaned["mo_sold"], "day": 15}
    )
    cleaned = cleaned.sort_values(["sale_date", CLEAN_ID]).reset_index(drop=True)

    payload = {
        "report": train_report.to_dict(),
        "state": state.to_dict(),
        "rows": {
            "train_raw": int(train_mask.sum()),
            "train_clean": len(train_clean),
            "test_clean": len(test_clean),
        },
        "outlier_rule_text": OUTLIER_RULES,
        "absence_to_none": [snake_case(c) for c in ABSENCE_TO_NONE],
        "absence_to_zero": [snake_case(c) for c in ABSENCE_TO_ZERO],
        "mode_imputed": list(MODE_IMPUTED),
    }

    if save:
        cleaned.to_parquet(DATA_PROCESSED / "listings_clean.parquet", index=False)
        with open(METRICS_DIR / "cleaning_report.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with open(DATA_PROCESSED / "cleaning_state.json", "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)
    return cleaned, payload


def load_cleaned() -> pd.DataFrame:
    path = DATA_PROCESSED / "listings_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run `uv run python scripts/02_clean_data.py` first."
        )
    return pd.read_parquet(path)


def load_cleaning_state() -> CleaningState:
    path = DATA_PROCESSED / "cleaning_state.json"
    if not path.exists():
        raise FileNotFoundError("Cleaning state not found; run the cleaning script first.")
    with open(path, encoding="utf-8") as handle:
        return CleaningState.from_dict(json.load(handle))
