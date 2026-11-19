"""Central configuration for the residential valuation project.

All paths are derived from the repository root so that no machine-specific
absolute path is required by any script, notebook, or documentation artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]
KAGGLE_DIR: Path = ROOT / "house-prices-advanced-regression-techniques"
TRAIN_CSV: Path = KAGGLE_DIR / "train.csv"
TEST_CSV: Path = KAGGLE_DIR / "test.csv"
DATA_DESCRIPTION: Path = KAGGLE_DIR / "data_description.txt"

DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
DATA_SYNTHETIC: Path = ROOT / "data" / "synthetic"

ARTIFACTS: Path = ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS / "models"
METRICS_DIR: Path = ARTIFACTS / "metrics"
FIGURES_DIR: Path = ARTIFACTS / "figures"
MONITORING_DIR: Path = ARTIFACTS / "monitoring"
REPORTS_DIR: Path = ARTIFACTS / "reports"

DOCS_DIR: Path = ROOT / "docs"
REPORT_DIR: Path = ROOT / "report"
SLIDES_DIR: Path = ROOT / "slides"

for _p in (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_SYNTHETIC,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    MONITORING_DIR,
    REPORTS_DIR,
    DOCS_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_FOLDS: int = 5
TARGET: str = "SalePrice"
ID_COLUMN: str = "Id"
CLEAN_ID: str = "id"

#: Fractions used by the evaluation harness.
CALIBRATION_FRACTION: float = 0.20
TEST_SIZE: float = 0.20

#: Default transaction period used when a caller omits the sale date; matches the
#: most recent year observed in the training data.
DEFAULT_SALE_YEAR: int = 2010
DEFAULT_SALE_MONTH: int = 6

#: Nominal coverage levels for prediction intervals.
COVERAGE_LEVELS: tuple[float, ...] = (0.80, 0.90)

# ---------------------------------------------------------------------------
# Business assumptions (Module 1)
# ---------------------------------------------------------------------------
#: Absolute error tolerance at which a valuation is considered operationally
#: acceptable (approximately the width of a typical negotiation margin).
ERROR_TOLERANCE_USD: float = 15_000.0

#: Relative error tolerance used for the MAPE-based KPI.
ERROR_TOLERANCE_PCT: float = 8.0

#: Loss asymmetry: an over-valuation is assumed to cost the platform more
#: (stale listing, failed transaction) than an under-valuation.
COST_OVER_VALUATION: float = 1.5
COST_UNDER_VALUATION: float = 1.0

#: Reference sale value used to express the business cost of error in USD.
REFERENCE_SALE_PRICE: float = 180_000.0

#: Retraining triggers (Module 7).
DRIFT_PSI_THRESHOLD: float = 0.20
DRIFT_MIN_FEATURES: int = 3
DRIFT_KS_PVALUE: float = 0.01
PERFORMANCE_DEGRADATION_THRESHOLD: float = 0.20
PERFORMANCE_CONSECUTIVE_WINDOWS: int = 2
COVERAGE_DROP_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Modeling configuration
# ---------------------------------------------------------------------------
#: Features that are genuinely observable when a property is listed. They are
#: the only inputs accepted by the deployed model; post-sale variables such as
#: days-on-market or sale condition are deliberately excluded to avoid
#: temporal leakage and to keep the inference contract realistic.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "SalePrice",
    "Id",
    "days_on_market",
    "listing_date",
    "sale_year",
    "sale_month",
    "sale_condition",
    "sale_type",
    "mo_sold",
    "yr_sold",
    "log_price",
    "price_per_sqm",
    "source_file",
)

#: Columns used only for temporal / business analysis, never as model inputs.
ANALYSIS_ONLY_COLUMNS: tuple[str, ...] = (
    "days_on_market",
    "sale_condition",
    "sale_type",
    "listing_date",
)

#: Neutral neighbourhood ordering used to seed the synthetic generator. The
#: tiers encode a documented expert prior about Ames sub-markets; they are not
#: estimated from the target variable.
NEIGHBOURHOOD_TIERS: dict[str, int] = {
    "NoRidge": 5,
    "NridgHt": 5,
    "StoneBr": 5,
    "Timber": 4,
    "Veenker": 4,
    "ClearCr": 4,
    "Somerst": 4,
    "Blmngtn": 4,
    "CollgCr": 4,
    "Gilbert": 4,
    "Crawfor": 4,
    "NWAmes": 3,
    "SawyerW": 3,
    "Mitchel": 3,
    "BrkSide": 3,
    "NAmes": 3,
    "Blueste": 3,
    "OldTown": 2,
    "Edwards": 2,
    "Sawyer": 2,
    "SWISU": 2,
    "IDOTRR": 2,
    "MeadowV": 1,
    "BrDale": 1,
    "NPkVill": 1,
}

#: Approximate centroid of each Ames neighbourhood used to synthesise a
#: plausible spatial layout (decimal degrees, WGS-84).
NEIGHBOURHOOD_CENTROIDS: dict[str, tuple[float, float]] = {
    "NoRidge": (42.0510, -93.6470),
    "NridgHt": (42.0460, -93.6560),
    "StoneBr": (42.0380, -93.6350),
    "Timber": (42.0290, -93.6570),
    "Veenker": (42.0420, -93.6280),
    "ClearCr": (42.0260, -93.6430),
    "Somerst": (42.0580, -93.6350),
    "Blmngtn": (42.0320, -93.6210),
    "CollgCr": (42.0210, -93.6290),
    "Gilbert": (42.0170, -93.6180),
    "Crawfor": (42.0350, -93.6640),
    "NWAmes": (42.0270, -93.6680),
    "SawyerW": (42.0140, -93.6420),
    "Mitchel": (42.0090, -93.6530),
    "BrkSide": (42.0450, -93.6180),
    "NAmes": (42.0180, -93.6510),
    "Blueste": (42.0600, -93.6460),
    "OldTown": (42.0390, -93.6120),
    "Edwards": (42.0110, -93.6640),
    "Sawyer": (42.0060, -93.6410),
    "SWISU": (42.0480, -93.6300),
    "IDOTRR": (42.0430, -93.6050),
    "MeadowV": (41.9980, -93.6320),
    "BrDale": (42.0530, -93.6260),
    "NPkVill": (42.0230, -93.6600),
}

#: Housing sub-market label used for reporting and stratified diagnostics.
NEIGHBOURHOOD_SEGMENT: dict[str, str] = {
    "NoRidge": "premium",
    "NridgHt": "premium",
    "StoneBr": "premium",
    "Timber": "established",
    "Veenker": "established",
    "ClearCr": "established",
    "Somerst": "established",
    "Blmngtn": "established",
    "CollgCr": "established",
    "Gilbert": "established",
    "Crawfor": "established",
    "NWAmes": "established",
    "SawyerW": "established",
    "Mitchel": "established",
    "BrkSide": "established",
    "NAmes": "established",
    "Blueste": "established",
    "OldTown": "value",
    "Edwards": "value",
    "Sawyer": "value",
    "SWISU": "value",
    "IDOTRR": "value",
    "MeadowV": "value",
    "BrDale": "value",
    "NPkVill": "value",
}

#: Macro anchors for the synthetic monthly market series (yyyy-mm, value).
MORTGAGE_RATE_ANCHORS: tuple[tuple[str, float], ...] = (
    ("2006-01", 6.40),
    ("2006-07", 6.80),
    ("2007-06", 6.65),
    ("2008-06", 6.30),
    ("2008-12", 5.20),
    ("2009-06", 5.00),
    ("2009-12", 4.80),
    ("2010-06", 4.70),
    ("2010-12", 4.90),
    ("2011-12", 4.60),
)

HPI_ANCHORS: tuple[tuple[str, float], ...] = (
    ("2006-01", 100.0),
    ("2006-12", 104.5),
    ("2007-06", 108.0),
    ("2008-06", 106.0),
    ("2009-06", 96.0),
    ("2010-06", 95.0),
    ("2010-12", 95.5),
    ("2011-12", 97.0),
)

MONTHS_SUPPLY_ANCHORS: tuple[tuple[str, float], ...] = (
    ("2006-01", 4.4),
    ("2007-06", 5.2),
    ("2008-12", 8.6),
    ("2009-12", 7.2),
    ("2010-12", 6.4),
    ("2011-12", 5.9),
)

UNEMPLOYMENT_ANCHORS: tuple[tuple[str, float], ...] = (
    ("2006-01", 3.9),
    ("2007-12", 4.3),
    ("2009-06", 6.0),
    ("2010-12", 5.4),
    ("2011-12", 5.0),
)


@dataclass(frozen=True)
class DataContract:
    """Column contract shared by the pipeline and the serving layer."""

    numeric: tuple[str, ...] = field(
        default=(
            "lot_frontage",
            "lot_area",
            "overall_qual",
            "overall_cond",
            "year_built",
            "year_remod_add",
            "mas_vnr_area",
            "total_bsmt_sf",
            "first_flr_sf",
            "second_flr_sf",
            "low_qual_fin_sf",
            "gr_liv_area",
            "bsmt_full_bath",
            "bsmt_half_bath",
            "full_bath",
            "half_bath",
            "bedroom_abv_gr",
            "kitchen_abv_gr",
            "tot_rms_abv_grd",
            "fireplaces",
            "garage_cars",
            "garage_area",
            "wood_deck_sf",
            "open_porch_sf",
            "enclosed_porch",
            "three_ssn_porch",
            "screen_porch",
            "pool_area",
            "misc_val",
            "total_sf",
            "total_bath",
            "total_porch_sf",
            "age_at_sale",
            "remod_age",
            "is_renovated",
            "has_pool",
            "has_garage",
            "has_fireplace",
            "has_second_floor",
            "liv_area_ratio",
            "rooms_per_1000sf",
            "avg_room_size",
            "garage_area_ratio",
            "quality_condition_index",
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
            "renovation_flag",
            "renovation_scope_code",
            "mortgage_rate_pct",
            "hpi_index",
            "hpi_yoy_pct",
            "months_supply",
            "local_unemployment_pct",
            "neigh_lat",
            "neigh_lon",
        )
    )


CONTRACT = DataContract()
