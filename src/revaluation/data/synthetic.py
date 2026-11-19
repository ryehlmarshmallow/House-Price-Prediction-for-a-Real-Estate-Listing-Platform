"""Synthetic contextual data generation (Module 1).

The Kaggle competition data provides structural attributes and the transaction
price, but it contains no information about neighbourhood amenities, renovation
history, marketing duration, or macro-financial conditions. This module
simulates those fields with an explicit, documented business logic:

* Location quality is driven by a tier assigned to each Ames neighbourhood.
  The tier is an expert prior about sub-market desirability and is *not*
  estimated from sale prices, which keeps the synthetic generator free of
  target leakage.
* Amenity distances, school ratings, walkability and crime are drawn from
  tier-dependent distributions with multiplicative noise.
* Renovation history is derived from the recorded construction and remodelling
  years, with a scope category that combines the remodelling lag and the
  observed overall quality.
* Marketing duration follows a negative-binomial-like process that responds to
  location, property quality, seasonality and the prevailing interest rate.
* Macro conditions follow monthly paths calibrated to the 2006-2011 U.S.
  housing cycle, with an AR(1) noise component.

Every generated column is listed in ``SYNTHETIC_SPECS`` and exported to the
data dictionary so that documentation and implementation cannot diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revaluation.config import (
    DATA_SYNTHETIC,
    HPI_ANCHORS,
    ID_COLUMN,
    MONTHS_SUPPLY_ANCHORS,
    MORTGAGE_RATE_ANCHORS,
    NEIGHBOURHOOD_CENTROIDS,
    NEIGHBOURHOOD_TIERS,
    RANDOM_STATE,
    UNEMPLOYMENT_ANCHORS,
)

MACRO_START = "2006-01"
MACRO_END = "2012-12"


# ---------------------------------------------------------------------------
# Documentation specification
# ---------------------------------------------------------------------------
SYNTHETIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "column": "neigh_lat",
        "dtype": "float",
        "unit": "decimal degrees",
        "range": "41.99 - 42.07",
        "logic": (
            "Neighbourhood centroid latitude (documented table) plus a jitter drawn from "
            "N(0, 0.0015) to represent intra-neighbourhood position."
        ),
        "assumption": "Property coordinates are available from the listing geocoder.",
    },
    {
        "column": "neigh_lon",
        "dtype": "float",
        "unit": "decimal degrees",
        "range": "-93.67 - -93.60",
        "logic": "Neighbourhood centroid longitude plus N(0, 0.0015) jitter.",
        "assumption": "Same as above.",
    },
    {
        "column": "dist_school_km",
        "dtype": "float",
        "unit": "km",
        "range": "0.10 - 8.00",
        "logic": (
            "Log-normal draw with median exp(1.90 - 0.22 * tier) and sigma 0.35, clipped. "
            "School accessibility improves with the neighbourhood tier."
        ),
        "assumption": "Distance to the nearest public elementary school, straight line.",
    },
    {
        "column": "dist_hospital_km",
        "dtype": "float",
        "unit": "km",
        "range": "0.30 - 20.00",
        "logic": "Log-normal median exp(5.00 - 0.55 * tier), sigma 0.30, clipped.",
        "assumption": "Distance to the nearest general hospital.",
    },
    {
        "column": "dist_transit_km",
        "dtype": "float",
        "unit": "km",
        "range": "0.20 - 15.00",
        "logic": "Log-normal median exp(3.20 - 0.42 * tier), sigma 0.33, clipped.",
        "assumption": "Distance to the nearest bus stop with a 15-minute service frequency.",
    },
    {
        "column": "dist_park_km",
        "dtype": "float",
        "unit": "km",
        "range": "0.10 - 10.00",
        "logic": "Log-normal median exp(1.60 - 0.15 * tier), sigma 0.30, clipped.",
        "assumption": "Distance to the nearest public park of at least 1 hectare.",
    },
    {
        "column": "school_rating",
        "dtype": "integer",
        "unit": "score 1-10",
        "range": "1 - 10",
        "logic": "clip(round(3.2 + 1.15 * tier + N(0, 0.9)), 1, 10).",
        "assumption": "State accountability rating rescaled to a ten-point scale.",
    },
    {
        "column": "walk_score",
        "dtype": "integer",
        "unit": "score 0-100",
        "range": "0 - 100",
        "logic": "clip(round(18 + 10.5 * tier + N(0, 8)), 0, 100).",
        "assumption": "Walk Score-style index of everyday errands on foot.",
    },
    {
        "column": "crime_index",
        "dtype": "integer",
        "unit": "index 0-100 (higher is worse)",
        "range": "0 - 100",
        "logic": "clip(round(78 - 10.5 * tier + N(0, 7)), 0, 100).",
        "assumption": "Inverse of a standardised property-crime rate at the census block level.",
    },
    {
        "column": "amenity_density_1_5km",
        "dtype": "integer",
        "unit": "count",
        "range": "0 - 40",
        "logic": "Poisson draw with mean 2 + 1.6 * tier.",
        "assumption": "Number of retail, education and health amenities within 1.5 km.",
    },
    {
        "column": "park_area_ha",
        "dtype": "float",
        "unit": "hectares",
        "range": "0.05 - 60.00",
        "logic": "Log-normal median exp(2.0 + 0.8 * tier), sigma 0.6, clipped.",
        "assumption": "Aggregate green space within the neighbourhood catchment.",
    },
    {
        "column": "amenity_score",
        "dtype": "float",
        "unit": "index 0-100",
        "range": "0 - 100",
        "logic": (
            "Weighted composite of inverse distances (school 0.25, hospital 0.15, transit 0.20), "
            "school rating (0.15), walk score (0.15) and inverse crime (0.10); z-scored with fixed "
            "reference moments and mapped linearly to 0-100."
        ),
        "assumption": "A single interpretable accessibility index for agents and buyers.",
    },
    {
        "column": "renovation_flag",
        "dtype": "integer (0/1)",
        "unit": "flag",
        "range": "0 - 1",
        "logic": "1 when YearRemodAdd > YearBuilt, otherwise 0.",
        "assumption": "The county assessor records the last remodelling permit year.",
    },
    {
        "column": "renovation_scope",
        "dtype": "category",
        "unit": "label",
        "range": "None / Minor / Moderate / Major",
        "logic": (
            "None when the property was never remodelled. Otherwise the scope is Major when the "
            "remodelling lag is at least 20 years, Moderate when the lag is at least 8 years, and Minor "
            "otherwise."
        ),
        "assumption": "Larger lags and higher quality imply a deeper intervention.",
    },
    {
        "column": "days_on_market",
        "dtype": "integer",
        "unit": "days",
        "range": "1 - 365",
        "logic": (
            "Log-normal with mean 16 + 7.5 * (5 - tier) + 4 * max(0, 8 - OverallQual), multiplied by "
            "a month-of-year factor (0.80 in June, 1.25 in January), the interest-rate ratio "
            "(rate / 5.5)^0.9 and the supply ratio (months_supply / 6.5)^0.5; clipped to 1-365."
        ),
        "assumption": "Slow markets, weaker locations and winter listings stay on the market longer.",
    },
    {
        "column": "listing_date",
        "dtype": "date",
        "unit": "yyyy-mm-dd",
        "range": "2005-01 - 2011-12",
        "logic": "Sale month (day 15) minus days_on_market.",
        "assumption": "A listing is published at least one day before the transaction closes.",
    },
    {
        "column": "mortgage_rate_pct",
        "dtype": "float",
        "unit": "percent per annum",
        "range": "4.30 - 7.10",
        "logic": (
            "Monthly path interpolated from documented anchors (6.40% in 2006-01, 6.80% in 2006-07, "
            "5.00% in 2009-06, 4.90% in 2010-12) plus AR(1) noise with sigma 0.05."
        ),
        "assumption": "30-year fixed conforming mortgage rate; approximates FRED MORTGAGE30US.",
    },
    {
        "column": "hpi_index",
        "dtype": "float",
        "unit": "index (2006-01 = 100)",
        "range": "94 - 109",
        "logic": "Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.6.",
        "assumption": "Local house-price index, calibrated to the mild Ames cycle of 2006-2011.",
    },
    {
        "column": "hpi_yoy_pct",
        "dtype": "float",
        "unit": "percent",
        "range": "-12 - 12",
        "logic": "Year-over-year percentage change of hpi_index.",
        "assumption": "Derived from the simulated index.",
    },
    {
        "column": "months_supply",
        "dtype": "float",
        "unit": "months",
        "range": "3.8 - 9.5",
        "logic": "Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.15.",
        "assumption": "Inventory divided by monthly absorption rate.",
    },
    {
        "column": "local_unemployment_pct",
        "dtype": "float",
        "unit": "percent",
        "range": "3.7 - 6.3",
        "logic": "Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.05.",
        "assumption": "Story County unemployment rate, approximates BLS LAUS.",
    },
)


# ---------------------------------------------------------------------------
# Macro series
# ---------------------------------------------------------------------------
def _month_ordinal(period: str) -> int:
    year, month = period.split("-")
    return int(year) * 12 + int(month) - 1


def _monthly_index(start: str = MACRO_START, end: str = MACRO_END) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{start}-01", end=f"{end}-01", freq="MS")


def _interpolate_anchors(
    anchors: tuple[tuple[str, float], ...], index: pd.DatetimeIndex
) -> np.ndarray:
    x = np.array([_month_ordinal(period) for period, _ in anchors], dtype=float)
    y = np.array([value for _, value in anchors], dtype=float)
    target = np.array(
        [month.year * 12 + month.month - 1 for month in index], dtype=float
    )
    return np.interp(target, x, y)


def _ar1_noise(
    n: int, sigma: float, rho: float, rng: np.random.Generator
) -> np.ndarray:
    innovations = rng.normal(0.0, sigma, size=n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def build_macro_series(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return the monthly macro-financial series used by the simulation."""
    index = _monthly_index()
    rng = np.random.default_rng(seed)
    n = len(index)

    rate = _interpolate_anchors(MORTGAGE_RATE_ANCHORS, index) + _ar1_noise(
        n, 0.05, 0.75, rng
    )
    hpi = _interpolate_anchors(HPI_ANCHORS, index) + _ar1_noise(n, 0.6, 0.70, rng)
    supply = _interpolate_anchors(MONTHS_SUPPLY_ANCHORS, index) + _ar1_noise(
        n, 0.15, 0.65, rng
    )
    unemployment = _interpolate_anchors(UNEMPLOYMENT_ANCHORS, index) + _ar1_noise(
        n, 0.05, 0.80, rng
    )

    frame = pd.DataFrame(
        {
            "month": index.strftime("%Y-%m"),
            "mortgage_rate_pct": np.round(rate, 3),
            "hpi_index": np.round(hpi, 2),
            "months_supply": np.round(supply, 2),
            "local_unemployment_pct": np.round(unemployment, 2),
        }
    )
    frame["hpi_yoy_pct"] = np.round(frame["hpi_index"].pct_change(12) * 100, 2)
    frame["hpi_yoy_pct"] = frame["hpi_yoy_pct"].fillna(0.0)
    return frame


# ---------------------------------------------------------------------------
# Property-level generation
# ---------------------------------------------------------------------------
class DriftParameters:
    """Parameters used to simulate a shifted post-deployment environment.

    A neutral instance reproduces the training-time regime. Positive
    ``rate_shift`` and ``supply_shift`` values emulate a tightening market;
    ``amenity_improvement`` emulates gentrification (amenity distances shrink
    and quality scores rise); ``renovation_boost`` increases the share of
    recently renovated stock.
    """

    def __init__(
        self,
        rate_shift: float = 0.0,
        supply_shift: float = 0.0,
        amenity_improvement: float = 0.0,
        renovation_boost: float = 0.0,
    ) -> None:
        self.rate_shift = rate_shift
        self.supply_shift = supply_shift
        self.amenity_improvement = amenity_improvement
        self.renovation_boost = renovation_boost


def _normalise_neighbourhood(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    mapping = {name.lower(): name for name in NEIGHBOURHOOD_TIERS}
    normalised = values.str.lower().map(mapping)
    fallback = values.where(normalised.isna(), normalised)
    return fallback.fillna("NAmes")


def _season_factor(month: pd.Series) -> np.ndarray:
    factors = {
        1: 1.25,
        2: 1.20,
        3: 1.05,
        4: 0.95,
        5: 0.85,
        6: 0.80,
        7: 0.85,
        8: 0.95,
        9: 1.00,
        10: 1.05,
        11: 1.15,
        12: 1.25,
    }
    return month.map(factors).to_numpy(dtype=float)


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def generate_context(
    properties: pd.DataFrame,
    seed: int = RANDOM_STATE,
    drift: DriftParameters | None = None,
) -> pd.DataFrame:
    """Generate contextual fields for the supplied property attributes.

    Parameters
    ----------
    properties:
        Frame with at least ``Id``, ``Neighborhood``, ``YrSold``, ``MoSold``,
        ``YearBuilt``, ``YearRemodAdd``, ``OverallQual`` and ``OverallCond``.
    seed:
        Seed for the pseudo-random draws. The function is deterministic for a
        given input order and seed.
    drift:
        Optional shifted-regime parameters used by the monitoring simulation.
    """
    required = {
        ID_COLUMN,
        "Neighborhood",
        "YrSold",
        "MoSold",
        "YearBuilt",
        "YearRemodAdd",
        "OverallQual",
        "OverallCond",
    }
    missing = required.difference(properties.columns)
    if missing:
        raise ValueError(f"Missing columns required for synthetic generation: {sorted(missing)}")

    drift = drift or DriftParameters()
    rng = np.random.default_rng(seed)
    base = properties.copy()
    base = base.sort_values(ID_COLUMN).reset_index(drop=True)

    neighbourhood = _normalise_neighbourhood(base["Neighborhood"])
    tier = neighbourhood.map(NEIGHBOURHOOD_TIERS).fillna(3).astype(float).to_numpy()
    n = len(base)

    # --- Location -----------------------------------------------------------
    lat = np.array([NEIGHBOURHOOD_CENTROIDS.get(name, (42.03, -93.63))[0] for name in neighbourhood])
    lon = np.array([NEIGHBOURHOOD_CENTROIDS.get(name, (42.03, -93.63))[1] for name in neighbourhood])
    lat = lat + rng.normal(0.0, 0.0015, size=n)
    lon = lon + rng.normal(0.0, 0.0015, size=n)

    def _lognormal(median: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(rng.normal(np.log(median), sigma, size=n))

    dist_school = np.clip(_lognormal(1.90 - 0.22 * tier, 0.35), 0.10, 8.0)
    dist_hospital = np.clip(_lognormal(4.80 - 0.50 * tier, 0.30), 0.30, 20.0)
    dist_transit = np.clip(_lognormal(3.20 - 0.42 * tier, 0.33), 0.20, 15.0)
    dist_park = np.clip(_lognormal(1.60 - 0.15 * tier, 0.30), 0.10, 10.0)

    # Gentrification compresses effective distances and lifts the scores.
    improvement = drift.amenity_improvement
    distance_scale = 1.0 - 0.20 * improvement
    dist_school = np.clip(dist_school * distance_scale, 0.10, 8.0)
    dist_hospital = np.clip(dist_hospital * distance_scale, 0.30, 20.0)
    dist_transit = np.clip(dist_transit * distance_scale, 0.20, 15.0)
    dist_park = np.clip(dist_park * distance_scale, 0.10, 10.0)

    school_rating = np.clip(
        np.round(3.2 + 1.15 * tier + rng.normal(0.0, 0.9, size=n)), 1, 10
    )
    walk_score = np.clip(
        np.round(18 + 10.5 * tier + rng.normal(0.0, 8.0, size=n) + 6 * improvement), 0, 100
    )
    crime_index = np.clip(
        np.round(78 - 10.5 * tier + rng.normal(0.0, 7.0, size=n) - 4 * improvement), 0, 100
    )
    amenity_density = rng.poisson(np.clip(2 + 1.6 * tier + 2 * improvement, 0.1, None))
    park_area = np.clip(_lognormal(2.0 + 0.8 * tier, 0.6), 0.05, 60.0)

    # Documented composite index with fixed reference moments (mean 0, sd 1).
    components = np.column_stack(
        [
            _zscore(-np.log1p(dist_school), -1.20, 0.55),
            _zscore(-np.log1p(dist_hospital), -1.95, 0.45),
            _zscore(-np.log1p(dist_transit), -1.55, 0.50),
            _zscore(school_rating.astype(float), 6.4, 1.6),
            _zscore(walk_score.astype(float), 44.0, 13.0),
            _zscore(-crime_index.astype(float), -46.0, 9.0),
        ]
    )
    weights = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    composite = components @ weights
    amenity_score = np.clip(50.0 + 22.0 * composite, 0.0, 100.0)

    # --- Renovation ---------------------------------------------------------
    year_built = base["YearBuilt"].astype(float).to_numpy()
    year_remod = base["YearRemodAdd"].astype(float).to_numpy()
    overall_qual = base["OverallQual"].astype(float).to_numpy()
    renovation_flag = (year_remod > year_built).astype(int)
    lag = np.clip(year_remod - year_built, 0, None)
    scope = np.where(
        renovation_flag == 0,
        "None",
        np.where(lag >= 20, "Major", np.where(lag >= 8, "Moderate", "Minor")),
    )
    if drift.renovation_boost > 0:
        boost_mask = (renovation_flag == 0) & (
            rng.random(n) < drift.renovation_boost
        )
        scope = np.where(boost_mask, "Minor", scope)
        renovation_flag = np.where(boost_mask, 1, renovation_flag)

    # --- Marketing duration -------------------------------------------------
    month = base["MoSold"].astype(int).clip(1, 12)
    market = build_macro_series(seed)
    period = (
        base["YrSold"].astype(int).astype(str)
        + "-"
        + month.astype(str).str.zfill(2)
    )
    market_lookup = market.set_index("month")
    rate = period.map(market_lookup["mortgage_rate_pct"]).to_numpy(dtype=float) + drift.rate_shift
    supply = period.map(market_lookup["months_supply"]).to_numpy(dtype=float) + drift.supply_shift
    hpi = period.map(market_lookup["hpi_index"]).to_numpy(dtype=float)
    hpi_yoy = period.map(market_lookup["hpi_yoy_pct"]).to_numpy(dtype=float)
    unemployment = period.map(market_lookup["local_unemployment_pct"]).to_numpy(dtype=float)

    base_days = 16 + 7.5 * (5 - tier) + 4 * np.maximum(0, 8 - overall_qual)
    mean_days = (
        base_days
        * _season_factor(month)
        * np.power(np.clip(rate, 3.0, None) / 5.5, 0.9)
        * np.power(np.clip(supply, 1.0, None) / 6.5, 0.5)
    )
    days_on_market = np.clip(
        np.round(np.exp(rng.normal(np.log(np.clip(mean_days, 2.0, None)) - 0.12, 0.55))),
        1,
        365,
    ).astype(int)

    sale_day = pd.to_datetime(
        {"year": base["YrSold"].astype(int), "month": month, "day": 15}
    )
    listing_date = sale_day - pd.to_timedelta(days_on_market, unit="D")

    context = pd.DataFrame(
        {
            ID_COLUMN: base[ID_COLUMN].astype(int).to_numpy(),
            "neigh_lat": np.round(lat, 6),
            "neigh_lon": np.round(lon, 6),
            "dist_school_km": np.round(dist_school, 3),
            "dist_hospital_km": np.round(dist_hospital, 3),
            "dist_transit_km": np.round(dist_transit, 3),
            "dist_park_km": np.round(dist_park, 3),
            "school_rating": school_rating.astype(int),
            "walk_score": walk_score.astype(int),
            "crime_index": crime_index.astype(int),
            "amenity_density_1_5km": amenity_density.astype(int),
            "park_area_ha": np.round(park_area, 3),
            "amenity_score": np.round(amenity_score, 2),
            "renovation_flag": renovation_flag.astype(int),
            "renovation_scope": scope,
            "days_on_market": days_on_market,
            "listing_date": listing_date.dt.strftime("%Y-%m-%d"),
            "mortgage_rate_pct": np.round(rate, 3),
            "hpi_index": np.round(hpi, 2),
            "hpi_yoy_pct": np.round(hpi_yoy, 3),
            "months_supply": np.round(supply, 2),
            "local_unemployment_pct": np.round(unemployment, 3),
        }
    )
    return context


def build_context_dataset(
    properties: pd.DataFrame,
    path=None,
    seed: int = RANDOM_STATE,
    drift: DriftParameters | None = None,
) -> pd.DataFrame:
    """Generate and persist the contextual frame (parquet by default)."""
    context = generate_context(properties, seed=seed, drift=drift)
    if path is None:
        path = DATA_SYNTHETIC / "property_context.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(path, index=False)
    return context


def load_context(path=None) -> pd.DataFrame:
    if path is None:
        path = DATA_SYNTHETIC / "property_context.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Synthetic context not found. Run `uv run python scripts/01_generate_context.py` first."
        )
    return pd.read_parquet(path)


def synthetic_specs_frame() -> pd.DataFrame:
    """Tabular specification of every synthetic field for the data dictionary."""
    return pd.DataFrame(list(SYNTHETIC_SPECS))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from revaluation.data.load import load_combined

    combined = load_combined()
    ctx = build_context_dataset(combined)
    print(ctx.head())
    print(ctx.describe(include="all").T)


def expected_context(
    neighborhood: str,
    sale_year: int,
    sale_month: int,
) -> dict[str, float]:
    """Central (noise-free) contextual values for a listing profile.

    Used by the serving layer when a caller does not supply amenity or macro
    fields. The values are the medians of the generating distributions, so the
    service remains consistent with the simulation logic instead of inventing a
    separate defaulting rule.
    """
    name = _normalise_neighbourhood(pd.Series([neighborhood])).iloc[0]
    tier = NEIGHBOURHOOD_TIERS.get(name, 3)
    latitude, longitude = NEIGHBOURHOOD_CENTROIDS.get(name, (42.03, -93.63))

    dist_school = 1.90 - 0.22 * tier
    dist_hospital = 4.80 - 0.50 * tier
    dist_transit = 3.20 - 0.42 * tier
    dist_park = 1.60 - 0.15 * tier
    school_rating = float(np.clip(round(3.2 + 1.15 * tier), 1, 10))
    walk_score = float(np.clip(round(18 + 10.5 * tier), 0, 100))
    crime_index = float(np.clip(round(78 - 10.5 * tier), 0, 100))
    amenity_density = float(max(0, round(2 + 1.6 * tier)))
    park_area = 2.0 + 0.8 * tier

    components = np.array(
        [
            (-np.log1p(dist_school) + 1.20) / 0.55,
            (-np.log1p(dist_hospital) + 1.95) / 0.45,
            (-np.log1p(dist_transit) + 1.55) / 0.50,
            (school_rating - 6.4) / 1.6,
            (walk_score - 44.0) / 13.0,
            (-crime_index + 46.0) / 9.0,
        ]
    )
    weights = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    amenity_score = float(np.clip(50.0 + 22.0 * float(components @ weights), 0.0, 100.0))

    macro = build_macro_series().set_index("month")
    month_key = f"{int(sale_year):04d}-{int(np.clip(sale_month, 1, 12)):02d}"
    if month_key not in macro.index:
        available = macro.index
        month_key = available[0] if month_key < available[0] else available[-1]
    row = macro.loc[month_key]

    return {
        "neigh_lat": latitude,
        "neigh_lon": longitude,
        "dist_school_km": round(dist_school, 3),
        "dist_hospital_km": round(dist_hospital, 3),
        "dist_transit_km": round(dist_transit, 3),
        "dist_park_km": round(dist_park, 3),
        "school_rating": school_rating,
        "walk_score": walk_score,
        "crime_index": crime_index,
        "amenity_density_1_5km": amenity_density,
        "park_area_ha": round(park_area, 3),
        "amenity_score": round(amenity_score, 2),
        "mortgage_rate_pct": float(row["mortgage_rate_pct"]),
        "hpi_index": float(row["hpi_index"]),
        "hpi_yoy_pct": float(row["hpi_yoy_pct"]),
        "months_supply": float(row["months_supply"]),
        "local_unemployment_pct": float(row["local_unemployment_pct"]),
    }
