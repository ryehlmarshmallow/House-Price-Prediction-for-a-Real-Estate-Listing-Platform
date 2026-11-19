"""Tests for the synthetic contextual data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revaluation.data.synthetic import (
    DriftParameters,
    build_macro_series,
    expected_context,
    generate_context,
)


@pytest.fixture()
def properties() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Neighborhood": ["CollgCr", "NoRidge", "MeadowV", "OldTown"],
            "YrSold": [2006, 2008, 2010, 2007],
            "MoSold": [1, 6, 12, 9],
            "YearBuilt": [1970, 1990, 1950, 2000],
            "YearRemodAdd": [2000, 1990, 1980, 2005],
            "OverallQual": [6, 9, 4, 7],
            "OverallCond": [5, 8, 3, 7],
        }
    )


def test_generation_is_deterministic(properties: pd.DataFrame) -> None:
    first = generate_context(properties, seed=7)
    second = generate_context(properties, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_generation_does_not_read_the_target(properties: pd.DataFrame) -> None:
    baseline = generate_context(properties, seed=11)
    contaminated = properties.assign(
        SalePrice=[100_000.0, 500_000.0, 80_000.0, 220_000.0],
        log_price=np.log1p([100_000.0, 500_000.0, 80_000.0, 220_000.0]),
    )
    alternative = generate_context(contaminated, seed=11)
    pd.testing.assert_frame_equal(baseline, alternative)


def test_generated_ranges(properties: pd.DataFrame) -> None:
    context = generate_context(properties, seed=3)
    assert context["amenity_score"].between(0, 100).all()
    assert context["walk_score"].between(0, 100).all()
    assert context["crime_index"].between(0, 100).all()
    assert context["school_rating"].between(1, 10).all()
    assert context["days_on_market"].between(1, 365).all()
    assert (context["dist_school_km"] > 0).all()
    assert set(context["renovation_scope"]).issubset(
        {"None", "Minor", "Moderate", "Major"}
    )
    assert context["mortgage_rate_pct"].between(3.5, 8.5).all()


def test_drift_parameters_shift_the_context(properties: pd.DataFrame) -> None:
    neutral = generate_context(properties, seed=21)
    stress = generate_context(
        properties,
        seed=21,
        drift=DriftParameters(rate_shift=2.0, supply_shift=1.0, amenity_improvement=1.0),
    )
    assert stress["mortgage_rate_pct"].mean() > neutral["mortgage_rate_pct"].mean() + 1.5
    assert stress["dist_school_km"].mean() < neutral["dist_school_km"].mean()
    assert stress["walk_score"].mean() >= neutral["walk_score"].mean()


def test_macro_series_is_complete() -> None:
    macro = build_macro_series()
    assert len(macro) == 84
    assert macro[["mortgage_rate_pct", "hpi_index", "months_supply"]].notna().all().all()
    assert macro["mortgage_rate_pct"].between(3.5, 8.0).all()


def test_expected_context_is_consistent_with_tier() -> None:
    premium = expected_context("NoRidge", 2010, 6)
    value = expected_context("MeadowV", 2010, 6)
    assert premium["amenity_score"] > value["amenity_score"]
    assert premium["dist_school_km"] < value["dist_school_km"]
    assert premium["mortgage_rate_pct"] == value["mortgage_rate_pct"]
