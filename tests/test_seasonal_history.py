"""Tests for seasonally matched historical features."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.features.seasonal_history import (
    SEASONAL_FEATURE_COLUMNS,
    build_seasonal_history_features,
)


@pytest.fixture
def source():
    return pd.DataFrame({
        "FlightDate": ["2023-07-01"] * 3,
        "Origin": ["ATL"] * 3,
        "Reporting_Airline": ["DL", "DL", "UA"],
        "DepDel15": [1, 0, 1],
        "ArrDel15": [0, 1, 0],
        "Cancelled": [0, 0, 0],
        "Diverted": [0, 0, 0],
    })


@pytest.fixture
def flights():
    return pd.DataFrame({
        "FlightDate": ["2024-07-01"] * 3,
        "Origin": ["ATL"] * 3,
        "Reporting_Airline": ["DL", "UA", "ZZ"],
        "CRSDepTime": [800, 900, 1000],
    })


def run_features(flights, source, **kwargs):

    params = {
        "target_month": "2024-07",
        "source_month": "2023-07",
        "lag_days": 120,
        "min_airline_flights": 2,
    }

    params.update(kwargs)

    return build_seasonal_history_features(
        flights,
        source,
        **params,
    )


def test_origin_rates(flights, source):
    result = run_features(flights, source)

    assert np.allclose(
        result["origin_dep_delay_rate_ly"],
        2 / 3,
    )

    assert np.allclose(
        result["origin_outbound_arr_delay_rate_ly"],
        1 / 3,
    )


def test_direct_airline_match(flights, source):
    result = run_features(flights, source)

    assert result.loc[0, "history_status"] == "direct"

    assert np.isclose(
        result.loc[0, "airline_origin_dep_delay_rate_ly"],
        0.5,
    )


def test_low_sample_fallback(flights, source):
    result = run_features(flights, source)

    assert result.loc[1, "history_status"] == (
        "fallback_low_sample"
    )

    assert np.isclose(
        result.loc[1, "airline_origin_dep_delay_rate_ly"],
        2 / 3,
    )


def test_unknown_group_fallback(flights, source):
    result = run_features(flights, source)

    assert result.loc[2, "history_status"] == (
        "fallback_unknown_group"
    )

    assert np.isclose(
        result.loc[2, "airline_origin_arr_delay_rate_ly"],
        1 / 3,
    )


def test_incorrect_source_month(flights, source):
    with pytest.raises(ValueError, match="previous-year"):
        run_features(
            flights,
            source,
            source_month="2023-06",
        )


def test_source_not_available(flights, source):
    with pytest.raises(ValueError, match="not yet available"):
        run_features(
            flights,
            source,
            lag_days=400,
        )


def test_target_month_mismatch(flights, source):
    flights.loc[0, "FlightDate"] = "2024-08-01"

    with pytest.raises(ValueError, match="Target month"):
        run_features(flights, source)


def test_unresolved_timestamp(flights, source):
    flights.loc[0, "CRSDepTime"] = 2400

    result = run_features(flights, source)

    assert result.loc[0, "history_status"] == (
        "history_unavailable"
    )

    assert result.loc[
        0, list(SEASONAL_FEATURE_COLUMNS)
    ].isna().all()


def test_target_outcome_independence(flights, source):
    first = flights.assign(
        DepDel15=0,
        ArrDel15=0,
    )

    second = flights.assign(
        DepDel15=1,
        ArrDel15=1,
    )

    pd.testing.assert_frame_equal(
        run_features(first, source),
        run_features(second, source),
    )


def test_index_preservation(flights, source):
    flights.index = [20, 10, 30]

    result = run_features(flights, source)

    assert result.index.equals(flights.index)


def test_input_not_mutated(flights, source):
    original_flights = flights.copy(deep=True)
    original_source = source.copy(deep=True)

    run_features(flights, source)

    pd.testing.assert_frame_equal(
        flights,
        original_flights,
    )

    pd.testing.assert_frame_equal(
        source,
        original_source,
    )


def test_invalid_minimum_count(flights, source):
    with pytest.raises(ValueError, match="minimum"):
        run_features(
            flights,
            source,
            min_airline_flights=0,
        )