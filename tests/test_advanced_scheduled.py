"""Tests for advanced scheduled flight features."""

import pandas as pd
import pytest

from flight_delay.features.advanced_scheduled import (
    ADVANCED_FEATURE_COLUMNS,
    build_advanced_scheduled_features,
    build_holiday_features,
    hhmm_to_hour,
)


@pytest.fixture
def sample_flights():
    return pd.DataFrame({
        "FlightDate": ["2023-07-03", "2023-07-04"],
        "Origin": ["SFO", "ATL"],
        "Dest": ["JFK", "LAX"],
        "Reporting_Airline": ["UA", "DL"],
        "CRSDepTime": [1730, 800],
        "CRSArrTime": [155, 1650],
    })


def test_route_features(sample_flights):
    result = build_advanced_scheduled_features(
        sample_flights
    )

    assert result["Route"].tolist() == [
        "SFO_JFK", "ATL_LAX"
    ]

    assert result["Airline_Route"].tolist() == [
        "UA_SFO_JFK", "DL_ATL_LAX"
    ]


@pytest.mark.parametrize(
    ("hhmm", "expected"),
    [
        (0, 0),
        (30, 0),
        (800, 8),
        (1730, 17),
        (2359, 23),
        (2400, 0),
    ],
)
def test_clock_hour(hhmm, expected):
    result = hhmm_to_hour(pd.Series([hhmm]))

    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    "hhmm",
    [-1, 1260, 2401, 2500, 830.5, None],
)
def test_invalid_clock_time(hhmm):
    with pytest.raises(ValueError):
        hhmm_to_hour(pd.Series([hhmm]))


def test_holiday_proximity():
    dates = pd.Series([
        "2023-07-03",
        "2023-07-04",
        "2023-07-05",
    ])

    result = build_holiday_features(dates)

    assert result[
        "is_us_federal_holiday"
    ].tolist() == [0, 1, 0]

    assert result.loc[
        0, "days_until_us_federal_holiday"
    ] == 1

    assert result.loc[
        2, "days_since_us_federal_holiday"
    ] == 1


def test_observed_holiday():
    result = build_holiday_features(
        pd.Series(["2023-01-02"])
    )

    assert result.loc[
        0, "is_us_federal_holiday"
    ] == 1


def test_year_boundary():
    result = build_holiday_features(
        pd.Series(["2023-12-31"])
    )

    assert result.loc[
        0, "days_until_us_federal_holiday"
    ] == 1


def test_schema_and_index(sample_flights):
    sample_flights.index = [20, 10]

    original = sample_flights.copy(deep=True)

    result = build_advanced_scheduled_features(
        sample_flights
    )

    assert result.columns.tolist() == list(
        ADVANCED_FEATURE_COLUMNS
    )

    assert result.index.equals(sample_flights.index)
    assert not result.isna().any().any()

    pd.testing.assert_frame_equal(
        sample_flights, original
    )


def test_missing_input_rejected(sample_flights):
    with pytest.raises(ValueError):
        build_advanced_scheduled_features(
            sample_flights.drop(columns=["Origin"])
        )


def test_target_independence(sample_flights):
    first = sample_flights.assign(
        DepDel15=[0, 1],
        ArrDel15=[1, 0],
    )

    second = sample_flights.assign(
        DepDel15=[1, 0],
        ArrDel15=[0, 1],
    )

    pd.testing.assert_frame_equal(
        build_advanced_scheduled_features(first),
        build_advanced_scheduled_features(second),
    )
