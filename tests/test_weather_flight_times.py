"""Tests for BTS local-to-UTC departure conversion."""

import pandas as pd
import pytest

from flight_delay.weather.flight_times import (
    AIRPORT_TIMEZONES,
    BTSDepartureTimeError,
    convert_bts_departure_to_utc,
)


def make_flights(origin, date, clock):
    return pd.DataFrame(
        {
            "Origin": [origin],
            "FlightDate": [date],
            "CRSDepTime": [clock],
        }
    )


@pytest.mark.parametrize(
    "origin,date,clock,expected",
    [
        (
            "ATL",
            "2024-01-15",
            1354,
            "2024-01-15T18:54:00Z",
        ),
        (
            "ATL",
            "2024-07-15",
            1354,
            "2024-07-15T17:54:00Z",
        ),
        (
            "DFW",
            "2024-07-15",
            900,
            "2024-07-15T14:00:00Z",
        ),
        (
            "DEN",
            "2024-01-15",
            900,
            "2024-01-15T16:00:00Z",
        ),
        (
            "ORD",
            "2024-01-15",
            900,
            "2024-01-15T15:00:00Z",
        ),
        (
            "LAX",
            "2024-07-15",
            915,
            "2024-07-15T16:15:00Z",
        ),
        (
            "SFO",
            "2024-01-15",
            915,
            "2024-01-15T17:15:00Z",
        ),
    ],
)
def test_airport_timezone_conversion(origin, date, clock, expected):
    result = convert_bts_departure_to_utc(make_flights(origin, date, clock))

    assert result.flights.loc[0, "scheduled_departure_utc"] == pd.Timestamp(expected)

    assert result.flights.loc[0, "departure_time_status"] == "valid"


@pytest.mark.parametrize(
    "date",
    ["2024-03-10", "2024-11-03"],
)
def test_daylight_saving_transition_requires_review(date):
    # 02:30 does not exist during spring-forward.
    # 01:30 is ambiguous during fall-back.

    clock = 230 if date.endswith("03-10") else 130

    result = convert_bts_departure_to_utc(make_flights("SFO", date, clock))

    assert result.flights.loc[0, "departure_time_status"] == "dst_unresolvable"

    assert pd.isna(result.flights.loc[0, "scheduled_departure_utc"])


def test_midnight_2400_requires_review():
    result = convert_bts_departure_to_utc(make_flights("ATL", "2024-01-15", 2400))

    assert result.flights.loc[0, "departure_time_status"] == "midnight_2400_review"

    assert pd.isna(result.flights.loc[0, "scheduled_departure_utc"])


@pytest.mark.parametrize("clock", [2360, 2500])
def test_invalid_clock_rejected(clock):
    result = convert_bts_departure_to_utc(make_flights("ATL", "2024-01-15", clock))

    assert result.flights.loc[0, "departure_time_status"] == "invalid_time"


def test_invalid_date_rejected():
    result = convert_bts_departure_to_utc(make_flights("ATL", "not-a-date", 1354))

    assert result.flights.loc[0, "departure_time_status"] == "invalid_date"


def test_unsupported_airport_rejected():
    result = convert_bts_departure_to_utc(make_flights("JFK", "2024-01-15", 1354))

    assert result.flights.loc[0, "departure_time_status"] == "unsupported_airport"


def test_leading_zero_clock():
    result = convert_bts_departure_to_utc(make_flights("LAX", "2024-07-15", "0001"))

    assert result.flights.loc[0, "scheduled_departure_utc"] == pd.Timestamp(
        "2024-07-15T07:01:00Z"
    )


def test_original_dataframe_preserved():
    flights = make_flights("ATL", "2024-01-15", 1354)

    original = flights.copy(deep=True)

    convert_bts_departure_to_utc(flights)

    pd.testing.assert_frame_equal(
        flights,
        original,
    )


def test_missing_columns_rejected():
    with pytest.raises(
        BTSDepartureTimeError,
        match="Missing BTS columns",
    ):
        convert_bts_departure_to_utc(pd.DataFrame({"Origin": ["ATL"]}))


def test_all_six_airports_configured():
    assert set(AIRPORT_TIMEZONES) == {"ATL", "DFW", "DEN", "ORD", "LAX", "SFO"}


def test_summary_counts():
    flights = pd.DataFrame(
        {
            "Origin": ["ATL", "SFO"],
            "FlightDate": [
                "2024-01-15",
                "2024-01-15",
            ],
            "CRSDepTime": [1354, 2500],
        }
    )

    result = convert_bts_departure_to_utc(flights)

    assert result.summary["total_rows"] == 2
    assert result.summary["valid_rows"] == 1
    assert result.summary["unresolved_rows"] == 1
    assert result.summary["valid_rate"] == 0.5
