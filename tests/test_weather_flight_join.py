"""Offline tests for flight-to-weather point-in-time joins."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.features import MODEL_WEATHER_COLUMNS
from flight_delay.weather.flight_join import (
    FlightWeatherJoinError,
    join_weather_to_flights,
)


@pytest.fixture
def flights():
    frame = pd.DataFrame(
        {
            "Origin": ["ATL", "SFO", "LAX"],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:15:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T15:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    prediction = frame["scheduled_departure_utc"] - pd.Timedelta(hours=24)

    frame["forecast_run_at_utc"] = (prediction - pd.Timedelta(hours=3)).dt.floor("6h")

    frame["forecast_valid_at_utc"] = frame["scheduled_departure_utc"].dt.floor("h")

    frame["forecast_hour"] = (
        (
            frame["forecast_valid_at_utc"] - frame["forecast_run_at_utc"]
        ).dt.total_seconds()
        / 3600
    ).astype(int)

    return frame


@pytest.fixture
def weather_features(flights):
    rows = []

    for i, flight in enumerate(flights.itertuples(index=False)):
        record = {
            "Origin": flight.Origin,
            "forecast_run_at_utc": flight.forecast_run_at_utc,
            "forecast_valid_at_utc": flight.forecast_valid_at_utc,
            "forecast_hour": flight.forecast_hour,
            "assumed_available_at_utc": (
                flight.forecast_run_at_utc + pd.Timedelta(hours=3)
            ),
            "backend": "zarr",
            "fallback_reason": None,
        }

        for column in MODEL_WEATHER_COLUMNS:
            record[column] = float(i + 1)

        rows.append(record)

    return pd.DataFrame(rows)


def test_exact_three_flight_join(flights, weather_features):
    joined = join_weather_to_flights(flights, weather_features)

    assert len(joined) == 3
    assert joined["Origin"].tolist() == ["ATL", "SFO", "LAX"]

    assert joined["origin_temperature_c"].tolist() == [1.0, 2.0, 3.0]


def test_forecast_identity(flights, weather_features):
    joined = join_weather_to_flights(flights, weather_features)

    assert joined["forecast_hour"].tolist() == [27, 30, 27]

    assert joined["forecast_run_at_utc"].dt.hour.tolist() == [6, 6, 12]


def test_duplicate_flights_preserved(flights, weather_features):
    repeated = pd.concat(
        [flights, flights.iloc[[0]]],
        ignore_index=True,
    )

    joined = join_weather_to_flights(repeated, weather_features)

    assert len(joined) == 4
    assert joined["Origin"].tolist() == ["ATL", "SFO", "LAX", "ATL"]


def test_weather_provenance_preserved(flights, weather_features):
    weather_features.loc[0, "backend"] = "grib"
    weather_features.loc[0, "fallback_reason"] = "zarr_chunk_error"

    joined = join_weather_to_flights(flights, weather_features)

    assert joined.loc[0, "weather_backend"] == "grib"
    assert joined.loc[0, "weather_fallback_reason"] == "zarr_chunk_error"


def test_missing_weather_rejected(flights, weather_features):
    with pytest.raises(
        FlightWeatherJoinError,
        match="Missing weather forecast matches",
    ):
        join_weather_to_flights(
            flights,
            weather_features.iloc[:-1],
        )


def test_duplicate_weather_rejected(flights, weather_features):
    duplicated = pd.concat(
        [weather_features, weather_features.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        FlightWeatherJoinError,
        match="Duplicate weather forecast identity",
    ):
        join_weather_to_flights(flights, duplicated)


def test_future_weather_rejected(flights, weather_features):
    weather_features.loc[0, "assumed_available_at_utc"] += pd.Timedelta(days=2)

    with pytest.raises(
        FlightWeatherJoinError,
        match="unavailable at prediction time",
    ):
        join_weather_to_flights(flights, weather_features)


def test_invalid_forecast_run_rejected(flights, weather_features):
    flights.loc[0, "forecast_run_at_utc"] -= pd.Timedelta(hours=6)

    with pytest.raises(
        FlightWeatherJoinError,
        match="selection policy",
    ):
        join_weather_to_flights(flights, weather_features)


def test_wrong_valid_time_rejected(flights, weather_features):
    flights.loc[0, "forecast_valid_at_utc"] += pd.Timedelta(hours=1)

    with pytest.raises(
        FlightWeatherJoinError,
        match="departure hour",
    ):
        join_weather_to_flights(flights, weather_features)


def test_naive_departure_rejected(flights, weather_features):
    flights["scheduled_departure_utc"] = flights[
        "scheduled_departure_utc"
    ].dt.tz_localize(None)

    with pytest.raises(
        FlightWeatherJoinError,
        match="timezone-aware",
    ):
        join_weather_to_flights(flights, weather_features)


def test_nonfinite_weather_rejected(flights, weather_features):
    weather_features.loc[0, "origin_temperature_c"] = np.inf

    with pytest.raises(
        FlightWeatherJoinError,
        match="Nonfinite",
    ):
        join_weather_to_flights(flights, weather_features)


def test_inputs_are_not_modified(flights, weather_features):
    original_flights = flights.copy(deep=True)
    original_weather = weather_features.copy(deep=True)

    join_weather_to_flights(flights, weather_features)

    pd.testing.assert_frame_equal(flights, original_flights)

    pd.testing.assert_frame_equal(weather_features, original_weather)


def test_planner_prediction_time_mismatch_rejected(flights, weather_features):
    flights["prediction_at_utc"] = flights["scheduled_departure_utc"] - pd.Timedelta(
        hours=24
    )

    flights.loc[0, "prediction_at_utc"] += pd.Timedelta(minutes=5)

    with pytest.raises(
        FlightWeatherJoinError,
        match="Planner prediction_at_utc",
    ):
        join_weather_to_flights(flights, weather_features)


def test_valid_planner_prediction_time_preserved(flights, weather_features):
    flights["prediction_at_utc"] = flights["scheduled_departure_utc"] - pd.Timedelta(
        hours=24
    )

    joined = join_weather_to_flights(flights, weather_features)

    pd.testing.assert_series_equal(
        joined["prediction_at_utc"],
        flights["prediction_at_utc"],
    )
