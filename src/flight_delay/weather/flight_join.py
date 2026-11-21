"""Strict point-in-time join of flights and HRRR weather features."""

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    SUPPORTED_FORECAST_LEADS,
    WEATHER_AVAILABILITY_LAG_HOURS,
)
from flight_delay.weather.features import MODEL_WEATHER_COLUMNS


class FlightWeatherJoinError(ValueError):
    """Invalid flight-weather mapping, coverage, or timing."""


JOIN_KEYS = [
    "Origin",
    "forecast_run_at_utc",
    "forecast_valid_at_utc",
    "forecast_hour",
]

WEATHER_PROVENANCE = [
    "weather_assumed_available_at_utc",
    "weather_backend",
    "weather_fallback_reason",
]


def _utc_datetime(frame, column):
    """Require timezone-aware timestamps; never infer naive UTC."""

    try:
        values = pd.to_datetime(frame[column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise FlightWeatherJoinError(f"Invalid datetime values: {column}") from exc

    if not isinstance(values.dtype, pd.DatetimeTZDtype) or values.isna().any():
        raise FlightWeatherJoinError(f"{column} must contain timezone-aware timestamps")

    return values.dt.tz_convert("UTC")


def join_weather_to_flights(
    flight_mapping: pd.DataFrame,
    weather_features: pd.DataFrame,
    *,
    departure_col="scheduled_departure_utc",
) -> pd.DataFrame:
    """Join weather using exact forecast identity and T-24h checks."""

    if not isinstance(flight_mapping, pd.DataFrame):
        raise FlightWeatherJoinError("Flight mapping must be a DataFrame")

    if not isinstance(weather_features, pd.DataFrame):
        raise FlightWeatherJoinError("Weather features must be a DataFrame")

    if flight_mapping.empty:
        raise FlightWeatherJoinError("Empty flight mapping")

    required_flight = set(JOIN_KEYS + [departure_col])

    required_weather = set(
        JOIN_KEYS
        + MODEL_WEATHER_COLUMNS
        + [
            "assumed_available_at_utc",
            "backend",
            "fallback_reason",
        ]
    )

    if not required_flight.issubset(flight_mapping.columns):
        raise FlightWeatherJoinError(
            "Flight mapping missing columns: "
            f"{sorted(required_flight - set(flight_mapping.columns))}"
        )

    if not required_weather.issubset(weather_features.columns):
        raise FlightWeatherJoinError(
            "Weather features missing columns: "
            f"{sorted(required_weather - set(weather_features.columns))}"
        )

    reserved = set(
        MODEL_WEATHER_COLUMNS
        + WEATHER_PROVENANCE
        + ["__weather_order", "__weather_match"]
    )

    if reserved.intersection(flight_mapping.columns):
        raise FlightWeatherJoinError(
            "Flight mapping already contains reserved weather columns"
        )

    flights = flight_mapping.copy(deep=True)
    weather = weather_features.copy(deep=True)

    for column in (
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
    ):
        flights[column] = _utc_datetime(flights, column)
        weather[column] = _utc_datetime(weather, column)

    flights[departure_col] = _utc_datetime(flights, departure_col)

    weather["assumed_available_at_utc"] = _utc_datetime(
        weather, "assumed_available_at_utc"
    )

    # The prediction must always occur exactly 24 hours before
    # the scheduled departure.

    prediction_time = flights[departure_col] - pd.Timedelta(hours=24)

    # Validate the actual prediction timestamp supplied by Planner.
    if "prediction_at_utc" in flights.columns:
        planner_prediction = _utc_datetime(flights, "prediction_at_utc")

        if not planner_prediction.equals(prediction_time):
            raise FlightWeatherJoinError(
                "Planner prediction_at_utc differs from departure minus 24h"
            )

    if "prediction_time_utc" in flights.columns:
        supplied_prediction_time = _utc_datetime(flights, "prediction_time_utc")

        if not supplied_prediction_time.equals(prediction_time):
            raise FlightWeatherJoinError(
                "Prediction time differs from departure minus 24h"
            )

    else:
        flights["prediction_time_utc"] = prediction_time

    # The forecast must describe the scheduled departure hour.

    expected_valid = flights[departure_col].dt.floor("h")

    if not flights["forecast_valid_at_utc"].equals(expected_valid):
        raise FlightWeatherJoinError(
            "Forecast valid time does not match departure hour"
        )

    # Enforce the same forecast-selection policy as the planner:
    # latest six-hour cycle available three hours after initialization.

    expected_run = (
        prediction_time - pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)
    ).dt.floor("6h")

    if not flights["forecast_run_at_utc"].equals(expected_run):
        raise FlightWeatherJoinError(
            "Forecast run does not match T-24h selection policy"
        )

    calculated_leads = (
        flights["forecast_valid_at_utc"] - flights["forecast_run_at_utc"]
    ).dt.total_seconds() / 3600

    if (
        not np.isfinite(calculated_leads).all()
        or not calculated_leads.isin(SUPPORTED_FORECAST_LEADS).all()
        or not calculated_leads.eq(flights["forecast_hour"]).all()
    ):
        raise FlightWeatherJoinError("Forecast lead identity mismatch")

    if flights[JOIN_KEYS].isna().any().any():
        raise FlightWeatherJoinError("Missing flight forecast join key")

    if weather[JOIN_KEYS].isna().any().any():
        raise FlightWeatherJoinError("Missing weather forecast join key")

    if weather.duplicated(JOIN_KEYS).any():
        raise FlightWeatherJoinError("Duplicate weather forecast identity")

    values = weather[MODEL_WEATHER_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise FlightWeatherJoinError("Nonfinite model weather feature")

    selected = weather[
        JOIN_KEYS
        + MODEL_WEATHER_COLUMNS
        + [
            "assumed_available_at_utc",
            "backend",
            "fallback_reason",
        ]
    ].rename(
        columns={
            "assumed_available_at_utc": ("weather_assumed_available_at_utc"),
            "backend": "weather_backend",
            "fallback_reason": "weather_fallback_reason",
        }
    )

    # Keep original flight row order even when multiple flights share
    # the same airport/forecast combination.

    flights["__weather_order"] = np.arange(len(flights))

    joined = flights.merge(
        selected,
        on=JOIN_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="__weather_match",
    )

    if len(joined) != len(flights):
        raise FlightWeatherJoinError("Flight row count changed during weather join")

    if not joined["__weather_match"].eq("both").all():
        missing = joined.loc[
            joined["__weather_match"].eq("left_only"),
            JOIN_KEYS,
        ]

        raise FlightWeatherJoinError(
            f"Missing weather forecast matches: {missing.head(5).to_dict('records')}"
        )

    if joined[MODEL_WEATHER_COLUMNS].isna().any().any():
        raise FlightWeatherJoinError("Missing joined model weather features")

    # Check the actual prediction timestamp, not merely the
    # floored forecast-valid hour.

    if (
        not joined["weather_assumed_available_at_utc"]
        .le(joined["prediction_time_utc"])
        .all()
    ):
        raise FlightWeatherJoinError(
            "Weather information unavailable at prediction time"
        )

    joined = (
        joined.sort_values("__weather_order")
        .drop(columns=["__weather_order", "__weather_match"])
        .reset_index(drop=True)
    )

    return joined
