"""Tests for the unified weather data contract."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.config import (
    AIRPORTS,
    CORE_RAW_WEATHER_COLUMNS,
    MODEL_WEATHER_COLUMNS,
    RAW_WEATHER_COLUMNS,
    VARIABLE_BY_FEATURE,
    WEATHER_VARIABLES,
)
from flight_delay.weather.contracts import (
    WeatherContractError,
    validate_hourly_weather,
)


@pytest.fixture
def weather_frame():

    run_at = pd.Timestamp("2024-07-14T06:00:00Z")

    lead = 30

    records = []

    for airport in AIRPORTS:
        for variable in WEATHER_VARIABLES:
            missing = not variable.required

            records.append(
                {
                    "Origin": airport.code,
                    "feature": variable.feature,
                    "forecast_run_at_utc": run_at,
                    "forecast_valid_at_utc": (run_at + pd.Timedelta(hours=lead)),
                    "forecast_hour": lead,
                    "assumed_available_at_utc": (run_at + pd.Timedelta(hours=3)),
                    "value": np.nan if missing else 1.0,
                    "raw_value": -9999.0 if missing else 1.0,
                    "value_status": ("fill_value" if missing else "valid"),
                    "grid_latitude": airport.latitude,
                    "grid_longitude": airport.longitude,
                    "backend": "zarr",
                    "storage_dtype": "float32",
                    "units": variable.grib_units,
                    "source_units": None,
                    "fallback_reason": None,
                }
            )

    return pd.DataFrame(records)


def test_weather_configuration():

    assert len(AIRPORTS) == 6

    assert len(RAW_WEATHER_COLUMNS) == 9

    assert len(CORE_RAW_WEATHER_COLUMNS) == 8

    assert len(MODEL_WEATHER_COLUMNS) == 7

    assert "cloud_ceiling_raw" not in (CORE_RAW_WEATHER_COLUMNS)


def test_precipitation_configuration():

    variable = VARIABLE_BY_FEATURE["precipitation_1h"]

    assert variable.zarr_variable == "APCP_1hr_acc_fcst"

    assert variable.grib_step_type == "accum"


def test_valid_hourly_weather(weather_frame):

    assert len(weather_frame) == 54

    validate_hourly_weather(weather_frame)


def test_float16_supported(weather_frame):

    weather_frame["storage_dtype"] = "float16"

    validate_hourly_weather(weather_frame)


def test_invalid_storage_dtype(weather_frame):

    weather_frame["storage_dtype"] = "int32"

    with pytest.raises(
        WeatherContractError,
        match="Unsupported Zarr storage dtype",
    ):
        validate_hourly_weather(weather_frame)


def test_duplicate_record_rejected(weather_frame):

    weather_frame.iloc[-1] = weather_frame.iloc[0]

    with pytest.raises(
        WeatherContractError,
        match="Duplicate",
    ):
        validate_hourly_weather(weather_frame)


def test_incomplete_batch_rejected(weather_frame):

    frame = weather_frame.iloc[:-1].copy()

    with pytest.raises(
        WeatherContractError,
        match="Expected 54 records",
    ):
        validate_hourly_weather(frame)


def test_invalid_forecast_timestamp(weather_frame):

    weather_frame["forecast_valid_at_utc"] += pd.Timedelta(hours=1)

    with pytest.raises(
        WeatherContractError,
        match="Forecast valid time",
    ):
        validate_hourly_weather(weather_frame)


def test_naive_timestamp_rejected(weather_frame):

    weather_frame["forecast_run_at_utc"] = weather_frame[
        "forecast_run_at_utc"
    ].dt.tz_localize(None)

    with pytest.raises(
        WeatherContractError,
        match="timezone-aware UTC",
    ):
        validate_hourly_weather(weather_frame)


def test_invalid_availability_timestamp(weather_frame):

    weather_frame["assumed_available_at_utc"] += pd.Timedelta(hours=1)

    with pytest.raises(
        WeatherContractError,
        match="Incorrect assumed availability",
    ):
        validate_hourly_weather(weather_frame)


def test_optional_cloud_ceiling_missing(weather_frame):

    ceiling = weather_frame.loc[weather_frame["feature"].eq("cloud_ceiling_raw")]

    assert ceiling["value"].isna().all()

    validate_hourly_weather(weather_frame)


def test_core_missing_value_rejected(weather_frame):

    weather_frame.loc[0, "value"] = np.nan

    weather_frame.loc[0, "value_status"] = "source_missing"

    with pytest.raises(
        WeatherContractError,
        match="Missing required weather values",
    ):
        validate_hourly_weather(weather_frame)


def test_core_missing_allowed_before_fallback(weather_frame):

    weather_frame.loc[0, "value"] = np.nan

    weather_frame.loc[0, "value_status"] = "source_missing"

    validate_hourly_weather(
        weather_frame,
        require_core_values=False,
    )


def test_nonfinite_valid_value_rejected(weather_frame):

    weather_frame.loc[0, "value"] = np.inf

    with pytest.raises(
        WeatherContractError,
        match="Nonfinite value marked as valid",
    ):
        validate_hourly_weather(weather_frame)


def test_invalid_longitude_rejected(weather_frame):

    weather_frame.loc[0, "grid_longitude"] = 275.0

    with pytest.raises(
        WeatherContractError,
        match="Grid longitude",
    ):
        validate_hourly_weather(weather_frame)


def test_grib_fallback_supported(weather_frame):

    weather_frame.loc[0, "backend"] = "grib"

    weather_frame.loc[0, "storage_dtype"] = None

    weather_frame.loc[0, "fallback_reason"] = "missing_zarr_chunk"

    validate_hourly_weather(weather_frame)
