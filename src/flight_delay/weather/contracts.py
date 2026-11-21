"""Unified weather data contract and validation."""

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    CORE_RAW_WEATHER_COLUMNS,
    EXTENDED_CYCLE_HOURS,
    ORIGIN_AIRPORTS,
    RAW_WEATHER_COLUMNS,
    SUPPORTED_FORECAST_LEADS,
    WEATHER_AVAILABILITY_LAG_HOURS,
)

FORECAST_MATCH_KEYS = (
    "Origin",
    "feature",
    "forecast_run_at_utc",
    "forecast_valid_at_utc",
    "forecast_hour",
)


WEATHER_RECORD_COLUMNS = (
    *FORECAST_MATCH_KEYS,
    "value",
    "raw_value",
    "value_status",
    "assumed_available_at_utc",
    "grid_latitude",
    "grid_longitude",
    "backend",
    "storage_dtype",
    "units",
    "source_units",
    "fallback_reason",
)


SUPPORTED_BACKENDS = frozenset(
    {
        "zarr",
        "grib",
    }
)

SUPPORTED_VALUE_STATUSES = frozenset(
    {
        "valid",
        "fill_value",
        "nonfinite",
        "source_missing",
    }
)


class WeatherContractError(ValueError):
    """Raised when weather records violate the contract."""


def require(condition: bool, message: str) -> None:
    """Raise an explicit validation error."""

    if not condition:
        raise WeatherContractError(message)


def validate_hourly_weather(
    frame: pd.DataFrame,
    *,
    require_core_values: bool = True,
) -> None:
    """Validate a complete six-airport hourly weather batch."""

    missing_columns = set(WEATHER_RECORD_COLUMNS) - set(frame.columns)

    require(
        not missing_columns,
        f"Missing columns: {sorted(missing_columns)}",
    )

    expected_records = len(ORIGIN_AIRPORTS) * len(RAW_WEATHER_COLUMNS)

    require(
        len(frame) == expected_records,
        f"Expected {expected_records} records, got {len(frame)}",
    )

    require(
        not frame[list(FORECAST_MATCH_KEYS)].isna().any().any(),
        "Missing forecast identity",
    )

    require(
        not frame.duplicated(subset=["Origin", "feature"]).any(),
        "Duplicate airport-feature records",
    )

    require(
        set(frame["Origin"]) == set(ORIGIN_AIRPORTS),
        "Unexpected airport coverage",
    )

    require(
        set(frame["feature"]) == set(RAW_WEATHER_COLUMNS),
        "Unexpected feature coverage",
    )

    # Validate UTC timestamps without silently localizing them.

    timestamp_columns = (
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
        "assumed_available_at_utc",
    )

    for column in timestamp_columns:
        dtype = frame[column].dtype

        require(
            isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC",
            f"{column} must be timezone-aware UTC",
        )

        require(
            frame[column].notna().all(),
            f"Missing timestamps in {column}",
        )

        require(
            frame[column].nunique() == 1,
            f"Multiple timestamps in {column}",
        )

    require(
        frame["forecast_hour"].nunique() == 1,
        "Multiple forecast leads",
    )

    lead = frame["forecast_hour"].iloc[0]

    require(
        lead in SUPPORTED_FORECAST_LEADS,
        f"Unsupported forecast lead: {lead}",
    )

    run_at = frame["forecast_run_at_utc"].iloc[0]

    valid_at = frame["forecast_valid_at_utc"].iloc[0]

    available_at = frame["assumed_available_at_utc"].iloc[0]

    require(
        run_at.hour in EXTENDED_CYCLE_HOURS and run_at == run_at.floor("h"),
        "Invalid initialization cycle",
    )

    require(
        valid_at == run_at + pd.Timedelta(hours=lead),
        "Forecast valid time does not match lead",
    )

    require(
        available_at == (run_at + pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)),
        "Incorrect assumed availability timestamp",
    )

    # Validate source provenance.

    require(
        frame["backend"].isin(SUPPORTED_BACKENDS).all(),
        "Unsupported weather backend",
    )

    zarr_mask = frame["backend"].eq("zarr")

    require(
        frame.loc[zarr_mask, "storage_dtype"].isin(["float16", "float32"]).all(),
        "Unsupported Zarr storage dtype",
    )

    require(
        frame["value_status"].isin(SUPPORTED_VALUE_STATUSES).all(),
        "Unsupported weather value status",
    )

    # Validate normalized weather values.

    valid_mask = frame["value_status"].eq("valid")

    require(
        pd.api.types.is_numeric_dtype(frame["value"]),
        "Weather values must be numeric",
    )

    require(
        np.isfinite(frame.loc[valid_mask, "value"].to_numpy(dtype=float)).all(),
        "Nonfinite value marked as valid",
    )

    require(
        frame.loc[~valid_mask, "value"].isna().all(),
        "Missing weather values must be NaN",
    )

    # Validate spatial coordinates.

    require(
        frame["grid_latitude"].between(-90, 90).all(),
        "Invalid grid latitude",
    )

    require(
        frame["grid_longitude"].ge(-180).all()
        and frame["grid_longitude"].lt(180).all(),
        "Grid longitude must be normalized to [-180, 180)",
    )

    # Core model inputs must not contain unresolved missingness.

    if require_core_values:
        core_mask = frame["feature"].isin(CORE_RAW_WEATHER_COLUMNS)

        require(
            frame.loc[core_mask, "value"].notna().all(),
            "Missing required weather values",
        )
