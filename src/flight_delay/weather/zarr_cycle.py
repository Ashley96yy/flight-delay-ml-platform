"""Cycle-level HRRR Zarr acquisition."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    ORIGIN_AIRPORTS,
    RAW_WEATHER_COLUMNS,
    SUPPORTED_FORECAST_LEADS,
    VARIABLE_BY_FEATURE,
    WEATHER_AVAILABILITY_LAG_HOURS,
)
from flight_delay.weather.contracts import validate_hourly_weather
from flight_delay.weather.grid import (
    build_airport_grid_mapping,
    locate_chunk,
    validate_hrrr_coordinates,
)
from flight_delay.weather.zarr_backend import (
    ZarrChunkError,
    build_zarr_array_path,
    build_zarr_group_path,
    decode_zarr_chunk,
    load_zarr_array_metadata,
    normalize_zarr_value,
)


class ZarrCoordinateError(ValueError):
    """Invalid forecast or spatial coordinate information."""


@dataclass(frozen=True)
class ZarrCoordinateData:
    reference_time: object
    forecast_periods: np.ndarray
    valid_times: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray


@dataclass
class ZarrCycleResult:
    weather: pd.DataFrame
    chunk_audit: pd.DataFrame


def load_zarr_coordinates(fs, run_at, feature):
    """Read actual coordinate arrays from the variable group."""

    import xarray as xr

    group_path = build_zarr_group_path(run_at, feature)

    ds = xr.open_zarr(
        fs.get_mapper(group_path),
        consolidated=False,
        chunks=None,
        decode_times=True,
        decode_timedelta=True,
    )

    try:
        return ZarrCoordinateData(
            reference_time=np.asarray(ds["forecast_reference_time"].values).copy(),
            forecast_periods=np.asarray(ds["forecast_period"].values).copy(),
            valid_times=np.asarray(ds["time"].values).copy(),
            x_coords=np.asarray(ds["projection_x_coordinate"].values).copy(),
            y_coords=np.asarray(ds["projection_y_coordinate"].values).copy(),
        )
    finally:
        ds.close()


def validate_zarr_forecast_coordinates(
    coordinates,
    run_at,
    metadata,
):
    """Validate forecast identity and return lead-to-index mapping."""

    run_at = pd.Timestamp(run_at)

    if run_at.tzinfo is None:
        raise ZarrCoordinateError("Forecast initialization must be timezone-aware")

    run_at = run_at.tz_convert("UTC")

    references = pd.to_datetime(
        np.asarray(coordinates.reference_time).reshape(-1),
        utc=True,
    )

    if len(references) != 1 or references[0] != run_at:
        raise ZarrCoordinateError("Forecast initialization mismatch")

    periods = pd.to_timedelta(np.asarray(coordinates.forecast_periods).reshape(-1))

    valid_times = pd.to_datetime(
        np.asarray(coordinates.valid_times).reshape(-1),
        utc=True,
    )

    if (
        periods.isna().any()
        or valid_times.isna().any()
        or len(periods) != metadata.shape[0]
        or len(valid_times) != metadata.shape[0]
    ):
        raise ZarrCoordinateError("Invalid forecast coordinate lengths or values")

    hours = np.asarray(
        periods / pd.Timedelta(hours=1),
        dtype=float,
    )

    if not np.array_equal(
        hours,
        np.arange(1, metadata.shape[0] + 1),
    ):
        raise ZarrCoordinateError("Unexpected forecast lead sequence")

    expected_valid_times = run_at + periods

    if not valid_times.equals(expected_valid_times):
        raise ZarrCoordinateError("Forecast valid time mismatch")

    try:
        validate_hrrr_coordinates(
            coordinates.x_coords,
            coordinates.y_coords,
        )
    except ValueError as exc:
        raise ZarrCoordinateError("HRRR spatial coordinates mismatch") from exc

    if metadata.shape[1:] != (
        len(coordinates.y_coords),
        len(coordinates.x_coords),
    ):
        raise ZarrCoordinateError("Coordinate dimensions mismatch")

    return {int(lead): int(index) for index, lead in enumerate(hours)}


def extract_zarr_cycle(
    fs,
    run_at,
    forecast_leads,
    *,
    coordinate_loader: Callable | None = None,
) -> ZarrCycleResult:
    """Extract nine variables for six airports and requested leads."""

    run_at = pd.Timestamp(run_at)

    if run_at.tzinfo is None:
        raise ZarrCoordinateError("Forecast initialization must be timezone-aware")

    run_at = run_at.tz_convert("UTC")

    leads = list(forecast_leads)

    if (
        not leads
        or any(
            not isinstance(lead, (int, np.integer))
            or isinstance(lead, (bool, np.bool_))
            or lead not in SUPPORTED_FORECAST_LEADS
            for lead in leads
        )
        or len(leads) != len(set(leads))
    ):
        raise ZarrCoordinateError("Invalid or duplicate forecast leads")

    leads = sorted(leads)

    if coordinate_loader is None:
        coordinate_loader = load_zarr_coordinates

    records = []
    audits = []

    airport_mapping = None

    for feature in RAW_WEATHER_COLUMNS:
        metadata = load_zarr_array_metadata(fs, run_at, feature)

        coordinates = coordinate_loader(fs, run_at, feature)

        lead_mapping = validate_zarr_forecast_coordinates(
            coordinates,
            run_at,
            metadata,
        )

        if airport_mapping is None:
            airport_mapping = build_airport_grid_mapping(
                coordinates.x_coords,
                coordinates.y_coords,
            )

        array_path = build_zarr_array_path(run_at, feature)

        variable = VARIABLE_BY_FEATURE[feature]

        # Cache compressed objects within this variable and cycle.
        # All requested leads normally share one time chunk.

        chunk_cache = {}

        for airport in airport_mapping.itertuples(index=False):
            for lead in leads:
                array_index = lead_mapping[lead]

                location = locate_chunk(
                    array_index=array_index,
                    y_index=int(airport.y_index),
                    x_index=int(airport.x_index),
                    array_shape=metadata.shape,
                    chunk_shape=metadata.chunks,
                    dimension_separator=metadata.dimension_separator,
                )

                key = location.chunk_key

                if key not in chunk_cache:
                    object_path = f"{array_path}/{key}"

                    started = perf_counter()

                    try:
                        compressed = fs.cat_file(object_path)
                    except (OSError, KeyError) as exc:
                        raise ZarrChunkError(
                            f"Unable to read Zarr chunk: {object_path}"
                        ) from exc

                    download_seconds = perf_counter() - started

                    started = perf_counter()

                    chunk = decode_zarr_chunk(compressed, metadata)

                    decode_seconds = perf_counter() - started

                    chunk_cache[key] = chunk

                    audits.append(
                        {
                            "feature": feature,
                            "chunk_key": key,
                            "object_path": object_path,
                            "compressed_bytes": len(compressed),
                            "decoded_bytes": chunk.nbytes,
                            "download_seconds": download_seconds,
                            "decode_seconds": decode_seconds,
                            "storage_dtype": str(metadata.dtype),
                        }
                    )

                value = normalize_zarr_value(
                    chunk_cache[key][location.local_indices],
                    metadata,
                )

                # Cloud ceiling remains optional and its height
                # semantics are not yet normalized to GRIB units.
                units = (
                    variable.grib_units if variable.required else metadata.source_units
                )

                records.append(
                    {
                        "Origin": airport.airport,
                        "feature": feature,
                        "forecast_run_at_utc": run_at,
                        "forecast_valid_at_utc": (run_at + pd.Timedelta(hours=lead)),
                        "forecast_hour": lead,
                        "value": value.value,
                        "raw_value": value.raw_value,
                        "value_status": value.value_status,
                        "assumed_available_at_utc": (
                            run_at + pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)
                        ),
                        "grid_latitude": airport.grid_latitude,
                        "grid_longitude": airport.grid_longitude,
                        "backend": "zarr",
                        "storage_dtype": str(metadata.dtype),
                        "units": units,
                        "source_units": metadata.source_units,
                        "fallback_reason": None,
                    }
                )

        # Release decoded chunks before processing next variable.
        chunk_cache.clear()

    weather = pd.DataFrame(records)

    weather = weather.sort_values(["forecast_hour", "feature", "Origin"]).reset_index(
        drop=True
    )

    for lead in leads:
        hourly_frame = weather.loc[weather["forecast_hour"].eq(lead)]

        validate_hourly_weather(hourly_frame)

    expected_count = len(RAW_WEATHER_COLUMNS) * len(ORIGIN_AIRPORTS) * len(leads)

    if len(weather) != expected_count:
        raise ZarrChunkError("Incomplete cycle extraction")

    return ZarrCycleResult(
        weather=weather,
        chunk_audit=pd.DataFrame(audits),
    )
