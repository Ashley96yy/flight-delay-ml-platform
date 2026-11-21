"""HRRR spatial grid validation and airport mapping."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import pyproj

from flight_delay.weather.config import AIRPORTS

HRRR_GRID_SHAPE = (1059, 1799)

HRRR_GRID_SPACING_M = 3000.0

HRRR_X_START = -2697520.1425219304
HRRR_Y_START = -1587306.1525566636

GRID_COORDINATE_TOLERANCE_M = 0.01

MAX_AIRPORT_GRID_DISTANCE_KM = 3.0


HRRR_CRS = pyproj.CRS.from_proj4(
    "+proj=lcc "
    "+lat_0=38.5 "
    "+lon_0=-97.5 "
    "+lat_1=38.5 "
    "+lat_2=38.5 "
    "+a=6371229 "
    "+b=6371229 "
    "+units=m "
    "+no_defs"
)


FORWARD_TRANSFORMER = pyproj.Transformer.from_crs(
    HRRR_CRS.geodetic_crs,
    HRRR_CRS,
    always_xy=True,
)

INVERSE_TRANSFORMER = pyproj.Transformer.from_crs(
    HRRR_CRS,
    HRRR_CRS.geodetic_crs,
    always_xy=True,
)

HRRR_GEOD = HRRR_CRS.get_geod()


class WeatherGridError(ValueError):
    """Raised when spatial grid validation fails."""


def normalize_longitude(longitude):
    """Normalize longitude to [-180, 180)."""

    return (np.asarray(longitude) + 180) % 360 - 180


def validate_hrrr_coordinates(
    x_coords,
    y_coords,
) -> None:
    """Validate coordinates against the verified HRRR grid."""

    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    expected_y_size, expected_x_size = HRRR_GRID_SHAPE

    if x.shape != (expected_x_size,):
        raise WeatherGridError(f"Unexpected X coordinate shape: {x.shape}")

    if y.shape != (expected_y_size,):
        raise WeatherGridError(f"Unexpected Y coordinate shape: {y.shape}")

    if not np.isfinite(x).all():
        raise WeatherGridError("Nonfinite X coordinates")

    if not np.isfinite(y).all():
        raise WeatherGridError("Nonfinite Y coordinates")

    expected_x = HRRR_X_START + np.arange(expected_x_size) * HRRR_GRID_SPACING_M

    expected_y = HRRR_Y_START + np.arange(expected_y_size) * HRRR_GRID_SPACING_M

    if not np.allclose(
        x,
        expected_x,
        rtol=0,
        atol=GRID_COORDINATE_TOLERANCE_M,
    ):
        raise WeatherGridError("X coordinates do not match the verified HRRR grid")

    if not np.allclose(
        y,
        expected_y,
        rtol=0,
        atol=GRID_COORDINATE_TOLERANCE_M,
    ):
        raise WeatherGridError("Y coordinates do not match the verified HRRR grid")


def build_airport_grid_mapping(
    x_coords,
    y_coords,
) -> pd.DataFrame:
    """Map configured airports to nearest HRRR grid points."""

    validate_hrrr_coordinates(
        x_coords,
        y_coords,
    )

    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    records = []

    for airport in AIRPORTS:
        airport_x, airport_y = FORWARD_TRANSFORMER.transform(
            airport.longitude,
            airport.latitude,
        )

        if not (x[0] <= airport_x <= x[-1] and y[0] <= airport_y <= y[-1]):
            raise WeatherGridError(f"Airport outside HRRR domain: {airport.code}")

        x_index = int(np.argmin(np.abs(x - airport_x)))

        y_index = int(np.argmin(np.abs(y - airport_y)))

        grid_x = float(x[x_index])
        grid_y = float(y[y_index])

        grid_longitude, grid_latitude = INVERSE_TRANSFORMER.transform(
            grid_x,
            grid_y,
        )

        grid_longitude = float(normalize_longitude(grid_longitude))

        projected_distance_km = float(
            np.hypot(
                airport_x - grid_x,
                airport_y - grid_y,
            )
            / 1000
        )

        _, _, geographic_distance_m = HRRR_GEOD.inv(
            airport.longitude,
            airport.latitude,
            grid_longitude,
            grid_latitude,
        )

        geographic_distance_km = geographic_distance_m / 1000

        if (
            not np.isfinite(geographic_distance_km)
            or geographic_distance_km > MAX_AIRPORT_GRID_DISTANCE_KM
        ):
            raise WeatherGridError(
                f"Airport-grid distance exceeds tolerance: {airport.code}"
            )

        records.append(
            {
                "airport": airport.code,
                "airport_latitude": airport.latitude,
                "airport_longitude": airport.longitude,
                "airport_x": airport_x,
                "airport_y": airport_y,
                "x_index": x_index,
                "y_index": y_index,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_latitude": grid_latitude,
                "grid_longitude": grid_longitude,
                "projected_distance_km": projected_distance_km,
                "geographic_distance_km": geographic_distance_km,
            }
        )

    return pd.DataFrame(records)


@dataclass(frozen=True)
class ChunkLocation:
    """Location of a forecast value within a Zarr chunk."""

    chunk_indices: tuple[int, int, int]
    local_indices: tuple[int, int, int]
    chunk_key: str


def locate_chunk(
    *,
    array_index: int,
    y_index: int,
    x_index: int,
    array_shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    dimension_separator: str = ".",
) -> ChunkLocation:
    """Translate global indices into chunk-local indices."""

    indices = (
        array_index,
        y_index,
        x_index,
    )

    if len(array_shape) != 3 or len(chunk_shape) != 3:
        raise WeatherGridError("Expected three-dimensional array and chunk shapes")

    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
        for value in (*array_shape, *chunk_shape)
    ):
        raise WeatherGridError("Array and chunk dimensions must be positive integers")

    if any(
        not isinstance(index, Integral) or isinstance(index, bool) for index in indices
    ):
        raise WeatherGridError("Grid indices must be integers")

    if any(
        not 0 <= index < size
        for index, size in zip(
            indices,
            array_shape,
            strict=True,
        )
    ):
        raise WeatherGridError("Grid index out of bounds")

    if dimension_separator not in (".", "/"):
        raise WeatherGridError("Unsupported dimension separator")

    chunk_indices = tuple(
        int(index // size)
        for index, size in zip(
            indices,
            chunk_shape,
            strict=True,
        )
    )

    local_indices = tuple(
        int(index % size)
        for index, size in zip(
            indices,
            chunk_shape,
            strict=True,
        )
    )

    chunk_key = dimension_separator.join(map(str, chunk_indices))

    return ChunkLocation(
        chunk_indices=chunk_indices,
        local_indices=local_indices,
        chunk_key=chunk_key,
    )
