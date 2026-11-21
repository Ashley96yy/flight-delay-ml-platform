"""HRRR Zarr metadata validation and chunk decoding."""

import json
from dataclasses import dataclass
from typing import Any

import numcodecs
import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    EXTENDED_CYCLE_HOURS,
    VARIABLE_BY_FEATURE,
    ZARR_BUCKET,
)

EXPECTED_ARRAY_SHAPE = (48, 1059, 1799)

EXPECTED_CHUNK_SHAPE = (48, 150, 150)

EXPECTED_DIMENSIONS = (
    "time",
    "projection_y_coordinate",
    "projection_x_coordinate",
)


EXPECTED_GRIB_PARAMETERS = {
    "temperature_2m": (2, 0, 0, 0),
    "u_wind_10m": (2, 0, 2, 2),
    "v_wind_10m": (2, 0, 2, 3),
    "wind_gust": (2, 0, 2, 22),
    "visibility": (2, 0, 19, 0),
    "precipitation_1h": (2, 0, 1, 8),
    "surface_cape": (2, 0, 7, 6),
    "composite_reflectivity": (2, 0, 16, 196),
    "cloud_ceiling_raw": (2, 0, 3, 5),
}


# Known unit representations from the feasibility experiments.
# None is permitted only for previously observed missing metadata.

ALLOWED_SOURCE_UNITS = {
    "temperature_2m": {"K"},
    "u_wind_10m": {"m s-1", "m s**-1"},
    "v_wind_10m": {"m s-1", "m s**-1"},
    "wind_gust": {"m s-1", "m s**-1"},
    "visibility": {"m", None},
    "precipitation_1h": {"kg m-2", "kg m**-2", None},
    "surface_cape": {"J kg-1", "J kg**-1"},
    "composite_reflectivity": {"dB", "dBZ", None},
    "cloud_ceiling_raw": {"m", "gpm"},
}


class ZarrBackendError(RuntimeError):
    """Raised when HRRR Zarr acquisition fails."""


class ZarrMetadataError(ZarrBackendError):
    """Raised when array metadata is unsupported."""


class ZarrChunkError(ZarrBackendError):
    """Raised when a chunk cannot be decoded."""


@dataclass(frozen=True)
class ZarrArrayMetadata:
    """Validated Zarr array storage contract."""

    feature: str
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dtype: np.dtype
    order: str
    dimension_separator: str
    declared_fill_value: float
    stored_fill_value: float
    compressor: dict[str, Any]
    source_units: str | None

    @property
    def expected_chunk_bytes(self) -> int:
        return int(np.prod(self.chunks)) * self.dtype.itemsize


@dataclass(frozen=True)
class ZarrValue:
    """Extracted value with missing-value provenance."""

    value: float
    raw_value: float
    value_status: str


def build_zarr_group_path(
    run_at,
    feature: str,
) -> str:
    """Build the HRRR forecast variable group path."""

    if feature not in VARIABLE_BY_FEATURE:
        raise ZarrMetadataError(f"Unsupported weather feature: {feature}")

    try:
        timestamp = pd.Timestamp(run_at)

    except (ValueError, TypeError, OverflowError) as exc:
        raise ZarrMetadataError("Invalid forecast initialization timestamp") from exc

    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ZarrMetadataError("Timezone-aware initialization required")

    timestamp = timestamp.tz_convert("UTC")

    if timestamp.hour not in EXTENDED_CYCLE_HOURS or timestamp != timestamp.floor("h"):
        raise ZarrMetadataError("Invalid extended forecast cycle")

    variable = VARIABLE_BY_FEATURE[feature]

    date = timestamp.strftime("%Y%m%d")
    cycle = timestamp.strftime("%H")

    return (
        f"{ZARR_BUCKET}/sfc/{date}/"
        f"{date}_{cycle}z_fcst.zarr/"
        f"{variable.zarr_level}/"
        f"{variable.zarr_variable}"
    )


def build_zarr_array_path(
    run_at,
    feature: str,
) -> str:
    """Build the nested HRRR Zarr data array path."""

    group_path = build_zarr_group_path(
        run_at,
        feature,
    )

    variable = VARIABLE_BY_FEATURE[feature]

    return f"{group_path}/{variable.zarr_level}/{variable.zarr_variable}"


def read_zarr_json(fs, path: str) -> dict:
    """Read a Zarr JSON metadata object."""

    try:
        result = json.loads(fs.cat_file(path))

    except (OSError, ValueError) as exc:
        raise ZarrMetadataError(f"Unable to read Zarr metadata: {path}") from exc

    if not isinstance(result, dict):
        raise ZarrMetadataError(f"Invalid metadata object: {path}")

    return result


def parse_zarr_metadata(
    feature: str,
    metadata: dict,
    attributes: dict,
) -> ZarrArrayMetadata:
    """Validate the storage and variable metadata."""

    if feature not in VARIABLE_BY_FEATURE:
        raise ZarrMetadataError(f"Unsupported weather feature: {feature}")

    variable = VARIABLE_BY_FEATURE[feature]

    required_keys = {
        "zarr_format",
        "shape",
        "chunks",
        "dtype",
        "compressor",
        "filters",
        "order",
        "fill_value",
    }

    missing_keys = required_keys - set(metadata)

    if missing_keys:
        raise ZarrMetadataError(f"Missing metadata keys: {sorted(missing_keys)}")

    if metadata["zarr_format"] != 2:
        raise ZarrMetadataError("Only Zarr format 2 is supported")

    if tuple(metadata["shape"]) != EXPECTED_ARRAY_SHAPE:
        raise ZarrMetadataError("Unsupported Zarr array shape")

    if tuple(metadata["chunks"]) != EXPECTED_CHUNK_SHAPE:
        raise ZarrMetadataError("Unsupported Zarr chunk shape")

    if tuple(attributes.get("_ARRAY_DIMENSIONS", ())) != EXPECTED_DIMENSIONS:
        raise ZarrMetadataError("Unexpected Zarr array dimensions")

    try:
        dtype = np.dtype(metadata["dtype"])

    except (TypeError, ValueError) as exc:
        raise ZarrMetadataError("Invalid storage dtype") from exc

    if dtype.kind != "f" or dtype.itemsize not in (2, 4):
        raise ZarrMetadataError(f"Unsupported storage dtype: {dtype}")

    if metadata["filters"] not in (None, []):
        raise ZarrMetadataError("Unsupported Zarr filters")

    order = metadata["order"]

    if order not in ("C", "F"):
        raise ZarrMetadataError("Unsupported array order")

    separator = metadata.get("dimension_separator", ".")

    if separator not in (".", "/"):
        raise ZarrMetadataError("Unsupported dimension separator")

    # No additional scale or offset is applied in the pilots.

    if attributes.get("scale_factor", 1) != 1 or attributes.get("add_offset", 0) != 0:
        raise ZarrMetadataError("Unsupported scale or offset")

    # Verify the actual meteorological variable identity.

    grib_parameter = tuple(attributes.get("GRIB_PARAM", ()))

    if grib_parameter != EXPECTED_GRIB_PARAMETERS[feature]:
        raise ZarrMetadataError(f"GRIB parameter mismatch: {feature}")

    expected_long_name = f"{variable.zarr_level}/{variable.zarr_variable}"

    if attributes.get("long_name") != expected_long_name:
        raise ZarrMetadataError(f"Variable long_name mismatch: {feature}")

    if feature == "precipitation_1h":
        if attributes.get("cell_methods") != "time: sum":
            raise ZarrMetadataError("Invalid precipitation accumulation metadata")

    source_units = attributes.get("units")

    if source_units not in ALLOWED_SOURCE_UNITS[feature]:
        raise ZarrMetadataError(f"Unsupported source units: {source_units}")

    compressor = metadata["compressor"]

    if not isinstance(compressor, dict) or compressor.get("id") != "blosc":
        raise ZarrMetadataError("Unsupported Zarr compressor")

    try:
        numcodecs.get_codec(compressor)

    except (ValueError, TypeError, KeyError) as exc:
        raise ZarrMetadataError("Invalid compressor configuration") from exc

    # Compare decoded values against the stored representation
    # of the metadata-declared fill value.

    try:
        declared_fill = float(metadata["fill_value"])

        with np.errstate(over="ignore", invalid="ignore"):
            stored_fill = float(np.asarray(declared_fill, dtype=dtype))

    except (ValueError, TypeError, OverflowError) as exc:
        raise ZarrMetadataError("Invalid fill value") from exc

    if not np.isfinite(declared_fill) or not np.isfinite(stored_fill):
        raise ZarrMetadataError("Unsupported nonfinite fill value")

    return ZarrArrayMetadata(
        feature=feature,
        shape=tuple(metadata["shape"]),
        chunks=tuple(metadata["chunks"]),
        dtype=dtype,
        order=order,
        dimension_separator=separator,
        declared_fill_value=declared_fill,
        stored_fill_value=stored_fill,
        compressor=compressor.copy(),
        source_units=source_units,
    )


def load_zarr_array_metadata(
    fs,
    run_at,
    feature: str,
) -> ZarrArrayMetadata:
    """Load and validate array metadata from object storage."""

    array_path = build_zarr_array_path(
        run_at,
        feature,
    )

    metadata = read_zarr_json(
        fs,
        f"{array_path}/.zarray",
    )

    attributes = read_zarr_json(
        fs,
        f"{array_path}/.zattrs",
    )

    return parse_zarr_metadata(
        feature,
        metadata,
        attributes,
    )


def decode_zarr_chunk(
    compressed: bytes,
    metadata: ZarrArrayMetadata,
) -> np.ndarray:
    """Decode a compressed chunk using validated metadata."""

    if not compressed:
        raise ZarrChunkError("Empty compressed chunk")

    try:
        codec = numcodecs.get_codec(metadata.compressor)

        decoded = codec.decode(compressed)

        if memoryview(decoded).nbytes != (metadata.expected_chunk_bytes):
            raise ZarrChunkError("Unexpected decoded chunk size")

        array = np.frombuffer(
            decoded,
            dtype=metadata.dtype,
        )

        return array.reshape(
            metadata.chunks,
            order=metadata.order,
        )

    except ZarrChunkError:
        raise

    except Exception as exc:
        raise ZarrChunkError(f"Chunk decoding failed: {metadata.feature}") from exc


def normalize_zarr_value(
    raw_value,
    metadata: ZarrArrayMetadata,
) -> ZarrValue:
    """Normalize missing values without silently imputing them."""

    raw_value = float(raw_value)

    if raw_value == metadata.stored_fill_value:
        return ZarrValue(
            value=np.nan,
            raw_value=raw_value,
            value_status="fill_value",
        )

    if not np.isfinite(raw_value):
        return ZarrValue(
            value=np.nan,
            raw_value=raw_value,
            value_status="nonfinite",
        )

    return ZarrValue(
        value=raw_value,
        raw_value=raw_value,
        value_status="valid",
    )
