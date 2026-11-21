"""Offline tests for HRRR Zarr metadata and decoding."""

import json

import numcodecs
import numpy as np
import pytest

from flight_delay.weather.zarr_backend import (
    ZarrChunkError,
    ZarrMetadataError,
    build_zarr_array_path,
    build_zarr_group_path,
    decode_zarr_chunk,
    load_zarr_array_metadata,
    normalize_zarr_value,
    parse_zarr_metadata,
)

RUN_AT = "2024-07-14T06:00:00Z"


@pytest.fixture
def array_metadata():

    return {
        "zarr_format": 2,
        "shape": [48, 1059, 1799],
        "chunks": [48, 150, 150],
        "dtype": "<f4",
        "fill_value": -9999.0,
        "filters": None,
        "order": "C",
        "compressor": {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 9,
            "shuffle": 1,
            "blocksize": 0,
        },
    }


@pytest.fixture
def temperature_attributes():

    return {
        "_ARRAY_DIMENSIONS": [
            "time",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ],
        "GRIB_PARAM": [2, 0, 0, 0],
        "long_name": "2m_above_ground/TMP",
        "units": "K",
    }


@pytest.fixture
def validated_metadata(
    array_metadata,
    temperature_attributes,
):

    return parse_zarr_metadata(
        "temperature_2m",
        array_metadata,
        temperature_attributes,
    )


def test_zarr_group_path():

    path = build_zarr_group_path(
        RUN_AT,
        "temperature_2m",
    )

    assert path == ("hrrrzarr/sfc/20240714/20240714_06z_fcst.zarr/2m_above_ground/TMP")


def test_zarr_array_path():

    path = build_zarr_array_path(
        RUN_AT,
        "temperature_2m",
    )

    assert path.endswith("2m_above_ground/TMP/2m_above_ground/TMP")


def test_precipitation_array_path():

    path = build_zarr_array_path(
        RUN_AT,
        "precipitation_1h",
    )

    assert path.endswith("surface/APCP_1hr_acc_fcst/surface/APCP_1hr_acc_fcst")


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-07-14T06:00:00",
        "2024-07-14T07:00:00Z",
        "2024-07-14T06:30:00Z",
    ],
)
def test_invalid_initialization(timestamp):

    with pytest.raises(ZarrMetadataError):
        build_zarr_group_path(
            timestamp,
            "temperature_2m",
        )


def test_unknown_feature():

    with pytest.raises(
        ZarrMetadataError,
        match="Unsupported weather feature",
    ):
        build_zarr_group_path(
            RUN_AT,
            "unknown_feature",
        )


@pytest.mark.parametrize(
    "dtype,fill_value,expected_bytes",
    [
        ("<f2", -10000.0, 2160000),
        ("<f4", -9999.0, 4320000),
    ],
)
def test_supported_storage_metadata(
    array_metadata,
    temperature_attributes,
    dtype,
    fill_value,
    expected_bytes,
):

    array_metadata["dtype"] = dtype

    array_metadata["fill_value"] = fill_value

    result = parse_zarr_metadata(
        "temperature_2m",
        array_metadata,
        temperature_attributes,
    )

    assert result.dtype == np.dtype(dtype)

    assert result.stored_fill_value == fill_value

    assert result.expected_chunk_bytes == expected_bytes


@pytest.mark.parametrize(
    "key,value",
    [
        ("dtype", "<i4"),
        ("dtype", "<f8"),
        ("zarr_format", 3),
        ("shape", [24, 1059, 1799]),
        ("chunks", [24, 150, 150]),
        ("filters", [{"id": "delta"}]),
        ("order", "X"),
        ("dimension_separator", "-"),
        ("compressor", None),
        ("fill_value", None),
        ("fill_value", float("inf")),
    ],
)
def test_unsupported_metadata_rejected(
    array_metadata,
    temperature_attributes,
    key,
    value,
):

    array_metadata[key] = value

    with pytest.raises(ZarrMetadataError):
        parse_zarr_metadata(
            "temperature_2m",
            array_metadata,
            temperature_attributes,
        )


def test_missing_metadata_key(
    array_metadata,
    temperature_attributes,
):

    array_metadata.pop("dtype")

    with pytest.raises(
        ZarrMetadataError,
        match="Missing metadata keys",
    ):
        parse_zarr_metadata(
            "temperature_2m",
            array_metadata,
            temperature_attributes,
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("GRIB_PARAM", [2, 0, 1, 8]),
        ("long_name", "surface/TMP"),
        ("_ARRAY_DIMENSIONS", ["time", "x", "y"]),
        ("units", "C"),
        ("scale_factor", 0.1),
        ("add_offset", 273.15),
    ],
)
def test_incompatible_variable_attributes(
    array_metadata,
    temperature_attributes,
    key,
    value,
):

    temperature_attributes[key] = value

    with pytest.raises(ZarrMetadataError):
        parse_zarr_metadata(
            "temperature_2m",
            array_metadata,
            temperature_attributes,
        )


def test_missing_visibility_units(
    array_metadata,
    temperature_attributes,
):

    temperature_attributes.update(
        {
            "GRIB_PARAM": [2, 0, 19, 0],
            "long_name": "surface/VIS",
            "units": None,
        }
    )

    result = parse_zarr_metadata(
        "visibility",
        array_metadata,
        temperature_attributes,
    )

    assert result.source_units is None


def test_hourly_precipitation_metadata(
    array_metadata,
    temperature_attributes,
):

    temperature_attributes.update(
        {
            "GRIB_PARAM": [2, 0, 1, 8],
            "long_name": "surface/APCP_1hr_acc_fcst",
            "cell_methods": "time: sum",
            "units": None,
        }
    )

    result = parse_zarr_metadata(
        "precipitation_1h",
        array_metadata,
        temperature_attributes,
    )

    assert result.feature == "precipitation_1h"

    assert result.source_units is None


def test_invalid_precipitation_accumulation(
    array_metadata,
    temperature_attributes,
):

    temperature_attributes.update(
        {
            "GRIB_PARAM": [2, 0, 1, 8],
            "long_name": "surface/APCP_1hr_acc_fcst",
            "cell_methods": "time: mean",
            "units": None,
        }
    )

    with pytest.raises(
        ZarrMetadataError,
        match="precipitation accumulation",
    ):
        parse_zarr_metadata(
            "precipitation_1h",
            array_metadata,
            temperature_attributes,
        )


def test_fill_value_cast_to_storage_dtype(
    array_metadata,
    temperature_attributes,
):

    array_metadata["dtype"] = "<f2"

    array_metadata["fill_value"] = -9999.0

    result = parse_zarr_metadata(
        "temperature_2m",
        array_metadata,
        temperature_attributes,
    )

    assert result.declared_fill_value == -9999.0

    assert result.stored_fill_value == -10000.0


@pytest.mark.parametrize(
    "dtype,order",
    [
        ("<f2", "C"),
        ("<f4", "C"),
        ("<f4", "F"),
    ],
)
def test_chunk_roundtrip(
    array_metadata,
    temperature_attributes,
    dtype,
    order,
):

    array_metadata["dtype"] = dtype

    array_metadata["order"] = order

    metadata = parse_zarr_metadata(
        "temperature_2m",
        array_metadata,
        temperature_attributes,
    )

    original = np.full(
        metadata.chunks,
        280.25,
        dtype=metadata.dtype,
    )

    original[29, 77, 103] = 301.125

    codec = numcodecs.get_codec(metadata.compressor)

    compressed = codec.encode(original.tobytes(order=order))

    restored = decode_zarr_chunk(
        compressed,
        metadata,
    )

    assert restored.shape == metadata.chunks

    assert restored.dtype == metadata.dtype

    np.testing.assert_array_equal(
        original,
        restored,
    )


def test_empty_chunk_rejected(validated_metadata):

    with pytest.raises(
        ZarrChunkError,
        match="Empty compressed chunk",
    ):
        decode_zarr_chunk(
            b"",
            validated_metadata,
        )


def test_corrupt_chunk_rejected(validated_metadata):

    with pytest.raises(ZarrChunkError):
        decode_zarr_chunk(
            b"invalid compressed weather data",
            validated_metadata,
        )


def test_wrong_decoded_size(validated_metadata):

    original = np.zeros(
        (2, 2, 2),
        dtype=np.float32,
    )

    codec = numcodecs.get_codec(validated_metadata.compressor)

    compressed = codec.encode(original.tobytes())

    with pytest.raises(
        ZarrChunkError,
        match="Unexpected decoded chunk size",
    ):
        decode_zarr_chunk(
            compressed,
            validated_metadata,
        )


@pytest.mark.parametrize(
    "raw_value,expected_status",
    [
        (300.0, "valid"),
        (0.0, "valid"),
        (-10.0, "valid"),
        (-9999.0, "fill_value"),
        (float("nan"), "nonfinite"),
        (float("inf"), "nonfinite"),
        (float("-inf"), "nonfinite"),
    ],
)
def test_normalize_zarr_value(
    validated_metadata,
    raw_value,
    expected_status,
):

    result = normalize_zarr_value(
        raw_value,
        validated_metadata,
    )

    assert result.value_status == expected_status

    if expected_status == "valid":
        assert result.value == raw_value

    else:
        assert np.isnan(result.value)


def test_float16_fill_value(
    array_metadata,
    temperature_attributes,
):

    array_metadata["dtype"] = "<f2"

    array_metadata["fill_value"] = -10000.0

    metadata = parse_zarr_metadata(
        "temperature_2m",
        array_metadata,
        temperature_attributes,
    )

    result = normalize_zarr_value(
        -10000.0,
        metadata,
    )

    assert np.isnan(result.value)

    assert result.raw_value == -10000.0

    assert result.value_status == "fill_value"


class FakeFileSystem:
    """In-memory object store without network requests."""

    def __init__(self, objects):

        self.objects = objects
        self.requests = []

    def cat_file(self, path):

        self.requests.append(path)

        if path not in self.objects:
            raise FileNotFoundError(path)

        return self.objects[path]


def test_load_metadata_from_filesystem(
    array_metadata,
    temperature_attributes,
):

    path = build_zarr_array_path(
        RUN_AT,
        "temperature_2m",
    )

    fs = FakeFileSystem(
        {
            f"{path}/.zarray": json.dumps(array_metadata).encode(),
            f"{path}/.zattrs": json.dumps(temperature_attributes).encode(),
        }
    )

    metadata = load_zarr_array_metadata(
        fs,
        RUN_AT,
        "temperature_2m",
    )

    assert metadata.feature == "temperature_2m"

    assert fs.requests == [
        f"{path}/.zarray",
        f"{path}/.zattrs",
    ]


def test_missing_metadata_object():

    fs = FakeFileSystem({})

    with pytest.raises(
        ZarrMetadataError,
        match="Unable to read Zarr metadata",
    ):
        load_zarr_array_metadata(
            fs,
            RUN_AT,
            "temperature_2m",
        )


def test_invalid_json_metadata():

    path = build_zarr_array_path(
        RUN_AT,
        "temperature_2m",
    )

    fs = FakeFileSystem(
        {
            f"{path}/.zarray": b"not valid JSON",
        }
    )

    with pytest.raises(ZarrMetadataError):
        load_zarr_array_metadata(
            fs,
            RUN_AT,
            "temperature_2m",
        )
