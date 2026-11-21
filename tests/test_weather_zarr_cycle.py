"""Offline integration tests for cycle-level Zarr extraction."""

import json
from dataclasses import replace

import numcodecs
import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.config import (
    RAW_WEATHER_COLUMNS,
    VARIABLE_BY_FEATURE,
)
from flight_delay.weather.zarr_backend import (
    EXPECTED_GRIB_PARAMETERS,
    ZarrChunkError,
    build_zarr_array_path,
)
from flight_delay.weather.zarr_cycle import (
    ZarrCoordinateData,
    ZarrCoordinateError,
    extract_zarr_cycle,
)

RUN = pd.Timestamp("2024-07-14T06:00:00Z")


def synthetic_coordinates(fs, run_at, feature):
    del fs, feature

    run = pd.Timestamp(run_at).tz_convert("UTC")

    return ZarrCoordinateData(
        reference_time=run.tz_localize(None).to_datetime64(),
        forecast_periods=np.arange(1, 49).astype("timedelta64[h]"),
        valid_times=pd.date_range(
            run + pd.Timedelta(hours=1),
            periods=48,
            freq="h",
        )
        .tz_localize(None)
        .to_numpy(),
        x_coords=(-2697520.1425219304 + np.arange(1799) * 3000.0),
        y_coords=(-1587306.1525566636 + np.arange(1059) * 3000.0),
    )


class FakeZarrStore:
    """Synthetic object store with genuine Blosc chunks."""

    def __init__(self, dtype="<f4", missing_chunk=False):
        self.requests = []
        self.dtype = np.dtype(dtype)
        self.fill = -10000.0 if dtype == "<f2" else -9999.0
        self.missing_chunk = missing_chunk

        self.compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 9,
            "shuffle": 1,
            "blocksize": 0,
        }

        codec = numcodecs.get_codec(self.compressor)

        self.payloads = {}

        for feature in RAW_WEATHER_COLUMNS:
            initial = (
                self.fill
                if feature == "cloud_ceiling_raw"
                else 280.25
                if feature == "temperature_2m"
                else 3.5
            )

            array = np.full(
                (48, 150, 150),
                initial,
                dtype=self.dtype,
            )

            if feature == "temperature_2m":
                array[29, 77, 103] = 301.125

            self.payloads[feature] = codec.encode(array.tobytes())

    def cat_file(self, path):
        self.requests.append(path)

        for feature in RAW_WEATHER_COLUMNS:
            prefix = build_zarr_array_path(RUN, feature)

            if not path.startswith(prefix + "/"):
                continue

            suffix = path[len(prefix) + 1 :]

            if suffix == ".zarray":
                return json.dumps(
                    {
                        "zarr_format": 2,
                        "shape": [48, 1059, 1799],
                        "chunks": [48, 150, 150],
                        "dtype": self.dtype.str,
                        "fill_value": self.fill,
                        "filters": None,
                        "order": "C",
                        "compressor": self.compressor,
                    }
                ).encode()

            if suffix == ".zattrs":
                variable = VARIABLE_BY_FEATURE[feature]

                attributes = {
                    "_ARRAY_DIMENSIONS": [
                        "time",
                        "projection_y_coordinate",
                        "projection_x_coordinate",
                    ],
                    "GRIB_PARAM": list(EXPECTED_GRIB_PARAMETERS[feature]),
                    "long_name": (f"{variable.zarr_level}/{variable.zarr_variable}"),
                    "units": (
                        None
                        if feature
                        in {
                            "visibility",
                            "precipitation_1h",
                            "composite_reflectivity",
                        }
                        else "m"
                        if feature == "cloud_ceiling_raw"
                        else "m s-1"
                        if feature
                        in {
                            "u_wind_10m",
                            "v_wind_10m",
                            "wind_gust",
                        }
                        else "J kg-1"
                        if feature == "surface_cape"
                        else "K"
                    ),
                }

                if feature == "precipitation_1h":
                    attributes["cell_methods"] = "time: sum"

                return json.dumps(attributes).encode()

            if self.missing_chunk and feature == "temperature_2m" and suffix == "0.2.8":
                raise FileNotFoundError(path)

            return self.payloads[feature]

        raise FileNotFoundError(path)


@pytest.mark.parametrize("dtype", ["<f2", "<f4"])
def test_single_hour_cycle(dtype):
    fs = FakeZarrStore(dtype=dtype)

    result = extract_zarr_cycle(
        fs,
        RUN,
        [30],
        coordinate_loader=synthetic_coordinates,
    )

    assert len(result.weather) == 54
    assert len(result.chunk_audit) == 54

    assert result.weather["backend"].eq("zarr").all()
    assert result.weather["storage_dtype"].eq(str(np.dtype(dtype))).all()

    core = result.weather.loc[result.weather["feature"] != "cloud_ceiling_raw"]

    assert core["value"].notna().all()

    ceiling = result.weather.loc[result.weather["feature"] == "cloud_ceiling_raw"]

    assert ceiling["value"].isna().all()
    assert ceiling["value_status"].eq("fill_value").all()


def test_multi_lead_chunk_reuse():
    fs = FakeZarrStore()

    result = extract_zarr_cycle(
        fs,
        RUN,
        list(range(27, 33)),
        coordinate_loader=synthetic_coordinates,
    )

    assert len(result.weather) == 324

    # Six airports × nine variables.
    assert len(result.chunk_audit) == 54

    chunk_requests = [
        path for path in fs.requests if not path.endswith((".zarray", ".zattrs"))
    ]

    assert len(chunk_requests) == 54
    assert len(set(chunk_requests)) == 54

    assert result.weather.groupby("forecast_hour").size().eq(54).all()


def test_forecast_time_indexing():
    fs = FakeZarrStore()

    result = extract_zarr_cycle(
        fs,
        RUN,
        [29, 30],
        coordinate_loader=synthetic_coordinates,
    )

    temperature = result.weather.loc[
        (result.weather["feature"] == "temperature_2m")
        & (result.weather["Origin"] == "ATL")
    ].set_index("forecast_hour")

    assert temperature.loc[29, "value"] == 280.25
    assert temperature.loc[30, "value"] == 301.125


def test_missing_chunk_rejected():
    fs = FakeZarrStore(missing_chunk=True)

    with pytest.raises(
        ZarrChunkError,
        match="Unable to read Zarr chunk",
    ):
        extract_zarr_cycle(
            fs,
            RUN,
            [30],
            coordinate_loader=synthetic_coordinates,
        )


def test_forecast_valid_time_mismatch():
    fs = FakeZarrStore()

    def wrong_coordinates(fs, run_at, feature):
        coordinates = synthetic_coordinates(fs, run_at, feature)

        return replace(
            coordinates,
            valid_times=(coordinates.valid_times + np.timedelta64(1, "h")),
        )

    with pytest.raises(
        ZarrCoordinateError,
        match="Forecast valid time mismatch",
    ):
        extract_zarr_cycle(
            fs,
            RUN,
            [30],
            coordinate_loader=wrong_coordinates,
        )


def test_shifted_spatial_grid_rejected():
    fs = FakeZarrStore()

    def wrong_coordinates(fs, run_at, feature):
        coordinates = synthetic_coordinates(fs, run_at, feature)

        return replace(
            coordinates,
            x_coords=coordinates.x_coords + 3000,
        )

    with pytest.raises(
        ZarrCoordinateError,
        match="HRRR spatial coordinates mismatch",
    ):
        extract_zarr_cycle(
            fs,
            RUN,
            [30],
            coordinate_loader=wrong_coordinates,
        )


@pytest.mark.parametrize(
    "leads",
    [[], [30, 30], [26], [33], [30.5]],
)
def test_invalid_leads_rejected(leads):
    fs = FakeZarrStore()

    with pytest.raises(
        ZarrCoordinateError,
        match="Invalid or duplicate forecast leads",
    ):
        extract_zarr_cycle(
            fs,
            RUN,
            leads,
            coordinate_loader=synthetic_coordinates,
        )
