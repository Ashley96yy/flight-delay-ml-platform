"""Offline tests for GRIB hourly weather extraction."""

import numpy as np
import pytest

from flight_delay.weather.config import (
    AIRPORTS,
    VARIABLE_BY_FEATURE,
)
from flight_delay.weather.contracts import WeatherContractError
from flight_delay.weather.grib_backend import (
    GRIBDownloadError,
    GRIBInventoryError,
    build_grib_reference,
    build_weather_patterns,
)
from flight_delay.weather.grib_decoder import (
    GRIBDecodeError,
    reference_airport_grid,
)
from flight_delay.weather.grib_hour import extract_grib_hour

RUN = "2024-07-14T06:00:00Z"
REFERENCE = build_grib_reference(RUN, 30)


class FakeResponse:
    def __init__(
        self,
        payload,
        status=200,
        content_range=None,
    ):
        self.payload = payload
        self.status = status
        self.headers = {}

        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class FakeGRIBTransport:
    """Serve synthetic inventory and byte ranges."""

    def __init__(self):
        patterns = build_weather_patterns(30)

        self.features = list(patterns)

        lines = [
            f"{i}:{(i - 1) * 100}:d=2024071406{pattern}"
            for i, pattern in enumerate(patterns.values(), start=1)
        ]

        lines.append("10:900:d=2024071406:DUMMY:surface:30 hour fcst:")

        self.inventory = "\n".join(lines).encode()

        self.payloads = [
            b"GRIB" + feature.encode().ljust(92, b"_") + b"7777"
            for feature in self.features
        ]

        self.index_requests = 0
        self.range_requests = 0
        self.bad_range_status = False

    def index_opener(self, url, timeout):
        assert url == REFERENCE.index_url
        assert timeout == 30

        self.index_requests += 1
        return FakeResponse(self.inventory)

    def range_opener(self, request, timeout):
        assert timeout == 60

        self.range_requests += 1

        byte_range = request.get_header("Range")
        start, end = map(
            int,
            byte_range.removeprefix("bytes=").split("-"),
        )

        assert end - start + 1 == 100

        feature_index = start // 100

        return FakeResponse(
            self.payloads[feature_index],
            status=200 if self.bad_range_status else 206,
            content_range=f"bytes {start}-{end}/1000",
        )


class FakeECCodes:
    """Emulate ecCodes metadata, grid and bitmap behavior."""

    def __init__(self):
        self.released = []
        self.bad_units = False
        self.bad_valid_time = False
        self.bad_precipitation_interval = False
        self.bad_grid = False
        self.missing_core = False

    def codes_new_from_message(self, payload):
        return payload[4:-4].rstrip(b"_").decode()

    def codes_get(self, gid, key):
        variable = VARIABLE_BY_FEATURE[gid]

        values = {
            "dataDate": 20240714,
            "dataTime": 600,
            "validityDate": 20240715,
            "validityTime": (1300 if self.bad_valid_time else 1200),
            "endStep": 30,
            "startStep": (
                28
                if self.bad_precipitation_interval and gid == "precipitation_1h"
                else 29
                if gid == "precipitation_1h"
                else 0
            ),
            "shortName": variable.grib_short_name,
            "units": (
                "C"
                if self.bad_units and gid == "temperature_2m"
                else variable.grib_units
            ),
            "typeOfLevel": variable.grib_type_of_level,
            "stepType": variable.grib_step_type,
            "bitmapPresent": int(
                gid == "cloud_ceiling_raw"
                or (self.missing_core and gid == "temperature_2m")
            ),
        }

        return values[key]

    def codes_get_array(self, gid, key):
        assert key == "bitmap"

        if gid == "cloud_ceiling_raw":
            return np.array([1, 0, 0, 0, 0, 0])

        if self.missing_core and gid == "temperature_2m":
            return np.zeros(6, dtype=int)

        return np.ones(6, dtype=int)

    def codes_grib_find_nearest(self, gid, latitude, longitude):
        airport_index = next(
            i
            for i, airport in enumerate(AIRPORTS)
            if abs(airport.latitude - latitude) < 1e-8
            and abs(airport.longitude - longitude) < 1e-8
        )

        airport = AIRPORTS[airport_index]

        grid = reference_airport_grid().loc[airport.code]

        value = {
            "temperature_2m": 301.125,
            "precipitation_1h": 0.0,
            "composite_reflectivity": -10.0,
            "cloud_ceiling_raw": 2500.0,
        }.get(gid, 3.5)

        return [
            {
                "lat": (
                    float(grid["grid_latitude"])
                    + (0.01 if self.bad_grid and airport.code == "ATL" else 0.0)
                ),
                "lon": float(grid["grid_longitude"]),
                "value": value,
                "index": airport_index,
            }
        ]

    def codes_release(self, gid):
        self.released.append(gid)


@pytest.fixture
def fake_system():
    return FakeGRIBTransport(), FakeECCodes()


def run_fake_extraction(fake_system, **kwargs):
    transport, decoder = fake_system

    return extract_grib_hour(
        REFERENCE,
        index_opener=transport.index_opener,
        range_opener=transport.range_opener,
        eccodes_module=decoder,
        **kwargs,
    )


def test_complete_hourly_extraction(fake_system):
    result = run_fake_extraction(fake_system)

    assert len(result.weather) == 54
    assert len(result.request_audit) == 9

    assert result.weather["backend"].eq("grib").all()
    assert result.weather["storage_dtype"].isna().all()

    transport, decoder = fake_system

    assert transport.index_requests == 1
    assert transport.range_requests == 9
    assert len(decoder.released) == 9


def test_expected_missing_cloud_ceiling(fake_system):
    result = run_fake_extraction(fake_system)

    ceiling = result.weather.loc[result.weather["feature"] == "cloud_ceiling_raw"]

    assert ceiling["value"].isna().sum() == 5
    assert ceiling["value_status"].eq("source_missing").sum() == 5


def test_valid_core_weather_values(fake_system):
    result = run_fake_extraction(fake_system)

    core = result.weather.loc[result.weather["feature"] != "cloud_ceiling_raw"]

    assert core["value"].notna().all()


def test_negative_reflectivity_preserved(fake_system):
    result = run_fake_extraction(fake_system)

    reflectivity = result.weather.loc[
        result.weather["feature"] == "composite_reflectivity"
    ]

    assert reflectivity["value"].eq(-10.0).all()


def test_grib_units_preserved(fake_system):
    result = run_fake_extraction(fake_system)

    winds = result.weather.loc[result.weather["feature"] == "wind_gust"]

    assert winds["units"].eq("m s**-1").all()
    assert winds["source_units"].eq("m s**-1").all()


def test_fallback_reason_recorded(fake_system):
    result = run_fake_extraction(
        fake_system,
        fallback_reason="missing_zarr_chunk",
    )

    assert result.weather["fallback_reason"].eq("missing_zarr_chunk").all()


def test_wrong_units_rejected(fake_system):
    _, decoder = fake_system
    decoder.bad_units = True

    with pytest.raises(
        GRIBDecodeError,
        match="metadata mismatch",
    ):
        run_fake_extraction(fake_system)

    assert decoder.released == ["temperature_2m"]


def test_wrong_forecast_time_rejected(fake_system):
    _, decoder = fake_system
    decoder.bad_valid_time = True

    with pytest.raises(
        GRIBDecodeError,
        match="Forecast valid time mismatch",
    ):
        run_fake_extraction(fake_system)


def test_wrong_precipitation_interval(fake_system):
    _, decoder = fake_system
    decoder.bad_precipitation_interval = True

    with pytest.raises(
        GRIBDecodeError,
        match="precipitation accumulation interval",
    ):
        run_fake_extraction(fake_system)


def test_wrong_spatial_grid_rejected(fake_system):
    _, decoder = fake_system
    decoder.bad_grid = True

    with pytest.raises(
        GRIBDecodeError,
        match="GRIB/Zarr grid mismatch",
    ):
        run_fake_extraction(fake_system)


def test_missing_core_values_rejected(fake_system):
    _, decoder = fake_system
    decoder.missing_core = True

    with pytest.raises(
        WeatherContractError,
        match="Missing required weather values",
    ):
        run_fake_extraction(fake_system)


def test_index_fetch_failure(fake_system):
    transport, decoder = fake_system

    def failing_index_opener(url, timeout):
        raise OSError("Simulated index network failure")

    with pytest.raises(
        GRIBInventoryError,
        match="Failed to fetch GRIB inventory",
    ):
        extract_grib_hour(
            REFERENCE,
            index_opener=failing_index_opener,
            range_opener=transport.range_opener,
            eccodes_module=decoder,
        )


def test_range_response_failure(fake_system):
    transport, _ = fake_system
    transport.bad_range_status = True

    with pytest.raises(
        GRIBDownloadError,
        match="HTTP 206",
    ):
        run_fake_extraction(fake_system)
