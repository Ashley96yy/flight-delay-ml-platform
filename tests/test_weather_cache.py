"""Offline tests for persistent weather caching."""

import json

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.cache import (
    WeatherCache,
    WeatherCacheIntegrityError,
    WeatherCacheValidationError,
    cache_fingerprint,
)
from flight_delay.weather.config import (
    AIRPORTS,
    WEATHER_VARIABLES,
)

RUN = pd.Timestamp("2024-07-14T06:00:00Z")


@pytest.fixture
def weather():
    records = []

    for airport in AIRPORTS:
        for variable in WEATHER_VARIABLES:
            missing = not variable.required

            records.append(
                {
                    "Origin": airport.code,
                    "feature": variable.feature,
                    "forecast_run_at_utc": RUN,
                    "forecast_valid_at_utc": (RUN + pd.Timedelta(hours=30)),
                    "forecast_hour": 30,
                    "assumed_available_at_utc": (RUN + pd.Timedelta(hours=3)),
                    "value": np.nan if missing else 3.5,
                    "raw_value": -9999.0 if missing else 3.5,
                    "value_status": ("fill_value" if missing else "valid"),
                    "grid_latitude": airport.latitude,
                    "grid_longitude": airport.longitude,
                    "backend": "zarr",
                    "storage_dtype": "float32",
                    "units": variable.grib_units,
                    "source_units": variable.grib_units,
                    "fallback_reason": None,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def cache(tmp_path):
    return WeatherCache(tmp_path / "weather_cache")


def test_cache_miss(cache):
    assert cache.read_hour(RUN, 30) is None


def test_cache_fingerprint_deterministic():
    assert cache_fingerprint() == cache_fingerprint()
    assert len(cache_fingerprint()) == 16


def test_zarr_cache_roundtrip(cache, weather):
    cache.write_hour(RUN, 30, weather)

    restored = cache.read_hour(RUN, 30)

    assert restored is not None
    assert len(restored) == 54
    assert restored["backend"].eq("zarr").all()

    expected = weather.sort_values(["feature", "Origin"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(restored, expected)


def test_grib_cache_roundtrip(cache, weather):
    weather["backend"] = "grib"
    weather["storage_dtype"] = None
    weather["fallback_reason"] = "zarr_chunk_error"

    restored_path = cache.write_hour(RUN, 30, weather)

    restored = cache.read_hour(RUN, 30)

    assert restored_path.is_file()
    assert restored["backend"].eq("grib").all()
    assert restored["fallback_reason"].eq("zarr_chunk_error").all()


def test_invalid_hour_rejected(cache, weather):
    with pytest.raises(
        WeatherCacheValidationError,
        match="Unsupported forecast lead",
    ):
        cache.write_hour(RUN, 33, weather)


def test_naive_run_rejected(cache):
    with pytest.raises(
        WeatherCacheValidationError,
        match="Timezone-aware",
    ):
        cache.read_hour("2024-07-14 06:00:00", 30)


def test_wrong_forecast_identity_rejected(cache, weather):
    with pytest.raises(
        WeatherCacheValidationError,
        match="Cached forecast lead mismatch",
    ):
        cache.write_hour(RUN, 29, weather)


def test_missing_core_value_rejected(cache, weather):
    weather.loc[0, "value"] = np.nan
    weather.loc[0, "value_status"] = "source_missing"

    with pytest.raises(
        WeatherCacheValidationError,
        match="violates the data contract",
    ):
        cache.write_hour(RUN, 30, weather)


def test_mixed_backend_rejected(cache, weather):
    weather.loc[0, "backend"] = "grib"

    with pytest.raises(
        WeatherCacheValidationError,
        match="Mixed backends",
    ):
        cache.write_hour(RUN, 30, weather)


def test_checksum_corruption_detected(cache, weather):
    path = cache.write_hour(RUN, 30, weather)

    with path.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(
        WeatherCacheIntegrityError,
        match="checksum mismatch",
    ):
        cache.read_hour(RUN, 30)


def test_manifest_identity_corruption(cache, weather):
    cache.write_hour(RUN, 30, weather)

    manifest_path = cache.entry_directory(RUN, 30) / "manifest.json"

    manifest = json.loads(manifest_path.read_text())

    manifest["fingerprint"] = "invalid"

    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        WeatherCacheIntegrityError,
        match="identity mismatch",
    ):
        cache.read_hour(RUN, 30)


def test_uncommitted_payload_is_cache_miss(cache):
    directory = cache.entry_directory(RUN, 30)
    directory.mkdir(parents=True)

    (directory / ".interrupted.parquet.tmp").write_bytes(b"incomplete")

    assert cache.read_hour(RUN, 30) is None


def test_committed_missing_payload_rejected(cache, weather):
    path = cache.write_hour(RUN, 30, weather)
    path.unlink()

    with pytest.raises(
        WeatherCacheIntegrityError,
        match="payload is missing",
    ):
        cache.read_hour(RUN, 30)


def test_new_commit_replaces_manifest(cache, weather):
    first_path = cache.write_hour(RUN, 30, weather)

    modified = weather.copy(deep=True)
    modified.loc[0, "value"] = 4.5

    second_path = cache.write_hour(RUN, 30, modified)

    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path != second_path

    restored = cache.read_hour(RUN, 30)

    assert (
        restored.loc[
            (restored["feature"] == modified.loc[0, "feature"])
            & (restored["Origin"] == modified.loc[0, "Origin"]),
            "value",
        ].iloc[0]
        == 4.5
    )


def test_read_cycle_hits_and_misses(cache, weather):
    cache.write_hour(RUN, 30, weather)

    result = cache.read_cycle(RUN, [29, 30, 31])

    assert result[29] is None
    assert result[30] is not None
    assert result[31] is None
