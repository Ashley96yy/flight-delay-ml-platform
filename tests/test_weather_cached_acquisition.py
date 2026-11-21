"""Offline tests for cache-aware weather acquisition."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.cache import (
    WeatherCache,
    WeatherCacheIntegrityError,
    WeatherCacheValidationError,
)
from flight_delay.weather.cached_acquisition import (
    acquire_cached_weather_cycle,
)
from flight_delay.weather.config import AIRPORTS, WEATHER_VARIABLES
from flight_delay.weather.contracts import WeatherContractError
from flight_delay.weather.fallback import (
    WeatherAcquisitionError,
    WeatherCycleResult,
)

RUN = pd.Timestamp("2024-07-14T06:00:00Z")


@pytest.fixture
def cache(tmp_path):
    return WeatherCache(tmp_path / "cache")


def make_weather(run, leads, backend="zarr", reason=None):
    records = []

    for lead in leads:
        for airport in AIRPORTS:
            for variable in WEATHER_VARIABLES:
                missing = not variable.required

                records.append(
                    {
                        "Origin": airport.code,
                        "feature": variable.feature,
                        "forecast_run_at_utc": run,
                        "forecast_valid_at_utc": (run + pd.Timedelta(hours=lead)),
                        "forecast_hour": lead,
                        "assumed_available_at_utc": (run + pd.Timedelta(hours=3)),
                        "value": np.nan if missing else 3.5,
                        "raw_value": -9999.0 if missing else 3.5,
                        "value_status": (
                            ("source_missing" if backend == "grib" else "fill_value")
                            if missing
                            else "valid"
                        ),
                        "grid_latitude": airport.latitude,
                        "grid_longitude": airport.longitude,
                        "backend": backend,
                        "storage_dtype": ("float32" if backend == "zarr" else None),
                        "units": variable.grib_units,
                        "source_units": variable.grib_units,
                        "fallback_reason": reason,
                    }
                )

    return pd.DataFrame(records)


def make_acquirer(calls, backend="zarr", reason=None):
    def acquire(fs, run, leads):
        calls.append(tuple(leads))

        return WeatherCycleResult(
            weather=make_weather(run, leads, backend, reason),
            attempt_audit=pd.DataFrame(
                [
                    {
                        "backend": backend,
                        "status": "success",
                    }
                ]
            ),
            backend_audit=pd.DataFrame(
                [
                    {
                        "backend": backend,
                        "feature": "temperature_2m",
                    }
                ]
            ),
            backend=backend,
            fallback_reason=reason,
        )

    return acquire


def test_first_run_fetches_missing_hours(cache):
    calls = []

    result = acquire_cached_weather_cycle(
        cache,
        None,
        RUN,
        [27, 30],
        acquirer=make_acquirer(calls),
    )

    assert calls == [(27, 30)]
    assert len(result.weather) == 108

    assert result.cache_audit["cache_status"].eq("fetched").all()

    assert cache.read_hour(RUN, 27) is not None
    assert cache.read_hour(RUN, 30) is not None


def test_all_cached_never_calls_backend(cache):
    cache.write_hour(RUN, 27, make_weather(RUN, [27]))
    cache.write_hour(RUN, 30, make_weather(RUN, [30]))

    def forbidden_acquirer(fs, run, leads):
        raise AssertionError("Network acquisition not expected")

    result = acquire_cached_weather_cycle(
        cache,
        None,
        RUN,
        [27, 30],
        acquirer=forbidden_acquirer,
    )

    assert len(result.weather) == 108
    assert result.cache_audit["cache_status"].eq("hit").all()
    assert result.attempt_audit.empty


def test_partial_cache_only_fetches_missing(cache):
    cache.write_hour(RUN, 30, make_weather(RUN, [30]))

    calls = []

    result = acquire_cached_weather_cycle(
        cache,
        None,
        RUN,
        [27, 30],
        acquirer=make_acquirer(calls),
    )

    assert calls == [(27,)]
    assert result.cache_audit["cache_status"].tolist() == [
        "fetched",
        "hit",
    ]


def test_interrupted_write_resumes(tmp_path):
    class FailOnceCache(WeatherCache):
        fail_once = True

        def write_hour(self, run_at, lead, weather):
            if lead == 30 and self.fail_once:
                self.fail_once = False
                raise OSError("Simulated interruption")

            return super().write_hour(run_at, lead, weather)

    cache = FailOnceCache(tmp_path / "cache")
    calls = []

    with pytest.raises(
        OSError,
        match="Simulated interruption",
    ):
        acquire_cached_weather_cycle(
            cache,
            None,
            RUN,
            [27, 30],
            acquirer=make_acquirer(calls),
        )

    assert cache.read_hour(RUN, 27) is not None
    assert cache.read_hour(RUN, 30) is None

    result = acquire_cached_weather_cycle(
        cache,
        None,
        RUN,
        [27, 30],
        acquirer=make_acquirer(calls),
    )

    assert calls == [(27, 30), (30,)]
    assert len(result.weather) == 108
    assert result.cache_audit["cache_status"].tolist() == [
        "hit",
        "fetched",
    ]


def test_corrupt_cache_does_not_silently_refetch(cache):
    path = cache.write_hour(RUN, 30, make_weather(RUN, [30]))

    with path.open("ab") as handle:
        handle.write(b"corruption")

    calls = []

    with pytest.raises(
        WeatherCacheIntegrityError,
        match="checksum mismatch",
    ):
        acquire_cached_weather_cycle(
            cache,
            None,
            RUN,
            [30],
            acquirer=make_acquirer(calls),
        )

    assert calls == []


def test_acquisition_failure_preserves_cached_hour(cache):
    cache.write_hour(RUN, 27, make_weather(RUN, [27]))

    def failing_acquirer(fs, run, leads):
        raise WeatherAcquisitionError("Both weather backends failed")

    with pytest.raises(WeatherAcquisitionError):
        acquire_cached_weather_cycle(
            cache,
            None,
            RUN,
            [27, 30],
            acquirer=failing_acquirer,
        )

    assert cache.read_hour(RUN, 27) is not None
    assert cache.read_hour(RUN, 30) is None


def test_mixed_backend_provenance(cache):
    cache.write_hour(
        RUN,
        27,
        make_weather(
            RUN,
            [27],
            backend="grib",
            reason="zarr_chunk_error",
        ),
    )

    result = acquire_cached_weather_cycle(
        cache,
        None,
        RUN,
        [27, 30],
        acquirer=make_acquirer([]),
    )

    assert result.cache_audit["backend"].tolist() == [
        "grib",
        "zarr",
    ]

    assert (
        result.weather.loc[
            result.weather["forecast_hour"].eq(27),
            "backend",
        ]
        .eq("grib")
        .all()
    )


def test_duplicate_request_rejected(cache):
    with pytest.raises(
        WeatherCacheValidationError,
        match="Duplicate forecast leads",
    ):
        acquire_cached_weather_cycle(cache, None, RUN, [30, 30])


def test_invalid_backend_result_not_cached(cache):
    def broken_acquirer(fs, run, leads):
        result = make_acquirer([])(fs, run, leads)
        result.weather = result.weather.iloc[:-1]
        return result

    with pytest.raises(
        WeatherContractError,
        match="Incorrect cycle record count",
    ):
        acquire_cached_weather_cycle(
            cache,
            None,
            RUN,
            [30],
            acquirer=broken_acquirer,
        )

    assert cache.read_hour(RUN, 30) is None
