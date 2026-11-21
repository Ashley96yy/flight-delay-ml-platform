"""Offline tests for cache-aware batch request planning."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.batch_plan import (
    build_weather_batch_plan,
    pending_cycle_tasks,
)
from flight_delay.weather.cache import (
    WeatherCache,
    WeatherCacheIntegrityError,
)
from flight_delay.weather.config import (
    AIRPORTS,
    WEATHER_VARIABLES,
)
from flight_delay.weather.planner import WeatherPlanningError

RUN_06 = pd.Timestamp("2024-07-14T06:00:00Z")


@pytest.fixture
def flights():
    return pd.DataFrame(
        {
            "Origin": ["ATL", "DFW", "SFO", "LAX", "ORD"],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:00:00Z",
                    "2024-07-15T09:30:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T15:00:00Z",
                    "2024-07-15T16:05:00Z",
                ],
                utc=True,
            ),
        }
    )


@pytest.fixture
def cache(tmp_path):
    return WeatherCache(tmp_path / "cache")


def make_weather(run_at, lead):
    rows = []

    for airport in AIRPORTS:
        for variable in WEATHER_VARIABLES:
            missing = not variable.required

            rows.append(
                {
                    "Origin": airport.code,
                    "feature": variable.feature,
                    "forecast_run_at_utc": run_at,
                    "forecast_valid_at_utc": (run_at + pd.Timedelta(hours=lead)),
                    "forecast_hour": lead,
                    "assumed_available_at_utc": (run_at + pd.Timedelta(hours=3)),
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

    return pd.DataFrame(rows)


def test_empty_cache_batch_plan(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    assert plan.summary() == {
        "flight_count": 5,
        "unique_hourly_requests": 4,
        "unique_cycles": 2,
        "cached_hours": 0,
        "missing_hours": 4,
        "cycles_fully_cached": 0,
        "cycles_needing_acquisition": 2,
    }


def test_batch_plan_deduplicates_flights(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    first = plan.cycle_tasks.iloc[0]
    second = plan.cycle_tasks.iloc[1]

    assert first["forecast_leads"] == (27, 30)
    assert first["flight_count"] == 3

    assert second["forecast_leads"] == (27, 28)
    assert second["flight_count"] == 2


def test_partial_cache_batch_plan(flights, cache):
    cache.write_hour(RUN_06, 30, make_weather(RUN_06, 30))

    plan = build_weather_batch_plan(flights, cache)

    assert plan.summary()["cached_hours"] == 1
    assert plan.summary()["missing_hours"] == 3

    first = plan.cycle_tasks.iloc[0]

    assert first["cached_leads"] == (30,)
    assert first["missing_leads"] == (27,)
    assert first["status"] == "pending"


def test_all_cached_cycles_are_ready(flights, cache):
    initial = build_weather_batch_plan(flights, cache)

    for row in initial.hour_status.itertuples(index=False):
        cache.write_hour(
            row.forecast_run_at_utc,
            row.forecast_hour,
            make_weather(
                row.forecast_run_at_utc,
                row.forecast_hour,
            ),
        )

    plan = build_weather_batch_plan(flights, cache)

    assert plan.summary()["missing_hours"] == 0
    assert plan.summary()["cached_hours"] == 4
    assert plan.summary()["cycles_fully_cached"] == 2

    assert pending_cycle_tasks(plan).empty


def test_corrupt_cache_is_not_treated_as_miss(flights, cache):
    path = cache.write_hour(RUN_06, 30, make_weather(RUN_06, 30))

    with path.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(
        WeatherCacheIntegrityError,
        match="checksum mismatch",
    ):
        build_weather_batch_plan(flights, cache)


def test_batch_plan_does_not_modify_flights(flights, cache):
    original = flights.copy(deep=True)

    build_weather_batch_plan(flights, cache)

    pd.testing.assert_frame_equal(flights, original)


def test_dry_run_does_not_create_cache_files(flights, cache):
    build_weather_batch_plan(flights, cache)

    assert not cache.namespace.exists()


def test_custom_departure_and_origin_columns(flights, cache):
    renamed = flights.rename(
        columns={
            "Origin": "airport",
            "scheduled_departure_utc": "departure",
        }
    )

    plan = build_weather_batch_plan(
        renamed,
        cache,
        departure_col="departure",
        origin_col="airport",
    )

    assert plan.summary()["flight_count"] == 5
    assert plan.summary()["unique_cycles"] == 2


def test_naive_departures_rejected(flights, cache):
    flights["scheduled_departure_utc"] = flights[
        "scheduled_departure_utc"
    ].dt.tz_localize(None)

    with pytest.raises(
        WeatherPlanningError,
        match="timezone-aware",
    ):
        build_weather_batch_plan(flights, cache)
