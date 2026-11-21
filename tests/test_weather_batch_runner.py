"""Offline tests for parallel weather batch execution."""

import json
from threading import Lock
from time import sleep

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.batch_plan import build_weather_batch_plan
from flight_delay.weather.batch_runner import (
    WeatherBatchError,
    run_weather_batch,
)
from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.config import AIRPORTS, WEATHER_VARIABLES
from flight_delay.weather.fallback import (
    WeatherAcquisitionError,
    WeatherCycleResult,
)


@pytest.fixture
def cache(tmp_path):
    return WeatherCache(tmp_path / "cache")


@pytest.fixture
def flights():
    return pd.DataFrame(
        {
            "Origin": ["ATL", "DFW", "SFO", "LAX"],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:15:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T15:00:00Z",
                    "2024-07-15T16:05:00Z",
                ],
                utc=True,
            ),
        }
    )


def make_weather(run_at, leads):
    rows = []

    for lead in leads:
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


def fake_acquirer(calls, fail_hour=None):
    def acquire(fs, run_at, leads):
        calls.append((run_at.hour, tuple(leads)))

        if run_at.hour == fail_hour:
            raise WeatherAcquisitionError("Simulated weather acquisition failure")

        return WeatherCycleResult(
            weather=make_weather(run_at, leads),
            attempt_audit=pd.DataFrame([{"backend": "zarr", "status": "success"}]),
            backend_audit=pd.DataFrame(),
            backend="zarr",
            fallback_reason=None,
        )

    return acquire


def test_first_batch_fetches_all(flights, cache):
    plan = build_weather_batch_plan(flights, cache)
    calls = []

    result = run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer(calls),
        max_workers=2,
    )

    assert result.summary["succeeded_cycles"] == 2
    assert result.summary["failed_cycles"] == 0
    assert result.summary["fetched_hours"] == 4
    assert len(calls) == 2


def test_rerun_uses_cache_without_filesystem(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer([]),
    )

    def forbidden_factory():
        raise AssertionError("No filesystem needed")

    fresh_plan = build_weather_batch_plan(flights, cache)

    result = run_weather_batch(
        fresh_plan,
        cache,
        fs_factory=forbidden_factory,
        acquirer=fake_acquirer([]),
    )

    assert result.summary["cached_cycles"] == 2
    assert result.summary["fetched_hours"] == 0
    assert result.summary["cache_hits"] == 4


def test_partial_cache_only_fetches_missing(flights, cache):
    run = pd.Timestamp("2024-07-14T06:00:00Z")

    cache.write_hour(run, 30, make_weather(run, [30]))

    calls = []

    plan = build_weather_batch_plan(flights, cache)

    result = run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer(calls),
    )

    assert (6, (27,)) in calls
    assert (12, (27, 28)) in calls

    assert result.summary["cache_hits"] == 1
    assert result.summary["fetched_hours"] == 3


def test_one_cycle_failure_does_not_stop_other(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    result = run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer([], fail_hour=12),
    )

    assert result.summary["succeeded_cycles"] == 1
    assert result.summary["failed_cycles"] == 1

    failed = result.task_report.loc[result.task_report["status"].eq("failed")]

    assert len(failed) == 1
    assert failed.iloc[0]["error_type"] == "WeatherAcquisitionError"

    run06 = pd.Timestamp("2024-07-14T06:00:00Z")
    run12 = pd.Timestamp("2024-07-14T12:00:00Z")

    assert cache.read_hour(run06, 27) is not None
    assert cache.read_hour(run12, 27) is None


def test_resume_only_failed_cycle(flights, cache):
    first_plan = build_weather_batch_plan(flights, cache)

    run_weather_batch(
        first_plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer([], fail_hour=12),
    )

    calls = []

    second_plan = build_weather_batch_plan(flights, cache)

    result = run_weather_batch(
        second_plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer(calls),
    )

    assert calls == [(12, (27, 28))]
    assert result.summary["cached_cycles"] == 1
    assert result.summary["succeeded_cycles"] == 1
    assert result.summary["failed_cycles"] == 0


def test_duplicate_cycle_task_rejected(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    plan.cycle_tasks = pd.concat(
        [plan.cycle_tasks, plan.cycle_tasks.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        WeatherBatchError,
        match="Duplicate forecast cycle task",
    ):
        run_weather_batch(
            plan,
            cache,
            fs_factory=lambda: object(),
        )


def test_report_files_written(flights, cache, tmp_path):
    plan = build_weather_batch_plan(flights, cache)
    report_dir = tmp_path / "batch_report"

    result = run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=fake_acquirer([]),
        report_dir=report_dir,
    )

    assert (report_dir / "task_report.csv").is_file()
    assert (report_dir / "cache_audit.csv").is_file()
    assert (report_dir / "attempt_audit.csv").is_file()
    assert (report_dir / "summary.json").is_file()

    saved_summary = json.loads((report_dir / "summary.json").read_text())

    assert saved_summary == result.summary


def test_thread_concurrency_is_bounded(flights, cache):
    plan = build_weather_batch_plan(flights, cache)

    active = 0
    maximum_active = 0
    lock = Lock()

    def measured_acquirer(fs, run_at, leads):
        nonlocal active, maximum_active

        with lock:
            active += 1
            maximum_active = max(maximum_active, active)

        try:
            sleep(0.02)

            return fake_acquirer([])(fs, run_at, leads)
        finally:
            with lock:
                active -= 1

    result = run_weather_batch(
        plan,
        cache,
        fs_factory=lambda: object(),
        acquirer=measured_acquirer,
        max_workers=2,
    )

    assert maximum_active <= 2
    assert result.summary["failed_cycles"] == 0


@pytest.mark.parametrize("workers", [0, 5, True])
def test_invalid_worker_count(flights, cache, workers):
    plan = build_weather_batch_plan(flights, cache)

    with pytest.raises(
        WeatherBatchError,
        match="max_workers",
    ):
        run_weather_batch(
            plan,
            cache,
            fs_factory=lambda: object(),
            max_workers=workers,
        )
