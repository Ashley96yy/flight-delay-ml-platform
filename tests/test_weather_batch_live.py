"""Opt-in real HRRR batch acquisition and restart test."""

import os
from pathlib import Path

import pandas as pd
import pytest
import s3fs

from flight_delay.weather.batch_plan import build_weather_batch_plan
from flight_delay.weather.batch_runner import run_weather_batch
from flight_delay.weather.cache import WeatherCache

ROOT = Path(__file__).resolve().parents[1]

RUN_06 = pd.Timestamp("2024-07-14T06:00:00Z")
RUN_12 = pd.Timestamp("2024-07-14T12:00:00Z")


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_WEATHER_BATCH") != "1",
    reason="Real batch integration requires explicit opt-in",
)
def test_real_parallel_batch_and_resume():
    """Fetch missing hours, persist them, and verify zero-download rerun."""

    cache = WeatherCache(ROOT / "data/interim/weather_cache_v2")

    # 1. Require the previously verified F30 production cache.

    existing_f30 = cache.read_hour(RUN_06, 30)

    assert existing_f30 is not None, (
        "Missing production F30 cache; run Step 6B smoke test first"
    )
    assert len(existing_f30) == 54

    # 2. Reuse the same three-flight sample as Step 7A.

    flights = pd.DataFrame(
        {
            "Origin": ["ATL", "SFO", "LAX"],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:15:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T15:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    plan = build_weather_batch_plan(flights, cache)

    assert plan.summary()["unique_cycles"] == 2
    assert plan.summary()["unique_hourly_requests"] == 3

    expected_keys = {
        (RUN_06, 27),
        (RUN_06, 30),
        (RUN_12, 27),
    }

    actual_keys = {
        (
            pd.Timestamp(row.forecast_run_at_utc),
            int(row.forecast_hour),
        )
        for row in plan.hour_status.itertuples(index=False)
    }

    assert actual_keys == expected_keys, (
        "Unexpected forecast requests: refusing broader acquisition"
    )

    missing_before = plan.summary()["missing_hours"]

    assert missing_before <= 2, "Unexpected number of missing weather hours"

    print("\n=== Initial Batch Plan ===")
    print(plan.cycle_tasks.to_string(index=False))
    print("Missing hours:", missing_before)

    # 3. Run the real acquisition with bounded concurrency.

    def fs_factory():
        return s3fs.S3FileSystem(
            anon=True,
            skip_instance_cache=True,
        )

    first = run_weather_batch(
        plan,
        cache,
        fs_factory=fs_factory,
        max_workers=2,
        report_dir=(ROOT / "artifacts/weather_feasibility/batch_7c_first"),
    )

    print("\n=== First Batch Result ===")
    print(first.task_report.to_string(index=False))
    print(first.summary)

    assert first.summary["failed_cycles"] == 0
    assert first.summary["fetched_hours"] == missing_before
    assert first.summary["cache_hits"] == 3 - missing_before

    assert first.task_report["status"].isin(["succeeded", "cached"]).all()

    # 4. Verify every requested hour was persisted.

    for run_at, lead in expected_keys:
        weather = cache.read_hour(run_at, lead)

        assert weather is not None
        assert len(weather) == 54

        assert weather["forecast_hour"].eq(lead).all()
        assert weather["forecast_run_at_utc"].eq(run_at).all()

        assert weather["backend"].isin(["zarr", "grib"]).all()

        print(
            f"Verified {run_at} F{lead:02d}: "
            f"54 records, backend={weather['backend'].iloc[0]}"
        )

    # 5. Rebuild the plan from disk, as on a new process launch.

    resumed_plan = build_weather_batch_plan(
        flights,
        cache,
    )

    assert resumed_plan.summary()["missing_hours"] == 0
    assert resumed_plan.summary()["cached_hours"] == 3
    assert resumed_plan.summary()["cycles_fully_cached"] == 2

    # 6. A rerun must not instantiate S3 clients or acquire data.

    def forbidden_factory():
        raise AssertionError("Filesystem client must not be created on full cache hit")

    def forbidden_acquirer(*args, **kwargs):
        raise AssertionError("No backend acquisition expected on resume")

    second = run_weather_batch(
        resumed_plan,
        cache,
        fs_factory=forbidden_factory,
        acquirer=forbidden_acquirer,
        max_workers=2,
        report_dir=(ROOT / "artifacts/weather_feasibility/batch_7c_resume"),
    )

    print("\n=== Restart / Cache-Only Result ===")
    print(second.task_report.to_string(index=False))
    print(second.summary)

    assert second.summary["cached_cycles"] == 2
    assert second.summary["failed_cycles"] == 0
    assert second.summary["cache_hits"] == 3
    assert second.summary["fetched_hours"] == 0

    assert second.task_report["status"].eq("cached").all()

    print("\nPASS: Real parallel batch and restart recovery")
