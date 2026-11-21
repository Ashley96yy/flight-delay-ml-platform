"""Verify flight-weather joins and leakage using cached HRRR data.

Run from the project root:

    python scripts/verify_weather_coverage.py

This script is read-only and does not access the network.
"""

from pathlib import Path

import pandas as pd

from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.coverage_audit import audit_weather_coverage
from flight_delay.weather.features import (
    MODEL_WEATHER_COLUMNS,
    build_weather_feature_table,
)
from flight_delay.weather.flight_join import join_weather_to_flights
from flight_delay.weather.planner import build_forecast_request_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_ROOT = PROJECT_ROOT / "data/interim/weather_cache_v2"


def main():
    # --------------------------------------------------
    # Step 1: Create three example flights
    # --------------------------------------------------

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

    print("\n=== Step 1: Input Flights ===")
    print(flights.to_string(index=False))

    # --------------------------------------------------
    # Step 2: Build the forecast request plan
    # --------------------------------------------------

    plan = build_forecast_request_plan(flights)

    print("\n=== Step 2: Forecast Request Plan ===")
    print(
        plan.flight_mapping[
            [
                "Origin",
                "scheduled_departure_utc",
                "prediction_at_utc",
                "forecast_run_at_utc",
                "forecast_valid_at_utc",
                "forecast_hour",
            ]
        ].to_string(index=False)
    )

    print(
        "\nUnique hourly requests:",
        len(plan.hourly_requests),
    )

    # --------------------------------------------------
    # Step 3: Read previously cached HRRR data
    # --------------------------------------------------

    cache = WeatherCache(CACHE_ROOT)

    weather_parts = []

    for request in plan.hourly_requests.itertuples(index=False):
        run_at = request.forecast_run_at_utc
        lead = int(request.forecast_hour)

        hourly = cache.read_hour(run_at, lead)

        if hourly is None:
            raise RuntimeError(
                "Missing cached HRRR weather data: "
                f"{run_at}, F{lead:02d}. "
                "Run Step 7C first."
            )

        weather_parts.append(hourly)

        print(
            f"Cache HIT: {run_at}, F{lead:02d}, "
            f"{len(hourly)} records, "
            f"backend={hourly['backend'].iloc[0]}"
        )

    weather_long = pd.concat(
        weather_parts,
        ignore_index=True,
    )

    print("\n=== Step 3: Raw Weather ===")
    print("Raw weather records:", len(weather_long))

    # --------------------------------------------------
    # Step 4: Generate model-ready weather features
    # --------------------------------------------------

    weather_features = build_weather_feature_table(weather_long)

    print("\n=== Step 4: Model Weather Features ===")
    print("Airport-hour records:", len(weather_features))
    print("Weather feature count:", len(MODEL_WEATHER_COLUMNS))

    print(
        weather_features[["Origin", "forecast_hour"] + MODEL_WEATHER_COLUMNS]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # --------------------------------------------------
    # Step 5: Join weather features to flights
    # --------------------------------------------------

    joined = join_weather_to_flights(
        plan.flight_mapping,
        weather_features,
    )

    if len(joined) != len(flights):
        raise AssertionError("Flight row count changed during weather join")

    if joined[MODEL_WEATHER_COLUMNS].isna().any().any():
        raise AssertionError("Joined flight data contains missing weather features")

    print("\n=== Step 5: Real Flight-Weather Join ===")

    display_columns = [
        "Origin",
        "scheduled_departure_utc",
        "forecast_run_at_utc",
        "forecast_hour",
        "weather_backend",
    ] + MODEL_WEATHER_COLUMNS

    print(
        joined[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nPASS: Flight-weather join")

    # --------------------------------------------------
    # Step 6: Independent leakage and coverage audit
    # --------------------------------------------------

    audit = audit_weather_coverage(
        plan.flight_mapping,
        weather_features,
        strict=True,
    )

    print("\n=== Step 6: Leakage & Coverage Audit ===")

    print("\nOverall Summary:")
    for key, value in audit.summary.items():
        print(f"{key}: {value}")

    print("\nCoverage by Airport:")
    print(
        audit.by_airport.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nViolations:")

    if audit.violations.empty:
        print("None")
    else:
        print(audit.violations[["Origin", "status"]].to_string(index=False))

    # --------------------------------------------------
    # Step 7: Final assertions
    # --------------------------------------------------

    assert audit.summary["flight_count"] == 3
    assert audit.summary["matched_flights"] == 3
    assert audit.summary["ready_flights"] == 3
    assert audit.summary["missing_weather_flights"] == 0
    assert audit.summary["invalid_flights"] == 0
    assert audit.summary["coverage_rate"] == 1.0

    assert audit.violations.empty

    assert len(weather_long) == 162
    assert len(weather_features) == 18
    assert len(MODEL_WEATHER_COLUMNS) == 7

    print("\n" + "=" * 55)
    print("PASS: REAL WEATHER FEATURE VALIDATION")
    print("PASS: FLIGHT-WEATHER JOIN")
    print("PASS: POINT-IN-TIME LEAKAGE CHECK")
    print("PASS: 100% WEATHER COVERAGE")
    print("=" * 55)


if __name__ == "__main__":
    main()
