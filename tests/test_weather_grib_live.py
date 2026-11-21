"""Opt-in integration test for production GRIB extraction."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.contracts import (
    validate_hourly_weather,
)
from flight_delay.weather.grib_backend import (
    build_grib_reference,
)
from flight_delay.weather.grib_hour import (
    extract_grib_hour,
)

ROOT = Path(__file__).resolve().parents[1]

RUN_AT = pd.Timestamp("2024-07-14T06:00:00Z")
FORECAST_HOUR = 30

CACHE_PATH = ROOT / (
    "data/interim/weather_cache/583149f133579ba3/hrrr_20240714_06_f30.parquet"
)


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_WEATHER_GRIB") != "1",
    reason="Live GRIB test requires explicit opt-in",
)
def test_real_grib_against_cached_reference():
    """Compare production GRIB output with Notebook 09 cache."""

    reference_path = Path(
        os.getenv(
            "HRRR_GRIB_REFERENCE_PARQUET",
            str(CACHE_PATH),
        )
    ).expanduser()

    # Verify local reference before accessing the network.

    assert reference_path.is_file(), f"Missing reference cache: {reference_path}"

    reference_weather = pd.read_parquet(reference_path).copy()

    assert len(reference_weather) == 54

    # Run the production GRIB acquisition pipeline.

    reference = build_grib_reference(
        RUN_AT,
        FORECAST_HOUR,
    )

    result = extract_grib_hour(reference)

    weather = result.weather
    audit = result.request_audit

    assert len(weather) == 54
    assert len(audit) == 9

    assert weather["backend"].eq("grib").all()

    validate_hourly_weather(weather)

    keys = [
        "Origin",
        "feature",
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
        "forecast_hour",
    ]

    for frame in (reference_weather, weather):
        for column in (
            "forecast_run_at_utc",
            "forecast_valid_at_utc",
        ):
            frame[column] = pd.to_datetime(
                frame[column],
                utc=True,
            )

        assert not frame.duplicated(keys).any()

    comparison = reference_weather[keys + ["value"]].merge(
        weather[keys + ["value", "value_status"]],
        on=keys,
        how="outer",
        suffixes=("_reference", "_production"),
        validate="one_to_one",
        indicator=True,
    )

    assert len(comparison) == 54

    assert comparison["_merge"].eq("both").all()

    reference_values = comparison["value_reference"].to_numpy(dtype=float)

    production_values = comparison["value_production"].to_numpy(dtype=float)

    # Missing-value masks must match exactly.

    reference_missing = np.isnan(reference_values)

    production_missing = np.isnan(production_values)

    np.testing.assert_array_equal(
        reference_missing,
        production_missing,
    )

    # Numerical values must agree with the saved GRIB reference.

    valid = ~reference_missing

    assert np.isfinite(production_values[valid]).all()

    np.testing.assert_allclose(
        production_values[valid],
        reference_values[valid],
        rtol=0,
        atol=1e-6,
    )

    absolute_error = np.abs(production_values[valid] - reference_values[valid])

    print("\n=== Production GRIB Integration ===")

    print("Matched records:", len(comparison))
    print("Valid pairs:", valid.sum())
    print("Matching missing values:", reference_missing.sum())

    print(
        "Maximum absolute error:",
        absolute_error.max(),
    )

    print(
        "Downloaded MiB:",
        audit["downloaded_bytes"].sum() / 1024**2,
    )

    print(
        "GRIB messages:",
        len(audit),
    )

    # Preserve production evidence for later cache work.

    output_dir = ROOT / "artifacts/weather_feasibility"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "production_grib_20240714_06_f30.parquet"

    audit_path = output_dir / "production_grib_20240714_06_f30_audit.parquet"

    weather.to_parquet(
        output_path,
        index=False,
    )

    audit.to_parquet(
        audit_path,
        index=False,
    )

    print("Saved weather:", output_path)
    print("Saved audit:", audit_path)

    print("\nProduction GRIB integration: PASS")
