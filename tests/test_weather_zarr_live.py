"""Opt-in real archive integration test for HRRR Zarr."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import s3fs

from flight_delay.weather.config import RAW_WEATHER_COLUMNS
from flight_delay.weather.contracts import (
    FORECAST_MATCH_KEYS,
    validate_hourly_weather,
)
from flight_delay.weather.grid import (
    HRRR_GEOD,
    normalize_longitude,
)
from flight_delay.weather.zarr_cycle import extract_zarr_cycle

ROOT = Path(__file__).resolve().parents[1]

RUN_AT = pd.Timestamp("2024-07-14T06:00:00Z")
FORECAST_LEAD = 30

CACHE_FILENAME = "hrrr_20240714_06_f30.parquet"


def find_grib_reference() -> Path:
    """Find an existing GRIB cache; never download it."""

    override = os.getenv("HRRR_GRIB_REFERENCE_PARQUET")

    if override:
        path = Path(override).expanduser().resolve()

        if not path.is_file():
            raise AssertionError(f"Reference cache does not exist: {path}")

        return path

    cache_root = ROOT / "data/interim/weather_cache"

    candidates = sorted(cache_root.glob(f"*/{CACHE_FILENAME}"))

    if len(candidates) != 1:
        raise AssertionError(
            "Expected exactly one existing GRIB cache, "
            f"found {len(candidates)}: {candidates}. "
            "Set HRRR_GRIB_REFERENCE_PARQUET explicitly "
            "if multiple configurations exist."
        )

    return candidates[0]


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_WEATHER_ZARR") != "1",
    reason="Real Zarr integration requires explicit opt-in",
)
def test_real_zarr_against_cached_grib():
    """Compare production Zarr extraction with cached GRIB."""

    # 1. Verify reference availability before network access.

    grib_path = find_grib_reference()

    grib = pd.read_parquet(grib_path).copy()

    assert len(grib) == 54

    assert set(grib["feature"]) == set(RAW_WEATHER_COLUMNS)

    assert not grib.duplicated(subset=FORECAST_MATCH_KEYS).any()

    for column in (
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
    ):
        grib[column] = pd.to_datetime(
            grib[column],
            utc=True,
        )

    assert grib["forecast_run_at_utc"].eq(RUN_AT).all()

    assert grib["forecast_hour"].eq(FORECAST_LEAD).all()

    assert (
        grib["forecast_valid_at_utc"]
        .eq(RUN_AT + pd.Timedelta(hours=FORECAST_LEAD))
        .all()
    )

    print("\nGRIB reference:", grib_path)

    # 2. Execute the actual production Zarr extractor.

    fs = s3fs.S3FileSystem(anon=True)

    result = extract_zarr_cycle(
        fs,
        RUN_AT,
        [FORECAST_LEAD],
    )

    zarr = result.weather.copy()
    audit = result.chunk_audit.copy()

    assert len(zarr) == 54

    assert len(audit) == 54

    assert not audit.duplicated(subset=["feature", "chunk_key"]).any()

    assert audit.groupby("feature").size().eq(6).all()

    assert zarr["storage_dtype"].eq("float32").all()

    validate_hourly_weather(zarr)

    # 3. Match complete forecast identities.

    grib_compare = grib[
        list(FORECAST_MATCH_KEYS)
        + [
            "value",
            "grid_latitude",
            "grid_longitude",
        ]
    ].rename(
        columns={
            "value": "grib_value",
            "grid_latitude": "grib_latitude",
            "grid_longitude": "grib_longitude",
        }
    )

    zarr_compare = zarr[
        list(FORECAST_MATCH_KEYS)
        + [
            "value",
            "grid_latitude",
            "grid_longitude",
            "value_status",
        ]
    ].rename(
        columns={
            "value": "zarr_value",
            "grid_latitude": "zarr_latitude",
            "grid_longitude": "zarr_longitude",
        }
    )

    comparison = grib_compare.merge(
        zarr_compare,
        on=list(FORECAST_MATCH_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    assert len(comparison) == 54
    assert comparison["_merge"].eq("both").all()

    # 4. Verify spatial consistency.

    _, _, separation_m = HRRR_GEOD.inv(
        normalize_longitude(comparison["grib_longitude"].to_numpy()),
        comparison["grib_latitude"].to_numpy(),
        normalize_longitude(comparison["zarr_longitude"].to_numpy()),
        comparison["zarr_latitude"].to_numpy(),
    )

    assert np.isfinite(separation_m).all()
    assert (separation_m <= 2.0).all()

    # 5. Verify matching missing-value masks.

    grib_values = comparison["grib_value"].to_numpy(dtype=np.float64)

    zarr_values = comparison["zarr_value"].to_numpy(dtype=np.float64)

    grib_missing = np.isnan(grib_values)
    zarr_missing = np.isnan(zarr_values)

    assert np.array_equal(
        grib_missing,
        zarr_missing,
    )

    assert np.isfinite(grib_values[~grib_missing]).all()

    assert np.isfinite(zarr_values[~zarr_missing]).all()

    both_valid = ~grib_missing

    # 6. Verify exact equality after Float32 casting.

    expected_zarr = grib_values[both_valid].astype(np.float32).astype(np.float64)

    observed_zarr = zarr_values[both_valid]

    adjusted_matches = expected_zarr == observed_zarr

    assert adjusted_matches.all(), (
        f"{(~adjusted_matches).sum()} precision-adjusted mismatches"
    )

    # 7. Print validation summary.

    raw_error = np.abs(zarr_values[both_valid] - grib_values[both_valid])

    print("\n=== Production Zarr Integration ===")

    print("Matched records:", len(comparison))
    print("Valid numerical pairs:", both_valid.sum())
    print("Matching missing values:", grib_missing.sum())
    print("Adjusted matches:", adjusted_matches.sum())

    print(
        "Maximum raw numerical error:",
        raw_error.max(),
    )

    print(
        "Maximum grid separation (m):",
        separation_m.max(),
    )

    print(
        "Downloaded payload (MiB):",
        audit["compressed_bytes"].sum() / 1024**2,
    )

    print("\n=== Per-Variable Coverage ===")

    display_summary = (
        comparison.assign(
            missing=grib_missing,
        )
        .groupby("feature")
        .agg(
            records=("Origin", "size"),
            missing=("missing", "sum"),
        )
    )

    print(display_summary.to_string())

    # 8. Save verified artifacts for future reuse.

    output_dir = ROOT / "artifacts/weather_feasibility"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather_path = output_dir / "production_zarr_20240714_06_f30.parquet"

    audit_path = output_dir / "production_zarr_20240714_06_f30_audit.parquet"

    zarr.to_parquet(weather_path, index=False)
    audit.to_parquet(audit_path, index=False)

    print("\nSaved weather:", weather_path)
    print("Saved audit:", audit_path)

    print("\nProduction Zarr integration: PASS")
