"""Decode HRRR GRIB messages into airport-level weather records."""

from functools import lru_cache

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    AIRPORTS,
    GRID_MATCH_TOLERANCE_M,
    VARIABLE_BY_FEATURE,
    WEATHER_AVAILABILITY_LAG_HOURS,
)
from flight_delay.weather.grib_backend import GRIBBackendError
from flight_delay.weather.grid import (
    HRRR_GEOD,
    HRRR_GRID_SHAPE,
    HRRR_GRID_SPACING_M,
    HRRR_X_START,
    HRRR_Y_START,
    build_airport_grid_mapping,
    normalize_longitude,
)


class GRIBDecodeError(GRIBBackendError):
    """GRIB message decoding or validation failure."""


@lru_cache(maxsize=1)
def reference_airport_grid():
    """Return the validated HRRR grid for the six airports."""

    ny, nx = HRRR_GRID_SHAPE

    x = HRRR_X_START + np.arange(nx) * HRRR_GRID_SPACING_M
    y = HRRR_Y_START + np.arange(ny) * HRRR_GRID_SPACING_M

    return build_airport_grid_mapping(x, y).set_index("airport")


def decode_grib_airports(
    payload,
    feature,
    reference,
    *,
    eccodes_module=None,
):
    """Decode one GRIB message into six airport records."""

    if feature not in VARIABLE_BY_FEATURE:
        raise GRIBDecodeError(f"Unknown weather feature: {feature}")

    if not payload.startswith(b"GRIB") or not payload.endswith(b"7777"):
        raise GRIBDecodeError("Invalid GRIB message markers")

    if eccodes_module is None:
        import eccodes

        eccodes_module = eccodes

    variable = VARIABLE_BY_FEATURE[feature]
    expected_grid = reference_airport_grid()

    try:
        gid = eccodes_module.codes_new_from_message(payload)
    except Exception as exc:
        raise GRIBDecodeError("Cannot open GRIB message") from exc

    if gid is None:
        raise GRIBDecodeError("Empty GRIB message handle")

    try:

        def get(key):
            return eccodes_module.codes_get(gid, key)

        run_at = pd.to_datetime(
            f"{int(get('dataDate')):08d}{int(get('dataTime')):04d}",
            format="%Y%m%d%H%M",
            utc=True,
        )

        valid_at = pd.to_datetime(
            f"{int(get('validityDate')):08d}{int(get('validityTime')):04d}",
            format="%Y%m%d%H%M",
            utc=True,
        )

        end_step = int(get("endStep"))
        start_step = int(get("startStep"))

        if run_at != reference.run_at:
            raise GRIBDecodeError("Forecast initialization mismatch")

        if valid_at != reference.valid_at:
            raise GRIBDecodeError("Forecast valid time mismatch")

        if end_step != reference.forecast_hour:
            raise GRIBDecodeError("Forecast lead mismatch")

        if feature == "precipitation_1h" and start_step != end_step - 1:
            raise GRIBDecodeError("Incorrect precipitation accumulation interval")

        expected = {
            "shortName": variable.grib_short_name,
            "units": variable.grib_units,
            "typeOfLevel": variable.grib_type_of_level,
            "stepType": variable.grib_step_type,
        }

        actual = {key: get(key) for key in expected}

        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                raise GRIBDecodeError(
                    f"{feature}: metadata mismatch for {key}: "
                    f"{actual[key]!r} != {expected_value!r}"
                )

        bitmap_present = bool(get("bitmapPresent"))

        bitmap = (
            np.asarray(eccodes_module.codes_get_array(gid, "bitmap"))
            if bitmap_present
            else None
        )

        records = []

        for airport in AIRPORTS:
            nearest_matches = eccodes_module.codes_grib_find_nearest(
                gid,
                airport.latitude,
                airport.longitude,
            )

            if len(nearest_matches) != 1:
                raise GRIBDecodeError(f"Unexpected nearest-grid result: {airport.code}")

            nearest = nearest_matches[0]

            latitude = float(nearest["lat"])
            longitude = float(normalize_longitude(nearest["lon"]))

            expected_location = expected_grid.loc[airport.code]

            _, _, separation_m = HRRR_GEOD.inv(
                longitude,
                latitude,
                float(expected_location["grid_longitude"]),
                float(expected_location["grid_latitude"]),
            )

            if not np.isfinite(separation_m) or separation_m > GRID_MATCH_TOLERANCE_M:
                raise GRIBDecodeError(f"GRIB/Zarr grid mismatch: {airport.code}")

            raw_value = float(nearest["value"])

            missing_bitmap_value = False

            if bitmap is not None:
                grid_index = int(nearest["index"])

                if not 0 <= grid_index < len(bitmap):
                    raise GRIBDecodeError(f"Invalid bitmap index: {airport.code}")

                missing_bitmap_value = bitmap[grid_index] == 0

            if missing_bitmap_value:
                value = np.nan
                value_status = "source_missing"
            elif not np.isfinite(raw_value):
                value = np.nan
                value_status = "nonfinite"
            else:
                value = raw_value
                value_status = "valid"

            records.append(
                {
                    "Origin": airport.code,
                    "feature": feature,
                    "forecast_run_at_utc": reference.run_at,
                    "forecast_valid_at_utc": reference.valid_at,
                    "forecast_hour": reference.forecast_hour,
                    "assumed_available_at_utc": (
                        reference.run_at
                        + pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)
                    ),
                    "value": value,
                    "raw_value": raw_value,
                    "value_status": value_status,
                    "grid_latitude": latitude,
                    "grid_longitude": longitude,
                    "backend": "grib",
                    "storage_dtype": None,
                    "units": variable.grib_units,
                    "source_units": actual["units"],
                    "fallback_reason": None,
                }
            )

        return pd.DataFrame(records)

    except GRIBDecodeError:
        raise
    except Exception as exc:
        raise GRIBDecodeError(f"GRIB decoding failed: {feature}") from exc
    finally:
        eccodes_module.codes_release(gid)
