"""Zarr-first weather acquisition with GRIB fallback."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    ORIGIN_AIRPORTS,
    RAW_WEATHER_COLUMNS,
    SUPPORTED_FORECAST_LEADS,
)
from flight_delay.weather.contracts import (
    FORECAST_MATCH_KEYS,
    WeatherContractError,
    validate_hourly_weather,
)
from flight_delay.weather.grib_backend import build_grib_reference
from flight_delay.weather.grib_hour import extract_grib_hour
from flight_delay.weather.grid import WeatherGridError
from flight_delay.weather.zarr_backend import (
    ZarrBackendError,
    ZarrChunkError,
    ZarrMetadataError,
)
from flight_delay.weather.zarr_cycle import (
    ZarrCoordinateError,
    extract_zarr_cycle,
)


class WeatherAcquisitionError(RuntimeError):
    """Both backends failed or the acquisition request is invalid."""

    def __init__(self, message, attempt_audit=None):
        super().__init__(message)
        self.attempt_audit = pd.DataFrame(
            attempt_audit if attempt_audit is not None else []
        )


@dataclass
class WeatherCycleResult:
    """Validated weather and acquisition provenance."""

    weather: pd.DataFrame
    attempt_audit: pd.DataFrame
    backend_audit: pd.DataFrame
    backend: str
    fallback_reason: str | None


ZARR_RECOVERABLE_ERRORS = (
    ZarrBackendError,
    ZarrCoordinateError,
    WeatherGridError,
    WeatherContractError,
    OSError,
)


def classify_zarr_error(exc):
    """Assign stable machine-readable fallback reasons."""

    if isinstance(exc, ZarrMetadataError):
        return "zarr_metadata_error"
    if isinstance(exc, ZarrChunkError):
        return "zarr_chunk_error"
    if isinstance(exc, ZarrCoordinateError):
        return "zarr_coordinate_error"
    if isinstance(exc, WeatherGridError):
        return "zarr_grid_error"
    if isinstance(exc, WeatherContractError):
        return "zarr_contract_error"
    if isinstance(exc, OSError):
        return "zarr_io_error"
    return "zarr_backend_error"


def validate_cycle_output(
    weather,
    run_at,
    leads,
    backend,
    *,
    fallback_reason=None,
):
    """Validate the result before accepting either backend."""

    required = {
        *FORECAST_MATCH_KEYS,
        "backend",
        "fallback_reason",
    }

    if not isinstance(weather, pd.DataFrame):
        raise WeatherContractError("Backend output must be a DataFrame")

    if not required.issubset(weather.columns):
        raise WeatherContractError("Backend output missing identity columns")

    expected_count = len(leads) * len(ORIGIN_AIRPORTS) * len(RAW_WEATHER_COLUMNS)

    if len(weather) != expected_count:
        raise WeatherContractError("Incorrect cycle record count")

    if set(weather["forecast_hour"]) != set(leads):
        raise WeatherContractError("Unexpected forecast leads")

    if weather.duplicated(subset=list(FORECAST_MATCH_KEYS)).any():
        raise WeatherContractError("Duplicate cycle weather records")

    if not weather["backend"].eq(backend).all():
        raise WeatherContractError("Backend provenance mismatch")

    if fallback_reason is None:
        if weather["fallback_reason"].notna().any():
            raise WeatherContractError("Unexpected fallback reason")
    elif not weather["fallback_reason"].eq(fallback_reason).all():
        raise WeatherContractError("Missing or inconsistent fallback reason")

    for lead in leads:
        hourly = weather.loc[weather["forecast_hour"].eq(lead)]

        validate_hourly_weather(hourly)

        if not hourly["forecast_run_at_utc"].eq(run_at).all():
            raise WeatherContractError("Forecast initialization mismatch")

        if (
            not hourly["forecast_valid_at_utc"]
            .eq(run_at + pd.Timedelta(hours=lead))
            .all()
        ):
            raise WeatherContractError("Forecast valid time mismatch")


def _attempt_record(
    backend,
    status,
    seconds,
    *,
    forecast_hour=None,
    error=None,
    error_code=None,
):
    return {
        "backend": backend,
        "forecast_hour": forecast_hour,
        "status": status,
        "seconds": seconds,
        "error_code": error_code,
        "error_type": (type(error).__name__ if error is not None else None),
        "error_message": (str(error) if error is not None else None),
    }


def acquire_weather_cycle(
    zarr_fs,
    run_at,
    forecast_leads,
    *,
    zarr_extractor=extract_zarr_cycle,
    grib_extractor=extract_grib_hour,
):
    """Acquire a cycle using Zarr, then fall back to GRIB."""

    try:
        leads = list(forecast_leads)
    except TypeError as exc:
        raise WeatherAcquisitionError("Forecast leads must be iterable") from exc

    if (
        not leads
        or any(
            not isinstance(lead, (int, np.integer))
            or isinstance(lead, (bool, np.bool_))
            or lead not in SUPPORTED_FORECAST_LEADS
            for lead in leads
        )
        or len(leads) != len(set(leads))
    ):
        raise WeatherAcquisitionError("Invalid or duplicate forecast leads")

    leads = sorted(int(lead) for lead in leads)

    try:
        references = {lead: build_grib_reference(run_at, lead) for lead in leads}
    except (ValueError, RuntimeError) as exc:
        raise WeatherAcquisitionError("Invalid forecast initialization") from exc

    run_at = references[leads[0]].run_at

    attempts = []

    # Try the primary backend once for the whole cycle.

    started = perf_counter()

    try:
        zarr_result = zarr_extractor(
            zarr_fs,
            run_at,
            leads,
        )

        validate_cycle_output(
            zarr_result.weather,
            run_at,
            leads,
            "zarr",
        )

    except ZARR_RECOVERABLE_ERRORS as exc:
        reason = classify_zarr_error(exc)

        attempts.append(
            _attempt_record(
                "zarr",
                "failed",
                perf_counter() - started,
                error=exc,
                error_code=reason,
            )
        )

    else:
        attempts.append(
            _attempt_record(
                "zarr",
                "success",
                perf_counter() - started,
            )
        )

        return WeatherCycleResult(
            weather=zarr_result.weather,
            attempt_audit=pd.DataFrame(attempts),
            backend_audit=zarr_result.chunk_audit.assign(backend="zarr"),
            backend="zarr",
            fallback_reason=None,
        )

    # The entire requested cycle now uses GRIB.
    # Do not return partially extracted Zarr records.

    grib_parts = []
    grib_audits = []

    for lead in leads:
        started = perf_counter()

        try:
            grib_result = grib_extractor(
                references[lead],
                fallback_reason=reason,
            )

            validate_cycle_output(
                grib_result.weather,
                run_at,
                [lead],
                "grib",
                fallback_reason=reason,
            )

        except Exception as exc:
            attempts.append(
                _attempt_record(
                    "grib",
                    "failed",
                    perf_counter() - started,
                    forecast_hour=lead,
                    error=exc,
                    error_code="grib_fallback_failed",
                )
            )

            raise WeatherAcquisitionError(
                f"Zarr acquisition failed and GRIB fallback also failed at F{lead:02d}",
                attempt_audit=attempts,
            ) from exc

        attempts.append(
            _attempt_record(
                "grib",
                "success",
                perf_counter() - started,
                forecast_hour=lead,
            )
        )

        grib_parts.append(grib_result.weather)

        grib_audits.append(
            grib_result.request_audit.assign(
                backend="grib",
                forecast_hour=lead,
            )
        )

    weather = pd.concat(
        grib_parts,
        ignore_index=True,
    )

    validate_cycle_output(
        weather,
        run_at,
        leads,
        "grib",
        fallback_reason=reason,
    )

    return WeatherCycleResult(
        weather=weather,
        attempt_audit=pd.DataFrame(attempts),
        backend_audit=pd.concat(
            grib_audits,
            ignore_index=True,
        ),
        backend="grib",
        fallback_reason=reason,
    )
