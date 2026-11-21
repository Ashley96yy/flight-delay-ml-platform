"""Cache-aware weather acquisition and restart recovery."""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from flight_delay.weather.cache import (
    WeatherCache,
    WeatherCacheValidationError,
    validate_cache_identity,
)
from flight_delay.weather.contracts import (
    WeatherContractError,
)
from flight_delay.weather.fallback import (
    acquire_weather_cycle,
    validate_cycle_output,
)


@dataclass
class CachedCycleResult:
    """Weather records and cache/acquisition audit trails."""

    weather: pd.DataFrame
    cache_audit: pd.DataFrame
    attempt_audit: pd.DataFrame
    backend_audit: pd.DataFrame


def acquire_cached_weather_cycle(
    cache: WeatherCache,
    zarr_fs,
    run_at,
    forecast_leads,
    *,
    acquirer: Callable = acquire_weather_cycle,
) -> CachedCycleResult:
    """Reuse validated cached hours and acquire only missing hours."""

    try:
        requested_leads = list(forecast_leads)
    except TypeError as exc:
        raise WeatherCacheValidationError("Forecast leads must be iterable") from exc

    if not requested_leads:
        raise WeatherCacheValidationError("Forecast leads must not be empty")

    # Validate all requests before touching cache or network.

    validated = [validate_cache_identity(run_at, lead) for lead in requested_leads]

    run_at = validated[0][0]
    leads = sorted(lead for _, lead in validated)

    if len(leads) != len(set(leads)):
        raise WeatherCacheValidationError("Duplicate forecast leads")

    cached = {}
    missing = []
    audit_records = []

    # Cache corruption is deliberately fail-closed.
    # Do not silently treat corrupted data as a cache miss.

    for lead in leads:
        frame = cache.read_hour(run_at, lead)

        if frame is None:
            missing.append(lead)
        else:
            cached[lead] = frame

            audit_records.append(
                {
                    "forecast_hour": lead,
                    "cache_status": "hit",
                    "backend": frame["backend"].iloc[0],
                    "fallback_reason": (frame["fallback_reason"].iloc[0]),
                }
            )

    attempt_audit = pd.DataFrame()
    backend_audit = pd.DataFrame()

    if missing:
        # Acquire all missing leads together to preserve
        # Zarr's cross-lead chunk reuse.

        acquired = acquirer(
            zarr_fs,
            run_at,
            missing,
        )

        # Validate again at the cache boundary, even if the
        # underlying acquirer has already validated its output.

        validate_cycle_output(
            acquired.weather,
            run_at,
            missing,
            acquired.backend,
            fallback_reason=acquired.fallback_reason,
        )

        attempt_audit = acquired.attempt_audit.copy()
        backend_audit = acquired.backend_audit.copy()

        # Commit each hour independently. If the process stops
        # partway through, completed hours remain reusable.

        for lead in missing:
            hourly = acquired.weather.loc[
                acquired.weather["forecast_hour"].eq(lead)
            ].copy()

            if len(hourly) != 54:
                raise WeatherContractError(f"Incomplete fetched hour: F{lead:02d}")

            cache.write_hour(
                run_at,
                lead,
                hourly,
            )

            cached[lead] = hourly

            audit_records.append(
                {
                    "forecast_hour": lead,
                    "cache_status": "fetched",
                    "backend": hourly["backend"].iloc[0],
                    "fallback_reason": (hourly["fallback_reason"].iloc[0]),
                }
            )

    weather = (
        pd.concat(
            [cached[lead] for lead in leads],
            ignore_index=True,
        )
        .sort_values(["forecast_hour", "feature", "Origin"])
        .reset_index(drop=True)
    )

    cache_audit = (
        pd.DataFrame(audit_records).sort_values("forecast_hour").reset_index(drop=True)
    )

    return CachedCycleResult(
        weather=weather,
        cache_audit=cache_audit,
        attempt_audit=attempt_audit,
        backend_audit=backend_audit,
    )
