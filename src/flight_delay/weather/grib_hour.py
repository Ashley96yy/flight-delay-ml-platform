"""Acquire one complete hourly HRRR GRIB weather batch."""

from dataclasses import dataclass
from time import perf_counter
from urllib.request import urlopen

import pandas as pd

from flight_delay.weather.contracts import validate_hourly_weather
from flight_delay.weather.grib_backend import (
    GRIBInventoryError,
    download_grib_range,
    parse_grib_inventory,
    select_weather_messages,
)
from flight_delay.weather.grib_decoder import decode_grib_airports


@dataclass
class GRIBHourResult:
    weather: pd.DataFrame
    request_audit: pd.DataFrame


def fetch_grib_inventory(index_url, *, opener=urlopen):
    """Fetch and parse an HTTP GRIB index."""

    try:
        with opener(index_url, timeout=30) as response:
            text = response.read().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise GRIBInventoryError("Failed to fetch GRIB inventory") from exc

    return parse_grib_inventory(text.splitlines())


def extract_grib_hour(
    reference,
    *,
    index_opener=urlopen,
    range_opener=urlopen,
    eccodes_module=None,
    fallback_reason=None,
):
    """Extract nine weather variables for all six airports."""

    inventory = fetch_grib_inventory(
        reference.index_url,
        opener=index_opener,
    )

    messages = select_weather_messages(
        inventory,
        reference.forecast_hour,
    )

    weather_parts = []
    audit_records = []

    for message in messages.itertuples(index=False):
        started = perf_counter()

        payload = download_grib_range(
            reference.grib_url,
            message.start_byte,
            message.end_byte,
            opener=range_opener,
        )

        download_seconds = perf_counter() - started

        started = perf_counter()

        values = decode_grib_airports(
            payload,
            message.feature,
            reference,
            eccodes_module=eccodes_module,
        )

        decode_seconds = perf_counter() - started

        if fallback_reason is not None:
            values["fallback_reason"] = fallback_reason

        weather_parts.append(values)

        audit_records.append(
            {
                "feature": message.feature,
                "message": message.message,
                "start_byte": message.start_byte,
                "end_byte": message.end_byte,
                "downloaded_bytes": len(payload),
                "download_seconds": download_seconds,
                "decode_seconds": decode_seconds,
                "source_url": reference.grib_url,
            }
        )

    weather = (
        pd.concat(
            weather_parts,
            ignore_index=True,
        )
        .sort_values(["feature", "Origin"])
        .reset_index(drop=True)
    )

    validate_hourly_weather(weather)

    return GRIBHourResult(
        weather=weather,
        request_audit=pd.DataFrame(audit_records),
    )
