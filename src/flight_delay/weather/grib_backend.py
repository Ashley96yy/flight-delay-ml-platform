"""HRRR GRIB inventory parsing and byte-range acquisition."""

import re
from dataclasses import dataclass
from numbers import Integral
from urllib.request import Request, urlopen

import pandas as pd

from flight_delay.weather.config import (
    EXTENDED_CYCLE_HOURS,
    GRIB_BASE_URL,
    SUPPORTED_FORECAST_LEADS,
)


class GRIBBackendError(RuntimeError):
    """Base error for GRIB acquisition."""


class GRIBInventoryError(GRIBBackendError):
    """Invalid or incompatible GRIB inventory."""


class GRIBDownloadError(GRIBBackendError):
    """Invalid GRIB byte-range response."""


@dataclass(frozen=True)
class GRIBReference:
    run_at: pd.Timestamp
    forecast_hour: int
    valid_at: pd.Timestamp
    grib_url: str
    index_url: str


def build_grib_reference(run_at, forecast_hour):
    """Build a validated HRRR surface forecast reference."""

    try:
        run_at = pd.Timestamp(run_at)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GRIBBackendError("Invalid forecast initialization") from exc

    if pd.isna(run_at) or run_at.tzinfo is None:
        raise GRIBBackendError("Timezone-aware initialization required")

    run_at = run_at.tz_convert("UTC")

    if run_at.hour not in EXTENDED_CYCLE_HOURS or run_at != run_at.floor("h"):
        raise GRIBBackendError("Extended HRRR cycle required")

    if (
        not isinstance(forecast_hour, Integral)
        or isinstance(forecast_hour, bool)
        or forecast_hour not in SUPPORTED_FORECAST_LEADS
    ):
        raise GRIBBackendError("Unsupported forecast hour")

    forecast_hour = int(forecast_hour)

    date = run_at.strftime("%Y%m%d")
    cycle = run_at.strftime("%H")

    filename = f"hrrr.t{cycle}z.wrfsfcf{forecast_hour:02d}.grib2"

    grib_url = f"{GRIB_BASE_URL}/hrrr.{date}/conus/{filename}"

    return GRIBReference(
        run_at=run_at,
        forecast_hour=forecast_hour,
        valid_at=(run_at + pd.Timedelta(hours=forecast_hour)),
        grib_url=grib_url,
        index_url=grib_url + ".idx",
    )


def build_weather_patterns(forecast_hour):
    """Select the exact HRRR weather inventory definitions."""

    if forecast_hour not in SUPPORTED_FORECAST_LEADS:
        raise GRIBInventoryError("Unsupported forecast hour")

    instant = f"{forecast_hour} hour fcst"

    return {
        "temperature_2m": (f":TMP:2 m above ground:{instant}:"),
        "u_wind_10m": (f":UGRD:10 m above ground:{instant}:"),
        "v_wind_10m": (f":VGRD:10 m above ground:{instant}:"),
        "wind_gust": (f":GUST:surface:{instant}:"),
        "visibility": (f":VIS:surface:{instant}:"),
        "precipitation_1h": (
            f":APCP:surface:{forecast_hour - 1}-{forecast_hour} hour acc fcst:"
        ),
        "surface_cape": (f":CAPE:surface:{instant}:"),
        "composite_reflectivity": (f":REFC:entire atmosphere:{instant}:"),
        "cloud_ceiling_raw": (f":HGT:cloud ceiling:{instant}:"),
    }


def parse_grib_inventory(lines):
    """Parse inventory offsets and determine inclusive byte ranges."""

    records = []

    for line in lines:
        if not line.strip():
            continue

        parts = line.split(":", 2)

        if len(parts) != 3:
            raise GRIBInventoryError(f"Malformed inventory line: {line}")

        try:
            message = int(parts[0])
            start_byte = int(parts[1])
        except ValueError as exc:
            raise GRIBInventoryError(f"Invalid inventory offset: {line}") from exc

        if message <= 0 or start_byte < 0:
            raise GRIBInventoryError("Invalid message number or byte offset")

        records.append(
            {
                "message": message,
                "start_byte": start_byte,
                "inventory": line,
            }
        )

    if not records:
        raise GRIBInventoryError("Empty GRIB inventory")

    inventory = pd.DataFrame(records)

    if (
        not inventory["start_byte"].is_unique
        or not inventory["start_byte"].is_monotonic_increasing
        or not inventory["message"].is_unique
    ):
        raise GRIBInventoryError("Invalid inventory ordering or duplicate offsets")

    inventory["end_byte"] = (inventory["start_byte"].shift(-1) - 1).astype("Int64")

    return inventory


def select_weather_messages(inventory, forecast_hour):
    """Resolve exactly one GRIB message per weather variable."""

    patterns = build_weather_patterns(forecast_hour)

    selected = []

    for feature, pattern in patterns.items():
        matches = inventory.loc[
            inventory["inventory"].str.contains(
                pattern,
                regex=False,
                na=False,
            )
        ]

        if len(matches) != 1:
            raise GRIBInventoryError(
                f"{feature}: expected one message, found {len(matches)}"
            )

        record = matches.iloc[0]

        if pd.isna(record["end_byte"]):
            raise GRIBInventoryError(f"Unresolved final byte range: {feature}")

        start = int(record["start_byte"])
        end = int(record["end_byte"])

        if end < start:
            raise GRIBInventoryError(f"Invalid byte range: {feature}")

        selected.append(
            {
                "feature": feature,
                "message": int(record["message"]),
                "start_byte": start,
                "end_byte": end,
                "size_bytes": end - start + 1,
            }
        )

    return pd.DataFrame(selected)


def download_grib_range(
    url,
    start_byte,
    end_byte,
    *,
    opener=urlopen,
):
    """Download and validate one complete GRIB message."""

    if (
        not isinstance(start_byte, Integral)
        or not isinstance(end_byte, Integral)
        or start_byte < 0
        or end_byte < start_byte
    ):
        raise GRIBDownloadError("Invalid byte range")

    start_byte = int(start_byte)
    end_byte = int(end_byte)

    expected_size = end_byte - start_byte + 1

    request = Request(
        url,
        headers={
            "Range": f"bytes={start_byte}-{end_byte}",
            "Accept-Encoding": "identity",
        },
    )

    try:
        with opener(request, timeout=60) as response:
            if response.status != 206:
                raise GRIBDownloadError("Server did not return HTTP 206")

            content_range = response.headers.get("Content-Range", "")

            expected_pattern = (
                rf"bytes {start_byte}-{end_byte}/"
                r"(?:[0-9]+|\*)"
            )

            if not re.fullmatch(
                expected_pattern,
                content_range.strip(),
            ):
                raise GRIBDownloadError("Invalid Content-Range header")

            payload = response.read()

    except GRIBDownloadError:
        raise
    except OSError as exc:
        raise GRIBDownloadError("GRIB range request failed") from exc

    if len(payload) != expected_size:
        raise GRIBDownloadError("Downloaded payload size mismatch")

    if payload[:4] != b"GRIB" or payload[-4:] != b"7777":
        raise GRIBDownloadError("Invalid GRIB message markers")

    return payload
