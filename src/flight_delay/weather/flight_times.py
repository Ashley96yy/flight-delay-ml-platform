"""Convert BTS scheduled local departure times to UTC.

No rows are silently dropped. Ambiguous timestamps are flagged
for review rather than guessed.
"""

from dataclasses import dataclass

import pandas as pd

AIRPORT_TIMEZONES = {
    "ATL": "America/New_York",
    "DFW": "America/Chicago",
    "DEN": "America/Denver",
    "ORD": "America/Chicago",
    "LAX": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
}


class BTSDepartureTimeError(ValueError):
    """Invalid BTS departure-time conversion input."""


@dataclass
class BTSDepartureTimeResult:
    flights: pd.DataFrame
    summary: dict


def convert_bts_departure_to_utc(
    flights: pd.DataFrame,
) -> BTSDepartureTimeResult:
    """Convert FlightDate + CRSDepTime at the origin to UTC."""

    required = {"Origin", "FlightDate", "CRSDepTime"}

    if not isinstance(flights, pd.DataFrame):
        raise BTSDepartureTimeError("Input must be a DataFrame")

    missing = required - set(flights.columns)

    if missing:
        raise BTSDepartureTimeError(f"Missing BTS columns: {sorted(missing)}")

    result = flights.copy(deep=True).reset_index(drop=True)

    dates = pd.to_datetime(
        result["FlightDate"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    clock = pd.to_numeric(
        result["CRSDepTime"],
        errors="coerce",
    )

    origins = result["Origin"].astype("string").str.strip().str.upper()

    status = pd.Series(
        "valid",
        index=result.index,
        dtype="object",
    )

    # Midnight represented as 2400 requires a separate review.
    # Do not assume a calendar-day rollover without verification.
    midnight_2400 = clock.eq(2400).fillna(False)

    status.loc[midnight_2400] = "midnight_2400_review"

    valid_clock = (
        clock.notna()
        & clock.between(0, 2359)
        & clock.mod(1).eq(0)
        & clock.mod(100).lt(60)
    ).fillna(False)

    status.loc[status.eq("valid") & ~valid_clock] = "invalid_time"

    status.loc[status.eq("valid") & dates.isna()] = "invalid_date"

    status.loc[status.eq("valid") & ~origins.isin(AIRPORT_TIMEZONES)] = (
        "unsupported_airport"
    )

    utc = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    # Vectorized conversion for each origin time zone.

    for airport, timezone in AIRPORT_TIMEZONES.items():
        mask = status.eq("valid") & origins.eq(airport).fillna(False)

        indices = result.index[mask]

        if len(indices) == 0:
            continue

        airport_clock = clock.loc[indices].astype(int)

        hours = airport_clock // 100
        minutes = airport_clock % 100

        local_naive = (
            dates.loc[indices].dt.normalize()
            + pd.to_timedelta(hours, unit="h")
            + pd.to_timedelta(minutes, unit="m")
        )

        local_time = local_naive.dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )

        converted = local_time.dt.tz_convert("UTC")

        unresolved = converted.isna()

        if unresolved.any():
            status.loc[converted.index[unresolved]] = "dst_unresolvable"

        utc.loc[indices] = converted

    result["scheduled_departure_utc"] = utc
    result["departure_time_status"] = status

    valid_count = int(status.eq("valid").sum())

    summary = {
        "total_rows": len(result),
        "valid_rows": valid_count,
        "unresolved_rows": len(result) - valid_count,
        "valid_rate": (valid_count / len(result) if len(result) else 0.0),
        "status_counts": status.value_counts().to_dict(),
    }

    return BTSDepartureTimeResult(
        flights=result,
        summary=summary,
    )
