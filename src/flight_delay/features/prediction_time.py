"""Construct scheduled-departure prediction timestamps."""

import numpy as np
import pandas as pd

AIRPORT_TIMEZONES = {
    "ATL": "America/New_York",
    "DFW": "America/Chicago",
    "ORD": "America/Chicago",
    "DEN": "America/Denver",
    "LAX": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
}

PREDICTION_HORIZON_HOURS = 24


def build_prediction_timestamps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Construct UTC timestamps from local scheduled departure.

    Ambiguous, nonexistent and unresolved midnight times
    are flagged instead of silently assigning a timestamp.
    """
    required = {"FlightDate", "Origin", "CRSDepTime"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing timestamp columns: {sorted(missing)}"
        )

    data = df.reset_index(drop=True)

    dates = pd.to_datetime(
        data["FlightDate"],
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Missing flight dates")

    if dates.dt.tz is not None:
        raise ValueError("FlightDate must be timezone-naive")

    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("FlightDate must contain dates only")

    # Determine timezone using origin airport.
    timezones = data["Origin"].map(AIRPORT_TIMEZONES)

    if timezones.isna().any():
        raise ValueError("Unsupported origin airport")

    # Validate HHMM scheduled departure times.
    hhmm = pd.to_numeric(
        data["CRSDepTime"],
        errors="raise",
    )

    values = hhmm.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    if not np.isfinite(values).all():
        raise ValueError("Missing or invalid departure time")

    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Departure time must be integer HHMM")

    hhmm = hhmm.astype("int64")

    hours = hhmm // 100
    minutes = hhmm % 100

    valid_regular = (
        hours.between(0, 23)
        & minutes.between(0, 59)
    )

    special_midnight = hhmm.eq(2400)

    if not (valid_regular | special_midnight).all():
        raise ValueError("Invalid HHMM departure time")

    # Do not silently assign a date to special 2400 values.
    minute_offsets = (
        hours * 60 + minutes
    ).where(valid_regular, 0)

    local_departure = (
        dates
        + pd.to_timedelta(minute_offsets, unit="m")
    )

    local_departure = local_departure.mask(
        special_midnight
    )

    departure_utc = pd.Series(
        pd.NaT,
        index=data.index,
        dtype="datetime64[ns, UTC]",
    )

    status = pd.Series(
        "ok",
        index=data.index,
        dtype="string",
    )

    status.loc[special_midnight] = "unresolved_2400"

    # Vectorized timezone conversion by airport timezone.
    for zone in AIRPORT_TIMEZONES.values():
        mask = timezones.eq(zone) & valid_regular

        if not mask.any():
            continue

        localized = (
            local_departure.loc[mask]
            .dt.tz_localize(
                zone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
        )

        utc = localized.dt.tz_convert("UTC")

        departure_utc.loc[mask] = utc

        invalid_index = utc.index[utc.isna()]

        status.loc[invalid_index] = (
            "unresolved_dst"
        )

    # Subtract 24 elapsed hours after conversion to UTC.
    prediction_utc = (
        departure_utc
        - pd.Timedelta(
            hours=PREDICTION_HORIZON_HOURS
        )
    )

    result = pd.DataFrame({
        "origin_timezone": timezones,
        "scheduled_departure_utc": departure_utc,
        "prediction_at_utc": prediction_utc,
        "timestamp_status": status,
    })

    result.index = df.index

    return result
