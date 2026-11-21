"""Advanced scheduled flight features."""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

INTERACTION_FEATURE_COLUMNS = (
    "Route",
    "Airline_Route",
    "Departure_Hour",
    "Arrival_Hour",
)

HOLIDAY_FEATURE_COLUMNS = (
    "is_us_federal_holiday",
    "days_since_us_federal_holiday",
    "days_until_us_federal_holiday",
)

ADVANCED_FEATURE_COLUMNS = (
    INTERACTION_FEATURE_COLUMNS
    + HOLIDAY_FEATURE_COLUMNS
)


def hhmm_to_hour(values: pd.Series) -> pd.Series:
    """Convert scheduled HHMM values to local clock hours."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    )

    array = numeric.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    if not np.isfinite(array).all():
        raise ValueError("Missing scheduled time")

    if not np.equal(array, np.floor(array)).all():
        raise ValueError("Non-integer scheduled time")

    if not ((array >= 0) & (array <= 2400)).all():
        raise ValueError("Invalid scheduled time")

    numeric = numeric.astype("int64")

    hours = numeric // 100
    minutes = numeric % 100

    valid = (
        (hours.between(0, 23) & minutes.between(0, 59))
        | numeric.eq(2400)
    )

    if not valid.all():
        raise ValueError("Invalid HHMM scheduled time")

    # 2400 represents midnight for clock-hour features.
    return (hours % 24).astype("int8")


def build_holiday_features(
    flight_dates: pd.Series,
) -> pd.DataFrame:
    """Construct deterministic US federal holiday features."""

    dates = pd.to_datetime(
        flight_dates,
        errors="raise",
    )

    if dates.empty or dates.isna().any():
        raise ValueError("Missing flight dates")

    if dates.dt.tz is not None:
        raise ValueError("Expected local calendar dates")

    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected date-only values")

    calendar = USFederalHolidayCalendar()

    holidays = calendar.holidays(
        start=dates.min() - pd.Timedelta(days=366),
        end=dates.max() + pd.Timedelta(days=366),
    )

    holiday_days = holidays.to_numpy(
        dtype="datetime64[D]"
    )

    flight_days = dates.to_numpy(
        dtype="datetime64[D]"
    )

    previous_idx = np.searchsorted(
        holiday_days,
        flight_days,
        side="right",
    ) - 1

    next_idx = np.searchsorted(
        holiday_days,
        flight_days,
        side="left",
    )

    since_previous = (
        flight_days - holiday_days[previous_idx]
    ).astype("int64")

    until_next = (
        holiday_days[next_idx] - flight_days
    ).astype("int64")

    return pd.DataFrame({
        "is_us_federal_holiday": (
            since_previous == 0
        ).astype("int8"),

        "days_since_us_federal_holiday":
            since_previous.astype("int16"),

        "days_until_us_federal_holiday":
            until_next.astype("int16"),

    }, index=flight_dates.index)


def build_advanced_scheduled_features(
    flights: pd.DataFrame,
) -> pd.DataFrame:
    """Build scheduled interactions and calendar features."""

    required = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "CRSDepTime",
        "CRSArrTime",
    ]

    missing = set(required) - set(flights.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if flights.empty:
        raise ValueError("Empty flight dataset")

    if flights[required].isna().any().any():
        raise ValueError("Missing scheduled feature inputs")

    origin = flights["Origin"].astype("string")
    dest = flights["Dest"].astype("string")
    airline = flights["Reporting_Airline"].astype("string")

    result = pd.DataFrame(
        index=flights.index
    )

    result["Route"] = origin + "_" + dest

    result["Airline_Route"] = (
        airline + "_" + origin + "_" + dest
    )

    result["Departure_Hour"] = hhmm_to_hour(
        flights["CRSDepTime"]
    )

    result["Arrival_Hour"] = hhmm_to_hour(
        flights["CRSArrTime"]
    )

    holidays = build_holiday_features(
        flights["FlightDate"]
    )

    result = pd.concat(
        [result, holidays],
        axis=1,
    )

    return result[list(ADVANCED_FEATURE_COLUMNS)]
