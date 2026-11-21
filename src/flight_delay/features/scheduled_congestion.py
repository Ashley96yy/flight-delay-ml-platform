
"""Retrospective scheduled airport traffic features."""

import numpy as np
import pandas as pd

ORIGIN_AIRPORTS = (
    "ATL", "DFW", "DEN", "ORD", "LAX", "SFO"
)

RAW_SCHEDULE_COLUMNS = (
    "FlightDate",
    "Origin",
    "Reporting_Airline",
    "CRSDepTime",
)

ORIGIN_TRAFFIC_KEYS = [
    "FlightDate",
    "Origin",
    "Departure_Hour",
]

AIRLINE_TRAFFIC_KEYS = [
    *ORIGIN_TRAFFIC_KEYS,
    "Reporting_Airline",
]

DEPARTURE_TRAFFIC_FEATURES = (
    "origin_scheduled_departures_1h",
    "airline_origin_scheduled_departures_1h",
)


def prepare_departure_schedule(
    schedule: pd.DataFrame,
    month: str,
) -> tuple[pd.DataFrame, dict]:
    """Prepare raw scheduled departures without outcome filtering."""

    period = pd.Period(month, freq="M")

    if str(period) != month:
        raise ValueError("Invalid reporting month")

    missing = set(RAW_SCHEDULE_COLUMNS) - set(
        schedule.columns
    )

    if missing:
        raise ValueError(
            f"Missing scheduled columns: {sorted(missing)}"
        )

    if schedule["Origin"].isna().any():
        raise ValueError("Missing origin airport")

    data = schedule.loc[
        schedule["Origin"].isin(ORIGIN_AIRPORTS),
        list(RAW_SCHEDULE_COLUMNS),
    ].copy()

    if data.empty:
        raise ValueError("No in-scope scheduled departures")

    dates = pd.to_datetime(
        data["FlightDate"],
        errors="coerce",
    )

    valid_dates = (
        dates.notna()
        & dates.dt.to_period("M").eq(period)
        & dates.eq(dates.dt.normalize())
    )

    numeric = pd.to_numeric(
        data["CRSDepTime"],
        errors="coerce",
    )

    unresolved_2400 = numeric.eq(2400).fillna(False)

    valid_clock = (
        numeric.notna()
        & numeric.between(0, 2359)
        & numeric.mod(100).between(0, 59)
        & numeric.mod(1).eq(0)
    ).fillna(False)

    airline = data["Reporting_Airline"].astype("string")

    valid_airline = (
        airline.notna()
        & airline.str.strip().ne("")
    ).fillna(False)

    valid = valid_dates & valid_clock & valid_airline

    audit = {
        "Month": month,
        "ScopedRecords": len(data),
        "InvalidDates": int((~valid_dates).sum()),
        "MissingOrInvalidTimes": int(
            (~valid_clock & ~unresolved_2400).sum()
        ),
        "Unresolved2400": int(unresolved_2400.sum()),
        "MissingAirline": int((~valid_airline).sum()),
        "IncludedRecords": int(valid.sum()),
        "ExcludedRecords": int((~valid).sum()),
    }

    data = data.loc[valid].copy()

    data["FlightDate"] = dates.loc[valid].dt.normalize()

    data["Origin"] = data["Origin"].astype("string")
    data["Reporting_Airline"] = airline.loc[valid]

    data["Departure_Hour"] = (
        numeric.loc[valid].astype("int64") // 100
    ).astype("int8")

    return data, audit


def aggregate_departure_traffic(
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate scheduled departures by local clock hour."""

    if schedule.empty:
        raise ValueError("Empty prepared schedule")

    if schedule[AIRLINE_TRAFFIC_KEYS].isna().any().any():
        raise ValueError("Missing traffic aggregation keys")

    origin_feature, airline_feature = (
        DEPARTURE_TRAFFIC_FEATURES
    )

    origin_traffic = (
        schedule.groupby(
            ORIGIN_TRAFFIC_KEYS,
            observed=True,
            sort=True,
        )
        .size()
        .reset_index(name=origin_feature)
    )

    airline_traffic = (
        schedule.groupby(
            AIRLINE_TRAFFIC_KEYS,
            observed=True,
            sort=True,
        )
        .size()
        .reset_index(name=airline_feature)
    )

    for table, feature in [
        (origin_traffic, origin_feature),
        (airline_traffic, airline_feature),
    ]:
        if table[feature].sum() != len(schedule):
            raise ValueError("Traffic count mismatch")

    reaggregated = (
        airline_traffic.groupby(ORIGIN_TRAFFIC_KEYS)[
            airline_feature
        ]
        .sum()
        .sort_index()
    )

    expected = (
        origin_traffic.set_index(ORIGIN_TRAFFIC_KEYS)[
            origin_feature
        ]
        .sort_index()
    )

    pd.testing.assert_series_equal(
        reaggregated,
        expected,
        check_names=False,
        check_dtype=False,
    )

    return origin_traffic, airline_traffic


def build_departure_traffic_features(
    flights: pd.DataFrame,
    origin_traffic: pd.DataFrame,
    airline_traffic: pd.DataFrame,
    month: str,
) -> pd.DataFrame:
    """Attach scheduled traffic counts to prediction flights."""

    keys, audit = prepare_departure_schedule(
        flights,
        month,
    )

    if len(keys) != len(flights):
        raise ValueError(
            "Target flights contain unresolved schedule inputs"
        )

    if audit["ExcludedRecords"] != 0:
        raise ValueError("Invalid target schedule")

    keys = keys.reset_index(drop=True)

    keys["_row_id"] = np.arange(len(keys))

    joined = (
        keys.merge(
            origin_traffic,
            on=ORIGIN_TRAFFIC_KEYS,
            how="left",
            validate="many_to_one",
        )
        .merge(
            airline_traffic,
            on=AIRLINE_TRAFFIC_KEYS,
            how="left",
            validate="many_to_one",
        )
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    if len(joined) != len(flights):
        raise ValueError("Traffic feature row mismatch")

    np.testing.assert_array_equal(
        joined["_row_id"].to_numpy(),
        np.arange(len(flights)),
    )

    result = joined[
        list(DEPARTURE_TRAFFIC_FEATURES)
    ].copy()

    if result.isna().any().any():
        raise ValueError("Missing scheduled traffic match")

    if not result.ge(1).all().all():
        raise ValueError("Invalid traffic counts")

    if not result.mod(1).eq(0).all().all():
        raise ValueError("Non-integer traffic counts")

    origin_feature, airline_feature = (
        DEPARTURE_TRAFFIC_FEATURES
    )

    if result[airline_feature].gt(
        result[origin_feature]
    ).any():
        raise ValueError("Inconsistent traffic counts")

    # Restore original flight indices and ordering.
    result.index = flights.index

    return result.astype("int64")
