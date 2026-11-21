"""Seasonally matched historical delay features."""

import numpy as np
import pandas as pd

from flight_delay.features.availability import (
    build_proxy_availability_calendar,
)
from flight_delay.features.historical_aggregation import (
    aggregate_monthly_delay_counts,
)
from flight_delay.features.prediction_time import (
    build_prediction_timestamps,
)

SEASONAL_FEATURE_COLUMNS = (
    "origin_dep_delay_rate_ly",
    "origin_outbound_arr_delay_rate_ly",
    "airline_origin_dep_delay_rate_ly",
    "airline_origin_arr_delay_rate_ly",
)


def build_seasonal_history_features(
    flights: pd.DataFrame,
    source: pd.DataFrame,
    target_month: str,
    source_month: str,
    lag_days: int = 120,
    min_airline_flights: int = 100,
) -> pd.DataFrame:
    """Construct prior-year same-month delay statistics."""

    target_period = pd.Period(target_month, freq="M")
    source_period = pd.Period(source_month, freq="M")

    if (
        str(target_period) != target_month
        or str(source_period) != source_month
    ):
        raise ValueError("Invalid month format")

    if source_period != target_period - 12:
        raise ValueError(
            "Expected previous-year same-month source"
        )

    if (
        isinstance(min_airline_flights, bool)
        or not isinstance(min_airline_flights, int)
        or min_airline_flights < 1
    ):
        raise ValueError("Invalid minimum flight count")

    if flights.empty:
        raise ValueError("Empty target flights")

    dates = pd.to_datetime(flights["FlightDate"])

    if not dates.dt.to_period("M").eq(target_period).all():
        raise ValueError("Target month mismatch")

    # Validate prediction-time eligibility.
    calendar = build_proxy_availability_calendar(
        [source_month],
        lag_days=lag_days,
    )

    available_at = calendar.iloc[0][
        "assumed_available_at_utc"
    ]

    timestamps = build_prediction_timestamps(flights)

    resolved = timestamps["timestamp_status"].eq("ok")

    if not timestamps.loc[
        resolved, "prediction_at_utc"
    ].ge(available_at).all():
        raise ValueError("Historical source not yet available")

    # Aggregate historical source labels only.
    origin, airline = aggregate_monthly_delay_counts(
        source,
        report_month=source_month,
    )

    origin_dep, origin_arr, airline_dep, airline_arr = (
        SEASONAL_FEATURE_COLUMNS
    )

    origin[origin_dep] = (
        origin["dep_delay_count"]
        / origin["flight_count"]
    )

    origin[origin_arr] = (
        origin["arr_delay_count"]
        / origin["flight_count"]
    )

    airline[airline_dep] = (
        airline["dep_delay_count"]
        / airline["flight_count"]
    )

    airline[airline_arr] = (
        airline["arr_delay_count"]
        / airline["flight_count"]
    )

    keys = flights[
        ["Origin", "Reporting_Airline"]
    ].copy()

    keys["_row_id"] = np.arange(len(keys))

    result = (
        keys.merge(
            origin[["Origin", origin_dep, origin_arr]],
            on="Origin",
            how="left",
            validate="many_to_one",
        )
        .merge(
            airline[
                [
                    "Origin",
                    "Reporting_Airline",
                    "flight_count",
                    airline_dep,
                    airline_arr,
                ]
            ].rename(
                columns={
                    "flight_count": "history_flight_count"
                }
            ),
            on=["Origin", "Reporting_Airline"],
            how="left",
            validate="many_to_one",
        )
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    np.testing.assert_array_equal(
        result["_row_id"].to_numpy(),
        np.arange(len(flights)),
    )

    unknown = result["history_flight_count"].isna()

    low_sample = (
        result["history_flight_count"]
        .lt(min_airline_flights)
        .fillna(False)
    )

    result["history_status"] = "direct"

    result.loc[
        low_sample, "history_status"
    ] = "fallback_low_sample"

    result.loc[
        unknown, "history_status"
    ] = "fallback_unknown_group"

    fallback = unknown | low_sample

    result.loc[fallback, airline_dep] = (
        result.loc[fallback, origin_dep]
    )

    result.loc[fallback, airline_arr] = (
        result.loc[fallback, origin_arr]
    )

    # Preserve unresolved prediction timestamps as missing.
    unresolved = ~resolved.to_numpy()

    result.loc[
        unresolved,
        list(SEASONAL_FEATURE_COLUMNS),
    ] = np.nan

    result.loc[
        unresolved, "history_status"
    ] = "history_unavailable"

    result.loc[
        unresolved, "history_flight_count"
    ] = np.nan

    valid_features = result.loc[
        resolved.to_numpy(),
        list(SEASONAL_FEATURE_COLUMNS),
    ]

    if valid_features.isna().any().any():
        raise ValueError("Missing eligible historical features")

    if not (
        valid_features.ge(0).all().all()
        and valid_features.le(1).all().all()
    ):
        raise ValueError("Invalid historical probabilities")

    result["source_month"] = source_month
    result["assumed_available_at_utc"] = available_at

    result.index = flights.index

    return result[
        [
            *SEASONAL_FEATURE_COLUMNS,
            "history_flight_count",
            "history_status",
            "source_month",
            "assumed_available_at_utc",
        ]
    ]