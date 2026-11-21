"""Point-in-time historical delay feature generation."""

import numpy as np
import pandas as pd

from flight_delay.features.historical_lookup import (
    match_history_windows,
)

DEFAULT_MIN_AIRLINE_FLIGHTS = 100

HISTORICAL_FEATURE_COLUMNS = (
    "origin_dep_delay_rate_3m",
    "origin_outbound_arr_delay_rate_3m",
    "airline_origin_dep_delay_rate_3m",
    "airline_origin_arr_delay_rate_3m",
)


def prepare_snapshot(
    snapshot: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Validate and prepare a historical snapshot for joining."""

    metrics = [
        "flight_count",
        "dep_delay_rate",
        "arr_delay_rate",
    ]

    required = keys + metrics

    missing = set(required) - set(snapshot.columns)

    if missing:
        raise ValueError(
            f"Missing snapshot columns: {sorted(missing)}"
        )

    result = snapshot[required].copy()

    if result.isna().any().any():
        raise ValueError("Missing snapshot values")

    for key in keys:
        result[key] = result[key].astype("string")

    if result.duplicated(keys).any():
        raise ValueError("Duplicate snapshot keys")

    counts = result["flight_count"].to_numpy(dtype=float)

    if not (
        np.isfinite(counts).all()
        and (counts > 0).all()
        and np.equal(counts, np.floor(counts)).all()
    ):
        raise ValueError("Invalid snapshot flight counts")

    for column in ["dep_delay_rate", "arr_delay_rate"]:
        if not result[column].between(0, 1).all():
            raise ValueError(f"Invalid rates: {column}")

    return result.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in metrics
        }
    )


def build_historical_features(
    flights: pd.DataFrame,
    prediction_at_utc: pd.Series,
    windows: pd.DataFrame,
    origin_snapshots: pd.DataFrame,
    airline_snapshots: pd.DataFrame,
    min_airline_flights: int = DEFAULT_MIN_AIRLINE_FLIGHTS,
) -> pd.DataFrame:
    """Join historical features without changing flight row order."""

    required = {"Origin", "Reporting_Airline"}

    if not required.issubset(flights.columns):
        raise ValueError("Missing flight grouping columns")

    if not flights.index.equals(prediction_at_utc.index):
        raise ValueError("Flight and timestamp indices differ")

    if (
        isinstance(min_airline_flights, bool)
        or not isinstance(min_airline_flights, int)
        or min_airline_flights < 1
    ):
        raise ValueError("Invalid minimum flight count")

    matched = match_history_windows(
        prediction_at_utc,
        windows,
    ).reset_index(drop=True)

    complete = matched["history_status"].eq("ok")

    # Only complete, eligible windows can contribute features.
    keys = flights[
        ["Origin", "Reporting_Airline"]
    ].reset_index(drop=True).copy()

    keys["window_id"] = matched["window_id"].where(complete)
    keys["_row_id"] = np.arange(len(keys))

    for column in ["window_id", "Origin", "Reporting_Airline"]:
        keys[column] = keys[column].astype("string")

    origin_keys = ["window_id", "Origin"]
    airline_keys = [
        "window_id",
        "Origin",
        "Reporting_Airline",
    ]

    origin = prepare_snapshot(
        origin_snapshots,
        origin_keys,
        prefix="origin",
    )

    airline = prepare_snapshot(
        airline_snapshots,
        airline_keys,
        prefix="airline_origin",
    )

    joined = (
        keys
        .merge(
            origin,
            on=origin_keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        .merge(
            airline,
            on=airline_keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    if len(joined) != len(flights):
        raise ValueError("Historical join changed row count")

    origin_found = joined["origin_flight_count"].notna()

    airline_found = joined[
        "airline_origin_flight_count"
    ].notna()

    sufficient_history = (
        joined["airline_origin_flight_count"]
        .ge(min_airline_flights)
        .fillna(False)
    )

    # An airline-origin group must belong to its origin group.
    inconsistent = airline_found & (
        ~origin_found
        | joined["airline_origin_flight_count"].gt(
            joined["origin_flight_count"]
        )
    )

    if inconsistent.any():
        raise ValueError("Inconsistent historical snapshots")

    # Preserve window metadata for auditing.
    result = matched[
        [
            "window_id",
            "window_available_at_utc",
            "source_months",
            "history_status",
        ]
    ].copy()

    result["origin_history_count"] = joined[
        "origin_flight_count"
    ]

    result["airline_origin_history_count"] = joined[
        "airline_origin_flight_count"
    ]

    result["origin_dep_delay_rate_3m"] = joined[
        "origin_dep_delay_rate"
    ]

    result["origin_outbound_arr_delay_rate_3m"] = joined[
        "origin_arr_delay_rate"
    ]

    # Fall back to origin rates for small or unknown groups.
    result["airline_origin_dep_delay_rate_3m"] = joined[
        "airline_origin_dep_delay_rate"
    ].where(
        sufficient_history,
        joined["origin_dep_delay_rate"],
    )

    result["airline_origin_arr_delay_rate_3m"] = joined[
        "airline_origin_arr_delay_rate"
    ].where(
        sufficient_history,
        joined["origin_arr_delay_rate"],
    )

    status = pd.Series(
        "direct",
        index=result.index,
        dtype="string",
    )

    status.loc[
        airline_found & ~sufficient_history
    ] = "fallback_low_sample"

    status.loc[
        ~airline_found
    ] = "fallback_unknown_group"

    status.loc[
        ~origin_found
    ] = "missing_origin_history"

    status.loc[
        ~complete
    ] = "history_unavailable"

    result["airline_history_status"] = status

    result.index = flights.index

    return result
