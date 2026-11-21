"""Monthly historical delay count aggregation."""

import pandas as pd

REQUIRED_COLUMNS = (
    "FlightDate",
    "Origin",
    "Reporting_Airline",
    "DepDel15",
    "ArrDel15",
    "Cancelled",
    "Diverted",
)

COUNT_COLUMNS = (
    "flight_count",
    "dep_delay_count",
    "arr_delay_count",
)


def aggregate_monthly_delay_counts(
    df: pd.DataFrame,
    report_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate one month into origin and airline-origin counts.

    Only completed, non-diverted flights with valid delay labels
    may contribute to these historical statistics.

    Availability filtering is handled separately during
    point-in-time feature lookup.
    """
    period = pd.Period(report_month, freq="M")

    if str(period) != report_month:
        raise ValueError("Invalid reporting month format")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Historical source is empty")

    data = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    if data.isna().any().any():
        raise ValueError("Missing values in historical source")

    # Ensure all records belong to the requested month.
    dates = pd.to_datetime(
        data["FlightDate"],
        errors="raise",
    )

    if not dates.dt.to_period("M").eq(period).all():
        raise ValueError("Flight dates do not match reporting month")

    # Enforce the existing modeling cohort.
    if not data["Cancelled"].eq(0).all():
        raise ValueError("Cancelled flights found")

    if not data["Diverted"].eq(0).all():
        raise ValueError("Diverted flights found")

    # Validate binary labels.
    for target in ["DepDel15", "ArrDel15"]:
        if not data[target].isin([0, 1]).all():
            raise ValueError(f"Invalid binary label: {target}")

        data[target] = data[target].astype("int64")

    # Origin-level monthly statistics.
    origin_stats = (
        data.groupby(
            ["Origin"],
            observed=True,
            sort=True,
        )
        .agg(
            flight_count=("DepDel15", "size"),
            dep_delay_count=("DepDel15", "sum"),
            arr_delay_count=("ArrDel15", "sum"),
        )
        .reset_index()
    )

    # Airline-origin-level monthly statistics.
    airline_origin_stats = (
        data.groupby(
            ["Origin", "Reporting_Airline"],
            observed=True,
            sort=True,
        )
        .agg(
            flight_count=("DepDel15", "size"),
            dep_delay_count=("DepDel15", "sum"),
            arr_delay_count=("ArrDel15", "sum"),
        )
        .reset_index()
    )

    # Preserve source reporting month.
    origin_stats.insert(0, "report_month", report_month)

    airline_origin_stats.insert(
        0,
        "report_month",
        report_month,
    )

    # Reconcile aggregate counts against source records.
    for stats in [origin_stats, airline_origin_stats]:

        if stats["flight_count"].sum() != len(data):
            raise ValueError("Aggregated flight count mismatch")

        if stats["dep_delay_count"].sum() != data["DepDel15"].sum():
            raise ValueError("Departure delay count mismatch")

        if stats["arr_delay_count"].sum() != data["ArrDel15"].sum():
            raise ValueError("Arrival delay count mismatch")

    return origin_stats, airline_origin_stats


def build_rate_snapshots(
    monthly_stats: pd.DataFrame,
    windows: pd.DataFrame,
    group_keys: list[str],
) -> pd.DataFrame:
    """Build weighted delay-rate snapshots for complete windows."""

    required = [
        "report_month",
        *group_keys,
        *COUNT_COLUMNS,
    ]

    missing = set(required) - set(monthly_stats.columns)

    if missing:
        raise ValueError(
            f"Missing monthly columns: {sorted(missing)}"
        )

    if monthly_stats[required].isna().any().any():
        raise ValueError("Missing monthly statistics")

    if monthly_stats.duplicated(
        ["report_month", *group_keys]
    ).any():
        raise ValueError("Duplicate monthly aggregation keys")

    counts = monthly_stats[list(COUNT_COLUMNS)]

    if (
        not counts.ge(0).all().all()
        or not counts.mod(1).eq(0).all().all()
        or not monthly_stats["flight_count"].gt(0).all()
    ):
        raise ValueError("Invalid monthly counts")

    for column in ["dep_delay_count", "arr_delay_count"]:
        if not monthly_stats[column].le(
            monthly_stats["flight_count"]
        ).all():
            raise ValueError("Delay count exceeds flight count")

    if windows["window_id"].duplicated().any():
        raise ValueError("Duplicate window IDs")

    tables = []

    complete_windows = windows.loc[
        windows["window_complete"]
    ]

    for window in complete_windows.itertuples(index=False):

        source_months = tuple(window.source_months)

        if (
            len(set(source_months))
            != window.history_month_count
            or len(source_months)
            != window.history_month_count
        ):
            raise ValueError("Invalid source month window")

        eligible_stats = monthly_stats.loc[
            monthly_stats["report_month"].isin(
                source_months
            )
        ]

        if set(eligible_stats["report_month"]) != set(
            source_months
        ):
            raise ValueError(
                f"Incomplete source coverage: {window.window_id}"
            )

        snapshot = (
            eligible_stats
            .groupby(
                group_keys,
                observed=True,
                sort=True,
            )[list(COUNT_COLUMNS)]
            .sum()
            .reset_index()
        )

        snapshot.insert(
            0, "window_id", window.window_id
        )

        snapshot["dep_delay_rate"] = (
            snapshot["dep_delay_count"]
            / snapshot["flight_count"]
        )

        snapshot["arr_delay_rate"] = (
            snapshot["arr_delay_count"]
            / snapshot["flight_count"]
        )

        tables.append(snapshot)

    if not tables:
        return pd.DataFrame(columns=[
            "window_id",
            *group_keys,
            *COUNT_COLUMNS,
            "dep_delay_rate",
            "arr_delay_rate",
        ])

    return pd.concat(tables, ignore_index=True)
