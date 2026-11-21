"""Point-in-time historical feature window construction."""

import pandas as pd

from flight_delay.features.availability import (
    eligible_source_months,
)


def build_history_windows(
    calendar: pd.DataFrame,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """Build historical windows at source availability boundaries."""

    required = {
        "report_month",
        "assumed_available_at_utc",
    }

    if not required.issubset(calendar.columns):
        raise ValueError("Missing availability calendar columns")

    if calendar.empty:
        raise ValueError("Availability calendar is empty")

    if lookback_months < 1:
        raise ValueError("Lookback must be positive")

    ordered = calendar.sort_values(
        "assumed_available_at_utc"
    ).reset_index(drop=True)

    availability = ordered["assumed_available_at_utc"]

    if not isinstance(
        availability.dtype, pd.DatetimeTZDtype
    ):
        raise ValueError("Availability must be timezone-aware")

    if availability.isna().any():
        raise ValueError("Missing availability timestamps")

    if ordered["report_month"].duplicated().any():
        raise ValueError("Duplicate source months")

    if availability.duplicated().any():
        raise ValueError("Duplicate availability boundaries")

    if not ordered["report_month"].is_monotonic_increasing:
        raise ValueError("Source availability order is inconsistent")

    records = []

    for row in ordered.itertuples(index=False):

        eligible = eligible_source_months(
            ordered,
            prediction_at_utc=row.assumed_available_at_utc,
            lookback_months=lookback_months,
        )

        months = tuple(eligible["report_month"])

        records.append({
            "window_id": row.report_month,
            "window_available_at_utc":
                row.assumed_available_at_utc,
            "source_months": months,
            "history_month_count": len(months),
            "window_complete":
                len(months) == lookback_months,
        })

    return pd.DataFrame(records)


def match_history_windows(
    prediction_at_utc: pd.Series,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Match each prediction to its latest available window."""

    if not isinstance(
        prediction_at_utc.dtype, pd.DatetimeTZDtype
    ):
        raise ValueError("Prediction timestamps must be timezone-aware")

    if windows.empty:
        raise ValueError("Historical windows are empty")

    ordered = windows.sort_values(
        "window_available_at_utc"
    ).reset_index(drop=True)

    boundaries = pd.DatetimeIndex(
        ordered["window_available_at_utc"]
    )

    if boundaries.tz is None:
        raise ValueError("Window boundaries must be timezone-aware")

    if boundaries.hasnans or boundaries.has_duplicates:
        raise ValueError("Invalid window boundaries")

    cutoffs = pd.DatetimeIndex(prediction_at_utc)

    # Find the latest boundary <= prediction time.
    positions = boundaries.searchsorted(
        cutoffs,
        side="right",
    ) - 1

    unresolved = prediction_at_utc.isna().to_numpy()

    # Prevent missing prediction times from matching a window.
    positions[unresolved] = -1

    result = ordered.reindex(
        positions
    ).reset_index(drop=True)

    status = pd.Series(
        "ok",
        index=result.index,
        dtype="string",
    )

    status.loc[positions < 0] = "no_history"

    incomplete = (
        (positions >= 0)
        & ~result["window_complete"].eq(True)
    )

    status.loc[incomplete] = "insufficient_months"
    status.loc[unresolved] = "unresolved_timestamp"

    result["history_status"] = status

    result.index = prediction_at_utc.index

    return result
