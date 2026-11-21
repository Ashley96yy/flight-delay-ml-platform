
"""Source availability policies for historical BTS features."""

import re
from collections.abc import Iterable

import pandas as pd

DEFAULT_PROXY_LAG_DAYS = 120

SOURCE_PREFIX = (
    "On_Time_Reporting_Carrier_"
    "On_Time_Performance_1987_present"
)


def build_proxy_availability_calendar(
    months: Iterable[str],
    lag_days: int = DEFAULT_PROXY_LAG_DAYS,
) -> pd.DataFrame:
    """Build an assumed BTS availability calendar.

    This policy is for retrospective research only.
    It does NOT represent verified publication dates.
    """
    if (
        isinstance(lag_days, bool)
        or not isinstance(lag_days, int)
        or lag_days < 0
    ):
        raise ValueError("lag_days must be a non-negative integer")

    rows = []
    seen = set()

    for month in months:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ValueError(f"Invalid month format: {month}")

        period = pd.Period(month, freq="M")

        if str(period) != month:
            raise ValueError(f"Invalid reporting month: {month}")

        if month in seen:
            raise ValueError(f"Duplicate reporting month: {month}")

        seen.add(month)

        # First instant after the reporting month ends.
        next_month_start = (
            (period + 1)
            .start_time
            .tz_localize("UTC")
        )

        assumed_available_at = (
            next_month_start
            + pd.Timedelta(days=lag_days)
        )

        rows.append({
            "report_month": month,
            "source_filename": (
                f"{SOURCE_PREFIX}_{period.year}_"
                f"{period.month}.zip"
            ),
            "assumed_available_at_utc": assumed_available_at,
            "availability_policy": "research_proxy",
            "availability_evidence": "assumed_not_verified",
            "lag_days": lag_days,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("report_month")
        .reset_index(drop=True)
    )


def eligible_source_months(
    calendar: pd.DataFrame,
    prediction_at_utc: str | pd.Timestamp,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """Select recent source months available under the proxy policy."""
    cutoff = pd.Timestamp(prediction_at_utc)

    if cutoff.tzinfo is None:
        raise ValueError(
            "prediction_at_utc must be timezone-aware"
        )

    if lookback_months < 1:
        raise ValueError(
            "lookback_months must be positive"
        )

    cutoff = cutoff.tz_convert("UTC")

    available = calendar.loc[
        calendar["assumed_available_at_utc"] <= cutoff
    ].copy()

    return (
        available
        .sort_values("report_month")
        .tail(lookback_months)
        .reset_index(drop=True)
    )
