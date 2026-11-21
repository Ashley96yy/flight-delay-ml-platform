"""Point-in-time HRRR forecast request planning."""

from dataclasses import dataclass

import pandas as pd

from flight_delay.weather.config import (
    EXTENDED_CYCLE_HOURS,
    ORIGIN_AIRPORTS,
    PREDICTION_HORIZON_HOURS,
    SUPPORTED_FORECAST_LEADS,
    WEATHER_AVAILABILITY_LAG_HOURS,
)

REQUEST_KEYS = [
    "forecast_run_at_utc",
    "forecast_valid_at_utc",
    "forecast_hour",
    "assumed_available_at_utc",
]


class WeatherPlanningError(ValueError):
    """Raised when a valid weather request cannot be planned."""


@dataclass
class ForecastRequestPlan:
    """Flight mappings and deduplicated acquisition requests."""

    flight_mapping: pd.DataFrame
    hourly_requests: pd.DataFrame
    cycle_requests: pd.DataFrame


def build_flight_forecast_mapping(
    flights: pd.DataFrame,
    *,
    departure_col: str = "scheduled_departure_utc",
    origin_col: str = "Origin",
) -> pd.DataFrame:
    """Map scheduled flights to eligible HRRR forecasts."""

    required_columns = {
        departure_col,
        origin_col,
    }

    missing_columns = required_columns - set(flights.columns)

    if missing_columns:
        raise WeatherPlanningError(f"Missing columns: {sorted(missing_columns)}")

    if flights.empty:
        raise WeatherPlanningError("Flight data must not be empty")

    departures = flights[departure_col]

    # Require timezone-aware timestamps.
    # Never assume naive timestamps are UTC.

    if not isinstance(
        departures.dtype,
        pd.DatetimeTZDtype,
    ):
        raise WeatherPlanningError("Scheduled departure must be timezone-aware")

    if departures.isna().any():
        raise WeatherPlanningError("Missing scheduled departure timestamps")

    departures = departures.dt.tz_convert("UTC").reset_index(drop=True)

    origins = flights[origin_col].reset_index(drop=True)

    if not origins.isin(ORIGIN_AIRPORTS).all():
        raise WeatherPlanningError("Unsupported or missing origin airport")

    prediction_horizon = pd.Timedelta(hours=PREDICTION_HORIZON_HOURS)

    availability_lag = pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)

    prediction_at = departures - prediction_horizon

    # Select the latest eligible extended forecast cycle.

    run_at = (prediction_at - availability_lag).dt.floor("6h")

    available_at = run_at + availability_lag

    valid_at = departures.dt.floor("h")

    lead_hours = (valid_at - run_at) / pd.Timedelta(hours=1)

    if not lead_hours.isin(SUPPORTED_FORECAST_LEADS).all():
        raise WeatherPlanningError("Unsupported forecast lead")

    if not run_at.dt.hour.isin(EXTENDED_CYCLE_HOURS).all():
        raise WeatherPlanningError("Invalid extended forecast cycle")

    if not available_at.le(prediction_at).all():
        raise WeatherPlanningError("Forecast unavailable at prediction time")

    # Ensure the chosen cycle is the latest eligible one.

    next_cycle_available_at = available_at + pd.Timedelta(hours=6)

    if not next_cycle_available_at.gt(prediction_at).all():
        raise WeatherPlanningError("Selected forecast cycle is not the latest")

    mapping = pd.DataFrame(
        {
            "flight_row_position": range(len(flights)),
            "Origin": origins,
            "scheduled_departure_utc": departures,
            "prediction_at_utc": prediction_at,
            "forecast_run_at_utc": run_at,
            "forecast_valid_at_utc": valid_at,
            "forecast_hour": lead_hours.astype("int64"),
            "assumed_available_at_utc": available_at,
        }
    )

    return mapping


def select_hrrr_forecast(
    scheduled_departure_utc,
) -> dict:
    """Select an eligible forecast for a single departure."""

    try:
        departure = pd.Timestamp(scheduled_departure_utc)

    except (ValueError, TypeError, OverflowError) as exc:
        raise WeatherPlanningError("Invalid scheduled departure timestamp") from exc

    if pd.isna(departure):
        raise WeatherPlanningError("Missing scheduled departure timestamp")

    if departure.tzinfo is None:
        raise WeatherPlanningError("Scheduled departure must be timezone-aware")

    frame = pd.DataFrame(
        {
            "Origin": [ORIGIN_AIRPORTS[0]],
            "scheduled_departure_utc": [departure.tz_convert("UTC")],
        }
    )

    mapping = build_flight_forecast_mapping(frame)

    reference = mapping.iloc[0].to_dict()

    reference.pop("Origin")
    reference.pop("flight_row_position")

    return reference


def build_forecast_request_plan(
    flights: pd.DataFrame,
    *,
    departure_col: str = "scheduled_departure_utc",
    origin_col: str = "Origin",
) -> ForecastRequestPlan:
    """Build deduplicated hourly and cycle-level requests."""

    mapping = build_flight_forecast_mapping(
        flights,
        departure_col=departure_col,
        origin_col=origin_col,
    )

    # One request per unique forecast identity.
    # Hourly weather cache covers all six airports.

    hourly_requests = (
        mapping.groupby(
            REQUEST_KEYS,
            as_index=False,
            sort=True,
        )
        .agg(
            flight_count=("Origin", "size"),
            requested_airport_count=("Origin", "nunique"),
        )
        .sort_values(
            [
                "forecast_run_at_utc",
                "forecast_hour",
            ]
        )
        .reset_index(drop=True)
    )

    hourly_requests["cache_airport_count"] = len(ORIGIN_AIRPORTS)

    # Group multiple forecast leads into one Zarr task.

    cycle_requests = (
        hourly_requests.groupby(
            "forecast_run_at_utc",
            as_index=False,
            sort=True,
        )
        .agg(
            forecast_leads=(
                "forecast_hour",
                lambda leads: tuple(sorted(int(lead) for lead in leads)),
            ),
            forecast_hour_count=("forecast_hour", "size"),
            flight_count=("flight_count", "sum"),
        )
        .reset_index(drop=True)
    )

    if hourly_requests["flight_count"].sum() != len(flights):
        raise WeatherPlanningError("Hourly request flight count mismatch")

    if cycle_requests["flight_count"].sum() != len(flights):
        raise WeatherPlanningError("Cycle request flight count mismatch")

    return ForecastRequestPlan(
        flight_mapping=mapping,
        hourly_requests=hourly_requests,
        cycle_requests=cycle_requests,
    )


def summarize_request_plan(
    plan: ForecastRequestPlan,
) -> dict[str, int | float]:
    """Summarize acquisition workload and deduplication."""

    flight_count = len(plan.flight_mapping)

    hourly_count = len(plan.hourly_requests)

    cycle_count = len(plan.cycle_requests)

    return {
        "flight_count": flight_count,
        "unique_hourly_requests": hourly_count,
        "unique_cycle_requests": cycle_count,
        "hourly_request_reduction_pct": (100 * (1 - hourly_count / flight_count)),
        "cycle_request_reduction_pct": (100 * (1 - cycle_count / hourly_count)),
    }
