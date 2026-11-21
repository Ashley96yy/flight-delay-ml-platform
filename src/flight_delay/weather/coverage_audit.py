"""Independent point-in-time weather coverage audit."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flight_delay.weather.config import (
    SUPPORTED_FORECAST_LEADS,
    WEATHER_AVAILABILITY_LAG_HOURS,
)
from flight_delay.weather.features import MODEL_WEATHER_COLUMNS
from flight_delay.weather.flight_join import JOIN_KEYS


class WeatherCoverageAuditError(ValueError):
    """Invalid audit input or strict audit failure."""


@dataclass
class WeatherCoverageAudit:
    detail: pd.DataFrame
    by_airport: pd.DataFrame
    violations: pd.DataFrame
    summary: dict


def _utc(frame, column):
    try:
        values = pd.to_datetime(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise WeatherCoverageAuditError(f"Invalid datetime column: {column}") from exc

    if not isinstance(values.dtype, pd.DatetimeTZDtype) or values.isna().any():
        raise WeatherCoverageAuditError(
            f"{column} must be timezone-aware and nonmissing"
        )

    return values.dt.tz_convert("UTC")


def audit_weather_coverage(
    flight_mapping: pd.DataFrame,
    weather_features: pd.DataFrame,
    *,
    strict=False,
) -> WeatherCoverageAudit:
    """Audit forecast identities, point-in-time safety and coverage."""

    flight_required = set(
        JOIN_KEYS
        + [
            "scheduled_departure_utc",
            "prediction_at_utc",
        ]
    )

    weather_required = set(
        JOIN_KEYS + MODEL_WEATHER_COLUMNS + ["assumed_available_at_utc", "backend"]
    )

    if not isinstance(flight_mapping, pd.DataFrame):
        raise WeatherCoverageAuditError("Flight mapping must be a DataFrame")

    if not isinstance(weather_features, pd.DataFrame):
        raise WeatherCoverageAuditError("Weather features must be a DataFrame")

    if flight_mapping.empty:
        raise WeatherCoverageAuditError("Empty flight mapping")

    if not flight_required.issubset(flight_mapping.columns):
        raise WeatherCoverageAuditError("Missing flight audit columns")

    if not weather_required.issubset(weather_features.columns):
        raise WeatherCoverageAuditError("Missing weather audit columns")

    flights = flight_mapping.copy(deep=True)
    weather = weather_features.copy(deep=True)

    if "flight_row_position" in flights.columns and (
        flights["flight_row_position"].isna().any()
        or flights["flight_row_position"].duplicated().any()
    ):
        raise WeatherCoverageAuditError("Duplicate or missing flight row position")

    for column in (
        "scheduled_departure_utc",
        "prediction_at_utc",
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
    ):
        flights[column] = _utc(flights, column)

    for column in (
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
        "assumed_available_at_utc",
    ):
        weather[column] = _utc(weather, column)

    if flights[JOIN_KEYS].isna().any().any() or weather[JOIN_KEYS].isna().any().any():
        raise WeatherCoverageAuditError("Missing forecast join identity")

    if weather.duplicated(JOIN_KEYS).any():
        raise WeatherCoverageAuditError("Duplicate weather forecast identity")

    expected_prediction = flights["scheduled_departure_utc"] - pd.Timedelta(hours=24)

    expected_run = (
        expected_prediction - pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)
    ).dt.floor("6h")

    expected_valid = flights["scheduled_departure_utc"].dt.floor("h")

    expected_lead = (expected_valid - expected_run).dt.total_seconds() / 3600

    actual_lead = pd.to_numeric(
        flights["forecast_hour"],
        errors="coerce",
    )

    flights["planner_valid"] = (
        flights["prediction_at_utc"].eq(expected_prediction)
        & flights["forecast_run_at_utc"].eq(expected_run)
        & flights["forecast_valid_at_utc"].eq(expected_valid)
        & actual_lead.eq(expected_lead)
        & actual_lead.isin(SUPPORTED_FORECAST_LEADS)
    )

    flights["__audit_row"] = np.arange(len(flights))

    selected = weather[
        JOIN_KEYS + MODEL_WEATHER_COLUMNS + ["assumed_available_at_utc", "backend"]
    ].rename(
        columns={
            "assumed_available_at_utc": ("weather_available_at_utc"),
            "backend": "weather_backend",
        }
    )

    detail = flights.merge(
        selected,
        on=JOIN_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="__match",
    )

    detail = detail.sort_values("__audit_row").reset_index(drop=True)

    detail["weather_matched"] = detail["__match"].eq("both")

    numeric_features = (
        detail[MODEL_WEATHER_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    detail["features_valid"] = np.isfinite(numeric_features).all(axis=1)

    available = detail["weather_available_at_utc"]

    detail["time_safe"] = (
        detail["weather_matched"]
        & available.le(detail["prediction_at_utc"])
        & available.eq(
            detail["forecast_run_at_utc"]
            + pd.Timedelta(hours=WEATHER_AVAILABILITY_LAG_HOURS)
        )
    )

    detail["backend_valid"] = detail["weather_backend"].isin(["zarr", "grib"])

    detail["status"] = np.select(
        [
            ~detail["planner_valid"],
            ~detail["weather_matched"],
            ~detail["features_valid"],
            ~detail["time_safe"],
            ~detail["backend_valid"],
        ],
        [
            "invalid_planner",
            "missing_weather",
            "invalid_weather_features",
            "future_weather",
            "invalid_weather_backend",
        ],
        default="ready",
    )

    detail["ready"] = detail["status"].eq("ready")

    total = len(detail)
    ready = int(detail["ready"].sum())

    summary = {
        "flight_count": total,
        "matched_flights": int(detail["weather_matched"].sum()),
        "ready_flights": ready,
        "missing_weather_flights": int(detail["status"].eq("missing_weather").sum()),
        "invalid_flights": total - ready,
        "coverage_rate": ready / total,
    }

    by_airport = detail.groupby("Origin", as_index=False).agg(
        flights=("ready", "size"),
        ready=("ready", "sum"),
    )

    by_airport["coverage_rate"] = by_airport["ready"] / by_airport["flights"]

    violations = detail.loc[~detail["ready"]].copy()

    if strict and not violations.empty:
        counts = violations["status"].value_counts().to_dict()

        raise WeatherCoverageAuditError(f"Weather coverage audit failed: {counts}")

    return WeatherCoverageAudit(
        detail=detail.drop(columns=["__audit_row", "__match"]),
        by_airport=by_airport,
        violations=violations.drop(columns=["__audit_row", "__match"]),
        summary=summary,
    )
