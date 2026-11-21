"""Offline tests for independent weather coverage auditing."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.coverage_audit import (
    WeatherCoverageAuditError,
    audit_weather_coverage,
)
from flight_delay.weather.features import MODEL_WEATHER_COLUMNS
from flight_delay.weather.planner import build_forecast_request_plan


@pytest.fixture
def audit_inputs():
    flights = pd.DataFrame(
        {
            "Origin": ["ATL", "SFO", "LAX"],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:15:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T15:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    plan = build_forecast_request_plan(flights)

    mapping = plan.flight_mapping.copy()

    weather = mapping[
        [
            "Origin",
            "forecast_run_at_utc",
            "forecast_valid_at_utc",
            "forecast_hour",
        ]
    ].copy()

    weather["assumed_available_at_utc"] = weather["forecast_run_at_utc"] + pd.Timedelta(
        hours=3
    )

    weather["backend"] = "zarr"

    for column in MODEL_WEATHER_COLUMNS:
        weather[column] = 1.0

    return mapping, weather


def test_complete_coverage(audit_inputs):
    mapping, weather = audit_inputs

    result = audit_weather_coverage(mapping, weather, strict=True)

    assert result.summary["ready_flights"] == 3
    assert result.summary["coverage_rate"] == 1.0
    assert result.violations.empty


def test_missing_weather_reported(audit_inputs):
    mapping, weather = audit_inputs

    result = audit_weather_coverage(mapping, weather.iloc[:-1])

    assert result.summary["matched_flights"] == 2
    assert result.summary["ready_flights"] == 2
    assert result.summary["missing_weather_flights"] == 1
    assert result.violations.iloc[0]["status"] == "missing_weather"


def test_wrong_prediction_time_reported(audit_inputs):
    mapping, weather = audit_inputs
    mapping.loc[0, "prediction_at_utc"] += pd.Timedelta(minutes=5)

    result = audit_weather_coverage(mapping, weather)

    assert result.detail.loc[0, "status"] == "invalid_planner"


def test_future_weather_reported(audit_inputs):
    mapping, weather = audit_inputs

    weather.loc[0, "assumed_available_at_utc"] += pd.Timedelta(days=2)

    result = audit_weather_coverage(mapping, weather)

    assert result.detail.loc[0, "status"] == "future_weather"


def test_missing_numeric_feature_reported(audit_inputs):
    mapping, weather = audit_inputs

    weather.loc[0, "origin_temperature_c"] = np.nan

    result = audit_weather_coverage(mapping, weather)

    assert result.detail.loc[0, "status"] == ("invalid_weather_features")


def test_unknown_backend_reported(audit_inputs):
    mapping, weather = audit_inputs

    weather.loc[0, "backend"] = "unknown"

    result = audit_weather_coverage(mapping, weather)

    assert result.detail.loc[0, "status"] == ("invalid_weather_backend")


def test_duplicate_weather_identity_rejected(audit_inputs):
    mapping, weather = audit_inputs

    weather = pd.concat(
        [weather, weather.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        WeatherCoverageAuditError,
        match="Duplicate weather forecast identity",
    ):
        audit_weather_coverage(mapping, weather)


def test_duplicate_flight_position_rejected(audit_inputs):
    mapping, weather = audit_inputs

    mapping.loc[1, "flight_row_position"] = mapping.loc[0, "flight_row_position"]

    with pytest.raises(
        WeatherCoverageAuditError,
        match="flight row position",
    ):
        audit_weather_coverage(mapping, weather)


def test_strict_mode_rejects_missing_weather(audit_inputs):
    mapping, weather = audit_inputs

    with pytest.raises(
        WeatherCoverageAuditError,
        match="missing_weather",
    ):
        audit_weather_coverage(mapping, weather.iloc[:-1], strict=True)


def test_airport_coverage_report(audit_inputs):
    mapping, weather = audit_inputs

    result = audit_weather_coverage(mapping, weather.iloc[:-1])

    assert len(result.by_airport) == 3
    assert result.by_airport["flights"].sum() == 3
    assert result.by_airport["ready"].sum() == 2


def test_inputs_not_modified(audit_inputs):
    mapping, weather = audit_inputs

    original_mapping = mapping.copy(deep=True)
    original_weather = weather.copy(deep=True)

    audit_weather_coverage(mapping, weather)

    pd.testing.assert_frame_equal(mapping, original_mapping)

    pd.testing.assert_frame_equal(weather, original_weather)
