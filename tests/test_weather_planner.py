"""Tests for point-in-time weather request planning."""

import pandas as pd
import pytest

from flight_delay.weather.planner import (
    WeatherPlanningError,
    build_flight_forecast_mapping,
    build_forecast_request_plan,
    select_hrrr_forecast,
    summarize_request_plan,
)


@pytest.fixture
def sample_flights():

    return pd.DataFrame(
        {
            "Origin": [
                "ATL",
                "DFW",
                "DEN",
                "ORD",
                "LAX",
                "SFO",
            ],
            "scheduled_departure_utc": pd.to_datetime(
                [
                    "2024-07-15T09:00:00Z",
                    "2024-07-15T09:30:00Z",
                    "2024-07-15T10:00:00Z",
                    "2024-07-15T12:45:00Z",
                    "2024-07-15T14:59:00Z",
                    "2024-07-15T15:00:00Z",
                ],
                utc=True,
            ),
        }
    )


def test_select_forecast():

    reference = select_hrrr_forecast("2024-07-15T12:45:00Z")

    assert reference["forecast_run_at_utc"] == pd.Timestamp("2024-07-14T06:00:00Z")

    assert reference["forecast_hour"] == 30

    assert reference["forecast_valid_at_utc"] == pd.Timestamp("2024-07-15T12:00:00Z")

    assert reference["prediction_at_utc"] == pd.Timestamp("2024-07-14T12:45:00Z")


@pytest.mark.parametrize(
    "departure,expected_run,expected_lead",
    [
        (
            "2024-07-15T08:59:59Z",
            "2024-07-14T00:00:00Z",
            32,
        ),
        (
            "2024-07-15T09:00:00Z",
            "2024-07-14T06:00:00Z",
            27,
        ),
        (
            "2024-07-15T14:59:59Z",
            "2024-07-14T06:00:00Z",
            32,
        ),
        (
            "2024-07-15T15:00:00Z",
            "2024-07-14T12:00:00Z",
            27,
        ),
    ],
)
def test_forecast_availability_boundary(
    departure,
    expected_run,
    expected_lead,
):

    reference = select_hrrr_forecast(departure)

    assert reference["forecast_run_at_utc"] == pd.Timestamp(expected_run)

    assert reference["forecast_hour"] == expected_lead


def test_year_boundary():

    reference = select_hrrr_forecast("2024-01-01T00:30:00Z")

    assert reference["forecast_run_at_utc"] == pd.Timestamp("2023-12-30T18:00:00Z")

    assert reference["forecast_hour"] == 30


def test_no_future_information(sample_flights):

    mapping = build_flight_forecast_mapping(sample_flights)

    assert (mapping["assumed_available_at_utc"] <= mapping["prediction_at_utc"]).all()


def test_latest_eligible_forecast(sample_flights):

    mapping = build_flight_forecast_mapping(sample_flights)

    next_available_at = mapping["assumed_available_at_utc"] + pd.Timedelta(hours=6)

    assert (next_available_at > mapping["prediction_at_utc"]).all()


def test_naive_timestamp_rejected():

    with pytest.raises(
        WeatherPlanningError,
        match="timezone-aware",
    ):
        select_hrrr_forecast("2024-07-15 12:45:00")


def test_missing_timestamp_rejected(sample_flights):

    sample_flights.loc[0, "scheduled_departure_utc"] = pd.NaT

    with pytest.raises(
        WeatherPlanningError,
        match="Missing scheduled departure",
    ):
        build_flight_forecast_mapping(sample_flights)


def test_invalid_airport_rejected(sample_flights):

    sample_flights.loc[0, "Origin"] = "XXX"

    with pytest.raises(
        WeatherPlanningError,
        match="Unsupported or missing origin",
    ):
        build_flight_forecast_mapping(sample_flights)


def test_empty_input_rejected(sample_flights):

    with pytest.raises(
        WeatherPlanningError,
        match="must not be empty",
    ):
        build_flight_forecast_mapping(sample_flights.iloc[:0])


def test_missing_column_rejected(sample_flights):

    frame = sample_flights.drop(columns="Origin")

    with pytest.raises(
        WeatherPlanningError,
        match="Missing columns",
    ):
        build_flight_forecast_mapping(frame)


def test_timezone_conversion():

    reference = select_hrrr_forecast("2024-07-15T05:45:00-07:00")

    assert reference["scheduled_departure_utc"] == pd.Timestamp("2024-07-15T12:45:00Z")

    assert reference["forecast_hour"] == 30


def test_hourly_request_deduplication(
    sample_flights,
):

    plan = build_forecast_request_plan(sample_flights)

    assert len(plan.flight_mapping) == 6

    assert len(plan.hourly_requests) == 5

    assert plan.hourly_requests["flight_count"].sum() == 6

    assert plan.hourly_requests["cache_airport_count"].eq(6).all()


def test_cycle_request_grouping(sample_flights):

    plan = build_forecast_request_plan(sample_flights)

    assert len(plan.cycle_requests) == 2

    first = plan.cycle_requests.iloc[0]

    second = plan.cycle_requests.iloc[1]

    assert first["forecast_leads"] == (27, 28, 30, 32)

    assert first["flight_count"] == 5

    assert second["forecast_leads"] == (27,)

    assert second["flight_count"] == 1


def test_request_plan_summary(sample_flights):

    plan = build_forecast_request_plan(sample_flights)

    summary = summarize_request_plan(plan)

    assert summary["flight_count"] == 6

    assert summary["unique_hourly_requests"] == 5

    assert summary["unique_cycle_requests"] == 2

    assert summary["cycle_request_reduction_pct"] == pytest.approx(60.0)


def test_input_dataframe_unchanged(sample_flights):

    original = sample_flights.copy(deep=True)

    build_forecast_request_plan(sample_flights)

    pd.testing.assert_frame_equal(
        original,
        sample_flights,
    )


def test_duplicate_input_index_supported(
    sample_flights,
):

    sample_flights.index = [0, 0, 1, 1, 2, 2]

    mapping = build_flight_forecast_mapping(sample_flights)

    assert mapping["flight_row_position"].is_unique

    assert len(mapping) == len(sample_flights)
