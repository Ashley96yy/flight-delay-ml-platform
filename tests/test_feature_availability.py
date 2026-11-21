"""Unit tests for historical source availability."""

import pandas as pd
import pytest

from flight_delay.features.availability import (
    build_proxy_availability_calendar,
    eligible_source_months,
)


@pytest.fixture
def calendar():
    return build_proxy_availability_calendar(
        [
            "2023-07",
            "2023-08",
            "2023-09",
            "2023-10",
        ],
        lag_days=120,
    )


def test_calendar_has_expected_columns(calendar):
    expected = {
        "report_month",
        "source_filename",
        "assumed_available_at_utc",
        "availability_policy",
        "availability_evidence",
        "lag_days",
    }

    assert expected.issubset(calendar.columns)
    assert len(calendar) == 4


def test_calendar_uses_120_day_proxy():
    result = build_proxy_availability_calendar(
        ["2023-01"],
        lag_days=120,
    )

    assert result.loc[
        0, "assumed_available_at_utc"
    ] == pd.Timestamp("2023-06-01T00:00:00Z")

    assert result.loc[
        0, "availability_evidence"
    ] == "assumed_not_verified"


def test_select_latest_available_months(calendar):
    eligible = eligible_source_months(
        calendar,
        "2024-02-09T18:00:00Z",
        lookback_months=3,
    )

    assert eligible["report_month"].tolist() == [
        "2023-07",
        "2023-08",
        "2023-09",
    ]


def test_exact_availability_boundary():
    calendar = build_proxy_availability_calendar(
        ["2023-09"]
    )

    before = eligible_source_months(
        calendar,
        "2024-01-28T23:59:59Z",
    )

    at_cutoff = eligible_source_months(
        calendar,
        "2024-01-29T00:00:00Z",
    )

    assert before.empty
    assert at_cutoff["report_month"].tolist() == [
        "2023-09"
    ]


def test_lookback_limit(calendar):
    eligible = eligible_source_months(
        calendar,
        "2024-02-09T18:00:00Z",
        lookback_months=2,
    )

    assert eligible["report_month"].tolist() == [
        "2023-08",
        "2023-09",
    ]


def test_future_source_excluded():
    calendar = build_proxy_availability_calendar(
        ["2024-01"]
    )

    eligible = eligible_source_months(
        calendar,
        "2024-02-09T18:00:00Z",
    )

    assert eligible.empty


def test_invalid_reporting_month():
    with pytest.raises(ValueError):
        build_proxy_availability_calendar(
            ["2023-13"]
        )


def test_duplicate_reporting_month():
    with pytest.raises(ValueError):
        build_proxy_availability_calendar(
            ["2023-01", "2023-01"]
        )


def test_negative_reporting_lag():
    with pytest.raises(ValueError):
        build_proxy_availability_calendar(
            ["2023-01"],
            lag_days=-1,
        )


def test_naive_prediction_timestamp_rejected(calendar):
    with pytest.raises(ValueError):
        eligible_source_months(
            calendar,
            pd.Timestamp("2024-02-09 18:00:00"),
        )


def test_invalid_lookback():
    with pytest.raises(ValueError):
        eligible_source_months(
            build_proxy_availability_calendar(
                ["2023-01"]
            ),
            "2024-02-09T18:00:00Z",
            lookback_months=0,
        )
