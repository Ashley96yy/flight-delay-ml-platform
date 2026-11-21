"""Tests for point-in-time historical features."""

import pandas as pd
import pytest

from flight_delay.features.availability import (
    build_proxy_availability_calendar,
)
from flight_delay.features.historical_aggregation import (
    build_rate_snapshots,
)
from flight_delay.features.historical_features import (
    HISTORICAL_FEATURE_COLUMNS,
    build_historical_features,
)
from flight_delay.features.historical_lookup import (
    build_history_windows,
)


@pytest.fixture
def history():
    months = [
        "2023-01",
        "2023-02",
        "2023-03",
        "2023-04",
    ]

    calendar = build_proxy_availability_calendar(
        months, lag_days=0
    )

    windows = build_history_windows(
        calendar, lookback_months=3
    )

    origin = pd.DataFrame({
        "report_month": months,
        "Origin": ["ATL"] * 4,
        "flight_count": [100, 200, 300, 400],
        "dep_delay_count": [20, 60, 120, 200],
        "arr_delay_count": [10, 40, 90, 160],
    })

    airline = origin.copy()
    airline["Reporting_Airline"] = "DL"

    return windows, origin, airline


def generate(history, cutoffs, airlines=None, minimum=100):
    windows, origin, airline = history

    flights = pd.DataFrame({
        "Origin": ["ATL"] * len(cutoffs),
        "Reporting_Airline": (
            airlines
            if airlines is not None
            else ["DL"] * len(cutoffs)
        ),
    })

    timestamps = pd.Series(
        pd.to_datetime(cutoffs, utc=True)
    )

    return build_historical_features(
        flights=flights,
        prediction_at_utc=timestamps,
        windows=windows,
        origin_snapshots=build_rate_snapshots(
            origin, windows, ["Origin"]
        ),
        airline_snapshots=build_rate_snapshots(
            airline,
            windows,
            ["Origin", "Reporting_Airline"],
        ),
        min_airline_flights=minimum,
    )


def test_weighted_historical_rate(history):
    result = generate(
        history, ["2023-04-15"]
    )

    assert result.loc[
        0, "origin_dep_delay_rate_3m"
    ] == pytest.approx(200 / 600)

    assert result.loc[
        0, "origin_outbound_arr_delay_rate_3m"
    ] == pytest.approx(140 / 600)


def test_availability_boundary(history):
    result = generate(
        history,
        [
            "2023-04-30 23:59:59",
            "2023-05-01 00:00:00",
            "2023-05-01 00:00:01",
        ],
    )

    assert result["window_id"].tolist() == [
        "2023-03",
        "2023-04",
        "2023-04",
    ]


def test_future_source_perturbation(history):
    cutoffs = ["2023-04-15", "2023-05-15"]

    original = generate(history, cutoffs)

    windows, origin, airline = history

    modified_origin = origin.copy()
    modified_airline = airline.copy()

    # April data is unavailable before May 1.
    for table in [modified_origin, modified_airline]:
        table.loc[
            table["report_month"].eq("2023-04"),
            "dep_delay_count",
        ] = 400

    modified = generate(
        (windows, modified_origin, modified_airline),
        cutoffs,
    )

    columns = list(HISTORICAL_FEATURE_COLUMNS)

    # Earlier predictions must remain unchanged.
    pd.testing.assert_series_equal(
        original.loc[0, columns],
        modified.loc[0, columns],
    )

    # Later predictions may use the updated source.
    assert modified.loc[
        1, "origin_dep_delay_rate_3m"
    ] > original.loc[
        1, "origin_dep_delay_rate_3m"
    ]


def test_unknown_airline_fallback(history):
    result = generate(
        history,
        ["2023-04-15"],
        airlines=["ZZ"],
    )

    assert result.loc[
        0, "airline_history_status"
    ] == "fallback_unknown_group"

    assert result.loc[
        0, "airline_origin_dep_delay_rate_3m"
    ] == result.loc[
        0, "origin_dep_delay_rate_3m"
    ]


def test_low_sample_fallback(history):
    result = generate(
        history,
        ["2023-04-15"],
        minimum=601,
    )

    assert result.loc[
        0, "airline_history_status"
    ] == "fallback_low_sample"


def test_minimum_count_boundary(history):
    result = generate(
        history,
        ["2023-04-15"],
        minimum=600,
    )

    assert result.loc[
        0, "airline_history_status"
    ] == "direct"


def test_missing_timestamp(history):
    result = generate(history, [None])

    assert result.loc[
        0, "history_status"
    ] == "unresolved_timestamp"

    assert result[
        list(HISTORICAL_FEATURE_COLUMNS)
    ].isna().all().all()


def test_before_first_window(history):
    result = generate(
        history, ["2023-01-15"]
    )

    assert result.loc[
        0, "history_status"
    ] == "no_history"

    assert result[
        list(HISTORICAL_FEATURE_COLUMNS)
    ].isna().all().all()


def test_incomplete_window(history):
    result = generate(
        history, ["2023-03-15"]
    )

    assert result.loc[
        0, "history_status"
    ] == "insufficient_months"

    assert result[
        list(HISTORICAL_FEATURE_COLUMNS)
    ].isna().all().all()


def test_duplicate_source_keys_rejected(history):
    windows, origin, _ = history

    duplicate = pd.concat(
        [origin, origin.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        build_rate_snapshots(
            duplicate, windows, ["Origin"]
        )


def test_missing_source_month_rejected(history):
    windows, origin, _ = history

    incomplete = origin.loc[
        origin["report_month"].ne("2023-02")
    ]

    with pytest.raises(ValueError):
        build_rate_snapshots(
            incomplete, windows, ["Origin"]
        )


def test_unsorted_prediction_order(history):
    result = generate(
        history,
        [
            "2023-05-15",
            "2023-04-15",
            None,
            "2023-04-15",
        ],
    )

    assert len(result) == 4

    assert result.loc[0, "window_id"] == "2023-04"
    assert result.loc[1, "window_id"] == "2023-03"
    assert pd.isna(result.loc[2, "window_id"])
    assert result.loc[3, "window_id"] == "2023-03"
