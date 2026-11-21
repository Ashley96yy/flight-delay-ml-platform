
"""Tests for scheduled airport traffic features."""

import pandas as pd
import pytest

from flight_delay.features import scheduled_congestion as sc


@pytest.fixture
def raw_schedule():
    return pd.DataFrame({
        "FlightDate": ["2023-01-01"] * 6,
        "Origin": [
            "ATL", "ATL", "ATL",
            "ATL", "SFO", "BOS",
        ],
        "Reporting_Airline": [
            "DL", "DL", "AA",
            "DL", "UA", "B6",
        ],
        "CRSDepTime": [
            800, 815, 850, 900, 830, 800,
        ],
    })


def make_traffic(raw):
    prepared, audit = sc.prepare_departure_schedule(
        raw,
        "2023-01",
    )

    origin, airline = sc.aggregate_departure_traffic(
        prepared
    )

    return origin, airline, audit


def test_origin_scope(raw_schedule):
    prepared, audit = sc.prepare_departure_schedule(
        raw_schedule,
        "2023-01",
    )

    assert len(prepared) == 5
    assert "BOS" not in prepared["Origin"].values
    assert audit["ExcludedRecords"] == 0


def test_hourly_traffic_counts(raw_schedule):
    origin, airline, _ = make_traffic(raw_schedule)

    target = raw_schedule.iloc[[0, 2, 3, 4]]

    result = sc.build_departure_traffic_features(
        target,
        origin,
        airline,
        "2023-01",
    )

    assert result.iloc[:, 0].tolist() == [3, 3, 1, 1]
    assert result.iloc[:, 1].tolist() == [2, 1, 1, 1]


def test_outcome_independence(raw_schedule):
    first = raw_schedule.assign(
        Cancelled=0,
        Diverted=0,
        DepDel15=0,
        ArrDel15=0,
    )

    second = raw_schedule.assign(
        Cancelled=1,
        Diverted=1,
        DepDel15=1,
        ArrDel15=1,
    )

    origin_a, airline_a, _ = make_traffic(first)
    origin_b, airline_b, _ = make_traffic(second)

    pd.testing.assert_frame_equal(origin_a, origin_b)
    pd.testing.assert_frame_equal(airline_a, airline_b)


def test_index_and_order_preserved(raw_schedule):
    origin, airline, _ = make_traffic(raw_schedule)

    target = raw_schedule.iloc[[3, 0, 2]].copy()
    target.index = [50, 10, 30]

    result = sc.build_departure_traffic_features(
        target,
        origin,
        airline,
        "2023-01",
    )

    assert result.index.tolist() == [50, 10, 30]
    assert result.iloc[:, 0].tolist() == [1, 3, 3]


@pytest.mark.parametrize(
    "value",
    [-1, 1260, 2500, 830.5, None, 2400],
)
def test_invalid_schedule_time(raw_schedule, value):
    raw_schedule["CRSDepTime"] = (
        raw_schedule["CRSDepTime"].astype(object)
    )

    raw_schedule.loc[0, "CRSDepTime"] = value

    prepared, audit = sc.prepare_departure_schedule(
        raw_schedule,
        "2023-01",
    )

    assert len(prepared) == 4
    assert audit["ExcludedRecords"] == 1


def test_2400_audit(raw_schedule):
    raw_schedule.loc[0, "CRSDepTime"] = 2400

    _, audit = sc.prepare_departure_schedule(
        raw_schedule,
        "2023-01",
    )

    assert audit["Unresolved2400"] == 1
    assert audit["MissingOrInvalidTimes"] == 0


def test_missing_target_match(raw_schedule):
    origin, airline, _ = make_traffic(raw_schedule)

    target = raw_schedule.iloc[[0]].copy()
    target["CRSDepTime"] = 2300

    with pytest.raises(ValueError, match="Missing"):
        sc.build_departure_traffic_features(
            target,
            origin,
            airline,
            "2023-01",
        )


def test_duplicate_traffic_key_rejected(raw_schedule):
    origin, airline, _ = make_traffic(raw_schedule)

    origin = pd.concat(
        [origin, origin.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(pd.errors.MergeError):
        sc.build_departure_traffic_features(
            raw_schedule.iloc[[0]],
            origin,
            airline,
            "2023-01",
        )


def test_input_not_mutated(raw_schedule):
    original = raw_schedule.copy(deep=True)

    make_traffic(raw_schedule)

    pd.testing.assert_frame_equal(
        raw_schedule,
        original,
    )
