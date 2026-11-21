"""Tests for scheduled-departure prediction timestamps."""

import pandas as pd
import pytest

from flight_delay.features.prediction_time import (
    build_prediction_timestamps,
)


def make_flight(
    date="2024-01-15",
    origin="SFO",
    hhmm=800,
):
    return pd.DataFrame({
        "FlightDate": [date],
        "Origin": [origin],
        "CRSDepTime": [hhmm],
    })


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("ATL", "2024-01-15T13:00:00Z"),
        ("DFW", "2024-01-15T14:00:00Z"),
        ("ORD", "2024-01-15T14:00:00Z"),
        ("DEN", "2024-01-15T15:00:00Z"),
        ("LAX", "2024-01-15T16:00:00Z"),
        ("SFO", "2024-01-15T16:00:00Z"),
    ],
)
def test_winter_timezone_conversion(origin, expected):
    result = build_prediction_timestamps(
        make_flight(origin=origin)
    )

    assert result.loc[
        0, "scheduled_departure_utc"
    ] == pd.Timestamp(expected)


def test_summer_timezone_conversion():
    result = build_prediction_timestamps(
        make_flight(date="2024-07-15")
    )

    assert result.loc[
        0, "scheduled_departure_utc"
    ] == pd.Timestamp("2024-07-15T15:00:00Z")


def test_exact_prediction_horizon():
    result = build_prediction_timestamps(
        make_flight()
    )

    horizon = (
        result.loc[0, "scheduled_departure_utc"]
        - result.loc[0, "prediction_at_utc"]
    )

    assert horizon == pd.Timedelta(hours=24)


def test_prediction_horizon_across_dst():
    result = build_prediction_timestamps(
        make_flight(date="2024-03-10")
    )

    assert result.loc[
        0, "scheduled_departure_utc"
    ] == pd.Timestamp("2024-03-10T15:00:00Z")

    assert result.loc[
        0, "prediction_at_utc"
    ] == pd.Timestamp("2024-03-09T15:00:00Z")


def test_prediction_crosses_year_boundary():
    result = build_prediction_timestamps(
        make_flight(date="2023-01-01")
    )

    assert result.loc[
        0, "prediction_at_utc"
    ] == pd.Timestamp("2022-12-31T16:00:00Z")


@pytest.mark.parametrize(
    ("date", "hhmm"),
    [
        ("2024-03-10", 230),
        ("2024-11-03", 130),
    ],
)
def test_unresolved_dst(date, hhmm):
    result = build_prediction_timestamps(
        make_flight(date=date, hhmm=hhmm)
    )

    assert result.loc[
        0, "timestamp_status"
    ] == "unresolved_dst"

    assert pd.isna(
        result.loc[0, "prediction_at_utc"]
    )


def test_unresolved_2400():
    result = build_prediction_timestamps(
        make_flight(hhmm=2400)
    )

    assert result.loc[
        0, "timestamp_status"
    ] == "unresolved_2400"

    assert pd.isna(
        result.loc[0, "prediction_at_utc"]
    )


@pytest.mark.parametrize(
    "hhmm",
    [-1, 1260, 2500, 830.5, None],
)
def test_invalid_hhmm_rejected(hhmm):
    with pytest.raises(ValueError):
        build_prediction_timestamps(
            make_flight(hhmm=hhmm)
        )


def test_unknown_airport_rejected():
    with pytest.raises(ValueError):
        build_prediction_timestamps(
            make_flight(origin="ZZZ")
        )


def test_missing_column_rejected():
    raw = make_flight().drop(
        columns=["CRSDepTime"]
    )

    with pytest.raises(ValueError):
        build_prediction_timestamps(raw)


def test_index_and_input_preserved():
    raw = pd.concat(
        [make_flight(), make_flight(origin="ATL")],
        ignore_index=True,
    )

    raw.index = [10, 20]

    original = raw.copy(deep=True)

    result = build_prediction_timestamps(raw)

    assert result.index.equals(raw.index)
    assert len(result) == len(raw)

    pd.testing.assert_frame_equal(
        raw, original
    )
