"""Tests for monthly historical delay aggregation."""

import pandas as pd
import pytest

from flight_delay.features.historical_aggregation import (
    COUNT_COLUMNS,
    aggregate_monthly_delay_counts,
)


@pytest.fixture
def sample_flights():
    return pd.DataFrame({
        "FlightDate": pd.to_datetime([
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
        ]),
        "Origin": ["ATL", "ATL", "ATL", "DFW"],
        "Reporting_Airline": ["DL", "DL", "AA", "AA"],
        "DepDel15": [1, 0, 1, 0],
        "ArrDel15": [0, 1, 1, 0],
        "Cancelled": [0, 0, 0, 0],
        "Diverted": [0, 0, 0, 0],
    })


def test_origin_aggregation(sample_flights):
    origin, _ = aggregate_monthly_delay_counts(
        sample_flights, "2023-01"
    )

    atl = origin.set_index("Origin").loc["ATL"]

    assert atl["flight_count"] == 3
    assert atl["dep_delay_count"] == 2
    assert atl["arr_delay_count"] == 2


def test_airline_origin_aggregation(sample_flights):
    _, airline = aggregate_monthly_delay_counts(
        sample_flights, "2023-01"
    )

    atl_dl = airline.set_index(
        ["Origin", "Reporting_Airline"]
    ).loc[("ATL", "DL")]

    assert atl_dl["flight_count"] == 2
    assert atl_dl["dep_delay_count"] == 1
    assert atl_dl["arr_delay_count"] == 1


def test_aggregate_reconciliation(sample_flights):
    origin, airline = aggregate_monthly_delay_counts(
        sample_flights, "2023-01"
    )

    pd.testing.assert_frame_equal(
        origin.set_index("Origin")[list(COUNT_COLUMNS)],
        airline.groupby("Origin")[list(COUNT_COLUMNS)].sum(),
        check_dtype=False,
    )


def test_reporting_month_preserved(sample_flights):
    origin, airline = aggregate_monthly_delay_counts(
        sample_flights, "2023-01"
    )

    assert origin["report_month"].eq("2023-01").all()
    assert airline["report_month"].eq("2023-01").all()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Cancelled", 1),
        ("Diverted", 1),
        ("DepDel15", 2),
        ("ArrDel15", -1),
        ("Origin", None),
    ],
)
def test_invalid_records_rejected(
    sample_flights, column, value
):
    sample_flights.loc[0, column] = value

    with pytest.raises(ValueError):
        aggregate_monthly_delay_counts(
            sample_flights, "2023-01"
        )


def test_incorrect_reporting_month(sample_flights):
    with pytest.raises(ValueError):
        aggregate_monthly_delay_counts(
            sample_flights, "2023-02"
        )


def test_missing_column_rejected(sample_flights):
    invalid = sample_flights.drop(columns=["ArrDel15"])

    with pytest.raises(ValueError):
        aggregate_monthly_delay_counts(
            invalid, "2023-01"
        )


def test_empty_source_rejected(sample_flights):
    with pytest.raises(ValueError):
        aggregate_monthly_delay_counts(
            sample_flights.iloc[:0], "2023-01"
        )


def test_input_not_mutated(sample_flights):
    original = sample_flights.copy(deep=True)

    aggregate_monthly_delay_counts(
        sample_flights, "2023-01"
    )

    pd.testing.assert_frame_equal(
        sample_flights, original
    )
