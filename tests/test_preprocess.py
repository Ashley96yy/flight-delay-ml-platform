import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flight_delay.data.preprocess import prepare_modeling_cohort


@pytest.fixture
def sample_flights():
    """Small synthetic dataset covering different flight outcomes."""

    return pd.DataFrame({
        "FlightDate": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
        "Reporting_Airline": ["AA", "UA", "DL", "AA", "UA"],
        "Flight_Number_Reporting_Airline": [
            "101", "202", "303", "404", "505"
        ],
        "Origin": ["ATL", "SFO", "SEA", "DFW", "ORD"],
        "Dest": ["JFK", "LAX", "SFO", "MIA", "DEN"],
        "CRSDepTime": [800, 1200, 1500, 900, 1400],
        "CRSArrTime": [1100, 1400, 1700, 1200, 1600],
        "CRSElapsedTime": [180, 120, 120, 180, 120],
        "Distance": [760, 337, 679, 1121, 888],

        "DepDelay": [15, -5, 0, None, 0],
        "DepDel15": [1, 0, 0, None, 0],

        "ArrDelay": [14, 20, 0, None, None],
        "ArrDel15": [0, 1, 0, None, None],

        "Cancelled": [0, 0, 0, 1, 0],
        "Diverted": [0, 0, 0, 0, 1],
    })


def test_filters_eligible_flights(sample_flights):
    """Only completed flights from selected airports are retained."""

    result = prepare_modeling_cohort(sample_flights)

    assert len(result) == 2
    assert set(result["Origin"]) == {"ATL", "SFO"}

    assert result["Cancelled"].eq(0).all()
    assert result["Diverted"].eq(0).all()


def test_output_data_types(sample_flights):
    """Dates and labels should have normalized types."""

    result = prepare_modeling_cohort(sample_flights)

    assert pd.api.types.is_datetime64_any_dtype(
        result["FlightDate"]
    )

    assert result["DepDel15"].dtype == "int8"
    assert result["ArrDel15"].dtype == "int8"


def test_original_dataframe_unchanged(sample_flights):
    """Preprocessing must not modify its input."""

    original = sample_flights.copy(deep=True)

    prepare_modeling_cohort(sample_flights)

    assert_frame_equal(sample_flights, original)


def test_preprocessing_is_idempotent(sample_flights):
    """Running preprocessing twice should produce the same output."""

    first = prepare_modeling_cohort(sample_flights)
    second = prepare_modeling_cohort(first)

    assert_frame_equal(first, second)


def test_invalid_input_rejected(sample_flights):
    """Invalid labels must fail validation before preprocessing."""

    invalid = sample_flights.copy()

    # A 15-minute departure delay must have label 1.
    invalid.loc[0, "DepDel15"] = 0

    with pytest.raises(ValueError, match="Label inconsistency"):
        prepare_modeling_cohort(invalid)
