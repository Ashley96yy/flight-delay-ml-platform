import pandas as pd
import pytest

from flight_delay.data.validate import validate_bts_data


@pytest.fixture
def valid_bts_df():
    """Create a small synthetic BTS dataset for testing."""

    return pd.DataFrame({
        "FlightDate": [
            "2023-01-02", "2023-01-03", "2023-01-04"
        ],
        "Reporting_Airline": ["AA", "UA", "DL"],
        "Flight_Number_Reporting_Airline": ["101", "202", "303"],
        "Origin": ["ATL", "LAX", "SEA"],
        "Dest": ["JFK", "SFO", "ORD"],
        "CRSDepTime": [800, 1300, 1500],
        "CRSArrTime": [1100, 1500, 1900],
        "CRSElapsedTime": [120, 90, 240],
        "Distance": [800, 337, 1600],

        "DepDelay": [15.0, -5.0, None],
        "DepDel15": [1.0, 0.0, None],

        "ArrDelay": [14.0, 20.0, None],
        "ArrDel15": [0.0, 1.0, None],

        "Cancelled": [0.0, 0.0, 1.0],
        "Diverted": [0.0, 0.0, 0.0],
    })


def test_valid_bts_data(valid_bts_df):
    """Valid BTS records should pass validation."""

    result = validate_bts_data(valid_bts_df)

    assert result["raw_records"] == 3
    assert result["six_airport_records"] == 2
    assert result["eligible_records"] == 2


def test_missing_required_column(valid_bts_df):
    """Required schema fields must be present."""

    df = valid_bts_df.drop(columns=["ArrDel15"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_bts_data(df)


def test_invalid_binary_label(valid_bts_df):
    """Binary labels must contain only 0, 1, or missing values."""

    df = valid_bts_df.copy()
    df.loc[0, "DepDel15"] = 2

    with pytest.raises(ValueError, match="Invalid binary labels"):
        validate_bts_data(df)


def test_label_inconsistency(valid_bts_df):
    """Labels must agree with the 15-minute delay rule."""

    df = valid_bts_df.copy()

    # Departure delay = 15 minutes, so the label must be 1.
    df.loc[0, "DepDel15"] = 0

    with pytest.raises(ValueError, match="Label inconsistency"):
        validate_bts_data(df)


def test_missing_label_in_completed_flight(valid_bts_df):
    """Completed flights must have valid delay outcomes."""

    df = valid_bts_df.copy()

    df.loc[0, "ArrDelay"] = None
    df.loc[0, "ArrDel15"] = None

    with pytest.raises(ValueError, match="Missing labels"):
        validate_bts_data(df)


def test_missing_baseline_feature(valid_bts_df):
    """Required modeling features cannot be missing."""

    df = valid_bts_df.copy()
    df.loc[0, "Distance"] = None

    with pytest.raises(ValueError, match="Missing baseline features"):
        validate_bts_data(df)


def test_invalid_flight_status(valid_bts_df):
    """Cancelled and Diverted must be valid binary indicators."""

    df = valid_bts_df.copy()
    df.loc[0, "Cancelled"] = 2

    with pytest.raises(ValueError, match="Invalid flight status"):
        validate_bts_data(df)


def test_label_delay_missingness_mismatch(valid_bts_df):
    """A missing label with nonmissing delay must be rejected."""

    df = valid_bts_df.copy()
    df.loc[0, "ArrDel15"] = None

    with pytest.raises(ValueError, match="Missingness mismatch"):
        validate_bts_data(df)


def test_duplicate_candidate_keys_reported(valid_bts_df):
    """Repeated candidate flight keys should be reported."""
    duplicated = pd.concat(
        [valid_bts_df, valid_bts_df.iloc[[0]]],
        ignore_index=True,
    )

    report = validate_bts_data(duplicated)

    assert report["duplicate_candidate_key_records"] == 2


def test_incomplete_candidate_keys_reported(valid_bts_df):
    """Missing flight identifiers should be reported."""
    df = valid_bts_df.copy()
    df.loc[0, "Flight_Number_Reporting_Airline"] = None

    report = validate_bts_data(df)

    assert report["incomplete_candidate_key_records"] == 1
