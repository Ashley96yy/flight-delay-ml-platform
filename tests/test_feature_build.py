import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flight_delay.features.build import (
    FEATURE_COLUMNS,
    build_baseline_features,
    hhmm_to_minutes,
)


@pytest.fixture
def sample_features():
    """Two scheduled flights without outcome fields."""

    return pd.DataFrame({
        "FlightDate": ["2023-07-15", "2023-07-17"],
        "Reporting_Airline": ["UA", "AA"],
        "Origin": ["SFO", "DFW"],
        "Dest": ["JFK", "LAX"],
        "CRSDepTime": [2330, 5],
        "CRSArrTime": [2400, 1435],
        "CRSElapsedTime": [330, 180],
        "Distance": [2586, 1235],
    })


@pytest.mark.parametrize(
    "hhmm,expected",
    [
        (5, 5),
        (930, 570),
        (1435, 875),
        (2359, 1439),
        (2400, 0),
    ],
)
def test_hhmm_conversion(hhmm, expected):
    result = hhmm_to_minutes(pd.Series([hhmm]))
    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    "invalid",
    [-1, 2360, 2500, 930.5],
)
def test_invalid_hhmm(invalid):
    with pytest.raises(ValueError):
        hhmm_to_minutes(pd.Series([invalid]))


def test_feature_shape_and_schema(sample_features):
    result = build_baseline_features(sample_features)

    assert result.shape == (2, 17)
    assert tuple(result.columns) == FEATURE_COLUMNS
    assert result.isna().sum().sum() == 0


def test_calendar_features(sample_features):
    result = build_baseline_features(sample_features)

    assert result["month"].tolist() == [7, 7]
    assert result["day_of_week"].tolist() == [5, 0]
    assert result["is_weekend"].tolist() == [1, 0]


def test_midnight_cyclical_encoding(sample_features):
    """23:30 and 00:30 should be near each other."""

    df = sample_features.copy()
    df.loc[1, "CRSDepTime"] = 30

    result = build_baseline_features(df)

    first = result.loc[
        0, ["dep_time_sin", "dep_time_cos"]
    ].to_numpy(dtype=float)

    second = result.loc[
        1, ["dep_time_sin", "dep_time_cos"]
    ].to_numpy(dtype=float)

    distance = np.linalg.norm(first - second)

    assert distance < 0.3


def test_missing_required_feature(sample_features):
    df = sample_features.drop(columns=["Distance"])

    with pytest.raises(ValueError, match="Missing required"):
        build_baseline_features(df)


def test_outcome_column_rejected(sample_features):
    df = sample_features.copy()
    df["DepDelay"] = [25, 0]

    with pytest.raises(ValueError):
        build_baseline_features(df)


def test_missing_input_value(sample_features):
    df = sample_features.copy()
    df.loc[0, "Origin"] = None

    with pytest.raises(ValueError, match="Missing values"):
        build_baseline_features(df)


def test_input_not_modified(sample_features):
    original = sample_features.copy(deep=True)

    build_baseline_features(sample_features)

    assert_frame_equal(sample_features, original)


def test_deterministic_output(sample_features):
    first = build_baseline_features(sample_features)
    second = build_baseline_features(sample_features)

    assert_frame_equal(first, second)
