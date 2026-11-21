
"""Tests for LightGBM feature preparation."""

import pandas as pd
import pytest

from flight_delay.features.build import build_baseline_features
from flight_delay.models.lightgbm_baseline import (
    fit_category_levels,
    prepare_lightgbm_features,
)


@pytest.fixture
def sample_features():
    raw = pd.DataFrame(
        {
            "FlightDate": [
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
            ],
            "Reporting_Airline": ["AA", "DL", "AA"],
            "Origin": ["DFW", "DEN", "DFW"],
            "Dest": ["ATL", "LAX", "ORD"],
            "CRSDepTime": [800, 1420, 2230],
            "CRSArrTime": [1110, 1655, 30],
            "CRSElapsedTime": [120, 165, 120],
            "Distance": [700, 900, 800],
        }
    )

    return build_baseline_features(raw)


def test_category_levels_learned_from_training(sample_features):
    levels = fit_category_levels(sample_features)

    assert set(levels) == {
        "Reporting_Airline",
        "Origin",
        "Dest",
    }

    assert set(levels["Reporting_Airline"]) == {
        "AA",
        "DL",
    }


def test_prepared_features_have_same_shape(sample_features):
    levels = fit_category_levels(sample_features)

    prepared = prepare_lightgbm_features(
        sample_features,
        levels,
    )

    assert prepared.shape == sample_features.shape
    assert list(prepared.columns) == list(sample_features.columns)


def test_categorical_dtypes(sample_features):
    levels = fit_category_levels(sample_features)

    prepared = prepare_lightgbm_features(
        sample_features,
        levels,
    )

    for column in levels:
        assert isinstance(
            prepared[column].dtype,
            pd.CategoricalDtype,
        )


def test_unknown_category_becomes_missing(sample_features):
    levels = fit_category_levels(sample_features)

    validation = sample_features.copy()
    validation.loc[0, "Dest"] = "ZZZ"

    prepared = prepare_lightgbm_features(
        validation,
        levels,
    )

    assert pd.isna(prepared.loc[0, "Dest"])
    assert "ZZZ" not in prepared["Dest"].cat.categories


def test_training_categories_not_modified(sample_features):
    levels = fit_category_levels(sample_features)

    validation = sample_features.copy()
    validation.loc[0, "Reporting_Airline"] = "NEW"

    prepare_lightgbm_features(validation, levels)

    assert "NEW" not in levels["Reporting_Airline"]
    assert set(levels["Reporting_Airline"]) == {
        "AA",
        "DL",
    }


def test_input_dataframe_not_mutated(sample_features):
    levels = fit_category_levels(sample_features)

    original = sample_features.copy(deep=True)

    prepare_lightgbm_features(
        sample_features,
        levels,
    )

    pd.testing.assert_frame_equal(
        sample_features,
        original,
    )


def test_missing_category_mapping_raises(sample_features):
    levels = fit_category_levels(sample_features)
    levels.pop("Dest")

    with pytest.raises(ValueError):
        prepare_lightgbm_features(
            sample_features,
            levels,
        )


def test_forbidden_feature_rejected(sample_features):
    invalid_features = sample_features.assign(
        DepDel15=1
    )

    with pytest.raises(ValueError):
        fit_category_levels(invalid_features)
