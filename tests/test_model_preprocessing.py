import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flight_delay.features.build import build_baseline_features
from flight_delay.models.preprocessing import (
    make_model_preprocessor,
    validate_model_features,
)


@pytest.fixture
def sample_X():
    raw = pd.DataFrame({
        "FlightDate": [
            "2023-01-02",
            "2023-07-15",
            "2023-11-20",
        ],
        "Reporting_Airline": ["UA", "AA", "UA"],
        "Origin": ["SFO", "DFW", "ATL"],
        "Dest": ["LAX", "ORD", "JFK"],
        "CRSDepTime": [930, 2330, 800],
        "CRSArrTime": [1100, 2400, 1500],
        "CRSElapsedTime": [90, 180, 330],
        "Distance": [350, 800, 2586],
    })

    return build_baseline_features(raw)


def test_preprocessing_output_is_sparse(sample_X):
    validate_model_features(sample_X)

    preprocessor = make_model_preprocessor()
    result = preprocessor.fit_transform(sample_X)

    assert sparse.issparse(result)
    assert result.shape[0] == 3
    assert result.shape[1] > 17


def test_unknown_categories_are_supported(sample_X):
    preprocessor = make_model_preprocessor()
    preprocessor.fit(sample_X)

    validation = sample_X.iloc[[0]].copy()

    validation["Reporting_Airline"] = "ZZ"
    validation["Origin"] = "ORD"
    validation["Dest"] = "BOS"

    result = preprocessor.transform(validation)

    categorical_slice = preprocessor.output_indices_[
        "categorical"
    ]

    assert result[:, categorical_slice].nnz == 0


def test_training_statistics_are_preserved(sample_X):
    preprocessor = make_model_preprocessor()
    preprocessor.fit(sample_X)

    scaler = preprocessor.named_transformers_["numeric"]
    original_mean = scaler.mean_.copy()

    validation = sample_X.iloc[[0]].copy()
    validation["Distance"] = 9999

    preprocessor.transform(validation)

    np.testing.assert_array_equal(
        scaler.mean_,
        original_mean,
    )


def test_target_column_rejected(sample_X):
    invalid = sample_X.copy()
    invalid["DepDel15"] = [0, 1, 0]

    with pytest.raises(ValueError, match="schema"):
        validate_model_features(invalid)


def test_missing_feature_rejected(sample_X):
    invalid = sample_X.drop(columns=["Distance"])

    with pytest.raises(ValueError, match="schema"):
        validate_model_features(invalid)


def test_missing_value_rejected(sample_X):
    invalid = sample_X.copy()
    invalid.loc[0, "Distance"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        validate_model_features(invalid)
