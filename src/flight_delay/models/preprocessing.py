import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.features.build import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


def validate_model_features(X: pd.DataFrame) -> None:
    """Validate the engineered model feature contract."""

    if not X.columns.is_unique:
        raise ValueError("Duplicate feature columns detected.")

    if tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature schema or column order mismatch."
        )

    if X.isna().any().any():
        raise ValueError("Model features contain missing values.")

    numeric_values = X[
        list(NUMERIC_FEATURE_COLUMNS)
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Nonfinite numerical features detected.")


def make_model_preprocessor() -> ColumnTransformer:
    """Create an unfitted preprocessing transformer."""

    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    numerical = StandardScaler()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical,
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            (
                "numeric",
                numerical,
                list(NUMERIC_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
