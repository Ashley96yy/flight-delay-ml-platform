
"""Feature preparation utilities for LightGBM."""

from collections.abc import Mapping

import pandas as pd

from flight_delay.features.build import (
    CATEGORICAL_FEATURE_COLUMNS,
)
from flight_delay.models.preprocessing import (
    validate_model_features,
)


def fit_category_levels(
    X_train: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    """Learn categorical levels from training data only."""
    validate_model_features(X_train)

    return {
        col: tuple(
            X_train[col]
            .astype("string")
            .dropna()
            .unique()
            .tolist()
        )
        for col in CATEGORICAL_FEATURE_COLUMNS
    }


def prepare_lightgbm_features(
    X: pd.DataFrame,
    category_levels: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Convert categorical columns using train-derived levels."""
    validate_model_features(X)

    X_out = X.copy()

    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in category_levels:
            raise ValueError(
                f"Missing category levels for {col}"
            )

        dtype = pd.CategoricalDtype(
            categories=list(category_levels[col]),
            ordered=False,
        )

        values = X_out[col].astype("string")
        known_values = values.isin(dtype.categories)

        X_out[col] = (
            values
            .where(known_values, pd.NA)
            .astype(dtype)
        )

    return X_out
