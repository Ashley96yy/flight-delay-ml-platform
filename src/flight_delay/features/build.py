import numpy as np
import pandas as pd

from flight_delay.features.schema import (
    BASELINE_FEATURES,
    validate_baseline_features,
)

CATEGORICAL_FEATURE_COLUMNS = (
    "Reporting_Airline",
    "Origin",
    "Dest",
)

NUMERIC_FEATURE_COLUMNS = (
    "CRSElapsedTime",
    "Distance",
    "month",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "dep_time_sin",
    "dep_time_cos",
    "arr_time_sin",
    "arr_time_cos",
)

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS
)


def hhmm_to_minutes(values: pd.Series) -> pd.Series:
    """Convert BTS HHMM times to minutes after midnight."""

    hhmm = pd.to_numeric(values, errors="raise")

    if hhmm.isna().any():
        raise ValueError("Scheduled times contain missing values.")

    valid_numbers = (
        hhmm.ge(0)
        & hhmm.le(2400)
        & hhmm.mod(1).eq(0)
    )

    if not valid_numbers.all():
        raise ValueError("Invalid HHMM scheduled time.")

    hhmm = hhmm.astype("int32")

    valid_time = (
        (
            hhmm.lt(2400)
            & (hhmm % 100).lt(60)
        )
        | hhmm.eq(2400)
    )

    if not valid_time.all():
        raise ValueError("Invalid HHMM scheduled time.")

    # Represent 24:00 as midnight on the daily clock.
    hhmm = hhmm.where(hhmm != 2400, 0)

    return (hhmm // 100) * 60 + (hhmm % 100)


def build_baseline_features(
    X_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build scheduled-flight baseline features.

    Requires the eight approved input columns.
    Does not fit parameters or use target labels.
    """

    if not X_raw.columns.is_unique:
        raise ValueError("Input columns must be unique.")

    missing = set(BASELINE_FEATURES) - set(X_raw.columns)

    if missing:
        raise ValueError(
            f"Missing required baseline columns: {sorted(missing)}"
        )

    # Reject additional fields, including actual outcomes.
    validate_baseline_features(X_raw.columns)

    if X_raw.isna().any().any():
        raise ValueError(
            "Missing values in baseline input features."
        )

    features = X_raw[
        [
            "Reporting_Airline",
            "Origin",
            "Dest",
            "CRSElapsedTime",
            "Distance",
        ]
    ].copy()

    for column in ("CRSElapsedTime", "Distance"):
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    # Calendar features.
    dates = pd.to_datetime(
        X_raw["FlightDate"],
        errors="raise",
    )

    features["month"] = dates.dt.month
    features["day_of_week"] = dates.dt.day_of_week
    features["day_of_month"] = dates.dt.day

    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype("int8")

    # Cyclical calendar features.
    month_angle = (
        2 * np.pi * (features["month"] - 1) / 12
    )

    weekday_angle = (
        2 * np.pi * features["day_of_week"] / 7
    )

    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    features["weekday_sin"] = np.sin(weekday_angle)
    features["weekday_cos"] = np.cos(weekday_angle)

    # Scheduled time features.
    dep_minutes = hhmm_to_minutes(
        X_raw["CRSDepTime"]
    )
    arr_minutes = hhmm_to_minutes(
        X_raw["CRSArrTime"]
    )

    dep_angle = 2 * np.pi * dep_minutes / 1440
    arr_angle = 2 * np.pi * arr_minutes / 1440

    features["dep_time_sin"] = np.sin(dep_angle)
    features["dep_time_cos"] = np.cos(dep_angle)

    features["arr_time_sin"] = np.sin(arr_angle)
    features["arr_time_cos"] = np.cos(arr_angle)

    # Guarantee stable column order.
    features = features.loc[:, list(FEATURE_COLUMNS)]

    if features.isna().any().any():
        raise ValueError("Engineered features contain missing values.")

    numeric_values = features[
        list(NUMERIC_FEATURE_COLUMNS)
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Engineered features contain nonfinite values.")

    return features
