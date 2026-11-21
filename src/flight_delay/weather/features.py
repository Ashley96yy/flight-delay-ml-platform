"""Build model-ready weather features from validated HRRR records."""

import numpy as np
import pandas as pd

from flight_delay.weather.config import RAW_WEATHER_COLUMNS
from flight_delay.weather.contracts import (
    WeatherContractError,
    validate_hourly_weather,
)


class WeatherFeatureError(ValueError):
    """Invalid or incomplete weather feature input."""


WEATHER_KEYS = [
    "Origin",
    "forecast_run_at_utc",
    "forecast_valid_at_utc",
    "assumed_available_at_utc",
    "forecast_hour",
]

MODEL_WEATHER_COLUMNS = [
    "origin_temperature_c",
    "origin_wind_speed_10m",
    "origin_wind_gust",
    "origin_visibility_km",
    "origin_precipitation_1h",
    "origin_surface_cape",
    "origin_composite_reflectivity",
]

REQUIRED_RAW_FEATURES = [
    feature for feature in RAW_WEATHER_COLUMNS if feature != "cloud_ceiling_raw"
]


def build_weather_feature_table(
    weather_long: pd.DataFrame,
) -> pd.DataFrame:
    """Convert complete hourly weather batches to model features."""

    if not isinstance(weather_long, pd.DataFrame) or weather_long.empty:
        raise WeatherFeatureError("Weather input must be a nonempty DataFrame")

    required_columns = set(
        WEATHER_KEYS + ["feature", "value", "backend", "fallback_reason"]
    )

    missing_columns = required_columns - set(weather_long.columns)

    if missing_columns:
        raise WeatherFeatureError(f"Missing weather columns: {sorted(missing_columns)}")

    weather = weather_long.copy(deep=True)

    identity = WEATHER_KEYS + ["feature"]

    if weather.duplicated(identity).any():
        raise WeatherFeatureError("Duplicate weather feature identity")

    # Validate each forecast hour independently.
    # Every hour must represent six airports and nine variables.

    for (_, _), hourly in weather.groupby(
        ["forecast_run_at_utc", "forecast_hour"],
        sort=False,
        dropna=False,
    ):
        if (
            hourly["backend"].nunique(dropna=False) != 1
            or hourly["fallback_reason"].nunique(dropna=False) != 1
        ):
            raise WeatherFeatureError("Mixed backend provenance within a forecast hour")

        try:
            validate_hourly_weather(hourly)
        except WeatherContractError as exc:
            raise WeatherFeatureError("Hourly weather contract failed") from exc

    # Enforce the T-24-hour feature availability boundary.
    # Forecast valid time is the floored scheduled departure hour.

    earliest_prediction_time = weather["forecast_valid_at_utc"] - pd.Timedelta(hours=24)

    if not weather["assumed_available_at_utc"].le(earliest_prediction_time).all():
        raise WeatherFeatureError("Weather forecast is unavailable at T-24h")

    # Pivot, rather than pivot_table, so duplicate rows
    # cannot be silently averaged.

    wide = weather.pivot(
        index=WEATHER_KEYS,
        columns="feature",
        values="value",
    ).reset_index()

    wide.columns.name = None

    missing_features = set(RAW_WEATHER_COLUMNS) - set(wide.columns)

    if missing_features:
        raise WeatherFeatureError(
            f"Missing raw weather features: {sorted(missing_features)}"
        )

    if wide[REQUIRED_RAW_FEATURES].isna().any().any():
        raise WeatherFeatureError("Missing required raw weather values")

    # Perform explicit unit conversions.

    wide["origin_temperature_c"] = wide["temperature_2m"] - 273.15

    wide["origin_wind_speed_10m"] = np.hypot(
        wide["u_wind_10m"],
        wide["v_wind_10m"],
    )

    wide["origin_wind_gust"] = wide["wind_gust"]

    wide["origin_visibility_km"] = wide["visibility"] / 1000.0

    wide["origin_precipitation_1h"] = wide["precipitation_1h"]

    wide["origin_surface_cape"] = wide["surface_cape"]

    wide["origin_composite_reflectivity"] = wide["composite_reflectivity"]

    values = wide[MODEL_WEATHER_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise WeatherFeatureError("Nonfinite model weather features")

    # Carry source provenance into the feature table.

    provenance_columns = WEATHER_KEYS + [
        "backend",
        "fallback_reason",
    ]

    provenance = weather[provenance_columns].drop_duplicates()

    if len(provenance) != len(wide):
        raise WeatherFeatureError("Inconsistent weather source provenance")

    result = wide[WEATHER_KEYS + MODEL_WEATHER_COLUMNS].merge(
        provenance,
        on=WEATHER_KEYS,
        how="left",
        validate="one_to_one",
    )

    return result.sort_values(
        [
            "forecast_run_at_utc",
            "forecast_hour",
            "Origin",
        ]
    ).reset_index(drop=True)
