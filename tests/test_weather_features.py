"""Offline tests for model-ready weather features."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.config import AIRPORTS, WEATHER_VARIABLES
from flight_delay.weather.features import (
    MODEL_WEATHER_COLUMNS,
    WeatherFeatureError,
    build_weather_feature_table,
)

RUN = pd.Timestamp("2024-07-14T06:00:00Z")

RAW_VALUES = {
    "temperature_2m": 300.0,
    "u_wind_10m": 3.0,
    "v_wind_10m": 4.0,
    "wind_gust": 7.0,
    "visibility": 12000.0,
    "precipitation_1h": 2.5,
    "surface_cape": 100.0,
    "composite_reflectivity": -10.0,
    "cloud_ceiling_raw": np.nan,
}


def make_weather(leads=(30,), backend="zarr"):
    rows = []

    for lead in leads:
        for airport in AIRPORTS:
            for variable in WEATHER_VARIABLES:
                value = RAW_VALUES[variable.feature]
                missing = pd.isna(value)

                rows.append(
                    {
                        "Origin": airport.code,
                        "feature": variable.feature,
                        "forecast_run_at_utc": RUN,
                        "forecast_valid_at_utc": (RUN + pd.Timedelta(hours=lead)),
                        "forecast_hour": lead,
                        "assumed_available_at_utc": (RUN + pd.Timedelta(hours=3)),
                        "value": value,
                        "raw_value": (-9999.0 if missing else value),
                        "value_status": (
                            "source_missing"
                            if missing and backend == "grib"
                            else "fill_value"
                            if missing
                            else "valid"
                        ),
                        "grid_latitude": airport.latitude,
                        "grid_longitude": airport.longitude,
                        "backend": backend,
                        "storage_dtype": ("float32" if backend == "zarr" else None),
                        "units": variable.grib_units,
                        "source_units": variable.grib_units,
                        "fallback_reason": None,
                    }
                )

    return pd.DataFrame(rows)


def test_single_hour_produces_six_rows():
    result = build_weather_feature_table(make_weather())

    assert len(result) == 6
    assert set(MODEL_WEATHER_COLUMNS).issubset(result.columns)


def test_temperature_conversion():
    result = build_weather_feature_table(make_weather())

    np.testing.assert_allclose(
        result["origin_temperature_c"],
        26.85,
        atol=1e-10,
    )


def test_wind_vector_magnitude():
    result = build_weather_feature_table(make_weather())

    assert result["origin_wind_speed_10m"].eq(5.0).all()


def test_visibility_conversion():
    result = build_weather_feature_table(make_weather())

    assert result["origin_visibility_km"].eq(12.0).all()


def test_other_features_preserve_values():
    result = build_weather_feature_table(make_weather())

    expected = {
        "origin_wind_gust": 7.0,
        "origin_precipitation_1h": 2.5,
        "origin_surface_cape": 100.0,
        "origin_composite_reflectivity": -10.0,
    }

    for column, value in expected.items():
        assert result[column].eq(value).all()


def test_optional_ceiling_missing_is_allowed():
    result = build_weather_feature_table(make_weather())

    assert len(result) == 6
    assert result[MODEL_WEATHER_COLUMNS].notna().all().all()


def test_multiple_forecast_leads():
    result = build_weather_feature_table(make_weather(leads=(27, 30)))

    assert len(result) == 12
    assert result.groupby("forecast_hour").size().eq(6).all()


def test_grib_weather_is_supported():
    result = build_weather_feature_table(make_weather(backend="grib"))

    assert result["backend"].eq("grib").all()


def test_duplicate_raw_record_rejected():
    weather = make_weather()

    weather = pd.concat(
        [weather, weather.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        WeatherFeatureError,
        match="Duplicate weather feature identity",
    ):
        build_weather_feature_table(weather)


def test_missing_variable_rejected():
    weather = make_weather()

    weather = weather.loc[weather["feature"] != "wind_gust"]

    with pytest.raises(
        WeatherFeatureError,
        match="Hourly weather contract failed",
    ):
        build_weather_feature_table(weather)


def test_missing_core_value_rejected():
    weather = make_weather()

    row = weather.index[weather["feature"] == "temperature_2m"][0]

    weather.loc[row, "value"] = np.nan
    weather.loc[row, "value_status"] = "source_missing"

    with pytest.raises(WeatherFeatureError):
        build_weather_feature_table(weather)


def test_mixed_backend_within_hour_rejected():
    weather = make_weather()
    weather.loc[0, "backend"] = "grib"

    with pytest.raises(
        WeatherFeatureError,
        match="Mixed backend provenance",
    ):
        build_weather_feature_table(weather)


def test_input_not_modified():
    weather = make_weather()
    original = weather.copy(deep=True)

    build_weather_feature_table(weather)

    pd.testing.assert_frame_equal(weather, original)


def test_empty_input_rejected():
    with pytest.raises(
        WeatherFeatureError,
        match="nonempty DataFrame",
    ):
        build_weather_feature_table(pd.DataFrame())
