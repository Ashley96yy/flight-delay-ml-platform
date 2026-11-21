"""Request-only checks; no API responses or weather coverage are fabricated."""

from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from flight_delay.weather.api_providers import (
    API_FEATURES,
    ForecastQuery,
    OpenMeteoSingleRunsAdapter,
    VisualCrossingRunAdapter,
)


@pytest.mark.parametrize("bad", [None, "NaT", "2024-07-15T09:15:00"])
def test_query_rejects_missing_or_naive_departures(bad):
    with pytest.raises(ValueError, match="timezone-aware"):
        ForecastQuery("ATL", bad)


@pytest.mark.parametrize("origin", ["XXX", "JFK", None])
def test_query_uses_current_six_origin_contract(origin):
    with pytest.raises(ValueError, match="origin"):
        ForecastQuery(origin, "2024-07-15T09:15Z")


@pytest.mark.parametrize("features", [(), ("visibility",), API_FEATURES * 2, []])
def test_unsupported_missing_or_duplicate_features_rejected(features):
    with pytest.raises(ValueError, match="features"):
        ForecastQuery("ATL", "2024-07-15T09:15Z", features)


def test_timezone_conversion_and_t24_cutoff():
    query = ForecastQuery("SFO", "2024-07-15T02:15:00-07:00")
    assert query.prediction_at_utc == pd.Timestamp("2024-07-14T09:15Z")
    assert query.valid_at_utc == pd.Timestamp("2024-07-15T09:00Z")


def test_single_run_selection_uses_its_own_lag_not_hrrr_lag():
    query = ForecastQuery("ATL", "2024-07-15T09:15Z")
    plan = OpenMeteoSingleRunsAdapter().plan(query)
    assert plan.run_at_utc == pd.Timestamp("2024-07-14T00:00Z")
    assert plan.assumed_available_at_utc == pd.Timestamp("2024-07-14T06:00Z")
    assert plan.assumed_available_at_utc <= query.prediction_at_utc
    params = parse_qs(urlparse(plan.url).query)
    assert params["run"] == ["2024-07-14T00:00"]
    assert params["models"] == ["ecmwf_ifs"]
    assert params["wind_speed_unit"] == ["ms"]
    assert params["forecast_hours"] == ["48"]
    assert len(params["hourly"][0].split(",")) == 4
    assert plan.blockers  # A plausible timestamp is not historical evidence.


@pytest.mark.parametrize(
    "departure,missing",
    [
        ("2023-07-15T09:15Z", True),
        ("2024-03-14T09:15Z", True),  # Run falls before archive start.
        ("2024-03-15T09:15Z", False),
        ("2025-07-15T09:15Z", False),
    ],
)
def test_ifs_documented_start_is_checked_on_run_not_departure(departure, missing):
    plan = OpenMeteoSingleRunsAdapter().plan(ForecastQuery("ATL", departure))
    assert ("outside_documented_ifs_archive" in plan.blockers) == missing
    assert any("hindcast" in reason for reason in plan.blockers)


def test_visual_crossing_is_archived_run_not_observation_request():
    query = ForecastQuery("ATL", "2023-07-15T09:15Z")
    plan = VisualCrossingRunAdapter().plan(query)
    assert "/historicalforecast/" in plan.url
    assert "/byrun/2023-07-13/2023-07-13?" in plan.url
    assert plan.assumed_available_at_utc is None
    assert plan.paid and plan.blockers
    assert "key=" not in plan.url
    assert dict(plan.params)["forecastDataset"] == "gfs"


def test_visual_crossing_start_is_not_claimed_before_2020():
    plan = VisualCrossingRunAdapter().plan(ForecastQuery("ATL", "2020-01-01T09:15Z"))
    assert "outside_documented_historical_forecast_archive" in plan.blockers


def test_request_identity_deduplicates_shared_run_but_separates_sources():
    adapter = OpenMeteoSingleRunsAdapter()
    a = adapter.plan(ForecastQuery("ATL", "2024-07-15T09:15Z"))
    b = adapter.plan(ForecastQuery("ATL", "2024-07-15T09:45Z"))
    c = adapter.plan(ForecastQuery("SFO", "2024-07-15T09:15Z"))
    d = adapter.plan(ForecastQuery("ATL", "2024-07-15T15:15Z"))
    assert a.request_key == b.request_key
    assert len({a.request_key, c.request_key, d.request_key}) == 3
    assert replace(a, model="different").request_key != a.request_key
    assert replace(a, params=tuple(reversed(a.params))).request_key == a.request_key
