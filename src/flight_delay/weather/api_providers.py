"""Historical forecast request adapters; no I/O or HRRR contract changes.

These are candidate request plans, not evidence of historical availability.
All production candidates remain blocked pending the evidence in docs/v5_api.md.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import pandas as pd

from flight_delay.weather.config import AIRPORTS

API_FEATURES = (
    "origin_temperature_c",
    "origin_wind_speed_10m",
    "origin_wind_gust",
    "origin_precipitation_1h",
)
AIRPORT_BY_CODE = {airport.code: airport for airport in AIRPORTS}


def utc_timestamp(value) -> pd.Timestamp:
    """Reject missing/naive times instead of guessing a timezone."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class ForecastQuery:
    """One origin's forecast at scheduled departure, decided exactly T-24h."""

    origin: str
    scheduled_departure_utc: pd.Timestamp
    features: tuple[str, ...] = API_FEATURES

    def __post_init__(self):
        if self.origin not in AIRPORT_BY_CODE:
            raise ValueError("Unsupported origin airport")
        object.__setattr__(
            self,
            "scheduled_departure_utc",
            utc_timestamp(self.scheduled_departure_utc),
        )
        if (
            not isinstance(self.features, tuple)
            or not self.features
            or len(set(self.features)) != len(self.features)
            or not set(self.features).issubset(API_FEATURES)
        ):
            raise ValueError("Invalid, duplicate, or unsupported API features")

    @property
    def prediction_at_utc(self):
        return self.scheduled_departure_utc - pd.Timedelta(hours=24)

    @property
    def valid_at_utc(self):
        return self.scheduled_departure_utc.floor("h")


@dataclass(frozen=True)
class APIRequestPlan:
    """Secret-free request identity and explicit evidence blockers.

    available_at is a planning assumption, never a measured release timestamp.
    A raw JSON response must not be passed to the HRRR feature/join pipeline.
    """

    provider: str
    model: str
    endpoint: str
    params: tuple[tuple[str, str], ...]
    run_at_utc: pd.Timestamp
    run_reference_kind: str
    assumed_available_at_utc: pd.Timestamp | None
    paid: bool
    blockers: tuple[str, ...]
    evidence_url: str

    @property
    def url(self) -> str:
        return f"{self.endpoint}?{urlencode(self.params)}"

    @property
    def request_key(self) -> str:
        # Forecast target/cutoff are intentionally not identity fields: several
        # flights can share a single airport/run/variable payload.
        payload = (self.provider, self.model, self.endpoint, sorted(self.params))
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ForecastAPIAdapter(Protocol):
    """Replace provider-specific URL/model selection behind this boundary."""

    def plan(self, query: ForecastQuery) -> APIRequestPlan: ...


class OpenMeteoSingleRunsAdapter:
    """Explicit IFS run requests, quarantined for hindcast/availability review."""

    _variables = {
        "origin_temperature_c": "temperature_2m",
        "origin_wind_speed_10m": "wind_speed_10m",
        "origin_wind_gust": "wind_gusts_10m",
        "origin_precipitation_1h": "precipitation",
    }

    def plan(self, query: ForecastQuery) -> APIRequestPlan:
        # Six hours is a conservative *candidate* global-model release lag,
        # not proof that the archived record existed then.
        run = (query.prediction_at_utc - pd.Timedelta(hours=6)).floor("6h")
        airport = AIRPORT_BY_CODE[query.origin]
        blockers = [
            "historical_operational_run_provenance_unverified_hindcast_warning",
            "historical_publication_time_unverified",
            "hourly_variables_units_and_interpolation_unverified",
        ]
        if run < pd.Timestamp("2024-03-14T00:00:00Z"):
            blockers.append("outside_documented_ifs_archive")
        return APIRequestPlan(
            provider="open_meteo_single_runs",
            model="ecmwf_ifs",
            endpoint="https://single-runs-api.open-meteo.com/v1/forecast",
            params=(
                ("latitude", str(airport.latitude)),
                ("longitude", str(airport.longitude)),
                ("models", "ecmwf_ifs"),
                ("run", run.strftime("%Y-%m-%dT%H:%M")),
                ("hourly", ",".join(self._variables[f] for f in query.features)),
                ("forecast_hours", "48"),
                ("timezone", "GMT"),
                ("temperature_unit", "celsius"),
                ("wind_speed_unit", "ms"),
                ("precipitation_unit", "mm"),
            ),
            run_at_utc=run,
            run_reference_kind="model_initialization_candidate",
            assumed_available_at_utc=run + pd.Timedelta(hours=6),
            paid=False,
            blockers=tuple(blockers),
            evidence_url="https://open-meteo.com/en/docs/single-runs-api",
        )


class VisualCrossingRunAdapter:
    """Daily archived forecast candidate; paid execution is not implemented."""

    _variables = {
        "origin_temperature_c": "temp",
        "origin_wind_speed_10m": "windspeed",
        "origin_wind_gust": "windgust",
        "origin_precipitation_1h": "precip",
    }

    def plan(self, query: ForecastQuery) -> APIRequestPlan:
        airport = AIRPORT_BY_CODE[query.origin]
        # Choose the preceding UTC day; midnight storage time is not a proven
        # model initialization or publication timestamp. Do not fabricate one.
        run_date = query.prediction_at_utc.floor("D") - pd.Timedelta(days=1)
        date = run_date.strftime("%Y-%m-%d")
        fields = ["rundatetime", "datetime", "datetimeEpoch"]
        fields.extend(self._variables[f] for f in query.features)
        blockers = [
            "paid_historical_forecast_addon_not_authorized",
            "daily_snapshot_initialization_and_publication_time_unverified",
            "model_specific_coverage_and_hourly_field_semantics_unverified",
        ]
        if run_date < pd.Timestamp("2020-01-01T00:00:00Z"):
            blockers.append("outside_documented_historical_forecast_archive")
        return APIRequestPlan(
            provider="visual_crossing_historical_forecast",
            model="gfs",
            endpoint=(
                "https://weather.visualcrossing.com/VisualCrossingWebServices/"
                f"rest/api/v1/historicalforecast/{airport.latitude},"
                f"{airport.longitude}/byrun/{date}/{date}"
            ),
            params=(
                ("unitGroup", "metric"),
                ("contentType", "json"),
                ("include", "hours"),
                ("forecastdays", "3"),
                ("forecastDataset", "gfs"),
                ("elements", ",".join(fields)),
            ),
            run_at_utc=run_date,
            run_reference_kind="daily_snapshot_basis_date_not_initialization",
            assumed_available_at_utc=None,
            paid=True,
            blockers=tuple(blockers),
            evidence_url=(
                "https://www.visualcrossing.com/resources/documentation/"
                "weather-api/historical-forecast-api/"
            ),
        )
