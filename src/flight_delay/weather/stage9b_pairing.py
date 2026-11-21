"""Build the frozen same-sample Stage 9B matrices without fitting any model."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.features.advanced_scheduled import (
    INTERACTION_FEATURE_COLUMNS,
    build_advanced_scheduled_features,
)
from flight_delay.features.build import (
    CATEGORICAL_FEATURE_COLUMNS,
    build_baseline_features,
)
from flight_delay.features.schema import BASELINE_FEATURES
from flight_delay.weather.cache import WeatherCache, sha256_file
from flight_delay.weather.config import ORIGIN_AIRPORTS
from flight_delay.weather.features import (
    MODEL_WEATHER_COLUMNS,
    build_weather_feature_table,
)
from flight_delay.weather.flight_join import JOIN_KEYS, join_weather_to_flights
from flight_delay.weather.flight_times import convert_bts_departure_to_utc
from flight_delay.weather.planner import build_flight_forecast_mapping
from flight_delay.weather.stage9b_protocol import allowed_hours, frozen_protocol

PIT_STATUS = "retrospective_exploratory"
TARGETS = ["DepDel15", "ArrDel15"]
IDENTITY = sorted(BASELINE_FEATURES | {"Flight_Number_Reporting_Airline"})
CATEGORICAL = (*CATEGORICAL_FEATURE_COLUMNS, "Route", "Airline_Route")


@dataclass(frozen=True)
class PairedInputs:
    manifest: pd.DataFrame
    baseline: pd.DataFrame
    weather: pd.DataFrame
    training_weather_profile: pd.DataFrame
    params_sha256: str
    manifest_sha256: str
    pit_status: str = PIT_STATUS


def _flight_identity(values):
    return hashlib.sha256(json.dumps(list(map(str, values))).encode()).hexdigest()


def _require_daily_binary_labels(frame):
    if not frame[TARGETS].isin([0, 1]).all().all():
        raise ValueError("Frozen cohort has missing or invalid binary labels")
    counts = frame.groupby(["split", "date"])[TARGETS].nunique()
    if len(counts) != 24 or not counts.eq(2).all().all():
        raise ValueError("Each frozen date needs both classes for both targets")


def load_frozen_cohort(data_dir):
    """Select dates and flights from schedule columns only, then attach labels."""
    source_hashes = frozen_protocol()["source_hashes"]
    cohorts = []
    for key, day in allowed_hours().items():
        date = pd.Timestamp(day["date"])
        file = data_dir / f"bts_{date:%Y_%m}_completed.parquet"
        digest = sha256_file(file)
        if source_hashes.get(file.name) != digest:
            raise ValueError(
                f"BTS source is outside the verified Stage 8 inputs: {file.name}"
            )
        frame = pd.read_parquet(file, columns=IDENTITY + TARGETS)
        local_date = pd.to_datetime(frame.FlightDate, errors="raise").dt.normalize()
        frame = frame.loc[local_date.eq(date) & frame.Origin.isin(ORIGIN_AIRPORTS)]
        frame = convert_bts_departure_to_utc(frame).flights
        departure = frame.scheduled_departure_utc.dt
        frame = frame.loc[
            departure.hour.eq(13) & departure.normalize().dt.tz_localize(None).eq(date)
        ].copy()
        expected_counts = {airport: int(day[airport]) for airport in ORIGIN_AIRPORTS}
        if frame.Origin.value_counts().to_dict() != expected_counts:
            raise ValueError(f"Frozen schedule support changed on {date.date()}")
        mapping = build_flight_forecast_mapping(frame)
        if set(
            mapping[
                ["forecast_run_at_utc", "forecast_hour", "forecast_valid_at_utc"]
            ].itertuples(index=False, name=None)
        ) != {key}:
            raise ValueError(
                f"Frozen forecast key does not match the planner on {date.date()}"
            )
        frame["flight_id"] = [
            _flight_identity(row)
            for row in frame[IDENTITY].itertuples(index=False, name=None)
        ]
        frame["date"] = day["date"]
        frame["split"] = day["split"]
        frame["bts_sha256"] = digest
        cohorts.append(frame.sort_values("flight_id"))

    result = pd.concat(cohorts, ignore_index=True)
    if not result.flight_id.is_unique:
        raise ValueError("Duplicate flight ID in the frozen cohort")
    _require_daily_binary_labels(result)
    return result


def _cache_weather(cache):
    parts, evidence = [], []
    for run, lead, valid in allowed_hours():
        entry = cache.entry_directory(run, lead) / "manifest.json"
        if not entry.is_file():
            raise ValueError(f"Missing Stage 9B weather cache for {valid.isoformat()}")
        receipt = entry.read_bytes()
        raw = cache.read_hour(run, lead)
        if raw is None or entry.read_bytes() != receipt:
            raise ValueError("Cache changed while preparing paired weather")
        if not raw["forecast_valid_at_utc"].eq(valid).all():
            raise ValueError("Cache does not match frozen valid time")
        manifest = json.loads(receipt)
        features = build_weather_feature_table(raw)
        parts.append(features)
        evidence.append(
            features[JOIN_KEYS].assign(
                weather_cache_sha256=manifest["sha256"],
                weather_manifest_sha256=hashlib.sha256(receipt).hexdigest(),
                weather_cache_fingerprint=manifest["fingerprint"],
            )
        )
    return pd.concat(parts, ignore_index=True), pd.concat(evidence, ignore_index=True)


def _require_training_variation(weather):
    training = weather.loc[weather.forecast_valid_at_utc.dt.year.eq(2023)]
    grouped = training.groupby("Origin")[MODEL_WEATHER_COLUMNS]
    counts = grouped.nunique()
    if int(counts.gt(1).sum().ge(4).sum()) < 2:
        raise ValueError(
            "Fewer than two weather fields vary across dates at four training airports"
        )
    return pd.concat(
        {
            "unique": counts,
            "min": grouped.min(),
            "max": grouped.max(),
            "iqr": grouped.quantile(0.75) - grouped.quantile(0.25),
        },
        axis=1,
    )


def prepare_pairs(data_dir: Path, cache_root: Path, params: Path) -> PairedInputs:
    """Validate row identity, schedule and every weather hour before pairing."""
    protocol = frozen_protocol()
    digest = sha256_file(params)
    if digest != protocol["params_sha256"]:
        raise ValueError("Stage 8 frozen parameters changed")
    cohort = load_frozen_cohort(data_dir)
    features, evidence = _cache_weather(WeatherCache(cache_root))
    weather_profile = _require_training_variation(features)
    mapping = build_flight_forecast_mapping(cohort)
    joined = join_weather_to_flights(mapping, features)
    if joined.flight_row_position.tolist() != list(range(len(cohort))):
        raise ValueError("Joined cohort changed identity, order or row count")
    manifest = joined[
        JOIN_KEYS
        + [
            "scheduled_departure_utc",
            "prediction_time_utc",
            "weather_assumed_available_at_utc",
            "weather_backend",
        ]
    ].merge(
        evidence,
        on=JOIN_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if not manifest[JOIN_KEYS].equals(joined[JOIN_KEYS]):
        raise ValueError("Cache evidence merge changed forecast keys or order")
    if manifest.weather_cache_sha256.isna().any():
        raise ValueError("Missing forecast cache evidence")
    manifest = pd.concat(
        [cohort[["flight_id", "split", "date", "bts_sha256", *TARGETS]], manifest],
        axis=1,
    ).set_index("flight_id")
    manifest["pit_status"] = PIT_STATUS

    baseline = build_baseline_features(cohort[sorted(BASELINE_FEATURES)])
    advance = build_advanced_scheduled_features(cohort)
    baseline = pd.concat([baseline, advance[list(INTERACTION_FEATURE_COLUMNS)]], axis=1)
    if list(baseline.columns) != protocol["feature_columns"]:
        raise ValueError("Stage 8 V3-A feature contract changed")
    mask = cohort["split"].eq("train")
    for column in CATEGORICAL:
        values = baseline[column].astype("string")
        levels = sorted(values.loc[mask].dropna().unique())
        baseline[column] = values.where(values.isin(levels)).astype(
            pd.CategoricalDtype(levels)
        )
    baseline.index = manifest.index
    weather = joined[MODEL_WEATHER_COLUMNS].set_axis(manifest.index)
    if not np.isfinite(weather.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite paired weather feature")
    augmented = pd.concat([baseline, weather], axis=1)
    return PairedInputs(
        manifest,
        baseline,
        augmented,
        weather_profile,
        digest,
        hashlib.sha256(manifest.to_csv().encode()).hexdigest(),
    )
