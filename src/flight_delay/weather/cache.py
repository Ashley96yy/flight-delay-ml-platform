"""Validated, content-addressed weather Parquet cache."""

import hashlib
import json
import os
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from uuid import uuid4

import pandas as pd

from flight_delay.weather.config import (
    AIRPORTS,
    GRIB_BASE_URL,
    RAW_WEATHER_COLUMNS,
    SUPPORTED_FORECAST_LEADS,
    WEATHER_AVAILABILITY_LAG_HOURS,
    WEATHER_CONTRACT_VERSION,
    WEATHER_VARIABLES,
    ZARR_BUCKET,
)
from flight_delay.weather.contracts import (
    WeatherContractError,
    validate_hourly_weather,
)

CACHE_SCHEMA_VERSION = 1


class WeatherCacheError(RuntimeError):
    """Base error for weather cache operations."""


class WeatherCacheIntegrityError(WeatherCacheError):
    """Stored cache contents failed verification."""


class WeatherCacheValidationError(WeatherCacheError):
    """Weather records or cache identity are invalid."""


def cache_fingerprint() -> str:
    """Identify the weather contract and acquisition configuration."""

    configuration = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "weather_contract_version": WEATHER_CONTRACT_VERSION,
        "availability_lag_hours": (WEATHER_AVAILABILITY_LAG_HOURS),
        "forecast_leads": list(SUPPORTED_FORECAST_LEADS),
        "raw_variables": list(RAW_WEATHER_COLUMNS),
        "variable_definitions": [asdict(variable) for variable in WEATHER_VARIABLES],
        "airports": [asdict(airport) for airport in AIRPORTS],
        "zarr_bucket": ZARR_BUCKET,
        "grib_base_url": GRIB_BASE_URL,
    }

    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:16]


def validate_cache_identity(run_at, lead):
    """Require an eligible UTC forecast initialization and lead."""

    try:
        run = pd.Timestamp(run_at)
    except (ValueError, TypeError, OverflowError) as exc:
        raise WeatherCacheValidationError("Invalid initialization timestamp") from exc

    if pd.isna(run) or run.tzinfo is None:
        raise WeatherCacheValidationError("Timezone-aware initialization required")

    run = run.tz_convert("UTC")

    if run.hour not in (0, 6, 12, 18) or run != run.floor("h"):
        raise WeatherCacheValidationError("Invalid extended forecast cycle")

    if (
        not isinstance(lead, Integral)
        or isinstance(lead, bool)
        or lead not in SUPPORTED_FORECAST_LEADS
    ):
        raise WeatherCacheValidationError("Unsupported forecast lead")

    return run, int(lead)


def sha256_file(path: Path) -> str:
    """Compute a streaming checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_cached_hour(
    frame: pd.DataFrame,
    run_at: pd.Timestamp,
    lead: int,
) -> None:
    """Check completeness, source identity and temporal identity."""

    try:
        validate_hourly_weather(frame)
    except (WeatherContractError, KeyError, TypeError) as exc:
        raise WeatherCacheValidationError(
            "Cached weather violates the data contract"
        ) from exc

    if not frame["forecast_run_at_utc"].eq(run_at).all():
        raise WeatherCacheValidationError("Cached forecast initialization mismatch")

    if not frame["forecast_hour"].eq(lead).all():
        raise WeatherCacheValidationError("Cached forecast lead mismatch")

    if not frame["backend"].nunique(dropna=False) == 1:
        raise WeatherCacheValidationError(
            "Mixed backends within one hourly cache entry"
        )

    if frame["fallback_reason"].nunique(dropna=False) != 1:
        raise WeatherCacheValidationError("Inconsistent fallback provenance")


class WeatherCache:
    """Store complete hourly weather batches with a commit manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.fingerprint = cache_fingerprint()
        self.namespace = self.root / self.fingerprint

    def entry_directory(self, run_at, lead) -> Path:
        run, lead = validate_cache_identity(run_at, lead)

        name = f"hrrr_{run.strftime('%Y%m%d_%H')}_f{lead:02d}"

        return self.namespace / name

    def read_hour(
        self,
        run_at,
        lead,
    ) -> pd.DataFrame | None:
        """Read verified weather or return None for a cache miss."""

        run, lead = validate_cache_identity(run_at, lead)

        directory = self.entry_directory(run, lead)
        manifest_path = directory / "manifest.json"

        # No committed manifest means no completed cache entry.
        # Orphan temporary files are never treated as valid data.

        if not manifest_path.is_file():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise WeatherCacheIntegrityError("Unreadable cache manifest") from exc

        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "run_at": run.isoformat(),
            "forecast_hour": lead,
            "record_count": (len(AIRPORTS) * len(RAW_WEATHER_COLUMNS)),
        }

        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in expected.items()
        ):
            raise WeatherCacheIntegrityError("Cache manifest identity mismatch")

        checksum = manifest.get("sha256")
        filename = manifest.get("filename")

        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)
            or filename != f"weather-{checksum}.parquet"
        ):
            raise WeatherCacheIntegrityError("Invalid cached payload reference")

        payload_path = directory / filename

        if not payload_path.is_file():
            raise WeatherCacheIntegrityError("Committed weather payload is missing")

        if sha256_file(payload_path) != checksum:
            raise WeatherCacheIntegrityError("Cached weather checksum mismatch")

        try:
            frame = pd.read_parquet(payload_path)
        except Exception as exc:
            raise WeatherCacheIntegrityError("Unreadable cached Parquet") from exc

        try:
            _validate_cached_hour(frame, run, lead)
        except WeatherCacheValidationError as exc:
            raise WeatherCacheIntegrityError(
                "Cached weather failed contract validation"
            ) from exc

        if frame["backend"].iloc[0] != manifest.get("backend"):
            raise WeatherCacheIntegrityError("Cached backend provenance mismatch")

        return frame

    def write_hour(
        self,
        run_at,
        lead,
        weather: pd.DataFrame,
    ) -> Path:
        """Atomically commit a validated hourly weather batch."""

        run, lead = validate_cache_identity(run_at, lead)

        if not isinstance(weather, pd.DataFrame):
            raise WeatherCacheValidationError("Weather must be a DataFrame")

        frame = weather.copy(deep=True)

        _validate_cached_hour(frame, run, lead)

        frame = frame.sort_values(["feature", "Origin"]).reset_index(drop=True)

        directory = self.entry_directory(run, lead)
        directory.mkdir(parents=True, exist_ok=True)

        token = uuid4().hex

        temporary_parquet = directory / f".{token}.parquet.tmp"
        temporary_manifest = directory / f".{token}.json.tmp"

        try:
            frame.to_parquet(
                temporary_parquet,
                index=False,
            )

            # Ensure payload bytes are flushed before committing
            # the manifest that will refer to them.

            with temporary_parquet.open("rb") as handle:
                os.fsync(handle.fileno())

            checksum = sha256_file(temporary_parquet)

            filename = f"weather-{checksum}.parquet"
            destination = directory / filename

            os.replace(
                temporary_parquet,
                destination,
            )

            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "run_at": run.isoformat(),
                "forecast_hour": lead,
                "record_count": len(frame),
                "backend": frame["backend"].iloc[0],
                "sha256": checksum,
                "filename": filename,
            }

            with temporary_manifest.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    manifest,
                    handle,
                    sort_keys=True,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())

            # Last operation: publish the committed manifest.
            os.replace(
                temporary_manifest,
                directory / "manifest.json",
            )

            return destination

        finally:
            temporary_parquet.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    def read_cycle(self, run_at, forecast_leads):
        """Return per-hour hits and misses without network access."""

        return {int(lead): self.read_hour(run_at, lead) for lead in forecast_leads}
