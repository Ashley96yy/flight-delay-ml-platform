"""Strict MOS join guard sharing legacy identity/time utilities, not HRRR policy."""

import numpy as np
import pandas as pd

from flight_delay.weather.flight_join import JOIN_KEYS, _utc_datetime
from flight_delay.weather.iem_mos import MOSContractError


def audit_mos_for_flights(flight_mapping, forecasts):
    """Audit explicitly selected vintages at exact scheduled departure instants.

    No nearest-time/asof join, cycle-lag assumption, interpolation or forward
    fill. All parsed archive rows currently fail the publication evidence gate.
    """
    required_flights = set(JOIN_KEYS + ["scheduled_departure_utc"])
    required_weather = set(
        JOIN_KEYS
        + ["model", "availability_status", "published_at_utc", "publication_evidence"]
    )
    if not required_flights <= set(flight_mapping) or not required_weather <= set(
        forecasts
    ):
        raise MOSContractError("Missing MOS flight audit identity or provenance")
    flights, weather = flight_mapping.copy(), forecasts.copy()
    if flights.empty:
        raise MOSContractError("Empty flight sample")
    if weather["model"].nunique() != 1:
        raise MOSContractError(
            "Audit one model at a time; never merge GFS and NBS identities"
        )
    for col in [
        "scheduled_departure_utc",
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
    ]:
        flights[col] = _utc_datetime(flights, col)
    for col in ["forecast_run_at_utc", "forecast_valid_at_utc"]:
        weather[col] = _utc_datetime(weather, col)
    if flights[JOIN_KEYS].isna().any().any() or weather[JOIN_KEYS].isna().any().any():
        raise MOSContractError("Missing MOS forecast identity")
    if weather.duplicated(JOIN_KEYS).any():
        raise MOSContractError("Duplicate MOS weather identity")
    if set(forecasts.columns).intersection(flights.columns) - set(JOIN_KEYS):
        raise MOSContractError("Flight input contains reserved forecast columns")
    cutoff = flights["scheduled_departure_utc"] - pd.Timedelta(hours=24)
    for col in ("prediction_at_utc", "prediction_time_utc"):
        if col in flights and not _utc_datetime(flights, col).equals(cutoff):
            raise MOSContractError("Supplied prediction time differs from T-24h")
    flights["prediction_at_utc"] = cutoff
    flights["__mos_order"] = np.arange(len(flights))
    result = flights.merge(
        weather,
        on=JOIN_KEYS,
        how="left",
        validate="many_to_one",
        indicator="__mos_match",
    )
    result = result.sort_values("__mos_order").reset_index(drop=True)
    lead = (
        result["forecast_valid_at_utc"] - result["forecast_run_at_utc"]
    ).dt.total_seconds() / 3600
    identity_valid = lead.eq(result["forecast_hour"]) & lead.gt(0)
    valid_time = result["forecast_valid_at_utc"].eq(result["scheduled_departure_utc"])
    result["weather_matched"] = result["__mos_match"].eq("both") & valid_time
    publication = pd.to_datetime(result["published_at_utc"], utc=True)
    # Do not interpret naive publication timestamps as UTC, even if pandas can.
    nonmissing = result["published_at_utc"].dropna()
    if any(pd.Timestamp(t).tzinfo is None for t in nonmissing):
        raise MOSContractError("MOS publication timestamps must be timezone-aware")
    has_evidence = result["publication_evidence"].map(
        lambda v: isinstance(v, str) and bool(v.strip())
    )
    verified = (
        result["availability_status"].eq("availability_verified")
        & has_evidence
        & publication.notna()
    )
    result["status"] = np.select(
        [
            ~identity_valid,
            ~result["weather_matched"],
            result["forecast_run_at_utc"].gt(result["prediction_at_utc"]),
            ~verified,
            publication.lt(result["forecast_run_at_utc"]),
            publication.gt(result["prediction_at_utc"]),
        ],
        [
            "invalid_identity",
            "missing_exact_valid_time",
            "forecast_initialized_after_cutoff",
            "availability_unverified",
            "publication_before_initialization",
            "published_after_cutoff",
        ],
        default="ready",
    )
    result["ready"] = result["status"].eq("ready")
    return result.drop(columns=["__mos_order", "__mos_match"])


def join_mos_to_flights(flight_mapping, forecasts):
    """Fail closed until every selected MOS vintage has publication evidence."""
    result = audit_mos_for_flights(flight_mapping, forecasts)
    if not result["ready"].all():
        raise MOSContractError(
            f"MOS point-in-time audit failed: {result['status'].value_counts().to_dict()}"
        )
    return result
