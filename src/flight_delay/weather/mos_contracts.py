"""Independent MOS contract: categories and probabilities retain their meaning."""

import hashlib
import json

import numpy as np
import pandas as pd

from flight_delay.weather.api_providers import utc_timestamp
from flight_delay.weather.iem_mos import MOSContractError, validate_plan

CONTRACT_VERSION = "iem_mos_v1_research"
RAW_FIELDS = ("tmp", "wsp", "gst", "vis", "cig", "lcb", "p06", "t06_1", "t03")
FEATURE_COLUMNS = (
    "mos_temperature_c",
    "mos_wind_speed_mps",
    "mos_gust_mps",
    "mos_gfs_visibility_category",
    "mos_gfs_ceiling_category",
    "mos_nbm_visibility_miles",
    "mos_nbm_ceiling_hundreds_ft",
    "mos_nbm_ceiling_unlimited",
    "mos_pop_6h_pct",
    "mos_tstorm_6h_pct",
    "mos_tstorm_3h_pct",
)


def _utc_from_explicit_api_field(value):
    # IEM handler explicitly SELECTs runtime/ftime AT TIME ZONE 'UTC' into
    # these *_utc fields, then pandas emits timezone-less UTC strings. This
    # source-specific interpretation is not a global naive-time relaxation.
    if not isinstance(value, str) or not value:
        raise MOSContractError("Missing explicit IEM UTC field")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise MOSContractError("Missing explicit IEM UTC field")
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _decode(value, *, missing, minimum=None, maximum=None):
    if value is None:
        return np.nan, "source_missing"
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise MOSContractError("Non-numeric or nonfinite MOS encoding")
    if value in missing:
        return np.nan, "missing_code"
    if (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    ):
        raise MOSContractError("MOS value outside documented domain")
    return float(value), "valid"


def parse_mos_response(raw: bytes, plan, receipt: dict) -> pd.DataFrame:
    """Validate run identity, preserve raw encodings and derive native features.

    All rows remain availability_unverified; neither retrieval time nor model
    cycle is used as publication time. No hourly interpolation or imputation.
    """
    request = validate_plan(plan)
    if (
        receipt.get("sha256") != hashlib.sha256(raw).hexdigest()
        or receipt.get("request_key") != plan.request_key
    ):
        raise MOSContractError("Response provenance hash/request mismatch")
    try:
        payload = json.loads(raw)
        if not isinstance(payload.get("schema"), dict) or not isinstance(
            payload["data"], list
        ):
            raise ValueError()
    except (ValueError, TypeError, KeyError, AttributeError):
        raise MOSContractError("Invalid MOS JSON table-schema payload") from None
    records = []
    for source in payload["data"]:
        if (
            not isinstance(source, dict)
            or source.get("station") not in request.stations
            or source.get("model") != request.model
        ):
            raise MOSContractError("Unexpected model/station in MOS response")
        run = _utc_from_explicit_api_field(source.get("runtime_utc"))
        valid = _utc_from_explicit_api_field(source.get("ftime_utc"))
        if run != request.runtime_utc or valid <= run:
            raise MOSContractError("Wrong vintage or nonpositive forecast lead")
        if run != run.floor("h") or valid != valid.floor("h"):
            raise MOSContractError("MOS forecast identity is not an exact hour")
        for key, timestamp in (("runtime", run), ("ftime", valid)):
            if key in source and _utc_from_explicit_api_field(source[key]) != timestamp:
                raise MOSContractError("Conflicting legacy and explicit UTC fields")
        record = {
            "Origin": source["station"][1:],
            "station": source["station"],
            "model": request.model,
            "provider": "iem_mos",
            "backend": "iem_mos",
            "contract_version": CONTRACT_VERSION,
            "model_version": "not_exposed_by_archive_api",
            "forecast_run_at_utc": run,
            "forecast_valid_at_utc": valid,
            "forecast_hour": int((valid - run).total_seconds() / 3600),
            "published_at_utc": pd.NaT,
            "publication_evidence": None,
            "availability_status": "availability_unverified",
            "model_ready": False,
            "request_key": plan.request_key,
            "response_sha256": receipt["sha256"],
            "retrieved_at_utc": utc_timestamp(receipt["retrieved_at_utc"]),
            "source_url": plan.url,
        }
        values = {}
        for field in RAW_FIELDS:
            value = source.get(field)
            record["mos_raw_" + field] = value
            missing = (
                {-99}
                if request.model == "NBS"
                else {
                    "tmp": {999},
                    "wsp": {99},
                    "gst": {99},
                    "vis": {9},
                    "cig": {9},
                }.get(field, {999})
            )
            minimum, maximum = None, None
            if field in ("wsp", "gst", "vis", "lcb"):
                minimum = 0
            if field in ("p06", "t06_1", "t03"):
                minimum, maximum = 0, 100
            if field == "vis" and request.model == "GFS":
                minimum, maximum = 1, 7
            if field == "cig" and request.model == "GFS":
                minimum, maximum = 1, 8
            if field in ("cig", "lcb") and request.model == "NBS" and value == -88:
                values[field] = np.nan
                record["mos_status_" + field] = "unlimited"
                continue
            if field == "cig" and request.model == "NBS":
                minimum = 0
            values[field], status = _decode(
                value, missing=missing, minimum=minimum, maximum=maximum
            )
            if (
                field in ("cig", "vis")
                and request.model == "GFS"
                and status == "valid"
                and not values[field].is_integer()
            ):
                raise MOSContractError("GFS category must be an integer")
            record["mos_status_" + field] = status
        record.update({name: np.nan for name in FEATURE_COLUMNS})
        record.update(
            mos_temperature_c=(values["tmp"] - 32) * 5 / 9,
            mos_wind_speed_mps=values["wsp"] * (1852 / 3600),
            mos_gust_mps=values["gst"] * (1852 / 3600),
            mos_pop_6h_pct=values["p06"],
            mos_tstorm_6h_pct=values["t06_1"],
            mos_tstorm_3h_pct=values["t03"],
        )
        if request.model == "GFS":
            record.update(
                mos_gfs_visibility_category=values["vis"],
                mos_gfs_ceiling_category=values["cig"],
            )
        else:
            record.update(
                mos_nbm_visibility_miles=values["vis"] / 10,
                mos_nbm_ceiling_hundreds_ft=values["cig"],
                mos_nbm_ceiling_unlimited=(
                    1.0
                    if source.get("cig") == -88
                    else 0.0
                    if record["mos_status_cig"] == "valid"
                    else np.nan
                ),
            )
        # The API's concatenated t06 can be null when its second component is
        # absent although t06_1 exists. Preserve each component, do not substitute
        # severe thunderstorm conditional probability for thunderstorm probability.
        for field in ("t06", "t06_2", "q06", "p12", "wdr"):
            record["mos_raw_" + field] = source.get(field)
        records.append(record)
    if not records:
        raise MOSContractError("Empty MOS response")
    result = pd.DataFrame(records)
    result["published_at_utc"] = pd.to_datetime(result["published_at_utc"], utc=True)
    identity = ["station", "model", "forecast_run_at_utc", "forecast_valid_at_utc"]
    if result.duplicated(identity).any():
        raise MOSContractError("Duplicate MOS forecast identity")
    return result.sort_values(identity).reset_index(drop=True)
