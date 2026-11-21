"""Bounded native-grid alignment, with publication approval kept separate.

Unverified runs can produce geometry/coverage diagnostics, never interpolated
weather values or model-ready features. No HRRR policy is changed here.
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flight_delay.weather.api_providers import utc_timestamp
from flight_delay.weather.flight_join import _utc_datetime
from flight_delay.weather.iem_mos import MOSContractError, MOSRunRequest
from flight_delay.weather.mos_contracts import CONTRACT_VERSION, FEATURE_COLUMNS

CONTINUOUS_FIELDS = frozenset({"mos_temperature_c", "mos_wind_speed_mps"})
PERIOD_FIELDS = frozenset({"mos_pop_6h_pct", "mos_tstorm_6h_pct", "mos_tstorm_3h_pct"})
MODEL_FIELDS = {
    "GFS": (
        "mos_temperature_c",
        "mos_wind_speed_mps",
        "mos_gfs_visibility_category",
        "mos_gfs_ceiling_category",
        "mos_pop_6h_pct",
        "mos_tstorm_6h_pct",
    ),
    "NBS": (
        "mos_temperature_c",
        "mos_wind_speed_mps",
        "mos_gust_mps",
        "mos_nbm_visibility_miles",
        "mos_nbm_ceiling_hundreds_ft",
        "mos_nbm_ceiling_unlimited",
        "mos_pop_6h_pct",
        "mos_tstorm_3h_pct",
    ),
}


@dataclass(frozen=True)
class RunAvailabilityEvidence:
    """Externally reviewed evidence for the complete station/run payload.

    A receipt establishes available-by, not exact publication. Schedule-based
    guesses are a separate kind that can never clear the strict audit. Nothing
    in this module manufactures evidence from runtime or today's retrieval.
    """

    model: str
    station: str
    model_run_time: pd.Timestamp
    response_sha256: str
    kind: str = "unverified"
    available_at: pd.Timestamp | None = None
    evidence_reference: str | None = None
    covers_entire_station_run: bool = False

    def __post_init__(self):
        request = MOSRunRequest(self.model, self.model_run_time, (self.station,))
        object.__setattr__(self, "model_run_time", request.runtime_utc)
        if not isinstance(self.response_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.response_sha256
        ):
            raise MOSContractError("Evidence requires the original payload SHA256")
        if type(self.covers_entire_station_run) is not bool:
            raise MOSContractError("Evidence scope must be an explicit boolean")
        if self.kind not in {
            "unverified",
            "schedule_assumption",
            "historical_full_run_receipt",
            "verified_publication",
        }:
            raise MOSContractError("Unsupported availability evidence kind")
        if self.available_at is not None:
            timestamp = utc_timestamp(self.available_at)
            if timestamp < request.runtime_utc:
                raise MOSContractError(
                    "Availability cannot precede model initialization"
                )
            object.__setattr__(self, "available_at", timestamp)
        if self.kind != "unverified" and self.available_at is None:
            raise MOSContractError("Availability evidence needs an explicit timestamp")
        if self.kind in {"historical_full_run_receipt", "verified_publication"} and (
            not isinstance(self.evidence_reference, str)
            or not self.evidence_reference.strip()
            or not self.covers_entire_station_run
        ):
            raise MOSContractError("Verified evidence must cover the full station/run")


def _availability(evidence, model, origin, run, payload_hash, cutoff):
    matches = [
        e
        for e in evidence
        if (e.model, e.station, e.model_run_time) == (model, "K" + origin, run)
    ]
    if len(matches) > 1:
        raise MOSContractError("Conflicting evidence for station/model/run")
    result = {
        "availability_status": "availability_unverified",
        "actual_publication_time_utc": pd.NaT,
        "verified_available_by_utc": pd.NaT,
        "assumed_available_time_utc": pd.NaT,
        "availability_evidence_kind": "unverified",
        "availability_evidence_reference": None,
    }
    if run > cutoff:
        result["availability_status"] = "forecast_initialized_after_cutoff"
    if not matches:
        return result
    e = matches[0]
    if e.response_sha256 != payload_hash:
        raise MOSContractError(
            "Availability evidence belongs to a different payload vintage"
        )
    result.update(
        availability_evidence_kind=e.kind,
        availability_evidence_reference=e.evidence_reference,
    )
    if e.kind == "schedule_assumption":
        result["assumed_available_time_utc"] = e.available_at
        return result
    if e.kind == "unverified":
        return result
    if e.kind == "verified_publication":
        result["actual_publication_time_utc"] = e.available_at
    result["verified_available_by_utc"] = e.available_at
    result["availability_status"] = (
        "availability_verified"
        if e.available_at <= cutoff
        else "available_after_cutoff"
    )
    return result


def _validated_weather(forecasts):
    required = {
        "Origin",
        "model",
        "provider",
        "backend",
        "contract_version",
        "forecast_run_at_utc",
        "forecast_valid_at_utc",
        "forecast_hour",
        "response_sha256",
        "request_key",
        *FEATURE_COLUMNS,
    }
    if (
        not isinstance(forecasts, pd.DataFrame)
        or forecasts.empty
        or not required <= set(forecasts)
    ):
        raise MOSContractError("Missing native MOS alignment contract")
    frame = forecasts.copy(deep=True)
    if (
        not frame.provider.eq("iem_mos").all()
        or not frame.backend.eq("iem_mos").all()
        or not frame.contract_version.eq(CONTRACT_VERSION).all()
    ):
        raise MOSContractError(
            "Only native forecast records are permitted, not observations"
        )
    for col in ("forecast_run_at_utc", "forecast_valid_at_utc"):
        frame[col] = _utc_datetime(frame, col)
    if not frame.forecast_valid_at_utc.eq(
        frame.forecast_valid_at_utc.dt.floor("h")
    ).all():
        raise MOSContractError("Native forecast valid times must be exact hours")
    identity = ["Origin", "model", "forecast_run_at_utc", "forecast_valid_at_utc"]
    if frame[identity].isna().any().any() or frame.duplicated(identity).any():
        raise MOSContractError("Missing or duplicate MOS forecast identity")
    lead = (
        frame.forecast_valid_at_utc - frame.forecast_run_at_utc
    ).dt.total_seconds() / 3600
    if not lead.gt(0).all() or not lead.eq(frame.forecast_hour).all():
        raise MOSContractError("Forecast lead identity mismatch")
    for (origin, model, run), group in frame.groupby(identity[:3], sort=False):
        MOSRunRequest(model, run, ("K" + origin,))
        if (
            group.response_sha256.nunique(dropna=False) != 1
            or group.request_key.nunique(dropna=False) != 1
        ):
            raise MOSContractError("Do not mix payload versions within one station/run")
        for column in ("response_sha256", "request_key"):
            value = group[column].iloc[0]
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise MOSContractError("Missing source payload/request hash")
    return frame.sort_values(identity)


def align_mos_forecasts(
    flight_requests,
    forecasts,
    *,
    strategy="conservative",
    evidence=(),
    max_past_gap_hours=3.0,
    max_bracket_hours=6.0,
):
    """Plan A or B and emit numeric features only for evidence-approved runs.

    Flights must name their selected model/run explicitly. No automatic latest
    run selection or cross-run bracket construction. Probabilities are retained
    only at their exact native period endpoint. Nonlinear fields in strategy B
    use the explicitly aged conservative point; they are never linearly blended.
    """
    if strategy not in {"conservative", "interpolation"}:
        raise MOSContractError("Unknown alignment strategy")
    for limit in (max_past_gap_hours, max_bracket_hours):
        if isinstance(limit, bool) or not math.isfinite(limit) or not 0 < limit <= 24:
            raise MOSContractError("Alignment gaps must be finite, positive and <=24h")
    evidence = tuple(evidence)
    if not all(isinstance(e, RunAvailabilityEvidence) for e in evidence):
        raise MOSContractError(
            "Only explicit reviewed availability evidence is accepted"
        )
    weather = _validated_weather(forecasts)
    required = {"Origin", "model", "forecast_run_at_utc", "scheduled_departure_utc"}
    if (
        not isinstance(flight_requests, pd.DataFrame)
        or flight_requests.empty
        or not required <= set(flight_requests)
    ):
        raise MOSContractError("Missing explicit flight/model/run request")
    flights = flight_requests.copy(deep=True)
    for col in ("scheduled_departure_utc", "forecast_run_at_utc"):
        flights[col] = _utc_datetime(flights, col)
    cutoff = flights.scheduled_departure_utc - pd.Timedelta(hours=24)
    for col in (
        "prediction_at_utc",
        "prediction_time_utc",
        "flight_prediction_time_utc",
    ):
        if col in flights and not _utc_datetime(flights, col).equals(cutoff):
            raise MOSContractError("Supplied cutoff is not exactly T-24h")
    rows = []
    for position, flight in enumerate(flights.to_dict("records")):
        origin, model, run = (
            flight["Origin"],
            flight["model"],
            flight["forecast_run_at_utc"],
        )
        MOSRunRequest(model, run, ("K" + origin,))
        target = flight["scheduled_departure_utc"]
        prediction = target - pd.Timedelta(hours=24)
        group = weather.loc[
            weather.Origin.eq(origin)
            & weather.model.eq(model)
            & weather.forecast_run_at_utc.eq(run)
        ]
        result = {
            "input_row_position": position,
            "sample_id": flight.get("sample_id", str(position)),
            "Origin": origin,
            "model": model,
            "strategy": strategy,
            "model_run_time_utc": run,
            "scheduled_departure_utc": target,
            "flight_prediction_time_utc": prediction,
            "forecast_age_at_prediction_hours": (prediction - run).total_seconds()
            / 3600,
            "forecast_lead_at_target_hours": (target - run).total_seconds() / 3600,
            "temporal_match": False,
            "alignment_status": "missing_selected_run",
            "left_valid_time_utc": pd.NaT,
            "right_valid_time_utc": pd.NaT,
            "valid_time_gap_hours": np.nan,
            "right_gap_hours": np.nan,
            "bracket_span_hours": np.nan,
            "right_weight": np.nan,
            "planned_interpolation": False,
            "interpolation_executed": False,
            "strict_pit_pass": False,
            "model_ready": False,
            "source_response_sha256": None,
            "source_request_key": None,
            "source_contract_version": CONTRACT_VERSION,
        }
        result.update({f: np.nan for f in FEATURE_COLUMNS})
        methods, covered = {}, []
        if group.empty:
            result.update(_availability((), model, origin, run, None, prediction))
        else:
            payload_hash = group.response_sha256.iloc[0]
            result.update(
                source_response_sha256=payload_hash,
                source_request_key=group.request_key.iloc[0],
            )
            result.update(
                _availability(evidence, model, origin, run, payload_hash, prediction)
            )
            left = group.loc[group.forecast_valid_at_utc.le(target)].tail(1)
            right = group.loc[group.forecast_valid_at_utc.ge(target)].head(1)
            if left.empty:
                result["alignment_status"] = "no_past_forecast"
            else:
                left = left.iloc[0]
                gap = (target - left.forecast_valid_at_utc).total_seconds() / 3600
                result.update(
                    left_valid_time_utc=left.forecast_valid_at_utc,
                    valid_time_gap_hours=gap,
                )
                result["alignment_status"] = "past_gap_exceeded"
                if gap <= max_past_gap_hours:
                    result.update(
                        temporal_match=True, alignment_status="conservative_point"
                    )
                    right_row = None
                    if gap == 0:
                        result.update(
                            alignment_status="exact_native_point",
                            right_weight=0.0,
                            right_valid_time_utc=left.forecast_valid_at_utc,
                            right_gap_hours=0.0,
                            bracket_span_hours=0.0,
                        )
                    elif strategy == "interpolation":
                        result.update(
                            temporal_match=False, alignment_status="no_same_run_bracket"
                        )
                        if not right.empty:
                            right_row = right.iloc[0]
                            right_gap = (
                                right_row.forecast_valid_at_utc - target
                            ).total_seconds() / 3600
                            span = gap + right_gap
                            result.update(
                                right_valid_time_utc=right_row.forecast_valid_at_utc,
                                right_gap_hours=right_gap,
                                bracket_span_hours=span,
                                right_weight=gap / span,
                            )
                            if (
                                right_gap <= max_past_gap_hours
                                and span <= max_bracket_hours
                            ):
                                result.update(
                                    temporal_match=True,
                                    alignment_status="same_run_bracket",
                                    planned_interpolation=True,
                                )
                            else:
                                result["alignment_status"] = "bracket_gap_exceeded"
                    if result["temporal_match"]:
                        approved = (
                            result["availability_status"] == "availability_verified"
                        )
                        for field in MODEL_FIELDS[model]:
                            value = left[field]
                            method = "exact" if gap == 0 else "conservative_point"
                            valid = pd.notna(value) and np.isfinite(value)
                            if field in PERIOD_FIELDS and gap != 0:
                                valid = False
                                method = "period_end_mismatch"
                            elif (
                                strategy == "interpolation"
                                and gap != 0
                                and field in CONTINUOUS_FIELDS
                            ):
                                valid = (
                                    valid
                                    and pd.notna(right_row[field])
                                    and np.isfinite(right_row[field])
                                )
                                method = (
                                    "linear"
                                    if valid
                                    else "missing_interpolation_endpoint"
                                )
                                # Arithmetic is NOT performed on unverified runs.
                                if valid and approved:
                                    weight = result["right_weight"]
                                    value = (1 - weight) * value + weight * right_row[
                                        field
                                    ]
                                    result["interpolation_executed"] = True
                            elif not valid:
                                method = "source_missing_or_unlimited"
                            methods[field] = method
                            if valid:
                                covered.append(field)
                                if approved:
                                    result[field] = float(value)
                        result["strict_pit_pass"] = approved
                        result["model_ready"] = approved and all(
                            f in covered for f in CONTINUOUS_FIELDS
                        )
        result["potential_field_count"] = len(covered)
        result["potential_field_denominator"] = len(MODEL_FIELDS[model])
        result["potential_fields_json"] = json.dumps(covered)
        result["field_methods_json"] = json.dumps(methods, sort_keys=True)
        result["materialized_field_count"] = sum(
            pd.notna(result[f]) for f in MODEL_FIELDS[model]
        )
        result["strict_status"] = (
            "ready"
            if result["strict_pit_pass"]
            else result["alignment_status"]
            if not result["temporal_match"]
            else result["availability_status"]
        )
        rows.append(result)
    frame = pd.DataFrame(rows)
    for col in (
        "actual_publication_time_utc",
        "verified_available_by_utc",
        "assumed_available_time_utc",
        "left_valid_time_utc",
        "right_valid_time_utc",
    ):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def require_model_ready_alignment(*args, **kwargs):
    """Strict model entry point; geometry-only coverage cannot bypass it."""
    frame = align_mos_forecasts(*args, **kwargs)
    if not frame.model_ready.all():
        raise MOSContractError(
            f"MOS alignment is not model-ready: {frame.strict_status.value_counts().to_dict()}"
        )
    return frame
