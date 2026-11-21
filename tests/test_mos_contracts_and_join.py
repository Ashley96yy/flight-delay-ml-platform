"""Synthetic MOS records exercise native units, missingness and strict PIT gates."""

import hashlib
import json

import pandas as pd
import pytest

from flight_delay.weather.iem_mos import IEMMOSProvider, MOSContractError, MOSRunRequest
from flight_delay.weather.mos_contracts import parse_mos_response
from flight_delay.weather.mos_flight_join import (
    audit_mos_for_flights,
    join_mos_to_flights,
)


def fixture(model="GFS", **changes):
    run = "2023-01-15T00:00:00" if model == "GFS" else "2023-01-15T01:00:00"
    row = {
        "model": model,
        "station": "KATL",
        "runtime_utc": run,
        "ftime_utc": "2023-01-16T06:00:00",
        "tmp": 32,
        "wsp": 10,
        "gst": None,
        "vis": 7 if model == "GFS" else 100,
        "cig": 8 if model == "GFS" else -88,
        "p06": 99,
        "t06": None,
        "t06_1": 12,
        "t06_2": None,
    }
    row.update(changes)
    return row


def parsed(model="GFS", rows=None):
    rows = rows if rows is not None else [fixture(model)]
    run = "2023-01-15T00:00Z" if model == "GFS" else "2023-01-15T01:00Z"
    p = IEMMOSProvider().plan(MOSRunRequest(model, run, ("KATL",)))
    raw = json.dumps({"schema": {"fields": []}, "data": rows}).encode()
    receipt = {
        "request_key": p.request_key,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved_at_utc": "2026-11-21T00:00:00Z",
    }
    return parse_mos_response(raw, p, receipt)


def flight(frame, departure="2023-01-16T06:00Z"):
    cols = ["Origin", "forecast_run_at_utc", "forecast_valid_at_utc", "forecast_hour"]
    f = frame[cols].copy()
    f["scheduled_departure_utc"] = pd.Timestamp(departure)
    return f


def test_gfs_preserves_categories_and_probability_meanings():
    frame = parsed()
    r = frame.iloc[0]
    assert r.mos_temperature_c == pytest.approx(0)
    assert r.mos_wind_speed_mps == pytest.approx(10 * 1852 / 3600)
    assert r.mos_gfs_visibility_category == 7 and r.mos_gfs_ceiling_category == 8
    assert pd.isna(r.mos_nbm_visibility_miles)
    assert r.mos_pop_6h_pct == 99  # Valid 99% is not a missing code.
    assert r.mos_tstorm_6h_pct == 12  # concatenated t06 is null, t06_1 exists.
    assert pd.isna(r.mos_gust_mps)
    assert not any(
        "cape" in col or "reflectivity" in col or "precipitation_1h" in col
        for col in frame
    )
    assert r.availability_status == "availability_unverified" and not r.model_ready
    assert pd.isna(r.published_at_utc)


def test_nbm_units_and_unlimited_ceiling_are_distinct_from_gfs_categories():
    r = parsed("NBS").iloc[0]
    assert r.mos_nbm_visibility_miles == 10
    assert r.mos_nbm_ceiling_unlimited == 1
    assert pd.isna(r.mos_nbm_ceiling_hundreds_ft)
    assert r.mos_status_cig == "unlimited"
    assert pd.isna(r.mos_gfs_visibility_category)


@pytest.mark.parametrize(
    "model,changes,field",
    [
        ("GFS", {"tmp": 999}, "tmp"),
        ("GFS", {"wsp": 99}, "wsp"),
        ("GFS", {"vis": 9}, "vis"),
        ("GFS", {"cig": 9}, "cig"),
        ("GFS", {"p06": 999}, "p06"),
        ("NBS", {"gst": -99}, "gst"),
        ("NBS", {"tmp": -99}, "tmp"),
        ("NBS", {"vis": -99}, "vis"),
    ],
)
def test_field_specific_missing_codes_are_not_features(model, changes, field):
    r = parsed(model, [fixture(model, **changes)]).iloc[0]
    assert r["mos_status_" + field] == "missing_code"


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "NBS"},
        {"station": "KJFK"},
        {"runtime_utc": "2023-01-16T00:00"},
        {"ftime_utc": "2023-01-14T00:00"},
        {"runtime_utc": None},
        {"runtime": "2023-01-15 01:00"},
        {"vis": 8},
        {"cig": 1.5},
        {"p06": 101},
        {"wsp": -1},
        {"tmp": True},
        {"tmp": float("nan")},
    ],
)
def test_invalid_identity_time_or_field_fails_closed(changes):
    with pytest.raises(MOSContractError):
        parsed(rows=[fixture(**changes)])


def test_duplicate_forecasts_are_never_averaged():
    with pytest.raises(MOSContractError, match="Duplicate"):
        parsed(rows=[fixture(), fixture()])


def test_no_observation_or_publication_time_is_inferred_from_retrieval():
    w = parsed()
    report = audit_mos_for_flights(flight(w), w)
    assert report.status.tolist() == ["availability_unverified"]
    assert not report.ready.any()
    with pytest.raises(MOSContractError, match="availability_unverified"):
        join_mos_to_flights(flight(w), w)


def verified_fixture():
    # TEST ONLY: external publication evidence is never invented for real data.
    w = parsed()
    w["published_at_utc"] = pd.Timestamp("2023-01-15T05:00Z")
    w["publication_evidence"] = "synthetic-test-receipt"
    w["availability_status"] = "availability_verified"
    return w


def test_exact_cutoff_can_pass_only_with_explicit_publication_evidence():
    w = verified_fixture()
    f = flight(w)
    assert join_mos_to_flights(f, w).ready.all()
    w["published_at_utc"] = pd.Timestamp("2023-01-15T06:00Z")
    assert audit_mos_for_flights(f, w).ready.all()


@pytest.mark.parametrize(
    "timestamp,status",
    [
        ("2023-01-15T06:00:01Z", "published_after_cutoff"),
        ("2023-01-14T23:59Z", "publication_before_initialization"),
    ],
)
def test_strict_publication_boundary(timestamp, status):
    w = verified_fixture()
    w["published_at_utc"] = pd.Timestamp(timestamp)
    assert audit_mos_for_flights(flight(w), w).status.tolist() == [status]


def test_off_grid_departures_are_not_interpolated_or_floored():
    w = verified_fixture()
    assert audit_mos_for_flights(flight(w, "2023-01-16T06:15Z"), w).status.tolist() == [
        "missing_exact_valid_time"
    ]


def test_future_vintage_is_rejected_even_if_http_response_exists():
    w = parsed()
    f = flight(w)
    f["scheduled_departure_utc"] = f["forecast_valid_at_utc"]
    w["forecast_run_at_utc"] += pd.Timedelta(hours=12)
    w["forecast_hour"] -= 12
    f["forecast_run_at_utc"] = w["forecast_run_at_utc"]
    f["forecast_hour"] = w["forecast_hour"]
    assert audit_mos_for_flights(f, w).status.tolist() == [
        "forecast_initialized_after_cutoff"
    ]


def test_wrong_supplied_cutoff_cannot_be_overwritten():
    w = parsed()
    f = flight(w)
    f["prediction_at_utc"] = pd.Timestamp("2023-01-15T07:00Z")
    with pytest.raises(MOSContractError, match="T-24h"):
        audit_mos_for_flights(f, w)


def test_duplicate_weather_and_mixed_models_rejected():
    w = parsed()
    with pytest.raises(MOSContractError, match="Duplicate"):
        audit_mos_for_flights(flight(w), pd.concat([w, w]))
    with pytest.raises(MOSContractError, match="one model"):
        audit_mos_for_flights(flight(w), pd.concat([w, parsed("NBS")]))


def test_naive_publication_is_not_silently_localized():
    w = verified_fixture()
    w["published_at_utc"] = pd.Timestamp("2023-01-15T05:00")
    with pytest.raises(MOSContractError, match="timezone-aware"):
        audit_mos_for_flights(flight(w), w)
