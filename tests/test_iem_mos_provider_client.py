"""Synthetic transport fixtures test safeguards without touching external APIs."""

import json
from dataclasses import replace

import pytest

from flight_delay.weather.api_client import APIClientError, APILimits, RequestLimitError
from flight_delay.weather.iem_mos import (
    IEMMOSClient,
    IEMMOSProvider,
    MOSContractError,
    MOSHTTPError,
    MOSRunRequest,
    MOSTransportError,
    validate_plan,
)


def plan(model="GFS", run="2023-01-15T00:00Z"):
    return IEMMOSProvider().plan(MOSRunRequest(model, run))


def body():
    return json.dumps({"schema": {"fields": []}, "data": []}).encode()


def forbidden(*args):
    pytest.fail("No network call expected")


@pytest.mark.parametrize(
    "model,run",
    [
        ("GFS", "2023-01-15T01:00Z"),
        ("NBS", "2023-01-15T00:00Z"),
        ("NBS", "2026-06-15T01:00Z"),
        ("HRRR", "2023-01-15T00:00Z"),
        ("GFS", "2023-01-15T00:00"),
        ("GFS", "2023-01-15T00:01Z"),
    ],
)
def test_invalid_product_or_vintage_policy(model, run):
    with pytest.raises(ValueError):
        MOSRunRequest(model, run)


@pytest.mark.parametrize("stations", [(), ("KJFK",), ("KATL",) * 2, ["KATL"]])
def test_station_contract(stations):
    with pytest.raises(MOSContractError):
        MOSRunRequest("GFS", "2023-01-15T00:00Z", stations)


def test_native_request_and_plan_keep_publication_unknown():
    p = plan()
    assert len([k for k, _ in p.params if k == "station"]) == 6
    assert p.assumed_available_at_utc is None
    assert "availability_unverified" in p.blockers
    assert not p.paid
    assert validate_plan(p).model == "GFS"
    a = MOSRunRequest("NBS", "2024-01-15T01:00Z")
    assert a.runtime_utc.hour == 1
    assert MOSRunRequest("NBS", "2026-06-15T00:00Z").runtime_utc.hour == 0


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint": "https://mesonet.agron.iastate.edu/api/1/asos.json"},
        {"paid": True},
        {"model": "NBS"},
        {"params": (("station", "KATL"), ("model", "GFS"))},
    ],
)
def test_client_rejects_plan_tampering_before_transport(tmp_path, change):
    with pytest.raises(MOSContractError):
        IEMMOSClient(tmp_path, allow_network=True, transport=forbidden).fetch(
            replace(plan(), **change)
        )


def test_dry_run_deduplicates_and_never_certifies_model_use():
    r = MOSRunRequest("GFS", "2023-01-15T00:00Z")
    report = IEMMOSProvider().dry_run([r, r], max_requests=1)
    assert report["unique_http_requests"] == 1
    assert report["http_attempts"] == 0
    assert report["within_request_budget"]
    assert not report["model_ready"]
    assert not report["ready_for_live"]


def test_default_network_off_creates_no_cache(tmp_path):
    client = IEMMOSClient(tmp_path / "absent", transport=forbidden)
    with pytest.raises(APIClientError, match="Network disabled"):
        client.fetch(plan())
    assert not (tmp_path / "absent").exists()


def test_cache_survives_restart_and_replays_original_receipt(tmp_path):
    calls = []

    def transport(*args):
        calls.append(args)
        return body()

    client = IEMMOSClient(tmp_path, allow_network=True, transport=transport)
    raw, receipt = client.fetch(plan())
    replay, cached = IEMMOSClient(tmp_path, transport=forbidden).fetch(plan())
    assert raw == replay
    assert receipt["retrieved_at_utc"] == cached["retrieved_at_utc"]
    assert cached["cache_status"] == "hit"
    assert cached["availability_status"] == "availability_unverified"
    assert not cached["model_ready"]
    assert len(calls) == len(json.loads(client.ledger_path.read_text())) == 1


@pytest.mark.parametrize(
    "field,value", [("body_base64", "YmFk"), ("schema_version", 2)]
)
def test_cache_corruption_fails_closed(tmp_path, field, value):
    client = IEMMOSClient(tmp_path, allow_network=True, transport=lambda *a: body())
    client.fetch(plan())
    path = client._cache_path(plan())
    d = json.loads(path.read_text())
    d[field] = value
    path.write_text(json.dumps(d))
    with pytest.raises(APIClientError, match="integrity"):
        IEMMOSClient(tmp_path, allow_network=True, transport=forbidden).fetch(plan())


def test_budget_survives_restart_and_failed_requests_consume_it(tmp_path):
    def failing(*args):
        raise MOSHTTPError(404)

    limits = APILimits(max_requests=1)
    client = IEMMOSClient(
        tmp_path, allow_network=True, limits=limits, transport=failing
    )
    with pytest.raises(APIClientError):
        client.fetch(plan())
    restarted = IEMMOSClient(
        tmp_path, allow_network=True, limits=limits, transport=forbidden
    )
    with pytest.raises(RequestLimitError, match="Persistent"):
        restarted.fetch(plan())
    assert len(json.loads(client.ledger_path.read_text())) == 1
    assert not (tmp_path / "writer.lock").exists()


@pytest.mark.parametrize(
    "error", [MOSHTTPError(429), MOSHTTPError(503), MOSTransportError("timeout")]
)
def test_transient_errors_retry_once_with_each_attempt_counted(tmp_path, error):
    calls = []
    waits = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return body()

    client = IEMMOSClient(
        tmp_path, allow_network=True, transport=transport, sleeper=waits.append
    )
    client.fetch(plan())
    ledger = json.loads(client.ledger_path.read_text())
    assert len(calls) == len(ledger) == 2
    assert [r["status"] for r in ledger] == ["error", "success"]
    assert waits and max(waits) <= 1


def test_retry_cannot_exceed_budget(tmp_path):
    calls = []

    def transport(*args):
        calls.append(args)
        raise MOSHTTPError(503)

    client = IEMMOSClient(
        tmp_path,
        allow_network=True,
        limits=APILimits(max_requests=1),
        transport=transport,
        sleeper=lambda _: None,
    )
    with pytest.raises(RequestLimitError):
        client.fetch(plan())
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"not json", b"[]", b"x" * 101])
def test_bad_responses_are_not_cached_or_retried(tmp_path, payload):
    calls = []

    def transport(*args):
        calls.append(args)
        return payload

    client = IEMMOSClient(
        tmp_path,
        allow_network=True,
        limits=APILimits(max_response_bytes=100),
        transport=transport,
    )
    with pytest.raises(APIClientError):
        client.fetch(plan())
    assert len(calls) == 1
    assert not client._cache_path(plan()).exists()


def test_active_writer_prevents_network(tmp_path):
    (tmp_path / "writer.lock").write_text("active")
    with pytest.raises(APIClientError, match="writer"):
        IEMMOSClient(tmp_path, allow_network=True, transport=forbidden).fetch(plan())


def test_corrupt_ledger_does_not_reset_budget(tmp_path):
    (tmp_path / "attempts.json").write_text("bad")
    with pytest.raises(APIClientError, match="ledger"):
        IEMMOSClient(tmp_path, allow_network=True, transport=forbidden).fetch(plan())
