"""Client safeguards use synthetic transport bytes, never real API calls."""

from dataclasses import replace
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from flight_delay.weather.api_client import (
    APIClientError,
    APILimits,
    HistoricalForecastClient,
    RequestLimitError,
    _NoRedirects,
    bounded_json_get,
)
from flight_delay.weather.api_providers import (
    ForecastQuery,
    OpenMeteoSingleRunsAdapter,
    VisualCrossingRunAdapter,
)


def candidate():
    return OpenMeteoSingleRunsAdapter().plan(ForecastQuery("ATL", "2024-07-15T09:15Z"))


def synthetic_approved_plan():
    # TEST ONLY: exercises client mechanics; no provider is approved by this.
    return replace(candidate(), blockers=())


def forbidden(*args):
    pytest.fail("Transport must not be called")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_requests": 0},
        {"max_requests": True},
        {"max_requests": 1.5},
        {"max_response_bytes": -1},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"min_interval_seconds": -1},
        {"min_interval_seconds": float("inf")},
    ],
)
def test_invalid_limits_fail_before_any_io(overrides):
    with pytest.raises(ValueError):
        APILimits(**overrides)


def test_default_client_cannot_fetch():
    client = HistoricalForecastClient(transport=forbidden)
    with pytest.raises(APIClientError, match="Network disabled"):
        client.fetch(synthetic_approved_plan())
    assert client.attempts == 0


def test_unresolved_real_candidate_cannot_fetch_even_with_network_enabled():
    client = HistoricalForecastClient(allow_network=True, transport=forbidden)
    with pytest.raises(APIClientError, match="evidence"):
        client.fetch(candidate())
    assert client.attempts == 0


def test_paid_execution_is_impossible_in_this_stage():
    paid = VisualCrossingRunAdapter().plan(ForecastQuery("ATL", "2023-07-15T09:15Z"))
    client = HistoricalForecastClient(allow_network=True, transport=forbidden)
    with pytest.raises(APIClientError, match="Paid API execution"):
        client.fetch(replace(paid, blockers=()))
    assert client.attempts == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "http://single-runs-api.open-meteo.com/v1/forecast",
        "https://single-runs-api.open-meteo.com.evil.example/v1/forecast",
    ],
)
def test_other_products_and_hosts_are_rejected(endpoint):
    client = HistoricalForecastClient(allow_network=True, transport=forbidden)
    with pytest.raises(APIClientError, match="Endpoint"):
        client.fetch(replace(synthetic_approved_plan(), endpoint=endpoint))


def test_dry_run_deduplicates_and_reports_unknowns_without_using_budget():
    client = HistoricalForecastClient(
        limits=APILimits(max_requests=1), transport=forbidden
    )
    report = client.dry_run([candidate(), candidate()])
    assert report["unique_http_requests"] == 1
    assert report["input_queries"] == 2
    assert report["http_attempts"] == client.attempts == 0
    assert report["within_request_budget"]
    assert not report["ready_for_live"]
    assert report["observed_weather_coverage"] is None
    assert report["model_metrics"] is None
    assert report["estimated_provider_billing_units"] is None
    assert report["requests"][0]["blockers"]


def test_dry_run_detects_budget_overflow_without_sending_anything():
    other = OpenMeteoSingleRunsAdapter().plan(ForecastQuery("SFO", "2024-07-15T09:15Z"))
    client = HistoricalForecastClient(
        limits=APILimits(max_requests=1), transport=forbidden
    )
    report = client.dry_run([candidate(), other])
    assert not report["within_request_budget"]
    assert client.attempts == 0


def test_success_is_bounded_hashed_and_remains_raw_only():
    calls = []

    def transport(*args):
        calls.append(args)
        return b'{"fixture": true}'

    client = HistoricalForecastClient(
        allow_network=True,
        transport=transport,
        limits=APILimits(max_requests=1, max_response_bytes=100, timeout_seconds=3),
    )
    result = client.fetch(synthetic_approved_plan())
    assert result["payload"] == {"fixture": True}
    assert result["status"] == "raw_response_only_not_model_ready"
    assert len(result["response_sha256"]) == 64
    assert result["response_bytes"] == len(b'{"fixture": true}')
    assert calls[0][1:] == (3, 100)
    with pytest.raises(RequestLimitError, match="budget exhausted"):
        client.fetch(synthetic_approved_plan())
    assert len(calls) == client.attempts == 1


def test_failed_attempt_consumes_quota_and_does_not_retry_or_leak_error():
    calls = []

    def transport(*args):
        calls.append(args)
        raise TimeoutError("sensitive provider URL")

    client = HistoricalForecastClient(
        allow_network=True,
        transport=transport,
        limits=APILimits(max_requests=1),
    )
    with pytest.raises(APIClientError, match="^HTTP transport failed$"):
        client.fetch(synthetic_approved_plan())
    with pytest.raises(RequestLimitError, match="budget exhausted"):
        client.fetch(synthetic_approved_plan())
    assert len(calls) == 1


def test_pacing_rejects_immediate_second_attempt_without_sleep_or_quota_use():
    now = [10.0]
    client = HistoricalForecastClient(
        allow_network=True,
        transport=lambda *args: b"{}",
        clock=lambda: now[0],
        limits=APILimits(max_requests=3, min_interval_seconds=2),
    )
    client.fetch(synthetic_approved_plan())
    with pytest.raises(RequestLimitError, match="interval"):
        client.fetch(synthetic_approved_plan())
    assert client.attempts == 1
    now[0] += 2
    client.fetch(synthetic_approved_plan())
    assert client.attempts == 2


@pytest.mark.parametrize(
    "payload", [b"x" * 101, b"bad", b"[]", b'{"error":true}', b"\xff"]
)
def test_invalid_payload_cannot_be_accepted(payload):
    client = HistoricalForecastClient(
        allow_network=True,
        transport=lambda *args: payload,
        limits=APILimits(max_response_bytes=100),
    )
    with pytest.raises(APIClientError):
        client.fetch(synthetic_approved_plan())
    assert client.attempts == 1


def test_http_reader_reads_at_most_limit_plus_one_and_rejects_large_response(
    monkeypatch,
):
    response = BytesIO(b"x" * 200)

    class Opener:
        def open(self, url, timeout):
            return response

    monkeypatch.setattr(
        "flight_delay.weather.api_client.build_opener", lambda *a: Opener()
    )
    with pytest.raises(APIClientError, match="byte limit"):
        bounded_json_get(candidate().url, 1, 100)


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("secret", 429, "secret", {}, None),
        URLError("secret"),
    ],
)
def test_http_errors_are_sanitized_and_never_retried(monkeypatch, error):
    calls = []

    class Opener:
        def open(self, url, timeout):
            calls.append(url)
            raise error

    monkeypatch.setattr(
        "flight_delay.weather.api_client.build_opener", lambda *a: Opener()
    )
    with pytest.raises(APIClientError) as caught:
        bounded_json_get(candidate().url, 1, 100)
    assert "secret" not in str(caught.value)
    assert len(calls) == 1


def test_redirects_cannot_bypass_request_budget():
    with pytest.raises(APIClientError, match="Redirects"):
        _NoRedirects().redirect_request(None, None, 302, "", {}, "https://elsewhere")
