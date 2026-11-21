"""Bounded, default-off JSON client for small historical forecast pilots."""

import hashlib
import json
import math
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, build_opener

from flight_delay.weather.api_providers import APIRequestPlan


class APIClientError(RuntimeError):
    """A network, policy, payload, or quota check failed."""


class RequestLimitError(APIClientError):
    """The local request budget or pacing limit has been reached."""


@dataclass(frozen=True)
class APILimits:
    """Local per-client limits, independent of a provider's billing units."""

    max_requests: int = 10
    max_response_bytes: int = 1_000_000
    timeout_seconds: float = 10.0
    min_interval_seconds: float = 1.0

    def __post_init__(self):
        for value in (self.max_requests, self.max_response_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "Request and response limits must be positive integers"
                )
        for value in (self.timeout_seconds, self.min_interval_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Timeout and interval must be finite and nonnegative")
        if self.timeout_seconds == 0:
            raise ValueError("Timeout must be positive")


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIClientError("Redirects are disabled; no follow-up request sent")


def bounded_json_get(url: str, timeout: float, max_bytes: int) -> bytes:
    """One HTTPS attempt; no redirects or automatic retries, bounded read."""
    try:
        with build_opener(_NoRedirects()).open(url, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
    except HTTPError as exc:
        # Never echo URLs or server bodies which could contain credentials.
        raise APIClientError(f"HTTP request failed with status {exc.code}") from None
    except (URLError, OSError, TimeoutError):
        raise APIClientError("HTTP transport failed") from None
    if len(payload) > max_bytes:
        raise APIClientError("Response exceeds byte limit")
    return payload


class HistoricalForecastClient:
    """Quarantine raw responses; no normalization, joining or model training.

    Paid plans are always rejected in this stage, even with network enabled.
    Every transport attempt (including errors) consumes a request. A batch
    must share one client; limits are not persistent across processes.
    """

    def __init__(
        self,
        *,
        limits=APILimits(),
        allow_network=False,
        transport=bounded_json_get,
        clock=monotonic,
    ):
        self.limits = limits
        self.allow_network = allow_network
        self._transport = transport
        self._clock = clock
        self._attempts = 0
        self._last_attempt = None
        self._lock = Lock()

    @property
    def attempts(self):
        return self._attempts

    def dry_run(self, plans: list[APIRequestPlan]) -> dict:
        """Deduplicate URLs without consuming quota or invoking transport."""
        unique = {plan.request_key: plan for plan in plans}
        requests = list(unique.values())
        return {
            "mode": "dry_run",
            "input_queries": len(plans),
            "unique_http_requests": len(requests),
            "http_attempts": 0,
            "remaining_local_requests": self.limits.max_requests - self.attempts,
            "within_request_budget": (
                len(requests) <= self.limits.max_requests - self.attempts
            ),
            "estimated_provider_billing_units": None,
            "observed_weather_coverage": None,
            "model_metrics": None,
            "ready_for_live": bool(requests)
            and self.allow_network
            and len(requests) <= self.limits.max_requests - self.attempts
            and all(not p.blockers and not p.paid for p in requests),
            "requests": [
                {
                    "provider": p.provider,
                    "model": p.model,
                    "request_key": p.request_key,
                    "url": p.url,
                    "run_at_utc": p.run_at_utc.isoformat(),
                    "run_reference_kind": p.run_reference_kind,
                    "assumed_available_at_utc": (
                        p.assumed_available_at_utc.isoformat()
                        if p.assumed_available_at_utc is not None
                        else None
                    ),
                    "paid": p.paid,
                    "blockers": list(p.blockers),
                    "evidence_url": p.evidence_url,
                }
                for p in requests
            ],
        }

    def fetch(self, plan: APIRequestPlan) -> dict:
        """Fetch one evidence-approved free request, or fail before transport."""
        if not self.allow_network:
            raise APIClientError("Network disabled; use dry_run")
        if plan.paid:
            raise APIClientError("Paid API execution is unavailable in stage 1")
        if plan.blockers:
            raise APIClientError("Historical forecast evidence is unresolved")
        if not plan.url.startswith(
            "https://single-runs-api.open-meteo.com/v1/forecast?"
        ):
            raise APIClientError("Endpoint is not approved for this client")
        with self._lock:
            if self._attempts >= self.limits.max_requests:
                raise RequestLimitError("Request budget exhausted")
            now = self._clock()
            if (
                self._last_attempt is not None
                and now - self._last_attempt < self.limits.min_interval_seconds
            ):
                raise RequestLimitError("Request interval limit reached; retry later")
            self._attempts += 1
            self._last_attempt = now
        try:
            raw = self._transport(
                plan.url,
                self.limits.timeout_seconds,
                self.limits.max_response_bytes,
            )
        except APIClientError:
            raise
        except Exception:
            raise APIClientError("HTTP transport failed") from None
        if not isinstance(raw, bytes) or len(raw) > self.limits.max_response_bytes:
            raise APIClientError("Invalid payload or response exceeds byte limit")
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            raise APIClientError("Response is not valid JSON") from None
        if not isinstance(data, dict) or data.get("error"):
            raise APIClientError("Response is not a successful JSON object")
        return {
            "request_key": plan.request_key,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "status": "raw_response_only_not_model_ready",
            "payload": data,
        }
