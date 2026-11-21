"""IEM archived operational MOS: bounded run requests and raw-only acquisition.

Archive access is distinct from historical availability certification. This
module never labels a forecast safe for a model merely because HTTP succeeds.
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from uuid import uuid4

import pandas as pd

from flight_delay.weather.api_client import (
    APIClientError,
    APILimits,
    RequestLimitError,
    _NoRedirects,
)
from flight_delay.weather.api_providers import APIRequestPlan, utc_timestamp
from flight_delay.weather.config import ORIGIN_AIRPORTS

ENDPOINT = "https://mesonet.agron.iastate.edu/api/1/mos.json"
STATIONS = tuple("K" + code for code in ORIGIN_AIRPORTS)
MODELS = ("GFS", "NBS")


class MOSContractError(ValueError):
    """MOS identity, time, encoding or provenance is invalid."""


class MOSHTTPError(APIClientError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"IEM HTTP status {status}")


class MOSTransportError(APIClientError):
    """A transient network/timeout failure, with no sensitive error details."""


@dataclass(frozen=True)
class MOSRunRequest:
    model: str
    runtime_utc: pd.Timestamp
    stations: tuple[str, ...] = STATIONS

    def __post_init__(self):
        run = utc_timestamp(self.runtime_utc)
        if self.model not in MODELS:
            raise MOSContractError("Only explicitly supported GFS/NBS products allowed")
        if (
            not isinstance(self.stations, tuple)
            or not 1 <= len(self.stations) <= 6
            or len(set(self.stations)) != len(self.stations)
            or not set(self.stations) <= set(STATIONS)
        ):
            raise MOSContractError("Invalid, duplicate or unsupported MOS station")
        if run != run.floor("h"):
            raise MOSContractError("MOS cycle must be an exact hour")
        if self.model == "GFS":
            hours = (0, 6, 12, 18)
        elif run < pd.Timestamp("2026-05-05T00:00Z"):
            hours = (1, 7, 13, 19)
        else:
            hours = (0, 6, 12, 18)
        if run.hour not in hours:
            raise MOSContractError("Cycle is not in the retained historical archive")
        object.__setattr__(self, "runtime_utc", run)
        object.__setattr__(self, "stations", tuple(sorted(self.stations)))


class IEMMOSProvider:
    """Adapt native MOS run requests to the shared immutable API plan boundary.

    MOS has its own field contract and acquisition request type. The HRRR API
    feature list is deliberately not reused or silently translated.
    """

    def plan(self, request: MOSRunRequest) -> APIRequestPlan:
        if not isinstance(request, MOSRunRequest):
            raise MOSContractError("IEM needs a native MOSRunRequest")
        params = tuple(("station", s) for s in request.stations) + (
            ("model", request.model),
            ("runtime", request.runtime_utc.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
        return APIRequestPlan(
            provider="iem_mos",
            model=request.model,
            endpoint=ENDPOINT,
            params=params,
            run_at_utc=request.runtime_utc,
            run_reference_kind="model_cycle_not_publication_time",
            assumed_available_at_utc=None,
            paid=False,
            blockers=("availability_unverified", "native_mos_contract_required"),
            evidence_url="https://mesonet.agron.iastate.edu/mos/",
        )

    def dry_run(self, requests, *, max_requests=14) -> dict:
        from flight_delay.weather.api_client import HistoricalForecastClient

        plans = [self.plan(r) for r in requests]
        report = HistoricalForecastClient(
            limits=APILimits(max_requests=max_requests)
        ).dry_run(plans)
        report["purpose"] = "raw_archive_research_only"
        report["model_ready"] = False
        return report


def validate_plan(plan: APIRequestPlan) -> MOSRunRequest:
    """Reject other products, implicit latest runs and injected parameters."""
    if plan.endpoint != ENDPOINT or plan.provider != "iem_mos" or plan.paid:
        raise MOSContractError("Only the free explicit-run IEM MOS endpoint is allowed")
    params = list(plan.params)
    if {k for k, _ in params} != {"station", "model", "runtime"}:
        raise MOSContractError("Unexpected request parameters")
    if (
        sum(k == "model" for k, _ in params) != 1
        or sum(k == "runtime" for k, _ in params) != 1
    ):
        raise MOSContractError("Model and runtime must be explicit and unique")
    request = MOSRunRequest(
        dict(params)["model"],
        dict(params)["runtime"],
        tuple(v for k, v in params if k == "station"),
    )
    canonical = IEMMOSProvider().plan(request)
    if plan.request_key != canonical.request_key:
        raise MOSContractError("Noncanonical or inconsistent request identity")
    if plan.run_at_utc != request.runtime_utc or plan.model != request.model:
        raise MOSContractError("Plan metadata does not match URL identity")
    return request


def iem_json_get(url, timeout, max_bytes):
    """One bounded GET; redirects disabled and HTTP status preserved for retry."""
    request = Request(
        url, headers={"User-Agent": "FlightDelayResearch/0.1 (tiny MOS pilot)"}
    )
    try:
        with build_opener(_NoRedirects()).open(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
    except HTTPError as exc:
        raise MOSHTTPError(exc.code) from None
    except (URLError, OSError, TimeoutError):
        raise MOSTransportError("IEM transport failed") from None
    if len(body) > max_bytes:
        raise APIClientError("IEM response exceeds byte limit")
    return body


def _atomic_json(path, value):
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class IEMMOSClient:
    """Free archive inspection client. Returned data is NOT leakage-approved.

    Cache and append-only attempt receipts survive restart. One pilot must share
    this cache/ledger; parallel writers are unsupported and rejected with a lock.
    Failed and retried HTTP attempts each consume the persistent request budget.
    """

    def __init__(
        self,
        cache_dir,
        *,
        allow_network=False,
        limits=APILimits(
            max_requests=14,
            max_response_bytes=2_000_000,
            timeout_seconds=20,
            min_interval_seconds=1,
        ),
        max_retries=1,
        transport=iem_json_get,
        clock=monotonic,
        sleeper=sleep,
    ):
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("Retries must be an integer between zero and two")
        self.cache_dir = Path(cache_dir)
        self.allow_network = allow_network
        self.limits = limits
        self.max_retries = max_retries
        self.transport, self.clock, self.sleeper = transport, clock, sleeper
        self._last_attempt = None

    @property
    def ledger_path(self):
        return self.cache_dir / "attempts.json"

    def _ledger(self):
        if not self.ledger_path.exists():
            return []
        try:
            rows = json.loads(self.ledger_path.read_text())
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                raise ValueError()
            return rows
        except (ValueError, OSError):
            raise APIClientError("Corrupt persistent MOS request ledger") from None

    def _cache_path(self, plan):
        return self.cache_dir / (plan.request_key + ".json")

    def _read_cache(self, plan):
        path = self._cache_path(plan)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
            raw = base64.b64decode(value["body_base64"], validate=True)
            receipt = value["receipt"]
            if (
                value["schema_version"] != 1
                or receipt["request_key"] != plan.request_key
                or receipt["url"] != plan.url
                or receipt["sha256"] != hashlib.sha256(raw).hexdigest()
                or receipt["bytes"] != len(raw)
                or len(raw) > self.limits.max_response_bytes
            ):
                raise ValueError()
            utc_timestamp(receipt["retrieved_at_utc"])
            return raw, dict(receipt, cache_status="hit")
        except (ValueError, KeyError, TypeError, OSError):
            raise APIClientError(
                "MOS cache failed identity/integrity validation"
            ) from None

    def fetch(self, plan):
        """Inspect a planned free archive run despite unresolved release evidence.

        This is a separate research-only operation from the stage-1 client's
        evidence-approved fetch. No source blocker is removed or overridden.
        """
        validate_plan(plan)
        cached = self._read_cache(plan)
        if cached is not None:
            return cached
        if not self.allow_network:
            raise APIClientError("Network disabled and MOS cache is missing")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = self.cache_dir / "writer.lock"
        try:
            handle = lock.open("x")
        except FileExistsError:
            raise APIClientError(
                "Another MOS writer is active; no HTTP request sent"
            ) from None
        try:
            handle.close()
            for retry in range(self.max_retries + 1):
                ledger = self._ledger()
                if len(ledger) >= self.limits.max_requests:
                    raise RequestLimitError("Persistent MOS request budget exhausted")
                # This is a tiny pilot ceiling, not an account-wide API rate.
                total_bytes = sum(r.get("bytes", 0) for r in ledger)
                if total_bytes + self.limits.max_response_bytes > 8_000_000:
                    raise RequestLimitError("MOS pilot response-byte budget exhausted")
                if self._last_attempt is not None:
                    wait = self.limits.min_interval_seconds - (
                        self.clock() - self._last_attempt
                    )
                    if wait > 0:
                        self.sleeper(wait)
                self._last_attempt = self.clock()
                receipt = {
                    "request_key": plan.request_key,
                    "url": plan.url,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "attempt": len(ledger) + 1,
                    "retry": retry,
                    "status": "started",
                    "bytes": 0,
                    "availability_status": "availability_unverified",
                    "model_ready": False,
                }
                ledger.append(receipt)
                _atomic_json(self.ledger_path, ledger)  # reserve BEFORE transport
                try:
                    raw = self.transport(
                        plan.url,
                        self.limits.timeout_seconds,
                        self.limits.max_response_bytes,
                    )
                    if (
                        not isinstance(raw, bytes)
                        or len(raw) > self.limits.max_response_bytes
                    ):
                        raise APIClientError("Invalid or oversized MOS response")
                    receipt.update(
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
                    )
                    value = json.loads(raw)
                    if not isinstance(value, dict) or not isinstance(
                        value.get("data"), list
                    ):
                        raise APIClientError(
                            "MOS response is not a table-schema object"
                        )
                except Exception as exc:
                    retryable = isinstance(exc, MOSHTTPError) and exc.status in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    # Timeout/I/O errors from injectable transports can retry.
                    retryable |= isinstance(
                        exc, (MOSTransportError, TimeoutError, URLError, OSError)
                    )
                    receipt.update(
                        status="error",
                        error_type=type(exc).__name__,
                        http_status=getattr(exc, "status", None),
                    )
                    _atomic_json(self.ledger_path, ledger)
                    if retryable and retry < self.max_retries:
                        self.sleeper(1 + retry)
                        continue
                    raise APIClientError(
                        "MOS request failed; see persisted attempt receipt"
                    ) from None
                receipt.update(status="success", http_status=200)
                _atomic_json(self.ledger_path, ledger)
                _atomic_json(
                    self._cache_path(plan),
                    {
                        "schema_version": 1,
                        "receipt": receipt,
                        "body_base64": base64.b64encode(raw).decode("ascii"),
                    },
                )
                return raw, dict(receipt, cache_status="miss")
        finally:
            lock.unlink(missing_ok=True)
