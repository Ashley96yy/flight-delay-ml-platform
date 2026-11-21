"""Bounded single-hour acquisition behind the frozen Stage 9A whitelist."""

import hashlib
import http.client
import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd

from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.config import RAW_WEATHER_COLUMNS
from flight_delay.weather.features import build_weather_feature_table
from flight_delay.weather.grib_backend import (
    build_grib_reference,
    parse_grib_inventory,
    select_weather_messages,
)
from flight_delay.weather.grib_decoder import decode_grib_airports
from flight_delay.weather.stage9b_protocol import allowed_hours, hour_key

MAX_ATTEMPTS = 240
HOUR_BYTES = 16 * 1024**2
TOTAL_BYTES = 384 * 1024**2
INDEX_BYTES = 64 * 1024
READ_CHUNK = 64 * 1024


class AcquisitionGateError(ValueError):
    """The given object, transfer, or campaign exceeds a hard safety bound."""


@contextmanager
def single_https_response(url, headers):
    """Open one HTTPS request with no redirect, automatic retry, or body buffering."""
    target = urlsplit(url)
    if (
        target.scheme != "https"
        or target.netloc != "noaa-hrrr-bdp-pds.s3.amazonaws.com"
        or target.query
        or target.fragment
    ):
        raise AcquisitionGateError("Unexpected NOAA HTTPS origin")
    connection = http.client.HTTPSConnection(target.netloc, timeout=30)
    try:
        connection.request("GET", target.path, headers=headers)
        with connection.getresponse() as response:
            yield response
    finally:
        connection.close()


@contextmanager
def campaign_ledger(cache_root):
    """Hold an exclusive local POSIX lock for the entire acquire/cache commit."""
    import fcntl

    cache_root.mkdir(parents=True, exist_ok=True)
    db_path = cache_root / "stage9b_budget.sqlite"
    with (cache_root / "stage9b.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AcquisitionGateError(
                "Stage 9B acquisition already in progress"
            ) from exc
        lock.seek(0)
        marker = lock.read()
        if marker and not db_path.is_file():
            raise AcquisitionGateError(
                "Initialized campaign has lost its budget ledger"
            )
        with closing(sqlite3.connect(db_path, isolation_level=None)) as db:
            db.execute("PRAGMA synchronous = FULL")
            if (
                not marker
                and not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='campaign'"
                ).fetchone()
            ):
                if db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
                    raise AcquisitionGateError(
                        "Existing budget ledger has an unknown schema"
                    )
                db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE campaign (id TEXT PRIMARY KEY);
                    CREATE TABLE requests (
                        id INTEGER PRIMARY KEY, hour TEXT NOT NULL, field TEXT NOT NULL,
                        url TEXT NOT NULL, requested_range TEXT,
                        reserved INTEGER NOT NULL DEFAULT 0, actual INTEGER, status INTEGER,
                        etag TEXT, content_range TEXT, digest TEXT, outcome TEXT NOT NULL DEFAULT 'incomplete',
                        attempted_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE committed (
                        hour TEXT PRIMARY KEY, parquet_sha256 TEXT NOT NULL,
                        first_attempt INTEGER NOT NULL, last_attempt INTEGER NOT NULL
                    );
                """)
                db.execute("INSERT INTO campaign VALUES(?)", (uuid4().hex,))
                db.execute("COMMIT")

            try:
                ids = db.execute("SELECT id FROM campaign").fetchall()
                db.execute("SELECT COUNT(*) FROM requests")
                db.execute("SELECT COUNT(*) FROM committed")
            except sqlite3.DatabaseError as exc:
                raise AcquisitionGateError(
                    "Campaign ledger is unreadable or truncated"
                ) from exc
            if len(ids) != 1 or not re.fullmatch(r"[0-9a-f]{32}", ids[0][0]):
                raise AcquisitionGateError("Invalid campaign ledger identity")
            expected = ids[0][0].encode()
            if marker and marker != expected:
                raise AcquisitionGateError("Campaign marker and budget ledger disagree")
            if not marker:
                lock.write(expected)
                lock.flush()
                os.fsync(lock.fileno())
            yield db


def _remaining(db, hour):
    total, per_hour = db.execute(
        "SELECT COALESCE(SUM(reserved), 0), "
        "COALESCE(SUM(CASE WHEN hour=? THEN reserved ELSE 0 END), 0) FROM requests",
        (hour,),
    ).fetchone()
    return min(HOUR_BYTES - per_hour, TOTAL_BYTES - total)


def _fetch(db, hour, field, url, opener, expected_hash, bounds=None, object_id=None):
    """Charge the attempt and pessimistic byte reservation before any body read."""
    if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= MAX_ATTEMPTS:
        raise AcquisitionGateError("Campaign HTTP attempt budget exhausted")
    if _remaining(db, hour) <= 0:
        raise AcquisitionGateError("Campaign response body budget exhausted")
    headers = {"Accept-Encoding": "identity", "Connection": "close"}
    if bounds is not None:
        start, end = bounds
        headers["Range"] = f"bytes={start}-{end}"
        if object_id:
            headers["If-Match"] = object_id[0]
    attempt = db.execute(
        "INSERT INTO requests(hour,field,url,requested_range) VALUES(?,?,?,?)",
        (hour, field, url, headers.get("Range")),
    ).lastrowid

    status, etag, digest, outcome = None, None, None, "failed"
    content_range = None
    response_identity = None
    body = bytearray()
    try:
        with opener(url, headers) as response:
            status = response.status
            if status != (200 if bounds is None else 206):
                raise AcquisitionGateError(f"Unexpected HTTP {status} for {field}")
            if response.getheader("Content-Encoding") not in (None, "identity"):
                raise AcquisitionGateError("Response encoding must be identity")
            if response.getheader("Transfer-Encoding") is not None:
                raise AcquisitionGateError("Transfer-Encoding is not supported")
            declared = response.getheader("Content-Length", "")
            if not re.fullmatch(r"[0-9]+", declared):
                raise AcquisitionGateError("Content-Length is required")
            length = int(declared)
            if length <= 0 or length + 1 > _remaining(db, hour):
                raise AcquisitionGateError("Response body budget exceeded")
            etag = response.getheader("ETag")
            if bounds is None:
                if length > INDEX_BYTES:
                    raise AcquisitionGateError("Index body limit exceeded")
            else:
                content_range = response.getheader("Content-Range", "")
                match = re.fullmatch(rf"bytes {start}-{end}/([0-9]+)", content_range)
                if (
                    length != end - start + 1
                    or not match
                    or int(match[1]) <= end
                    or length == int(match[1])
                ):
                    raise AcquisitionGateError(
                        "Content-Range does not identify the requested partial object"
                    )
                if (
                    not etag
                    or not re.fullmatch(r'"[^"\r\n]+"', etag)
                    or (object_id and (etag, int(match[1])) != object_id)
                ):
                    raise AcquisitionGateError(
                        "Unverifiable or changed GRIB object identity"
                    )

                response_identity = (etag, int(match[1]))

            # Reserve the declared bytes plus one bounded EOF probe durably. On a
            # hard crash the reservation remains and counts against both limits.
            db.execute(
                "UPDATE requests SET reserved=? WHERE id=?", (length + 1, attempt)
            )
            while len(body) <= length:
                want = min(READ_CHUNK, length + 1 - len(body))
                if not want:
                    break
                part = response.read(want)
                if len(part) > want:
                    raise AcquisitionGateError(
                        "Transport violated bounded read contract"
                    )
                if not part:
                    break
                body.extend(part)
            if len(body) != length:
                raise AcquisitionGateError(
                    "Response length differs from declared length"
                )
            digest = hashlib.sha256(body).hexdigest()
            if expected_hash is not None and digest != expected_hash:
                raise AcquisitionGateError(f"SHA-256 mismatch for {field}")
        outcome = "verified_response"
        return bytes(body), response_identity
    except http.client.IncompleteRead as exc:
        body.extend(exc.partial)
        raise
    finally:
        db.execute(
            "UPDATE requests SET actual=?,status=?,etag=?,content_range=?,digest=?,outcome=? WHERE id=?",
            (len(body), status, etag, content_range, digest, outcome, attempt),
        )


def _expected_hashes(expected_sha256, names):
    supplied = expected_sha256 or {}
    if set(supplied) - names or any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in supplied.values()
    ):
        raise AcquisitionGateError("Malformed trusted response SHA-256 mapping")
    return supplied


def acquire_frozen_hour(
    cache_root,
    *,
    forecast_run_at_utc,
    forecast_hour,
    forecast_valid_at_utc,
    opener,
    expected_sha256=None,
    decoder=decode_grib_airports,
):
    """Acquire exactly one whitelisted hour via an explicitly supplied transport."""
    run, lead, valid = hour_key(
        forecast_run_at_utc, forecast_hour, forecast_valid_at_utc
    )
    if (run, lead, valid) not in allowed_hours():
        raise AcquisitionGateError("Forecast identity is not in the Stage 9A whitelist")
    ref = build_grib_reference(run, lead)
    if ref.valid_at != valid:
        raise AcquisitionGateError("Forecast valid time mismatch")
    cache = WeatherCache(cache_root)
    hour = f"{run.isoformat()}/F{lead:02d}/{valid.isoformat()}"
    with campaign_ledger(cache.root.resolve()) as db:
        receipt = db.execute(
            "SELECT parquet_sha256 FROM committed WHERE hour=?", (hour,)
        ).fetchone()
        cached = cache.read_hour(run, lead)
        if receipt:
            manifest = cache.entry_directory(run, lead) / "manifest.json"
            if (
                cached is None
                or json.loads(manifest.read_bytes())["sha256"] != receipt[0]
            ):
                raise AcquisitionGateError(
                    "Previously committed cache is missing or changed"
                )
        if cached is not None:
            build_weather_feature_table(cached)
            return cached
        first = db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM requests").fetchone()[
            0
        ]
        supplied = _expected_hashes(expected_sha256, {"index", *RAW_WEATHER_COLUMNS})
        index, _ = _fetch(
            db, hour, "index", ref.index_url, opener, supplied.get("index")
        )
        lines = index.decode("utf-8").splitlines()
        if not all(
            line.split(":")[2:3] == [f"d={run:%Y%m%d%H}"]
            for line in lines
            if line.strip()
        ):
            raise AcquisitionGateError("Index contains the wrong forecast run")
        ranges = select_weather_messages(parse_grib_inventory(lines), lead)
        if int(ranges["size_bytes"].sum()) + len(ranges) > _remaining(db, hour):
            raise AcquisitionGateError(
                "Indexed ranges exceed the remaining body budget"
            )
        attempts = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        if attempts + len(ranges) > MAX_ATTEMPTS:
            raise AcquisitionGateError(
                "Indexed ranges exceed the remaining request budget"
            )

        parts = []
        object_id = None
        for row in ranges.itertuples(index=False):
            payload, object_id = _fetch(
                db,
                hour,
                row.feature,
                ref.grib_url,
                opener,
                supplied.get(row.feature),
                (row.start_byte, row.end_byte),
                object_id,
            )
            if not payload.startswith(b"GRIB") or not payload.endswith(b"7777"):
                raise AcquisitionGateError("Invalid GRIB message framing")
            parts.append(decoder(payload, row.feature, ref))
        frame = pd.concat(parts, ignore_index=True)
        if (
            not frame["feature"].isin(RAW_WEATHER_COLUMNS).all()
            or not frame["forecast_run_at_utc"].eq(run).all()
            or not frame["forecast_hour"].eq(lead).all()
            or not frame["forecast_valid_at_utc"].eq(valid).all()
            or not frame["backend"].eq("grib").all()
        ):
            raise AcquisitionGateError("Decoded forecast identity or backend mismatch")
        build_weather_feature_table(frame)
        path = cache.write_hour(run, lead, frame)
        verified = cache.read_hour(run, lead)
        if verified is None:
            raise AcquisitionGateError("Committed cache cannot be reread")
        last = db.execute("SELECT MAX(id) FROM requests").fetchone()[0]
        db.execute(
            "INSERT INTO committed VALUES(?,?,?,?)",
            (hour, path.stem.removeprefix("weather-"), first, last),
        )
        return verified
