"""Stage 9B safety, pairing and evaluation tests. All responses are simulated."""

import hashlib
import http.client
import io
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather import stage9b_acquisition as acq
from flight_delay.weather import stage9b_pairing as pairing
from flight_delay.weather.cache import WeatherCache, WeatherCacheIntegrityError
from flight_delay.weather.config import AIRPORTS, WEATHER_VARIABLES
from flight_delay.weather.features import WeatherFeatureError
from flight_delay.weather.grib_backend import GRIBInventoryError, build_weather_patterns
from flight_delay.weather.stage9b_evaluation import (
    bootstrap_daily_ap,
    decision,
    evaluate_predictions,
)
from flight_delay.weather.stage9b_protocol import allowed_hours, frozen_hours

KEYS = list(allowed_hours())
MESSAGE = b"GRIBpayload7777"


def synthetic_hour(run, lead, valid, n=0):
    """Build explicitly synthetic values in the exact raw weather schema."""
    return pd.DataFrame(
        [
            {
                "Origin": airport.code,
                "feature": variable.feature,
                "forecast_run_at_utc": run,
                "forecast_valid_at_utc": valid,
                "forecast_hour": lead,
                "assumed_available_at_utc": run + pd.Timedelta(hours=3),
                "value": float(n + index + 1),
                "raw_value": float(n + index + 1),
                "value_status": "valid",
                "grid_latitude": airport.latitude,
                "grid_longitude": airport.longitude,
                "backend": "grib",
                "storage_dtype": None,
                "units": variable.grib_units,
                "source_units": variable.grib_units,
                "fallback_reason": None,
            }
            for index, airport in enumerate(AIRPORTS)
            for variable in WEATHER_VARIABLES
        ]
    )


class MockResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.body = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.reads = []

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read(self, amount):
        assert 0 <= amount <= acq.READ_CHUNK
        self.reads.append(amount)
        return self.body.read(amount)


def index_body(run, lead, *, missing=False, stride=len(MESSAGE)):
    patterns = list(build_weather_patterns(lead).values())
    if missing:
        patterns.pop()
    return "".join(
        f"{i + 1}:{i * stride}:d={run:%Y%m%d%H}{pattern}\n"
        for i, pattern in enumerate(patterns + [":spare:"])
    ).encode()


def mock_archive(key, *, bad_range=None, index=None):
    run, lead, _ = key
    responses, calls = [], []

    @contextmanager
    def opener(url, headers):
        calls.append((url, headers.copy()))
        if url.endswith(".idx"):
            response = MockResponse(
                200, index if index is not None else index_body(run, lead)
            )
        else:
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            response = MockResponse(
                206,
                MESSAGE,
                {"Content-Range": f"bytes {start}-{end}/9999", "ETag": '"object"'},
            )
            if bad_range:
                bad_range(response)
        responses.append(response)
        yield response

    return opener, responses, calls


def acquire(root, key, opener, **kwargs):
    return acq.acquire_frozen_hour(
        root,
        forecast_run_at_utc=key[0],
        forecast_hour=key[1],
        forecast_valid_at_utc=key[2],
        opener=opener,
        decoder=lambda payload, feature, ref: synthetic_hour(*key).query(
            "feature == @feature"
        ),
        **kwargs,
    )


def ledger(root, query):
    with sqlite3.connect(root / "stage9b_budget.sqlite") as db:
        return db.execute(query).fetchall()


def test_full_hour_has_ten_receipts_and_verified_cache_hit(tmp_path):
    key = KEYS[0]
    opener, _, calls = mock_archive(key)
    result = acquire(tmp_path, key, opener)
    assert len(result) == 54
    assert len(calls) == 10
    assert calls[1][1].get("If-Match") is None
    assert [headers["If-Match"] for _, headers in calls[2:]] == ['"object"'] * 8
    observed = ledger(tmp_path, "SELECT reserved,actual,digest,outcome FROM requests")
    assert all(
        reserved == actual + 1 and len(h) == 64 and status == "verified_response"
        for reserved, actual, h, status in observed
    )
    assert ledger(tmp_path, "SELECT first_attempt,last_attempt FROM committed") == [
        (1, 10)
    ]
    restored = acquire(
        tmp_path, key, lambda *_: pytest.fail("cache hit requested HTTP")
    )
    pd.testing.assert_frame_equal(result, restored)

    cache = WeatherCache(tmp_path)
    path = next(cache.entry_directory(*key[:2]).glob("weather-*.parquet"))
    path.write_bytes(b"corrupt")
    with pytest.raises(WeatherCacheIntegrityError, match="checksum"):
        acquire(
            tmp_path, key, lambda *_: pytest.fail("corrupt cache was fetched again")
        )
    assert len(ledger(tmp_path, "SELECT id FROM requests")) == 10


@pytest.mark.parametrize(
    "alter",
    [
        lambda r: setattr(r, "status", 200),
        lambda r: r.headers.update({"Content-Range": "bytes 0-4/9999"}),
        lambda r: r.headers.update({"Content-Length": str(len(MESSAGE) + 1)}),
        lambda r: r.headers.update({"ETag": "unquoted"}),
        lambda r: r.headers.update({"Transfer-Encoding": "chunked"}),
    ],
)
def test_unsafe_range_is_rejected_before_body(tmp_path, alter):
    opener, responses, calls = mock_archive(KEYS[0], bad_range=alter)
    with pytest.raises(acq.AcquisitionGateError):
        acquire(tmp_path, KEYS[0], opener)
    assert len(calls) == 2
    assert responses[-1].reads == []
    assert WeatherCache(tmp_path).read_hour(*KEYS[0][:2]) is None


@pytest.mark.parametrize("offset", (-1, 1))
def test_short_and_unending_streams_fail_closed(tmp_path, offset):
    key = KEYS[0]

    class Endless:
        def read(self, n):
            return b"X" * n

    def alter(r):
        r.body = io.BytesIO(MESSAGE[:-1]) if offset < 0 else Endless()

    opener, responses, _ = mock_archive(key, bad_range=alter)
    with pytest.raises(acq.AcquisitionGateError, match="length differs"):
        acquire(tmp_path, key, opener)
    assert max(responses[-1].reads) <= len(MESSAGE) + 1
    assert ledger(tmp_path, "SELECT reserved,actual FROM requests WHERE id=2") == [
        (len(MESSAGE) + 1, len(MESSAGE) + offset)
    ]
    assert WeatherCache(tmp_path).read_hour(*key[:2]) is None


def test_whitelist_validates_all_three_fields(tmp_path):
    run, lead, valid = KEYS[0]
    fail = lambda *_: pytest.fail("unauthorized network attempt")  # noqa: E731
    for key in (
        (run + pd.Timedelta(days=1), lead, valid),
        (run, lead + 1, valid),
        (run, lead, valid + pd.Timedelta(hours=1)),
    ):
        with pytest.raises(acq.AcquisitionGateError, match="whitelist"):
            acquire(tmp_path, key, fail)
    assert not (tmp_path / "stage9b_budget.sqlite").exists()


@pytest.mark.parametrize(
    "reason", ["missing", "wrong_run", "index_hash", "range_hash", "overlarge"]
)
def test_index_and_payload_failure_leave_no_cache(tmp_path, reason):
    run, lead, _ = KEYS[0]
    index = index_body(
        run + pd.Timedelta(days=int(reason == "wrong_run")),
        lead,
        missing=reason == "missing",
        stride=acq.HOUR_BYTES if reason == "overlarge" else len(MESSAGE),
    )
    expected = {
        "index_hash": {"index": "0" * 64},
        "range_hash": {next(iter(build_weather_patterns(lead))): "0" * 64},
    }.get(reason)
    opener, _, calls = mock_archive(KEYS[0], index=index)
    with pytest.raises((acq.AcquisitionGateError, GRIBInventoryError)):
        acquire(tmp_path, KEYS[0], opener, expected_sha256=expected)
    assert len(calls) == (2 if reason == "range_hash" else 1)
    assert WeatherCache(tmp_path).read_hour(run, lead) is None


@pytest.mark.parametrize("budget", ["hour", "total", "attempt"])
def test_persistent_budget_does_not_reset_between_invocations(
    tmp_path, monkeypatch, budget
):
    key = KEYS[0]
    size = len(index_body(*key[:2])) + 1
    if budget == "hour":
        monkeypatch.setattr(acq, "HOUR_BYTES", size)
    elif budget == "total":
        monkeypatch.setattr(acq, "TOTAL_BYTES", size)
    else:
        monkeypatch.setattr(acq, "MAX_ATTEMPTS", 1)
    opener, _, calls = mock_archive(key)
    with pytest.raises(acq.AcquisitionGateError, match="remaining"):
        acquire(tmp_path, key, opener)
    other = KEYS[1] if budget != "hour" else key
    with pytest.raises(acq.AcquisitionGateError, match="exhausted"):
        acquire(tmp_path, other, opener)
    assert len(calls) == 1


def test_connection_failure_is_charged_across_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(acq, "MAX_ATTEMPTS", 1)

    def connection_failure(*_):
        raise OSError("DNS failure")

    with pytest.raises(OSError, match="DNS"):
        acquire(tmp_path, KEYS[0], connection_failure)
    with pytest.raises(acq.AcquisitionGateError, match="attempt budget exhausted"):
        acquire(tmp_path, KEYS[0], lambda *_: pytest.fail("retried over budget"))
    assert ledger(tmp_path, "SELECT outcome,reserved,actual FROM requests") == [
        ("failed", 0, 0)
    ]


def test_exclusive_lock_and_lost_budget_ledger(tmp_path):
    with acq.campaign_ledger(tmp_path):
        with pytest.raises(acq.AcquisitionGateError, match="already in progress"):
            with acq.campaign_ledger(tmp_path):
                pytest.fail("Concurrent lock succeeded")
    (tmp_path / "stage9b_budget.sqlite").unlink()
    with pytest.raises(acq.AcquisitionGateError, match="lost"):
        with acq.campaign_ledger(tmp_path):
            pytest.fail("Lost ledger was reset")


def test_os_exit_retains_attempt_and_unsettled_bytes(tmp_path):
    script = """
import os
from pathlib import Path
from contextlib import contextmanager
from flight_delay.weather.stage9b_acquisition import _fetch,campaign_ledger
class Response:
    status = 200
    def getheader(self,key,default=None):
        return "9" if key == "Content-Length" else default
    def read(self,n):
        os._exit(43)
@contextmanager
def opener(*args):
    yield Response()
with campaign_ledger(Path(os.environ["STAGE9B_CRASH_DIR"])) as db:
    _fetch(db, "test", "index", "mock", opener, None)
"""
    env = dict(os.environ, STAGE9B_CRASH_DIR=str(tmp_path))
    env["PYTHONPATH"] = str(Path(acq.__file__).resolve().parents[2])
    result = subprocess.run([sys.executable, "-c", script], env=env, check=False)
    assert result.returncode == 43
    with acq.campaign_ledger(tmp_path) as db:
        assert db.execute(
            "SELECT reserved,actual,outcome FROM requests"
        ).fetchall() == [(10, None, "incomplete")]


def test_decode_or_weather_contract_failure_never_commits(tmp_path):
    key = KEYS[0]
    opener, _, _ = mock_archive(key)

    def wrong_decode(payload, feature, ref):
        part = synthetic_hour(*key).query("feature == @feature").copy()
        part["Origin"] = "ATL"
        return part

    with pytest.raises(WeatherFeatureError):
        acq.acquire_frozen_hour(
            tmp_path,
            forecast_run_at_utc=key[0],
            forecast_hour=key[1],
            forecast_valid_at_utc=key[2],
            opener=opener,
            decoder=wrong_decode,
        )
    assert WeatherCache(tmp_path).read_hour(*key[:2]) is None


@pytest.fixture
def mock_cohort(monkeypatch, tmp_path):
    params = tmp_path / "fixed.json"
    params.write_bytes(b"test parameter fixture")
    protocol = pairing.frozen_protocol()
    protocol["params_sha256"] = hashlib.sha256(params.read_bytes()).hexdigest()
    monkeypatch.setattr(pairing, "frozen_protocol", lambda: protocol)
    records = []
    for n, (key, info) in enumerate(allowed_hours().items()):
        for i in range(4):
            records.append(
                {
                    "FlightDate": pd.Timestamp(info["date"]),
                    "Reporting_Airline": "DL",
                    "Origin": "ATL",
                    "Dest": "LGA" if n < 12 else "BOS",
                    "CRSDepTime": 900 + i,
                    "CRSArrTime": 1100 + i,
                    "CRSElapsedTime": 120.0,
                    "Distance": 760.0,
                    "scheduled_departure_utc": key[2] + pd.Timedelta(minutes=i),
                    "flight_id": f"{n}-{i}",
                    "split": info["split"],
                    "date": info["date"],
                    "DepDel15": i % 2,
                    "ArrDel15": i % 2,
                    "bts_sha256": "test",
                }
            )
    cohort = pd.DataFrame(records)
    monkeypatch.setattr(pairing, "load_frozen_cohort", lambda _: cohort)
    return params, cohort


@pytest.fixture
def synthetic_cache(tmp_path):
    cache = WeatherCache(tmp_path / "weather")
    for i, key in enumerate(KEYS):
        cache.write_hour(*key[:2], synthetic_hour(*key, n=i))
    return cache


def test_identical_feature_rows_and_provenance(mock_cohort, synthetic_cache, tmp_path):
    params, cohort = mock_cohort
    result = pairing.prepare_pairs(tmp_path, synthetic_cache.root, params)
    assert result.manifest.index.tolist() == cohort.flight_id.tolist()
    pd.testing.assert_frame_equal(
        result.baseline, result.weather[result.baseline.columns]
    )
    pd.testing.assert_frame_equal(
        result.manifest[pairing.TARGETS],
        cohort.set_index("flight_id")[pairing.TARGETS],
    )
    assert len(result.baseline) == 96
    assert (
        result.baseline.loc[result.manifest["split"] == "validation", "Dest"]
        .isna()
        .all()
    )
    assert result.manifest.weather_cache_sha256.str.len().eq(64).all()
    assert result.training_weather_profile["unique"].eq(12).all().all()
    assert result.manifest.forecast_hour.eq(31).all()
    assert (
        (result.manifest.scheduled_departure_utc - result.manifest.prediction_time_utc)
        .eq(pd.Timedelta(hours=24))
        .all()
    )
    assert result.pit_status == "retrospective_exploratory"


@pytest.mark.parametrize("error", ["missing", "duplicate", "time", "airport", "nan"])
def test_join_failure_never_drops_rows(
    mock_cohort, synthetic_cache, tmp_path, monkeypatch, error
):
    params, _ = mock_cohort
    if error == "missing":
        path = synthetic_cache.entry_directory(*KEYS[0][:2]) / "manifest.json"
        path.unlink()
    else:
        real = pairing._cache_weather

        def damaged(cache):
            features, evidence = real(cache)
            if error == "duplicate":
                features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
            elif error == "time":
                features.loc[0, "forecast_run_at_utc"] -= pd.Timedelta(hours=6)
            elif error == "airport":
                features.loc[0, "Origin"] = "NONEXISTENT"
            else:
                features.loc[0, "origin_temperature_c"] = np.nan
            return features, evidence

        monkeypatch.setattr(pairing, "_cache_weather", damaged)
    with pytest.raises(ValueError):
        pairing.prepare_pairs(tmp_path, synthetic_cache.root, params)


def test_frozen_params_are_checked_before_pairing(
    mock_cohort, synthetic_cache, tmp_path
):
    params, _ = mock_cohort
    params.write_bytes(b"edited params")
    with pytest.raises(ValueError, match="parameters changed"):
        pairing.prepare_pairs(tmp_path, synthetic_cache.root, params)


def _validation_dates():
    return [r["date"] for r in frozen_hours() if r["split"] == "validation"]


def test_reproducible_quarter_bootstrap_and_leave_out():
    frame = pd.DataFrame(
        {"DepDel15": np.linspace(0.01, 0.12, 12), "ArrDel15": [0.0] * 12},
        index=_validation_dates(),
    )
    first = bootstrap_daily_ap(frame)
    pd.testing.assert_frame_equal(first, bootstrap_daily_ap(frame))
    assert first.loc["DepDel15", "mean"] == pytest.approx(0.065)
    assert first.loc["DepDel15", "leave_one_date_out_min"] == pytest.approx(0.06)
    assert first.loc["DepDel15", "lower"] == pytest.approx(0.06)
    assert first.loc["DepDel15", "upper"] == pytest.approx(0.07)
    with pytest.raises(ValueError, match="12 ordered"):
        bootstrap_daily_ap(frame.iloc[:-1])


@pytest.mark.parametrize(
    "dep,arr,baseline,splits,outcome",
    [
        ((0.01, 0.001, 0.02, 0.001), -0.01, True, 1, "exploratory_signal"),
        ((0.01, 0, 0.02, 0.001), -0.01, True, 1, "inconclusive"),
        ((0.01, 0.001, 0.02, 0), -0.01, True, 1, "inconclusive"),
        ((0.01, 0.001, 0.02, 0.001), -0.01001, True, 1, "inconclusive"),
        ((0.005, 0, 0.009, 0.003), 0, True, 1, "below_practical_threshold"),
        ((0.005, 0, 0.01, 0.003), 0, True, 1, "inconclusive"),
        ((0.02, 0.01, 0.03, 0.01), 0, False, 1, "unverifiable"),
        ((0.02, 0.01, 0.03, 0.01), 0, True, 0, "unverifiable"),
    ],
)
def test_exact_decision_thresholds(dep, arr, baseline, splits, outcome):
    stats = pd.DataFrame(
        [dep, (0, arr, 0, 0)],
        index=pairing.TARGETS,
        columns=["mean", "lower", "upper", "leave_one_date_out_min"],
    )
    assert decision(stats, baseline, splits) == outcome


def test_evaluator_rejects_reordered_predictions(
    mock_cohort, synthetic_cache, tmp_path
):
    params, _ = mock_cohort
    paired = pairing.prepare_pairs(tmp_path, synthetic_cache.root, params)
    validation = paired.manifest.loc[paired.manifest["split"] == "validation"]
    scores = validation[pairing.TARGETS].astype(float) * 0.5 + 0.25
    result = evaluate_predictions(paired, scores, scores, departure_weather_splits=1)
    assert result["decision"] == "below_practical_threshold"
    assert result["ap_intervals"]["mean"].eq(0).all()
    assert len(result["daily"]) == 48
    with pytest.raises(ValueError, match="identity"):
        evaluate_predictions(
            paired, scores, scores.iloc[::-1], departure_weather_splits=1
        )


def test_truncated_ledger_fails_closed(tmp_path):
    with acq.campaign_ledger(tmp_path):
        pass
    (tmp_path / "stage9b_budget.sqlite").write_bytes(b"")
    with pytest.raises(acq.AcquisitionGateError, match="unreadable or truncated"):
        with acq.campaign_ledger(tmp_path):
            pytest.fail("Truncated ledger restarted")


def test_default_transport_has_one_attempt_and_does_not_follow_redirect(monkeypatch):
    response = MockResponse(302, b"")
    calls = []

    class Connection:
        def __init__(self, host, timeout):
            calls.append(("connect_ready", host, timeout))

        def request(self, method, path, headers):
            calls.append((method, path, headers))

        @contextmanager
        def getresponse(self):
            yield response

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    with acq.single_https_response(
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/example.idx",
        {},
    ) as got:
        assert got.status == 302
    assert calls == [
        ("connect_ready", "noaa-hrrr-bdp-pds.s3.amazonaws.com", 30),
        ("GET", "/example.idx", {}),
        ("close",),
    ]


def test_object_etag_change_after_first_range_does_not_commit(tmp_path):
    opener, _, calls = mock_archive(KEYS[0])

    @contextmanager
    def changed(url, headers):
        with opener(url, headers) as response:
            if len(calls) == 3:
                response.headers["ETag"] = '"changed"'
            yield response

    with pytest.raises(acq.AcquisitionGateError, match="changed GRIB object identity"):
        acquire(tmp_path, KEYS[0], changed)
    assert WeatherCache(tmp_path).read_hour(*KEYS[0][:2]) is None


def test_frozen_whitelist_tampering_and_budget_constants(tmp_path, monkeypatch):
    from flight_delay.weather import stage9b_protocol as protocol

    assert len(KEYS) == 24 and all(lead == 31 for _, lead, _ in KEYS)
    assert acq.MAX_ATTEMPTS == 240
    assert acq.HOUR_BYTES == 16 * 1024**2
    assert acq.TOTAL_BYTES == 384 * 1024**2
    (tmp_path / "stage9a_protocol.json").write_bytes(
        (protocol._ROOT / "stage9a_protocol.json").read_bytes()
    )
    (tmp_path / "stage9a_hours.csv").write_text("other,date")
    monkeypatch.setattr(protocol, "_ROOT", tmp_path)
    with pytest.raises(ValueError, match="whitelist integrity"):
        protocol.frozen_hours()


def test_constant_training_weather_fails_before_pairing(mock_cohort, tmp_path):
    params, _ = mock_cohort
    cache = WeatherCache(tmp_path / "constant")
    for key in KEYS:
        cache.write_hour(*key[:2], synthetic_hour(*key))
    with pytest.raises(ValueError, match="Fewer than two"):
        pairing.prepare_pairs(tmp_path, cache.root, params)


def test_missing_committed_manifest_does_not_trigger_second_download(tmp_path):
    key = KEYS[0]
    opener, _, _ = mock_archive(key)
    acquire(tmp_path, key, opener)
    entry = WeatherCache(tmp_path).entry_directory(*key[:2])
    (entry / "manifest.json").unlink()
    with pytest.raises(acq.AcquisitionGateError, match="Previously committed"):
        acquire(
            tmp_path, key, lambda *_: pytest.fail("committed hour was downloaded again")
        )
    assert len(ledger(tmp_path, "SELECT id FROM requests")) == 10
