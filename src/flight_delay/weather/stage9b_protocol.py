"""Integrity-checked Stage 9A experiment inputs."""

import csv
import json
from pathlib import Path

import pandas as pd

from flight_delay.weather.cache import sha256_file

_ROOT = Path(__file__).resolve().parent
_PROTOCOL_SHA256 = "7e1894edbc5da85c04ad041a29a9a11d0fa003b571d2e88e20e89d9c0bcddd45"


def frozen_protocol():
    path = _ROOT / "stage9a_protocol.json"
    if sha256_file(path) != _PROTOCOL_SHA256:
        raise ValueError("Stage 9A protocol integrity failure")
    return json.loads(path.read_bytes())


def frozen_hours():
    path = _ROOT / "stage9a_hours.csv"
    if sha256_file(path) != frozen_protocol()["hours_sha256"]:
        raise ValueError("Stage 9A forecast whitelist integrity failure")
    with path.open(newline="", encoding="utf-8") as handle:
        hours = list(csv.DictReader(handle))
    if len(hours) != 24:
        raise ValueError("Stage 9A requires exactly 24 hours")
    return hours


def hour_key(run, lead, valid):
    if not isinstance(lead, int) or isinstance(lead, bool):
        raise ValueError("Stage 9A lead must be an integer")
    run, valid = pd.Timestamp(run), pd.Timestamp(valid)
    if pd.isna(run) or pd.isna(valid) or run.tzinfo is None or valid.tzinfo is None:
        raise ValueError("Timezone-aware run and valid time are required")
    return run.tz_convert("UTC"), lead, valid.tz_convert("UTC")


def allowed_hours():
    return {
        hour_key(row["run_utc"], int(row["lead"]), row["valid_utc"]): row
        for row in frozen_hours()
    }
