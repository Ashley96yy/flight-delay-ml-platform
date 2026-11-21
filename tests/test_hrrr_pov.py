"""Stage 4 offline pilot safeguards."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hrrr_pov.py"
SPEC = importlib.util.spec_from_file_location("hrrr_pov", SCRIPT)
pov = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pov)


def _flights(date):
    return pd.DataFrame(
        {
            "FlightDate": pd.to_datetime([date, date]),
            "Reporting_Airline": ["DL", "DL"],
            "Origin": ["ATL", "ATL"],
            "Dest": ["SFO", "LAX"],
            "CRSDepTime": [800, 830],
            "CRSArrTime": [1020, 1105],
            "CRSElapsedTime": [320, 335],
            "Distance": [2139, 1946],
            "Flight_Number_Reporting_Airline": [12, 13],
            "DepDel15": [0, 1],
            "ArrDel15": [1, 0],
        }
    )


def test_cohort_ids_are_independent_of_labels_and_2025(tmp_path):
    for date in pov.DATES:
        path = tmp_path / f"bts_{date[:7].replace('-', '_')}_completed.parquet"
        _flights(date).to_parquet(path)
    (tmp_path / "bts_2025_01_completed.parquet").write_text("unreadable")
    first, hashes = pov.load_cohort(tmp_path)
    for date in pov.DATES:
        frame = _flights(date)
        frame["DepDel15"] = 1 - frame["DepDel15"]
        frame.to_parquet(
            tmp_path / f"bts_{date[:7].replace('-', '_')}_completed.parquet"
        )
    second, new_hashes = pov.load_cohort(tmp_path)
    assert set(first.year) == {2023, 2024}
    assert len(first) == 8
    assert first.flight_id.equals(second.flight_id)
    assert hashes != new_hashes


def test_minimum_additions_counts_required_dates_and_cached_hours():
    jan, jul = pov.DATES[:2]
    keys = [
        (pd.Timestamp("2023-01-14T06:00Z"), 31),
        (pd.Timestamp("2023-07-14T06:00Z"), 31),
        (pd.Timestamp("2023-07-14T06:00Z"), 32),
    ]
    rows = [
        (*key, date, airport)
        for key, date in zip(keys, [jan, jul, jul], strict=True)
        for airport in pov.ORIGIN_AIRPORTS
    ]
    mapping = pd.DataFrame(rows, columns=pov.HOUR_KEYS + pov.STRATA)
    assert pov.minimum_additions(mapping, {keys[0]}, 12, [jan, jul]) == [keys[1]]
    assert pov.minimum_additions(mapping, {keys[0]}, 18, [jan, jul]) == keys[1:]
    assert pov.minimum_additions(mapping.iloc[:6], set(), 6, [jan, jul]) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"forecast_hour": 31.0},
        {"kind": "assumed_availability"},
        {"covers_full_hour": False},
        {"available_by_utc": "2023-01-14T08:00:00"},
        {"available_by_utc": "2023-01-14T05:59:59Z"},
        {"cache_sha256": "different"},
    ],
)
def test_reviewed_evidence_rejects_unverified_or_wrong_payload(tmp_path, changes):
    cache = pov.WeatherCache(tmp_path / "cache")
    key = (pd.Timestamp("2023-01-14T06:00Z"), 31)
    directory = cache.entry_directory(*key)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"sha256": "bound"}))
    evidence_file = tmp_path / "evidence.json"
    evidence = {
        "forecast_run_at_utc": key[0].isoformat(),
        "forecast_hour": key[1],
        "available_by_utc": "2023-01-14T08:00:00Z",
        "kind": "historical_full_hour_receipt",
        "covers_full_hour": True,
        "evidence_reference": "reviewed test fixture",
        "cache_sha256": "bound",
    }
    evidence_file.write_text(json.dumps([evidence]))
    assert pov.approved_hours(cache, {key}, evidence_file) == {
        key: pd.Timestamp("2023-01-14T08:00Z")
    }
    evidence_file.write_text(json.dumps([evidence | changes]))
    with pytest.raises(ValueError):
        pov.approved_hours(cache, {key}, evidence_file)


def test_fit_uses_same_order_and_targets_for_both_arms(tmp_path, monkeypatch):
    import lightgbm

    operations = []

    class Classifier:
        def __init__(self, **params):
            self.task = params["label"]

        def fit(self, features, target):
            operations.append(
                (self.task, tuple(features.index), tuple(target), len(features.columns))
            )

        def predict_proba(self, features):
            operations.append((self.task, tuple(features.index)))
            return np.array([[0.25, 0.75], [0.75, 0.25]])

    monkeypatch.setattr(lightgbm, "LGBMClassifier", Classifier)
    flights = pd.concat(
        [
            _flights("2023-01-15").assign(year=2023),
            _flights("2024-01-15").assign(year=2024),
        ],
        ignore_index=True,
    ).assign(flight_id=list("abcd"))
    joined = pd.DataFrame(
        {"flight_row_position": [1, 3, 0, 2]}
        | {column: [1.0, 2.0, 3.0, 4.0] for column in pov.MODEL_WEATHER_COLUMNS}
    )
    pov.fit_comparison(
        flights, joined, {task: {"label": task} for task in pov.TARGETS}, tmp_path
    )
    for left, right in ((0, 4), (2, 6)):
        assert operations[left][:-1] == operations[right][:-1]
        assert operations[left][1] == (0, 2)
        assert operations[left + 1] == operations[right + 1]
        assert operations[left + 1][1] == (1, 3)
        assert operations[right][-1] - operations[left][-1] == len(
            pov.MODEL_WEATHER_COLUMNS
        )
    predictions = pd.read_csv(tmp_path / "paired_validation_predictions.csv")
    assert predictions.flight_id.tolist() == ["d", "c"]
    metrics = pd.read_csv(tmp_path / "exploratory_metrics.csv")
    assert len(metrics) == 4
    assert metrics["mode"].eq("exploratory").all()


@pytest.mark.parametrize("available, expected", [(None, 0), ("13:00", 2), ("13:01", 1)])
def test_runner_keeps_receipt_cutoff_separate_from_lag(tmp_path, available, expected):
    from argparse import Namespace

    from flight_delay.weather.config import AIRPORTS, WEATHER_VARIABLES

    for date in pov.DATES:
        _flights(date).to_parquet(
            tmp_path / f"bts_{date[:7].replace('-', '_')}_completed.parquet"
        )
    run_at = pd.Timestamp("2023-01-14T06:00Z")
    key = (run_at, 31)
    records = [
        {
            "Origin": airport.code,
            "feature": variable.feature,
            "forecast_run_at_utc": run_at,
            "forecast_valid_at_utc": run_at + pd.Timedelta(hours=31),
            "forecast_hour": 31,
            "assumed_available_at_utc": run_at + pd.Timedelta(hours=3),
            "value": 3.5 if variable.required else np.nan,
            "raw_value": 3.5 if variable.required else -9999.0,
            "value_status": "valid" if variable.required else "fill_value",
            "grid_latitude": airport.latitude,
            "grid_longitude": airport.longitude,
            "backend": "zarr",
            "storage_dtype": "float32",
            "units": variable.grib_units,
            "source_units": variable.grib_units,
            "fallback_reason": None,
        }
        for airport in AIRPORTS
        for variable in WEATHER_VARIABLES
    ]
    cache = pov.WeatherCache(tmp_path / "cache")
    cache.write_hour(*key, pd.DataFrame(records))
    manifest = json.loads((cache.entry_directory(*key) / "manifest.json").read_text())
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            [
                {
                    "forecast_run_at_utc": run_at.isoformat(),
                    "forecast_hour": 31,
                    "available_by_utc": f"2023-01-14T{available}:00Z",
                    "kind": "historical_full_hour_receipt",
                    "covers_full_hour": True,
                    "evidence_reference": "reviewed test fixture",
                    "cache_sha256": manifest["sha256"],
                }
            ]
            if available
            else []
        )
    )
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps({task: {"min_child_samples": 1} for task in pov.TARGETS})
    )
    output = tmp_path / "plan"
    summary = pov.run(
        Namespace(
            data_dir=tmp_path,
            cache=cache.root,
            params=params,
            output=output,
            reviewed_evidence=evidence,
            grib_cost_reference=None,
            fit=True,
            mode="exploratory",
        )
    )
    assert summary["exploratory"]["splits"]["2023"]["eligible"] == 2
    assert summary["strict_weather"]["splits"]["2023"]["eligible"] == expected
    assert not summary["training_executed"]
    assert not (output / "exploratory_metrics.csv").exists()
