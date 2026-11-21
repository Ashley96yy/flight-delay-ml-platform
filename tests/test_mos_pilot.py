"""The pilot scope is fixed and test-year label columns cannot be read."""

import importlib.util
from pathlib import Path

import pandas as pd


def load_pilot():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_iem_mos.py"
    spec = importlib.util.spec_from_file_location("mos_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_is_only_twelve_exact_run_requests():
    requests = load_pilot().pilot_requests()
    assert len(requests) == 12
    assert {r.runtime_utc.year for r in requests} == {2023, 2024, 2025}
    assert {r.runtime_utc.month for r in requests} == {1, 7}
    assert all(len(r.stations) == 6 for r in requests)
    assert {r.runtime_utc.hour for r in requests if r.model == "NBS"} == {1}


def test_default_dry_run_cannot_construct_network_client(tmp_path, monkeypatch):
    module = load_pilot()

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run constructed acquisition client")

    monkeypatch.setattr(module, "IEMMOSClient", forbidden)
    module.collect(tmp_path / "out", tmp_path / "cache")
    assert (tmp_path / "out/dry_run.json").exists()
    assert not (tmp_path / "cache").exists()


def test_schedule_sample_projects_no_labels_even_in_2025(tmp_path, monkeypatch):
    module = load_pilot()
    calls = []

    def read(path, *, columns):
        calls.append((path, columns))
        assert not {"DepDel15", "ArrDel15", "DepDelay", "ArrDelay"}.intersection(
            columns
        )
        year, month = path.name.split("_")[1:3]
        return pd.DataFrame(
            [
                {
                    "FlightDate": f"{year}-{month}-16",
                    "Reporting_Airline": "AA",
                    "Flight_Number_Reporting_Airline": 1,
                    "Origin": origin,
                    "Dest": "SEA",
                    "CRSDepTime": 900,
                }
                for origin in ("ATL", "DFW", "DEN", "ORD", "LAX", "SFO")
            ]
        )[columns]

    monkeypatch.setattr(module.pd, "read_parquet", read)
    sample, sources = module.schedule_sample(tmp_path)
    assert len(calls) == len(sources) == 6
    assert len(sample) == 36
    assert sample.departure_time_status.eq("valid").all()
    assert len([p for p, _ in calls if "2025" in p.name]) == 2
