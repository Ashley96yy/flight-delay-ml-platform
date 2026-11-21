
"""Tests for historical source provenance."""

import json

import pytest

from flight_delay.features.availability import (
    build_proxy_availability_calendar,
)
from flight_delay.features.source_provenance import (
    load_source_provenance,
)


@pytest.fixture
def sample_sources(tmp_path):
    calendar = build_proxy_availability_calendar(
        ["2023-01", "2023-02"],
        lag_days=120,
    )

    for row in calendar.itertuples(index=False):
        manifest = {
            "period": row.report_month,
            "dataset_version": "completed_six_origin_v1",
            "generated_at_utc": "2026-11-06T00:00:00Z",
            "source": {
                "filename": row.source_filename,
                "size_bytes": 1000,
                "sha256": "a" * 64,
            },
            "processed": {
                "filename": "sample.parquet",
                "size_bytes": 500,
                "sha256": "b" * 64,
                "row_count": 100,
            },
            "schema": {
                "sha256": "c" * 64,
            },
        }

        path = tmp_path / f"{row.report_month}.json"
        path.write_text(json.dumps(manifest))

    return tmp_path, calendar


def test_valid_provenance(sample_sources):
    manifest_dir, calendar = sample_sources

    result = load_source_provenance(
        manifest_dir,
        calendar,
    )

    assert len(result) == 2
    assert result["schema_sha256"].nunique() == 1
    assert result["processed_row_count"].tolist() == [
        100, 100
    ]


def test_missing_manifest_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    (manifest_dir / "2023-02.json").unlink()

    with pytest.raises(FileNotFoundError):
        load_source_provenance(manifest_dir, calendar)


def test_period_mismatch_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    path = manifest_dir / "2023-01.json"
    manifest = json.loads(path.read_text())
    manifest["period"] = "2023-03"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="period mismatch"):
        load_source_provenance(manifest_dir, calendar)


def test_source_filename_mismatch_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    path = manifest_dir / "2023-01.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["filename"] = "wrong.zip"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="filename mismatch"):
        load_source_provenance(manifest_dir, calendar)


def test_invalid_sha256_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    path = manifest_dir / "2023-01.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["sha256"] = "invalid"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Invalid SHA256"):
        load_source_provenance(manifest_dir, calendar)


def test_schema_mismatch_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    path = manifest_dir / "2023-02.json"
    manifest = json.loads(path.read_text())
    manifest["schema"]["sha256"] = "d" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Inconsistent schemas"):
        load_source_provenance(manifest_dir, calendar)


def test_dataset_version_mismatch_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    path = manifest_dir / "2023-02.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_version"] = "different_version"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="dataset versions"):
        load_source_provenance(manifest_dir, calendar)


def test_duplicate_calendar_month_rejected(sample_sources):
    manifest_dir, calendar = sample_sources

    duplicated = calendar.iloc[[0, 0, 1]].copy()

    with pytest.raises(ValueError, match="Duplicate"):
        load_source_provenance(manifest_dir, duplicated)
