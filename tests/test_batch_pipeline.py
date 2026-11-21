import json
import sys

import pytest

from flight_delay.data import batch_pipeline as batch


def fake_monthly_pipeline(year, month, raw_dir, processed_dir):
    """Return a synthetic monthly ETL result."""
    return {
        "year": year,
        "month": month,
        "raw_records": 100,
        "six_airport_records": 30,
        "eligible_records": 25,
        "output_path": f"bts_{year}_{month:02d}.parquet",
    }


def test_iter_months_cross_year():
    """Month iteration should handle year boundaries."""
    months = list(batch.iter_months("2023-12", "2024-02"))

    assert months == [
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


@pytest.mark.parametrize(
    "start,end",
    [
        ("2023-00", "2023-02"),
        ("2023-13", "2024-01"),
        ("2023-03", "2023-02"),
    ],
)
def test_invalid_date_ranges(start, end):
    """Invalid ranges must be rejected."""
    with pytest.raises(ValueError):
        list(batch.iter_months(start, end))


def test_successful_batch(tmp_path, monkeypatch):
    """Two successful months should produce a JSON summary."""
    downloaded = []

    def fake_download(year, month, raw_dir):
        downloaded.append((year, month))

    monkeypatch.setattr(batch, "download_bts_month", fake_download)
    monkeypatch.setattr(
        batch, "run_monthly_pipeline", fake_monthly_pipeline
    )

    summary = batch.run_batch_pipeline(
        start="2023-01",
        end="2023-02",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        summary_dir=tmp_path / "reports",
    )

    assert downloaded == [(2023, 1), (2023, 2)]
    assert summary["requested_months"] == 2
    assert summary["attempted_months"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0

    report_path = (
        tmp_path / "reports"
        / "batch_2023-01_to_2023-02_summary.json"
    )

    assert report_path.is_file()

    saved = json.loads(report_path.read_text())
    assert [r["status"] for r in saved["results"]] == [
        "success", "success"
    ]


def test_offline_skips_download(tmp_path, monkeypatch):
    """Offline mode must never call the downloader."""

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Downloader must not be called")

    monkeypatch.setattr(
        batch, "download_bts_month", unexpected_download
    )
    monkeypatch.setattr(
        batch, "run_monthly_pipeline", fake_monthly_pipeline
    )

    summary = batch.run_batch_pipeline(
        start="2023-01",
        end="2023-01",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        summary_dir=tmp_path / "reports",
        offline=True,
    )

    assert summary["succeeded"] == 1
    assert summary["failed"] == 0


@pytest.mark.parametrize(
    "continue_on_error,expected_attempts",
    [
        (False, 2),
        (True, 3),
    ],
)
def test_failure_reporting(
    tmp_path, monkeypatch, continue_on_error, expected_attempts
):
    """Failed months must be recorded and surfaced."""

    def sometimes_fails(year, month, raw_dir, processed_dir):
        if month == 2:
            raise ValueError("Synthetic February failure")
        return fake_monthly_pipeline(
            year, month, raw_dir, processed_dir
        )

    monkeypatch.setattr(
        batch, "download_bts_month",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        batch, "run_monthly_pipeline", sometimes_fails
    )

    with pytest.raises(RuntimeError, match="1 month"):
        batch.run_batch_pipeline(
            start="2023-01",
            end="2023-03",
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            summary_dir=tmp_path / "reports",
            continue_on_error=continue_on_error,
        )

    report_path = (
        tmp_path / "reports"
        / "batch_2023-01_to_2023-03_summary.json"
    )

    report = json.loads(report_path.read_text())

    assert report["attempted_months"] == expected_attempts
    assert report["failed"] == 1
    assert report["results"][1]["period"] == "2023-02"
    assert report["results"][1]["status"] == "failed"
    assert report["results"][1]["error_type"] == "ValueError"


def test_dry_run_does_not_execute(
    monkeypatch, capsys
):
    """Dry-run should only print planned months."""

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("Batch ETL must not execute")

    monkeypatch.setattr(
        batch, "run_batch_pipeline", unexpected_execution
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_pipeline",
            "--start", "2023-01",
            "--end", "2023-02",
            "--dry-run",
        ],
    )

    batch.main()

    output = capsys.readouterr().out

    assert "2023-01" in output
    assert "2023-02" in output
