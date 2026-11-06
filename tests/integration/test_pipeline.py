import zipfile
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flight_delay.data.pipeline import run_monthly_pipeline
from flight_delay.data.storage import load_parquet


@pytest.fixture
def sample_bts_df():
    """Four synthetic flights covering different outcomes."""
    return pd.DataFrame({
        "FlightDate": [
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
        "Reporting_Airline": ["AA", "UA", "DL", "AA"],
        "Flight_Number_Reporting_Airline": [
            "101", "202", "303", "404"
        ],
        "Origin": ["ATL", "SFO", "SEA", "DFW"],
        "Dest": ["JFK", "LAX", "ORD", "MIA"],
        "CRSDepTime": [900, 1200, 1530, 800],
        "CRSArrTime": [1200, 1500, 1800, 1100],
        "CRSElapsedTime": [180, 150, 120, 140],
        "Distance": [1000, 350, 1700, 1100],
        "DepDelay": [15, -3, 0, None],
        "DepDel15": [1, 0, 0, None],
        "ArrDelay": [12, 25, 0, None],
        "ArrDel15": [0, 1, 0, None],
        "Cancelled": [0, 0, 0, 1],
        "Diverted": [0, 0, 0, 0],
    })


def create_test_zip(df, raw_dir):
    """Create a small BTS-style ZIP file."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / (
        "On_Time_Reporting_Carrier_On_Time_Performance_"
        "1987_present_2023_1.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr("test_flights.csv", df.to_csv(index=False))

    return zip_path


def test_pipeline_success(sample_bts_df, tmp_path):
    """Valid ZIP should produce a verified Parquet file."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    create_test_zip(sample_bts_df, raw_dir)

    result = run_monthly_pipeline(
        year=2023,
        month=1,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert result["raw_records"] == 4
    assert result["six_airport_records"] == 3
    assert result["eligible_records"] == 2

    output_path = Path(result["output_path"])
    assert output_path.is_file()

    cleaned = load_parquet(output_path)

    assert len(cleaned) == 2
    assert set(cleaned["Origin"]) == {"ATL", "SFO"}
    assert cleaned["Flight_Number_Reporting_Airline"].dtype == "string"
    assert cleaned["DepDel15"].dtype == "int8"
    assert cleaned["ArrDel15"].dtype == "int8"


def test_pipeline_repeatable(sample_bts_df, tmp_path):
    """Repeated execution should not append records."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    create_test_zip(sample_bts_df, raw_dir)

    def run():
        return run_monthly_pipeline(
            2023, 1, raw_dir, processed_dir
        )

    first_result = run()
    first = load_parquet(first_result["output_path"])

    second_result = run()
    second = load_parquet(second_result["output_path"])

    assert_frame_equal(first, second)
    assert len(second) == 2
    assert len(list(processed_dir.glob("*.parquet"))) == 1


@pytest.mark.parametrize("month", [0, 13])
def test_invalid_month(tmp_path, month):
    """Months outside 1-12 must be rejected."""
    with pytest.raises(ValueError, match="Month"):
        run_monthly_pipeline(
            2023, month, tmp_path, tmp_path
        )


def test_missing_zip(tmp_path):
    """Missing source files must cause an explicit failure."""
    with pytest.raises(FileNotFoundError):
        run_monthly_pipeline(
            2023,
            1,
            tmp_path / "raw",
            tmp_path / "processed",
        )


def test_wrong_month(sample_bts_df, tmp_path):
    """Records from another month must be rejected."""
    df = sample_bts_df.copy()
    df.loc[0, "FlightDate"] = "2023-02-01"

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    create_test_zip(df, raw_dir)

    with pytest.raises(ValueError, match="outside"):
        run_monthly_pipeline(
            2023, 1, raw_dir, processed_dir
        )

    assert not list(processed_dir.glob("*.parquet"))


def test_invalid_label(sample_bts_df, tmp_path):
    """Invalid labels must stop the pipeline before writing."""
    df = sample_bts_df.copy()

    # A departure delay of 15 minutes requires label 1.
    df.loc[0, "DepDel15"] = 0

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    create_test_zip(df, raw_dir)

    with pytest.raises(ValueError, match="Label inconsistency"):
        run_monthly_pipeline(
            2023, 1, raw_dir, processed_dir
        )

    assert not list(processed_dir.glob("*.parquet"))
