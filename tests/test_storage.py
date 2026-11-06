import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flight_delay.data.storage import load_parquet, save_parquet


@pytest.fixture
def sample_df():
    """Create a small sample for Parquet storage tests."""
    return pd.DataFrame({
        "FlightDate": pd.to_datetime([
            "2023-01-01",
            "2023-01-02",
        ]),
        "Origin": ["SFO", "LAX"],
        "Dest": ["JFK", "ORD"],
        "DepDel15": pd.Series([1, 0], dtype="int8"),
        "ArrDel15": pd.Series([0, 1], dtype="int8"),
    })


def test_parquet_round_trip(sample_df, tmp_path):
    """Saved and loaded data should be identical."""
    path = tmp_path / "flights.parquet"

    save_parquet(sample_df, path)
    loaded = load_parquet(path)

    assert_frame_equal(sample_df, loaded, check_dtype=True)


def test_creates_parent_directory(sample_df, tmp_path):
    """Storage should create missing output directories."""
    path = tmp_path / "nested" / "flights.parquet"

    saved_path = save_parquet(sample_df, path)

    assert saved_path == path
    assert path.is_file()


def test_overwrite_without_duplicates(sample_df, tmp_path):
    """Repeated saves should replace rather than append."""
    path = tmp_path / "flights.parquet"

    save_parquet(sample_df, path)
    save_parquet(sample_df, path)

    loaded = load_parquet(path)

    assert len(loaded) == len(sample_df)
    assert_frame_equal(sample_df, loaded)


def test_invalid_extension(sample_df, tmp_path):
    """Non-Parquet output paths should be rejected."""
    path = tmp_path / "flights.csv"

    with pytest.raises(ValueError, match="parquet"):
        save_parquet(sample_df, path)


def test_missing_file(tmp_path):
    """Loading a nonexistent file should raise an error."""
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        load_parquet(path)
