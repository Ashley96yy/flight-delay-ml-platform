import pandas as pd
import pytest

from flight_delay.data import dataset
from flight_delay.data.storage import save_parquet


def create_month_file(directory, year=2023, month=1, dates=None):
    """Create one small processed BTS-style Parquet file."""

    if dates is None:
        dates = [
            f"{year}-{month:02d}-01",
            f"{year}-{month:02d}-02",
        ]

    df = pd.DataFrame({
        "FlightDate": pd.to_datetime(dates),
        "Reporting_Airline": ["UA", "AA"],
        "Origin": ["SFO", "DFW"],
        "Dest": ["LAX", "ORD"],
        "CRSDepTime": [930, 2330],
        "CRSArrTime": [1100, 2400],
        "CRSElapsedTime": [90, 120],
        "Distance": [350, 800],
        "DepDel15": [1, 0],
        "ArrDel15": [0, 1],
        "Cancelled": [0, 0],
        "Diverted": [0, 0],
    })

    path = (
        directory
        / f"bts_{year}_{month:02d}_completed.parquet"
    )

    save_parquet(df, path)

    return path


def test_split_boundaries():
    """All split ranges must match the evaluation plan."""

    assert dataset.months_for_split("train")[0] == (2023, 1)
    assert dataset.months_for_split("train")[-1] == (2023, 12)

    assert dataset.months_for_split("validation")[0] == (
        2024, 1
    )
    assert dataset.months_for_split("validation")[-1] == (
        2024, 8
    )

    assert dataset.months_for_split("calibration") == [
        (2024, 9),
        (2024, 10),
    ]

    assert dataset.months_for_split("threshold") == [
        (2024, 11),
        (2024, 12),
    ]

    assert dataset.months_for_split("test")[0] == (2025, 1)


def test_split_months_do_not_overlap():
    """Chronological splits must be disjoint."""

    names = list(dataset.SPLIT_RANGES)
    all_months = []

    for name in names:
        all_months.extend(dataset.months_for_split(name))

    assert len(all_months) == 36
    assert len(set(all_months)) == 36


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        dataset.months_for_split("random")


def test_test_set_locked(tmp_path):
    """Test data must not be accessible by default."""

    with pytest.raises(PermissionError, match="locked"):
        dataset.load_temporal_split(
            "test",
            processed_dir=tmp_path,
        )


def test_dataset_loading_and_alignment(tmp_path, monkeypatch):
    """Features, targets, and metadata must stay aligned."""

    monkeypatch.setitem(
        dataset.SPLIT_RANGES,
        "train",
        ("2023-01", "2023-01"),
    )

    create_month_file(tmp_path)

    result = dataset.load_temporal_split(
        "train",
        processed_dir=tmp_path,
    )

    assert result.X.shape == (2, 17)
    assert result.y_departure.tolist() == [1, 0]
    assert result.y_arrival.tolist() == [0, 1]

    assert result.X["Origin"].tolist() == ["SFO", "DFW"]
    assert result.flight_dates.dt.year.eq(2023).all()

    assert "DepDel15" not in result.X.columns
    assert "ArrDel15" not in result.X.columns


def test_missing_monthly_file(tmp_path, monkeypatch):
    monkeypatch.setitem(
        dataset.SPLIT_RANGES,
        "train",
        ("2023-01", "2023-01"),
    )

    with pytest.raises(FileNotFoundError):
        dataset.load_temporal_split(
            "train",
            processed_dir=tmp_path,
        )


def test_wrong_month_rejected(tmp_path, monkeypatch):
    monkeypatch.setitem(
        dataset.SPLIT_RANGES,
        "train",
        ("2023-01", "2023-01"),
    )

    create_month_file(
        tmp_path,
        dates=["2023-02-01", "2023-02-02"],
    )

    with pytest.raises(ValueError, match="outside expected"):
        dataset.load_temporal_split(
            "train",
            processed_dir=tmp_path,
        )


def test_invalid_modeling_cohort(tmp_path, monkeypatch):
    monkeypatch.setitem(
        dataset.SPLIT_RANGES,
        "train",
        ("2023-01", "2023-01"),
    )

    path = create_month_file(tmp_path)
    df = pd.read_parquet(path)

    df.loc[0, "Cancelled"] = 1
    save_parquet(df, path)

    with pytest.raises(ValueError, match="Invalid modeling cohort"):
        dataset.load_temporal_split(
            "train",
            processed_dir=tmp_path,
        )
