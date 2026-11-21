from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flight_delay.data.storage import load_parquet
from flight_delay.data.validate import ORIGIN_AIRPORTS
from flight_delay.features.build import build_baseline_features

# Fixed chronological evaluation policy.
SPLIT_RANGES = {
    "train": ("2023-01", "2023-12"),
    "validation": ("2024-01", "2024-08"),
    "calibration": ("2024-09", "2024-10"),
    "threshold": ("2024-11", "2024-12"),
    "test": ("2025-01", "2025-12"),
}

BASELINE_INPUT_COLUMNS = (
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
)

REQUIRED_DATASET_COLUMNS = set(BASELINE_INPUT_COLUMNS) | {
    "DepDel15",
    "ArrDel15",
    "Cancelled",
    "Diverted",
}


@dataclass
class TemporalDataset:
    """Aligned features, targets, and flight-date metadata."""

    X: pd.DataFrame
    y_departure: pd.Series
    y_arrival: pd.Series
    flight_dates: pd.Series


def months_for_split(name: str) -> list[tuple[int, int]]:
    """Return the months belonging to a chronological split."""

    if name not in SPLIT_RANGES:
        raise ValueError(f"Unknown dataset split: {name}")

    start, end = SPLIT_RANGES[name]

    return [
        (period.year, period.month)
        for period in pd.period_range(
            start=start,
            end=end,
            freq="M",
        )
    ]


def load_temporal_split(
    name: str,
    processed_dir: str | Path = "data/processed",
    *,
    allow_test: bool = False,
) -> TemporalDataset:
    """Load and transform a chronological dataset split."""

    # Check access before reading any files.
    months = months_for_split(name)

    if name == "test" and not allow_test:
        raise PermissionError(
            "The 2025 test split is locked until final evaluation."
        )

    processed_dir = Path(processed_dir)

    feature_parts = []
    departure_parts = []
    arrival_parts = []
    date_parts = []

    for year, month in months:
        period = f"{year}-{month:02d}"

        path = (
            processed_dir
            / f"bts_{year}_{month:02d}_completed.parquet"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing dataset for {period}: {path}"
            )

        df = load_parquet(path)

        # Validate required columns.
        missing = REQUIRED_DATASET_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {period}: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError(f"Empty monthly dataset: {period}")

        # Verify the month represented by each file.
        dates = pd.to_datetime(
            df["FlightDate"],
            errors="raise",
        )

        expected = pd.Period(period, freq="M")

        if not dates.dt.to_period("M").eq(expected).all():
            raise ValueError(
                f"Flight dates outside expected month: {period}"
            )

        # Check the established modeling cohort contract.
        eligible = (
            df["Origin"].isin(ORIGIN_AIRPORTS)
            & df["Cancelled"].eq(0)
            & df["Diverted"].eq(0)
            & df["DepDel15"].isin([0, 1])
            & df["ArrDel15"].isin([0, 1])
        )

        if not eligible.all():
            raise ValueError(
                f"Invalid modeling cohort in {period}"
            )

        # Feature engineering uses only approved inputs.
        X_month = build_baseline_features(
            df.loc[:, list(BASELINE_INPUT_COLUMNS)]
        )

        feature_parts.append(X_month)

        # Preserve row order so X and y remain aligned.
        departure_parts.append(df["DepDel15"])
        arrival_parts.append(df["ArrDel15"])
        date_parts.append(dates)

    return TemporalDataset(
        X=pd.concat(feature_parts, ignore_index=True),
        y_departure=pd.concat(
            departure_parts, ignore_index=True
        ).astype("int8"),
        y_arrival=pd.concat(
            arrival_parts, ignore_index=True
        ).astype("int8"),
        flight_dates=pd.concat(
            date_parts, ignore_index=True
        ),
    )
