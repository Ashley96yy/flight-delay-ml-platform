import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from flight_delay.data.preprocess import prepare_modeling_cohort
from flight_delay.data.storage import load_parquet, save_parquet
from flight_delay.data.validate import REQUIRED_COLUMNS, validate_bts_data

logger = logging.getLogger(__name__)


def run_monthly_pipeline(
    year: int,
    month: int,
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> dict:
    """Process one month of BTS data from local ZIP to Parquet."""

    # Step 1: Validate input parameters.
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    filename = (
        "On_Time_Reporting_Carrier_On_Time_Performance_"
        f"1987_present_{year}_{month}.zip"
    )

    zip_path = raw_dir / filename

    if not zip_path.is_file():
        raise FileNotFoundError(f"BTS ZIP not found: {zip_path}")

    logger.info("Processing BTS data: %s-%02d", year, month)

    # Step 2: Read selected columns from the source ZIP.
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [
            name for name in zf.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV in ZIP, found {len(csv_files)}"
            )

        with zf.open(csv_files[0]) as file:
            raw_df = pd.read_csv(
                file,
                usecols=lambda col: col in REQUIRED_COLUMNS,
                dtype={"Flight_Number_Reporting_Airline": "string"},
                low_memory=False,
            )

    logger.info("Loaded %s records", f"{len(raw_df):,}")

    # Step 3: Check the month represented by the source data.
    flight_dates = pd.to_datetime(
        raw_df["FlightDate"],
        errors="coerce",
    )

    valid_month = (
        flight_dates.notna()
        & flight_dates.dt.year.eq(year)
        & flight_dates.dt.month.eq(month)
    )

    if not valid_month.all():
        raise ValueError(
            "Source contains invalid dates or records "
            "outside the requested year/month."
        )

    # Step 4: Validate the source data.
    audit = validate_bts_data(raw_df)

    logger.info("Data validation passed")

    # Step 5: Prepare the modeling cohort.
    cleaned_df = prepare_modeling_cohort(raw_df)

    if len(cleaned_df) != audit["eligible_records"]:
        raise ValueError("Preprocessing row count mismatch.")

    logger.info(
        "Eligible flights: %s",
        f"{len(cleaned_df):,}",
    )

    # Step 6: Save Parquet.
    output_path = (
        processed_dir
        / f"bts_{year}_{month:02d}_completed.parquet"
    )

    save_parquet(cleaned_df, output_path)

    # Step 7: Verify persisted data.
    reloaded_df = load_parquet(output_path)

    assert_frame_equal(
        cleaned_df,
        reloaded_df,
        check_dtype=True,
    )

    logger.info("Parquet verification passed")

    # Step 8: Return a structured summary.
    return {
        "year": year,
        "month": month,
        "raw_records": audit["raw_records"],
        "six_airport_records": audit["six_airport_records"],
        "eligible_records": audit["eligible_records"],
        "output_path": str(output_path),
    }


def main():
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Run monthly BTS ETL pipeline"
    )

    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result = run_monthly_pipeline(
        year=args.year,
        month=args.month,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    print("\n=== ETL Summary ===")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
