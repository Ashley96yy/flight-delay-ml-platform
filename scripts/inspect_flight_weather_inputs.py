"""Inspect historical flight Parquet files before weather acquisition.

This is a read-only diagnostic script.
It does not download weather or modify existing datasets.

Run:
    python scripts/inspect_flight_weather_inputs.py
"""

import re
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

FILE_PATTERN = re.compile(r"bts_(20\d{2})_(0[1-9]|1[0-2])_completed\.parquet")

YEARS = (2023, 2024, 2025)

IMPORTANT_COLUMNS = (
    "Origin",
    "FlightDate",
    "FL_DATE",
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "CRSDepTime",
    "scheduled_departure_utc",
    "prediction_at_utc",
    "DepDel15",
    "ArrDel15",
    "Cancelled",
    "Diverted",
)


def discover_files():
    records = []

    if not DATA_ROOT.is_dir():
        print(f"Data directory does not exist: {DATA_ROOT}")
        return records

    for path in sorted(DATA_ROOT.rglob("*.parquet")):
        match = FILE_PATTERN.fullmatch(path.name)

        if match is None:
            continue

        year = int(match.group(1))
        month = int(match.group(2))

        if year not in YEARS:
            continue

        parquet = pq.ParquetFile(path)

        records.append(
            {
                "year": year,
                "month": month,
                "path": path,
                "rows": parquet.metadata.num_rows,
                "columns": parquet.schema_arrow.names,
            }
        )

    return records


def inspect_schema(record):
    path = record["path"]
    columns = record["columns"]

    print("\n=== Schema Inspection ===")
    print("File:", path.relative_to(PROJECT_ROOT))
    print("Rows:", record["rows"])
    print("Column count:", len(columns))

    print("\nRelevant columns:")

    column_lookup = {column.lower(): column for column in columns}

    for candidate in IMPORTANT_COLUMNS:
        match = column_lookup.get(candidate.lower())

        if match is not None:
            print(f"  FOUND   {match}")
        else:
            print(f"  MISSING {candidate}")

    print("\n=== Sample Records ===")

    available = [
        column_lookup[candidate.lower()]
        for candidate in IMPORTANT_COLUMNS
        if candidate.lower() in column_lookup
    ]

    available = list(dict.fromkeys(available))

    if not available:
        print("No recognizable flight columns found.")
        return

    parquet = pq.ParquetFile(path)

    batches = parquet.iter_batches(
        batch_size=5,
        columns=available,
    )

    first_batch = next(batches, None)

    if first_batch is None:
        print("Parquet file has no rows.")
        return

    sample = first_batch.to_pandas()

    print(sample.to_string(index=False))


def main():
    print("=== Historical BTS Data Inventory ===")
    print("Data root:", DATA_ROOT)

    records = discover_files()

    print("\nDiscovered monthly Parquet files:", len(records))

    by_key = {(record["year"], record["month"]): record for record in records}

    for year in YEARS:
        available = [month for month in range(1, 13) if (year, month) in by_key]

        missing = [month for month in range(1, 13) if (year, month) not in by_key]

        total_rows = sum(by_key[(year, month)]["rows"] for month in available)

        print(f"\n=== Year {year} ===")
        print("Available months:", available)
        print("Missing months:", missing)
        print("Monthly files:", len(available))
        print("Total rows:", f"{total_rows:,}")

    if not records:
        print("\nNo files matched bts_YYYY_MM_completed.parquet.")
        print("Other available Parquet paths:")

        if DATA_ROOT.is_dir():
            for path in sorted(DATA_ROOT.rglob("*.parquet"))[:20]:
                print(" ", path.relative_to(PROJECT_ROOT))

        return

    print("\n=== Available Monthly Files ===")

    for record in records:
        print(f"{record['year']}-{record['month']:02d}: {record['rows']:,} rows")

    # Inspect the earliest and latest available schemas.
    inspect_schema(records[0])

    if len(records) > 1:
        inspect_schema(records[-1])

    print("\n=== Inventory Complete ===")
    print("No weather downloads performed.")
    print("No existing datasets modified.")


if __name__ == "__main__":
    main()
