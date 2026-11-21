import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flight_delay.data.batch_pipeline import iter_months
from flight_delay.data.download_bts import build_bts_filename
from flight_delay.data.storage import load_parquet
from flight_delay.data.validate import validate_bts_data

DATASET_VERSION = "completed_six_origin_v1"


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 checksum in chunks."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_monthly_manifest(
    year: int,
    month: int,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> dict:
    """Build provenance metadata for one processed BTS month."""

    period = f"{year}-{month:02d}"

    raw_path = Path(raw_dir) / build_bts_filename(year, month)
    parquet_path = (
        Path(processed_dir)
        / f"bts_{year}_{month:02d}_completed.parquet"
    )

    for path in (raw_path, parquet_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    df = load_parquet(parquet_path)

    # Reuse established data quality rules.
    audit = validate_bts_data(df)

    actual_periods = pd.to_datetime(
        df["FlightDate"]
    ).dt.to_period("M")

    if not actual_periods.eq(pd.Period(period)).all():
        raise ValueError(f"Unexpected flight dates: {period}")

    schema = [
        {"column": column, "dtype": str(df[column].dtype)}
        for column in df.columns
    ]

    schema_bytes = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    schema_sha256 = hashlib.sha256(
        schema_bytes
    ).hexdigest()

    return {
        "period": period,
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "filename": raw_path.name,
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "processed": {
            "filename": parquet_path.name,
            "size_bytes": parquet_path.stat().st_size,
            "sha256": sha256_file(parquet_path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "departure_delay_rate": float(
                df["DepDel15"].mean()
            ),
            "arrival_delay_rate": float(
                df["ArrDel15"].mean()
            ),
        },
        "schema": {
            "columns": schema,
            "sha256": schema_sha256,
        },
        "quality": {
            "incomplete_candidate_keys": audit[
                "incomplete_candidate_key_records"
            ],
            "duplicate_candidate_keys": audit[
                "duplicate_candidate_key_records"
            ],
        },
    }


def save_manifest(manifest: dict, directory: str | Path) -> Path:
    """Atomically write a monthly JSON manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / f"{manifest['period']}.json"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)

        json.dump(manifest, file, indent=2)
        file.write("\n")

    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Generate BTS monthly data manifests"
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--processed-dir", default="data/processed"
    )
    parser.add_argument(
        "--manifest-dir", default="data/interim/manifests"
    )

    args = parser.parse_args()

    schema_hashes = set()

    for year, month in iter_months(args.start, args.end):
        manifest = build_monthly_manifest(
            year,
            month,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
        )

        path = save_manifest(
            manifest,
            args.manifest_dir,
        )

        schema_hashes.add(manifest["schema"]["sha256"])

        print(
            f"{manifest['period']} | "
            f"rows={manifest['processed']['row_count']:,} | "
            f"manifest={path}"
        )

    if len(schema_hashes) != 1:
        raise RuntimeError(
            "Schema mismatch detected across processed months."
        )

    print("\nMonthly manifests generated successfully!")
    print("Cross-month schema fingerprints match.")


if __name__ == "__main__":
    main()
