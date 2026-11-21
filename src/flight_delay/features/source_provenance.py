"""Historical BTS source provenance and availability metadata."""

import json
import re
from pathlib import Path

import pandas as pd

REQUIRED_CALENDAR_COLUMNS = {
    "report_month",
    "source_filename",
    "assumed_available_at_utc",
    "availability_policy",
    "availability_evidence",
    "lag_days",
}


def _validate_sha256(value: str, field_name: str) -> None:
    """Validate the format of a SHA256 digest."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", value
    ):
        raise ValueError(f"Invalid SHA256: {field_name}")


def load_source_provenance(
    manifest_dir: Path,
    availability_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Combine monthly manifests with source availability metadata.

    SHA256 values identify current files; they do not establish
    historical publication time or historical version availability.
    """
    missing_columns = (
        REQUIRED_CALENDAR_COLUMNS
        - set(availability_calendar.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing calendar columns: {sorted(missing_columns)}"
        )

    if availability_calendar["report_month"].duplicated().any():
        raise ValueError("Duplicate report months in calendar")

    if availability_calendar.empty:
        raise ValueError("Availability calendar is empty")

    manifest_dir = Path(manifest_dir)
    records = []

    for row in availability_calendar.itertuples(index=False):
        month = row.report_month
        path = manifest_dir / f"{month}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing monthly manifest: {path}"
            )

        with path.open(encoding="utf-8") as f:
            manifest = json.load(f)

        required_sections = {
            "period",
            "dataset_version",
            "generated_at_utc",
            "source",
            "processed",
            "schema",
        }

        if not isinstance(manifest, dict) or not (
            required_sections <= manifest.keys()
        ):
            raise ValueError(
                f"Invalid manifest structure: {path}"
            )

        if manifest["period"] != month:
            raise ValueError(
                f"Manifest period mismatch: {path}"
            )

        source = manifest["source"]
        processed = manifest["processed"]
        schema = manifest["schema"]

        if source["filename"] != row.source_filename:
            raise ValueError(
                f"Source filename mismatch: {month}"
            )

        _validate_sha256(
            source["sha256"], "source.sha256"
        )
        _validate_sha256(
            processed["sha256"], "processed.sha256"
        )
        _validate_sha256(
            schema["sha256"], "schema.sha256"
        )

        if processed["row_count"] <= 0:
            raise ValueError(
                f"Invalid processed row count: {month}"
            )

        generated_at = pd.Timestamp(
            manifest["generated_at_utc"]
        )

        if generated_at.tzinfo is None:
            raise ValueError(
                f"Manifest timestamp must have timezone: {month}"
            )

        records.append({
            "report_month": month,
            "manifest_path": str(path),
            "dataset_version": manifest["dataset_version"],
            "manifest_generated_at_utc": (
                generated_at.tz_convert("UTC")
            ),
            "source_sha256": source["sha256"],
            "source_size_bytes": source["size_bytes"],
            "processed_filename": processed["filename"],
            "processed_sha256": processed["sha256"],
            "processed_size_bytes": processed["size_bytes"],
            "processed_row_count": processed["row_count"],
            "schema_sha256": schema["sha256"],
        })

    metadata = pd.DataFrame(records)

    provenance = availability_calendar.merge(
        metadata,
        on="report_month",
        how="left",
        validate="one_to_one",
    )

    # All selected monthly files should use the same schema.
    if provenance["schema_sha256"].nunique() != 1:
        raise ValueError(
            "Inconsistent schemas across source months"
        )

    if provenance["dataset_version"].nunique() != 1:
        raise ValueError(
            "Inconsistent processed dataset versions"
        )

    return (
        provenance
        .sort_values("report_month")
        .reset_index(drop=True)
    )
