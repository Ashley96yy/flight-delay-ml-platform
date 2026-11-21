import argparse
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

from flight_delay.data.download_bts import download_bts_month
from flight_delay.data.pipeline import run_monthly_pipeline

logger = logging.getLogger(__name__)


def parse_year_month(value: str) -> date:
    """Parse a YYYY-MM value into the first day of the month."""

    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError(f"Invalid year-month format: {value}")

    year, month = map(int, value.split("-"))

    if year < 1987:
        raise ValueError("Year must be 1987 or later.")

    return date(year, month, 1)


def iter_months(start: str, end: str):
    """Generate inclusive (year, month) pairs."""

    current = parse_year_month(start)
    last = parse_year_month(end)

    if current > last:
        raise ValueError("Start month must not exceed end month.")

    while current <= last:
        yield current.year, current.month

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def run_batch_pipeline(
    start: str,
    end: str,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    summary_dir: str | Path = "data/interim",
    offline: bool = False,
    continue_on_error: bool = False,
) -> dict:
    """Run monthly BTS ETL over an inclusive date range."""

    months = list(iter_months(start, end))
    records = []

    for year, month in months:
        period = f"{year}-{month:02d}"

        logger.info("Starting %s", period)

        try:
            # Reuse valid raw files or download missing ones.
            if not offline:
                download_bts_month(
                    year=year,
                    month=month,
                    raw_dir=raw_dir,
                )

            # Run the existing validated monthly ETL.
            result = run_monthly_pipeline(
                year=year,
                month=month,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )

            records.append({
                "period": period,
                "status": "success",
                **result,
            })

            logger.info("Completed %s", period)

        except Exception as exc:
            logger.exception("Failed processing %s", period)

            records.append({
                "period": period,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

            if not continue_on_error:
                break

    # Build a structured execution report.
    succeeded = sum(
        item["status"] == "success" for item in records
    )
    failed = sum(
        item["status"] == "failed" for item in records
    )

    summary = {
        "start": start,
        "end": end,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_months": len(months),
        "attempted_months": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "results": records,
    }

    # Save the execution report locally.
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        summary_dir / f"batch_{start}_to_{end}_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    logger.info("Batch summary saved: %s", summary_path)

    if failed:
        raise RuntimeError(
            f"{failed} month(s) failed. "
            f"Check the report: {summary_path}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-month BTS ETL"
    )

    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--processed-dir", default="data/processed"
    )
    parser.add_argument(
        "--summary-dir", default="data/interim"
    )
    parser.add_argument(
        "--offline", action="store_true"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true"
    )
    parser.add_argument(
        "--dry-run", action="store_true"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    if args.dry_run:
        print("Months to process:")
        for year, month in iter_months(args.start, args.end):
            print(f"  {year}-{month:02d}")
        return

    summary = run_batch_pipeline(
        start=args.start,
        end=args.end,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        summary_dir=args.summary_dir,
        offline=args.offline,
        continue_on_error=args.continue_on_error,
    )

    print("\n=== Batch ETL Summary ===")
    print("Requested:", summary["requested_months"])
    print("Succeeded:", summary["succeeded"])
    print("Failed:", summary["failed"])


if __name__ == "__main__":
    main()
