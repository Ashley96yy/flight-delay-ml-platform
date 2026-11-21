"""Plan historical HRRR weather demand for 2023-2025.

This script:
- Reads only three BTS columns per monthly Parquet.
- Converts scheduled local departure times to UTC.
- Reports unresolved local timestamps without guessing.
- Reuses the tested forecast request planner.
- Deduplicates weather requests across month/year boundaries.
- Checks existing production cache without network access.
- Writes planning reports, never weather data.

Run:
    python scripts/plan_historical_weather.py
"""

import json
from pathlib import Path

import pandas as pd

from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.flight_times import (
    convert_bts_departure_to_utc,
)
from flight_delay.weather.planner import (
    build_forecast_request_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data/processed"

OUTPUT_ROOT = PROJECT_ROOT / "artifacts/weather_feasibility/historical_plan"

CACHE_ROOT = PROJECT_ROOT / "data/interim/weather_cache_v2"

YEARS = (2023, 2024, 2025)

READ_COLUMNS = [
    "Origin",
    "FlightDate",
    "CRSDepTime",
]

REQUEST_KEYS = [
    "forecast_run_at_utc",
    "forecast_hour",
]

TIME_STATUSES = [
    "valid",
    "invalid_time",
    "invalid_date",
    "unsupported_airport",
    "midnight_2400_review",
    "dst_unresolvable",
]

# Observed in the previous real GRIB F30 integration test.
# Used only as an indicative all-GRIB transfer scenario.
REFERENCE_GRIB_MIB_PER_HOUR = 11.594222068786621


def monthly_paths():
    """Require all expected monthly datasets to be present."""

    paths = []

    missing = []

    for year in YEARS:
        for month in range(1, 13):
            path = DATA_ROOT / f"bts_{year}_{month:02d}_completed.parquet"

            if not path.is_file():
                missing.append(str(path))
            else:
                paths.append((year, month, path))

    if missing:
        raise FileNotFoundError("Missing monthly BTS datasets:\n" + "\n".join(missing))

    return paths


def process_month(year, month, path):
    """Process one month, retaining only compact request metadata."""

    flights = pd.read_parquet(
        path,
        columns=READ_COLUMNS,
    )

    converted = convert_bts_departure_to_utc(flights)

    status = converted.flights["departure_time_status"]

    counts = status.value_counts().to_dict()

    report = {
        "year": year,
        "month": month,
        "total_flights": len(flights),
    }

    for name in TIME_STATUSES:
        report[name] = int(counts.get(name, 0))

    report["unresolved"] = report["total_flights"] - report["valid"]

    report["valid_rate"] = (
        report["valid"] / report["total_flights"] if report["total_flights"] else 0.0
    )

    valid_flights = converted.flights.loc[
        status.eq("valid"),
        ["Origin", "scheduled_departure_utc"],
    ].copy()

    # A month can theoretically contain no convertible flights.
    # Do not call the planner with an empty frame.

    if valid_flights.empty:
        report["monthly_unique_hours"] = 0
        report["monthly_unique_cycles"] = 0

        return report, pd.DataFrame(columns=["year", "month"] + REQUEST_KEYS)

    plan = build_forecast_request_plan(valid_flights)

    requests = plan.hourly_requests[REQUEST_KEYS].copy()

    if requests.duplicated(REQUEST_KEYS).any():
        raise ValueError(f"Duplicate planner requests: {year}-{month:02d}")

    requests["year"] = year
    requests["month"] = month

    report["monthly_unique_hours"] = len(requests)

    report["monthly_unique_cycles"] = requests["forecast_run_at_utc"].nunique()

    if len(plan.flight_mapping) != len(valid_flights):
        raise ValueError(f"Planner lost flight rows: {year}-{month:02d}")

    return report, requests


def build_cycle_tasks(hour_status):
    """Aggregate unique hours by HRRR initialization cycle."""

    records = []

    for run_at, group in hour_status.groupby(
        "forecast_run_at_utc",
        sort=True,
    ):
        leads = sorted(int(x) for x in group["forecast_hour"])

        missing = sorted(
            int(x)
            for x in group.loc[
                group["cache_status"].eq("miss"),
                "forecast_hour",
            ]
        )

        records.append(
            {
                "forecast_run_at_utc": run_at,
                "forecast_leads": ",".join(map(str, leads)),
                "missing_leads": ",".join(map(str, missing)),
                "hour_count": len(leads),
                "cached_hours": len(leads) - len(missing),
                "missing_hours": len(missing),
                "status": ("ready" if not missing else "pending"),
            }
        )

    return pd.DataFrame(records)


def main():
    print("=== Historical Weather Demand Planning ===")
    print("Years:", YEARS)
    print("BTS data root:", DATA_ROOT)
    print("Network access: DISABLED (cache-only inspection)")

    month_reports = []
    request_parts = []

    # Step 1: Process 36 months independently.

    for year, month, path in monthly_paths():
        report, requests = process_month(
            year,
            month,
            path,
        )

        month_reports.append(report)
        request_parts.append(requests)

        print(
            f"{year}-{month:02d} | "
            f"flights={report['total_flights']:,} | "
            f"valid={report['valid']:,} | "
            f"unresolved={report['unresolved']:,} | "
            f"unique_hours={report['monthly_unique_hours']:,}"
        )

    monthly = pd.DataFrame(month_reports)

    # Step 2: Deduplicate across all month boundaries.

    all_requests = pd.concat(
        request_parts,
        ignore_index=True,
    )

    all_requests["forecast_run_at_utc"] = pd.to_datetime(
        all_requests["forecast_run_at_utc"],
        utc=True,
    )

    all_requests["forecast_hour"] = all_requests["forecast_hour"].astype(int)

    unique_requests = (
        all_requests[REQUEST_KEYS]
        .drop_duplicates()
        .sort_values(REQUEST_KEYS)
        .reset_index(drop=True)
    )

    if unique_requests.duplicated(REQUEST_KEYS).any():
        raise ValueError("Request deduplication failed")

    # Step 3: Check the production cache.
    # read_hour performs manifest, checksum and contract checks.
    # Corrupt cache entries cause an error rather than a false hit.

    cache = WeatherCache(CACHE_ROOT)

    cache_records = []

    for request in unique_requests.itertuples(index=False):
        run_at = request.forecast_run_at_utc
        lead = int(request.forecast_hour)

        cached = cache.read_hour(run_at, lead)

        cache_records.append(
            {
                "forecast_run_at_utc": run_at,
                "forecast_hour": lead,
                "cache_status": ("hit" if cached is not None else "miss"),
                "backend": (cached["backend"].iloc[0] if cached is not None else None),
            }
        )

    hour_status = pd.DataFrame(cache_records)

    cycle_tasks = build_cycle_tasks(hour_status)

    # Step 4: Summarize year-level request demand.
    # Deduplication is performed separately within each dataset year.

    yearly_reports = []

    for year in YEARS:
        year_months = monthly.loc[monthly["year"].eq(year)]

        year_requests = all_requests.loc[
            all_requests["year"].eq(year),
            REQUEST_KEYS,
        ].drop_duplicates()

        yearly_reports.append(
            {
                "year": year,
                "total_flights": int(year_months["total_flights"].sum()),
                "valid_flights": int(year_months["valid"].sum()),
                "unresolved_flights": int(year_months["unresolved"].sum()),
                "unique_hours": len(year_requests),
                "unique_cycles": (year_requests["forecast_run_at_utc"].nunique()),
            }
        )

    yearly = pd.DataFrame(yearly_reports)

    # Step 5: Overall statistics.

    total_hours = len(hour_status)

    cached_hours = int(hour_status["cache_status"].eq("hit").sum())

    missing_hours = total_hours - cached_hours

    total_cycles = len(cycle_tasks)

    pending_cycles = int(cycle_tasks["status"].eq("pending").sum())

    summary = {
        "total_flights": int(monthly["total_flights"].sum()),
        "valid_flights": int(monthly["valid"].sum()),
        "unresolved_flights": int(monthly["unresolved"].sum()),
        "unique_hourly_requests": total_hours,
        "unique_cycles": total_cycles,
        "cached_hours": cached_hours,
        "missing_hours": missing_hours,
        "cached_cycles": total_cycles - pending_cycles,
        "pending_cycles": pending_cycles,
        "logical_weather_records": total_hours * 54,
        "estimated_all_grib_transfer_gib": round(
            missing_hours * REFERENCE_GRIB_MIB_PER_HOUR / 1024,
            2,
        ),
    }

    # Step 6: Save compact planning reports only.

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_csv(
        OUTPUT_ROOT / "monthly_time_quality.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_ROOT / "yearly_summary.csv",
        index=False,
    )

    hour_status.to_parquet(
        OUTPUT_ROOT / "unique_hourly_requests.parquet",
        index=False,
    )

    cycle_tasks.to_csv(
        OUTPUT_ROOT / "cycle_tasks.csv",
        index=False,
    )

    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Step 7: Display results.

    print("\n=== Yearly Summary ===")
    print(
        yearly.to_string(
            index=False,
        )
    )

    print("\n=== Unresolved Local Times ===")

    for status_name in TIME_STATUSES:
        if status_name == "valid":
            continue

        count = int(monthly[status_name].sum())

        print(f"{status_name}: {count:,}")

    print("\n=== Overall Weather Demand ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Cache Backend Distribution ===")

    print(
        hour_status.loc[
            hour_status["cache_status"].eq("hit"),
            "backend",
        ]
        .value_counts()
        .to_string()
    )

    print("\nReports saved to:", OUTPUT_ROOT)

    print("\nPASS: Historical weather planning completed")
    print("No weather was downloaded.")
    print("No original BTS files were modified.")


if __name__ == "__main__":
    main()
