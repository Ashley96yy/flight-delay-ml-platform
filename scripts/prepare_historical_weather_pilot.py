"""Review DST exceptions and prepare a bounded HRRR pilot.

Read-only for the BTS and weather cache datasets.
Does not download weather.

Run:
    python scripts/prepare_historical_weather_pilot.py
"""

from pathlib import Path

import pandas as pd

from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.flight_times import (
    convert_bts_departure_to_utc,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data/processed"

PLAN_ROOT = ROOT / "artifacts/weather_feasibility/historical_plan"

CACHE_ROOT = ROOT / "data/interim/weather_cache_v2"

YEARS = (2023, 2024, 2025)


def review_dst_exceptions():
    """Inspect November departure-time exceptions."""

    exceptions = []

    for year in YEARS:
        path = DATA_ROOT / f"bts_{year}_11_completed.parquet"

        flights = pd.read_parquet(
            path,
            columns=[
                "Origin",
                "FlightDate",
                "CRSDepTime",
            ],
        )

        converted = convert_bts_departure_to_utc(flights)

        unresolved = converted.flights.loc[
            converted.flights["departure_time_status"].ne("valid"),
            [
                "Origin",
                "FlightDate",
                "CRSDepTime",
                "departure_time_status",
            ],
        ].copy()

        unresolved.insert(0, "year", year)

        # Retain the row position in the source month
        # for later inspection if required.
        unresolved["monthly_row_position"] = unresolved.index

        exceptions.append(unresolved)

    result = pd.concat(
        exceptions,
        ignore_index=True,
    )

    print("\n=== DST Exception Review ===")

    if result.empty:
        print("No unresolved November timestamps")
    else:
        print(result.to_string(index=False))

    return result


def select_pilot_requests():
    """Select one real F30 forecast request per year."""

    path = PLAN_ROOT / "unique_hourly_requests.parquet"

    if not path.is_file():
        raise FileNotFoundError(f"Missing historical planning artifact: {path}")

    requests = pd.read_parquet(path)

    requests["forecast_run_at_utc"] = pd.to_datetime(
        requests["forecast_run_at_utc"],
        utc=True,
    )

    requests["forecast_hour"] = requests["forecast_hour"].astype(int)

    cache = WeatherCache(CACHE_ROOT)

    selections = []

    for year in YEARS:
        target = pd.Timestamp(f"{year}-07-14T06:00:00Z")

        # Restrict the pilot to one July 06Z F30
        # forecast per year. Never expand automatically
        # into a full month or full year.

        candidates = requests.loc[
            requests["forecast_run_at_utc"].dt.year.eq(year)
            & requests["forecast_run_at_utc"].dt.month.eq(7)
            & requests["forecast_run_at_utc"].dt.hour.eq(6)
            & requests["forecast_hour"].eq(30)
        ].copy()

        if candidates.empty:
            raise RuntimeError(f"No eligible July F30 pilot for {year}")

        distance = (candidates["forecast_run_at_utc"] - target).abs()

        selected = candidates.loc[distance.idxmin()]

        run_at = selected["forecast_run_at_utc"]
        lead = int(selected["forecast_hour"])

        cached = cache.read_hour(run_at, lead)

        selections.append(
            {
                "year": year,
                "forecast_run_at_utc": run_at,
                "forecast_hour": lead,
                "cache_status": ("hit" if cached is not None else "miss"),
                "backend": (cached["backend"].iloc[0] if cached is not None else None),
            }
        )

    result = pd.DataFrame(selections)

    if len(result) != 3:
        raise AssertionError("Pilot must contain exactly three requests")

    print("\n=== Historical HRRR Pilot Selection ===")
    print(result.to_string(index=False))

    print("\nCache hits:", int(result["cache_status"].eq("hit").sum()))

    print("Cache misses:", int(result["cache_status"].eq("miss").sum()))

    return result


def main():
    print("=== Historical Weather Pilot Preparation ===")
    print("Years:", YEARS)
    print("Weather downloads: NONE")

    exceptions = review_dst_exceptions()

    pilot = select_pilot_requests()

    PLAN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    exception_path = PLAN_ROOT / "dst_exceptions.csv"
    pilot_path = PLAN_ROOT / "pilot_requests.csv"

    exceptions.to_csv(
        exception_path,
        index=False,
    )

    pilot.to_csv(
        pilot_path,
        index=False,
    )

    print("\n=== Output Artifacts ===")
    print(exception_path)
    print(pilot_path)

    print("\nPASS: Historical pilot preparation complete")
    print("No weather downloaded.")
    print("No original BTS files modified.")


if __name__ == "__main__":
    main()
