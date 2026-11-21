"""Cache-aware batch planning for historical weather acquisition."""

from dataclasses import dataclass

import pandas as pd

from flight_delay.weather.cache import WeatherCache
from flight_delay.weather.planner import build_forecast_request_plan


@dataclass
class WeatherBatchPlan:
    """Flight mappings and per-cycle acquisition workload."""

    flight_mapping: pd.DataFrame
    hourly_requests: pd.DataFrame
    cycle_tasks: pd.DataFrame
    hour_status: pd.DataFrame

    def summary(self) -> dict[str, int]:
        """Summarize requests after cache deduplication."""

        return {
            "flight_count": len(self.flight_mapping),
            "unique_hourly_requests": len(self.hourly_requests),
            "unique_cycles": len(self.cycle_tasks),
            "cached_hours": int(self.hour_status["cache_status"].eq("hit").sum()),
            "missing_hours": int(self.hour_status["cache_status"].eq("miss").sum()),
            "cycles_fully_cached": int(self.cycle_tasks["status"].eq("ready").sum()),
            "cycles_needing_acquisition": int(
                self.cycle_tasks["status"].eq("pending").sum()
            ),
        }


def build_weather_batch_plan(
    flights: pd.DataFrame,
    cache: WeatherCache,
    *,
    departure_col: str = "scheduled_departure_utc",
    origin_col: str = "Origin",
) -> WeatherBatchPlan:
    """Plan weather acquisition without any network requests."""

    request_plan = build_forecast_request_plan(
        flights,
        departure_col=departure_col,
        origin_col=origin_col,
    )

    cycle_records = []
    hourly_records = []

    for cycle in request_plan.cycle_requests.itertuples(index=False):
        run_at = cycle.forecast_run_at_utc
        requested_leads = tuple(int(lead) for lead in cycle.forecast_leads)

        cached_leads = []
        missing_leads = []

        for lead in requested_leads:
            # read_hour verifies manifest, checksum and
            # the complete 54-record weather contract.

            cached = cache.read_hour(run_at, lead)

            if cached is None:
                missing_leads.append(lead)
                cache_status = "miss"
                backend = None
                fallback_reason = None
            else:
                cached_leads.append(lead)
                cache_status = "hit"
                backend = cached["backend"].iloc[0]

                fallback_reason = cached["fallback_reason"].iloc[0]

                if pd.isna(fallback_reason):
                    fallback_reason = None

            hourly_records.append(
                {
                    "forecast_run_at_utc": run_at,
                    "forecast_hour": lead,
                    "cache_status": cache_status,
                    "backend": backend,
                    "fallback_reason": fallback_reason,
                }
            )

        cycle_records.append(
            {
                "forecast_run_at_utc": run_at,
                "forecast_leads": requested_leads,
                "cached_leads": tuple(cached_leads),
                "missing_leads": tuple(missing_leads),
                "flight_count": int(cycle.flight_count),
                "hourly_request_count": len(requested_leads),
                "status": ("ready" if not missing_leads else "pending"),
            }
        )

    cycle_tasks = pd.DataFrame(cycle_records)

    hour_status = pd.DataFrame(hourly_records)

    # Every unique hourly request must appear exactly once.

    if len(hour_status) != len(request_plan.hourly_requests):
        raise ValueError("Batch planning lost or duplicated hourly requests")

    return WeatherBatchPlan(
        flight_mapping=request_plan.flight_mapping,
        hourly_requests=request_plan.hourly_requests,
        cycle_tasks=cycle_tasks,
        hour_status=hour_status,
    )


def pending_cycle_tasks(
    plan: WeatherBatchPlan,
) -> pd.DataFrame:
    """Return cycles requiring acquisition."""

    return (
        plan.cycle_tasks.loc[plan.cycle_tasks["status"].eq("pending")]
        .copy()
        .reset_index(drop=True)
    )
