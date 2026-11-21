"""Network-free historical forecast API planning CLI (no live switch)."""

import argparse
import json
from pathlib import Path

from flight_delay.weather.api_client import APILimits, HistoricalForecastClient
from flight_delay.weather.api_providers import (
    ForecastQuery,
    OpenMeteoSingleRunsAdapter,
    VisualCrossingRunAdapter,
)

ADAPTERS = {
    "open-meteo": OpenMeteoSingleRunsAdapter,
    "visual-crossing": VisualCrossingRunAdapter,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provider", choices=ADAPTERS, default="open-meteo")
    parser.add_argument("--max-requests", type=int, default=10)
    args = parser.parse_args(argv)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not 1 <= len(records) <= 100:
        parser.error("Input must contain 1–100 scheduled-flight queries")
    adapter = ADAPTERS[args.provider]()
    plans = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "Origin",
            "scheduled_departure_utc",
        }:
            parser.error("Each query needs only Origin and scheduled_departure_utc")
        plans.append(
            adapter.plan(
                ForecastQuery(
                    record["Origin"],
                    record["scheduled_departure_utc"],
                )
            )
        )
    client = HistoricalForecastClient(limits=APILimits(max_requests=args.max_requests))
    report = client.dry_run(plans)
    report["query_mapping"] = [
        {
            "input_position": i,
            "Origin": record["Origin"],
            "scheduled_departure_utc": record["scheduled_departure_utc"],
            "request_key": plan.request_key,
        }
        for i, (record, plan) in enumerate(zip(records, plans, strict=True))
    ]
    print(json.dumps(report, indent=2))
    return 0 if report["within_request_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
