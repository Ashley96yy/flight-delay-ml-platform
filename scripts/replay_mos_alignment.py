"""Offline comparison of A/B against the immutable Stage-2 tiny sample."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from flight_delay.weather.flight_join import _utc_datetime
from flight_delay.weather.iem_mos import IEMMOSProvider, MOSRunRequest
from flight_delay.weather.mos_alignment import (
    MODEL_FIELDS,
    align_mos_forecasts,
)
from flight_delay.weather.mos_contracts import parse_mos_response


def replay(source, output):
    """Read saved receipts/responses/schedules only; no transport or labels."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipts = {
        r["request_key"]: r for r in json.loads((source / "receipts.json").read_text())
    }
    sample = pd.read_csv(source / "flight_schedule_sample.csv")
    if {"DepDel15", "ArrDel15", "DepDelay", "ArrDelay"}.intersection(sample.columns):
        raise ValueError("Replay sample must not contain outcome labels")
    sample["scheduled_departure_utc"] = _utc_datetime(sample, "scheduled_departure_utc")
    requests, forecasts, hashes = [], [], {}
    for year in (2023, 2024, 2025):
        for date in ("01-15", "07-15"):
            for model, hour in (("GFS", "00"), ("NBS", "01")):
                query = MOSRunRequest(model, f"{year}-{date}T{hour}:00Z")
                plan = IEMMOSProvider().plan(query)
                name = (
                    model.lower()
                    + "_"
                    + query.runtime_utc.strftime("%Y%m%d_%H")
                    + ".json"
                )
                raw = (source / "raw" / name).read_bytes()
                hashes[name] = hashlib.sha256(raw).hexdigest()
                frame = parse_mos_response(raw, plan, receipts[plan.request_key])
                forecasts.append(frame)
                local_date = (query.runtime_utc + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                group = sample.loc[sample.FlightDate.eq(local_date)]
                mapping = group[
                    ["sample_id", "Origin", "scheduled_departure_utc"]
                ].copy()
                mapping["model"] = model
                mapping["forecast_run_at_utc"] = query.runtime_utc
                requests.append(mapping)
    requests = pd.concat(requests, ignore_index=True)
    forecasts = pd.concat(forecasts, ignore_index=True)
    if len(requests) != 72 or len(sample) != 36:
        raise ValueError("Expected the frozen Stage-2 36-flight / 72-check sample")
    comparison, field_rows, detail_parts = [], [], []
    for strategy in ("conservative", "interpolation"):
        # No publication evidence is invented or inherited from runtime.
        result = align_mos_forecasts(requests, forecasts, strategy=strategy)
        result.to_csv(output / (strategy + "_detail.csv"), index=False)
        detail_parts.append(result)
        for model, group in [("ALL", result), *result.groupby("model", sort=True)]:
            comparison.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "flight_model_checks": len(group),
                    "temporal_matches": int(group.temporal_match.sum()),
                    "temporal_match_rate": float(group.temporal_match.mean()),
                    "max_past_valid_gap_hours": float(
                        group.loc[group.temporal_match, "valid_time_gap_hours"].max()
                    ),
                    "max_right_valid_gap_hours": float(group.right_gap_hours.max())
                    if strategy == "interpolation"
                    and group.right_gap_hours.notna().any()
                    else None,
                    "max_bracket_span_hours": float(group.bracket_span_hours.max())
                    if strategy == "interpolation"
                    and group.bracket_span_hours.notna().any()
                    else None,
                    "potential_field_values": int(group.potential_field_count.sum()),
                    "potential_field_denominator": int(
                        group.potential_field_denominator.sum()
                    ),
                    "potential_field_coverage": float(
                        group.potential_field_count.sum()
                        / group.potential_field_denominator.sum()
                    ),
                    "materialized_field_values": int(
                        group.materialized_field_count.sum()
                    ),
                    "strict_pit_passes": int(group.strict_pit_pass.sum()),
                    "strict_pit_pass_rate": float(group.strict_pit_pass.mean()),
                    "planned_interpolations": int(group.planned_interpolation.sum()),
                    "executed_interpolations": int(group.interpolation_executed.sum()),
                    "availability_status": "availability_unverified",
                }
            )
        for model, group in result.groupby("model", sort=True):
            lists = group.potential_fields_json.map(json.loads)
            for field in MODEL_FIELDS[model]:
                count = int(lists.map(lambda fields: field in fields).sum())
                field_rows.append(
                    {
                        "strategy": strategy,
                        "model": model,
                        "field": field,
                        "candidate_available": count,
                        "checks": len(group),
                        "potential_coverage_rate": count / len(group),
                        "materialized_available": int(group[field].notna().sum()),
                    }
                )
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(field_rows).to_csv(output / "field_coverage.csv", index=False)
    detail = pd.concat(detail_parts, ignore_index=True)
    by_year = (
        detail.assign(year=detail.scheduled_departure_utc.dt.year)
        .groupby(["strategy", "year"], as_index=False)
        .agg(
            checks=("temporal_match", "size"),
            temporal_matches=("temporal_match", "sum"),
            strict_passes=("strict_pit_pass", "sum"),
        )
    )
    by_year.to_csv(output / "by_year.csv", index=False)
    summary = {
        "mode": "offline_replay",
        "new_mos_json_requests": 0,
        "raw_response_files": 12,
        "forecast_records": len(forecasts),
        "schedule_rows": len(sample),
        "flight_model_checks_per_strategy": len(requests),
        "max_past_gap_hours": 3.0,
        "max_bracket_span_hours": 6.0,
        "availability_status": "availability_unverified",
        "actual_interpolated_values_emitted": 0,
        "labels_read": False,
        "model_trained": False,
        "comparison": [r for r in comparison if r["model"] == "ALL"],
        "raw_response_sha256": hashes,
        "schedule_sha256": hashlib.sha256(
            (source / "flight_schedule_sample.csv").read_bytes()
        ).hexdigest(),
    }
    (output / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/iem_mos_pilot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/iem_mos_stage3"))
    args = parser.parse_args()
    summary = replay(args.source, args.output)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
