"""Fixed 12-request MOS feasibility pilot; default is dry run, never training."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from flight_delay.weather.api_client import APILimits
from flight_delay.weather.flight_times import convert_bts_departure_to_utc
from flight_delay.weather.iem_mos import (
    IEMMOSClient,
    IEMMOSProvider,
    MOSRunRequest,
)
from flight_delay.weather.mos_contracts import RAW_FIELDS, parse_mos_response
from flight_delay.weather.mos_flight_join import audit_mos_for_flights

# Deliberately no flight outcome/label columns, including in the 2025 metadata check.
SCHEDULE_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]


def pilot_requests():
    return [
        MOSRunRequest(model, f"{year}-{date}T{hour}:00:00Z")
        for year in (2023, 2024, 2025)
        for date in ("01-15", "07-15")
        for model, hour in (("GFS", "00"), ("NBS", "01"))
    ]


def request_name(request):
    return request.model.lower() + "_" + request.runtime_utc.strftime("%Y%m%d_%H")


def collect(output, cache, *, allow_network=False):
    provider = IEMMOSProvider()
    requests = pilot_requests()
    output.mkdir(parents=True, exist_ok=True)
    report = provider.dry_run(requests, max_requests=14)
    report["station_run_model_cells"] = len(requests) * 6
    report["max_total_response_bytes"] = 8_000_000
    (output / "dry_run.json").write_text(json.dumps(report, indent=2) + "\n")
    if not allow_network:
        return
    client = IEMMOSClient(
        cache,
        allow_network=True,
        limits=APILimits(
            max_requests=14,
            max_response_bytes=2_000_000,
            timeout_seconds=20,
            min_interval_seconds=1,
        ),
    )
    (output / "raw").mkdir(exist_ok=True)
    receipts = []
    for request in requests:
        plan = provider.plan(request)
        try:
            raw, receipt = client.fetch(plan)
            (output / "raw" / (request_name(request) + ".json")).write_bytes(raw)
            receipts.append(
                dict(
                    receipt,
                    model=request.model,
                    runtime=request.runtime_utc.isoformat(),
                )
            )
        except Exception as exc:
            receipts.append(
                {
                    "model": request.model,
                    "runtime": request.runtime_utc.isoformat(),
                    "error_type": type(exc).__name__,
                }
            )
        (output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


def schedule_sample(flight_dir):
    """One label-blind flight per origin on each sampled next local day."""
    parts, sources = [], []
    for year in (2023, 2024, 2025):
        for month in (1, 7):
            path = flight_dir / f"bts_{year}_{month:02d}_completed.parquet"
            # Column projection ensures test labels are never read into memory.
            frame = pd.read_parquet(path, columns=SCHEDULE_COLUMNS)
            local_date = f"{year}-{month:02d}-16"
            selected = frame.loc[
                pd.to_datetime(frame["FlightDate"])
                .dt.strftime("%Y-%m-%d")
                .eq(local_date)
            ].copy()
            selected["sample_id"] = selected.apply(
                lambda row: hashlib.sha256(
                    ("42|" + "|".join(str(row[c]) for c in SCHEDULE_COLUMNS)).encode()
                ).hexdigest(),
                axis=1,
            )
            if selected["sample_id"].duplicated().any():
                raise ValueError("Duplicate complete flight identity in sample source")
            selected = (
                selected.sort_values("sample_id").groupby("Origin", sort=True).head(1)
            )
            converted = convert_bts_departure_to_utc(selected)
            parts.append(converted.flights)
            sources.append(
                {
                    "file": str(path),
                    "local_date": local_date,
                    "rows_selected": len(selected),
                    "columns_read": SCHEDULE_COLUMNS,
                    "selected_schedule_sha256": hashlib.sha256(
                        selected[SCHEDULE_COLUMNS].to_csv(index=False).encode()
                    ).hexdigest(),
                }
            )
    return pd.concat(parts, ignore_index=True), sources


def analyze(output, *, flight_dir=None):
    receipts = json.loads((output / "receipts.json").read_text())
    by_request = {r["request_key"]: r for r in receipts if "request_key" in r}
    coverage, missingness, frames, errors = [], [], [], []
    provider = IEMMOSProvider()
    for request in pilot_requests():
        plan = provider.plan(request)
        path = output / "raw" / (request_name(request) + ".json")
        if plan.request_key not in by_request or not path.exists():
            errors.append(
                {
                    "model": request.model,
                    "runtime": request.runtime_utc.isoformat(),
                    "error": "missing_receipt_or_response",
                }
            )
            for station in request.stations:
                coverage.append(
                    {
                        "model": request.model,
                        "runtime": request.runtime_utc.isoformat(),
                        "station": station,
                        "records": 0,
                        "run_present": False,
                    }
                )
            continue
        try:
            frame = parse_mos_response(
                path.read_bytes(), plan, by_request[plan.request_key]
            )
        except Exception as exc:
            errors.append(
                {
                    "model": request.model,
                    "runtime": request.runtime_utc.isoformat(),
                    "error": type(exc).__name__,
                }
            )
            continue
        frames.append(frame)
        for station in request.stations:
            group = frame.loc[frame.station.eq(station)]
            steps = (
                group.forecast_valid_at_utc.sort_values()
                .diff()
                .dropna()
                .dt.total_seconds()
                / 3600
            )
            coverage.append(
                {
                    "model": request.model,
                    "runtime": request.runtime_utc.isoformat(),
                    "station": station,
                    "records": len(group),
                    "run_present": not group.empty,
                    "min_lead_hours": group.forecast_hour.min(),
                    "max_lead_hours": group.forecast_hour.max(),
                    "observed_step_hours": ",".join(
                        str(int(v)) for v in sorted(steps.unique())
                    ),
                    "records_at_least_24h_lead": int(group.forecast_hour.ge(24).sum()),
                    "availability_verified_records": 0,
                }
            )
            for field in RAW_FIELDS:
                status = group["mos_status_" + field]
                missing = status.isin(["source_missing", "missing_code"])
                missingness.append(
                    {
                        "model": request.model,
                        "runtime": request.runtime_utc.isoformat(),
                        "station": station,
                        "field": field,
                        "rows": len(group),
                        "valid_values": int(status.eq("valid").sum()),
                        "null_values": int(status.eq("source_missing").sum()),
                        "missing_codes": int(status.eq("missing_code").sum()),
                        "unlimited_codes": int(status.eq("unlimited").sum()),
                        "missing_fraction": float(missing.mean())
                        if len(group)
                        else None,
                    }
                )
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(output / "coverage.csv", index=False)
    missingness = pd.DataFrame(missingness)
    missingness.to_csv(output / "field_checks.csv", index=False)
    if not missingness.empty:
        totals = missingness.groupby(["model", "field"], as_index=False)[
            ["rows", "valid_values", "null_values", "missing_codes", "unlimited_codes"]
        ].sum()
        totals["missing_fraction"] = (
            totals.null_values + totals.missing_codes
        ) / totals.rows
        totals.to_csv(output / "field_summary.csv", index=False)
    summary = {
        "requested_run_model_station_cells": 72,
        "present_cells": int(coverage.run_present.sum()) if not coverage.empty else 0,
        "parsed_records": sum(len(f) for f in frames),
        "availability_status": "availability_unverified",
        "model_ready": False,
        "verified_publication_records": 0,
        "errors": errors,
        "coverage_scope": "six fixed run dates; not annual or flight coverage",
    }
    if flight_dir is not None:
        sample, sources = schedule_sample(flight_dir)
        sample.to_csv(output / "flight_schedule_sample.csv", index=False)
        (output / "flight_schedule_sources.json").write_text(
            json.dumps(sources, indent=2) + "\n"
        )
        audits = []
        for frame in frames:
            run = frame.forecast_run_at_utc.iloc[0]
            # Fixed archive run paired with next LOCAL date schedules; no latest
            # run assumption, no targets or flight outcomes are involved.
            local_date = (run + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            group = sample.loc[
                pd.to_datetime(sample.FlightDate).dt.strftime("%Y-%m-%d").eq(local_date)
            ]
            group = group.loc[group.departure_time_status.eq("valid")]
            mapping = group[["sample_id", "Origin", "scheduled_departure_utc"]].copy()
            mapping["forecast_run_at_utc"] = run
            mapping["forecast_valid_at_utc"] = mapping.scheduled_departure_utc
            mapping["forecast_hour"] = (
                mapping.forecast_valid_at_utc - run
            ).dt.total_seconds() / 3600
            # Float representation avoids pandas' mixed int/float join warning;
            # identities remain exact and fractional leads cannot match the grid.
            weather = frame.copy()
            weather["forecast_hour"] = weather.forecast_hour.astype(float)
            result = audit_mos_for_flights(mapping, weather)
            audits.append(result)
        audit = pd.concat(audits, ignore_index=True)
        audit.to_csv(output / "flight_point_in_time_audit.csv", index=False)
        summary.update(
            schedule_sample_rows=len(sample),
            flight_model_checks=len(audit),
            exact_valid_time_matches=int(audit.weather_matched.sum()),
            ready_flight_model_checks=int(audit.ready.sum()),
            flight_audit_status_counts=audit.status.value_counts().to_dict(),
            flight_labels_read=False,
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/iem_mos_pilot"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/interim/iem_mos_pilot_cache")
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--flights-dir", type=Path)
    args = parser.parse_args()
    if args.analyze_only and args.allow_network:
        parser.error("Analysis-only mode cannot enable network")
    if not args.analyze_only:
        collect(args.output, args.cache, allow_network=args.allow_network)
    if args.analyze_only or args.allow_network:
        print(json.dumps(analyze(args.output, flight_dir=args.flights_dir), indent=2))
    else:
        print("Dry run saved. No network requests sent.")


if __name__ == "__main__":
    main()
