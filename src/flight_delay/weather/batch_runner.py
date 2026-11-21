"""Bounded parallel execution of cached weather cycle tasks."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import local
from time import perf_counter
from uuid import uuid4

import pandas as pd

from flight_delay.weather.batch_plan import WeatherBatchPlan
from flight_delay.weather.cache import WeatherCache, validate_cache_identity
from flight_delay.weather.cached_acquisition import acquire_cached_weather_cycle
from flight_delay.weather.fallback import acquire_weather_cycle


class WeatherBatchError(ValueError):
    """Invalid batch execution configuration or request plan."""


@dataclass
class WeatherBatchResult:
    task_report: pd.DataFrame
    cache_audit: pd.DataFrame
    attempt_audit: pd.DataFrame
    summary: dict


def _validate_tasks(plan):
    """Validate and deduplicate the cycle task identities."""

    tasks = []

    seen = set()

    for row in plan.cycle_tasks.itertuples(index=False):
        leads = tuple(int(lead) for lead in row.forecast_leads)

        if not leads or len(leads) != len(set(leads)):
            raise WeatherBatchError("Invalid or duplicate forecast leads")

        normalized = [
            validate_cache_identity(row.forecast_run_at_utc, lead) for lead in leads
        ]

        run_at = normalized[0][0]

        if run_at in seen:
            raise WeatherBatchError("Duplicate forecast cycle task")

        seen.add(run_at)

        tasks.append((run_at, tuple(sorted(leads))))

    return tasks


def _atomic_write(path, writer):
    """Publish a report file using atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_batch_report(result, report_dir):
    """Save task, cache, attempt and summary reports."""

    root = Path(report_dir)

    for filename, frame in (
        ("task_report.csv", result.task_report),
        ("cache_audit.csv", result.cache_audit),
        ("attempt_audit.csv", result.attempt_audit),
    ):
        _atomic_write(
            root / filename,
            lambda path, data=frame: data.to_csv(path, index=False),
        )

    _atomic_write(
        root / "summary.json",
        lambda path: path.write_text(
            json.dumps(result.summary, indent=2),
            encoding="utf-8",
        ),
    )


def run_weather_batch(
    plan: WeatherBatchPlan,
    cache: WeatherCache,
    *,
    fs_factory,
    acquirer=acquire_weather_cycle,
    max_workers=2,
    report_dir=None,
) -> WeatherBatchResult:
    """Execute cycle tasks with bounded concurrency and failure isolation."""

    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 4
    ):
        raise WeatherBatchError("max_workers must be an integer from 1 to 4")

    if not callable(fs_factory) or not callable(acquirer):
        raise WeatherBatchError("fs_factory and acquirer must be callable")

    tasks = _validate_tasks(plan)

    # One filesystem client per worker thread, not per cycle.
    worker_local = local()

    def worker_filesystem():
        if not hasattr(worker_local, "fs"):
            worker_local.fs = fs_factory()

        return worker_local.fs

    def execute_task(run_at, leads):
        started = perf_counter()

        task_identity = {
            "forecast_run_at_utc": run_at,
            "forecast_leads": ",".join(f"F{lead:02d}" for lead in leads),
        }

        try:
            # Re-check cache at execution time; the original dry-run
            # plan may have become stale.

            existing = cache.read_cycle(run_at, leads)

            if all(frame is not None for frame in existing.values()):
                cache_rows = [
                    {
                        "forecast_run_at_utc": run_at,
                        "forecast_hour": lead,
                        "cache_status": "hit",
                        "backend": frame["backend"].iloc[0],
                    }
                    for lead, frame in existing.items()
                ]

                return (
                    {
                        **task_identity,
                        "status": "cached",
                        "cache_hits": len(leads),
                        "fetched_hours": 0,
                        "seconds": perf_counter() - started,
                        "error_type": None,
                        "error_message": None,
                    },
                    cache_rows,
                    [],
                )

            result = acquire_cached_weather_cycle(
                cache,
                worker_filesystem(),
                run_at,
                leads,
                acquirer=acquirer,
            )

            if len(result.cache_audit) != len(leads):
                raise WeatherBatchError("Incomplete cache acquisition audit")

            cache_rows = result.cache_audit.assign(forecast_run_at_utc=run_at).to_dict(
                "records"
            )

            attempt_rows = result.attempt_audit.assign(
                forecast_run_at_utc=run_at
            ).to_dict("records")

            hit_count = sum(record["cache_status"] == "hit" for record in cache_rows)

            fetched_count = sum(
                record["cache_status"] == "fetched" for record in cache_rows
            )

            return (
                {
                    **task_identity,
                    "status": "succeeded",
                    "cache_hits": hit_count,
                    "fetched_hours": fetched_count,
                    "seconds": perf_counter() - started,
                    "error_type": None,
                    "error_message": None,
                },
                cache_rows,
                attempt_rows,
            )

        except Exception as exc:
            # Do not allow one failed cycle to stop other cycles.
            # Persist failure details in the task report.

            failure_attempts = getattr(exc, "attempt_audit", None)

            attempt_rows = []

            if isinstance(failure_attempts, pd.DataFrame):
                attempt_rows = failure_attempts.assign(
                    forecast_run_at_utc=run_at
                ).to_dict("records")

            return (
                {
                    **task_identity,
                    "status": "failed",
                    "cache_hits": None,
                    "fetched_hours": None,
                    "seconds": perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                [],
                attempt_rows,
            )

    reports = []
    cache_records = []
    attempt_records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(execute_task, run_at, leads) for run_at, leads in tasks
        ]

        for future in as_completed(futures):
            report, cache_rows, attempt_rows = future.result()

            reports.append(report)
            cache_records.extend(cache_rows)
            attempt_records.extend(attempt_rows)

    task_report = pd.DataFrame(reports)

    if not task_report.empty:
        task_report = task_report.sort_values("forecast_run_at_utc").reset_index(
            drop=True
        )

    cache_audit = pd.DataFrame(cache_records)
    attempt_audit = pd.DataFrame(attempt_records)

    summary = {
        "planned_cycles": len(tasks),
        "planned_hourly_requests": len(plan.hourly_requests),
        "cached_cycles": sum(row["status"] == "cached" for row in reports),
        "succeeded_cycles": sum(row["status"] == "succeeded" for row in reports),
        "failed_cycles": sum(row["status"] == "failed" for row in reports),
        "cache_hits": sum(row.get("cache_status") == "hit" for row in cache_records),
        "fetched_hours": sum(
            row.get("cache_status") == "fetched" for row in cache_records
        ),
    }

    result = WeatherBatchResult(
        task_report=task_report,
        cache_audit=cache_audit,
        attempt_audit=attempt_audit,
        summary=summary,
    )

    if report_dir is not None:
        save_batch_report(result, report_dir)

    return result
