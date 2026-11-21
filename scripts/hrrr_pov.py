"""Plan and optionally run the paired HRRR experiment from local data only."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.features.advanced_scheduled import (
    INTERACTION_FEATURE_COLUMNS,
    build_advanced_scheduled_features,
)
from flight_delay.features.build import (
    CATEGORICAL_FEATURE_COLUMNS,
    build_baseline_features,
)
from flight_delay.features.schema import BASELINE_FEATURES
from flight_delay.weather.batch_plan import build_weather_batch_plan
from flight_delay.weather.cache import WeatherCache, sha256_file
from flight_delay.weather.config import ORIGIN_AIRPORTS, WEATHER_AVAILABILITY_LAG_HOURS
from flight_delay.weather.coverage_audit import audit_weather_coverage
from flight_delay.weather.features import (
    MODEL_WEATHER_COLUMNS,
    build_weather_feature_table,
)
from flight_delay.weather.flight_join import join_weather_to_flights
from flight_delay.weather.flight_times import convert_bts_departure_to_utc

DATES = ("2023-01-15", "2023-07-15", "2024-01-15", "2024-07-15")
IDENTITY = sorted(BASELINE_FEATURES | {"Flight_Number_Reporting_Airline"})
TARGETS = {"Departure": "DepDel15", "Arrival": "ArrDel15"}
HOUR_KEYS = ["forecast_run_at_utc", "forecast_hour"]
STRATA = ["sample_date", "Origin"]
CATEGORIES = (*CATEGORICAL_FEATURE_COLUMNS, "Route", "Airline_Route")
MIN_VALIDATION = 100
MIN_PER_CLASS = 10


def load_cohort(data_dir):
    """Freeze a label-blind sample of up to 200 flights per airport/date."""
    parts, sources = [], {}
    columns = [*IDENTITY, *TARGETS.values()]
    for date in DATES:
        month = date[:7].replace("-", "_")
        path = data_dir / f"bts_{month}_completed.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[
            pd.to_datetime(frame.FlightDate).dt.strftime("%Y-%m-%d").eq(date)
            & frame.Origin.isin(ORIGIN_AIRPORTS)
        ]
        frame = convert_bts_departure_to_utc(frame).flights
        departure = frame.scheduled_departure_utc.dt
        frame = frame.loc[
            departure.strftime("%Y-%m-%d").eq(date) & departure.hour.between(9, 15)
        ].copy()
        frame["flight_id"] = [
            hashlib.sha256(json.dumps(["42", *map(str, values)]).encode()).hexdigest()
            for values in frame[IDENTITY].itertuples(index=False, name=None)
        ]
        if frame.flight_id.duplicated().any():
            raise ValueError(f"Duplicate scheduled flight identity on {date}")
        sampled = frame.sort_values("flight_id").groupby("Origin", sort=True).head(200)
        parts.append(sampled.assign(sample_date=date, year=int(date[:4])))
        sources[path.name] = sha256_file(path)
    return pd.concat(parts, ignore_index=True), sources


def approved_hours(cache, hits, evidence_path):
    """Read external review decisions keyed to the exact committed cache payload."""
    if evidence_path is None:
        return {}
    evidence = json.loads(evidence_path.read_text())
    if not isinstance(evidence, list):
        raise ValueError("Reviewed evidence must be a JSON list")
    accepted = {}
    for entry in evidence:
        run = pd.Timestamp(entry["forecast_run_at_utc"])
        available = pd.Timestamp(entry["available_by_utc"])
        if (
            pd.isna(run)
            or pd.isna(available)
            or run.tzinfo is None
            or available.tzinfo is None
            or available < run
            or entry["kind"]
            not in {"verified_publication", "historical_full_hour_receipt"}
            or type(entry["forecast_hour"]) is not int
            or entry["covers_full_hour"] is not True
            or not isinstance(entry["evidence_reference"], str)
            or not entry["evidence_reference"].strip()
        ):
            raise ValueError("Invalid historically reviewed HRRR evidence")
        key = (run, entry["forecast_hour"])
        if key not in hits:
            raise ValueError("Receipt does not refer to a committed pilot cache hour")
        manifest = json.loads(
            (cache.entry_directory(*key) / "manifest.json").read_text()
        )
        if entry["cache_sha256"] != manifest["sha256"] or key in accepted:
            raise ValueError("Receipt payload mismatch or duplicate receipt")
        accepted[key] = available
    return accepted


def minimum_additions(mapping, hit_keys, minimum_rows, dates):
    """Find the smallest set of missing hours meeting label-blind support gates."""
    counts = mapping.groupby(HOUR_KEYS + STRATA).size().reset_index(name="n")
    groups = dict(tuple(counts.groupby(HOUR_KEYS)))
    options = sorted(groups.keys() - hit_keys)
    required = {(date, airport) for date in dates for airport in ORIGIN_AIRPORTS}
    supported = {
        key: set(frame[STRATA].itertuples(index=False, name=None))
        for key, frame in groups.items()
    }
    sizes = {key: int(frame.n.sum()) for key, frame in groups.items()}
    cached = groups.keys() & hit_keys
    base = set().union(*(supported[key] for key in cached))
    baseline_size = sum(sizes[key] for key in cached)
    for n in range(len(options) + 1):
        for keys in itertools.combinations(options, n):
            if (
                baseline_size + sum(sizes[key] for key in keys) >= minimum_rows
                and base.union(*(supported[key] for key in keys)) == required
            ):
                return list(keys)
    return None


def sample_diagnostics(mapping, eligible, cohort, training_minimum, weather):
    report, blockers = {}, []
    for year, minimum in ((2023, training_minimum), (2024, MIN_VALIDATION)):
        split = mapping.loc[mapping.year.eq(year)]
        matched = split.loc[eligible.loc[split.index]]
        labels = cohort.iloc[matched.flight_row_position][list(TARGETS.values())]
        strata = matched[STRATA].drop_duplicates()
        counts = {
            target: {
                "positive": int(labels[target].eq(1).sum()),
                "negative": int(labels[target].eq(0).sum()),
            }
            for target in TARGETS.values()
        }
        values = weather.loc[matched.index]
        report[str(year)] = {
            "cohort": len(split),
            "eligible": len(matched),
            "eligible_fraction": len(matched) / len(split) if len(split) else 0.0,
            "airport_date_strata": len(strata),
            "source_forecast_hours": len(matched[HOUR_KEYS].drop_duplicates()),
            "airport_forecast_groups": len(
                matched[[*HOUR_KEYS, "Origin"]].drop_duplicates()
            ),
            "field_counts_all_cohort": {
                column: int(np.isfinite(weather.loc[split.index, column]).sum())
                for column in MODEL_WEATHER_COLUMNS
            },
            "label_counts": counts,
        }
        if len(matched) < minimum or len(strata) != 2 * len(ORIGIN_AIRPORTS):
            blockers.append(
                f"{year}: need {minimum} rows and all 12 airport/date strata"
            )
        if any(min(count.values()) < MIN_PER_CLASS for count in counts.values()):
            blockers.append(
                f"{year}: need {MIN_PER_CLASS} positives and negatives per target"
            )
        if not values.empty and not values.nunique().gt(1).any():
            blockers.append(f"{year}: no variation in eligible weather features")
        if not labels.isin([0, 1]).all().all():
            blockers.append(f"{year}: nonbinary or missing targets")
    return report, blockers


def fit_comparison(flights, joined, params, output):
    """Use the same ordered row positions in both model arms."""
    from lightgbm import LGBMClassifier
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        auc,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )

    selected = flights.iloc[joined.flight_row_position].reset_index(drop=True)
    weather = joined[MODEL_WEATHER_COLUMNS].reset_index(drop=True)
    train = selected.year.eq(2023)
    base = build_baseline_features(selected[sorted(BASELINE_FEATURES)])
    advanced = build_advanced_scheduled_features(selected)
    base = pd.concat([base, advanced[list(INTERACTION_FEATURE_COLUMNS)]], axis=1)
    for column in CATEGORIES:
        levels = sorted(base.loc[train, column].astype("string").unique())
        base[column] = base[column].astype("string").astype(pd.CategoricalDtype(levels))
    predictions = selected.loc[~train, ["flight_id", *TARGETS.values()]].copy()
    metrics, bins = [], []
    for arm, features in (
        ("V3-A", base),
        ("V5-HRRR", pd.concat([base, weather], axis=1)),
    ):
        for task, target in TARGETS.items():
            model = LGBMClassifier(**params[task])
            model.fit(features.loc[train], selected.loc[train, target])
            probabilities = model.predict_proba(features.loc[~train])[:, 1]
            y = selected.loc[~train, target]
            precision, recall, _ = precision_recall_curve(y, probabilities)
            predictions[f"{arm}_{target}"] = probabilities
            metrics.append(
                {
                    "mode": "exploratory",
                    "arm": arm,
                    "target": target,
                    "pr_auc_trapezoid": auc(recall, precision),
                    "average_precision": average_precision_score(y, probabilities),
                    "roc_auc": roc_auc_score(y, probabilities),
                    "f1_at_0_5": f1_score(y, probabilities >= 0.5, zero_division=0),
                    "brier": brier_score_loss(y, probabilities),
                }
            )
            observed, predicted = calibration_curve(y, probabilities, n_bins=5)
            bins.extend(
                {
                    "arm": arm,
                    "target": target,
                    "observed": obs,
                    "predicted": pred,
                }
                for obs, pred in zip(observed, predicted, strict=True)
            )
    predictions.to_csv(output / "paired_validation_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "exploratory_metrics.csv", index=False)
    pd.DataFrame(bins).to_csv(output / "exploratory_calibration.csv", index=False)


def run(args):
    if args.fit and args.mode != "exploratory":
        raise ValueError("Fitting requires explicitly exploratory mode")
    cohort, source_hashes = load_cohort(args.data_dir)
    params = json.loads(args.params.read_text())
    # Two maximum-size minimum leaves is a practical fit guard, not a power claim.
    training_minimum = 2 * max(params[task]["min_child_samples"] for task in TARGETS)
    cache = WeatherCache(args.cache)
    plan = build_weather_batch_plan(cohort, cache)
    mapping = plan.flight_mapping.assign(
        sample_date=cohort.sample_date,
        year=cohort.year,
        flight_id=cohort.flight_id,
    )
    hit_hours = plan.hour_status.loc[plan.hour_status.cache_status.eq("hit"), HOUR_KEYS]
    hits = set(hit_hours.itertuples(index=False, name=None))
    approved = approved_hours(cache, hits, args.reviewed_evidence)
    forecast_rows = [cache.read_hour(*key) for key in sorted(hits)]
    cache_hashes = {
        f"{run.isoformat()}/F{lead:02d}": json.loads(
            (cache.entry_directory(run, lead) / "manifest.json").read_text()
        )["sha256"]
        for run, lead in sorted(hits)
    }
    exploratory = pd.Series(False, index=mapping.index)
    strict = exploratory.copy()
    weather = pd.DataFrame(np.nan, index=mapping.index, columns=MODEL_WEATHER_COLUMNS)
    mapping["verified_available_by_utc"] = pd.to_datetime(
        [
            approved.get(key)
            for key in mapping[HOUR_KEYS].itertuples(index=False, name=None)
        ],
        utc=True,
    )
    if forecast_rows:
        features = build_weather_feature_table(
            pd.concat(forecast_rows, ignore_index=True)
        )
        audit = audit_weather_coverage(mapping, features)
        weather = audit.detail[MODEL_WEATHER_COLUMNS].where(audit.detail.time_safe)
        exploratory = audit.detail.ready
        strict = exploratory & mapping.verified_available_by_utc.le(
            mapping.prediction_at_utc
        )
        mapping["legacy_lag_audit_status"] = audit.detail.status
    else:
        mapping["legacy_lag_audit_status"] = "missing_weather"
    mapping["exploratory_weather_eligible"] = exploratory
    mapping["strict_weather_eligible"] = strict
    mapping["historical_availability_status"] = "missing_weather"
    mapping.loc[exploratory, "historical_availability_status"] = (
        "availability_unverified"
    )
    mapping.loc[
        exploratory & mapping.verified_available_by_utc.notna() & ~strict,
        "historical_availability_status",
    ] = "available_after_cutoff"
    mapping.loc[strict, "historical_availability_status"] = "availability_verified"

    args.output.mkdir(parents=True, exist_ok=True)
    # The report is rebuilt from source on every run. Refuse stale training outputs.
    if (
        any(args.output.glob("*metrics.csv"))
        or (args.output / "paired_validation_predictions.csv").exists()
    ):
        raise ValueError("Use a new output directory for a rerun after fitting")
    mapping.to_csv(args.output / "cohort_audit.csv", index=False)
    plan.hour_status.to_csv(args.output / "all_hour_requests.csv", index=False)
    plan.cycle_tasks.to_csv(args.output / "cycle_tasks.csv", index=False)

    minimum_plan = []
    possible = True
    for year, minimum in ((2023, training_minimum), (2024, MIN_VALIDATION)):
        additional = minimum_additions(
            mapping.loc[mapping.year.eq(year)],
            hits,
            minimum,
            [date for date in DATES if date.startswith(str(year))],
        )
        if additional is None:
            possible = False
            continue
        for run, lead in additional:
            minimum_plan.append(
                {
                    "year": year,
                    "forecast_run_at_utc": run.isoformat(),
                    "forecast_hour": lead,
                    "forecast_valid_at_utc": (
                        run + pd.Timedelta(hours=lead)
                    ).isoformat(),
                }
            )
    (args.output / "minimum_additions.json").write_text(
        json.dumps(minimum_plan, indent=2) + "\n"
    )
    cost = None
    if args.grib_cost_reference:
        sizes = pd.read_csv(args.grib_cost_reference)["downloaded_bytes"]
        if sizes.empty or not sizes.ge(0).all() or not sizes.map(math.isfinite).all():
            raise ValueError("Invalid historical GRIB cost reference")
        cost = {
            "historical_reference_bytes_per_hour_min": int(sizes.min()),
            "historical_reference_bytes_per_hour_max": int(sizes.max()),
            "min_additions_estimated_payload_bytes_low": int(sizes.min())
            * len(minimum_plan),
            "min_additions_estimated_payload_bytes_high": int(sizes.max())
            * len(minimum_plan),
            "all_missing_estimated_payload_bytes_low": int(sizes.min())
            * plan.summary()["missing_hours"],
            "all_missing_estimated_payload_bytes_high": int(sizes.max())
            * plan.summary()["missing_hours"],
            "range_gets_per_hour": 9,
            "inventory_gets_per_hour": 1,
            "min_additions_estimated_requests_no_retry": 10 * len(minimum_plan),
            "all_missing_estimated_requests_no_retry": 10
            * plan.summary()["missing_hours"],
            "reference_sha256": sha256_file(args.grib_cost_reference),
        }
    explor_report, blockers = sample_diagnostics(
        mapping, exploratory, cohort, training_minimum, weather
    )
    strict_report, strict_blockers = sample_diagnostics(
        mapping, strict, cohort, training_minimum, weather.where(strict)
    )
    report = {
        "weather_network_requests": 0,
        "weather_network_bytes": 0,
        "availability_lag_hours": WEATHER_AVAILABILITY_LAG_HOURS,
        "cohort_rule": "seed-42 SHA256, upper bound 200 per airport/date, UTC hours 09–15",
        "source_hashes": source_hashes,
        "params_sha256": sha256_file(args.params),
        "cohort_sha256": sha256_file(args.output / "cohort_audit.csv"),
        "cache_payload_hashes": cache_hashes,
        "weather_plan": plan.summary(),
        "minimum_additional_hours": len(minimum_plan) if possible else None,
        "minimum_plan_note": "Row and airport/date support only. Class counts and weather variance still require review.",
        "grib_range_estimate": cost,
        "exploratory": {"splits": explor_report, "blockers": blockers},
        "strict_weather": {"splits": strict_report, "blockers": strict_blockers},
        "strict_experiment": "blocked: source BTS schedule snapshots at T-24h are unverified",
        "training_requested": args.fit,
        "training_executed": False,
    }
    if args.fit and not blockers:
        valid = mapping.loc[exploratory].reset_index(drop=True)
        audit_weather_coverage(valid, features, strict=True)
        joined = join_weather_to_flights(valid, features)
        fit_comparison(cohort, joined, params, args.output)
        report["training_executed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "cache", "params", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reviewed-evidence", type=Path)
    parser.add_argument("--grib-cost-reference", type=Path)
    parser.add_argument("--mode", choices=["strict", "exploratory"], default="strict")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.fit and args.mode != "exploratory":
        parser.error("--fit requires --mode exploratory and the sample gates")
    report = run(args)
    print(json.dumps(report, indent=2))
    return 2 if args.fit and not report["training_executed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
