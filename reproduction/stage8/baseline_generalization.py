"""Evaluate frozen V3-A on the predeclared Stage 8 schedule-only cohort."""

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from flight_delay.data.manifest import sha256_file
from flight_delay.features.advanced_scheduled import (
    INTERACTION_FEATURE_COLUMNS,
    build_advanced_scheduled_features,
)
from flight_delay.features.build import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_baseline_features,
)
from flight_delay.features.schema import BASELINE_FEATURES
from flight_delay.weather.flight_times import (
    AIRPORT_TIMEZONES,
    convert_bts_departure_to_utc,
)

FROZEN_PARAM_HASH = "2f36b7b62d622d7458abd2bf54711ec24163df425b287b67ebb442c64be4b65c"
CATEGORIES = (*CATEGORICAL_FEATURE_COLUMNS, "Route", "Airline_Route")
FLIGHT_IDENTITY = [*sorted(BASELINE_FEATURES), "Flight_Number_Reporting_Airline"]
TARGETS = {"Departure": "DepDel15", "Arrival": "ArrDel15"}
CONTEXT_DATES = ("2024-01-15", "2024-07-15")
YEARS = (2023, 2024)


def load_data(data_dir, manifests):
    """Verify the exact, fixed month files before applying the cohort selector."""
    cohorts, input_audit = [], []
    for year in YEARS:
        for month in range(1, 13):
            path = data_dir / f"bts_{year}_{month:02d}_completed.parquet"
            manifest = json.loads((manifests / f"{year}-{month:02d}.json").read_text())
            expected = manifest["processed"]
            sha = sha256_file(path)
            if path.name != expected["filename"] or sha != expected["sha256"]:
                raise ValueError(f"Processed input manifest mismatch: {path.name}")
            local = pd.read_parquet(path, columns=[*FLIGHT_IDENTITY, *TARGETS.values()])
            local["FlightDate"] = pd.to_datetime(local["FlightDate"], errors="raise")
            dates = local["FlightDate"].dt
            if (
                len(local) != expected["row_count"]
                or not (dates.year.eq(year) & dates.month.eq(month)).all()
            ):
                raise ValueError(
                    f"Input row-count or partition date mismatch: {path.name}"
                )
            origins = local.loc[local["Origin"].isin(AIRPORT_TIMEZONES)]
            converted = convert_bts_departure_to_utc(origins)
            scheduled = converted.flights.scheduled_departure_utc.dt
            in_window = scheduled.hour.between(
                9, 15
            ) & scheduled.normalize().dt.tz_localize(None).eq(
                converted.flights["FlightDate"]
            )
            cohort = converted.flights.loc[
                in_window, [*FLIGHT_IDENTITY, *TARGETS.values()]
            ]
            if cohort.duplicated(FLIGHT_IDENTITY).any():
                raise ValueError(f"Duplicate scheduled identities in {path.name}")
            if not cohort[list(TARGETS.values())].isin([0, 1]).all().all():
                raise ValueError(f"Invalid binary label in {path.name}")
            cohorts.append(cohort)
            input_audit.append(
                {
                    "file": path.name,
                    "file_sha256": sha,
                    "raw_rows": len(local),
                    "airport_rows": len(origins),
                    "selected_rows": len(cohort),
                    "dates": int(cohort.FlightDate.nunique()),
                    "utc_conversion_status": converted.summary["status_counts"],
                }
            )
    return pd.concat(cohorts, ignore_index=True), input_audit


def build_features(cohort, train, primary):
    """Build ordered V3-A inputs and report train-to-validation category coverage."""
    base = build_baseline_features(cohort[sorted(BASELINE_FEATURES)])
    advanced = build_advanced_scheduled_features(cohort)
    X = pd.concat([base, advanced[list(INTERACTION_FEATURE_COLUMNS)]], axis=1)
    if list(X) != [*FEATURE_COLUMNS, *INTERACTION_FEATURE_COLUMNS]:
        raise ValueError("Out-of-contract V3-A feature matrix")
    coverage = []
    for column in CATEGORIES:
        values = X[column].astype("string")
        levels = sorted(values.loc[train].dropna().unique())
        known = values.isin(levels)
        X[column] = values.where(known).astype(pd.CategoricalDtype(levels))
        for partition, mask in (
            ("validation_primary", primary),
            ("validation_all", ~train),
        ):
            unseen = values[mask & ~known]
            coverage.append(
                {
                    "partition": partition,
                    "feature": column,
                    "training_levels": len(levels),
                    "rows": int(mask.sum()),
                    "unknown_rows": len(unseen),
                    "unknown_levels": int(unseen.nunique()),
                    "unknown_fraction": float(len(unseen) / mask.sum()),
                }
            )
    return X, coverage


def metrics(y, score, train_prior):
    """Calculate ranking only when the group contains both binary outcomes."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or not len(y) or not np.isfinite(score).all():
        raise ValueError("Invalid evaluation vectors")
    n, positives = len(y), int(y.sum())
    prevalence = positives / n
    both_classes = 0 < positives < n
    ap = float(average_precision_score(y, score)) if both_classes else None
    roc = float(roc_auc_score(y, score)) if both_classes else None
    cuts = np.quantile(score, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        "rows": n,
        "positive_rows": positives,
        "prevalence": prevalence,
        "ap": ap,
        "ap_minus_prevalence": ap - prevalence if both_classes else None,
        "roc_auc": roc,
        "brier": float(brier_score_loss(y, score)),
        "brier_train_prior": prevalence * (1 - train_prior) ** 2
        + (1 - prevalence) * train_prior**2,
        "mean_score": float(score.mean()),
        "score_min": float(cuts[0]),
        "score_p05": float(cuts[1]),
        "score_p50": float(cuts[2]),
        "score_p95": float(cuts[3]),
        "score_max": float(cuts[4]),
        "distinct_scores": int(np.unique(score).size),
        "scores_at_least_0_5": int((score >= 0.5).sum()),
    }


def block_intervals(dates):
    """Bootstrap equal-weight daily differences over fixed seven-day blocks."""
    dates = dates.dropna(subset=["ap_minus_prevalence", "roc_auc"]).copy()
    day = pd.to_datetime(dates["group"])
    dates["ap_delta"] = dates["ap_minus_prevalence"]
    dates["auc_delta"] = dates["roc_auc"] - 0.5
    dates["block"] = (day - pd.Timestamp("2024-01-01")).dt.days // 7
    # Assign each block to the quarter of its start to retain full week blocks.
    dates["quarter"] = (
        pd.Timestamp("2024-01-01") + pd.to_timedelta(dates["block"] * 7, unit="D")
    ).dt.quarter
    totals = dates.groupby(["quarter", "block"], as_index=False).agg(
        dates=("group", "size"),
        ap_delta=("ap_delta", "sum"),
        auc_delta=("auc_delta", "sum"),
    )
    rng = np.random.default_rng(42)
    draws = np.zeros((2000, 3))
    for _, quarter in totals.groupby("quarter"):
        rows = quarter[["dates", "ap_delta", "auc_delta"]].to_numpy(dtype=float)
        indices = rng.integers(0, len(rows), size=(len(draws), len(rows)))
        draws += rows[indices].sum(axis=1)
    percentiles = np.quantile(draws[:, 1:] / draws[:, :1], [0.025, 0.975], axis=0)
    return {
        "comparable_dates": len(dates),
        "blocks": len(totals),
        "ap_minus_prevalence": {
            "daily_mean": float(dates["ap_delta"].mean()),
            "positive_date_fraction": float(dates["ap_delta"].gt(0).mean()),
            "interval_95": percentiles[:, 0].tolist(),
        },
        "roc_auc_minus_half": {
            "daily_mean": float(dates["auc_delta"].mean()),
            "positive_date_fraction": float(dates["auc_delta"].gt(0).mean()),
            "interval_95": percentiles[:, 1].tolist(),
        },
    }


def evaluate(cohort, X, masks, parameters):
    pooled, slices, states, inference = [], [], [], {}
    strata = cohort[["Origin", "Reporting_Airline"]].assign(
        month=cohort.FlightDate.dt.strftime("%Y-%m"),
        date=cohort.FlightDate.dt.strftime("%Y-%m-%d"),
    )
    train = masks["train"]
    for task, target in TARGETS.items():
        labels = cohort[target]
        prior = float(labels.loc[train].mean())
        model = lgb.LGBMClassifier(**parameters[task])
        model.fit(X.loc[train], labels.loc[train], categorical_feature=list(CATEGORIES))
        if model.classes_.tolist() != [0, 1]:
            raise ValueError("Unexpected model class ordering")
        scores = pd.Series(model.predict_proba(X)[:, 1], index=cohort.index)
        booster = model.booster_
        states.append(
            {
                "target": target,
                "requested_trees": parameters[task]["n_estimators"],
                "fitted_trees": booster.num_trees(),
                "fitted_splits": int(booster.feature_importance("split").sum()),
                "split_count_by_feature": dict(
                    zip(
                        X.columns,
                        map(int, booster.feature_importance("split")),
                        strict=True,
                    )
                ),
            }
        )
        for partition, mask in masks.items():
            pooled.append(
                {
                    "partition": partition,
                    "target": target,
                    "dates": int(cohort.loc[mask, "FlightDate"].nunique()),
                    **metrics(labels.loc[mask], scores.loc[mask], prior),
                }
            )
        current = []
        for partition in ("train", "validation_primary", "validation_all"):
            mask = masks[partition]
            for dimension in strata.columns:
                groups = strata.loc[mask].groupby(dimension).groups
                for group, index in groups.items():
                    current.append(
                        {
                            "partition": partition,
                            "target": target,
                            "dimension": dimension,
                            "group": group,
                            "dates": int(cohort.loc[index, "FlightDate"].nunique()),
                            **metrics(labels.loc[index], scores.loc[index], prior),
                        }
                    )
        current = pd.DataFrame(current)
        daily = current[
            (current.partition == "validation_primary") & (current.dimension == "date")
        ]
        inference[target] = block_intervals(daily)
        slices.append(current)
    return pd.DataFrame(pooled), pd.concat(slices, ignore_index=True), states, inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "manifest-dir", "params", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.params) != FROZEN_PARAM_HASH:
        parser.error("Stage 8 requires the original frozen parameter file")
    parameters = json.loads(args.params.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    cohort, input_audit = load_data(args.data_dir, args.manifest_dir)
    train = cohort.FlightDate.dt.year.eq(2023)
    primary = ~train & ~cohort.FlightDate.isin(pd.to_datetime(CONTEXT_DATES))
    masks = {
        "train": train,
        "validation_primary": primary,
        "validation_all": ~train,
        "validation_jan_aug": primary & cohort.FlightDate.dt.month.le(8),
        "validation_sep_dec": primary & cohort.FlightDate.dt.month.ge(9),
        **{
            f"context_{day}": cohort.FlightDate.eq(pd.Timestamp(day))
            for day in CONTEXT_DATES
        },
    }
    if cohort.loc[train, "FlightDate"].max() >= cohort.loc[~train, "FlightDate"].min():
        raise ValueError("Non-chronological training boundary")
    X, coverage = build_features(cohort, train, primary)
    pooled, slices, states, inference = evaluate(cohort, X, masks, parameters)
    pooled.to_csv(args.output / "pooled_metrics.csv", index=False)
    slices.to_csv(args.output / "group_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "category_coverage.csv", index=False)
    summary = {
        "protocol_sha256": sha256_file(args.protocol),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "params_sha256": FROZEN_PARAM_HASH,
        "feature_columns": list(X.columns),
        "fit_year": 2023,
        "validation_year": 2024,
        "context_dates": CONTEXT_DATES,
        "versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": input_audit,
        "lightgbm": states,
        "day_block_inference": inference,
        "weather_requests": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
