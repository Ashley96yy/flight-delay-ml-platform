"""Frozen Stage 9A evaluation for an aligned pair of prediction tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from flight_delay.weather.stage9b_pairing import PIT_STATUS, TARGETS, PairedInputs
from flight_delay.weather.stage9b_protocol import frozen_hours, frozen_protocol


def bootstrap_daily_ap(daily_differences: pd.DataFrame):
    """Sample whole paired dates within quarters, sharing draws across targets."""
    spec = frozen_protocol()["bootstrap"]
    dates = pd.DatetimeIndex(daily_differences.index)
    expected = [row["date"] for row in frozen_hours() if row["split"] == "validation"]
    if dates.strftime("%Y-%m-%d").tolist() != expected:
        raise ValueError("Bootstrap requires the 12 ordered frozen validation dates")
    values = daily_differences[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite paired date scores")
    replicates = np.zeros((spec["n_resamples"], len(TARGETS)))
    rng = np.random.default_rng(spec["seed"])
    for quarter in range(1, 5):
        block = values[dates.quarter == quarter]
        if len(block) != 3:
            raise ValueError("Bootstrap requires three complete dates per quarter")
        index = rng.integers(0, len(block), size=(len(replicates), len(block)))
        replicates += block[index].sum(axis=1)
    tail = (1 - spec["confidence"]) / 2
    limits = np.quantile(replicates / len(values), [tail, 1 - tail], axis=0)
    return pd.DataFrame(
        {
            "mean": values.mean(axis=0),
            "lower": limits[0],
            "upper": limits[1],
            "leave_one_date_out_min": (
                (values.sum(axis=0) - values) / (len(values) - 1)
            ).min(axis=0),
        },
        index=TARGETS,
    )


def decision(intervals, baseline_ranking_valid, departure_weather_splits):
    spec = frozen_protocol()["decision"]
    dep = intervals.loc["DepDel15"]
    arr = intervals.loc["ArrDel15"]
    if not baseline_ranking_valid or departure_weather_splits <= 0:
        return "unverifiable"
    if (
        dep["mean"] >= spec["departure_min_ap_gain"]
        and dep["lower"] > 0
        and dep["leave_one_date_out_min"] > 0
        and arr["lower"] >= spec["arrival_lower_bound_floor"]
    ):
        return "exploratory_signal"
    if dep["upper"] < spec["departure_min_ap_gain"]:
        return "below_practical_threshold"
    return "inconclusive"


def _scores(labels, scores):
    return {
        "ap": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "brier": brier_score_loss(labels, scores),
    }


def evaluate_predictions(
    paired: PairedInputs,
    baseline_scores: pd.DataFrame,
    weather_scores: pd.DataFrame,
    *,
    departure_weather_splits: int,
):
    """Assess precomputed validation scores. There is no model fitting path."""
    if (
        not isinstance(departure_weather_splits, int)
        or isinstance(departure_weather_splits, bool)
        or departure_weather_splits < 0
    ):
        raise ValueError(
            "Provide the audited nonnegative departure weather split count"
        )
    mask = paired.manifest["split"].eq("validation")
    validation = paired.manifest.loc[mask]
    if (
        paired.pit_status != PIT_STATUS
        or not validation["pit_status"].eq(PIT_STATUS).all()
    ):
        raise ValueError("Stage 9B evaluation is retrospective_exploratory")
    for arm in (baseline_scores, weather_scores):
        if not arm.index.equals(validation.index) or list(arm.columns) != TARGETS:
            raise ValueError(
                "Prediction identity, order or targets differ from paired validation"
            )
        scores = arm.to_numpy(dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(
                "Prediction probabilities must be finite and within [0, 1]"
            )

    rows, pooled, predictions = [], [], {"V3-A": baseline_scores, "V5": weather_scores}
    for arm, scores in predictions.items():
        for target in TARGETS:
            labels = validation[target]
            if set(labels.unique()) != {0, 1}:
                raise ValueError("Invalid validation outcomes")
            pooled.append(
                {
                    "arm": arm,
                    "target": target,
                    "prevalence": labels.mean(),
                    "constant_scores": scores[target].nunique() == 1,
                    **_scores(labels, scores[target]),
                }
            )
        for date, group in validation.groupby("date", sort=True):
            for target in TARGETS:
                labels = group[target]
                if set(labels.unique()) != {0, 1}:
                    raise ValueError("Frozen validation date lacks both classes")
                rows.append(
                    {
                        "date": date,
                        "arm": arm,
                        "target": target,
                        "rows": len(group),
                        "prevalence": labels.mean(),
                        **_scores(labels, scores.loc[group.index, target]),
                    }
                )
    daily = pd.DataFrame(rows)
    metrics = ["ap", "roc_auc", "brier"]
    pivot = daily.pivot(index=["date", "target"], columns="arm", values=metrics)
    differences = pivot.xs("V5", level="arm", axis=1) - pivot.xs(
        "V3-A", level="arm", axis=1
    )
    ap_differences = differences["ap"].unstack("target")
    intervals = bootstrap_daily_ap(ap_differences)
    dep_base = daily.loc[(daily["arm"] == "V3-A") & (daily["target"] == "DepDel15")]
    baseline_valid = (
        baseline_scores["DepDel15"].nunique() > 1
        and (dep_base["ap"] - dep_base["prevalence"]).mean() > 0
    )
    return {
        "pit_status": PIT_STATUS,
        "decision": decision(intervals, baseline_valid, departure_weather_splits),
        "daily": daily,
        "daily_paired_differences": differences.reset_index(),
        "pooled": pd.DataFrame(pooled),
        "ap_intervals": intervals,
    }
