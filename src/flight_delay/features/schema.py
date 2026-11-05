TARGETS = {
    "departure": "DepDel15",
    "arrival": "ArrDel15",
}

# Features allowed for the initial T-24h baseline.
BASELINE_FEATURES = frozenset({
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
})


def validate_baseline_features(columns):
    """Reject fields outside the baseline feature allowlist."""
    invalid = set(columns) - BASELINE_FEATURES

    if invalid:
        raise ValueError(
            f"Invalid or potentially leaking features: {sorted(invalid)}"
        )
