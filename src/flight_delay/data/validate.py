import pandas as pd

from flight_delay.features.schema import BASELINE_FEATURES

ORIGIN_AIRPORTS = {
    "ATL", "DFW", "DEN", "ORD", "LAX", "SFO"
}

LABEL_PAIRS = {
    "DepDel15": "DepDelay",
    "ArrDel15": "ArrDelay",
}

CANDIDATE_KEY_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
]

REQUIRED_COLUMNS = BASELINE_FEATURES | {
    "Cancelled",
    "Diverted",
    "DepDel15",
    "ArrDel15",
    "DepDelay",
    "ArrDelay",
    "Flight_Number_Reporting_Airline",
}


def validate_bts_data(df: pd.DataFrame) -> dict:
    """Validate a BTS pilot dataset without modifying it."""

    # 1. Validate required columns.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Validate flight status fields.
    for col in ["Cancelled", "Diverted"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Invalid flight status: {col}")

    # 3. Validate binary labels and consistency.
    for label_col, delay_col in LABEL_PAIRS.items():
        label = df[label_col]
        delay = df[delay_col]

        invalid = label.notna() & ~label.isin([0, 1])
        if invalid.any():
            raise ValueError(f"Invalid binary labels: {label_col}")

        if not label.isna().equals(delay.isna()):
            raise ValueError(
                f"Missingness mismatch: {label_col}, {delay_col}"
            )

        valid = label.notna()
        expected = (delay.loc[valid] >= 15).astype(int)
        actual = label.loc[valid].astype(int)

        if not expected.equals(actual):
            raise ValueError(f"Label inconsistency: {label_col}")

    # 4. Identify completed flights.
    completed = (
        df["Cancelled"].eq(0)
        & df["Diverted"].eq(0)
    )

    # 5. Audit missing labels in completed flights. Real BTS data may contain
    # completed records without one or both outcomes, so missing labels are
    # counted rather than imputed or rejected.
    target_columns = list(LABEL_PAIRS)
    missing_dep_label = completed & df["DepDel15"].isna()
    missing_arr_label = completed & df["ArrDel15"].isna()
    missing_any_label = missing_dep_label | missing_arr_label

    # 6. Build the common modeling cohort. Both targets must be observed so
    # departure and arrival models use the same source rows.
    in_scope = df["Origin"].isin(ORIGIN_AIRPORTS)
    labels_present = df[target_columns].notna().all(axis=1)
    eligible = in_scope & completed & labels_present

    if df.loc[eligible, list(BASELINE_FEATURES)].isna().any().any():
        raise ValueError("Missing baseline features in eligible flights")

    # 7. Audit candidate flight keys without removing records.
    key_df = df[CANDIDATE_KEY_COLUMNS]
    complete_key_mask = key_df.notna().all(axis=1)
    incomplete_key_records = int((~complete_key_mask).sum())
    complete_keys = key_df.loc[complete_key_mask]
    duplicate_key_records = int(
        complete_keys.duplicated(
            subset=CANDIDATE_KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    return {
        "raw_records": len(df),
        "six_airport_records": int(in_scope.sum()),
        "eligible_records": int(eligible.sum()),
        "completed_missing_dep_label_records": int(
            missing_dep_label.sum()
        ),
        "completed_missing_arr_label_records": int(
            missing_arr_label.sum()
        ),
        "completed_missing_label_records": int(
            missing_any_label.sum()
        ),
        "in_scope_completed_missing_dep_label_records": int(
            (in_scope & missing_dep_label).sum()
        ),
        "in_scope_completed_missing_arr_label_records": int(
            (in_scope & missing_arr_label).sum()
        ),
        "in_scope_completed_missing_label_records": int(
            (in_scope & missing_any_label).sum()
        ),
        "incomplete_candidate_key_records": incomplete_key_records,
        "duplicate_candidate_key_records": duplicate_key_records,
    }
