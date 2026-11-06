import pandas as pd

from flight_delay.data.validate import (
    ORIGIN_AIRPORTS,
    validate_bts_data,
)


def prepare_modeling_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the initial completed-flight modeling cohort.

    The function validates input data, filters eligible
    records, and normalizes selected data types.

    It does not modify the original DataFrame.
    """

    # Step 1: Validate data before preprocessing.
    validate_bts_data(df)

    # Step 2: Restrict flights to selected origin airports.
    airport_mask = df["Origin"].isin(ORIGIN_AIRPORTS)

    # Step 3: Keep completed flights only.
    completed_mask = (
        df["Cancelled"].eq(0)
        & df["Diverted"].eq(0)
    )

    # Step 4: Require valid labels for both tasks.
    valid_labels_mask = (
        df["DepDel15"].isin([0, 1])
        & df["ArrDel15"].isin([0, 1])
    )

    # Step 5: Apply filtering without modifying raw data.
    eligible_mask = (
        airport_mask
        & completed_mask
        & valid_labels_mask
    )

    cleaned = df.loc[eligible_mask].copy()

    # Step 6: Normalize date and label types.
    cleaned["FlightDate"] = pd.to_datetime(
        cleaned["FlightDate"],
        errors="raise",
    )

    for target in ["DepDel15", "ArrDel15"]:
        cleaned[target] = cleaned[target].astype("int8")

    # Step 7: Reset DataFrame index.
    return cleaned.reset_index(drop=True)
