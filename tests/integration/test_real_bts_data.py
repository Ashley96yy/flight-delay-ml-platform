import zipfile
from pathlib import Path

import pandas as pd
import pytest

from flight_delay.data.validate import (
    REQUIRED_COLUMNS,
    validate_bts_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ZIP_NAME = (
    "On_Time_Reporting_Carrier_On_Time_Performance_"
    "1987_present_2023_1.zip"
)

ZIP_PATH = RAW_DIR / ZIP_NAME


@pytest.mark.integration
def test_january_2023_bts_data():

    if not ZIP_PATH.exists():
        pytest.skip("January 2023 BTS dataset is not available.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        csv_name = next(
            name for name in zf.namelist()
            if name.lower().endswith(".csv")
        )

        with zf.open(csv_name) as file:
            df = pd.read_csv(
                file,
                usecols=list(REQUIRED_COLUMNS),
                low_memory=False,
            )

    # Validate real BTS records.
    result = validate_bts_data(df)

    assert result["raw_records"] == 538837
    assert result["six_airport_records"] == 115330
    assert result["eligible_records"] == 112548
