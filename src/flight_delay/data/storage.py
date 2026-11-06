import os
import tempfile
from pathlib import Path

import pandas as pd


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a DataFrame as Parquet using an atomic file replacement."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix != ".parquet":
        raise ValueError("Output file must have a .parquet extension")

    # Write to a temporary file in the same directory.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f"{output_path.stem}_",
        suffix=".parquet",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        df.to_parquet(
            temp_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        # Replace the final file only after writing succeeds.
        os.replace(temp_path, output_path)

    finally:
        temp_path.unlink(missing_ok=True)

    return output_path


def load_parquet(
    input_path: str | Path,
) -> pd.DataFrame:
    """Load a Parquet dataset into a Pandas DataFrame."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {input_path}"
        )

    return pd.read_parquet(
        input_path,
        engine="pyarrow",
    )
