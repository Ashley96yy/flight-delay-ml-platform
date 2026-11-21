import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

BTS_BASE_URL = "https://transtats.bts.gov/PREZIP/"


def build_bts_filename(year: int, month: int) -> str:
    """Build the official monthly BTS ZIP filename."""

    if year < 1987:
        raise ValueError("Year must be 1987 or later.")

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    return (
        "On_Time_Reporting_Carrier_On_Time_Performance_"
        f"1987_present_{year}_{month}.zip"
    )


def validate_zip(path: Path) -> None:
    """Verify that a ZIP is readable and contains one CSV."""

    if not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid ZIP file: {path}")

    with zipfile.ZipFile(path) as zf:
        csv_files = [
            info for info in zf.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one CSV, found {len(csv_files)}"
            )

        if csv_files[0].file_size == 0:
            raise ValueError("CSV inside ZIP is empty.")

        bad_member = zf.testzip()
        if bad_member is not None:
            raise ValueError(
                f"Corrupted ZIP member: {bad_member}"
            )


def download_bts_month(
    year: int,
    month: int,
    raw_dir: str | Path = "data/raw",
) -> Path:
    """Download and validate one monthly BTS ZIP."""

    filename = build_bts_filename(year, month)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / filename

    # Reuse an existing valid file.
    if output_path.exists():
        try:
            validate_zip(output_path)
            print(f"Using cached file: {output_path}")
            return output_path
        except ValueError:
            print("Cached ZIP is invalid. Re-downloading.")

    url = f"{BTS_BASE_URL}{filename}"

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    # Create a temporary file in the destination directory.
    with tempfile.NamedTemporaryFile(
        dir=raw_dir,
        suffix=".part",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        print(f"Downloading: {url}")

        with urlopen(request, timeout=90) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        validate_zip(temp_path)

        # Publish only after successful validation.
        os.replace(temp_path, output_path)

    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Download completed: {output_path}")

    return output_path
