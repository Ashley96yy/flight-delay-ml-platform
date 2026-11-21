import io
import zipfile
from urllib.error import URLError

import pytest

from flight_delay.data import download_bts as downloader


def make_zip_bytes(files=None):
    """Create a small ZIP archive in memory."""
    if files is None:
        files = {"flights.csv": "FlightDate\n2023-01-01\n"}

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for name, content in files.items():
            zf.writestr(name, content)

    return buffer.getvalue()


def test_valid_filename():
    """Valid year and month produce the expected BTS filename."""
    filename = downloader.build_bts_filename(2023, 1)

    assert filename == (
        "On_Time_Reporting_Carrier_On_Time_Performance_"
        "1987_present_2023_1.zip"
    )


@pytest.mark.parametrize(
    "year, month, error",
    [
        (1986, 1, "Year"),
        (2023, 0, "Month"),
        (2023, 13, "Month"),
    ],
)
def test_invalid_date_parameters(year, month, error):
    """Invalid year or month must be rejected."""
    with pytest.raises(ValueError, match=error):
        downloader.build_bts_filename(year, month)


def test_invalid_zip_archive(tmp_path):
    """A non-ZIP file must be rejected."""
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(ValueError, match="Invalid ZIP"):
        downloader.validate_zip(path)


@pytest.mark.parametrize(
    "files",
    [
        {"readme.txt": "hello"},
        {"a.csv": "x\n1", "b.csv": "x\n2"},
        {"empty.csv": ""},
    ],
)
def test_invalid_zip_contents(tmp_path, files):
    """ZIP must contain exactly one nonempty CSV."""
    path = tmp_path / "invalid.zip"
    path.write_bytes(make_zip_bytes(files))

    with pytest.raises(ValueError):
        downloader.validate_zip(path)


def test_valid_cache_skips_network(tmp_path, monkeypatch):
    """Existing valid ZIP should not trigger a download."""
    filename = downloader.build_bts_filename(2023, 1)
    path = tmp_path / filename
    path.write_bytes(make_zip_bytes())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Network should not be called")

    monkeypatch.setattr(
        downloader, "urlopen", fail_if_called
    )

    result = downloader.download_bts_month(
        2023, 1, raw_dir=tmp_path
    )

    assert result == path
    assert result.read_bytes() == make_zip_bytes()


def test_download_new_file(tmp_path, monkeypatch):
    """A valid network response should be saved as ZIP."""
    content = make_zip_bytes()

    def fake_urlopen(request, timeout):
        return io.BytesIO(content)

    monkeypatch.setattr(
        downloader, "urlopen", fake_urlopen
    )

    result = downloader.download_bts_month(
        2023, 1, raw_dir=tmp_path
    )

    assert result.is_file()
    assert result.read_bytes() == content
    downloader.validate_zip(result)


def test_invalid_cache_redownloads(tmp_path, monkeypatch):
    """A corrupted cached file should be replaced."""
    filename = downloader.build_bts_filename(2023, 1)
    path = tmp_path / filename
    path.write_bytes(b"corrupted cache")

    content = make_zip_bytes()

    def fake_urlopen(request, timeout):
        return io.BytesIO(content)

    monkeypatch.setattr(
        downloader, "urlopen", fake_urlopen
    )

    result = downloader.download_bts_month(
        2023, 1, raw_dir=tmp_path
    )

    assert result.read_bytes() == content
    downloader.validate_zip(result)


def test_network_failure_cleanup(tmp_path, monkeypatch):
    """Network failure should not leave partial downloads."""
    def fake_urlopen(request, timeout):
        raise URLError("Network unavailable")

    monkeypatch.setattr(
        downloader, "urlopen", fake_urlopen
    )

    with pytest.raises(URLError):
        downloader.download_bts_month(
            2023, 1, raw_dir=tmp_path
        )

    assert not list(tmp_path.glob("*.zip"))
    assert not list(tmp_path.glob("*.part"))


def test_invalid_download_cleanup(tmp_path, monkeypatch):
    """Invalid downloaded content must not be published."""
    def fake_urlopen(request, timeout):
        return io.BytesIO(b"invalid zip data")

    monkeypatch.setattr(
        downloader, "urlopen", fake_urlopen
    )

    with pytest.raises(ValueError, match="Invalid ZIP"):
        downloader.download_bts_month(
            2023, 1, raw_dir=tmp_path
        )

    assert not list(tmp_path.glob("*.zip"))
    assert not list(tmp_path.glob("*.part"))
