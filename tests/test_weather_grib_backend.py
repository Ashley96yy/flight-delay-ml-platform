"""Offline tests for HRRR GRIB inventory and range requests."""

import pytest

from flight_delay.weather.grib_backend import (
    GRIBBackendError,
    GRIBDownloadError,
    GRIBInventoryError,
    build_grib_reference,
    build_weather_patterns,
    download_grib_range,
    parse_grib_inventory,
    select_weather_messages,
)


def test_reference_urls():
    ref = build_grib_reference("2024-07-14T06:00:00Z", 30)

    assert ref.grib_url.endswith("hrrr.20240714/conus/hrrr.t06z.wrfsfcf30.grib2")

    assert ref.index_url == ref.grib_url + ".idx"


@pytest.mark.parametrize(
    "run,lead",
    [
        ("2024-07-14T06:00:00", 30),
        ("2024-07-14T07:00:00Z", 30),
        ("2024-07-14T06:30:00Z", 30),
        ("2024-07-14T06:00:00Z", 33),
    ],
)
def test_invalid_reference(run, lead):
    with pytest.raises(GRIBBackendError):
        build_grib_reference(run, lead)


def test_precipitation_pattern():
    patterns = build_weather_patterns(30)

    assert patterns["precipitation_1h"] == (":APCP:surface:29-30 hour acc fcst:")


def example_inventory():
    patterns = build_weather_patterns(30)

    lines = [
        f"{i}:{(i - 1) * 100}:d=2024071406{pattern}"
        for i, pattern in enumerate(
            [*patterns.values(), ":DUMMY:surface:30 hour fcst:"],
            start=1,
        )
    ]

    return parse_grib_inventory(lines)


def test_inventory_ranges():
    inventory = example_inventory()

    assert len(inventory) == 10
    assert inventory.iloc[0]["start_byte"] == 0
    assert inventory.iloc[0]["end_byte"] == 99


def test_select_nine_variables():
    selected = select_weather_messages(example_inventory(), 30)

    assert len(selected) == 9
    assert selected["feature"].is_unique
    assert selected["size_bytes"].eq(100).all()


def test_missing_variable_rejected():
    inventory = example_inventory().iloc[1:].copy()

    with pytest.raises(
        GRIBInventoryError,
        match="temperature_2m",
    ):
        select_weather_messages(inventory, 30)


def test_duplicate_variable_rejected():
    inventory = example_inventory()
    inventory.loc[1, "inventory"] = inventory.loc[0, "inventory"]

    with pytest.raises(
        GRIBInventoryError,
        match="temperature_2m",
    ):
        select_weather_messages(inventory, 30)


def test_duplicate_offsets_rejected():
    with pytest.raises(GRIBInventoryError):
        parse_grib_inventory(
            [
                "1:0:d=2024071406:TMP:",
                "2:0:d=2024071406:UGRD:",
            ]
        )


def test_nonmonotonic_offsets_rejected():
    with pytest.raises(GRIBInventoryError):
        parse_grib_inventory(
            [
                "1:100:d=2024071406:TMP:",
                "2:50:d=2024071406:UGRD:",
            ]
        )


class FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status=206,
        content_range="bytes 100-111/1000",
    ):
        self.payload = payload
        self.status = status
        self.headers = {"Content-Range": content_range}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def make_opener(
    payload,
    *,
    status=206,
    content_range="bytes 100-111/1000",
):
    def opener(request, timeout):
        assert timeout == 60
        assert request.get_header("Range") == ("bytes=100-111")

        return FakeResponse(
            payload,
            status=status,
            content_range=content_range,
        )

    return opener


VALID_PAYLOAD = b"GRIB12347777"


def test_valid_range_download():
    payload = download_grib_range(
        "https://example.com/weather.grib2",
        100,
        111,
        opener=make_opener(VALID_PAYLOAD),
    )

    assert payload == VALID_PAYLOAD


def test_non_partial_response_rejected():
    with pytest.raises(
        GRIBDownloadError,
        match="HTTP 206",
    ):
        download_grib_range(
            "https://example.com/weather.grib2",
            100,
            111,
            opener=make_opener(
                VALID_PAYLOAD,
                status=200,
            ),
        )


def test_wrong_content_range_rejected():
    with pytest.raises(
        GRIBDownloadError,
        match="Content-Range",
    ):
        download_grib_range(
            "https://example.com/weather.grib2",
            100,
            111,
            opener=make_opener(
                VALID_PAYLOAD,
                content_range="bytes 0-11/1000",
            ),
        )


def test_truncated_payload_rejected():
    with pytest.raises(
        GRIBDownloadError,
        match="payload size mismatch",
    ):
        download_grib_range(
            "https://example.com/weather.grib2",
            100,
            111,
            opener=make_opener(b"GRIB7777"),
        )


def test_invalid_grib_markers():
    with pytest.raises(
        GRIBDownloadError,
        match="GRIB message markers",
    ):
        download_grib_range(
            "https://example.com/weather.grib2",
            100,
            111,
            opener=make_opener(b"XXXX1234YYYY"),
        )
