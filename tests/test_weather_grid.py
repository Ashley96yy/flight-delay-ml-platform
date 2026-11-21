"""Tests for HRRR spatial grid and chunk mapping."""

import numpy as np
import pytest

from flight_delay.weather.grid import (
    WeatherGridError,
    build_airport_grid_mapping,
    locate_chunk,
    normalize_longitude,
    validate_hrrr_coordinates,
)


@pytest.fixture
def hrrr_coordinates():
    """Reproduce the previously validated HRRR coordinates."""

    x = -2697520.1425219304 + np.arange(1799) * 3000.0

    y = -1587306.1525566636 + np.arange(1059) * 3000.0

    return x, y


def test_valid_hrrr_coordinates(hrrr_coordinates):

    x, y = hrrr_coordinates

    validate_hrrr_coordinates(x, y)


def test_airport_coverage(hrrr_coordinates):

    mapping = build_airport_grid_mapping(*hrrr_coordinates)

    assert len(mapping) == 6

    assert mapping["airport"].is_unique

    assert set(mapping["airport"]) == {
        "ATL",
        "DFW",
        "DEN",
        "ORD",
        "LAX",
        "SFO",
    }


@pytest.mark.parametrize(
    "airport,expected_x,expected_y",
    [
        ("ATL", 1303, 377),
        ("DFW", 914, 321),
        ("DEN", 695, 587),
        ("ORD", 1164, 672),
        ("LAX", 260, 433),
        ("SFO", 178, 595),
    ],
)
def test_airport_grid_indices(
    hrrr_coordinates,
    airport,
    expected_x,
    expected_y,
):

    mapping = build_airport_grid_mapping(*hrrr_coordinates).set_index("airport")

    row = mapping.loc[airport]

    assert row["x_index"] == expected_x
    assert row["y_index"] == expected_y


def test_airport_grid_distance(hrrr_coordinates):

    mapping = build_airport_grid_mapping(*hrrr_coordinates)

    assert mapping["geographic_distance_km"].ge(0).all()

    assert mapping["geographic_distance_km"].max() == pytest.approx(
        1.730590,
        abs=0.001,
    )


def test_grid_longitude_normalized(hrrr_coordinates):

    mapping = build_airport_grid_mapping(*hrrr_coordinates)

    assert mapping["grid_longitude"].ge(-180).all()

    assert mapping["grid_longitude"].lt(180).all()


def test_longitude_normalization():

    result = normalize_longitude([275.0, 240.0, 180.0, -120.0])

    np.testing.assert_allclose(
        result,
        [-85.0, -120.0, -180.0, -120.0],
    )


def test_invalid_x_shape(hrrr_coordinates):

    x, y = hrrr_coordinates

    with pytest.raises(
        WeatherGridError,
        match="Unexpected X coordinate shape",
    ):
        validate_hrrr_coordinates(x[:-1], y)


def test_invalid_y_shape(hrrr_coordinates):

    x, y = hrrr_coordinates

    with pytest.raises(
        WeatherGridError,
        match="Unexpected Y coordinate shape",
    ):
        validate_hrrr_coordinates(x, y[:-1])


def test_nonfinite_coordinates(hrrr_coordinates):

    x, y = hrrr_coordinates

    x[0] = np.nan

    with pytest.raises(
        WeatherGridError,
        match="Nonfinite X coordinates",
    ):
        validate_hrrr_coordinates(x, y)


def test_shifted_grid_rejected(hrrr_coordinates):

    x, y = hrrr_coordinates

    with pytest.raises(
        WeatherGridError,
        match="X coordinates do not match",
    ):
        validate_hrrr_coordinates(
            x + 3000,
            y,
        )


def test_reversed_grid_rejected(hrrr_coordinates):

    x, y = hrrr_coordinates

    with pytest.raises(
        WeatherGridError,
        match="Y coordinates do not match",
    ):
        validate_hrrr_coordinates(
            x,
            y[::-1],
        )


def test_irregular_grid_rejected(hrrr_coordinates):

    x, y = hrrr_coordinates

    x[500] += 100

    with pytest.raises(
        WeatherGridError,
        match="X coordinates do not match",
    ):
        validate_hrrr_coordinates(x, y)


@pytest.mark.parametrize(
    "airport,expected_key,expected_local",
    [
        ("ATL", "0.2.8", (29, 77, 103)),
        ("DFW", "0.2.6", (29, 21, 14)),
        ("DEN", "0.3.4", (29, 137, 95)),
        ("ORD", "0.4.7", (29, 72, 114)),
        ("LAX", "0.2.1", (29, 133, 110)),
        ("SFO", "0.3.1", (29, 145, 28)),
    ],
)
def test_airport_chunk_location(
    hrrr_coordinates,
    airport,
    expected_key,
    expected_local,
):

    mapping = build_airport_grid_mapping(*hrrr_coordinates).set_index("airport")

    row = mapping.loc[airport]

    location = locate_chunk(
        array_index=29,
        y_index=int(row["y_index"]),
        x_index=int(row["x_index"]),
        array_shape=(48, 1059, 1799),
        chunk_shape=(48, 150, 150),
    )

    assert location.chunk_key == expected_key

    assert location.local_indices == expected_local


def test_multiple_leads_share_chunk():

    locations = [
        locate_chunk(
            array_index=lead - 1,
            y_index=377,
            x_index=1303,
            array_shape=(48, 1059, 1799),
            chunk_shape=(48, 150, 150),
        )
        for lead in range(27, 33)
    ]

    assert {location.chunk_key for location in locations} == {"0.2.8"}

    assert [location.local_indices[0] for location in locations] == [
        26,
        27,
        28,
        29,
        30,
        31,
    ]


def test_multiple_time_chunks_supported():

    location = locate_chunk(
        array_index=29,
        y_index=377,
        x_index=1303,
        array_shape=(48, 1059, 1799),
        chunk_shape=(12, 150, 150),
    )

    assert location.chunk_key == "2.2.8"

    assert location.local_indices == (5, 77, 103)


def test_slash_dimension_separator():

    location = locate_chunk(
        array_index=29,
        y_index=377,
        x_index=1303,
        array_shape=(48, 1059, 1799),
        chunk_shape=(48, 150, 150),
        dimension_separator="/",
    )

    assert location.chunk_key == "0/2/8"


@pytest.mark.parametrize(
    "indices",
    [
        (-1, 377, 1303),
        (48, 377, 1303),
        (29, 1059, 1303),
        (29, 377, 1799),
    ],
)
def test_out_of_bounds_index(indices):

    with pytest.raises(
        WeatherGridError,
        match="Grid index out of bounds",
    ):
        locate_chunk(
            array_index=indices[0],
            y_index=indices[1],
            x_index=indices[2],
            array_shape=(48, 1059, 1799),
            chunk_shape=(48, 150, 150),
        )


def test_invalid_chunk_shape():

    with pytest.raises(
        WeatherGridError,
        match="positive integers",
    ):
        locate_chunk(
            array_index=29,
            y_index=377,
            x_index=1303,
            array_shape=(48, 1059, 1799),
            chunk_shape=(48, 0, 150),
        )


def test_invalid_dimension_separator():

    with pytest.raises(
        WeatherGridError,
        match="Unsupported dimension separator",
    ):
        locate_chunk(
            array_index=29,
            y_index=377,
            x_index=1303,
            array_shape=(48, 1059, 1799),
            chunk_shape=(48, 150, 150),
            dimension_separator="-",
        )


def test_fractional_index_rejected():

    with pytest.raises(
        WeatherGridError,
        match="Grid indices must be integers",
    ):
        locate_chunk(
            array_index=29.5,
            y_index=377,
            x_index=1303,
            array_shape=(48, 1059, 1799),
            chunk_shape=(48, 150, 150),
        )
