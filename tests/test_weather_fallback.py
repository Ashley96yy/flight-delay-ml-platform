"""Offline fault-injection tests for Zarr-to-GRIB fallback."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.config import (
    AIRPORTS,
    WEATHER_VARIABLES,
)
from flight_delay.weather.contracts import WeatherContractError
from flight_delay.weather.fallback import (
    WeatherAcquisitionError,
    acquire_weather_cycle,
)
from flight_delay.weather.grib_backend import GRIBDownloadError
from flight_delay.weather.grib_hour import GRIBHourResult
from flight_delay.weather.zarr_backend import (
    ZarrChunkError,
    ZarrMetadataError,
)
from flight_delay.weather.zarr_cycle import (
    ZarrCoordinateError,
    ZarrCycleResult,
)

RUN = pd.Timestamp("2024-07-14T06:00:00Z")


def make_weather(run_at, leads, backend, reason=None):
    """Build a complete synthetic backend output."""

    records = []

    for lead in leads:
        for airport in AIRPORTS:
            for variable in WEATHER_VARIABLES:
                missing = not variable.required

                records.append(
                    {
                        "Origin": airport.code,
                        "feature": variable.feature,
                        "forecast_run_at_utc": run_at,
                        "forecast_valid_at_utc": (run_at + pd.Timedelta(hours=lead)),
                        "forecast_hour": lead,
                        "assumed_available_at_utc": (run_at + pd.Timedelta(hours=3)),
                        "value": np.nan if missing else 3.5,
                        "raw_value": (-9999.0 if missing else 3.5),
                        "value_status": ("fill_value" if missing else "valid"),
                        "grid_latitude": airport.latitude,
                        "grid_longitude": airport.longitude,
                        "backend": backend,
                        "storage_dtype": ("float32" if backend == "zarr" else None),
                        "units": variable.grib_units,
                        "source_units": variable.grib_units,
                        "fallback_reason": reason,
                    }
                )

    return pd.DataFrame(records)


def make_zarr_result(run_at, leads):
    return ZarrCycleResult(
        weather=make_weather(run_at, leads, "zarr"),
        chunk_audit=pd.DataFrame([{"feature": "temperature_2m", "chunk_key": "0.2.8"}]),
    )


def make_grib_extractor(calls, fail_at=None):
    def extractor(reference, *, fallback_reason):
        calls.append(reference.forecast_hour)

        if reference.forecast_hour == fail_at:
            raise GRIBDownloadError("Simulated GRIB download failure")

        return GRIBHourResult(
            weather=make_weather(
                reference.run_at,
                [reference.forecast_hour],
                "grib",
                reason=fallback_reason,
            ),
            request_audit=pd.DataFrame(
                [
                    {
                        "feature": "temperature_2m",
                        "downloaded_bytes": 100,
                    }
                ]
            ),
        )

    return extractor


def test_zarr_success_does_not_call_grib():
    grib_calls = []

    result = acquire_weather_cycle(
        None,
        RUN,
        [27, 30],
        zarr_extractor=lambda fs, run, leads: make_zarr_result(run, leads),
        grib_extractor=make_grib_extractor(grib_calls),
    )

    assert result.backend == "zarr"
    assert result.fallback_reason is None
    assert len(result.weather) == 108
    assert grib_calls == []

    assert result.attempt_audit["status"].tolist() == ["success"]


def test_zarr_chunk_failure_falls_back_to_grib():
    grib_calls = []

    def broken_zarr(fs, run, leads):
        raise ZarrChunkError("Missing chunk 0.2.8")

    result = acquire_weather_cycle(
        None,
        RUN,
        [30, 27],
        zarr_extractor=broken_zarr,
        grib_extractor=make_grib_extractor(grib_calls),
    )

    assert result.backend == "grib"
    assert result.fallback_reason == "zarr_chunk_error"
    assert len(result.weather) == 108

    assert grib_calls == [27, 30]

    assert result.weather["fallback_reason"].eq("zarr_chunk_error").all()

    assert result.attempt_audit["status"].tolist() == [
        "failed",
        "success",
        "success",
    ]


@pytest.mark.parametrize(
    "error,reason",
    [
        (
            ZarrMetadataError("Invalid metadata"),
            "zarr_metadata_error",
        ),
        (
            ZarrCoordinateError("Invalid coordinates"),
            "zarr_coordinate_error",
        ),
        (
            WeatherContractError("Missing core weather"),
            "zarr_contract_error",
        ),
        (
            OSError("Network failure"),
            "zarr_io_error",
        ),
    ],
)
def test_recoverable_zarr_errors(error, reason):
    grib_calls = []

    def broken_zarr(fs, run, leads):
        raise error

    result = acquire_weather_cycle(
        None,
        RUN,
        [30],
        zarr_extractor=broken_zarr,
        grib_extractor=make_grib_extractor(grib_calls),
    )

    assert result.backend == "grib"
    assert result.fallback_reason == reason
    assert grib_calls == [30]


def test_invalid_zarr_output_triggers_fallback():
    grib_calls = []

    def invalid_zarr(fs, run, leads):
        result = make_zarr_result(run, leads)

        result.weather.loc[0, "value"] = np.nan
        result.weather.loc[0, "value_status"] = "source_missing"

        return result

    result = acquire_weather_cycle(
        None,
        RUN,
        [30],
        zarr_extractor=invalid_zarr,
        grib_extractor=make_grib_extractor(grib_calls),
    )

    assert result.backend == "grib"
    assert result.fallback_reason == "zarr_contract_error"


def test_grib_failure_raises_composite_error():
    def broken_zarr(fs, run, leads):
        raise ZarrChunkError("Missing Zarr chunk")

    with pytest.raises(
        WeatherAcquisitionError,
        match="GRIB fallback also failed",
    ) as info:
        acquire_weather_cycle(
            None,
            RUN,
            [27, 30],
            zarr_extractor=broken_zarr,
            grib_extractor=make_grib_extractor(
                [],
                fail_at=30,
            ),
        )

    audit = info.value.attempt_audit

    assert audit["status"].tolist() == [
        "failed",
        "success",
        "failed",
    ]

    assert isinstance(
        info.value.__cause__,
        GRIBDownloadError,
    )


def test_programming_errors_are_not_hidden():
    grib_calls = []

    def buggy_zarr(fs, run, leads):
        raise RuntimeError("Programming bug")

    with pytest.raises(
        RuntimeError,
        match="Programming bug",
    ):
        acquire_weather_cycle(
            None,
            RUN,
            [30],
            zarr_extractor=buggy_zarr,
            grib_extractor=make_grib_extractor(grib_calls),
        )

    assert grib_calls == []


@pytest.mark.parametrize(
    "leads",
    [
        [],
        [30, 30],
        [26],
        [33],
        [30.5],
    ],
)
def test_invalid_leads_rejected_before_acquisition(leads):
    with pytest.raises(
        WeatherAcquisitionError,
        match="Invalid or duplicate forecast leads",
    ):
        acquire_weather_cycle(
            None,
            RUN,
            leads,
        )


def test_invalid_run_rejected_before_acquisition():
    with pytest.raises(
        WeatherAcquisitionError,
        match="Invalid forecast initialization",
    ):
        acquire_weather_cycle(
            None,
            "2024-07-14T07:00:00Z",
            [30],
        )


def test_fallback_preserves_hourly_identity():
    def broken_zarr(fs, run, leads):
        raise ZarrChunkError("Missing chunk")

    result = acquire_weather_cycle(
        None,
        RUN,
        [27, 28, 29, 30, 31, 32],
        zarr_extractor=broken_zarr,
        grib_extractor=make_grib_extractor([]),
    )

    assert len(result.weather) == 324

    assert result.weather.groupby("forecast_hour").size().eq(54).all()

    assert set(result.weather["forecast_hour"]) == {27, 28, 29, 30, 31, 32}

    assert not result.weather.duplicated(
        subset=[
            "Origin",
            "feature",
            "forecast_run_at_utc",
            "forecast_valid_at_utc",
            "forecast_hour",
        ]
    ).any()
