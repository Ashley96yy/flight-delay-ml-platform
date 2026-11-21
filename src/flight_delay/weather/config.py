"""Configuration for HRRR weather acquisition."""

from dataclasses import dataclass
from typing import Final

WEATHER_CONTRACT_VERSION: Final = "v1"

PREDICTION_HORIZON_HOURS: Final = 24

WEATHER_AVAILABILITY_LAG_HOURS: Final = 3

EXTENDED_CYCLE_HOURS: Final = (0, 6, 12, 18)

SUPPORTED_FORECAST_LEADS: Final = tuple(range(27, 33))

GRID_MATCH_TOLERANCE_M: Final = 2.0

ZARR_BUCKET: Final = "hrrrzarr"

GRIB_BASE_URL: Final = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class Airport:
    code: str
    latitude: float
    longitude: float


AIRPORTS: Final = (
    Airport("ATL", 33.6367, -84.4281),
    Airport("DFW", 32.8968, -97.0380),
    Airport("DEN", 39.8561, -104.6737),
    Airport("ORD", 41.9742, -87.9073),
    Airport("LAX", 33.9425, -118.4081),
    Airport("SFO", 37.6213, -122.3790),
)

ORIGIN_AIRPORTS: Final = tuple(airport.code for airport in AIRPORTS)


@dataclass(frozen=True)
class WeatherVariable:
    feature: str
    zarr_level: str
    zarr_variable: str
    grib_short_name: str
    grib_units: str
    grib_type_of_level: str
    grib_step_type: str
    required: bool = True


WEATHER_VARIABLES: Final = (
    WeatherVariable(
        "temperature_2m",
        "2m_above_ground",
        "TMP",
        "2t",
        "K",
        "heightAboveGround",
        "instant",
    ),
    WeatherVariable(
        "u_wind_10m",
        "10m_above_ground",
        "UGRD",
        "10u",
        "m s**-1",
        "heightAboveGround",
        "instant",
    ),
    WeatherVariable(
        "v_wind_10m",
        "10m_above_ground",
        "VGRD",
        "10v",
        "m s**-1",
        "heightAboveGround",
        "instant",
    ),
    WeatherVariable(
        "wind_gust",
        "surface",
        "GUST",
        "gust",
        "m s**-1",
        "surface",
        "instant",
    ),
    WeatherVariable(
        "visibility",
        "surface",
        "VIS",
        "vis",
        "m",
        "surface",
        "instant",
    ),
    WeatherVariable(
        "precipitation_1h",
        "surface",
        "APCP_1hr_acc_fcst",
        "tp",
        "kg m**-2",
        "surface",
        "accum",
    ),
    WeatherVariable(
        "surface_cape",
        "surface",
        "CAPE",
        "cape",
        "J kg**-1",
        "surface",
        "instant",
    ),
    WeatherVariable(
        "composite_reflectivity",
        "entire_atmosphere",
        "REFC",
        "refc",
        "dB",
        "atmosphere",
        "instant",
    ),
    WeatherVariable(
        "cloud_ceiling_raw",
        "cloud_ceiling",
        "HGT",
        "gh",
        "gpm",
        "cloudCeiling",
        "instant",
        required=False,
    ),
)


VARIABLE_BY_FEATURE: Final = {
    variable.feature: variable for variable in WEATHER_VARIABLES
}

RAW_WEATHER_COLUMNS: Final = tuple(variable.feature for variable in WEATHER_VARIABLES)

CORE_RAW_WEATHER_COLUMNS: Final = tuple(
    variable.feature for variable in WEATHER_VARIABLES if variable.required
)

MODEL_WEATHER_COLUMNS: Final = (
    "origin_temperature_c",
    "origin_wind_speed_10m",
    "origin_wind_gust",
    "origin_visibility_km",
    "origin_precipitation_1h",
    "origin_surface_cape",
    "origin_composite_reflectivity",
)
