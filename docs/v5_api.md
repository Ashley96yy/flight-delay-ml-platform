# V5: API-first, small-data-first, evidence-driven

Research checked: 2026-11-20. Stage 1 is request planning and client safeguards.
There is no live weather acquisition, measured API coverage, weather model fit,
or V3/V5 score in this stage. URLs in plans are candidates, not validated API
responses. `examples/weather_api_queries.json` contains invented schedule times
for exercising planning only, not actual flight samples or weather fixtures.

## Provider decision

| Product | Documented history / semantics | Suitability for a 2023–2025 T-24h study | Stage 1 action |
| --- | --- | --- | --- |
| Open-Meteo Historical Weather | ERA5/ERA5-Land reanalysis and analysis products | Reject as forecast inputs | No adapter |
| Open-Meteo Historical Forecast | Stitches initial forecast hours into continuous series | Cannot reconstruct the forecast available T-24h from the date alone | No adapter |
| Open-Meteo Previous Runs | Fixed 1–7 day offsets; mostly from Jan 2024; GFS 2m temperature from Mar 2021, JMA from 2018 | Partial candidate; a day offset is not proof of publication before a particular flight cutoff. Do not infer 2023 coverage for wind/precipitation from temperature coverage | Research only |
| Open-Meteo Single Runs | Explicit initialization, IFS from 2024-03-14, other models from 2026-04-02. Documentation explicitly mentions IFS Cycle 49R1 hindcasts | 2023 outside documented IFS archive; older records require operational-versus-hindcast confirmation. Initialization is not public release | Free endpoint request adapter; all real plans blocked |
| Visual Crossing Historical Forecast | Historical forecasts from 2020-01-01; byrun/bytarget modes; midnight UTC daily stored forecasts. Historical Forecast Addon required | Best documented API candidate for the full calendar span. Exact hourly fields, model-specific history, original run identity, and publication timing still need verification | By-run request adapter; paid calls prohibited |
| Weatherbit historical forecasts | Custom paid cold-archive reports; hourly 48h forecasts; limited city points and initialization options | Possible alternative after checking airport locations and historical dates; not a ready low-cost point API | No order or call |

Sources (provider documentation, not empirical coverage checks):

1. [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api)
2. [Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api)
3. [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api)
4. [Single Runs](https://open-meteo.com/en/docs/single-runs-api)
5. [Open-Meteo pricing and weighted call accounting](https://open-meteo.com/en/pricing)
6. [Visual Crossing history, dates, midnight snapshots and billing](https://www.visualcrossing.com/resources/documentation/weather-data/how-to-query-weather-forecasts-from-the-past-historical-forecasts/)
7. [Visual Crossing byrun parameters and required addon](https://www.visualcrossing.com/resources/documentation/weather-api/historical-forecast-api/)
8. [Weatherbit custom historical forecast reports](https://help.weatherbit.io/faq/how-do-i-place-an-order-for-historical-forecast-data/)

**Decision:** do not claim that a free, fully eligible 2023–2025 multi-variable
API has been established. API-first is a preference among sources that pass
the evidence gate. Keep HRRR as the existing alternative, with its original
availability assumptions disclosed. No silent substitution of observations,
seamless historical weather, retrospectively recomputed forecasts, or a different
provider/model for unavailable years.

## Compatibility boundary

Existing `config.py`, `contracts.py`, planner, cache, Zarr/GRIB fallback, feature
builder, flight join and leakage audit are unchanged. The current v1 contract is
specifically HRRR: ATL/DFW/DEN/ORD/LAX/SFO, six-airport × nine-variable hourly
batches, 00/06/12/18 cycles, F27–F32, and assumed initialization +3h availability.
Core fields include composite reflectivity; API providers need not offer it.
Broadening `SUPPORTED_BACKENDS` alone would incorrectly certify new sources.

`ForecastAPIAdapter.plan(ForecastQuery)` is the replaceable provider boundary.
Queries use current airport coordinates and scheduled departure only, derive
the exact T-24h cutoff, and reject naive timestamps. API plans use their own run
selection; IFS's candidate +6h lag does not alter HRRR's +3h policy. Visual
Crossing's stored day is explicitly marked as a snapshot basis date, not a
verified initialization or availability timestamp. Future dates may be planned;
planning never asserts archive presence.

Stage 1 deliberately stops at **quarantined raw JSON**. These four candidate
features are a new explicit API subset: Celsius temperature, 10m wind speed,
gust speed and hourly precipitation. Confirm provider-specific units and
accumulation windows before mapping them: Visual Crossing metric wind speed
must not be assumed to be m/s. Never zero-fill absent CAPE, visibility, ceiling,
or reflectivity to satisfy HRRR's schema.

Before normalized API rows can enter a model, add a separate versioned contract
containing provider/product/model/model-version, original initialization,
valid time, lead, publication time or documented conservative bound, evidence
reference and verification status, units/accumulation interval, actual location
and interpolation semantics, retrieval timestamp, and raw response SHA256.
Require an original operational forecast, publication <= exact flight cutoff,
and exact departure-hour identity. Retrieval today does not establish historical
availability. A hindcast warning is a blocking issue until resolved with the
provider, even if its endpoint supports `run=`.

The eventual normalized provider interface should produce that common schema;
an HRRR bridge will validate legacy records first and map them while preserving
their source and assumed availability. Keep caches separate by contract,
provider, model/version, run, location, variable selection, units and evidence
policy. Existing HRRR cache keys and objects remain valid and untouched.

For a future pilot: try an evidence-approved API source; if unavailable, read
validated existing HRRR cache only and record the fallback reason per identity.
Cache misses become reported missingness. Do not automatically invoke the
existing downloader or batch runner. New bounded HRRR acquisition would be a
separate, explicitly configured stage. Policy, quota, paid-access, or provenance
failures must remain visible; they must not trigger a large fallback download.

## Client and dry run

```bash
PYTHONPATH=src python -m flight_delay.weather.api_dry_run \
  --input examples/weather_api_queries.json --provider open-meteo
PYTHONPATH=src python -m flight_delay.weather.api_dry_run \
  --input examples/weather_api_queries.json --provider visual-crossing
```

The CLI has no live or paid option and accepts at most 100 label-free queries.
It preserves input-to-request mapping and deduplicates common airport/run/field
payloads. A plan exceeding the request cap exits 2; an in-budget blocked plan
exits 0 with `ready_for_live=false` and explicit blockers. That is planning
success, not source approval. Coverage, billing units and model metrics are null.

The client defaults to no network, ten attempts total, one second between
attempts, ten-second timeout per HTTP operation and one-megabyte maximum response
per attempt. Timeout is not a total wall-clock deadline. Limits apply to a
single shared client instance; they are not an account-wide or durable quota.
No automatic retry, redirect or sleep is performed. A 429/timeout consumes one
attempt and raises an error. Paid execution is unconditionally unavailable in
this stage. The only permitted network host/product is the free Single Runs
endpoint, and unresolved source evidence blocks it before transport. No keys
are read, stored, or logged. Current real adapters all have evidence blockers.

HTTP attempts differ from provider billing units, which can be weighted by
variables, locations, dates and products. Do not label the local request count
as cost. Phase 2 needs a reviewed response sample, provider-specific decoder,
archival timing evidence, separate cache namespace and persistent pilot budget
before enabling bounded live acquisition or fitting a weather model.
