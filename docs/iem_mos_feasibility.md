# IEM MOS feasibility — V5 research stage 2

Checked 2026-11-20 America/Los_Angeles. Raw receipts use UTC (2026-11-21).

**Decision: historical archive acquisition is feasible and free; strict T−24h
model use is NOT yet established.** The pilot found all 72 requested
station/run/model cells, but none has independently verified historical
publication time. No V5 model was trained, no score improvement is claimed, and
the MOS join remains blocked for the acquired rows.

## What the source establishes

IEM describes a realtime archive of NWS operational station forecast guidance,
not an observation endpoint or an explicitly reconstructed hindcast product.
Its published table lists GFS from December 2003 and NBS from November 2018.
For our years, NBS retained cycles are 01/07/13/19 UTC. Current retained cycles
changed in May 2026; do not project today's 00/06/12/18 policy back into 2023.
The pilot uses exact historical cycles and verifies each returned station,
model, initialization and valid time against its request.

This establishes archive existence and run identity for the sampled records,
not immutable original versions or original public release time. The API has
no per-row issue, first-seen, distribution or publication timestamp and does
not expose the exact model version. A nominal model cycle is not a release
timestamp. IEM's general statement that products are processed in realtime
does not establish a dated historical release bound for each record.

Every parsed row therefore has `published_at_utc=null`,
`availability_status=availability_unverified`, `model_ready=false`. Retrieval
today and a plausible assumed lag cannot promote it to leakage-approved data.

## Access, price, license and durability

The public API is `GET https://mesonet.agron.iastate.edu/api/1/mos.json`, with
repeated four-character `station` parameters, explicit `model` and UTC `runtime`.
The provider-maintained handler permits at most six stations per request.
Without explicit runtime it can select a recent/latest run; this project
forbids that mode. The raw response is a JSON Table Schema object.

IEM states that its data/services are freely available, including lawful
commercial use; its disclaimer places website materials in the public domain
and appreciates attribution. No account, key, subscription or payment was used.
We did not establish a published numeric request quota. The project's local
14-attempt ceiling, one-second pacing and byte caps are our policy, not IEM
service guarantees. IEM offers the service as-is without warranty; long-term
uptime, archive completeness and immutable versions are not guaranteed. Keep
raw receipts and content hashes rather than depending on perpetual API access.

## Actual small-sample results

Run dates: **January 15 and July 15 in each of 2023, 2024, 2025**. GFS 00 UTC;
NBS 01 UTC. Each request asks for all six airport stations:

| Airport | IEM station | GFS sampled runs present | NBS sampled runs present |
| --- | --- | --- | --- |
| ATL | KATL | 6/6 | 6/6 |
| DFW | KDFW | 6/6 | 6/6 |
| DEN | KDEN | 6/6 | 6/6 |
| ORD | KORD | 6/6 | 6/6 |
| LAX | KLAX | 6/6 | 6/6 |
| SFO | KSFO | 6/6 | 6/6 |

| Year | GFS cells / records | NBS cells / records |
| --- | --- | --- |
| 2023 | 12/12; 252 rows | 12/12; 276 rows |
| 2024 | 12/12; 252 rows | 12/12; 276 rows |
| 2025 | 12/12; 252 rows | 12/12; 276 rows |

Across the 12 distinct run/model requests there are **1,584 distinct forecast
rows**, with no unexpected station/model/run or duplicate identity detected.
This is 72/72 *sampled station/run/model cells*, not annual coverage or usable
flight coverage. Six dates only represent winter and summer, not every season
or weather regime. See `artifacts/iem_mos_pilot/coverage.csv` for every cell.

GFS has 21 valid times per station, leads **6–72h**, and observed **3h and 6h**
steps. NBS has 23 valid times per station, leads **5–71h** from its 01 UTC
reference, and **3h** steps. These are native time grids, not hourly forecasts.
No interpolation, nearest-time match, forward fill or observation fill was used.

## Fields and independent semantics

The counts below are non-missing native values, including documented unlimited
ceiling codes where applicable, divided by all returned rows of that model.
Period probabilities are naturally absent at non-reporting endpoints; their
null rate must not be interpreted as random API outages.

| Raw field | GFS, 756 rows | NBS, 828 rows | Meaning / handling |
| --- | --- | --- | --- |
| TMP | 756/756 | 828/828 | Fahrenheit; derive Celsius explicitly |
| WSP | 756/756 | 828/828 | Knots; derive m/s explicitly |
| GST | 0/756 | 828/828 | Native gust; GFS absence stays missing |
| VIS | 756/756 | 828/828 | GFS category 1–7; NBS tenths of miles, capped/rounded per card |
| CIG | 756/756 | 828/828 | GFS category 1–8; NBS hundreds of feet; -88 denotes unlimited |
| LCB | 0/756 | 828/828 | Lowest cloud base, separate from ceiling; raw/status retained |
| P06 | 396/756 | 396/828 | 6h precipitation probability, percent; not precipitation amount |
| T06_1 | 432/756 | 36/828 | Parsed first component of native 6h thunderstorm field |
| T03 | 0/756 | 828/828 | 3h thunderstorm probability, percent; not a 6h field |

NBS CIG contains 259 numeric heights and **569 unlimited** codes. NBS LCB has
748 numeric values and **80 unlimited** codes. Unlimited is not a negative
height or an ordinary missing observation. The parser retains raw values,
records an explicit status, and uses a separate ceiling-unlimited flag.

GFS categories are never converted into fabricated exact distances/heights.
Field-specific GFS 9/99/999 missing codes and NBS -99 are decoded separately;
99% probability remains a valid value. IEM normalizes WGS to GST, and the
concatenated T06 field can be null when T06_2 is absent even though T06_1 exists.
We retain both components; a severe-thunderstorm conditional probability is
not substituted for ordinary thunderstorm probability. MOS probabilities are
never mapped to HRRR CAPE, reflectivity, or hourly precipitation accumulation.
The full per-station field/null/code counts are in `field_checks.csv` and
`field_summary.csv`.

## Genuine schedule checks, no labels

We projected only six identity/schedule columns from the existing processed
BTS files, selected one flight per airport on January 16 and July 16 in each
year using a deterministic seed-42 hash, and reused the existing local-to-UTC
departure converter. This produced **36 actual scheduled flights**. Each is
checked against the corresponding previous-day GFS and NBS run: **72 checks**.
These are fixed-vintage feasibility checks, not a claim that these are the
latest historically available runs.

- **68/72:** no MOS forecast at the exact scheduled instant; refused to round,
  interpolate, or treat the grid as hourly.
- **4/72:** exact forecast identity and sufficient nominal lead, but
  `availability_unverified`.
- **0/72:** ready for strict T−24h model use.

The four matches are SFO's sampled July 2024 flight and ORD's sampled January
2025 flight, each with both models. Their leads are 54/53h and 36/35h. A lead
greater than 24h is necessary here but does not prove public availability at
the cutoff. No flight outcome column, including 2025 labels, was read. This
small hash sample is a diagnostic of alignment losses, not a general flight
coverage estimate or a training dataset.

## Actual network accounting

Dry run planned **12 distinct** data requests. One preliminary GFS request
inspected the schema before the bounded client existed; the client subsequently
retrieved that same run again plus the other eleven. Thus actual MOS data HTTP
attempts = **13**, successful responses = **13**, data errors = **0**, retries =
**0**. This exploratory duplicate is explicitly counted, not hidden as a cache
hit. Total response-body bytes = **1,637,532** (about 1.56 MiB); distinct planned
responses total **1,514,155** bytes. HTTP headers/TLS traffic are not measured.
Counts exclude documentation, GitHub and file-delivery tool retrievals, whose
underlying HTTP counts are not exposed.

`network_attempts.json`, `exploratory_request.json`, `receipts.json` and
`network_summary.json` preserve request URLs, times, status, size and SHA256.
`raw/` preserves all twelve planned responses and the exploratory duplicate.
Some NOAA VLab documentation pages returned 403 through the browser retrieval
tool; official indexed card text and other NOAA pages were used instead.
Those documentation errors are recorded separately in `research_access_notes.json`.

## Evidence references

- [IEM archive history and realtime origin](https://mesonet.agron.iastate.edu/mos/)
- [IEM API access and as-is policy](https://mesonet.agron.iastate.edu/api/)
- [IEM public-domain terms](https://mesonet.agron.iastate.edu/disclaimer.php)
- [Provider API implementation](https://github.com/akrherz/iem-web-services/blob/main/src/iemws/services/mos.py), reviewed blob `d69bfae970e9ebdd8f73ce08437771d33b94c843`.
  Its SQL explicitly converts runtime/ftime to UTC and exposes no publication
  field. This justifies source-specific interpretation of the naive `*_utc`
  JSON strings, not a relaxation of generic timezone checks.
- [NOAA MOS overview and time grid](https://www.noaa.gov/jetstream/upper-air-charts/weather-models/jetstream-max-model-output-statistics)
- [GFS MAV code definitions](https://vlab.noaa.gov/web/mdl/mav-card)
- [NBM 4.0](https://vlab.noaa.gov/web/mdl/nbm-textcard-v4.0),
  [4.1](https://vlab.noaa.gov/web/mdl/nbm-textcard-v4.1),
  [4.2](https://vlab.noaa.gov/web/mdl/nbm-textcard-v4.2) field definitions.

**Next gate:** establish an independently supportable historical publication
time or conservative upper bound for each chosen vintage, and prespecify a
native-grid alignment policy. Until then, use the provider for evidence
collection and parser validation only. Keep HRRR as an explicit alternative;
no API or MOS failure automatically invokes a GRIB/Zarr download.
