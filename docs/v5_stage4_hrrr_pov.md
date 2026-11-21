# Stage 4 HRRR Proof-of-Value

Stage 4 ran the pilot offline against four local BTS Parquet files and the
existing `weather_cache_v2`. Logged traffic for HRRR or MOS data was **zero
requests and zero bytes**. The runner contains no acquisition path and left
the production HRRR selection, cache validation, weather features and join
rules intact.

## Cohort and measured results

Date strata are January 15 and July 15 in 2023 and 2024, with ATL, DFW, DEN,
ORD, LAX and SFO as origins. The pilot uses 09:00–15:59 UTC to include the
cached forecast vintage and keep requests bounded. Date filtering uses the
local BTS date and the converted UTC departure date. Inside each airport/date
stratum, scheduled identity with SHA256 seed 42 ranks up to 200 completed
flights. Labels are excluded from sample selection. Holiday/weekend dates,
early UTC hours, the completed-flight filter and two dates per year limit
what this pilot can represent. 2023 is train, 2024 validation, and 2025 is
reserved. The runner does not open any 2025 files.

| Split | BTS cohort | Complete exploratory weather | Airport/date strata covered | Airport/forecast groups | Strict weather |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 train | 1,587 | 0 | 0/12 | 0 | 0 |
| 2024 validation | 1,705 | 470 (27.57%) | 6/12 | 13 | 0 |
| Combined | 3,292 | 470 (14.28%) | 6/24 | 13 | 0 |

All 470 rows with usable exploratory weather had all seven features. Their
departure target has 88 positives and 382 negatives. Their arrival target
has 79 positives and 391 negatives. They cover only July 2024. Other rows
had zero weather fields in this run. Three valid v2 cache hours were
`2024-07-14 06Z F27`, `06Z F30`, and `12Z F27`. The 2023 legacy
Parquet files in `weather_cache` have no current commit manifests and
lack fields required by today's contract. Recover them only if independent
raw provenance, units, grid coordinates and missing-value statuses can be
reconstructed and validated. The filename and an archived HRRR URL do not
prove historical availability.

## Historical availability and fit gates

The existing planner chooses the latest six-hour cycle at or before T−24h
minus three hours. It uses the forecast valid hour containing the scheduled
departure. The historical forecast run and lead are checked against the
current cache manifest and contract. The three-hour lag is an exploratory
assumption. NOAA's inventory supports the forecast lead and cycle structure.
A current operational completion feed cannot determine availability for a
specific historical payload.
[NOAA HRRR product inventory](https://www.nco.ncep.noaa.gov/pmb/products/hrrr/),
[NCEP realtime production status](https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat_new/).

**Strict PIT** requires an independently reviewed, matching complete-hour
publication or trusted historical receipt. The optional local evidence JSON
is a list of review records with `forecast_run_at_utc`, integer
`forecast_hour`, `available_by_utc`, `kind`
(`verified_publication` or `historical_full_hour_receipt`),
`covers_full_hour: true`, `evidence_reference`, and the exact v2 manifest
`cache_sha256`. A reviewer must bind the archived product version and content
to the extracted payload before supplying this record. The tool checks the
binding and compares the verified upper bound with each flight cutoff. A
locally entered assertion alone supplies no credible evidence. Strict training
also needs a versioned schedule available at T−24h. The retrospective BTS
CRS fields lack that historical schedule vintage, so this runner reports
strict weather coverage and keeps strict fitting disabled.

**Exploratory** keeps the 3-hour assumption and retrospectively recorded
schedule fields explicit. `--mode exploratory --fit` requires coverage
of all 12 airport/date strata per split, 800 training rows with the current
frozen parameters, 100 validation rows, at least 10 positives and 10 negatives
for each target in each split, complete binary targets and some weather
variation in both splits. The training minimum is twice the largest frozen
minimum leaf size. These are execution gates. Statistical support depends on
independent weather cases and more than two days per split.

The two arms share the same ordered eligible row IDs, year split, 21 V3-A
scheduled/interaction features, category levels learned from 2023, and frozen
LightGBM parameters from the existing V3-A workflow. V5 adds seven origin
HRRR features. Neither arm uses observed weather, flight status or actual
times as features. For both labels, the script reports trapezoidal PR-AUC
and average precision separately, ROC-AUC, F1 at the fixed 0.5 threshold,
Brier score, reliability bins and validation predictions. For an expanded
pilot, compare paired differences with airport/date or weather-episode
blocking and enough independent dates before interpreting any change as a
generalizable gain.

No BTS-backed model was fit. The explicit fit request exited 2, gave the same
coverage diagnostics and wrote no predictions or metrics. Strict weather
and current exploratory model comparison are both **No-Go**.

## Offline Dry Run

Run from the integrated working tree using the existing project environment.

```bash
PYTHONPATH=src python scripts/hrrr_pov.py \
  --data-dir data/processed \
  --cache data/interim/weather_cache_v2 \
  --params artifacts/hyperparameter_optimization/best_params.json \
  --grib-cost-reference artifacts/weather_feasibility/historical_grib_acquisition.csv \
  --output /tmp/hrrr-pov-dry
```

The summary records source, cache, parameter and cohort audit hashes.
`cohort_audit.csv` contains schedule/forecast mapping and SHA256 flight IDs,
with no individual labels. `all_hour_requests.csv` and `cycle_tasks.csv`
are the complete bounded plan. `minimum_additions.json` plans the minimum
cardinality set of missing hours that could satisfy the specified row and
airport/date support gates given the frozen cohort. This search ignores
labels and assumes each added hour passes cache and feature checks. It gives
no guarantee about class support, weather variance, or receipt evidence.
The five-hour plan projects 818 training rows and 652 validation rows
(12 strata each), spread across 24 and 19 airport/forecast groups
respectively, if every planned payload is usable. These are candidate
counts calculated from flight-to-hour mappings.

| Plan | Forecast hours | Estimated GRIB requests without retry | Estimated range payload |
| --- | ---: | ---: | ---: |
| Full pilot | 27 across eight cycles, 3 cached, 24 missing | 240 additional | 287,980,776–290,823,624 bytes |
| Minimum pilot completion | 5 additional | 50 additional | 59,995,995–60,588,255 bytes |

The estimate uses the prior six-hour GRIB range measurement, 11,999,199 to
12,117,651 bytes per hour. The existing path uses one inventory plus nine
variable range requests per forecast hour. Estimates exclude inventory bytes,
headers and retries. They are unmeasured for the proposed hours and do
not estimate Zarr chunk costs. The exact five-hour GRIB candidate list is:

| Run UTC | Lead | Valid UTC |
| --- | ---: | --- |
| 2023-01-14 06:00 | F31 | 2023-01-15 13:00 |
| 2023-01-14 06:00 | F32 | 2023-01-15 14:00 |
| 2023-07-14 06:00 | F31 | 2023-07-15 13:00 |
| 2023-07-14 12:00 | F27 | 2023-07-15 15:00 |
| 2024-01-14 06:00 | F31 | 2024-01-15 13:00 |

After original-worktree integration, the lowest-cost next action is to look
for a contract-valid local copy of the July 2023 F31 payload. If exploratory
work proceeds and the local search fails, begin with one listed forecast
hour and a hard request/byte budget, verify the existing pipeline end to end,
and rerun the dry plan. Keep this separate from a Strict PIT track, which
needs historical weather evidence and a schedule snapshot before training.
Reserve broader collection and 2025 evaluation for a later decision.
