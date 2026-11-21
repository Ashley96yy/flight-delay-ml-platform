# V5 Stage 9B Phase 1 · Offline preflight

**Decision — READY for separately authorized Phase 2.** Phase 1 ran zero
real weather HTTP requests and zero model fits. Strict PIT remains
unverified. No batch runner or training CLI was introduced.

## Safety gate implementation

The Stage 9A archive was verified against SHA-256
`c5a8861bd03f0836d88058082d3a1e01a2fb1b39710c693a103a0f31f3d5d5be`
and its internal file manifest. The shipped forecast list is byte-for-byte
identical to the Stage 9A output. Its SHA-256 is
`2a0fb0cdd6d0537b4e71ae3b281153e886278b272280ce7dba3b004354d085ac`.
The protocol loader checks its pinned JSON digest and the forecast-list
digest before use. A forecast is accepted only by exact UTC run, integer lead,
and valid time. The project constructs the NOAA index and range URLs
from that validated identity.

`stage9b_acquisition.py` exposes a guarded single-hour function. The direct
HTTPS transport issues one GET, with no automatic retries or redirects.
The ordinary GRIB modules and older exploratory runner are untouched.
Any future execution must use the guarded function and the same dedicated
local campaign cache root for every invocation or restart. The lock and
journal persist in that root. Mutations of campaign files outside this
function or switching cache roots are outside its accounting.

- A POSIX exclusive lock covers cache validation, network operations and
  publication. Another process using the same root is rejected. An
  initialized marker is bound to one SQLite journal. A lost, truncated,
  or mismatched SQLite journal fails closed.
- SQLite uses fully synchronized commits. HTTP attempts are committed
  before opening a connection, and include connection and response
  failures. Byte reservations are committed before body reads. Each
  reservation includes the expected length and one bounded EOF probe.
  Reservations are retained on error and abrupt exit. The journal stores
  observed body bytes when available. An abrupt interruption preserves
  the attempt and reserved bytes even if the exact body bytes are unknown.
- The limits are 240 HTTP attempts, 16 MiB attributable response body
  per hour and 384 MiB overall. Index size is limited to 64 KiB. The helper
  reads at most 64 KiB per call, subject to the smaller remaining
  reservation. Limits cover application-read response body bytes.
  HTTP/TLS headers, kernel buffering and retransmission bytes fall
  outside that count.
- The index must describe the requested run and contain all nine exact
  fields. Range responses must be 206 and match Content-Range and
  Content-Length. Range responses also agree on the object length and
  strong ETag. Subsequent requests use If-Match. A full-object 200 or a
  redirect stops before application body reads. Streaming overruns, short
  or invalidly framed messages and mismatched trusted SHA-256s fail closed.
  Response digests are retained for auditing. Independent source digests
  are checked when supplied. An observed digest establishes what was read.
- A cache hour is published using the existing atomic manifest after
  all nine messages are decoded and the six-airport contract and seven
  derived features pass. Cache hits are verified via the existing SHA-256
  check and yield zero HTTP attempts. Loss or replacement of a previously
  journaled cache entry causes a hard failure. Forecast-key, URL, requested
  ranges, byte charges and digests are retained in the journal, with a
  link to the committed Parquet SHA-256 after successful publication.

## Offline paired validation

The existing BTS completed Parquet inputs were verified against the 24
Stage 8 SHA-256s. Only the 2023 and 2024 partitions were accessed. We kept
one day per month and six origins. The frozen rule is the third Tuesday
of odd months and the third Saturday of even months, scheduled
departure 13:00–13:59 UTC with the same local calendar day. Each hour is
the preceding day's 06Z run, F31, valid at 13Z UTC.

Real BTS schedules and labels were paired offline with 24 synthetic,
contract-valid weather hours in a separate scratch directory. The result
has 3,021 training flights and 3,230 validation flights, each split
spanning 12 frozen dates. Negating both labels in memory left the selected
IDs and their order unchanged. Withholding one synthetic forecast
manifest produced an explicit failure. Unit cases also rejected missing,
duplicated, nonfinite, wrong-run and wrong-airport weather and constant
training weather.

Both matrices share the same ordered manifest and label vectors. The
21 baseline columns reproduce the Stage 8 contract, including train-only
category levels. V5 appends seven weather columns. The resulting shapes
are 6,251 × 21 and 6,251 × 28. The existing many-to-one join uses airport,
run, lead and valid time. Additional order and cardinality checks guard
against row copy or loss. Each manifest row carries the forecast keys,
assumed availability, backend, committed Parquet SHA-256, weather
manifest digest and cache fingerprint. The 24-hour boundary is audited
using the existing 3-hour forecast availability assumption.

The result is explicitly `retrospective_exploratory`. Historical full-hour
publication and historical BTS schedule availability at T−24h are unverified.
The dry run generated no new scores or model metrics.

## Frozen evaluation protocol

The parameter file required by the loader retains SHA-256
`2f36b7b62d622d7458abd2bf54711ec24163df425b287b67ebb442c64be4b65c`.
The Stage 9A DESIGN and DECISION files define all requested resampling
settings and the arrival guardrail. No missing requested criterion was
filled in after inspection.

| Item | Frozen rule |
| --- | --- |
| Primary | Equal-date DepDel15 AP difference, V5 minus V3-A |
| Secondary | Date-paired ROC-AUC and Brier differences, plus pooled descriptive metrics |
| Bootstrap | Whole paired dates, 3 draws per calendar quarter, 10,000 resamples, seed 42, 95% percentile interval |
| Robustness | Mean departure AP difference remains positive after deleting each date in turn |
| Arrival guardrail | Arrival AP difference interval lower bound ≥ −0.01 |
| Signal | Departure equal-date AP gain ≥ +0.01, interval lower bound > 0, leave-one-out positive, arrival guardrail met |
| Unverifiable | Missing inputs or class support, non-viable baseline ranking, or no weather splits in the departure V5 model |
| Below practical threshold | Departure interval upper bound < +0.01 |
| Inconclusive | Remaining cases, with no additional date selection or parameter search |

Evaluation accepts already aligned prediction tables only. The Phase 1
tests exercised it with synthetic scores.

## Reproduction and verification

Exact local equivalents of the test and lint invocations are below.
The report bundle includes the socket-disabled pytest wrapper
`run_pytest_offline.py`, the real-BTS/synthetic-weather offline runner
`offline_preflight.py`, and the commands' recorded outputs.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  python /path/to/run_pytest_offline.py tests/test_stage9b_phase1.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  python /path/to/run_pytest_offline.py
ruff check --no-cache src scripts tests
ruff format --check --no-cache src/flight_delay/weather/stage9b*.py tests/test_stage9b_phase1.py
```

The focused suite yielded **44 passed**. Full regression yielded
**703 passed, 3 skipped**, with only the three opt-in real-weather tests
skipped. Both runs recorded zero socket guard calls. Ruff checks and
format checks passed. The report bundle records the offline manifest
SHA-256 and the injected missing-weather error.

Phase 2 still requires separate explicit authorization. Probe attempts,
downloads, data selection changes and model fits remained outside this
preflight.
