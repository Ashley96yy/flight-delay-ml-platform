# Stage 2 integration, verification and reproduction

## Source and integration status

The original local project remains at
`/Users/dingyuyao/Documents/Project/flight-delay-ml-platform`, local HEAD
`b53a8b1`. It contains uncommitted V5 weather code and tests. The remote contents
lookup did not find `src/flight_delay/weather/api_providers.py`, and local
inspection confirmed the stage-1 adapter patch was not applied there.

The current stage-1 Library bundle was retrieved and checked against the
previously tested bundle: SHA256
`08b95ccad61b8a5cf2ad35107a7a8278a58c7aae4697506a609b5d4629257238`.
It was byte-identical. The prior patch applies cleanly to the original working
copy. All 86 existing snapshot files other than the intentionally updated empty
README matched the original byte-for-byte.

This chat's writable project root is still the DATA266 Lab1 directory, not the
original flight-delay directory. Attempting to attach the original checkout
through the app was unavailable. No shell write to the unauthorized original
directory was attempted. Integration and testing therefore use
`/tmp/flight-delay-iem-stage2`, a local Git copy of the validated stage-1 snapshot.
**The original repository has NOT been patched or pushed.** The deliverable
includes both a cumulative patch for the original current V5 working copy and
an incremental stage-2 patch for a copy that already has stage 1. Apply one,
not both. They are not intended for an older checkout missing the existing V5
modules. No original data, notebooks, environment or unrelated edits are bundled.

## Additive implementation

- `iem_mos.py`: `IEMMOSProvider`, independent `MOSRunRequest`, shared
  `APIRequestPlan`; six-station explicit-run requests; research-only default-off
  client; bounded GET, timeout, finite retries, persistent budget and hashed cache.
- `mos_contracts.py`: model-specific encodings, raw-field preservation,
  source-specific explicit UTC handling, separate MOS features and provenance.
  The parser checks response hash, station/model/run and duplicate identities.
- `mos_flight_join.py`: reuses the existing join identity keys and strict UTC
  validator. Adds a source-specific MOS audit/guard without changing the HRRR
  planner, +3h assumption, F27–F32 constraints or ready rules. Unknown
  publication time, off-grid flights and future vintages are rejected.
- `scripts/probe_iem_mos.py`: fixed tiny dry-run/acquisition/analysis CLI,
  separate archive and flight coverage, label-free schedule projection and reports.
- Three new test files cover provider/client, field contract/join and pilot scope.
- Reports/raw responses in `artifacts/iem_mos_pilot`; feasibility and updated
  experiment design in `docs/`. README links to them; no dependency added.

The existing `HistoricalForecastClient` is unchanged: its evidence-approved
fetch still blocks unresolved candidates. `IEMMOSClient.fetch` is explicitly
**research acquisition**, not model approval, and preserves all source blockers.
No HRRR fallback, GRIB request or Zarr request is triggered by this new path.

## Budget, cache and failure behavior

Default network is off. Explicit opt-in allows only the free IEM MOS endpoint
with one GFS/NBS product, an exact supported retained cycle and at most six
known airport stations. No latest/implicit runtime, extra parameters or paid
endpoints. Cache hits can replay with network disabled; SHA256, URL, request
identity, byte count and receipt time must agree. Cache corruption fails closed.

The attempt ledger is reserved atomically before each transport attempt, so
failed and interrupted attempts consume quota. Every retry is separately
counted. Retryable HTTP statuses are 408/429/500/502/503/504 and transient I/O
timeouts; schema/contract failures and 404 are not automatically retried.
Retries default to one, with a hard configurable maximum of two. Redirects are
disabled. Default timeout is 20 seconds per HTTP operation (not a total wall-clock
deadline), maximum response is 2 MB, known response-body budget is 8 MB,
default attempt cap 14, pacing is at least one second within a client process.
Quota persists within one pilot cache directory; it is not an IEM account quota.

One writer per cache directory is supported, protected by an exclusive lock.
A crash can leave `writer.lock`; do not remove it without checking the old
process has ended. This fails closed rather than automatically resetting a
budget. Exact received byte counts are available for successful responses;
network transfer overhead or a partial body lost to a transport exception is
not measurable by this interface. The live pilot had no such errors.

## Actual checks

| Version | Test result |
| --- | --- |
| Existing V5 before stage 1 | 487 passed, 3 skipped (previous stage reproduced) |
| Integrated stage 1, rerun this stage | 543 passed, 3 skipped |
| Stage 2 final full suite | **608 passed, 3 skipped** |

Stage 2 adds **65 tests**. Existing tests were preserved, including all three
opt-in live HRRR tests (still skipped). The January BTS integration test uses
the existing local ZIP through a read-only source symlink, with no download.
Ruff lint passes over source, tests and the new pilot script; format checks
pass for the new files. Cache replay is covered by restart/integrity tests.
Real response parsing and all derived CSVs were reproduced offline after
collection; no extra data calls were needed for analysis.

The existing local virtual environment has a native-readline startup crash
in pytest's optional `_readline_workaround`. Only that helper is disabled in
the runner process; capture fixtures and all test bodies remain enabled. No
pytest package, environment dependency or original project file was edited:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/flight-delay-mpl \
"$PROJECT_PYTHON" -c 'import _pytest.capture; _pytest.capture._readline_workaround=lambda:None; import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider"]))'
```

`PROJECT_PYTHON` refers to the existing flight-delay `.venv/bin/python`. Normally
configured Python environments can simply run `PYTHONPATH=src python -m pytest`.

## Reproduce without any network

After applying the cumulative patch, the saved raw responses and receipts are
already present. Run:

```bash
PYTHONPATH=src python scripts/probe_iem_mos.py --analyze-only
```

To also repeat the schedule audit with existing local BTS files (six schedule
columns only, including the 2025 metadata checks):

```bash
PYTHONPATH=src python scripts/probe_iem_mos.py --analyze-only \
  --flights-dir /Users/dingyuyao/Documents/Project/flight-delay-ml-platform/data/processed
```

A fresh request plan, still without network:

```bash
PYTHONPATH=src python scripts/probe_iem_mos.py --output /tmp/iem-new-plan
```

The explicit `--allow-network` option is for future deliberate bounded rechecks;
it is unnecessary to reproduce this report. Use one dedicated persistent cache
for a pilot. The local collection actually used a client cap of 13 plus one
pre-client exploration, totaling 13 successful attempts within the planned
14-attempt ceiling. Network-only totals and raw receipts are documented in
`iem_mos_feasibility.md`.
