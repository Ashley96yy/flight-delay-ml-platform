# V5 API stage 1 — implementation and validation

Completed 2026-11-20 against the existing local V5 working copy at
`/Users/dingyuyao/Documents/Project/flight-delay-ml-platform`.

**Result:** API-first/small-data-first/evidence-driven planning and bounded
client infrastructure are ready as an additive patch. This stage has not
activated a production API provider or trained a weather-enhanced model.

## Files

| Change | File | Purpose |
| --- | --- | --- |
| New | `src/flight_delay/weather/api_providers.py` | Replaceable planning protocol, T-24h queries, explicit Single Runs and byrun adapters; source/evidence blockers |
| New | `src/flight_delay/weather/api_client.py` | Default-off JSON client, attempt/byte/time/pacing limits, paid/host/provenance guards, response hash |
| New | `src/flight_delay/weather/api_dry_run.py` | Label-free bounded CLI; request deduplication and input mapping; no live switch |
| New | `tests/test_weather_api_providers.py` | Temporal policy, archive boundaries, request identity, unsupported inputs |
| New | `tests/test_weather_api_client.py` | Dry-run isolation, paid/evidence/host guards, quotas, pacing, errors, response limits, redirects |
| New | `tests/test_weather_api_dry_run.py` | CLI mapping, budget exit status, input/label limits and no-live enforcement |
| New | `examples/weather_api_queries.json` | Three synthetic schedule-only planner inputs |
| New | `docs/v5_api.md` | Official-source research, compatibility boundary, future normalized contract and HRRR fallback design |
| New | `docs/v5_proof_of_value.md` | Frozen paired V3-A/V5 development experiment; no 2025 access |
| New | `docs/v5_stage1_report.md` | This report |
| New | `artifacts/weather_api_stage1/open_meteo_dry_run.json` | Actual network-free planning output |
| New | `artifacts/weather_api_stage1/visual_crossing_dry_run.json` | Actual network-free planning output |
| Modified | `README.md` | Entry points and current readiness limits; original file was empty |

No pre-existing source file or test was edited. HRRR contract, backend list,
cache fingerprint, planners, fallback, batch runner, weather features, flight
join and leakage checks are preserved. No dependency was added.

## Research conclusion

Visual Crossing's documented historical forecasts cover dates from 2020, but
access requires a paid addon and temporal/field evidence still needs review.
The implementation cannot make a paid call. Open-Meteo's stitched historical
series is not a T-24h run archive; Single Runs lacks documented 2023 IFS history
and explicitly carries a hindcast warning for part of its historical dataset.
Previous Runs has field-dependent history and does not resolve all timing
questions. No free complete eligible 2023–2025 source is claimed. Official
links, unresolved questions and source-specific decisions are in `v5_api.md`.

## Actual checks

| Check | Result |
| --- | --- |
| Unmodified existing implementation | **487 passed, 3 skipped** |
| New API tests only | **56 passed** |
| Full suite after addition | **543 passed, 3 skipped** |
| `ruff check src tests` | Passed |
| `ruff format --check` on all six new Python files | Passed |
| `git apply --check` against the original local V5 working copy | Passed; check only, no files applied |
| Existing snapshot files compared byte-for-byte with the original | 86 matching, excluding intentionally updated README |
| Open-Meteo dry run | 3 input queries, 2 unique planned HTTP requests, 0 actual attempts |
| Visual Crossing dry run | 3 input queries, 2 unique planned HTTP requests, 0 actual attempts |

Both reports have `ready_for_live=false`, explicit evidence blockers, and null
coverage, billing estimates and model metrics. The three skipped tests are the
existing opt-in live HRRR Zarr/GRIB/batch tests; they were not enabled. The real
January 2023 BTS integration test used the already-existing local ZIP via a
read-only source symlink. No dataset was downloaded for these checks. Existing
downloader unit tests print simulated download messages from mocked transports.

The local Python 3.12.2 virtual environment's native `readline` extension crashes
during pytest's optional startup workaround. Tests were run with only that
workaround disabled in the runner process; capture fixtures and test bodies
remain intact. No environment packages or pytest files were edited. Command:

```bash
# PROJECT_PYTHON is the existing flight-delay .venv/bin/python.
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/flight-delay-mpl \
"$PROJECT_PYTHON" -c 'import _pytest.capture; _pytest.capture._readline_workaround = lambda: None; import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider"]))'
```

## Integration and scope

This chat has write access to the Lab1 workspace, not the source flight-delay
directory. Development and the local Git snapshot are therefore isolated at
`/tmp/flight-delay-api-stage1`. The source tree, its staging state, local data,
notebooks, environment and unrelated uncommitted changes were not modified.
The patch is intended for the **current local V5 working copy**, which already
contains the modules under test, not an older remote checkout missing them.
No GitHub commit, branch, PR or push was created.

The delivery bundle contains only this stage's changed/new files, the additive
patch and import instructions; it excludes the existing repository, datasets,
environment, credentials and other local changes. Before applying the patch,
check the destination to avoid overwriting existing work:

```bash
cd /Users/dingyuyao/Documents/Project/flight-delay-ml-platform
git apply --check /path/to/v5_api_stage1.patch
git apply /path/to/v5_api_stage1.patch
```

Next development gate: verify an original historical operational run and its
publication/field semantics, review cost if a paid provider is chosen, then add
a normalized decoder and separate bounded cache. Only after that should the
fixed paired experiment in `v5_proof_of_value.md` run. This stage does not
claim API coverage, leakage-free API model inputs, fitted models or score gains.
