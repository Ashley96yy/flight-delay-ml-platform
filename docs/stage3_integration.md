# Stage 3 integration status and safe import

The original project remains on `main` at `b53a8b1`. Stage 1/2 API files are
still absent there; existing V5 modules include uncommitted user work. The
current chat still grants write access to the Lab1 workspace and scratch,
not `/Users/dingyuyao/Documents/Project/flight-delay-ml-platform`.

Consequently, **the original repository has not been modified, committed or
pushed**, and `main` has not been merged. Development uses an independent
local copy at `/tmp/flight-delay-iem-stage3`, branch
`feature/v5-api-iem-mos`, based on Stage-2 commit `b733a69`. Branch/commit names
in the delivery refer only to that independent copy.

The latest Stage-2 bundle was materialized and its exact SHA256 verified:
`014f89d8a82578d70c4eabb248220977ff0575ff428d2a0902b3c8ff6c2cd2e4`.
It is byte-identical to the prior delivery. Its cumulative patch passed
`git apply --check` against the original current V5 working copy, and the
integrated independent copy reproduced **608 passed, 3 skipped** before edits.
Stage 1 was not applied a second time.

## This stage

- New `src/flight_delay/weather/mos_alignment.py`: bounded A/B alignment and
  typed, payload-bound publication evidence; no unverified model-value output.
- New `scripts/replay_mos_alignment.py`: offline-only comparison on saved
  forecasts/schedules; no access to labels, network or model training.
- New `tests/test_mos_alignment.py`: 39 tests; full regression 647/3 skipped.
- New evidence, alignment and integration reports plus replay/probe receipts.
- README links to the current stage. No HRRR business logic, dependency,
  existing source module or existing test is changed.

The cumulative delivery patch is for the **current original V5 working copy**
that has neither stage installed. The incremental Stage-3 patch is for a copy
that already has Stage 2. Use one route, not both. An older remote checkout
without the uncommitted V5 prerequisites cannot be assumed compatible.

## Apply when the original directory is writable

```bash
cd /Users/dingyuyao/Documents/Project/flight-delay-ml-platform
git switch -c feature/v5-api-iem-mos
git apply --check /absolute/path/to/v5_stages1_2_3_cumulative.patch
git apply /absolute/path/to/v5_stages1_2_3_cumulative.patch
```

Inspect the original diff/staging state first. If the branch already exists,
inspect it instead of replacing it. Creating the branch keeps unrelated
uncommitted work in place; it is not a backup of that work. Do not run
`git reset --hard`, force-checkout, indiscriminate `git add .`, or copy a whole
replacement repository over the current folder. Existing V5 dependencies also
need a separate review before any remote-ready commit; do not add the virtual
environment or large datasets. No commit/merge/push on the original is claimed.

Run Ruff and full pytest after applying. The independent development copy's
tests and import checks do not replace regression in a newly modified original
working tree. The existing Python 3.12 native-readline workaround and test
command are documented in `v5_mos_integration.md`.
