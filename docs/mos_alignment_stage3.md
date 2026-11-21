# Stage 3 — bounded MOS alignment and offline comparison

## Design

The new `mos_alignment.py` consumes the independent MOS contract and explicit
flight/model/run requests. It reuses strict UTC validation and leaves all
existing HRRR and Stage-2 join/audit logic unchanged. Neither strategy searches
other runs to improve coverage. A group must have one station/model/run,
one request identity and one original response hash; duplicate points and
mixed versions fail closed.

**A — Conservative:** retain the closest native point not later than departure,
with a maximum age relative to departure of 3 hours. Keep its original valid
timestamp and label it an aged point proxy, not an hourly forecast.

**B — Restricted interpolation:** find both bracketing native points in that
same group. Each point must lie within 3 hours of the target and their total
span must not exceed 6 hours. No extrapolation, cross-run bracket or missing-value
imputation. Only Celsius temperature and wind speed in m/s are allowlisted for
linear interpolation. A future *forecast valid time* can be a right endpoint
only if the entire matching run was already available at the prediction cutoff;
it is not a future observed measurement.

Both methods keep categories, physical/capped visibility, cloud ceiling and
unlimited flags at the conservative point; no linear blending of these fields
or gusts is implemented. Probability fields are emitted only at their exact
native period endpoint. Off-grid P06/T06/T03 remain unavailable rather than
shifting a 6h/3h probability window to describe departure. A different lagged
probability-context feature would require a separately named, prespecified
definition. Unlimited NBS ceiling keeps its flag and an absent numeric height;
this is not a random missing-source record.

Gap caps and field rules were fixed from the native 3h/6h grid and semantics,
not selected using delay labels or a validation score. These are conservative
research choices, not claims of meteorological optimality.

## Publication guard and provenance

`RunAvailabilityEvidence` separately represents unverified information,
schedule assumptions, verified publication or a historical full-run receipt.
Reviewed release/receipt evidence must bind the original payload SHA256,
station, model and nominal cycle and explicitly cover the complete station/run.
An assumed timestamp cannot pass the gate. A receipt is an available-by bound;
it never invents the exact publication instant.

Unverified runs produce **diagnostic metadata only**: source endpoints, source
identity/hash, candidate field presence, weights and gap/age measures. They do
not execute interpolation arithmetic or emit numerical weather features for
model use. `require_model_ready_alignment` rejects them. Synthetic unit-test
receipts exercise the approved path; none is applied to real responses.

Provenance includes nominal run time, target departure, exact T−24h cutoff,
actual publication/verified available-by/assumed availability separately,
left/right native valid times, left/right gaps, bracket span, right weight,
planned/executed interpolation flags, per-field methods, forecast age at the
prediction cutoff, target lead, source request key and source response hash.
Timing approval and required continuous-field completeness are separate checks.

## Actual offline replay

Input is exactly Stage 2's 12 saved JSON responses (1,584 forecast points) and
36 saved real schedules, yielding 72 flight/model checks per strategy. The
2025 portion is source/alignment diagnostics only; no delay label was read and
no predictor, threshold or model parameter was optimized from test outcomes.
No network client is used by the replay script.

| Measure | A Conservative | B Interpolation plan |
| --- | --- | --- |
| Candidate temporal matches | 72/72 (100%) | 72/72 (100%) |
| Maximum past valid-time gap | 2h 59m | 2h 59m |
| Maximum future endpoint gap | Not applicable | 2h 40m |
| Maximum bracketing span | Not applicable | 3h |
| Potential native field slots present | 344/504 (68.25%) | 344/504 (68.25%) |
| Temperature/wind speed potential coverage | 72/72 each | 72/72 each |
| Planned off-grid interpolation rows | 0 | 68 |
| Executed real interpolations | 0 | 0 |
| Materialized model field values | 0 | 0 |
| Availability | Unverified | Unverified |
| Strict point-in-time passes | **0/72** | **0/72** |

Stage 2 exact-point geometry matched 4/72. The larger Stage-3 *candidate*
matching rate comes from a different explicit alignment policy, not new data,
confirmed publication or a more permissive leakage gate. It is not model-ready
coverage. Both methods use the same 72 requests and have the same potential
field coverage in this sample. Therefore this replay establishes no empirical
benefit from interpolation over the simpler conservative candidate.

The 504 denominator consists of six GFS fields × 36 checks plus eight NBS
fields × 36 checks. GFS candidates have temperature/wind and category VIS/CIG
at all 36 points. NBS has temperature/wind/gust/visibility/unlimited flags at
all 36; numeric ceiling height exists at 12/36, with the unlimited indicator
carrying the other semantics. Probabilities retain their endpoint and are
available at only 2/36 checks per reported probability field. See
`field_coverage.csv` for exact fields/denominators. The field percentage is not
an API uptime estimate or an assumption that absent heights equal zero.

## Tests and reproduction

Stage 3 adds 39 tests: strict release/receipt boundaries and scope, assumptions,
payload binding, whole-run future endpoints, mixed-run/version rejection,
missing endpoints, category/window protection, gap caps, cross-midnight UTC
arithmetic, spring/fall DST offsets, ambiguous/nonexistent BTS times, preserved
order/duplicates and actual saved-response offline replay.

Full regression: **647 passed, 3 skipped**, compared with the reproduced
Stage-2 baseline **608 passed, 3 skipped**. Ruff lint/format passes for the
new code, tests and replay script. The existing native-readline pytest startup
workaround described in `v5_mos_integration.md` was used without modifying tests.

```bash
PYTHONPATH=src python scripts/replay_mos_alignment.py
```

This writes per-strategy details, per-field coverage, year-level counts and a
summary with immutable input hashes under `artifacts/iem_mos_stage3/`. It has
no network switch or training function. New weather JSON requests: **0**.
Two separate publication-text probes returned only 58 body bytes and two
application-level no-product messages; see `network_summary.json`.

**PoV decision: do not start V3-A/V5 training yet.** The blocker is historical
publication/full-run vintage evidence. Alignment mechanics are implemented,
but neither their 100% candidate matching rate nor passing software tests
certifies historical information availability.
