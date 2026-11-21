# Small-data Proof-of-Value specification (not executed)

Historical stage-1 draft. For current MOS-specific source, field and alignment
decisions, use [the updated MOS experiment specification](v5_mos_proof_of_value.md).

## Question and fixed comparison

Does weather known at the exact T-24h cutoff add predictive value to the existing
V3-A LightGBM baseline on the same flights? Use the current V3-A's 21 scheduled
and route-interaction features, as defined in `07_congestion_models.ipynb`, not
the V3-C1 congestion variant or V4 history features. V5 is precisely that V3-A
matrix plus a prespecified, validated weather subset. Record its actual column
names and feature version in the experiment manifest. Do not compare new sample
scores with the original full-data notebook scores.

## Label-blind sample, hard bounds, and time split

Preserve the established development boundary: 2023 training; Jan–Aug 2024
validation. For the first pilot choose origin ATL and SFO on these fixed local
flight dates, representing different seasons:

| Split | Dates | Maximum sampled flights per airport per date |
| --- | --- | --- |
| Train | 2023-01-15, 03-15, 05-15, 07-15, 09-15, 11-15 | 200 |
| Validation | 2024-01-15, 03-15, 05-15, 07-15 | 200 |

At most 2,400 training and 1,600 validation rows. This is a development smoke
study, with a narrow population and little power, not a production estimate.
Start with three label-free queries and the stage-1 dry run before this pilot.
The three included example queries are synthetic planner inputs only.

Derive airport-local dates using the existing timezone/timestamp resolution
code; retain its unresolved/DST diagnostics. Rank eligible rows by
SHA256(seed=42 + complete canonical flight identity), take the first 200 in each
cell and preserve the order. Labels must not enter sample selection, weather
requests, source choice, or coverage exclusions. Do not select only delayed or
severe-weather days. Do not move dates in response to model performance.

Save an immutable flight-identity list, source manifest SHA256s, labels SHA256,
sample seed, versions and split assignments before any fit. Duplicate flight
identities or unresolved prediction timestamps fail preflight. Read training
and development files only. Keep 2024 calibration/threshold partitions and all
2025 test rows/labels outside this stage. Do not create or alter test labels.

Deduplicate weather by provider/location/run/field selection and forecast hour.
Set a future shared-client hard cap of 100 HTTP attempts and 1 MB per response;
if the plan is larger, stop before acquisition and revise a label-blind design.
This is a ceiling, not a claim about how many requests will be required.
No annual loops, GRIB/Zarr bulk downloads or unapproved paid access. A paid API
requires an approved concrete quote/license and budget before implementation
can enable it; stage 1 has no paid execution path.

## Evidence and coverage gates before fitting

Freeze one eligible provider/model and a weather feature list across train and
validation. The initial API candidate subset is temperature, wind speed, gust
and 1h precipitation; units, accumulation intervals and actual archived
availability must first pass review. Reject hindcasts or unproven availability.
Open-Meteo IFS's documented archive does not cover this 2023 training design;
do not silently train only on 2024 or combine unreviewed sources across years.
If no source passes, report **blocked: no eligible historical forecast source**.

First report attempted identities, matched/missing forecasts, missing fields,
availability failures, fallback identities and coverage per airport/date/split.
Measure coverage on the frozen flight sample; do not infer it from HTTP 200 or
the number of planned URLs. Run the point-in-time audit before exposing weather
values to a model. Check initialization/lead/valid-time consistency and release
before each flight's exact cutoff; keep provenance out of predictor columns.

Primary comparison uses the same complete-weather intersection for V3 and V5,
with exactly matched flight identities/order and unmodified target arrays. Freeze
that intersection before fitting, and report every excluded row and reason.
Also report V3 on the full original sample to expose selection effects. Stop if
either target has only one class in a split. If HRRR cache fallback is used,
report the mixed-source design and separate source strata; do not describe it
as measured API coverage. Never use observations to fill failed forecasts.

## Fixed fitting and paired reporting

Fit fresh LightGBM instances for DepDel15 and ArrDel15 in both arms. Copy and
hash `artifacts/hyperparameter_optimization/best_params.json` as currently used
by V3, including seed=42. Keep parameters fixed, with no 2025 access, search or
early stopping on the comparison validation set. Fit categories and any
imputation on training only; carry identical preprocessing for shared columns.
The original `min_child_samples` may underfit this tiny sample; disclose that
limitation instead of retuning after looking at validation gains.

Report Average Precision (`average_precision_score`) as the primary PR measure,
ROC-AUC and Brier score as secondary metrics, and F1 only at a predeclared 0.5
threshold. Report V3, V5 and paired deltas for both targets, plus training time,
weather acquisition time, HTTP attempts and source/field coverage. Do not label
AP as trapezoidal PR-AUC. Use the same validation row IDs for all deltas.

For uncertainty, bootstrap paired predictions by airport-local flight day
(1,000 draws, seed=42). Report the small number of independent day groups and
the resulting exploratory uncertainty. Include prespecified airport/daypart
breakdowns only when sample/class counts support the metric. Do not turn a
subgroup chosen after seeing gains into a headline.

Proceed only if the source passes evidence review, coverage is adequate for
the declared population, and paired results suggest a useful gain relative to
cost and complexity. Agree a numerical utility target before the future fit.
No gain, wide uncertainty or inadequate coverage is a valid reported outcome.

## Planned outputs

- `sample_manifest.json`: identities, immutable hashes, split, seed, date/airport scope.
- `api_request_plan.json`: deduplicated workload, budget and evidence blockers.
- `weather_evidence.json` and `coverage.csv`: original forecasts, timing and gaps.
- `point_in_time_audit.json`: zero unreviewed source or timing violations.
- `experiment_config.json`: feature names, versions, fixed model and preprocessing.
- `paired_predictions.parquet`, `metrics.csv`, `paired_bootstrap.csv` and a limitations note.

Only planning and synthetic unit-test artifacts exist at stage 1. None of the
weather coverage, predictions, experimental metrics or fitted models above has
been generated or claimed.
