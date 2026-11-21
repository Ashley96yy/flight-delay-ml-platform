# V5 Stage 7 — Baseline diagnostics and experiment validity

**Decision: No-Go for a V5 lift claim or promotion.** The first arrival
run is a degenerate fit. The second round restores score variation and
exposes a train-to-validation ranking gap on a two-date split.
The existing experiments are reproducible exploratory diagnostics.

## Inputs and reproducibility

The audited branch is `feature/v5-api-iem-mos` at
`52ed386a084f0231fda90b088dae16fbc61d8d80` locally and on GitHub. Remote
`main` is `b53a8b1`. The original workspace was read-only to this session.
The patch adds reporting and explicit category handling. Its base is
the existing branch. No Git writeback, commit, push or merge was performed.

| Local run | Parameter source | Scope |
| --- | --- | --- |
| `/tmp/hrrr-pov-first-experiment` | Repository Optuna configuration from the original V1/V3 workflow | Frozen parameters |
| `/tmp/hrrr-pov-small-sample-diagnostic` | `/tmp/hrrr_pov_small_params.json` | Small-sample diagnostic configuration |

Both runs had identical cohort, BTS input and HRRR payload hashes.
The eligible sets were 818 training flights (137 departure positives and
130 arrival positives) and 652 validation flights (153 departure positives
and 177 arrival positives). Eight cached hourly payloads were read through
the existing cache contract. Training and validation had 24 and 19
airport/run/valid-time groups respectively, from three run times and just
two dates per split. They covered 51.54% and 38.24% of the frozen flight
cohorts. Strict weather PIT was 0 in both splits. Recovering cached bytes
does not verify historical publication time or the T−24h BTS schedule
snapshot. The audit operated on the four fixed 2023/2024 BTS inputs with
no new weather requests and no 2025 test labels.

The replay rebuilt both feature matrices and the weather join, verified all
validation flight IDs and labels in order, and reproduced all eight saved
model prediction vectors to CSV float precision. Patched end-to-end runs
reproduced both original prediction CSVs byte for byte. Package
`evidence/diagnostics.json` contains parameter hashes, per-model LightGBM
statistics, PR metrics and one-factor fit controls. The replay uses
LightGBM 4.7.0, pandas 3.1.1 and scikit-learn 1.9.1.

## Baseline consistency and categories

The PoV uses the same 17 scheduled columns and four interactions
(`Route`, `Airline_Route`, `Departure_Hour`, `Arrival_Hour`) as the V3-A
feature set in notebooks 05 and 07. The audited notebook cells were
05 cells 62, 68, 70, 72 and 74, and 07 cells 3–5 and 7, indexed from zero.
Both arms shared ordered rows, targets, the 21 baseline features and the
five train-derived pandas categorical definitions. V5 adds seven HRRR
fields. All numerical inputs were finite. Each model exposed classes
`[0, 1]`. Explicit notebook-style `categorical_feature` and LightGBM's
current automatic handling produced identical predictions in all eight
replays. The original large-cohort workflow used about 1.48 million
training rows. Its metric estimates and parameter regime should be
interpreted in that context.

| Categorical feature | Training levels | Unseen validation levels | Unseen validation rows |
| --- | ---: | ---: | ---: |
| Reporting_Airline | 15 | 0 | 0 |
| Origin | 6 | 0 | 0 |
| Dest | 177 | 13 | 15 of 652 (2.30%) |
| Route | 443 | 73 | 95 of 652 (14.57%) |
| Airline_Route | 636 | 186 | 216 of 652 (33.13%) |

Unseen categories became missing by design. A direct pandas cast raised a
future-compatibility warning, so the patch explicitly masks unseen values
before applying the same training dtype. The repeated fits used no
Route or Airline_Route splits in either round. `Dest` appears in the
second-round trees. Sparse route categories and novel route values limit
support for learning route interactions here. They do not explain the
constant first-round arrival scores.

## Training and ranking evidence

| Run | Arm | Target | Train ROC-AUC | Val ROC-AUC | Val AP | Distinct val scores | Trees / splits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | V3-A | Departure | 0.6739 | 0.4502 | 0.2273 | 215 | 100 / 100 |
| Frozen | V5-HRRR | Departure | 0.6795 | 0.4336 | 0.2221 | 199 | 100 / 100 |
| Frozen | V3-A | Arrival | 0.5000 | 0.5000 | 0.2715 | 1 | 1 / 0 |
| Frozen | V5-HRRR | Arrival | 0.5000 | 0.5000 | 0.2715 | 1 | 1 / 0 |
| Diagnostic | V3-A | Departure | 0.8761 | 0.4310 | 0.2254 | 629 | 100 / 591 |
| Diagnostic | V5-HRRR | Departure | 0.8723 | 0.4049 | 0.2084 | 619 | 100 / 583 |
| Diagnostic | V3-A | Arrival | 0.8184 | 0.4205 | 0.2352 | 610 | 200 / 1181 |
| Diagnostic | V5-HRRR | Arrival | 0.8281 | 0.4104 | 0.2381 | 604 | 200 / 1183 |

1. The arrival parameters requested 200 trees, a minimum of 400 instances
   per leaf and 90% bagging each round. The 818 eligible training rows
   cleared the runner's 800-row heuristic. Actual bag sizes in the
   verbosity-only replay ranged from 710 to 758, below the two-leaf
   requirement at the initial constant Hessian. Trees contained zero
   splits, and both arms predicted the training prevalence
   `130/818 = 0.158924` for every validation row. A single-factor
   training replay, disabling only bagging, restored 200 splits in
   each arrival arm. Lowering only the leaf minimum also restored splits.
   The controls diagnose feasibility and do not select a parameter policy.
2. The local second round changed six capacity, sampling and
   regularization settings together. It learned many splits in both
   arms, including 86 departure and 234 arrival weather splits in V5,
   but all validation AUCs were below 0.5 despite train AUCs of 0.818–0.876.
   Every second-round AP was below its validation prevalence baseline
   (departure 0.2347, arrival 0.2715). Brier scores also lost to the
   constant training prevalence in every nonconstant model in these
   two rounds. The larger train-validation gap supports overfit or
   nontransfer under this split. The bundled parameter changes do not
   isolate which change accounts for that gap.
3. Both frozen arrival models had trapezoidal PR-AUC 0.6357. Their
   tied scores give only the two PR endpoints and this integration
   returns `(1 + prevalence) / 2`. AP equals validation prevalence
   at 0.2715 and ROC-AUC is 0.5. Report AP beside its prevalence baseline
   and flag constant scores. Threshold F1 is separate from ranking.
   No first-round probability reached 0.5, and only 11 baseline and
   7 weather-arm departure scores reached it in the second round.

## Measured distribution and validity

| Eligible date | Rows | Weekday | Departure rate | Arrival rate |
| --- | ---: | --- | ---: | ---: |
| 2023-01-15 | 338 | Sunday | 12.13% | 14.50% |
| 2023-07-15 | 480 | Saturday | 20.00% | 16.88% |
| 2024-01-15 | 182 | Monday | 35.71% | 53.85% |
| 2024-07-15 | 470 | Monday | 18.72% | 16.81% |

Weekend status separates the training and validation eligible sets.
The January prevalence shift is large and the validation weather matching
selects different proportions of underlying flights by date. For second-
round V3-A arrival, January has a higher rate but lower mean score than
July (0.1144 vs 0.1491). ROC-AUC is 0.3439 within January and 0.5605 within
July. January delayed versus July nondelayed comparisons account for
38,318 of 84,075 validation positive-negative pairs, with concordance
0.2928. V5 exhibits the same direction and similarly poor concordance.
This directly links the ranking failure to the observed January slice and
cross-date misordering. These observations cannot identify whether the
shift was caused by weather, schedule, calendar mix or another factor.

The paired comparison is fair on its eligible flights and both runs use the
same evaluation data. It has only two weather dates per split and shares
forecasts across many flights. The repeated use of 2024 dates makes
the second round diagnostic. This is not an independent estimate of
parameter or weather lift. Strict PIT remains blocked by missing reviewed
historical weather availability and historical schedule snapshots.

## Minimal patch

The patch masks unseen categorical values explicitly. It augments the
existing exploratory metrics with validation prevalence, distinct scores,
count at the 0.5 threshold, constant-score status, requested and actual
trees, split count, and training AP/ROC-AUC. Original metric fields,
experimental scope and model parameters are retained. The focused test
checks matching paired matrices, unseen values and constant versus
nonconstant score diagnostics. A future run can detect the 0-split failure
directly without loading Booster internals from an earlier session.

Before estimating lift, freeze a validation protocol on more independent
weather days across calendar strata, with training-only temporal tuning
and separate weather-day holdouts. Require a nondegenerate baseline and
meaningful AP above prevalence on that design. Report per-date results
and uncertainty clustered by weather day. This Stage 7 review imposes
no new acquisition, release or parameter change.

## Verification

The original full regression had 659 passes and 3 opt-in network skips.
The full patched regression had 659 passes and the same 3 skips. It used
the original tests via symlinks with the modified PoV test and script
in scratch. The isolated patched PoV tests also had 12 passes. Ruff
lint and format checks passed. The branch, index and pre-existing
workspace changes stayed in place.

From the project root on `feature/v5-api-iem-mos`, check the extracted
patch with `git apply --check --index /path/to/stage7_baseline_diagnostics.patch`.
The patch includes only the PoV script, focused test and this report.
