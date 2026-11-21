# V3-A versus V5 MOS — experiment specification, not executed

This supersedes the provider/field assumptions in the stage-1 PoV draft. The
time split and paired-comparison requirements are unchanged. The source pilot
established archive access but **not** historical release availability. This
experiment must stay blocked until that evidence and alignment policy pass.

## Freeze before any model fit

Compare a freshly fitted V3-A LightGBM with V5 = the same V3-A columns plus a
single prespecified MOS feature set on **identical train/validation flight IDs,
row order and untouched target arrays**. V3-A means the existing 21 scheduled
and route-interaction features, not congestion V3-C1 or seasonal-history V4.
Use the existing `best_params.json` parameters and random seeds unchanged.
Record source-file, feature-manifest, configuration and sample hashes.

Keep 2023 training; 2024 development validation (Jan–Aug as currently used);
2025 final test sealed. The 2025 archive/schedule probes in this stage are only
source diagnostics: they must not select predictors, tune parameters, calibrate
probabilities or optimize thresholds. No 2025 label has been read here.

## Source and time policy

Prefer one verified NBS source across all development partitions for the first
experiment, because its native fields include gust and physical visibility.
GFS is a separate possible ablation with category-based ceiling/visibility and
no assumed gust values. Do not mix GFS/NBS meanings or change source by year.
NBM version changes and unknown exact API model version must be disclosed.
Freeze final field selection using 2023/2024 semantics and diagnostics only.

Publication evidence must show that every selected run was publicly available
at or before scheduled departure minus exactly 24 hours. Model initialization,
HTTP success, retrieval today or a guessed cycle+lag cannot satisfy this gate.
Verify that the record represents original operational guidance, with no
unreviewed later revision/hindcast dependency. Unknown evidence remains
`availability_unverified` and cannot be promoted to pass.

Initial alignment policy: use only flights whose scheduled UTC instant exactly
matches a native valid time. Report this restricted population and how many
otherwise eligible flights it excludes. Do not floor a 06:15 flight to 06:00 or
forward-fill 3h/6h records into hourly forecasts. Any future bounded nearest-time
or window-overlap policy needs a separate feature version and a prespecified
distance/window definition before viewing gains. Probability windows remain
explicitly 3h or 6h intervals ending at the native valid time; they are not
hourly event probabilities or amounts.

Candidate first NBS set: Celsius temperature, wind speed and gust in m/s,
visibility in miles, ceiling in hundreds of feet plus unlimited flag, and
native T03 probability. Keep P06 as an optional separate ablation only after
its endpoint sparsity and 6h window selection are defined. Do not replace a
missing P06 with zero or mislabel T03 as T06. GFS alternatives retain categorical
VIS/CIG and native T06_1; no synthesized CAPE, reflectivity or precipitation
amount. Missing values require a prespecified rule and train-only preprocessing.

## Small development sample and cost bound

Start with ATL and SFO, using six fixed 2023 dates spread across the year and
four fixed 2024 dates within Jan–Aug, as in the stage-1 sample plan. Apply a
label-blind hash sample with a cap of 200 flights per airport/date (at most
4,000 sampled rows before grid/evidence exclusions). Freeze the sample before
weather collection and before reading comparative metrics. If exact-grid
intersection or class counts are too small, report that limitation; do not
choose easier dates based on scores. Further sample expansion needs a new
label-blind manifest, not a full-year acquisition loop.

Run a request-only deduplication plan before acquisition. A new experiment gets
its own reviewed shared request budget (initial cap at most 100 attempts), with
persistent receipts, response-byte cap and cache. The present CLI is deliberately
limited to its fixed 12-request source pilot. It does not implement the later
experiment or train models. No paid provider or automatic HRRR acquisition.

## Fit and report paired results

Use the exact same complete-evidence/weather intersection for V3-A and V5.
Also show V3-A on the full frozen sample to make selection effects visible.
Fit categories, missing-value handling and any calibration using training data
only (calibration via a designated temporal training holdout or out-of-fold
predictions), never 2025. Keep model parameters fixed for the first comparison;
do not tune after inspecting validation gains. The original full-data
`min_child_samples` can underfit this tiny pilot and must be disclosed.

For DepDel15 and ArrDel15, report Average Precision as the primary PR summary,
clearly distinguishing AP from trapezoidal PR-AUC. If trapezoidal PR-AUC is
also reported, give its integration definition and the same paired IDs.
Include ROC-AUC, Brier score, reliability diagrams with bin counts, calibration
intercept/slope where supported, and F1 only at a predeclared threshold. Do not
optimize a threshold on the final test set.

Report airport and season strata with sample/positive counts. Bootstrap paired
prediction deltas by airport-local flight day, preserving both models' rows in
each resample; report uncertainty and the very small number of independent day
groups. No post-hoc subgroup becomes the headline. Show coverage, publication
verification, grid exclusions, field missingness, acquisition/fit time and
actual HTTP/body-byte cost alongside any gain. A blocked or no-gain result is
valid evidence. Agree a numerical utility target before the future fit.

No training, fitted probabilities, calibration output, PR score or gain exists
from this stage. The decision to expand comes after source eligibility and a
paired small-data result, not before.
