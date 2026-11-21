# Stage 8 baseline generalization protocol

This protocol was written before any Stage 8 fit or score inspection.

- Use the verified local BTS completed-flight Parquet and manifests for the
  24 months of 2023 and 2024. No weather inputs are used.
- Preserve the pilot domain of ATL, DFW, DEN, ORD, LAX and SFO origins,
  09:00–15:59 UTC scheduled departure, and agreement between the local flight
  date and UTC departure date. Use all rows in that domain on every available
  date. Read only the scheduled-feature inputs, scheduled flight number
  for identity checks, and the two approved target columns.
- Train on all 2023 dates. The primary 2024 evaluation omits January 15 and
  July 15, which the previous PoV already inspected. Report those dates as
  separate retrospective slices and include them in the complete monthly
  context. Also report January–August and September–December separately
  within the primary partition. The first block partly overlaps the old
  notebook's evaluation window.
- Use the original V3-A 21-column scheduled feature contract. Fit each of
  the five categorical vocabularies on 2023 rows alone, explicitly mapping
  unknown 2024 categories to missing. Pass those categorical columns to
  LightGBM. Fit one model for each target using the exact frozen repository
  parameter file with SHA-256
  `2f36b7b62d622d7458abd2bf54711ec24163df425b287b67ebb442c64be4b65c`.
  No early stopping, tuning, weather features, threshold fitting or
  model selection against 2024.
- Report pooled train and validation AP, positive rate, ROC-AUC, Brier
  score, Brier of the training target prior, score quantiles, requested
  and actual tree counts, splits and unique scores. Use score 0.5 only as
  a fixed reference in the distribution table.
- Report validation performance by airport, airline, calendar month and
  date. Train/validation distributions are reported on the same grouping.
  Keep single-class groups in the output with ranking metrics undefined.
  Preserve both January and July monthly context.
- Evaluate daily generalization using equal-date AP minus daily
  prevalence and ROC-AUC minus 0.5. Report the fraction of comparable
  dates above each reference. Estimate conditional intervals for the
  mean daily differences using a 2,000-draw bootstrap over nonoverlapping
  seven-calendar-day blocks anchored to 2024-01-01 with RNG seed 42.
  Resample blocks within calendar quarter and keep all flights within each
  date fixed. Describe ranking as supported for this domain and period
  when both bootstrap lower 95% bounds are above zero for both targets.
  Monthly and group variation remains relevant even if the gate passes.
- This retrospective diagnostic does not prove the flight schedule
  existed in the same form at T−24h. No inference is made about weather
  lift or performance on 2025.
