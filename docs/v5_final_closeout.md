# V5 Final Project Closeout

**Completed — Baseline ML Engineering; HRRR Enhancement Closed (No-Go)**

## Baseline and weather evidence

- **Baseline complete.** Stage 8 assessed cross-date generalization on
  398,612 flights in 2023 and 411,997 primary validation flights in 2024.
  Departure and arrival validation AP were 0.1932 and 0.2001, and ROC-AUC
  was 0.6257 and 0.6139, respectively. Both targets showed ranking signal
  across dates within the six-airport cohort. [Results and scope](../README.md#measured-baseline-results)
- **HRRR engineering PoV complete.** Earlier exploratory runs used verified
  HRRR caches. The acquisition path implements selective GRIB Range requests,
  decoding and feature validation, SHA-256 cache checks, and persistent
  request and byte limits. Offline preflight verified the same-cohort join
  and failure handling. HRRR's incremental predictive value remains
  unproven. [Baseline diagnostics](v5_stage7_baseline_diagnostics.md) · [Offline preflight](v5_stage9b_phase1_preflight.md)
- **Live smoke test stopped.** Stage 9B Phase 2A selected the 2023-01-16
  06:00 UTC forecast, F31, valid at 2023-01-17 13:00 UTC. DNS resolution for
  the NOAA S3 endpoint failed in the Work environment. The persistent ledger
  recorded one index request attempt, zero Range attempts, zero HTTP
  responses, zero response-body bytes, and no cache commit. Phase 2B did
  not run.
- **Strict T−24h point-in-time validation is unverified.** Historical
  availability is unverified for both complete HRRR forecast runs and the BTS
  schedules used as features. Exploratory weather runs used an assumed lag.

## Final decision

Close the weather enhancement track and stop data acquisition and tuning.
Keep the baseline, weather engineering proof of value, and audit records as
completed portfolio work. Any future restart is a new research stage with
its own scope and evidence requirements.
