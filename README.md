# Flight Delay ML Platform

**Completed — Baseline ML Engineering; HRRR Enhancement Closed (No-Go)**

An end-to-end ML engineering project that turns U.S. Bureau of Transportation
Statistics (BTS) flight records into reproducible datasets and evaluates
LightGBM models for flights departing or arriving at least 15 minutes late
(`DepDel15` and `ArrDel15`). The baseline uses scheduled flight information.
A separate weather proof of value exercised NOAA HRRR forecast acquisition
and caching. The incremental prediction value of weather remains unproven.

## Architecture

| Layer | Implementation |
| --- | --- |
| Data | Monthly BTS ingestion, schema and row validation, Parquet partitions, and manifests with SHA-256 checksums. [Data pipeline](src/flight_delay/data) |
| Features | A 21-column V3-A scheduled-flight contract with route and time interactions. Category vocabularies are learned on training data, with unseen validation categories mapped to missing. [Feature modules](src/flight_delay/features) · [Baseline audit](docs/v5_stage7_baseline_diagnostics.md) |
| Modeling and evaluation | Two LightGBM classifiers with frozen parameters, chronological train/validation splits, per-date and subgroup diagnostics, and automated tests. [Model code](src/flight_delay/models) |
| Weather PoV | NOAA HRRR selective GRIB byte-range requests, forecast identity checks, airport-level feature extraction, SHA-256-verified Parquet cache, and a persistent SQLite budget for the bounded acquisition path. [GRIB backend](src/flight_delay/weather/grib_backend.py) · [Offline safety preflight](docs/v5_stage9b_phase1_preflight.md) |

Core tools include Python 3.12, pandas, PyArrow, LightGBM, scikit-learn,
SQLite, pytest, and Ruff.

## Measured baseline results

The Stage 8 diagnostic trained on **398,612 flights across 365 dates in 2023**
and evaluated **411,997 flights across 364 dates in 2024**. It covers completed
flights departing from ATL, DFW, DEN, ORD, LAX, or SFO between 09:00 and 15:59
UTC, with matching local and UTC departure dates. The primary validation set
excludes January 15 and July 15, 2024, which were inspected in earlier pilots.

| Target | Validation positive rate | Validation Average Precision | Validation ROC-AUC |
| --- | ---: | ---: | ---: |
| Departure · `DepDel15` | 0.1303 | **0.1932** | **0.6257** |
| Arrival · `ArrDel15` | 0.1416 | **0.2001** | **0.6139** |

Average Precision exceeded the positive-rate reference for both targets.
The results show modest ranking ability across dates in this cohort.

## Reproduce the checks

On the `feature/v5-api-iem-mos` branch, use a Python 3.12 environment and run
the focused offline checks from the repository root.

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_stage9b_phase1.py
python -m ruff check src scripts tests
```

The Stage 8 reproduction materials are included in this repository:

- `reproduction/stage8/baseline_generalization.py`: archived evaluation script.
- `reproduction/stage8/PROTOCOL.md`: evaluation protocol.
- `reproduction/STAGE8_REPORT.md`: original experiment report.
- `reproduction/stage8/`: recorded metrics and summary.
- `artifacts/hyperparameter_optimization/best_params.json`: frozen model parameters.

An exact metric replay still requires the original 24 monthly 2023–2024
BTS Parquet inputs, which are not committed to GitHub.
The Stage 8 experiment was not rerun during this repository cleanup.

## Known limitations

- The reported scores cover six origins, one departure window, and one
  historical validation year. Part of 2024 overlapped earlier notebook
  evaluation. The results do not establish performance on unseen years.
- BTS schedule snapshots at T−24h and historical HRRR publication times
  remain unverified. The three-hour HRRR availability lag was an exploratory
  assumption.
- Earlier HRRR runs demonstrated selective retrieval and verified caching.
  The final bounded smoke test stopped at DNS resolution, so the final paired
  weather experiment did not run. [Final closeout](docs/v5_final_closeout.md)
