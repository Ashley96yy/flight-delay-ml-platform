# Flight Delay ML Platform

V5 weather development follows **API-first, small-data-first, evidence-driven**.

Stage 4 concludes the IEM timestamp investigation and adds an offline HRRR
pilot with separate strict and exploratory coverage reporting. The current
weather cache has no eligible 2023 training rows. No model was fit.

- [Final IEM publication-time findings](docs/iem_mos_availability_final.md)
- [HRRR pilot results and reproduction](docs/v5_stage4_hrrr_pov.md)

Stage 3 adds bounded conservative alignment and restricted interpolation,
with historical publication evidence kept separate from time matching:

- [Publication-time findings and evidence grades](docs/mos_availability_evidence.md)
- [Alignment rules and offline A/B results](docs/mos_alignment_stage3.md)
- [Integration status and safe cumulative import](docs/stage3_integration.md)

Both strategies match all 72 saved sample requests geometrically; **0/72 pass
the strict availability audit**. Unverified runs emit no model-ready weather
values, and no V5 model has been trained.

Current MOS research stage:

- [IEM historical MOS feasibility and measured small-sample results](docs/iem_mos_feasibility.md)
- [MOS integration, regression checks and offline reproduction](docs/v5_mos_integration.md)
- [Updated paired V3-A/V5 MOS experiment design](docs/v5_mos_proof_of_value.md)

IEM archived forecasts were retrieved for six airports on two dates in each
of 2023–2025. Historical publication time remains **availability_unverified**;
the model join rejects these rows. No model training or performance claim.

- [API research, provider boundary and safeguards](docs/v5_api.md)
- [Small-data V3-A versus V5 Proof-of-Value design](docs/v5_proof_of_value.md)

Stage 1 adds request-only provider adapters, a default-off bounded API client,
and a network-free dry run. Existing HRRR code and tests remain unchanged.
No API source has yet passed the full historical availability evidence gate.

```bash
PYTHONPATH=src python -m flight_delay.weather.api_dry_run \
  --input examples/weather_api_queries.json --provider open-meteo
```

The example times are synthetic planning inputs. Request counts are not measured
coverage, provider billing units or model performance. The CLI cannot execute
live or paid requests.
