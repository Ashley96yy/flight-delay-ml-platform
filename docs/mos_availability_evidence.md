# Stage 3 — historical MOS availability evidence

**Decision: availability_unverified for all saved real forecasts.** Nominal run
identity is well supported; an actual release timestamp or a verified historical
available-by bound for the complete matching payload has not been obtained.
Checks performed 2026-11-20 America/Los_Angeles; acquisition receipts use UTC.

## Five separate times

| Time | Meaning in this project | May establish strict availability? |
| --- | --- | --- |
| `model_run_time` | GFS nominal model reference; NBS nominal guidance cycle | No |
| `forecast_valid_time` | Time the point forecast describes, or a period's endpoint | No |
| `actual_publication_time` | Verified original release of the matching full run payload | Yes, if <= cutoff |
| `assumed_available_time` | Explicit schedule/lag assumption, kept separate | No |
| `flight_prediction_time` | Scheduled departure UTC minus exactly 24 elapsed hours | Defines the cutoff |

A historical full-run receipt can establish **available by** that receipt time
without establishing the exact publication time. Today's retrieval cannot
establish availability in 2023–2025. All timestamps must carry a timezone.

## Evidence grades and findings

Grades are project-specific: **A** = historical per-vintage release/receipt with
identity/content binding; **B** = primary implementation/schema or authoritative
field semantics; **C** = operational schedule/change notice or broad realtime
statement; **U** = unresolved. B/C are useful evidence, but cannot substitute
for A at the T−24h boundary.

| Evidence | Finding | Grade and limit |
| --- | --- | --- |
| [IEM MOS parser](https://github.com/akrherz/pyIEM/blob/main/src/pyiem/nws/products/mos.py) | `section_parser` derives `initts` from the station guidance header date/cycle; `MOSProduct.sql` writes that into `runtime`, and writes each forecast time into `ftime` | B: runtime is the nominal cycle, not IEM ingestion/release time. Current code is not proof of the exact historical ingestion version |
| [IEM API handler](https://github.com/akrherz/iem-web-services/blob/main/src/iemws/services/mos.py) | Explicit runtime selects the matching stored cycle. UTC output columns are derived from runtime/ftime | B: confirms identity and UTC interpretation, not release |
| [MOS table schema](https://github.com/akrherz/iem-database/blob/main/init/mos.sql) | Forecast table has runtime/ftime and weather values; no separate original publication or first-receipt column | B: the reviewed MOS schema/API cannot supply that timestamp |
| [AFOS schema](https://github.com/akrherz/iem-database/blob/main/init/afos.sql) and [retrieval code](https://github.com/akrherz/iem/blob/main/pylib/iemweb/afos/retrieve.py) | A separate product archive has `entered`; the retrieval path filters/orders by it | B/U: neither the column name nor retrieval code proves first-arrival semantics or a binding to the saved MOS payload. Do not relabel it publication time |
| [NWS SCN 22-131](https://www.weather.gov/media/notification/pdf2/scn22-131_nbm_v4.1.pdf), section 4 | NBS dissemination timing changes by cycle, including later average distribution for the 01/07/13/19 cycles | C: averages/change deltas are not per-run release guarantees or absolute upper bounds |
| [NWS NBM 4.2 timing table](https://www.weather.gov/media/mdl/nbm/docs_v42/Dissemination_Timing_Differences.pdf) and [archived NWS notice](https://mesonet.agron.iastate.edu/wx/afos/p.php?e=202404112110&pil=PNSWSH) | Distinguishes product cycles and dissemination channels/timing changes | C: supports separating cycle from distribution; does not certify the sampled historical runs |
| [IEM archive documentation](https://mesonet.agron.iastate.edu/mos/) | General realtime processing/archive origin | C: no per-vintage historical receipt |

Reviewed file blob identities are recorded in `artifacts/iem_mos_stage3/research_sources.json`; source links above use their maintained paths.

**Grade A evidence obtained: none.** GFS/NBS run times are not promoted to
publication times; no guessed fixed lag is accepted. WMO header timestamps must
also be interpreted for the specific product: a header date alone does not
prove first public receipt, whole-horizon completeness, or an unchanged vintage.

## Narrow original-product probes

Only two additional free text requests were made: nominal station identifiers
GFSATL and NBSATL, KATL substring filter, January 15, 2023 UTC, one result each.
Both returned HTTP 200 with an application-level “could not find” response,
29 bytes each. Their exact bodies and receipts are preserved under
`artifacts/iem_mos_stage3/publication_probe/`. This establishes no usable
publication evidence and **does not prove global absence of archived bulletins**;
the historical collective/split identifier mapping also needs confirmation.
No new MOS JSON, HRRR or observation data was acquired. Some NOAA VLab PDF
retrievals were blocked or too large; they were not used to manufacture timing
bounds. The NWS service notice and timing table above were read successfully.

## Cheapest next resolution

Before buying data, expanding dates or training, obtain an authoritative answer
to these three targeted questions from IEM/NOAA documentation or the archive
maintainer. No message has been sent on the user's behalf.

1. Which original collective/split product identifiers map to GFS/NBS cycles
   for the sampled years, and is the original complete station/run text retained?
2. Is any historical timestamp a true first-receipt/dissemination time, rather
   than a WMO validity/cycle field or a backfill insertion timestamp?
3. Can that timestamp bind the complete forecast horizon, station, nominal run
   and payload/version represented by the saved response, including corrections?

Then test one documented run per model with a small text/receipt query and
content comparison. A defensible historical available-by bound may suffice;
an exact publication instant is unnecessary. Until such evidence exists,
retain unverified status and keep the strict PoV blocked. More forecast values
would not solve this provenance gap.
