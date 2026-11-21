# IEM MOS historical availability decision

**Decision: close the IEM historical publication-time investigation.** The IEM
provider and archived pilot forecasts remain available for reproducibility.
The saved GFS and NBS records retain `availability_unverified`. The Stage 3
strict audit passed 0 of 72 flight/model checks. Stage 4 sent zero new forecast
or AFOS product-data requests.

| Clock | What it establishes |
| --- | --- |
| MOS `runtime` | Nominal station guidance cycle parsed from the forecast text. GFS uses the model reference cycle. NBS has a blend guidance cycle. |
| MOS `ftime` | UTC point or period endpoint the forecast describes. |
| WMO/MND product time | Timestamp parsed from the bulletin heading or product text. IEM stores this as AFOS `entered`. |
| Actual publication time | Time the original publisher made the specific version and relevant complete product available. Unverified for the pilot. |
| Actual receipt time | Time a trusted historical receiver obtained that version. A content-bound receipt can establish an upper bound on availability. Unverified for the pilot. |
| Assumed availability time | Policy estimate documented separately from publication and receipt. |
| Flight prediction time | Scheduled departure UTC minus 24 elapsed hours. Verified evidence must put availability at or before this cutoff. |

## Evidence

Grade A means a reliable historical release or receipt for the matching product
version and complete payload used by the audit. Grade B means primary source
semantics or implementation. Grade C means general operational context. No
Grade A evidence was obtained.

- **B, product identifiers.** The IEM finder documents `MAVxxx` for GFS MOS
  and `NBSxxx` for short-range NBM. The collective routes cover MAV and
  KWNO/NGRID NB-family products. The splitter constructs the station PIL
  from the first three characters of the collective PIL and the last three
  characters of the four-letter station. For the pilot airports the candidate
  identifiers therefore use ATL, DFW, DEN, ORD, LAX and SFO with the MAV or NBS
  prefix. Stage 3's `GFSATL` probe used the wrong identifier. The negative
  `NBSATL` response establishes no coverage result.
  [IEM finder](https://mesonet.agron.iastate.edu/wx/afos/),
  [routing source](https://github.com/akrherz/pyWWA/blob/main/pqact.d/pqact_afos.conf),
  [splitter](https://github.com/akrherz/pyWWA/blob/main/src/pywwa/workflows/split_mav.py).
- **B, AFOS semantics.** The realtime `afos_dump` workflow compares the
  current receipt-side clock `utcnow` with the parsed `nws.valid` and
  inserts `nws.valid` into `entered`. The splitter stores the collective
  bulletin's `prod.valid` and reconstructs the station header using it.
  The historical backfill importer also writes the old bulletin's parsed
  `prod.valid`. The WMO parser derives the time from the heading and can
  use MND metadata to determine the final parsed time. This schema and
  workflow provide no historical receipt timestamp through `entered`.
  [Realtime ingestion](https://github.com/akrherz/pyWWA/blob/main/src/pywwa/workflows/afos_dump.py),
  [splitter](https://github.com/akrherz/pyWWA/blob/main/src/pywwa/workflows/split_mav.py),
  [WMO parser](https://github.com/akrherz/pyIEM/blob/main/src/pyiem/wmo.py),
  [historical importer](https://github.com/akrherz/pyIEM/blob/main/scripts/util/poker2afos.py).
- **B, binding limits.** Station PIL and guidance cycle can narrow a text
  lookup. A matching pilot row would also require the correct year, forecast
  times and fields, a specific product version or correction, and evidence
  covering the complete payload. The MOS schema exposes nominal runtime and
  validity with no publication/receipt column. The archive documentation
  describes realtime and backfill sources. The reviewed sources do not
  establish the timestamp and content binding required for any sampled run.
  [MOS schema](https://github.com/akrherz/iem-database/blob/main/init/mos.sql),
  [AFOS API guide](https://mesonet.agron.iastate.edu/nws/text.php),
  [AFOS archive data documentation](https://mesonet.agron.iastate.edu/info/datasets/afos.html).
- **C, dissemination schedules.** NOAA's change notice describes NBM
  distribution offsets. WMO headings specify bulletin date/time metadata.
  Neither supplies a historical per-version receipt for the products in
  the pilot. [NOAA notice](https://www.weather.gov/media/notification/pdf2/scn22-131_nbm_v4.1.pdf),
  [NWS WMO heading reference](https://www.weather.gov/tg/head).

Reviewed Git blob identities for the key implementation sources are below.
They identify the reviewed source, with no claim about the exact historical
deployment version.

| Source | Git blob SHA |
| --- | --- |
| pyWWA routing | `cc5ef6d2ba8ebd31f214ed4f55159e77a999b1b0` |
| pyWWA split_mav | `992454d94b0085b827026d75802c7237ed983a38` |
| pyWWA afos_dump | `f394de82708ba99dc1d1c47144328ca7bbb91b6b` |
| pyIEM WMO parser | `55b4346596c43acd315c4cec9bfcbbff4d3c5739` |
