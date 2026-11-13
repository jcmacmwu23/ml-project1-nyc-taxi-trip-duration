# Measured results

## Evaluation

NYC TLC yellow taxi records, January–April 2024. The four source files contained **13,069,067 rows**. Ingestion admitted **13,069,002 pickups** and rejected 65. Both models trained on 250,000 January rows. Error is measured in minutes.

| Cohort | Labels | Model | MAE | Median absolute error | 90th-percentile absolute error |
|---|---:|---|---:|---:|---:|
| February validation | 2,970,999 | Historical median | 6.836 | 4.642 | 14.917 |
| February validation | 2,970,999 | Boosted tree | 6.655 | 4.561 | 14.618 |
| March 4–31 holdout | 3,205,822 | Historical median | 7.244 | 4.800 | 16.050 |
| March 4–31 holdout | 3,205,822 | Boosted tree | 7.058 | 4.711 | 15.755 |

Label coverage was 98.79% for validation and 98.78% for holdout. The tree improved February MAE by **2.65%**, below the predeclared **5%** release threshold. It passed the incumbent, slice, and prediction checks. The historical median stayed approved. A deliberately weak 120-minute constant had **104.641 minutes MAE**, and its promotion command was rejected.

## Replay and recovery

The April 1–7 replay published **874,423 predictions** across seven service dates using the approved median. An injected failure on April 1 stopped after staging. Retrying published **93,868** predictions, and repeating the command returned the same manifest and checksum.

The public-data replay stayed on the bootstrap version because neither candidate qualified for release. The synthetic-trip rehearsal promoted version 2 through the same gate, scored 1,680 pickups across seven dates, rolled the alias back to version 1, and republished the 240-row April 1 partition from the original snapshot. Tests also cover failure after commit, a retry after alias movement, and replay of a superseded date.

## Input shift and delayed error

April 1 had zero visible outcomes at the end of the pickup day. By April 4 at 06:00 New York time, **92,559** outcomes were visible, for **98.61% coverage** and **6.791 minutes MAE**. Pickup distribution distances stayed the same across the two reports.

| Service date | Pickup hour distance | Pickup zone distance | Matured MAE | Observed signals |
|---|---:|---:|---:|---|
| April 3 | 0.078 | 0.057 | 8.245 | Error increased |
| April 7 | 0.173 | 0.112 | 6.706 | Pickup hour and zone shifted |

The input threshold was 0.10 total variation. The error threshold was 15% above the approved version's February MAE of 6.836. April 3 MAE was 20.6% higher. April 7 shows how pickup mix can cross the input threshold while prediction error stays within bounds. The replay simulates label availability 24 hours after drop-off.

## Verification environment

A fresh native CPython 3.13.18 environment on macOS ARM installed the pinned dependencies and passed all five fixture tests. Manifests use the scorer source SHA-256 as the code revision because the project folder has no Git commit. Linux and the Databricks stretch port were outside this verification run.

## Resume bullet

- Built and verified a local CPU batch ML lifecycle for NYC taxi duration using DuckDB, scikit-learn, and MLflow. Replayed 874,423 predictions across seven daily partitions and verified idempotent retries and delayed-label monitoring, and rehearsed promotion and rollback with synthetic trips.
