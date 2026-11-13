# Five-minute demo

Run the TLC and fixture rehearsals in advance. Use the TLC replay for evaluation, rejection, retry, and monitoring. Use the synthetic-trip replay for promotion and rollback. Open both replay reports and their April 1 partition directories.

| Time | Show | Say |
|---|---|---|
| 0:00–0:45 | Architecture and source manifest | The model uses fields available at pickup. Replay outcomes arrive 24 hours after drop-off. Pickup validation and target eligibility run independently. |
| 0:45–1:40 | February decision, March report, and rejected candidate | Compare the tree and grouped median on the same rows. Show the improvement threshold and one protected slice. Open the weak model decision and show that the approved alias stayed on its prior version. |
| 1:40–2:35 | Injected failure, saved scoring plan, and retry | The first attempt stopped after staging. The retry reused the pinned version. A repeat returned the same committed manifest and checksum. |
| 2:35–3:25 | Early and matured April 1 reports | Show the same input-shift readings next to the changing label coverage and error status. On April 3, MAE increased while both input distances stayed below threshold. On April 7, both input distances crossed the threshold and MAE stayed within bounds. |
| 3:25–4:25 | Release history and April 1 pointer | Show the prior and replacement manifests, the supersession link, and the selected run. |
| 4:25–5:00 | Measured results | State the held-out MAE, replay row count, and simulated label assumption. The run used local CPU compute. |

## Python exercises

1. Add a `taxi status DATE` command that checks the active pointer, manifest checksum, and prediction row count. Test an absent partition and a tampered manifest.
2. Add a monitoring breakdown by pickup zone with label coverage, MAE, and minimum-volume suppression. Reconcile the zone totals to the overall report when some labels are unavailable.
3. Implement a dry-run backfill command that shows the current run, intended version, snapshot, and proposed supersession for each date. Test a range with an unscored date and verify that it writes no files.
