# NYC taxi duration and batch MLOps

Predict yellow taxi trip duration from pickup zone, hour, and weekday. The local CPU workflow compares a historical median with a scikit-learn boosted tree, releases approved MLflow versions, and publishes Parquet predictions by service date.

~~~mermaid
flowchart LR
    TLC["NYC TLC Jan–Apr 2024 Parquet"] --> D["DuckDB ingestion and source manifest"]
    D --> P["Pickup features"]
    D --> L["Trip outcomes and availability"]
    P --> E["Chronological fit and evaluation"]
    L -->|"Visible by cutoff"| E
    E --> R["MLflow and SQLite"]
    R --> G["Validation gate and release"]
    G --> A["Approved model alias"]
    A -->|"Resolve and persist once"| B["Daily batch plan"]
    P --> B
    B --> C["Immutable Parquet and manifest"]
    C --> I["Atomic active partition pointer"]
    I --> M["Input shift and delayed error"]
    L -->|"Join as of replay clock"| M
    G -->|"Restore version and replay a day"| B
~~~

## Clean setup and replay

Use native CPython 3.13 on macOS or Linux. Allow a few GB for dependencies, raw files, snapshots, and results. The first setup fetches public data and installs dependencies.

~~~sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e '.[test]'
export MLFLOW_ENABLE_TELEMETRY=false
python -m pytest -q
python scripts/replay.py --root .local/demo
~~~

Use a fresh registry directory for each rehearsal. The report is written to `.local/demo/reports/replay.json`. For offline verification, copy or link the original `raw/` directory into a new root and pass `--offline`. For a deterministic promotion and rollback rehearsal, run `python scripts/replay.py --root .local/rollback-demo --fixture`. All compute and artifacts stay local. Databricks Lakeflow and Unity Catalog remain stretch work.

## Data and release contracts

- Training uses January pickups and outcomes visible by February 3. Validation uses February pickups and outcomes visible by March 3. March 4–31 is held out with an April 3 label cutoff. April 1–7 is the daily scoring replay.
- Simulated outcomes arrive 24 hours after drop-off. Public TLC files typically arrive around two months after the reporting month. Outcomes outside 1–120 minutes are excluded from evaluation. All valid pickups are scored regardless of the eventual outcome. DST gaps or folds are excluded where the source offset cannot be established.
- The median groups by pickup zone, hour, and weekday/weekend. Groups with fewer than 30 examples fall back to the global median. Both models train on up to 250,000 deterministically selected January rows.
- February MAE must improve by at least 5% over the median and match or beat the incumbent. Weekday, weekend, peak, and off-peak slices with at least 1,000 rows may regress by at most 10% against the incumbent. Peak hours are 07–09 and 16–18 inclusive. Evaluation predictions must be finite and positive.
- Ingestion records source and normalized file hashes and pickup rejection counts. Batch input counts start at the validated pickup snapshot. Use the ingestion manifest for upstream rejections.
- Keep the registry and artifacts together. The artifacts contain serialized scikit-learn models. Load only artifacts from your own trusted replay directory.

[Architecture diagram](docs/architecture.png) · [Operator runbook](docs/runbook.md) · [Five-minute demo](docs/demo.md) · [Measured results](docs/results.md) · [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Run manifests include a SHA-256 fingerprint of the scorer source and record the Git commit when available.
