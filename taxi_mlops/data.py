"""TLC snapshot ingestion and point-in-time cohorts."""

import shutil
import urllib.request
from pathlib import Path

import duckdb

from .common import digest, identity, lock, read_json, sync_dir, verify_files, write_json

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 5))
SCHEMA_VERSION = 1
SPLITS = {
    "train": {"start": "2024-01-01", "end": "2024-02-01", "as_of": "2024-02-03T05:00:00Z"},
    "validation": {"start": "2024-02-01", "end": "2024-03-01", "as_of": "2024-03-03T05:00:00Z"},
    "test": {"start": "2024-03-04", "end": "2024-04-01", "as_of": "2024-04-03T04:00:00Z"},
}
REQUIRED = {
    "tpep_pickup_datetime": {"TIMESTAMP", "TIMESTAMP_MS", "TIMESTAMP_NS", "TIMESTAMP_S"},
    "tpep_dropoff_datetime": {"TIMESTAMP", "TIMESTAMP_MS", "TIMESTAMP_NS", "TIMESTAMP_S"},
    "PULocationID": {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"},
}
FEATURES = ("pickup_zone", "pickup_hour", "pickup_weekday")


def connect():
    con = duckdb.connect(config={"threads": 4, "memory_limit": "1GB"})
    con.execute("SET TimeZone = 'UTC'")
    return con


def parquet(path):
    return f"read_parquet('{str(path).replace(chr(39), chr(39) * 2)}')"


def copy_parquet(con, query, target):
    con.execute(f"COPY ({query}) TO '{str(target).replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET)")


def raw_source(root, month, offline):
    name = f"yellow_tripdata_{month}.parquet"
    path = root / "raw" / name
    url = f"{BASE_URL}/{name}"
    if not path.is_file():
        if offline:
            raise ValueError(f"Missing cached file {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(path)
    return path, {"month": month, "url": url, "bytes": path.stat().st_size, "sha256": digest(path)}


def schema(con, path):
    columns = {row[0]: row[1] for row in con.execute(f"DESCRIBE SELECT * FROM {parquet(path)}").fetchall()}
    for name, accepted_types in REQUIRED.items():
        if columns.get(name) not in accepted_types:
            raise ValueError(f"{path.name}: invalid required field {name} ({columns.get(name)})")
    return columns


def local_time_is_valid(local, utc):
    """Exclude timestamps in DST gaps or folds because the source has no offset."""
    return f"""(
        timezone('America/New_York', {utc}) = {local}
        AND timezone('America/New_York', {utc} - INTERVAL '1 hour') <> {local}
        AND timezone('America/New_York', {utc} + INTERVAL '1 hour') <> {local}
    )"""


def build_month(con, path, month, destination):
    con.execute(
        """CREATE OR REPLACE TEMP TABLE trips AS
        SELECT file_row_number AS source_row, tpep_pickup_datetime::TIMESTAMP AS pickup_local,
               tpep_dropoff_datetime::TIMESTAMP AS dropoff_local, PULocationID AS zone,
               timezone('America/New_York', pickup_local) AS pickup_at,
               timezone('America/New_York', dropoff_local) AS dropoff_at
        FROM read_parquet(?, file_row_number=true)""",
        [str(path)],
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW valid_pickups AS
        SELECT *, concat('{month}:', source_row) AS row_key
        FROM trips
        WHERE strftime(pickup_local, '%Y-%m') = '{month}'
          AND zone BETWEEN 1 AND 265
          AND {local_time_is_valid("pickup_local", "pickup_at")}"""
    )
    copy_parquet(
        con,
        """SELECT row_key, pickup_at, pickup_local::DATE AS service_date,
                  zone::SMALLINT AS pickup_zone, hour(pickup_local)::UTINYINT AS pickup_hour,
                  (isodow(pickup_local) - 1)::UTINYINT AS pickup_weekday
           FROM valid_pickups""",
        destination / f"pickups-{month}.parquet",
    )
    copy_parquet(
        con,
        f"""SELECT row_key, dropoff_at + INTERVAL '24 hours' AS label_available_at,
                   epoch(dropoff_at - pickup_at) / 60.0 AS duration_minutes
            FROM valid_pickups
            WHERE {local_time_is_valid("dropoff_local", "dropoff_at")}
              AND epoch(dropoff_at - pickup_at) BETWEEN 60 AND 7200""",
        destination / f"labels-{month}.parquet",
    )
    raw = con.execute("SELECT count(*) FROM trips").fetchone()[0]
    pickups = con.execute("SELECT count(*) FROM valid_pickups").fetchone()[0]
    eligible = con.execute(
        f"SELECT count(*) FROM {parquet(destination / f'labels-{month}.parquet')}"
    ).fetchone()[0]
    return {"source_rows": raw, "pickup_rows": pickups, "rejected_pickups": raw - pickups, "eligible_labels": eligible}


def ingest(root, offline=False, months=MONTHS):
    with lock(root / "locks" / "ingest"):
        sources = [raw_source(root, month, offline) for month in months]
        snapshot_id = identity({"sources": [info for _, info in sources], "schema_version": SCHEMA_VERSION})
        destination = root / "snapshots" / snapshot_id
        if destination.exists():
            verify_files(destination, read_json(destination / "manifest.json")["files"])
            return snapshot_id
        stage = destination.with_suffix(".staging")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        with connect() as con:
            for path, info in sources:
                info["schema"] = schema(con, path)
                info.update(build_month(con, path, info["month"], stage))
        write_json(stage / "manifest.json", {
            "snapshot": snapshot_id, "schema_version": SCHEMA_VERSION, "sources": [info for _, info in sources],
            "splits": SPLITS, "files": {p.name: digest(p) for p in sorted(stage.glob("*.parquet"))},
        })
        stage.rename(destination)
        sync_dir(destination.parent)
        return snapshot_id


def snapshot_dir(root, snapshot_id):
    if len(snapshot_id) != 64 or any(char not in "0123456789abcdef" for char in snapshot_id):
        raise ValueError("Expected a SHA-256 snapshot identifier")
    directory = root / "snapshots" / snapshot_id
    manifest = read_json(directory / "manifest.json")
    if manifest["snapshot"] != snapshot_id or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Snapshot identity or feature schema mismatch")
    return directory


def cohort_query(directory, split, labels=True, cap=None):
    month = split["start"][:7]
    pickups = parquet(directory / f"pickups-{month}.parquet")
    outcomes = parquet(directory / f"labels-{month}.parquet")
    query = f"SELECT p.* FROM {pickups} p WHERE service_date >= ?::DATE AND service_date < ?::DATE"
    params = [split["start"], split["end"]]
    if labels:
        query = f"""SELECT p.*, l.duration_minutes FROM ({query}) p
                    JOIN {outcomes} l USING (row_key) WHERE l.label_available_at <= ?::TIMESTAMPTZ"""
        params.append(split["as_of"])
    if cap is not None:
        query += " ORDER BY md5(row_key), row_key LIMIT ?"
        params.append(cap)
    return query, params


def batches(con, directory, split, labels=True, cap=None, size=65_536):
    query, params = cohort_query(directory, split, labels, cap)
    with con.execute(query, params).to_arrow_reader(size) as reader:
        for batch in reader:
            yield batch.to_pandas()
