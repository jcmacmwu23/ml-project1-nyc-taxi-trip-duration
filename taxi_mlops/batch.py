"""Pinned scoring runs and atomic publication for a local filesystem."""

import os
import re
import shutil
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from .common import code_revision, digest, identity, lock, now, read_json, sync_dir, write_json
from .data import MONTHS, SCHEMA_VERSION, batches, connect, parquet, snapshot_dir
from .models import approved, client, load_model, predict, recover_release

OUTPUT_SCHEMA = pa.schema([("row_key", pa.string()), ("prediction_minutes", pa.float64())])
RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}")


def verify_commit(directory):
    manifest = read_json(directory / "manifest.json")
    path = directory / "predictions.parquet"
    if digest(path) != manifest["sha256"]:
        raise ValueError("Committed predictions checksum mismatch")
    if pq.read_metadata(path).num_rows != manifest["counts"]["published"]:
        raise ValueError("Committed predictions row count mismatch")
    return manifest


def score(root, snapshot, service_date, run_id=None, version=None, replace=False, fail=None):
    day = date.fromisoformat(service_date)
    service_date = day.isoformat()
    month = service_date[:7]
    if month not in MONTHS:
        raise ValueError("Service date is outside the four-month source window")
    if replace and (run_id is None or version is None):
        raise ValueError("Replacement requires an explicit run ID and model version")
    run_id = run_id or f"initial-{snapshot[:16]}"
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid scoring run ID")
    directory = snapshot_dir(root, snapshot)
    partition = root / "predictions" / service_date
    run = partition / "runs" / run_id
    plan = partition / "plans" / f"{run_id}.json"
    pointer = partition / "active.json"
    revision = code_revision()
    with lock(root / "locks" / f"score-{service_date}"):
        active = read_json(pointer)["run_id"] if pointer.exists() else None
        if plan.exists():
            pinned = read_json(plan)
            if pinned["snapshot"] != snapshot or pinned["replace"] != replace or (
                version is not None and version != pinned["version"]
            ):
                raise ValueError("Run ID already belongs to different scoring inputs")
        else:
            if active is not None and not replace:
                raise ValueError("This date already has an active run. Use backfill to replace it.")
            registry = client(root)
            with lock(root / "locks" / "release"):
                recover_release(root, registry)
                if version is None:
                    version = approved(registry)
                else:
                    history = (root / "release" / "history").glob("*.json")
                    if version not in {read_json(path)["version"] for path in history}:
                        raise ValueError("Explicit scoring version must have been approved")
            pinned = {
                "run_id": run_id, "service_date": service_date, "snapshot": snapshot, "version": version,
                "code_revision": revision, "feature_schema_version": SCHEMA_VERSION,
                "replace": replace, "supersedes": active, "created_at": now(),
                "scoring_config": {"batch_rows": 65_536},
            }
            write_json(plan, pinned)

        if run.exists():
            manifest = verify_commit(run)
            if manifest["plan_sha256"] != identity(pinned):
                raise ValueError("Committed run does not match its pinned scoring plan")
            if active == run_id or (partition / "published" / f"{run_id}.json").exists():
                return manifest
        else:
            if pinned["code_revision"] != revision:
                raise ValueError("Retry requires the original scorer source revision")
            manifest = commit(root, directory, run, pinned, day, fail)
        if fail == "after_commit":
            raise RuntimeError("Injected failure after commit and before pointer update")
        if active != pinned["supersedes"]:
            raise ValueError("Active run changed since this run started")
        if active is not None:
            # Preserve evidence of prior publication before moving the pointer.
            write_json(partition / "published" / f"{active}.json", {"run_id": active})
        write_json(pointer, {"run_id": run_id, "manifest_sha256": digest(run / "manifest.json")})
        return manifest


def commit(root, directory, run, pinned, day, fail):
    filename = f"pickups-{day:%Y-%m}.parquet"
    checksums = read_json(directory / "manifest.json")["files"]
    if digest(directory / filename) != checksums[filename]:
        raise ValueError(f"Pickup snapshot checksum mismatch for {filename}")
    client(root)
    model = load_model(pinned["version"])
    run.parent.mkdir(parents=True, exist_ok=True)
    stage = run.with_suffix(".staging")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    path = stage / "predictions.parquet"
    split = {"start": day.isoformat(), "end": (day + timedelta(days=1)).isoformat()}
    count = 0
    with connect() as con, pq.ParquetWriter(path, OUTPUT_SCHEMA) as writer:
        for frame in batches(con, directory, split, labels=False, size=pinned["scoring_config"]["batch_rows"]):
            writer.write_table(pa.Table.from_arrays(
                [pa.array(frame.row_key, type=pa.string()), pa.array(predict(model, frame))],
                schema=OUTPUT_SCHEMA,
            ))
            count += len(frame)
    if count == 0:
        raise ValueError("No valid pickups in the requested partition")
    with connect() as con:
        actual, distinct = con.execute(
            f"SELECT count(*), count(DISTINCT row_key) FROM {parquet(path)}"
        ).fetchone()
    if actual != count or distinct != count:
        raise ValueError("Staged prediction rows failed reconciliation")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    manifest = {
        **pinned, "input_contract": "validated pickup snapshot",
        "counts": {"input": count, "rejected": 0, "scored": count, "published": count},
        "sha256": digest(path), "committed_at": now(),
        "plan_sha256": identity(pinned),
    }
    write_json(stage / "manifest.json", manifest)
    if fail == "after_stage":
        raise RuntimeError("Injected failure after staging and before commit")
    stage.rename(run)
    sync_dir(run.parent)
    return manifest
