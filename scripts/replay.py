"""Rehearse the lifecycle in a fresh state directory and save the evidence."""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from taxi_mlops.cli import dispatch, parser
from taxi_mlops.common import read_json, write_json
from taxi_mlops.models import approved, client


def write_fixture(root):
    """Create a small separable cohort whose baseline falls back to the global median."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    for month, days in ((1, [8]), (2, [5]), (3, [4]), (4, range(1, 8))):
        rows = []
        for day in days:
            for i in range(240):
                pickup = pd.Timestamp(year=2024, month=month, day=day, hour=i // 20, second=i % 60)
                rows.append({
                    "tpep_pickup_datetime": pickup,
                    "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=5 if i % 2 else 20),
                    "PULocationID": 161 + i % 2,
                })
        pd.DataFrame(rows).to_parquet(raw / f"yellow_tripdata_2024-{month:02}.parquet")


def replay(root, offline):
    def command(*args):
        print("taxi", *args, file=sys.stderr, flush=True)
        return dispatch(parser().parse_args(["--root", str(root), *map(str, args)]))

    if (root / "mlflow.db").exists():
        raise ValueError("Replay requires a fresh registry. Choose a new --root.")
    snapshot = command("ingest", *(["--offline"] if offline else []))["snapshot"]
    normal = command("train", snapshot)
    baseline = normal["versions"]["baseline"]
    candidate = normal["versions"]["candidate"]
    bootstrap = command("bootstrap", baseline, "--reason", "Historical median serving check")
    decision = command("evaluate", snapshot, "--candidate", candidate, "--baseline", baseline)
    promotion = None
    if decision["passed"]:
        promotion = command("promote", decision["id"], "--reason", "Passed the frozen February gate")
    test = command("test-report", decision["id"])

    weak = command("train", snapshot, "--weak")["versions"]
    rejection = command("evaluate", snapshot, "--candidate", weak["candidate"], "--baseline", weak["baseline"])
    try:
        command("promote", rejection["id"], "--reason", "Demonstrate the validation guard")
    except ValueError as error:
        rejection_message = str(error)
    else:
        raise AssertionError("Weak candidate was promoted")
    selected = candidate if promotion else baseline
    assert approved(client(root)) == selected

    first = date(2024, 4, 1)
    partition = root / "predictions" / str(first)
    try:
        command("score", snapshot, first, "--fail", "after_stage")
    except RuntimeError as error:
        failure = str(error)
    else:
        raise AssertionError("Failure injection did not run")
    assert not (partition / "active.json").exists()
    initial = [command("score", snapshot, first + timedelta(days=offset)) for offset in range(7)]
    assert all(run["version"] == selected for run in initial)
    assert command("score", snapshot, first) == initial[0]

    early = command("monitor", first, "--as-of", "2024-04-01T23:59:59-04:00", "--decision", decision["id"])
    matured = [
        command("monitor", day, "--as-of", f"{day + timedelta(days=3)}T06:00:00-04:00", "--decision", decision["id"])
        for day in (first + timedelta(days=offset) for offset in range(7))
    ]
    assert early["outcomes"]["labeled_rows"] == 0
    assert matured[0]["outcomes"]["labeled_rows"] > 0
    assert early["input"] == matured[0]["input"]

    rollback = replacement = None
    if promotion:
        rollback = command("rollback", baseline, "--reason", "Rehearse restoring the prior approved version")
        replacement = command("backfill", snapshot, first, first, "--version", baseline,
                              "--run-id", "rollback-replay")[0]
        assert replacement["supersedes"] == initial[0]["run_id"]
        assert replacement["counts"] == initial[0]["counts"]
        assert replacement["version"] == baseline
        assert command("score", snapshot, first) == initial[0]
        assert read_json(partition / "active.json")["run_id"] == "rollback-replay"

    report = {
        "snapshot": read_json(root / "snapshots" / snapshot / "manifest.json"),
        "training": normal, "bootstrap": bootstrap, "validation": decision, "promotion": promotion,
        "held_out": test, "rejected_candidate": rejection, "rejection_message": rejection_message,
        "injected_failure": failure, "runs": initial, "early_monitor": early,
        "matured_monitor": matured, "rollback": rollback, "replacement": replacement,
        "final_approved": approved(client(root)),
    }
    write_json(root / "reports" / "replay.json", report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--offline", action="store_true")
    cli.add_argument("--fixture", action="store_true", help="rehearse promotion and rollback with synthetic trips")
    args = cli.parse_args()
    if args.fixture:
        write_fixture(args.root)
    report = replay(args.root.resolve(), args.offline or args.fixture)
    print(json.dumps({
        "report": str(args.root / "reports" / "replay.json"),
        "published_rows": sum(run["counts"]["published"] for run in report["runs"]),
        "promotion": report["promotion"],
        "rollback": report["rollback"],
    }, indent=2))
