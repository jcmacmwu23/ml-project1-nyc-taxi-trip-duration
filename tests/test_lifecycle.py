import pandas as pd
import pytest

from taxi_mlops.batch import score
from taxi_mlops.common import read_json
from taxi_mlops.data import batches, connect, ingest, snapshot_dir
from taxi_mlops.models import approved, client, evaluate, release, train
from taxi_mlops.monitor import error_status, monitor, total_variation


def write_month(root, month, rows):
    path = root / "raw" / f"yellow_tripdata_{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def trip(pickup, duration=15, zone=161):
    timestamp = pd.Timestamp(pickup)
    return {
        "tpep_pickup_datetime": timestamp,
        "tpep_dropoff_datetime": timestamp + pd.Timedelta(minutes=duration),
        "PULocationID": zone,
        "DOLocationID": 100,
        "trip_distance": 4.2,
    }


def query_pickups(root, snapshot, split, labels):
    with connect() as con:
        return list(batches(con, snapshot_dir(root, snapshot), split, labels=labels))


def test_scoring_eligibility_and_features_ignore_outcomes(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    row = trip("2024-01-05 16:00")
    changed = {**row, "tpep_dropoff_datetime": pd.Timestamp("2024-01-05 21:00"),
               "DOLocationID": 200, "trip_distance": 90.0}
    write_month(root_a, "2024-01", [row])
    write_month(root_b, "2024-01", [changed])
    a = ingest(root_a, offline=True, months=("2024-01",))
    b = ingest(root_b, offline=True, months=("2024-01",))
    split = {"start": "2024-01-05", "end": "2024-01-06", "as_of": "2024-01-06T21:15:00Z"}
    assert a != b
    pd.testing.assert_frame_equal(query_pickups(root_a, a, split, False)[0],
                                  query_pickups(root_b, b, split, False)[0])
    assert query_pickups(root_a, a, split, True)[0].duration_minutes.tolist() == [15]
    assert query_pickups(root_b, b, split, True) == []
    assert query_pickups(root_a, a, {**split, "as_of": "2024-01-06T21:14:59Z"}, True) == []


@pytest.mark.parametrize("alteration", [{"PULocationID": "161"}, {"tpep_dropoff_datetime": "yesterday"}])
def test_ingest_rejects_invalid_schema(tmp_path, alteration):
    write_month(tmp_path, "2024-01", [{**trip("2024-01-05 16:00"), **alteration}])
    with pytest.raises(ValueError, match="invalid required field"):
        ingest(tmp_path, offline=True, months=("2024-01",))
    assert not list((tmp_path / "snapshots").glob("*/manifest.json"))


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory):
    root = tmp_path_factory.mktemp("lifecycle")
    for month, day in (("2024-01", "08"), ("2024-02", "05"), ("2024-04", "01")):
        rows = [trip(f"{month}-{day} {i // 20:02d}:00:{i % 60:02d}", 5 if i % 2 else 20, 161 + i % 2)
                for i in range(240)]
        write_month(root, month, rows)
    snapshot = ingest(root, True, ("2024-01", "2024-02", "2024-04"))
    normal = train(root, snapshot)["versions"]
    weak = train(root, snapshot, weak=True)["versions"]
    release(root, "bootstrap", normal["baseline"], reason="Fixture baseline")
    return root, snapshot, normal, weak


def test_reject_retry_rollback_and_monitor(lifecycle):
    root, snapshot, normal, weak = lifecycle
    baseline, candidate = normal["baseline"], normal["candidate"]
    bad = evaluate(root, snapshot, weak["candidate"], weak["baseline"])
    assert not bad["passed"]
    with pytest.raises(ValueError, match="passing validation"):
        release(root, "promote", decision_id=bad["id"], reason="Rejected fixture")
    assert approved(client(root)) == baseline

    decision = evaluate(root, snapshot, candidate, baseline)
    assert decision["passed"]
    release(root, "promote", decision_id=decision["id"], reason="Exercise rollback")

    day = "2024-04-01"
    partition = root / "predictions" / day
    with pytest.raises(RuntimeError, match="after staging"):
        score(root, snapshot, day, fail="after_stage")
    assert not (partition / "active.json").exists()
    with pytest.raises(RuntimeError, match="after commit"):
        score(root, snapshot, day, fail="after_commit")
    assert not (partition / "active.json").exists()

    release(root, "rollback", baseline, reason="Exercise version pin")
    original = score(root, snapshot, day)
    assert original["version"] == candidate
    assert score(root, snapshot, day) == original
    early = monitor(root, day, "2024-04-01T23:59:00-04:00", decision)
    late = monitor(root, day, "2024-04-04T06:00:00-04:00", decision)
    assert early["outcomes"]["labeled_rows"] == 0
    assert late["outcomes"]["labeled_rows"] == 240
    assert early["input"] == late["input"]

    replacement = score(root, snapshot, day, "restored", baseline, replace=True)
    assert replacement["version"] == baseline and replacement["supersedes"] == original["run_id"]
    assert score(root, snapshot, day) == original
    assert read_json(partition / "active.json")["run_id"] == "restored"
    assert len(list((partition / "runs").glob("*/predictions.parquet"))) == 2


def test_input_shift_and_error_are_independent():
    reference = {1: 0.5, 2: 0.5}
    assert total_variation(reference, {1: 0.8, 2: 0.2}) > 0.10
    assert error_status(2_000, 5, 5, True) == "within_threshold"
    assert total_variation(reference, reference) == 0
    assert error_status(2_000, 6, 5, False) == "pending"
    assert error_status(2_000, 6, 5, True) == "increased"
