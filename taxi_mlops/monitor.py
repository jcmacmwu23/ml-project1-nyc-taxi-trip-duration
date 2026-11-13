"""Compare pickup mix immediately and prediction error as outcomes arrive."""

from datetime import date, datetime

from .batch import verify_commit
from .common import digest, identity, read_json, write_json
from .data import SPLITS, connect, parquet, snapshot_dir

SHIFT_THRESHOLD = 0.10
ERROR_THRESHOLD = 0.15
MIN_LABELS = 1_000


def distribution(con, path, feature, start, end):
    rows = con.execute(
        f"""SELECT {feature}, count(*) FROM {parquet(path)}
        WHERE service_date >= ?::DATE AND service_date < ?::DATE GROUP BY 1""",
        [start, end],
    ).fetchall()
    count = sum(n for _, n in rows)
    if not count:
        raise ValueError("Monitoring reference and observed distributions require pickup rows")
    return {value: n / count for value, n in rows}, count


def total_variation(reference, observed):
    return 0.5 * sum(abs(reference.get(key, 0) - observed.get(key, 0)) for key in reference.keys() | observed.keys())


def error_status(rows, mae, reference_mae, final):
    if not final or rows < MIN_LABELS or reference_mae is None:
        return "pending" if not final else "insufficient_reference_or_labels"
    return "increased" if mae > reference_mae * (1 + ERROR_THRESHOLD) else "within_threshold"


def monitor(root, service_date, as_of, decision):
    day = date.fromisoformat(service_date)
    clock = datetime.fromisoformat(as_of)
    if clock.utcoffset() is None:
        raise ValueError("Monitoring cutoff requires an explicit timezone")
    service_date = day.isoformat()
    partition = root / "predictions" / service_date
    active = read_json(partition / "active.json")
    run = partition / "runs" / active["run_id"]
    if digest(run / "manifest.json") != active["manifest_sha256"]:
        raise ValueError("Published pointer manifest checksum mismatch")
    manifest = verify_commit(run)
    directory = snapshot_dir(root, manifest["snapshot"])
    reference = snapshot_dir(root, decision["snapshot"])
    split = SPLITS["validation"]
    month = service_date[:7]
    shifts = {}
    with connect() as con:
        for feature in ("pickup_zone", "pickup_hour"):
            baseline, ref_rows = distribution(
                con, reference / "pickups-2024-02.parquet", feature, split["start"], split["end"]
            )
            observed, rows = distribution(con, directory / f"pickups-{month}.parquet", feature, service_date,
                                          str(day.fromordinal(day.toordinal() + 1)))
            distance = total_variation(baseline, observed)
            shifts[feature] = {"total_variation": distance, "flag": distance >= SHIFT_THRESHOLD}
        predictions = parquet(run / "predictions.parquet")
        outcomes = parquet(directory / f"labels-{month}.parquet")
        labeled, mae, bias = con.execute(
            f"""SELECT count(*), avg(abs(prediction_minutes-duration_minutes)),
                       avg(prediction_minutes-duration_minutes)
                FROM {predictions} JOIN {outcomes} USING (row_key)
                WHERE label_available_at <= ?::TIMESTAMPTZ""",
            [clock.isoformat()],
        ).fetchone()
        final = con.execute(
            """SELECT ?::TIMESTAMPTZ >=
                timezone('America/New_York', ?::DATE::TIMESTAMP + INTERVAL '1 day')
                + INTERVAL '26 hours'""",
            [clock.isoformat(), service_date],
        ).fetchone()[0]
    baseline_metrics = decision["metrics"].get(str(manifest["version"]), {})
    reference_mae = baseline_metrics.get("mae")
    report = {
        "service_date": service_date, "as_of": clock.isoformat(), "run_id": manifest["run_id"],
        "version": manifest["version"], "snapshot": manifest["snapshot"], "reference_decision": decision["id"],
        "input": {"pickup_rows": rows, "reference_pickup_rows": ref_rows,
                  "reference_average_daily_rows": ref_rows / 29, "shift": shifts,
                  "rejected_scoring_rows": manifest["counts"]["rejected"],
                  "source_month_rejected_pickups": next(
                      source["rejected_pickups"] for source in read_json(directory / "manifest.json")["sources"]
                      if source["month"] == month
                  )},
        "outcomes": {"labeled_rows": labeled, "coverage": labeled / rows, "mae": mae,
                     "signed_error": bias, "reference_mae": reference_mae,
                     "label_window_closed": final, "status": error_status(labeled, mae, reference_mae, final)},
        "thresholds": {"total_variation": SHIFT_THRESHOLD, "mae_increase": ERROR_THRESHOLD, "min_labels": MIN_LABELS},
    }
    name = identity({"date": service_date, "run": manifest["run_id"], "as_of": report["as_of"], "decision": decision["id"]})
    write_json(root / "reports" / f"monitor-{name}.json", report)
    return report
