"""Fit, evaluate, and release pickup-only duration models."""

import getpass
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .common import code_revision, lock, now, read_json, write_json
from .data import FEATURES, SPLITS, batches, cohort_query, connect, snapshot_dir

MODEL = "yellow-taxi-duration"
ALIAS = "approved"
CAP = 250_000
MIN_GROUP = 30
MIN_SLICE = 1_000
SEED = 24
SLICES = ("weekday", "weekend", "peak", "off_peak")


def features(frame):
    return frame.loc[:, list(FEATURES)]


class HistoricalMedian(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.global_median_ = float(np.median(y))
        frame = X.assign(weekend=X.pickup_weekday >= 5, target=np.asarray(y))
        groups = frame.groupby(["pickup_zone", "pickup_hour", "weekend"]).target.agg(["median", "size"])
        self.medians_ = groups.loc[groups["size"] >= MIN_GROUP, "median"]
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_arrays([X.pickup_zone, X.pickup_hour, X.pickup_weekday >= 5])
        return self.medians_.reindex(keys).fillna(self.global_median_).to_numpy()


class ZoneCategories(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        counts = X.pickup_zone.value_counts().sort_index().sort_values(ascending=False, kind="stable")
        self.zones_ = list(counts.loc[counts >= MIN_GROUP].head(254).index)
        return self

    def transform(self, X):
        zones = X.pickup_zone.where(X.pickup_zone.isin(self.zones_), 0)
        return X.assign(pickup_zone=pd.Categorical(zones, categories=[0, *self.zones_]))


def client(root):
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)


def approved(registry, required=True):
    from mlflow.exceptions import MlflowException

    try:
        return int(registry.get_model_version_by_alias(MODEL, ALIAS).version)
    except MlflowException as error:
        if required or error.error_code not in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
            raise
        return None


def load_model(version):
    return mlflow.sklearn.load_model(f"models:/{MODEL}/{int(version)}")


def predict(model, frame):
    with threadpool_limits(limits=4):
        values = np.asarray(model.predict(features(frame)), dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Serving contract requires one finite positive prediction per pickup")
    return values


def train(root, snapshot, weak=False):
    directory = snapshot_dir(root, snapshot)
    with connect() as con:
        query, params = cohort_query(directory, SPLITS["train"])
        eligible_rows = con.execute(f"SELECT count(*) FROM ({query})", params).fetchone()[0]
        if eligible_rows == 0:
            raise ValueError("Training requires visible, eligible January labels")
        cohort = pd.concat(batches(con, directory, SPLITS["train"], cap=CAP), ignore_index=True)
    X = features(cohort)
    y = cohort.duration_minutes
    baseline = HistoricalMedian()
    if weak:
        from sklearn.dummy import DummyRegressor

        candidate = DummyRegressor(strategy="constant", constant=120)
    else:
        candidate = make_pipeline(
            ZoneCategories(),
            HistGradientBoostingRegressor(
                loss="absolute_error", categorical_features="from_dtype", early_stopping=False,
                max_iter=140, max_leaf_nodes=31, random_state=SEED,
            ),
        )
    registry = client(root)
    experiment = registry.get_experiment_by_name(MODEL)
    if experiment is None:
        experiment_id = registry.create_experiment(MODEL, artifact_location=(root / "mlartifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id

    versions = {}
    for role, model in (("baseline", baseline), ("candidate", candidate)):
        with threadpool_limits(limits=4):
            model.fit(X, y)
        example = X.head(5)
        signature = mlflow.models.infer_signature(example, predict(model, example))
        with mlflow.start_run(experiment_id=experiment_id, run_name=role) as run:
            mlflow.log_params({
                "role": role, "snapshot": snapshot, "train_rows": len(X),
                "train_cap": CAP, "eligible_train_rows": eligible_rows, "weak_candidate": weak and role == "candidate",
            })
            mlflow.log_dict({"splits": SPLITS, "code_revision": code_revision()}, "lineage.json")
            info = mlflow.sklearn.log_model(
                model, name=role, signature=signature, input_example=example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                registered_model_name=MODEL,
            )
            version = int(info.registered_model_version)
            for name, value in {"snapshot": snapshot, "role": role, "training_run": run.info.run_id}.items():
                registry.set_model_version_tag(MODEL, str(version), name, value)
            versions[role] = version
    return {"snapshot": snapshot, "train_rows": len(X), "eligible_train_rows": eligible_rows, "versions": versions}


def masks(frame):
    peak = frame.pickup_hour.isin([7, 8, 9, 16, 17, 18]).to_numpy()
    weekend = (frame.pickup_weekday >= 5).to_numpy()
    return {"weekday": ~weekend, "weekend": weekend, "peak": peak, "off_peak": ~peak}


def compare(directory, split, models):
    errors = {version: [] for version in models}
    slice_masks = {name: [] for name in SLICES}
    with connect() as con:
        pickups = sum(len(frame) for frame in batches(con, directory, split, labels=False))
        for frame in batches(con, directory, split):
            labels = frame.duration_minutes.to_numpy()
            for version, model in models.items():
                errors[version].append(np.abs(predict(model, frame) - labels))
            for name, mask in masks(frame).items():
                slice_masks[name].append(mask)
    if not pickups or not next(iter(errors.values())):
        raise ValueError("Evaluation requires pickup rows and visible, eligible labels")
    slice_masks = {name: np.concatenate(values) for name, values in slice_masks.items()}
    metrics = {}
    for version, parts in errors.items():
        error = np.concatenate(parts)
        metrics[str(version)] = {
            "pickup_rows": pickups, "label_rows": len(error), "coverage": len(error) / pickups,
            "mae": float(error.mean()), "median_absolute_error": float(np.median(error)),
            "p90_absolute_error": float(np.quantile(error, 0.9)),
            "slices": {
                name: {"rows": int(mask.sum()), "mae": float(error[mask].mean()) if mask.any() else None}
                for name, mask in slice_masks.items()
            },
        }
    return metrics


def validation_gate(metrics, candidate, baseline, incumbent):
    challenger = metrics[str(candidate)]
    benchmark = metrics[str(baseline)]
    current = metrics[str(incumbent or baseline)]
    return {
        "baseline_mae_improvement": challenger["mae"] <= 0.95 * benchmark["mae"] and benchmark["mae"] > 0,
        "incumbent_mae": challenger["mae"] <= current["mae"],
        "protected_slices": all(
            challenger["slices"][name]["mae"] <= 1.10 * segment["mae"]
            for name, segment in current["slices"].items() if segment["rows"] >= MIN_SLICE
        ),
        "serving_contract": True,  # compare loads and validates every prediction
    }


def evaluate(root, snapshot, candidate, baseline):
    directory = snapshot_dir(root, snapshot)
    registry = client(root)
    with lock(root / "locks" / "release"):
        incumbent = approved(registry, required=False)
    for version, role in ((candidate, "candidate"), (baseline, "baseline")):
        tags = registry.get_model_version(MODEL, str(version)).tags
        if tags.get("snapshot") != snapshot or tags.get("role") != role:
            raise ValueError(f"Version {version} is not the {role} for snapshot {snapshot}")
    versions = {candidate, baseline}
    if incumbent is not None:
        versions.add(incumbent)
    models = {version: load_model(version) for version in sorted(versions)}
    metrics = compare(directory, SPLITS["validation"], models)
    checks = validation_gate(metrics, candidate, baseline, incumbent)
    decision = {
        "id": uuid4().hex, "created_at": now(), "snapshot": snapshot, "candidate": candidate,
        "baseline": baseline, "incumbent": incumbent, "checks": checks, "passed": all(checks.values()),
        "split": SPLITS["validation"], "metrics": metrics, "code_revision": code_revision(),
        "thresholds": {"baseline_improvement": 0.05, "slice_regression": 0.10, "min_slice_rows": MIN_SLICE},
    }
    write_json(root / "decisions" / f"{decision['id']}.json", decision)
    tags = registry.get_model_version(MODEL, str(candidate)).tags
    registry.log_artifact(tags["training_run"], str(root / "decisions" / f"{decision['id']}.json"), "validation")
    for version, row in metrics.items():
        for name in ("mae", "median_absolute_error", "p90_absolute_error", "coverage"):
            tags = registry.get_model_version(MODEL, version).tags
            registry.log_metric(tags["training_run"], f"validation_{name}", row[name])
    return decision


def held_out(root, decision_id):
    decision = get_decision(root, decision_id)
    path = root / "reports" / f"test-{decision_id}.json"
    with lock(root / "locks" / f"test-{decision_id}"):
        if path.exists():
            return read_json(path)
        client(root)
        directory = snapshot_dir(root, decision["snapshot"])
        versions = {v: load_model(v) for v in (decision["candidate"], decision["baseline"])}
        report = {"decision": decision_id, "snapshot": decision["snapshot"], "split": SPLITS["test"],
                  "metrics": compare(directory, SPLITS["test"], versions)}
        write_json(path, report)
        return report


def get_decision(root, decision_id):
    if len(decision_id) != 32 or any(char not in "0123456789abcdef" for char in decision_id):
        raise ValueError("Invalid validation decision ID")
    return read_json(root / "decisions" / f"{decision_id}.json")


def recover_release(root, registry):
    pending = root / "release" / "pending.json"
    if not pending.exists():
        return
    event = read_json(pending)
    current = approved(registry, required=False)
    if current == event["previous"]:
        registry.set_registered_model_alias(MODEL, ALIAS, str(event["version"]))
    elif current != event["version"]:
        raise ValueError("Approved alias changed outside the release commands")
    write_json(root / "release" / "history" / f"{event['id']}.json", event)
    pending.unlink()


def release(root, action, version=None, decision_id=None, reason=None):
    registry = client(root)
    with lock(root / "locks" / "release"):
        recover_release(root, registry)
        previous = approved(registry, required=False)
        if action == "bootstrap":
            if previous is not None:
                raise ValueError("An approved model already exists")
            tags = registry.get_model_version(MODEL, str(version)).tags
            if tags.get("role") != "baseline":
                raise ValueError("Bootstrap requires a historical median baseline")
            predict(load_model(version), pd.DataFrame({
                "pickup_zone": [161, 265], "pickup_hour": [8, 23], "pickup_weekday": [0, 6],
            }))
        elif action == "promote":
            decision = get_decision(root, decision_id)
            if previous is None or not decision["passed"] or decision["incumbent"] != previous:
                raise ValueError("Promotion requires a passing validation decision for the current incumbent")
            version = decision["candidate"]
        elif action == "rollback":
            history = (root / "release" / "history").glob("*.json")
            prior_versions = {read_json(path)["version"] for path in history}
            if version not in prior_versions or version == previous:
                raise ValueError("Rollback requires a different, previously approved model version")
            load_model(version)
        event = {"id": uuid4().hex, "action": action, "previous": previous, "version": version,
                 "decision": decision_id, "reason": reason, "actor": getpass.getuser(), "at": now()}
        write_json(root / "release" / "pending.json", event)
        recover_release(root, registry)
        return event
