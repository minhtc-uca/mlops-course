"""Validate, fit on train, compare on validation and record two local runs."""

import argparse
import json
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from bike_demand.preflight import check_snapshot, sha256
from bike_demand.split import partition_summary, split_data
from bike_demand.tracking import (
    Tracking,
    record_metrics,
    timestamp,
    training_commit,
    working_tree_dirty,
    write_json,
)
from bike_demand.validate import FEATURES, load_data

DEFAULT_MODEL = {"n_estimators": 50, "max_depth": 10, "random_state": 42, "n_jobs": 1}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or set(config) != {"data", "features", "model"}:
        raise ValueError("config: require data, features and model")
    if config["data"] != "data/raw/hour.csv":
        raise ValueError("config: use the frozen data/raw/hour.csv snapshot")
    if config["features"] != list(FEATURES):
        raise ValueError(
            "config: require the contract's ordered features; no banned predictors"
        )
    if config["model"] != DEFAULT_MODEL:
        raise ValueError(
            "config: use the Week 1 RF preset; settings changes need review"
        )
    return config


def select_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    raise NotImplementedError(
        "TODO(Week 1, Lab 2): select only the ordered predictors, excluding "
        "labels and identities. See README 'Expected failures'."
    )


def compare_mean(
    train_target: pd.Series, validation_target: pd.Series
) -> tuple[float, float]:
    raise NotImplementedError(
        "TODO(Week 1, Lab 2): fit the comparator on training targets and "
        "compute validation MAE. See README 'Expected failures'."
    )


def fit_evaluate(train: pd.DataFrame, validation: pd.DataFrame, config: dict):
    raise NotImplementedError(
        "TODO(Week 1, Lab 2): fit the RF on train and evaluate only on "
        "validation rows. See README 'Expected failures'."
    )


def reload_validation(
    root: Path, validation: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, float]:
    # Only load locally produced/trusted model files, never arbitrary downloaded models.
    model = joblib.load(root / "artifacts/model.joblib")
    predictions = model.predict(validation.loc[:, features])
    if not np.isfinite(predictions).all():
        raise ValueError("reload: predictions must be finite")
    return predictions, float(mean_absolute_error(validation["cnt"], predictions))


def verify_readback(root: Path, validation: pd.DataFrame, result: dict) -> dict:
    metadata_path = root / "artifacts/model-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    model_path = root / "artifacts/model.joblib"
    for key in (
        "model_version",
        "model_run_id",
        "training_commit",
        "data_sha256",
        "features",
        "model_sha256",
    ):
        if metadata[key] != result[key]:
            raise ValueError(f"readback: metadata mismatch for {key}")
    if sha256(model_path) != result["model_sha256"]:
        raise ValueError("readback: saved model checksum mismatch")
    _, mae = reload_validation(root, validation, result["features"])
    if mae != result["validation_mae"]["random_forest"]:
        raise ValueError("readback: reloaded validation MAE differs from measured MAE")
    tracking = Tracking(root)
    with tempfile.TemporaryDirectory(prefix="bike-readback-") as temporary:
        destination = Path(temporary)
        for label, key in (
            ("naive_train_mean", "naive_run_id"),
            ("random_forest", "model_run_id"),
        ):
            run = tracking.read_run(result[key])
            if (
                run.info.status != "FINISHED"
                or run.data.tags["evaluation"] != "validation"
            ):
                raise ValueError("readback: run status/evaluation mismatch")
            if run.data.metrics["validation_mae"] != result["validation_mae"][label]:
                raise ValueError("readback: logged MAE mismatch")
            if run.data.params["data_sha256"] != result["data_sha256"]:
                raise ValueError("readback: logged snapshot mismatch")
            if run.data.params["training_commit"] != result["training_commit"]:
                raise ValueError("readback: logged commit mismatch")
            if json.loads(run.data.params["features"]) != result["features"]:
                raise ValueError("readback: logged feature order mismatch")
            context_path = tracking.download(
                result[key], "execution-context.json", destination / label
            )
            context = json.loads(context_path.read_text())
            if context["partitions"] != result["partitions"]:
                raise ValueError("readback: logged partition metadata mismatch")
        downloaded_model = tracking.download(
            result["model_run_id"], "model.joblib", destination
        )
        downloaded_metadata = tracking.download(
            result["model_run_id"], "model-metadata.json", destination
        )
        if downloaded_model.read_bytes() != model_path.read_bytes():
            raise ValueError("readback: logged model bytes differ from the saved model")
        if downloaded_metadata.read_bytes() != metadata_path.read_bytes():
            raise ValueError(
                "readback: logged metadata bytes differ from saved metadata"
            )
    return {
        "reload_mae": mae,
        "reload_mae_delta": mae - result["validation_mae"]["random_forest"],
        "metadata_matches": True,
        "logged_model_bytes_match": True,
    }


def train_baseline(root: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    frame = load_data(root / config["data"])
    snapshot = check_snapshot(root)
    partitions = split_data(frame)
    train, validation = partitions["train"], partitions["validation"]
    # Test partition is counted and retained, never passed to fitting/evaluation.
    mean, naive_mae = compare_mean(train["cnt"], validation["cnt"])
    model, predictions, model_mae = fit_evaluate(train, validation, config)
    if not np.isfinite(predictions).all():
        raise ValueError("model: predictions must be finite")
    context = {
        "timestamp": timestamp(),
        "training_commit": training_commit(root),
        "working_tree_dirty": working_tree_dirty(root),
        "data_sha256": snapshot["sha256"],
        "features": config["features"],
        "partitions": partition_summary(partitions),
        "config": config,
        "effective_model_settings": model.get_params(),
        "model_version": f"baseline-{uuid4().hex[:12]}",
    }
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "model.joblib"
    joblib.dump(model, model_path)
    reloaded_predictions, _ = reload_validation(root, validation, config["features"])
    if not np.array_equal(predictions, reloaded_predictions):
        raise ValueError("reload: saved model predictions differ from fitted model")
    tracking = Tracking(root)
    naive_id = tracking.start_run("naive_train_mean", context, {"train_mean": mean})
    tracking.finish(naive_id, naive_mae, [])
    model_id = tracking.start_run("random_forest", context, model.get_params())
    metadata = {**context, "model_run_id": model_id, "model_sha256": sha256(model_path)}
    metadata_path = artifacts / "model-metadata.json"
    write_json(metadata_path, metadata)
    tracking.finish(model_id, model_mae, [model_path, metadata_path])
    result = {
        **metadata,
        "naive_run_id": naive_id,
        "train_mean": mean,
        "validation_mae": {"naive_train_mean": naive_mae, "random_forest": model_mae},
    }
    result["readback"] = verify_readback(root, validation, result)
    result["readback"]["prediction_max_abs_delta"] = float(
        np.max(np.abs(predictions - reloaded_predictions))
    )
    record_metrics(root, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = train_baseline(Path.cwd(), args.config)
    except (ValueError, OSError, NotImplementedError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
