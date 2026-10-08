"""Provided local tracking plumbing. No running MLflow viewer is required."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mlflow import MlflowClient


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def training_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def working_tree_dirty(root: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


class Tracking:
    def __init__(self, root: Path):
        root = root.resolve()
        (root / "tracking/artifacts").mkdir(parents=True, exist_ok=True)
        # Absolute URI avoids dependence on cwd while retaining the specified backend.
        self.uri = f"sqlite:///{(root / 'tracking/mlflow.db').as_posix()}"
        self.client = MlflowClient(tracking_uri=self.uri)
        experiment = self.client.get_experiment_by_name("bike-demand-baseline")
        self.experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment(
                "bike-demand-baseline",
                artifact_location=(root / "tracking/artifacts").as_uri(),
            )
        )

    def start_run(self, name: str, context: dict, settings: dict) -> str:
        run = self.client.create_run(
            self.experiment_id,
            tags={
                "mlflow.runName": name,
                "evaluation": "validation",
                "training_commit": context["training_commit"],
                "data_sha256": context["data_sha256"],
                "working_tree_dirty": str(context["working_tree_dirty"]),
            },
        )
        run_id = run.info.run_id
        try:
            self.client.log_param(run_id, "predictor", name)
            self.client.log_param(run_id, "features", json.dumps(context["features"]))
            self.client.log_param(run_id, "data_sha256", context["data_sha256"])
            self.client.log_param(run_id, "training_commit", context["training_commit"])
            for key, value in settings.items():
                self.client.log_param(run_id, key, value)
            for name, summary in context["partitions"].items():
                for key, value in summary.items():
                    self.client.log_param(run_id, f"{name}_{key}", value)
            self.client.log_dict(run_id, context, "execution-context.json")
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        return run_id

    def finish(self, run_id: str, mae: float, artifacts: list[Path]) -> None:
        try:
            self.client.log_metric(run_id, "validation_mae", mae)
            for path in artifacts:
                self.client.log_artifact(run_id, str(path))
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        self.client.set_terminated(run_id, status="FINISHED")

    def read_run(self, run_id: str):
        return self.client.get_run(run_id)

    def download(self, run_id: str, name: str, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        return Path(
            self.client.download_artifacts(run_id, name, str(destination.resolve()))
        )


def record_metrics(root: Path, result: dict) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "metrics.json", result)
    with (reports / "metrics-history.jsonl").open("a") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")


def timestamp() -> str:
    return datetime.now(UTC).isoformat()
