import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyRegressor

from bike_demand import __version__, preflight
from bike_demand.tracking import Tracking, record_metrics, write_json
from bike_demand.train import load_config

pytestmark = pytest.mark.infra


def test_installed_package():
    assert version("bike-demand") == __version__ == "0.1.0"
    assert sys.version_info[:2] == (3, 12)
    config = tomllib.loads(Path("pyproject.toml").read_text())
    assert config["project"]["requires-python"] == ">=3.12,<3.13"
    assert Path(".python-version").read_text().strip() == "3.12"


def test_config():
    config = load_config(Path("configs/baseline.yaml"))
    assert len(config["features"]) == 12
    assert config["model"]["n_jobs"] == 1


def test_config_rejects_leaky_predictors(tmp_path):
    source = Path("configs/baseline.yaml").read_text().replace("- temp", "- cnt")
    config = tmp_path / "leak.yaml"
    config.write_text(source)
    with pytest.raises(ValueError, match="ordered features"):
        load_config(config)


def test_preflight_does_not_execute_exercises(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not run exercises")

    monkeypatch.setattr("bike_demand.validate.check_domains", forbidden)
    assert preflight.main() == 0


def test_snapshot_identity():
    manifest = preflight.check_snapshot(Path.cwd())
    assert manifest["rows"] == sum(manifest["partition_rows"].values())
    assert manifest["licence"] == "CC BY 4.0"


def test_snapshot_tampering_is_rejected(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    for name in ("data/raw/hour.csv", "data/manifest.json"):
        (tmp_path / name).write_bytes(Path(name).read_bytes())
    with (tmp_path / "data/raw/hour.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        preflight.check_snapshot(tmp_path)


def test_preflight_outside_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert preflight.main() == 1
    assert "Missing project file: uv.lock" in capsys.readouterr().out


def test_preflight_cli():
    result = subprocess.run(
        [sys.executable, "-m", "bike_demand.preflight"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tracking_readback_and_history(tmp_path):
    # Plumbing only: this tiny model is not a baseline solution or course result.
    context = {
        "training_commit": "synthetic-infra",
        "working_tree_dirty": False,
        "data_sha256": "synthetic-fixture",
        "features": ["x"],
        "partitions": {
            "train": {"rows": 2, "date_min": "2011-01-01", "date_max": "2011-01-02"}
        },
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(DummyRegressor().fit([[0], [1]], [1, 3]), model_path)
    metadata_path = tmp_path / "model-metadata.json"
    write_json(metadata_path, context)
    tracker = Tracking(tmp_path)
    run_id = tracker.start_run("infra_only", context, {"kind": "fixture"})
    tracker.finish(run_id, 1.0, [model_path, metadata_path])
    run = tracker.read_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics == {"validation_mae": 1.0}
    assert run.data.params["features"] == '["x"]'
    assert run.data.tags["evaluation"] == "validation"
    assert (
        tracker.download(run_id, "model.joblib", tmp_path / "readback").read_bytes()
        == model_path.read_bytes()
    )
    assert (
        tracker.download(
            run_id, "model-metadata.json", tmp_path / "readback"
        ).read_bytes()
        == metadata_path.read_bytes()
    )
    record_metrics(tmp_path, {"sequence": 1, "fixture_run": run_id})
    record_metrics(tmp_path, {"sequence": 2, "fixture_run": run_id})
    assert json.loads((tmp_path / "reports/metrics.json").read_text())["sequence"] == 2
    history = [
        json.loads(line)
        for line in (tmp_path / "reports/metrics-history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["sequence"] for row in history] == [1, 2]
