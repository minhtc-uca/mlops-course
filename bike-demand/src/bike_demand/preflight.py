"""Check the frozen snapshot and environment, without executing exercises."""

import csv
import hashlib
import json
import sys
import tempfile
from importlib import import_module
from pathlib import Path

REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "configs/baseline.yaml",
    "data/manifest.json",
    "data/raw/hour.csv",
    "reports/week-01-team.md",
    ".github/workflows/ci.yml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_project_files(root: Path) -> list[str]:
    return [
        f"Missing project file: {name}"
        for name in REQUIRED_FILES
        if not (root / name).is_file()
    ]


def check_snapshot(root: Path) -> dict:
    manifest = json.loads((root / "data/manifest.json").read_text())
    data = root / "data/raw/hour.csv"
    if sha256(data) != manifest["sha256"] or data.stat().st_size != manifest["bytes"]:
        raise ValueError(
            "hour.csv: snapshot checksum/size does not match data/manifest.json"
        )
    with data.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if header != manifest["header"] or rows != manifest["rows"]:
        raise ValueError(
            "hour.csv: snapshot header/rows do not match data/manifest.json"
        )
    return manifest


def check_runtime() -> list[str]:
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Use Python 3.12; run uv sync --locked.")
    for name in (
        "pandas",
        "numpy",
        "sklearn",
        "mlflow",
        "joblib",
        "yaml",
        "pytest",
        "ruff",
    ):
        try:
            import_module(name)
        except ImportError:
            errors.append(f"Missing {name}; run uv sync --locked.")
    return errors


def check_writable(root: Path) -> None:
    for name in ("artifacts", "reports", "tracking", "tracking/artifacts"):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass


def main() -> int:
    root = Path.cwd()
    errors = check_project_files(root) + check_runtime()
    if not errors:
        try:
            check_snapshot(root)
            check_writable(root)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            errors.append(str(error))
    if errors:
        print("Preflight failed:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print("Preflight passed: imports, frozen snapshot and writable outputs.")
    print("This does not complete exercises, verify GitHub CI, or certify other OSes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
