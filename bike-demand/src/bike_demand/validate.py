"""CSV loading and explicit contract checks; never repair the input."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)
REQUIRED = ("instant", "dteday", *FEATURES, "casual", "registered", "cnt")
DOMAINS = {
    "season": (1, 4),
    "yr": (0, 1),
    "mnth": (1, 12),
    "hr": (0, 23),
    "holiday": (0, 1),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
}
NORMALISED = ("temp", "atemp", "hum", "windspeed")
COUNTS = ("cnt", "casual", "registered")


def check_domains(frame: pd.DataFrame) -> None:
    raise NotImplementedError(
        "TODO(Week 1, Lab 2): validate domains, finite ranges and date "
        "windows. See README 'Expected failures'."
    )


def load_data(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("data: require at least one row")
    for field in REQUIRED:
        if frame[field].isna().any():
            raise ValueError(f"{field}: missing values are not allowed")
        try:
            if field == "dteday":
                frame[field] = pd.to_datetime(
                    frame[field], format="%Y-%m-%d", errors="raise"
                )
            else:
                frame[field] = pd.to_numeric(frame[field], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{field}: invalid {'date' if field == 'dteday' else 'numeric'} values"
            ) from error
    ids = frame["instant"]
    if (
        not (np.isfinite(ids) & (ids > 0) & (ids % 1 == 0)).all()
        or ids.duplicated().any()
    ):
        raise ValueError("instant: require unique positive integers")
    check_domains(frame)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    try:
        frame = load_data(args.data)
    except (ValueError, OSError, NotImplementedError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    print(f"Validated {len(frame)} rows; input bytes unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
