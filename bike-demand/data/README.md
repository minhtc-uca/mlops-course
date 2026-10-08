# Frozen UCI Bike Sharing snapshot

Source: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Download: [original ZIP](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip).

Attribution: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5W894>.

The UCI page was checked on 2026-10-07 and states **CC BY 4.0**:
<https://creativecommons.org/licenses/by/4.0/>. Sharing/adaptation requires
appropriate credit, a licence link and an indication of changes. The original
archive's documentation is retained in `UCI-original-Readme.txt`.

`raw/hour.csv` is extracted **without changing any bytes**. No `day.csv` is used.
The manifest records the measured SHA-256, byte size, header, row count, date
bounds and partition counts, not catalogue estimates. The source-normalised
weather values are preserved. Never overwrite the canonical CSV to create faults;
use copies or the tiny synthetic fixtures in `tests/fixtures/`.

The task estimates hourly rental count `cnt` from supplied calendar/weather.
It is not a station-level forecast or a forecast of future weather.

- Train: 2011-01-01 through 2011-12-31.
- Validation: 2012-01-01 through 2012-06-30.
- Test: 2012-07-01 through 2012-12-31. Count/retain these rows only in Weeks 1–4;
  evaluate the selected release candidate once in Week 5, without selecting a
  different model afterwards.

Missing hours are legitimate in the source; do not require 24 rows per day.
Preflight checks snapshot identity without executing the unfinished validator.
No runtime command downloads replacement data automatically.
