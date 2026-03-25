"""
Preprocessing utilities for time series.

Intended responsibilities:
- Load CSV with schema: timestamp,value
- Sort by timestamp
- Optional normalization / windowing
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv


@dataclass
class TimeSeries:
    timestamps: list[datetime]
    values: list[float]


def load_csv_time_series(csv_path: str | Path) -> TimeSeries:
    csv_path = Path(csv_path)
    timestamps: list[datetime] = []
    values: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError(f"CSV schema mismatch in {csv_path}: expected header timestamp,value")
        for row in reader:
            # NAB format typically uses "%Y-%m-%d %H:%M:%S"
            ts = datetime.strptime(row["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
            timestamps.append(ts)
            values.append(float(row["value"]))

    # Ensure chronological order
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    timestamps = [timestamps[i] for i in order]
    values = [values[i] for i in order]
    return TimeSeries(timestamps=timestamps, values=values)

