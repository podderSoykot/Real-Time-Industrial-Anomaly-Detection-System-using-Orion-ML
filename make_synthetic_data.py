"""
Generate a synthetic time-series dataset for this project.

Creates:
  data/synthetic_machine_failure.csv
  data/synthetic_machine_failure_train.csv
  data/synthetic_machine_failure_test.csv
  data/synthetic_machine_failure_meta.json

Schema for each CSV:
  timestamp,value
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Meta:
    name: str
    mode: str
    n_points: int
    start: str
    freq: str
    train_ratio: float
    train_rows: int
    test_rows: int
    anomaly_1: tuple[int, int, float]
    anomaly_2: tuple[int, int, float]
    seed: int
    value_mean: float
    value_std: float
    value_min: float
    value_max: float


def write_points_csv(path: Path, points: list[tuple[datetime, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        for ts, v in points:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{v:.10g}"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--start", type=str, default="2024-01-01 00:00:00")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anom1-start", type=int, default=400)
    parser.add_argument("--anom1-end", type=int, default=420)
    parser.add_argument("--anom1-shift", type=float, default=3.0)
    parser.add_argument("--anom2-start", type=int, default=1200)
    parser.add_argument("--anom2-end", type=int, default=1230)
    parser.add_argument("--anom2-shift", type=float, default=-2.0)
    args = parser.parse_args()

    random.seed(args.seed)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")

    points: list[tuple[datetime, float]] = []
    values: list[float] = []

    for i in range(args.n):
        ts = start_dt + timedelta(hours=i)
        baseline = math.sin(i / 40.0)  # np.sin(np.arange(2000)/40)
        noise = random.gauss(0.0, 0.1)  # np.random.normal(0,0.1,2000)
        v = baseline + noise
        points.append((ts, v))
        values.append(v)

    # Inject anomalies (machine failure simulation)
    for i in range(args.anom1_start, min(args.anom1_end, args.n)):
        ts, v = points[i]
        points[i] = (ts, v + args.anom1_shift)
        values[i] = v + args.anom1_shift

    for i in range(args.anom2_start, min(args.anom2_end, args.n)):
        ts, v = points[i]
        points[i] = (ts, v + args.anom2_shift)
        values[i] = v + args.anom2_shift

    train_rows = int(round(len(points) * args.train_ratio))
    train_rows = max(1, min(train_rows, len(points) - 1))
    train_pts = points[:train_rows]
    test_pts = points[train_rows:]

    mean_v = sum(values) / len(values)
    var_v = sum((x - mean_v) ** 2 for x in values) / max(1, len(values) - 1)
    std_v = math.sqrt(var_v)
    min_v = min(values)
    max_v = max(values)

    meta = Meta(
        name="synthetic_machine_failure",
        mode="synthetic_hourly",
        n_points=len(points),
        start=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        freq="H",
        train_ratio=args.train_ratio,
        train_rows=len(train_pts),
        test_rows=len(test_pts),
        anomaly_1=(args.anom1_start, args.anom1_end, args.anom1_shift),
        anomaly_2=(args.anom2_start, args.anom2_end, args.anom2_shift),
        seed=args.seed,
        value_mean=mean_v,
        value_std=std_v,
        value_min=min_v,
        value_max=max_v,
    )

    outdir = Path(args.outdir)
    full_csv = outdir / f"{meta.name}.csv"
    train_csv = outdir / f"{meta.name}_train.csv"
    test_csv = outdir / f"{meta.name}_test.csv"
    meta_path = outdir / f"{meta.name}_meta.json"

    write_points_csv(full_csv, points)
    write_points_csv(train_csv, train_pts)
    write_points_csv(test_csv, test_pts)
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    print("Created synthetic dataset:")
    print(" ", full_csv)
    print(" ", train_csv)
    print(" ", test_csv)
    print(" ", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

