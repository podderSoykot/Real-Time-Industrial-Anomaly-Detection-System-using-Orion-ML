"""
Generate synthetic time-series datasets for this project.

Default: larger hourly series with multiple regimes (sine, trend, random walk, dual seasonality).

Creates (per dataset name):
  data/<name>.csv
  data/<name>_train.csv
  data/<name>_test.csv
  data/<name>_meta.json

Schema for each CSV:
  timestamp,value
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Meta:
    name: str
    mode: str
    pattern: str
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
    regimes: list[dict[str, int | str]] = field(default_factory=list)


def write_points_csv(path: Path, points: list[tuple[datetime, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        for ts, v in points:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{v:.10g}"])


def _noise(rng: random.Random, sigma: float) -> float:
    return rng.gauss(0.0, sigma)


def series_sine_noise(i: int, rng: random.Random) -> float:
    return math.sin(i / 40.0) + _noise(rng, 0.1)


def series_linear_trend(i: int, rng: random.Random, *, offset: int = 0) -> float:
    j = i - offset
    return 0.15 + j * 0.00035 + _noise(rng, 0.12)


def series_random_walk(i: int, rng: random.Random, state: list[float], *, step: float = 0.18) -> float:
    if i == 0 or not state:
        state.append(rng.gauss(0.0, 0.2))
    else:
        state.append(state[-1] + rng.gauss(0.0, step))
    return state[-1]


def series_dual_seasonal(i: int, rng: random.Random) -> float:
    return math.sin(i / 40.0) + 0.35 * math.sin(i / 6.0) + _noise(rng, 0.08)


def series_stepwise(i: int, rng: random.Random, *, segment_len: int = 120) -> float:
    level = (i // segment_len) * 0.4
    return level + math.sin(i / 25.0) * 0.15 + _noise(rng, 0.06)


def build_series(pattern: str, n: int, seed: int) -> tuple[list[float], list[dict[str, int | str]]]:
    rng = random.Random(seed)
    values: list[float] = []
    regimes: list[dict[str, int | str]] = []
    walk_state: list[float] = []

    if pattern == "sine_noise":
        regimes.append({"type": "sine_noise", "start": 0, "end": n - 1})
        for i in range(n):
            values.append(series_sine_noise(i, rng))

    elif pattern == "linear_trend":
        regimes.append({"type": "linear_trend", "start": 0, "end": n - 1})
        for i in range(n):
            values.append(series_linear_trend(i, rng, offset=0))

    elif pattern == "random_walk":
        regimes.append({"type": "random_walk", "start": 0, "end": n - 1})
        for i in range(n):
            values.append(series_random_walk(i, rng, walk_state))

    elif pattern == "dual_seasonal":
        regimes.append({"type": "dual_seasonal", "start": 0, "end": n - 1})
        for i in range(n):
            values.append(series_dual_seasonal(i, rng))

    elif pattern == "stepwise":
        regimes.append({"type": "stepwise", "start": 0, "end": n - 1})
        for i in range(n):
            values.append(series_stepwise(i, rng))

    elif pattern in ("mixed_regimes", "machine_failure"):
        # Four contiguous regimes: industrial-style changing behavior
        q = n // 4
        bounds = [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)]
        labels = ("sine_noise", "linear_trend", "random_walk", "dual_seasonal")
        for (a, b), label in zip(bounds, labels):
            regimes.append({"type": label, "start": a, "end": b - 1})
        for i in range(n):
            if i < q:
                values.append(series_sine_noise(i, rng))
            elif i < 2 * q:
                values.append(series_linear_trend(i, rng, offset=q))
            elif i < 3 * q:
                values.append(series_random_walk(i, rng, walk_state))
            else:
                values.append(series_dual_seasonal(i, rng))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return values, regimes


def default_anomaly_windows(n: int, train_ratio: float) -> tuple[int, int, int, int]:
    train_rows = int(round(n * train_ratio))
    train_rows = max(1, min(train_rows, n - 1))
    width = max(20, min(80, n // 25))
    anom1_start = max(30, min(n // 6, train_rows - width - 20))
    anom1_end = min(anom1_start + width, train_rows - 5)
    if anom1_end <= anom1_start:
        anom1_start, anom1_end = 100, min(100 + width, train_rows - 1)

    anom2_start = max(train_rows + n // 15, int(n * 0.72))
    anom2_end = min(anom2_start + width, n - 1)
    if anom2_end <= anom2_start:
        anom2_start = min(train_rows + 50, n - width - 2)
        anom2_end = min(anom2_start + width, n - 1)

    return anom1_start, anom1_end, anom2_start, anom2_end


def slug_pattern(pattern: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", pattern.strip()).strip("_").lower()
    return s or "series"


def emit_dataset(
    *,
    name: str,
    pattern: str,
    n: int,
    start_dt: datetime,
    train_ratio: float,
    seed: int,
    anom1: tuple[int, int, float],
    anom2: tuple[int, int, float],
    outdir: Path,
) -> None:
    raw_values, regimes = build_series(pattern, n, seed)
    points: list[tuple[datetime, float]] = []
    values: list[float] = []
    for i in range(n):
        ts = start_dt + timedelta(hours=i)
        v = raw_values[i]
        points.append((ts, v))
        values.append(v)

    for i in range(anom1[0], min(anom1[1], n)):
        ts, v = points[i]
        points[i] = (ts, v + anom1[2])
        values[i] = v + anom1[2]

    for i in range(anom2[0], min(anom2[1], n)):
        ts, v = points[i]
        points[i] = (ts, v + anom2[2])
        values[i] = v + anom2[2]

    train_rows = int(round(len(points) * train_ratio))
    train_rows = max(1, min(train_rows, len(points) - 1))
    train_pts = points[:train_rows]
    test_pts = points[train_rows:]

    mean_v = sum(values) / len(values)
    var_v = sum((x - mean_v) ** 2 for x in values) / max(1, len(values) - 1)
    std_v = math.sqrt(var_v)

    meta = Meta(
        name=name,
        mode="synthetic_hourly",
        pattern=pattern,
        n_points=len(points),
        start=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        freq="H",
        train_ratio=train_ratio,
        train_rows=len(train_pts),
        test_rows=len(test_pts),
        anomaly_1=(anom1[0], anom1[1], anom1[2]),
        anomaly_2=(anom2[0], anom2[1], anom2[2]),
        seed=seed,
        value_mean=mean_v,
        value_std=std_v,
        value_min=min(values),
        value_max=max(values),
        regimes=regimes,
    )

    write_points_csv(outdir / f"{name}.csv", points)
    write_points_csv(outdir / f"{name}_train.csv", train_pts)
    write_points_csv(outdir / f"{name}_test.csv", test_pts)
    (outdir / f"{name}_meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def main() -> int:
    patterns_help = (
        "sine_noise | linear_trend | random_walk | dual_seasonal | stepwise | "
        "mixed_regimes (default main) | machine_failure (alias for mixed_regimes)"
    )
    parser = argparse.ArgumentParser(description="Generate synthetic hourly CSVs + meta JSON.")
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--n", type=int, default=10_000, help="Number of hourly points (default 10000).")
    parser.add_argument("--start", type=str, default="2024-01-01 00:00:00")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern",
        type=str,
        default="mixed_regimes",
        help=f"Primary series shape ({patterns_help}).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="synthetic_machine_failure",
        help="Base filename for the primary dataset (default synthetic_machine_failure).",
    )
    parser.add_argument("--anom1-start", type=int, default=None)
    parser.add_argument("--anom1-end", type=int, default=None)
    parser.add_argument("--anom1-shift", type=float, default=3.0)
    parser.add_argument("--anom2-start", type=int, default=None)
    parser.add_argument("--anom2-end", type=int, default=None)
    parser.add_argument("--anom2-shift", type=float, default=-2.5)
    parser.add_argument(
        "--also",
        type=str,
        default="",
        help=(
            "Comma-separated extra patterns to export as separate datasets "
            "(e.g. sine_noise,linear_trend,random_walk). Each gets synthetic_type_<pattern> files."
        ),
    )
    args = parser.parse_args()

    n = max(200, args.n)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    outdir = Path(args.outdir)

    d1s, d1e, d2s, d2e = default_anomaly_windows(n, args.train_ratio)
    a1s = args.anom1_start if args.anom1_start is not None else d1s
    a1e = args.anom1_end if args.anom1_end is not None else d1e
    a2s = args.anom2_start if args.anom2_start is not None else d2s
    a2e = args.anom2_end if args.anom2_end is not None else d2e
    anom1 = (a1s, a1e, args.anom1_shift)
    anom2 = (a2s, a2e, args.anom2_shift)

    primary_pattern = args.pattern.strip()
    if primary_pattern == "machine_failure":
        primary_pattern = "mixed_regimes"

    emit_dataset(
        name=args.name,
        pattern=primary_pattern,
        n=n,
        start_dt=start_dt,
        train_ratio=args.train_ratio,
        seed=args.seed,
        anom1=anom1,
        anom2=anom2,
        outdir=outdir,
    )

    extras = [p.strip() for p in args.also.split(",") if p.strip()]
    for i, pat in enumerate(extras):
        p = pat if pat != "machine_failure" else "mixed_regimes"
        slug = slug_pattern(p)
        # Slightly different seeds so walks differ
        emit_dataset(
            name=f"synthetic_type_{slug}",
            pattern=p,
            n=n,
            start_dt=start_dt,
            train_ratio=args.train_ratio,
            seed=args.seed + 17 + i,
            anom1=anom1,
            anom2=anom2,
            outdir=outdir,
        )

    print("Created synthetic dataset(s) in", outdir.resolve())
    print(" Primary:", args.name, f"pattern={primary_pattern} n={n}")
    for pat in extras:
        print(" Extra:  synthetic_type_%s" % slug_pattern(pat if pat != "machine_failure" else "mixed_regimes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
