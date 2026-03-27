"""
EDA script for project datasets.

Default target:
  data/synthetic_machine_failure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import math


TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]


def parse_ts(s: str) -> datetime:
    s = s.strip()
    last_err: Optional[Exception] = None
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise ValueError(f"Bad timestamp: {s!r}") from last_err


@dataclass
class EDAResult:
    file: str
    rows: int
    start_timestamp: str
    end_timestamp: str
    median_step_seconds: float
    value_mean: float
    value_std: float
    value_min: float
    value_max: float
    q25: float
    q50: float
    q75: float


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def analyze_csv(csv_path: Path) -> EDAResult:
    timestamps: list[datetime] = []
    values: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError(f"Expected timestamp,value columns in {csv_path}")
        for row in reader:
            timestamps.append(parse_ts(row["timestamp"]))
            values.append(float(row["value"]))

    if not timestamps:
        raise ValueError(f"No rows in {csv_path}")

    diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
    median_step = statistics.median(diffs) if diffs else float("nan")

    vals_sorted = sorted(values)
    return EDAResult(
        file=str(csv_path),
        rows=len(values),
        start_timestamp=timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
        end_timestamp=timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
        median_step_seconds=float(median_step),
        value_mean=float(statistics.mean(values)),
        value_std=float(statistics.stdev(values) if len(values) >= 2 else 0.0),
        value_min=float(min(values)),
        value_max=float(max(values)),
        q25=float(percentile(vals_sorted, 0.25)),
        q50=float(percentile(vals_sorted, 0.50)),
        q75=float(percentile(vals_sorted, 0.75)),
    )


def maybe_check_anomaly_windows(values: list[float], meta_path: Path) -> None:
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    a1 = meta.get("anomaly_1")
    a2 = meta.get("anomaly_2")
    for name, a in [("anomaly_1", a1), ("anomaly_2", a2)]:
        if not isinstance(a, list) or len(a) < 2:
            continue
        s, e = int(a[0]), int(a[1])
        if s < 0 or e > len(values) or s >= e:
            continue
        segment = values[s:e]
        print(f"{name}: idx[{s}:{e}] mean={statistics.mean(segment):.4f} min={min(segment):.4f} max={max(segment):.4f}")


def rolling_mean_std(values: list[float], window: int) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        m = statistics.mean(chunk)
        s = statistics.pstdev(chunk) if len(chunk) >= 2 else 0.0
        means.append(m)
        stds.append(s)
    return means, stds


def load_anomaly_windows(meta_path: Path, n: int) -> list[tuple[str, int, int, float]]:
    windows: list[tuple[str, int, int, float]] = []
    if not meta_path.exists():
        return windows
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("anomaly_1", "anomaly_2"):
        a = meta.get(key)
        if not isinstance(a, list) or len(a) < 3:
            continue
        s, e, shift = int(a[0]), int(a[1]), float(a[2])
        s = max(0, min(s, n))
        e = max(0, min(e, n))
        if s < e:
            windows.append((key, s, e, shift))
    return windows


def make_plots(
    timestamps: list[datetime],
    values: list[float],
    out_dir: Path,
    meta_path: Path,
    train_rows: Optional[int] = None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_anomaly_windows(meta_path, len(values))
    saved: list[Path] = []

    # 1) Time-series with anomaly windows
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, values, linewidth=1.0, color="steelblue", label="value")
    for name, s, e, shift in windows:
        color = "red" if shift > 0 else "orange"
        ax.axvspan(timestamps[s], timestamps[e - 1], color=color, alpha=0.2, label=name)
    if train_rows is not None and 0 < train_rows < len(values):
        ax.axvline(timestamps[train_rows], color="black", linestyle="--", linewidth=1.2, label="train/test split")
    ax.set_title("Time Series with Injected Anomaly Windows")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best")
    p1 = out_dir / "timeseries_with_anomalies.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1)

    # 2) Rolling mean + std bands
    window = max(24, math.ceil(len(values) * 0.02))
    rmean, rstd = rolling_mean_std(values, window)
    upper = [m + 2 * s for m, s in zip(rmean, rstd)]
    lower = [m - 2 * s for m, s in zip(rmean, rstd)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, values, color="#7fa6d8", linewidth=0.9, label="value")
    ax.plot(timestamps, rmean, color="#1f4e79", linewidth=1.5, label=f"rolling mean (w={window})")
    ax.fill_between(timestamps, lower, upper, color="#1f4e79", alpha=0.15, label="mean ± 2*std")
    ax.set_title("Rolling Mean and Volatility Envelope")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    p2 = out_dir / "rolling_mean_std.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2)

    # 3) Distribution view
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(values, bins=40, color="#4c78a8", alpha=0.9, edgecolor="white")
    ax1.set_title("Value Distribution (Histogram)")
    ax1.set_xlabel("value")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.2)
    ax2.boxplot(values, vert=True, patch_artist=True, boxprops=dict(facecolor="#72b7b2"))
    ax2.set_title("Value Distribution (Box Plot)")
    ax2.set_ylabel("value")
    ax2.grid(alpha=0.2)
    p3 = out_dir / "distribution_hist_box.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(p3)

    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/synthetic_machine_failure.csv")
    parser.add_argument("--meta", type=str, default="data/synthetic_machine_failure_meta.json")
    parser.add_argument("--out", type=str, default="data/eda_summary.json")
    parser.add_argument("--plots-dir", type=str, default="data/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    result = analyze_csv(csv_path)
    print("EDA summary")
    for k, v in asdict(result).items():
        print(f"- {k}: {v}")

    # Optional anomaly window summary if meta exists.
    timestamps: list[datetime] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        values = []
        for r in csv.DictReader(f):
            timestamps.append(parse_ts(r["timestamp"]))
            values.append(float(r["value"]))
    maybe_check_anomaly_windows(values, Path(args.meta))

    train_rows: Optional[int] = None
    meta_path = Path(args.meta)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tr = meta.get("train_rows")
        if isinstance(tr, int):
            train_rows = tr

    plot_paths = make_plots(
        timestamps=timestamps,
        values=values,
        out_dir=Path(args.plots_dir),
        meta_path=meta_path,
        train_rows=train_rows,
    )
    print("\nSaved plots:")
    for p in plot_paths:
        print("-", p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
