from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class SensorMeta:
    sensor_id: str
    curve_type: str
    amplitude: float
    noise_std: float
    accel_mag: float
    shock_mag: float
    shock_sigma_sec: float
    anomaly_label_start_sec: int
    anomaly_label_end_sec: int
    state_low_start_sec: int
    state_moderate_start_sec: int
    state_high_start_sec: int


@dataclass
class MachineMeta:
    machine_id: str
    machine_name: str
    failure_time_sec: int
    failure_timestamp: str
    sensors: list[SensorMeta]


@dataclass
class PMeta:
    name: str
    mode: str
    start: str
    interval_sec: int
    days: int
    total_points: int
    horizon_sec: int
    seed: int
    machines: list[MachineMeta]


def _degradation_curve(u: float, curve_type: str, *, rng: random.Random, amplitude: float) -> float:
    # u in [0,1]
    if curve_type == "linear":
        return amplitude * u
    if curve_type == "weibull":
        b = rng.uniform(1.2, 3.2)
        return amplitude * (u**b)
    if curve_type == "exp":
        alpha = rng.uniform(2.0, 6.0)
        denom = math.exp(alpha) - 1.0
        if denom <= 1e-12:
            return amplitude * u
        return amplitude * (math.exp(alpha * u) - 1.0) / denom
    # fallback
    return amplitude * u


def _write_sensor_csv(
    out_csv: Path,
    *,
    start_dt: datetime,
    interval_sec: int,
    num_points: int,
    failure_time_sec: int,
    horizon_sec: int,
    state_low_start_sec: int,
    state_moderate_start_sec: int,
    state_high_start_sec: int,
    machine_name: str,
    sensor_id: str,
    base_level: float,
    curve_type: str,
    amplitude: float,
    noise_std: float,
    accel_mag: float,
    shock_mag: float,
    shock_sigma_sec: float,
    rng: random.Random,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # Supervised columns:
        # - label: 0/1 anomaly window (any of low/moderate/high)
        # - state: 0 none, 1 low, 2 moderate, 3 high
        w.writerow(["timestamp", "value", "label", "state", "machine_name", "sensor_id"])

        for row in _iter_sensor_rows(
            start_dt=start_dt,
            interval_sec=interval_sec,
            num_points=num_points,
            failure_time_sec=failure_time_sec,
            horizon_sec=horizon_sec,
            state_low_start_sec=state_low_start_sec,
            state_moderate_start_sec=state_moderate_start_sec,
            state_high_start_sec=state_high_start_sec,
            machine_name=machine_name,
            sensor_id=sensor_id,
            base_level=base_level,
            curve_type=curve_type,
            amplitude=amplitude,
            noise_std=noise_std,
            accel_mag=accel_mag,
            shock_mag=shock_mag,
            shock_sigma_sec=shock_sigma_sec,
            rng=rng,
        ):
            w.writerow(row)


def _iter_sensor_rows(
    *,
    start_dt: datetime,
    interval_sec: int,
    num_points: int,
    failure_time_sec: int,
    horizon_sec: int,
    state_low_start_sec: int,
    state_moderate_start_sec: int,
    state_high_start_sec: int,
    machine_name: str,
    sensor_id: str,
    base_level: float,
    curve_type: str,
    amplitude: float,
    noise_std: float,
    accel_mag: float,
    shock_mag: float,
    shock_sigma_sec: float,
    rng: random.Random,
) -> list[list[Any]]:
    """
    Generate rows for a sensor time series.
    Returned as an eager list to keep this generator simple; total rows should be manageable.
    """
    duration_sec = (num_points - 1) * interval_sec if num_points > 1 else interval_sec
    out: list[list[Any]] = []
    for i in range(num_points):
        t_sec = i * interval_sec
        ts = start_dt + timedelta(seconds=t_sec)

        u = (t_sec / duration_sec) if duration_sec > 0 else 0.0
        u = _clamp(u, 0.0, 1.0)

        degr = _degradation_curve(u, curve_type, rng=rng, amplitude=amplitude)

        state = 0
        if state_low_start_sec <= t_sec < failure_time_sec:
            if t_sec >= state_high_start_sec:
                state = 3
            elif t_sec >= state_moderate_start_sec:
                state = 2
            else:
                state = 1

        label = 1 if state > 0 else 0

        if label == 1:
            x = (t_sec - state_low_start_sec) / max(1.0, float(horizon_sec))
            accel = accel_mag * (x**2)
            sigma = noise_std * (1.0 + 0.8 * x)
        else:
            accel = 0.0
            sigma = noise_std

        if t_sec >= failure_time_sec:
            dx = float(t_sec - failure_time_sec)
            shock = shock_mag * math.exp(-(dx * dx) / (2.0 * (shock_sigma_sec * shock_sigma_sec)))
        else:
            shock = 0.0

        noise = rng.gauss(0.0, sigma)
        value = base_level + degr + accel + shock + noise

        out.append([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{value:.10g}", int(label), int(state), machine_name, sensor_id])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate supervised predictive maintenance dataset.")
    p.add_argument("--outdir", type=str, default="data/predictive_maintenance_pm")
    p.add_argument("--name", type=str, default="pm_sensor_supervised")
    p.add_argument("--machines", type=int, default=1, help="How many machines to generate (default 1).")
    p.add_argument("--sensors-per-machine", type=int, default=10)
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--interval-sec", type=int, default=1)
    p.add_argument("--start", type=str, default="2024-01-01 00:00:00")
    p.add_argument("--horizon-hours", type=float, default=12.0, help="Label window length before failure.")
    p.add_argument("--state-high-fraction", type=float, default=0.33, help="Within horizon: fraction assigned to HIGH (near failure).")
    p.add_argument("--state-moderate-fraction", type=float, default=0.33, help="Within horizon: fraction assigned to MODERATE (middle). LOW gets the remainder.")
    p.add_argument("--single-csv", action="store_true", help="Generate a single CSV instead of one CSV per sensor.")
    p.add_argument(
        "--total-records",
        type=int,
        default=100000,
        help="When --single-csv is set: total number of rows across all machines/sensors.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    interval_sec = args.interval_sec
    horizon_sec = int(args.horizon_hours * 3600.0)
    if args.state_high_fraction < 0 or args.state_moderate_fraction < 0 or (args.state_high_fraction + args.state_moderate_fraction) >= 1.0:
        raise ValueError("state-high-fraction and state-moderate-fraction must be >=0 and sum to < 1.0")

    duration_sec_full = args.days * 24 * 3600
    duration_sec_full = max(interval_sec, duration_sec_full)

    total_series = int(args.machines * args.sensors_per_machine)
    if total_series <= 0:
        raise ValueError("machines * sensors-per-machine must be > 0")

    if args.single_csv:
        total_records = int(args.total_records)
        if total_records <= 0:
            raise ValueError("--total-records must be > 0")
        base_points = total_records // total_series
        remainder = total_records % total_series
        if base_points < 2 and remainder == 0:
            raise ValueError("Not enough points per series to build a time axis.")
        max_points = base_points + (1 if remainder > 0 else 0)
        duration_sec = (max_points - 1) * interval_sec
        num_points_per_series_default = base_points
    else:
        duration_sec = duration_sec_full
        num_points_per_series_default = duration_sec // interval_sec
        remainder = 0
        base_points = num_points_per_series_default
        max_points = num_points_per_series_default

    # Effective horizon can't exceed the time span of the generated samples.
    horizon_sec = int(args.horizon_hours * 3600.0)
    horizon_sec_eff = int(min(horizon_sec, max(1, duration_sec)))
    if horizon_sec_eff < 1:
        horizon_sec_eff = 1

    machines_meta: list[MachineMeta] = []

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare single-csv writer if requested.
    single_csv_path = None
    single_csv_f = None
    single_csv_writer = None
    if args.single_csv:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        single_csv_path = outdir / f"{args.name}__single.csv"
        single_csv_f = single_csv_path.open("w", encoding="utf-8", newline="")
        single_csv_writer = csv.writer(single_csv_f)
        single_csv_writer.writerow(["timestamp", "value", "label", "state", "machine_name", "sensor_id"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for m in range(args.machines):
        machine_id = f"M{m+1:02d}"
        machine_name = f"Machine-{machine_id}"

        # Choose failure time (end-exclusive label window). Keep it away from edges.
        failure_time_sec = int(rng.uniform(0.6 * duration_sec, 0.9 * duration_sec))
        # Ensure failure isn't too close to start/end relative to horizon.
        min_failure = horizon_sec_eff + interval_sec
        min_failure = min(min_failure, max(1, duration_sec - interval_sec))
        failure_time_sec = max(min_failure, min(failure_time_sec, duration_sec - interval_sec))

        failure_ts = start_dt + timedelta(seconds=failure_time_sec)

        sensors_meta: list[SensorMeta] = []

        label_start = failure_time_sec - horizon_sec_eff
        label_end = failure_time_sec  # end-exclusive
        state_high_start_sec = int(round(failure_time_sec - (horizon_sec_eff * args.state_high_fraction)))
        state_moderate_start_sec = int(
            round(failure_time_sec - (horizon_sec_eff * (args.state_high_fraction + args.state_moderate_fraction)))
        )
        state_low_start_sec = label_start

        for s in range(args.sensors_per_machine):
            sensor_id = f"{machine_id}-S{s+1:02d}"
            curve_type = rng.choice(["linear", "weibull", "exp"])

            # Sensor-specific parameters (so signals differ per sensor)
            amplitude = rng.uniform(0.6, 2.0)
            noise_std = rng.uniform(0.03, 0.15)
            accel_mag = rng.uniform(0.3, 2.2)
            shock_mag = rng.uniform(2.0, 6.0)
            shock_sigma_sec = rng.uniform(20.0, 120.0)

            # Base level could differ per sensor too
            base_level = rng.uniform(5.0, 12.0)

            # Deterministic per-sensor RNG so series repeat for given seed.
            sensor_seed = rng.randint(0, 10**9)
            sensor_rng = random.Random(sensor_seed)

            sensors_meta.append(
                SensorMeta(
                    sensor_id=sensor_id,
                    curve_type=curve_type,
                    amplitude=float(amplitude),
                    noise_std=float(noise_std),
                    accel_mag=float(accel_mag),
                    shock_mag=float(shock_mag),
                    shock_sigma_sec=float(shock_sigma_sec),
                    anomaly_label_start_sec=int(label_start),
                    anomaly_label_end_sec=int(label_end),
                    state_low_start_sec=int(state_low_start_sec),
                    state_moderate_start_sec=int(state_moderate_start_sec),
                    state_high_start_sec=int(state_high_start_sec),
                )
            )

            # Determine points count for this series.
            series_idx = m * args.sensors_per_machine + s
            num_points = base_points + (1 if series_idx < remainder else 0)
            num_points = max(2, int(num_points))

            if args.single_csv:
                rows = _iter_sensor_rows(
                    start_dt=start_dt,
                    interval_sec=interval_sec,
                    num_points=num_points,
                    failure_time_sec=failure_time_sec,
                    horizon_sec=horizon_sec_eff,
                    state_low_start_sec=int(state_low_start_sec),
                    state_moderate_start_sec=int(state_moderate_start_sec),
                    state_high_start_sec=int(state_high_start_sec),
                    machine_name=machine_name,
                    sensor_id=sensor_id,
                    base_level=base_level,
                    curve_type=curve_type,
                    amplitude=amplitude,
                    noise_std=noise_std,
                    accel_mag=accel_mag,
                    shock_mag=shock_mag,
                    shock_sigma_sec=shock_sigma_sec,
                    rng=sensor_rng,
                )
                assert single_csv_writer is not None
                for row in rows:
                    single_csv_writer.writerow(row)
            else:
                sensor_csv = outdir / f"{args.name}__{machine_id}__{sensor_id}.csv"
                _write_sensor_csv(
                    sensor_csv,
                    start_dt=start_dt,
                    interval_sec=interval_sec,
                    num_points=num_points_per_series_default,
                    failure_time_sec=failure_time_sec,
                    horizon_sec=horizon_sec_eff,
                    state_low_start_sec=int(state_low_start_sec),
                    state_moderate_start_sec=int(state_moderate_start_sec),
                    state_high_start_sec=int(state_high_start_sec),
                    machine_name=machine_name,
                    sensor_id=sensor_id,
                    base_level=base_level,
                    curve_type=curve_type,
                    amplitude=amplitude,
                    noise_std=noise_std,
                    accel_mag=accel_mag,
                    shock_mag=shock_mag,
                    shock_sigma_sec=shock_sigma_sec,
                    rng=sensor_rng,
                )

        machines_meta.append(
            MachineMeta(
                machine_id=machine_id,
                machine_name=machine_name,
                failure_time_sec=failure_time_sec,
                failure_timestamp=failure_ts.strftime("%Y-%m-%d %H:%M:%S"),
                sensors=sensors_meta,
            )
        )

    if single_csv_f is not None:
        single_csv_f.close()

    meta = PMeta(
        name=args.name,
        mode="predictive_maintenance_supervised",
        start=args.start,
        interval_sec=interval_sec,
        days=args.days if not args.single_csv else int(round(duration_sec / (24 * 3600))),
        total_points=max_points,
        horizon_sec=horizon_sec,
        seed=args.seed,
        machines=machines_meta,
    )

    meta_path = outdir / f"{args.name}__meta.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    if args.single_csv:
        assert single_csv_path is not None
        print(f"Generated single predictive maintenance CSV: {single_csv_path.resolve()}")
        print(f"- rows (total): {args.total_records}")
        print(f"- machines: {args.machines}")
        print(f"- sensors/machine: {args.sensors_per_machine}")
    else:
        print(f"Generated predictive maintenance dataset in: {outdir.resolve()}")
        print(f"- machines: {args.machines}")
        print(f"- sensors/machine: {args.sensors_per_machine}")
        print(f"- days: {args.days}, interval_sec: {args.interval_sec} => points: {num_points_per_series_default}")
        print(f"- horizon_hours: {args.horizon_hours} => label window: {horizon_sec_eff} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

