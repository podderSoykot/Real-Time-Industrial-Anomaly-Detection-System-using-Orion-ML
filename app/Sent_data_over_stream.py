"""
Synthetic producer for the stream API mounted on ``app.api`` (same process as ``POST /stream/ingest``
and ``WebSocket /stream/ws``).

Run the API first, e.g.::

  python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

Then either HTTP (no extra deps)::

  python -m app.Sent_data_over_stream --base-url http://127.0.0.1:8000 --mode http

Or WebSocket (``pip install websockets`` if missing)::

  python -m app.Sent_data_over_stream --base-url http://127.0.0.1:8000 --mode ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

_DEMO_ASSETS: list[dict[str, str]] = [
    {
        "machine_name": "Hydraulic Press-07",
        "place": "Plant A — Bay 2",
        "line": "Assembly Line 3",
        "sensor_id": "S-TEMP-01",
        "zone": "North cell",
        "shift": "A",
    },
    {
        "machine_name": "CNC Mill-12",
        "place": "Plant B — Cell 5",
        "line": "Machining Line 1",
        "sensor_id": "S-VIB-04",
        "zone": "West mezzanine",
        "shift": "B",
    },
    {
        "machine_name": "Conveyor M-03",
        "place": "Plant A — Bay 1",
        "line": "Packaging Line 2",
        "sensor_id": "S-AMP-02",
        "zone": "Loading dock",
        "shift": "A",
    },
    {
        "machine_name": "Welding Robot R5",
        "place": "Plant C — Weld shop",
        "line": "Line 7",
        "sensor_id": "S-CURR-09",
        "zone": "Spark bay",
        "shift": "Night",
    },
]

_NOTES = [
    "Spike vs rolling mean",
    "Check coupling",
    "Maintenance window due",
    "Vibration harmonics",
]


def _asset_fields(
    *,
    machine_name: str | None,
    place: str | None,
    line: str | None,
    sensor_id: str | None,
    zone: str | None,
    shift: str | None,
    notes: str | None,
    random_notes: bool,
) -> dict[str, str]:
    if machine_name or place or line or sensor_id or zone or shift:
        base = _DEMO_ASSETS[0]
        out = {
            "machine_name": (machine_name or base["machine_name"]).strip(),
            "place": (place or base["place"]).strip(),
            "line": (line or base["line"]).strip(),
            "sensor_id": (sensor_id or base["sensor_id"]).strip(),
            "zone": (zone or base["zone"]).strip(),
            "shift": (shift or base["shift"]).strip(),
        }
        if notes is not None:
            out["notes"] = notes.strip()
        elif random_notes and random.random() < 0.1:
            out["notes"] = random.choice(_NOTES)
        else:
            out["notes"] = ""
        return out

    asset = dict(random.choice(_DEMO_ASSETS))
    if notes is not None:
        asset["notes"] = notes.strip()
    elif random_notes and random.random() < 0.12:
        asset["notes"] = random.choice(_NOTES)
    else:
        asset["notes"] = ""
    return asset


def _http_base_to_ws_url(base_url: str) -> str:
    """``http(s)://host:port`` → ``ws(s)://host:port/stream/ws``."""
    u = base_url.strip().rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://") :] + "/stream/ws"
    if u.startswith("http://"):
        return "ws://" + u[len("http://") :] + "/stream/ws"
    raise ValueError(f"Expected base_url to start with http:// or https://, got {base_url!r}")


def iter_synthetic_values(
    *,
    step: float = 0.5,
    amplitude: float = 2.0,
    baseline: float = 10.0,
    spike_prob: float = 0.02,
    spike_mag: float = 6.0,
) -> Iterator[float]:
    """Yield synthetic sensor readings: sine trend + Gaussian noise + rare spikes."""
    i = 0.0
    while True:
        v = baseline + math.sin(i / 8.0) * amplitude + random.gauss(0, 0.35)
        if random.random() < spike_prob:
            v += random.choice([-spike_mag, spike_mag])
        yield v
        i += step


def send_synthetic_to_stream_http(
    base_url: str,
    *,
    interval_sec: float = 0.5,
    max_points: int | None = None,
    machine_name: str | None = None,
    place: str | None = None,
    line: str | None = None,
    sensor_id: str | None = None,
    zone: str | None = None,
    shift: str | None = None,
    notes: str | None = None,
    random_notes: bool = True,
) -> None:
    """
    Push samples with ``POST {base_url}/stream/ingest``.
    Subscribers on ``WebSocket /stream/ws`` receive the same points from the server.
    """
    ingest_url = urljoin(base_url.rstrip("/") + "/", "stream/ingest")
    gen = iter_synthetic_values()
    n = 0
    while max_points is None or n < max_points:
        value = next(gen)
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "value": value,
            "timestamp": ts,
            **_asset_fields(
                machine_name=machine_name,
                place=place,
                line=line,
                sensor_id=sensor_id,
                zone=zone,
                shift=shift,
                notes=notes,
                random_notes=random_notes,
            ),
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            ingest_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=15.0) as resp:
                resp.read()
        except URLError as e:
            raise RuntimeError(f"POST failed ({ingest_url}): {e}") from e
        n += 1
        time.sleep(interval_sec)


async def send_synthetic_to_stream_websocket(
    base_url: str,
    *,
    interval_sec: float = 0.5,
    max_points: int | None = None,
    machine_name: str | None = None,
    place: str | None = None,
    line: str | None = None,
    sensor_id: str | None = None,
    zone: str | None = None,
    shift: str | None = None,
    notes: str | None = None,
    random_notes: bool = True,
) -> None:
    """
    Connect to ``WebSocket`` and send JSON objects with ``value``, ``timestamp``, and asset fields.
    Requires the ``websockets`` package (``pip install websockets``).
    """
    try:
        import websockets
    except ImportError as e:
        raise ImportError(
            "WebSocket mode needs `websockets`. Install with: pip install websockets"
        ) from e

    ws_url = _http_base_to_ws_url(base_url)

    gen = iter_synthetic_values()
    n = 0

    async with websockets.connect(ws_url) as ws:

        async def _drain_incoming() -> None:
            while True:
                await ws.recv()

        drain = asyncio.create_task(_drain_incoming())
        try:
            while max_points is None or n < max_points:
                value = next(gen)
                payload = {
                    "value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **_asset_fields(
                        machine_name=machine_name,
                        place=place,
                        line=line,
                        sensor_id=sensor_id,
                        zone=zone,
                        shift=shift,
                        notes=notes,
                        random_notes=random_notes,
                    ),
                }
                await ws.send(json.dumps(payload))
                n += 1
                await asyncio.sleep(interval_sec)
        finally:
            drain.cancel()
            try:
                await drain
            except asyncio.CancelledError:
                pass


def run_synthetic_producer(
    base_url: str,
    *,
    mode: str = "http",
    interval_sec: float = 0.5,
    max_points: int | None = None,
    machine_name: str | None = None,
    place: str | None = None,
    line: str | None = None,
    sensor_id: str | None = None,
    zone: str | None = None,
    shift: str | None = None,
    notes: str | None = None,
    random_notes: bool = True,
) -> None:
    """Dispatch to HTTP or WebSocket producer."""
    if mode == "http":
        send_synthetic_to_stream_http(
            base_url,
            interval_sec=interval_sec,
            max_points=max_points,
            machine_name=machine_name,
            place=place,
            line=line,
            sensor_id=sensor_id,
            zone=zone,
            shift=shift,
            notes=notes,
            random_notes=random_notes,
        )
    elif mode in ("ws", "websocket"):
        asyncio.run(
            send_synthetic_to_stream_websocket(
                base_url,
                interval_sec=interval_sec,
                max_points=max_points,
                machine_name=machine_name,
                place=place,
                line=line,
                sensor_id=sensor_id,
                zone=zone,
                shift=shift,
                notes=notes,
                random_notes=random_notes,
            )
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'http' or 'ws')")


def main() -> None:
    p = argparse.ArgumentParser(description="Send synthetic data to the stream API.")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API root (same host/port as uvicorn).",
    )
    p.add_argument("--mode", choices=("http", "ws"), default="http", help="HTTP ingest or WebSocket send.")
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between samples.")
    p.add_argument("--max-points", type=int, default=None, help="Stop after N samples (default: run forever).")
    p.add_argument("--machine", default=None, help="Fixed machine name (default: rotate demo assets).")
    p.add_argument("--place", default=None, help="Fixed site / place.")
    p.add_argument("--line", default=None, help="Fixed production line.")
    p.add_argument("--sensor", default=None, help="Fixed sensor id.")
    p.add_argument("--zone", default=None, help="Fixed floor / area.")
    p.add_argument("--shift", default=None, help="Fixed shift id.")
    p.add_argument("--notes", default=None, help="Fixed notes string (omit for random occasional notes).")
    p.add_argument(
        "--no-random-notes",
        action="store_true",
        help="Do not attach random notes when rotating demo assets.",
    )
    args = p.parse_args()
    run_synthetic_producer(
        args.base_url,
        mode=args.mode,
        interval_sec=args.interval,
        max_points=args.max_points,
        machine_name=args.machine,
        place=args.place,
        line=args.line,
        sensor_id=args.sensor,
        zone=args.zone,
        shift=args.shift,
        notes=args.notes,
        random_notes=not args.no_random_notes,
    )


if __name__ == "__main__":
    main()
