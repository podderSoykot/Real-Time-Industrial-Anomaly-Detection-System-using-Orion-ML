"""
Continuous sensor streaming (WebSocket + ingest) mounted on the main API in ``api.py``.

Paths (when included with prefix ``/stream``):

- ``GET /stream/`` — service info
- ``GET /stream/health`` — buffer + subscriber stats
- ``POST /stream/ingest`` — push a sample
- ``WebSocket /stream/ws`` — history snapshot, then live JSON points

By default a **demo sine wave** is emitted (set ``STREAM_SYNTHETIC=0`` to disable). ``api`` lifespan starts workers.

Standalone (optional)::

  python -m uvicorn app.stream_api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import statistics
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Demo / default asset context (override with env or per-request fields on ingest).
DEFAULT_MACHINE = os.environ.get("STREAM_MACHINE_NAME", "Hydraulic Press-07")
DEFAULT_PLACE = os.environ.get("STREAM_PLACE", "Plant A — Bay 2")
DEFAULT_LINE = os.environ.get("STREAM_LINE", "Assembly Line 3")
DEFAULT_SENSOR = os.environ.get("STREAM_SENSOR_ID", "S-TEMP-01")
DEFAULT_ZONE = os.environ.get("STREAM_ZONE", "North cell")
DEFAULT_SHIFT = os.environ.get("STREAM_SHIFT", "A")

# Rotating demo assets for synthetic stream (machine, site, line, sensor, area, shift).
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
    {
        "machine_name": "Chiller Unit C2",
        "place": "Utilities — Roof",
        "line": "HVAC",
        "sensor_id": "S-PRESS-11",
        "zone": "Mechanical yard",
        "shift": "C",
    },
]

_SYNTHETIC_NOTES = [
    "Spike vs rolling mean",
    "Check coupling / alignment",
    "Maintenance window due",
    "Vibration harmonics",
    "Thermal drift suspected",
    "Compare with baseline from last week",
]

MAX_HISTORY = 2000
BROADCAST_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
CLIENTS: set[WebSocket] = set()
_HISTORY_LOCK = asyncio.Lock()
_HISTORY: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)

_broadcast_task: asyncio.Task | None = None
_synthetic_task: asyncio.Task | None = None


def synthetic_stream_enabled() -> bool:
    """Default on for local dashboards; set ``STREAM_SYNTHETIC=0`` (or ``false``) to disable."""
    raw = os.environ.get("STREAM_SYNTHETIC", "1")
    v = (raw or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _rolling_zscore_flag(values: list[float], *, window: int = 40, z: float = 3.0) -> tuple[float, bool]:
    if len(values) < 5:
        return 0.0, False
    w = values[-window:]
    mu = statistics.mean(w)
    sd = statistics.pstdev(w)
    if sd < 1e-12:
        return 0.0, False
    last = values[-1]
    zs = (last - mu) / sd
    return float(abs(zs)), bool(abs(zs) > z)


def _resolve_asset_fields(
    *,
    machine_name: str | None,
    place: str | None,
    line: str | None,
    sensor_id: str | None,
    zone: str | None,
    shift: str | None,
    notes: str | None,
) -> dict[str, str]:
    return {
        "machine_name": (machine_name or DEFAULT_MACHINE).strip(),
        "place": (place or DEFAULT_PLACE).strip(),
        "line": (line or DEFAULT_LINE).strip(),
        "sensor_id": (sensor_id or DEFAULT_SENSOR).strip(),
        "zone": (zone or DEFAULT_ZONE).strip(),
        "shift": (shift or DEFAULT_SHIFT).strip(),
        "notes": (notes or "").strip(),
    }


async def _append_point(
    ts: datetime,
    value: float,
    *,
    machine_name: str | None = None,
    place: str | None = None,
    line: str | None = None,
    sensor_id: str | None = None,
    zone: str | None = None,
    shift: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    asset = _resolve_asset_fields(
        machine_name=machine_name,
        place=place,
        line=line,
        sensor_id=sensor_id,
        zone=zone,
        shift=shift,
        notes=notes,
    )
    async with _HISTORY_LOCK:
        vals = [float(p["value"]) for p in _HISTORY]
        vals.append(value)
        score, is_anomaly = _rolling_zscore_flag(vals)
        row = {
            "timestamp": ts.isoformat(),
            "value": value,
            "score": score,
            "is_anomaly": is_anomaly,
            **asset,
        }
        _HISTORY.append(row)
    await BROADCAST_QUEUE.put({"type": "point", **row})
    return row


class IngestBody(BaseModel):
    value: float = Field(..., description="Sensor reading.")
    timestamp: datetime | None = Field(
        default=None,
        description="Optional UTC timestamp; defaults to server time.",
    )
    machine_name: str | None = Field(default=None, description="Equipment or machine id.")
    place: str | None = Field(default=None, description="Site / building / zone.")
    line: str | None = Field(default=None, description="Production line or cell.")
    sensor_id: str | None = Field(default=None, description="Sensor channel id.")
    zone: str | None = Field(default=None, description="Floor / area code.")
    shift: str | None = Field(default=None, description="Shift id (e.g. A, Night).")
    notes: str | None = Field(default=None, description="Free-text context.")


stream_router = APIRouter(prefix="/stream", tags=["stream"])


@stream_router.get("/")
def stream_root() -> dict:
    return {
        "service": "industrial-sensor-stream",
        "ingest": "POST /stream/ingest",
        "websocket": "WS /stream/ws",
        "health": "/stream/health",
        "synthetic": synthetic_stream_enabled(),
    }


@stream_router.get("/health")
def stream_health() -> dict:
    return {
        "status": "ok",
        "clients": len(CLIENTS),
        "buffered_points": len(_HISTORY),
        "synthetic": synthetic_stream_enabled(),
    }


@stream_router.post("/ingest")
async def ingest(body: IngestBody) -> dict:
    ts = body.timestamp or _utc_now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    row = await _append_point(
        ts,
        body.value,
        machine_name=body.machine_name,
        place=body.place,
        line=body.line,
        sensor_id=body.sensor_id,
        zone=body.zone,
        shift=body.shift,
        notes=body.notes,
    )
    return {"accepted": True, "point": row}


async def _broadcast_worker() -> None:
    while True:
        msg = await BROADCAST_QUEUE.get()
        dead: list[WebSocket] = []
        text = json.dumps(msg)
        for ws in CLIENTS:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            CLIENTS.discard(ws)


async def _synthetic_worker() -> None:
    i = 0.0
    while True:
        await asyncio.sleep(0.5)
        v = 10.0 + math.sin(i / 8.0) * 2.0 + random.gauss(0, 0.35)
        if random.random() < 0.02:
            v += random.choice([-6.0, 6.0])
        asset = random.choice(_DEMO_ASSETS)
        note = ""
        if random.random() < 0.12:
            note = random.choice(_SYNTHETIC_NOTES)
        await _append_point(
            _utc_now(),
            v,
            machine_name=asset["machine_name"],
            place=asset["place"],
            line=asset["line"],
            sensor_id=asset["sensor_id"],
            zone=asset["zone"],
            shift=asset["shift"],
            notes=note or None,
        )
        i += 1.0


async def start_stream_workers() -> None:
    global _broadcast_task, _synthetic_task
    _broadcast_task = asyncio.create_task(_broadcast_worker())
    if synthetic_stream_enabled():
        logger.info("Synthetic demo stream enabled — emit sensor-like values (set STREAM_SYNTHETIC=0 to disable)")
        _synthetic_task = asyncio.create_task(_synthetic_worker())


async def stop_stream_workers() -> None:
    global _broadcast_task, _synthetic_task
    for t in (_synthetic_task, _broadcast_task):
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    _broadcast_task = None
    _synthetic_task = None


@stream_router.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    CLIENTS.add(ws)
    try:
        async with _HISTORY_LOCK:
            hist = list(_HISTORY)
        await ws.send_text(json.dumps({"type": "history", "points": hist}))
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=3600.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping", "t": _utc_now().isoformat()}))
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "value" in payload:
                ts_s = payload.get("timestamp")
                ts = (
                    datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
                    if isinstance(ts_s, str)
                    else _utc_now()
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                pv = payload.get("machine_name")
                pl = payload.get("place")
                ln = payload.get("line")
                sid = payload.get("sensor_id")
                zn = payload.get("zone")
                sh = payload.get("shift")
                nt = payload.get("notes")
                await _append_point(
                    ts,
                    float(payload["value"]),
                    machine_name=str(pv) if pv is not None else None,
                    place=str(pl) if pl is not None else None,
                    line=str(ln) if ln is not None else None,
                    sensor_id=str(sid) if sid is not None else None,
                    zone=str(zn) if zn is not None else None,
                    shift=str(sh) if sh is not None else None,
                    notes=str(nt) if nt is not None else None,
                )
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)


@asynccontextmanager
async def _standalone_lifespan(_app: FastAPI):
    await start_stream_workers()
    try:
        yield
    finally:
        await stop_stream_workers()


def create_standalone_app() -> FastAPI:
    """Optional second process; same routes as ``app.include_router(stream_router)`` in ``api.py``."""
    standalone = FastAPI(title="Industrial Sensor Stream API", version="1.0.0", lifespan=_standalone_lifespan)
    standalone.include_router(stream_router)
    return standalone


app = create_standalone_app()


def create_app() -> FastAPI:
    return app
