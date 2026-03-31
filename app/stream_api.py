"""
Continuous sensor streaming (WebSocket + ingest) mounted on the main API in ``api.py``.

Paths (when included with prefix ``/stream``):

- ``GET /stream/`` — service info
- ``GET /stream/health`` — buffer + subscriber stats
- ``GET /stream/history`` — last points snapshot (HTTP-friendly)
- ``POST /stream/ingest`` — push a sample
- ``WebSocket /stream/ws`` — history snapshot, then live JSON points

By default a **demo sine wave** is emitted (set ``STREAM_SYNTHETIC=0`` to disable). ``api`` lifespan starts workers.

Streaming anomaly scoring mode:
- ``STREAM_DETECT_MODE=rolling_zscore`` (default): rolling z-score per sensor
- ``STREAM_DETECT_MODE=orion_ml``: tries ``ORION_PRETRAINED_PATH``; otherwise falls back to baseline
- ``STREAM_DETECT_MODE=baseline_zscore``: fits baseline from ``STREAM_TRAIN_CSV`` (or default training CSV)

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
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    # Allow `src.*` imports when running `app.stream_api` directly.
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import OrionTimeSeriesModel, orion_import_available
from src.preprocess import TimeSeries, load_csv_time_series

logger = logging.getLogger(__name__)

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# Demo / default asset context (override with env or per-request fields on ingest).
DEFAULT_MACHINE = os.environ.get("STREAM_MACHINE_NAME", "Hydraulic Press-07")
DEFAULT_PLACE = os.environ.get("STREAM_PLACE", "Plant A — Bay 2")
DEFAULT_LINE = os.environ.get("STREAM_LINE", "Assembly Line 3")
DEFAULT_SENSOR = os.environ.get("STREAM_SENSOR_ID", "S-TEMP-01")
DEFAULT_ZONE = os.environ.get("STREAM_ZONE", "North cell")
DEFAULT_SHIFT = os.environ.get("STREAM_SHIFT", "A")

# Detection accuracy / tuning
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


STREAM_DETECT_MODE = os.environ.get("STREAM_DETECT_MODE", "rolling_zscore").strip().lower()
STREAM_ROLLING_WINDOW = _env_int("STREAM_ROLLING_WINDOW", 40)
STREAM_Z_THRESHOLD = _env_float("STREAM_Z_THRESHOLD", 3.0)
STREAM_INFER_WINDOW_POINTS = _env_int("STREAM_INFER_WINDOW_POINTS", 250)
STREAM_PREDICT_EVERY_N = max(1, _env_int("STREAM_PREDICT_EVERY_N", 1))

# Queue / reliability hardening
MAX_HISTORY = _env_int("STREAM_MAX_HISTORY", 2000)
STREAM_BROADCAST_QUEUE_MAXSIZE = max(1, _env_int("STREAM_BROADCAST_QUEUE_MAXSIZE", 1000))
STREAM_INGEST_MAX_PER_MINUTE = max(1, _env_int("STREAM_INGEST_MAX_PER_MINUTE", 240))
STREAM_HISTORY_DEFAULT_LIMIT = max(1, _env_int("STREAM_HISTORY_DEFAULT_LIMIT", 200))

# WebSocket robustness
WS_SEND_TIMEOUT_SEC = max(1.0, _env_float("WS_SEND_TIMEOUT_SEC", 5.0))
WS_RECV_TIMEOUT_SEC = max(1.0, _env_float("WS_RECV_TIMEOUT_SEC", 30.0))

# DBSCAN clustering model for streaming anomaly detection.
STREAM_DBSCAN_MODEL_PATH = Path(
    os.environ.get("STREAM_DBSCAN_MODEL_PATH", str(_PROJECT_ROOT / "src" / "General_machine" / "dbscan_model.joblib"))
)
if not STREAM_DBSCAN_MODEL_PATH.is_absolute():
    STREAM_DBSCAN_MODEL_PATH = _PROJECT_ROOT / STREAM_DBSCAN_MODEL_PATH
STREAM_DBSCAN_SCORE_THRESHOLD = _env_float("STREAM_DBSCAN_SCORE_THRESHOLD", 0.0)


def _pretrained_orion_path() -> Path:
    env = os.environ.get("ORION_PRETRAINED_PATH")
    if env:
        p = Path(env.strip())
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / "data" / "models" / "orion_pretrained.pkl"


def _default_train_csv_path() -> Path:
    return _PROJECT_ROOT / "data" / "synthetic_machine_failure_train.csv"

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

# Streaming state (in-memory demo / prototype)
BROADCAST_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=STREAM_BROADCAST_QUEUE_MAXSIZE)
CLIENTS: set[WebSocket] = set()
_HISTORY_LOCK = asyncio.Lock()
_HISTORY: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)

_broadcast_task: asyncio.Task | None = None
_synthetic_task: asyncio.Task | None = None

# Per-asset buffers to avoid mixing sensors in the anomaly score.
_VALUES_BY_KEY: dict[str, deque[float]] = {}
_TIMES_BY_KEY: dict[str, deque[datetime]] = {}
_POINT_COUNTER_BY_KEY: dict[str, int] = {}
_LAST_DETECTION_BY_KEY: dict[str, tuple[float, bool]] = {}

# In-memory ingest rate limiter (simple prototype safeguard).
_INGEST_RATE: dict[str, deque[float]] = {}
_INGEST_RATE_LOCK = asyncio.Lock()

_DETECT_MODEL: Any | None = None
_DETECT_MODEL_SOURCE: str | None = None


def _asset_key(asset: dict[str, str]) -> str:
    # Prefer sensor-level granularity; otherwise fall back to broader asset identity.
    sensor = asset.get("sensor_id") or ""
    if sensor:
        return f"sensor:{sensor}|shift:{asset.get('shift','')}|zone:{asset.get('zone','')}"
    return "|".join(
        [
            f"machine:{asset.get('machine_name','')}",
            f"place:{asset.get('place','')}",
            f"line:{asset.get('line','')}",
            f"zone:{asset.get('zone','')}",
            f"shift:{asset.get('shift','')}",
        ]
    )


def synthetic_stream_enabled() -> bool:
    """Default on for local dashboards; set ``STREAM_SYNTHETIC=0`` (or ``false``) to disable."""
    raw = os.environ.get("STREAM_SYNTHETIC", "1")
    v = (raw or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _rolling_zscore_flag(values: list[float], *, window: int, z: float) -> tuple[float, bool]:
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
        key = _asset_key(asset)
        per_key_maxlen = max(STREAM_ROLLING_WINDOW, STREAM_INFER_WINDOW_POINTS)
        ts_q = _TIMES_BY_KEY.setdefault(key, deque(maxlen=per_key_maxlen))
        vals_q = _VALUES_BY_KEY.setdefault(key, deque(maxlen=per_key_maxlen))

        _POINT_COUNTER_BY_KEY[key] = _POINT_COUNTER_BY_KEY.get(key, 0) + 1
        points_seen_for_key = _POINT_COUNTER_BY_KEY[key]

        ts_q.append(ts)
        vals_q.append(float(value))

        vals_list = list(vals_q)
        score, is_anomaly = _rolling_zscore_flag(
            vals_list,
            window=STREAM_ROLLING_WINDOW,
            z=STREAM_Z_THRESHOLD,
        )

        # Optional accuracy improvement: use trained model for per-asset windows.
        if STREAM_DETECT_MODE != "rolling_zscore" and _DETECT_MODEL is not None:
            do_predict = points_seen_for_key == 1 or (points_seen_for_key % STREAM_PREDICT_EVERY_N == 0)
            if not do_predict:
                last = _LAST_DETECTION_BY_KEY.get(key)
                if last is not None:
                    score, is_anomaly = last
            else:
                n = min(STREAM_INFER_WINDOW_POINTS, len(vals_q))
                if n >= 5:
                    sub_ts = list(ts_q)[-n:]
                    sub_vals = vals_list[-n:]
                    try:
                        res = _DETECT_MODEL.predict(TimeSeries(timestamps=sub_ts, values=sub_vals))
                        if res.scores:
                            score = float(res.scores[-1])
                        # DBSCAN uses distance-based scores; we override the anomaly decision using score threshold.
                        if STREAM_DETECT_MODE == "dbscan_cluster":
                            is_anomaly = score > STREAM_DBSCAN_SCORE_THRESHOLD
                        else:
                            if res.is_anomaly:
                                is_anomaly = bool(res.is_anomaly[-1])
                        _LAST_DETECTION_BY_KEY[key] = (score, is_anomaly)
                    except Exception:
                        logger.exception("Stream detection failed for %s; falling back to rolling_zscore.", key)

        row = {
            "timestamp": ts.isoformat(),
            "value": value,
            "score": score,
            "is_anomaly": is_anomaly,
            **asset,
        }
        _HISTORY.append(row)

    # Bounded queue: if overloaded, drop oldest broadcast(s) instead of blocking ingest.
    msg = {"type": "point", **row}
    try:
        BROADCAST_QUEUE.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            _ = BROADCAST_QUEUE.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            BROADCAST_QUEUE.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("Broadcast queue full; dropping point for %s", asset.get("sensor_id"))
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
        "history": "/stream/history",
        "synthetic": synthetic_stream_enabled(),
    }


@stream_router.get("/health")
def stream_health() -> dict:
    return {
        "status": "ok",
        "clients": len(CLIENTS),
        "buffered_points": len(_HISTORY),
        "synthetic": synthetic_stream_enabled(),
        "detect_mode": STREAM_DETECT_MODE,
        "model_loaded": _DETECT_MODEL is not None,
        "model_source": _DETECT_MODEL_SOURCE,
        "dbscan_score_threshold": STREAM_DBSCAN_SCORE_THRESHOLD if STREAM_DETECT_MODE == "dbscan_cluster" else None,
    }


@stream_router.get("/history")
async def stream_history(
    limit: int = STREAM_HISTORY_DEFAULT_LIMIT,
    sensor_id: str | None = None,
    machine_name: str | None = None,
) -> dict:
    limit = max(1, min(int(limit), MAX_HISTORY))
    async with _HISTORY_LOCK:
        pts = list(_HISTORY)[-limit:]
    if sensor_id is not None:
        pts = [p for p in pts if p.get("sensor_id") == sensor_id]
    if machine_name is not None:
        pts = [p for p in pts if p.get("machine_name") == machine_name]
    return {
        "points": pts,
        "limit": limit,
        "filtered": {"sensor_id": sensor_id, "machine_name": machine_name},
    }


async def _check_ingest_rate(request: Request) -> None:
    if STREAM_INGEST_MAX_PER_MINUTE <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window_sec = 60.0
    async with _INGEST_RATE_LOCK:
        dq = _INGEST_RATE.get(client)
        if dq is None:
            dq = deque()
            _INGEST_RATE[client] = dq
        while dq and dq[0] < (now - window_sec):
            dq.popleft()
        if len(dq) >= STREAM_INGEST_MAX_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many ingest requests (rate limit).")
        dq.append(now)


@stream_router.post("/ingest")
async def ingest(request: Request, body: IngestBody) -> dict:
    await _check_ingest_rate(request)
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
                # Prevent a single slow client from stalling broadcast for everyone.
                await asyncio.wait_for(ws.send_text(text), timeout=WS_SEND_TIMEOUT_SEC)
            except Exception:
                dead.append(ws)
        for ws in dead:
            CLIENTS.discard(ws)


def _init_detection_model_sync() -> tuple[Any | None, str | None]:
    """
    Initialize stream detection model once at server start.

    - `rolling_zscore`: no model (pure rolling statistics)
    - `orion_ml`: try pretrained pickle; else fall back to baseline_zscore
    - `baseline_zscore`: fit baseline from CSV (fast + dependency-light)
    - `dbscan_cluster`: load DBSCAN model from `STREAM_DBSCAN_MODEL_PATH` (scikit-learn)
    """
    detect_mode = STREAM_DETECT_MODE
    if detect_mode not in {"rolling_zscore", "orion_ml", "baseline_zscore", "dbscan_cluster"}:
        logger.warning("Unknown STREAM_DETECT_MODE=%r; falling back to rolling_zscore", detect_mode)
        return None, "rolling_zscore"

    if detect_mode == "rolling_zscore":
        return None, "rolling_zscore"

    if detect_mode == "dbscan_cluster":
        if joblib is None:
            logger.warning("STREAM_DETECT_MODE=dbscan_cluster but joblib could not be imported.")
            return None, "rolling_zscore"
        model_path = STREAM_DBSCAN_MODEL_PATH
        if not model_path.exists():
            logger.warning("DBSCAN model file not found at %s (set STREAM_DBSCAN_MODEL_PATH).", model_path)
            return None, "dbscan_missing"
        try:
            state = joblib.load(model_path)
            if isinstance(state, dict) and state.get("type") == "dbscan_core_distance":
                from src.General_machine.dbscan_training import DbscanAnomalyModel

                m = DbscanAnomalyModel.from_state(state)
                return m, f"dbscan_model:{model_path.name}"

            # Backward compatibility if older files stored the full instance.
            if hasattr(state, "predict"):
                return state, f"dbscan_model:{model_path.name}"

            logger.warning("DBSCAN model loaded but has no usable `predict()`; falling back to rolling_zscore.")
            return None, "rolling_zscore"
        except Exception:
            logger.exception("Failed to load DBSCAN model from %s.", model_path)
            return None, "dbscan_load_failed"

    baseline_q = _env_float("STREAM_BASELINE_THRESHOLD_QUANTILE", 0.995)
    train_csv_env = os.environ.get("STREAM_TRAIN_CSV")
    train_csv = Path(train_csv_env.strip()) if train_csv_env else _default_train_csv_path()
    if not train_csv.is_absolute():
        train_csv = _PROJECT_ROOT / train_csv

    if detect_mode == "orion_ml":
        pkl_path = _pretrained_orion_path()
        if pkl_path.exists() and orion_import_available():
            try:
                cfg: dict[str, Any] = {}
                pipeline = os.environ.get("STREAM_ORION_PIPELINE")
                if pipeline:
                    cfg["pipeline"] = pipeline
                m = OrionTimeSeriesModel.from_orion_pickle(pkl_path, config=cfg or None)
                return m, "pretrained_orion_pickle"
            except Exception:
                logger.exception("Failed to load Orion pretrained pickle (%s). Falling back to baseline.", pkl_path)

    if not train_csv.exists():
        logger.warning("Training CSV for baseline not found (%s). Staying on rolling_zscore.", train_csv)
        return None, "rolling_zscore"

    try:
        train_series = load_csv_time_series(train_csv)
        m = OrionTimeSeriesModel(config={"use_orion": False, "threshold_quantile": baseline_q})
        m.fit(train_series)
        return m, "fitted_baseline_train_csv"
    except Exception:
        logger.exception("Failed to fit baseline model from %s. Staying on rolling_zscore.", train_csv)
        return None, "rolling_zscore"


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

    global _DETECT_MODEL, _DETECT_MODEL_SOURCE
    if _DETECT_MODEL is None and _DETECT_MODEL_SOURCE is None:
        # Model init may load pickle / parse CSV; keep it off the event loop.
        _DETECT_MODEL, _DETECT_MODEL_SOURCE = await asyncio.to_thread(_init_detection_model_sync)

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

        # Optional query params to reduce payload size.
        # Example: ws://host:8000/stream/ws?limit=200&sensor_id=S-TEMP-01
        limit_s = ws.query_params.get("limit")
        sensor_id_q = ws.query_params.get("sensor_id")
        machine_name_q = ws.query_params.get("machine_name")
        try:
            limit = max(1, min(int(limit_s), MAX_HISTORY)) if limit_s else STREAM_HISTORY_DEFAULT_LIMIT
        except Exception:
            limit = STREAM_HISTORY_DEFAULT_LIMIT

        if sensor_id_q is not None:
            hist = [p for p in hist[-limit:] if p.get("sensor_id") == sensor_id_q]
        elif machine_name_q is not None:
            hist = [p for p in hist[-limit:] if p.get("machine_name") == machine_name_q]
        else:
            hist = hist[-limit:]

        await ws.send_text(json.dumps({"type": "history", "points": hist, "limit": limit}))

        last_ping_at = time.time()
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=WS_RECV_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                last_ping_at = time.time()
                await ws.send_text(json.dumps({"type": "ping", "t": _utc_now().isoformat()}))
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            # Heartbeat response from client (optional).
            if payload.get("type") == "pong":
                continue

            if "value" not in payload:
                continue

            # Parse timestamp: accept ISO strings, "Z" suffix, or numeric unix seconds.
            ts_s = payload.get("timestamp")
            try:
                if isinstance(ts_s, (int, float)):
                    v = float(ts_s)
                    # Heuristic: if it's likely milliseconds, convert to seconds.
                    if v > 1e12:
                        v = v / 1000.0
                    ts = datetime.fromtimestamp(v, tz=timezone.utc)
                elif isinstance(ts_s, str) and ts_s.strip():
                    ts = datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = _utc_now()
            except Exception:
                ts = _utc_now()

            try:
                v = float(payload["value"])
            except Exception:
                continue

            pv = payload.get("machine_name")
            pl = payload.get("place")
            ln = payload.get("line")
            sid = payload.get("sensor_id")
            zn = payload.get("zone")
            sh = payload.get("shift")
            nt = payload.get("notes")

            row = await _append_point(
                ts,
                v,
                machine_name=str(pv) if pv is not None else None,
                place=str(pl) if pl is not None else None,
                line=str(ln) if ln is not None else None,
                sensor_id=str(sid) if sid is not None else None,
                zone=str(zn) if zn is not None else None,
                shift=str(sh) if sh is not None else None,
                notes=str(nt) if nt is not None else None,
            )

            # Acknowledge the ingest so clients can surface failures.
            try:
                await asyncio.wait_for(
                    ws.send_text(json.dumps({"type": "ack", "point": row})),
                    timeout=WS_SEND_TIMEOUT_SEC,
                )
            except Exception:
                pass
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
