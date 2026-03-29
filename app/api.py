"""
FastAPI anomaly detection API.

**Orion inference:** if ``data/models/orion_pretrained.pkl`` exists (or ``ORION_PRETRAINED_PATH``),
it is loaded at startup. ``POST /detect`` with ``use_orion=true`` and ``refit_from_train=false``
(default) runs **inference only** on that pickle—no refit per request.

Train the pickle with::

  python train_orion.py

Run with the **same Python** that has ``orion-ml`` + TensorFlow (the one you used for ``train_orion.py``).
If ``.venv310`` in the repo has no Orion (common on Windows), use a short-path venv, e.g.::

  cd <repo-root>
  C:\\venv\\orion310\\Scripts\\python.exe -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

Or from ``app/`` with that interpreter::

  C:\\venv\\orion310\\Scripts\\python.exe -m uvicorn api:app --reload
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.model import OrionTimeSeriesModel, orion_import_available
from src.preprocess import TimeSeries, load_csv_time_series

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = _PROJECT_ROOT / "data" / "synthetic_machine_failure_train.csv"
DEFAULT_QUANTILE = 0.995

_ORION_PRETRAINED: OrionTimeSeriesModel | None = None

# Swagger UI often pre-fills optional strings with the literal "string" — treat as "use default".
_TRAIN_CSV_PLACEHOLDERS = frozenset({"", "string", "null", "none", "optional"})


def pretrained_orion_path() -> Path:
    env = os.environ.get("ORION_PRETRAINED_PATH")
    if env:
        p = Path(env.strip())
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / "data" / "models" / "orion_pretrained.pkl"


def _try_load_orion_pretrained() -> None:
    global _ORION_PRETRAINED
    _ORION_PRETRAINED = None
    path = pretrained_orion_path()
    if not path.exists():
        logger.info("No pretrained Orion pickle at %s — will refit from train_csv when use_orion=true", path)
        return
    if not orion_import_available():
        logger.warning(
            "Pretrained pickle exists at %s but ``orion`` cannot be imported in this interpreter (%s). "
            "Start uvicorn with the Python where you ran ``train_orion.py`` (TensorFlow + orion-ml installed), "
            "e.g. C:\\venv\\orion310\\Scripts\\python.exe -m uvicorn app.api:app --reload",
            path,
            sys.executable,
        )
        return
    try:
        _ORION_PRETRAINED = OrionTimeSeriesModel.from_orion_pickle(path)
        logger.info("Loaded pretrained Orion model from %s", path)
    except Exception:
        logger.exception("Failed to load pretrained Orion from %s", path)


def _resolve_train_csv_path(train_csv: Optional[str]) -> Path:
    if train_csv is None:
        return DEFAULT_TRAIN_PATH
    s = train_csv.strip()
    if not s or s.lower() in _TRAIN_CSV_PLACEHOLDERS:
        return DEFAULT_TRAIN_PATH
    p = Path(s)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


class PointIn(BaseModel):
    timestamp: datetime
    value: float


class DetectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "points": [
                    {"timestamp": "2024-06-01T12:00:00", "value": 0.0},
                    {"timestamp": "2024-06-01T13:00:00", "value": 2.5},
                    {"timestamp": "2024-06-01T14:00:00", "value": 0.05},
                ],
                "threshold_quantile": 0.99,
                "use_orion": True,
                "refit_from_train": False,
                "train_csv": None,
            }
        }
    )

    points: list[PointIn] = Field(default_factory=list, description="Time-ordered sensor points to score.")
    threshold_quantile: float = Field(
        default=0.995,
        description="Only used when ``use_orion`` is false (z-score baseline). Ignored for pretrained Orion.",
    )
    use_orion: bool = Field(
        default=True,
        description="If true, use pretrained Orion pickle when available (see ``refit_from_train``); else z-score.",
    )
    refit_from_train: bool = Field(
        default=False,
        description=(
            "If false and a pretrained Orion ``.pkl`` exists, **skip training** and only run ``predict``. "
            "If true, fit Orion (slow) from ``train_csv`` on every request."
        ),
    )
    train_csv: Optional[str] = Field(
        default=None,
        description=(
            "Train CSV for **baseline** mode or when ``refit_from_train`` is true (Orion). "
            "Relative paths are from the **repository root**."
        ),
    )


class DetectResponse(BaseModel):
    threshold: float
    total_points: int
    anomaly_count: int
    anomalies: list[dict]
    meta: dict[str, Any] = Field(default_factory=dict, description="Backend + inference_source, etc.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_orion_pretrained()
    yield


app = FastAPI(title="Industrial Anomaly Detection API", version="1.0.0", lifespan=lifespan)


def _load_train_series(
    train_csv: Optional[str],
    *,
    q: float,
    use_orion: bool,
) -> OrionTimeSeriesModel:
    path = _resolve_train_csv_path(train_csv)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Training CSV not found: {path}")
    train_series = load_csv_time_series(path)
    model = OrionTimeSeriesModel(
        config={
            "use_orion": use_orion,
            "threshold_quantile": q,
        }
    )
    try:
        model.fit(train_series)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model fit failed: {e}") from e
    return model


def _request_to_series(points: list[PointIn]) -> TimeSeries:
    if not points:
        raise HTTPException(status_code=400, detail="`points` must not be empty.")
    points_sorted = sorted(points, key=lambda p: p.timestamp)
    return TimeSeries(
        timestamps=[p.timestamp for p in points_sorted],
        values=[p.value for p in points_sorted],
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "industrial-anomaly-api",
        "health": "/health",
        "detect": "POST /detect",
        "docs": "/docs",
        "pretrained_orion_loaded": _ORION_PRETRAINED is not None,
        "pretrained_orion_path": str(pretrained_orion_path()),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pretrained_orion_loaded": _ORION_PRETRAINED is not None}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    q = min(max(req.threshold_quantile, 0.50), 0.9999)

    if req.use_orion and not req.refit_from_train and _ORION_PRETRAINED is not None:
        model = _ORION_PRETRAINED
        inference_source = "pretrained_orion_pickle"
    elif req.use_orion:
        model = _load_train_series(req.train_csv, q=q, use_orion=True)
        inference_source = "refit_train_csv"
    else:
        model = _load_train_series(req.train_csv, q=q, use_orion=False)
        inference_source = "baseline_zscore_train_csv"

    series = _request_to_series(req.points)

    try:
        result = model.predict(series)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    anomalies = []
    for ts, value, score, flag in zip(series.timestamps, series.values, result.scores, result.is_anomaly):
        if flag:
            anomalies.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "value": value,
                    "score": score,
                }
            )
    meta = dict(result.meta or {})
    meta["fitted_backend"] = model.fitted_backend
    meta["inference_source"] = inference_source
    return DetectResponse(
        threshold=result.threshold,
        total_points=len(series.values),
        anomaly_count=len(anomalies),
        anomalies=anomalies,
        meta=meta,
    )


def create_app():
    return app
