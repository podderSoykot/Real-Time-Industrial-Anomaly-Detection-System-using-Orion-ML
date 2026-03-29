"""
FastAPI service for anomaly detection.

Run:
  uvicorn app.api:app --reload
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import OrionTimeSeriesModel
from src.preprocess import TimeSeries, load_csv_time_series


DEFAULT_TRAIN_PATH = Path("data/synthetic_machine_failure_train.csv")
DEFAULT_QUANTILE = 0.995


class PointIn(BaseModel):
    timestamp: datetime
    value: float


class DetectRequest(BaseModel):
    points: list[PointIn] = Field(default_factory=list)
    threshold_quantile: float = 0.995
    train_csv: Optional[str] = None


class DetectResponse(BaseModel):
    threshold: float
    total_points: int
    anomaly_count: int
    anomalies: list[dict]


app = FastAPI(title="Industrial Anomaly Detection API", version="1.0.0")


def _load_train_series(train_csv: Optional[str], q: float) -> OrionTimeSeriesModel:
    path = Path(train_csv) if train_csv else DEFAULT_TRAIN_PATH
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Training CSV not found: {path}")
    train_series = load_csv_time_series(path)
    model = OrionTimeSeriesModel(config={"threshold_quantile": q})
    model.fit(train_series)
    return model


def _request_to_series(points: list[PointIn]) -> TimeSeries:
    if not points:
        raise HTTPException(status_code=400, detail="`points` must not be empty.")
    points_sorted = sorted(points, key=lambda p: p.timestamp)
    return TimeSeries(
        timestamps=[p.timestamp for p in points_sorted],
        values=[p.value for p in points_sorted],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    q = min(max(req.threshold_quantile, 0.50), 0.9999)
    model = _load_train_series(req.train_csv, q=q)
    series = _request_to_series(req.points)

    result = model.predict(series)
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
    return DetectResponse(
        threshold=result.threshold,
        total_points=len(series.values),
        anomaly_count=len(anomalies),
        anomalies=anomalies,
    )


def create_app():
    return app

