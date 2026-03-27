"""
Detection entrypoints (placeholder).

This is where you would:
- Load a CSV series
- Preprocess (windowing / normalization)
- Run Orion ML pipeline
- Output anomalies (timestamps + scores)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .preprocess import TimeSeries
from .model import OrionTimeSeriesModel


@dataclass
class AnomalyPoint:
    timestamp: datetime
    score: float


def detect_anomalies(model: OrionTimeSeriesModel, series: TimeSeries) -> list[AnomalyPoint]:
    result = model.predict(series)
    out: list[AnomalyPoint] = []
    for ts, score, flag in zip(series.timestamps, result.scores, result.is_anomaly):
        if flag:
            out.append(AnomalyPoint(timestamp=ts, score=score))
    return out

