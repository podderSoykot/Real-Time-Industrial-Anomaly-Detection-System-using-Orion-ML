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
from typing import Iterable

from .preprocess import TimeSeries
from .model import DetectionResult, OrionTimeSeriesModel


@dataclass
class AnomalyPoint:
    timestamp: datetime
    score: float


def detect_anomalies(model: OrionTimeSeriesModel, series: TimeSeries) -> list[AnomalyPoint]:
    result = model.predict(series)
    # Placeholder: if result.scores is aligned to input points, map them.
    # Since this is a stub, return empty for now.
    _ = result
    return []

