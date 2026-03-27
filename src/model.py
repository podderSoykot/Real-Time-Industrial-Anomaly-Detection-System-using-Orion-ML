"""
Simple anomaly model for time-series values.

This implementation is dependency-light and works on `timestamp,value` data.
It learns baseline mean/std on train data and uses absolute z-score as anomaly
score. Threshold is set from a chosen train quantile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import statistics

from .preprocess import TimeSeries


@dataclass
class DetectionResult:
    scores: list[float]
    is_anomaly: list[bool]
    threshold: float
    meta: dict[str, Any] | None = None


class OrionTimeSeriesModel:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._model: dict[str, float] | None = None

    def fit(self, train_series: TimeSeries) -> None:
        if not train_series.values:
            raise ValueError("Training series is empty.")

        vals = train_series.values
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) >= 2 else 1e-9
        std_v = max(std_v, 1e-9)

        train_scores = [abs((v - mean_v) / std_v) for v in vals]
        q = float(self.config.get("threshold_quantile", 0.995))
        q = min(max(q, 0.50), 0.9999)
        threshold = _quantile(train_scores, q)

        self._model = {
            "mean": mean_v,
            "std": std_v,
            "threshold": threshold,
            "threshold_quantile": q,
        }

    def predict(self, series: TimeSeries) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        mean_v = self._model["mean"]
        std_v = self._model["std"]
        threshold = self._model["threshold"]

        scores = [abs((v - mean_v) / std_v) for v in series.values]
        flags = [s > threshold for s in scores]
        return DetectionResult(
            scores=scores,
            is_anomaly=flags,
            threshold=threshold,
            meta={"mean": mean_v, "std": std_v},
        )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (k - f) * (xs[c] - xs[f])

