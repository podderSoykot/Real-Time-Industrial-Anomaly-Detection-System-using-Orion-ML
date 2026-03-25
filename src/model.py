"""
Model interface for Orion ML (placeholder).

Orion ML integration will depend on the exact Orion API you use.
This file provides a minimal structure so you can plug in your model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectionResult:
    # For now keep this generic; extend when Orion pipeline outputs are known.
    scores: list[float]
    threshold: float | None = None
    meta: dict[str, Any] | None = None


class OrionTimeSeriesModel:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._model: Any = None

    def fit(self, train_series: Any) -> None:
        # TODO: integrate Orion ML pipeline fitting here.
        self._model = True

    def predict(self, series: Any) -> DetectionResult:
        # TODO: integrate Orion ML detection here.
        return DetectionResult(scores=[], threshold=None, meta={"stub": True})

