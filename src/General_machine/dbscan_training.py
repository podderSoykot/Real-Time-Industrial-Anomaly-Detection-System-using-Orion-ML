from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.model import DetectionResult
from src.preprocess import TimeSeries, load_csv_time_series


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast rolling mean/std with a fixed window.
    For edges (< window) it uses the available prefix length.
    """
    n = len(values)
    means = np.empty(n, dtype=float)
    stds = np.empty(n, dtype=float)

    # For prototype simplicity and clarity: compute per index; n is small (stream windows).
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        means[i] = float(np.mean(chunk))
        # Population std (consistent with pstdev usage elsewhere).
        stds[i] = float(np.std(chunk, ddof=0)) if len(chunk) >= 2 else 0.0
    return means, stds


def _build_features(values: np.ndarray, window: int) -> np.ndarray:
    """
    Build a fixed-size feature vector per sample.

    Feature order:
      [value, rolling_mean, rolling_std, slope]
    """
    values = values.astype(float)
    mean, std = _rolling_mean_std(values, window=window)
    slope = np.empty_like(values)
    slope[0] = 0.0
    slope[1:] = values[1:] - values[:-1]
    # Avoid extremely tiny std; helps stability for standardization.
    std = np.where(std < 1e-12, 0.0, std)
    return np.stack([values, mean, std, slope], axis=1)


@dataclass
class DbscanConfig:
    window_size: int = 40
    eps: float = 0.7
    min_samples: int = 10
    metric: str = "euclidean"


class DbscanAnomalyModel:
    """
    DBSCAN clustering wrapper for anomaly detection.

    DBSCAN has no official `predict()` for new points. For online inference we:
      - Train DBSCAN
      - Keep core samples (`components_`)
      - For new samples, compute distance to nearest core sample
      - Mark anomaly if distance > eps
    """

    def __init__(self, config: DbscanConfig):
        self.config = config
        self._scaler: StandardScaler | None = None
        self._nn: NearestNeighbors | None = None
        self._core_components: np.ndarray | None = None
        self._fitted = False

    def fit(self, series: TimeSeries) -> "DbscanAnomalyModel":
        values = np.asarray(series.values, dtype=float)
        X = _build_features(values, window=self.config.window_size)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        db = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
        )
        labels = db.fit_predict(Xs)

        core_mask = np.asarray(db.core_sample_indices_, dtype=int)
        if core_mask.size == 0:
            # Degenerate case: no cores found -> treat everything as anomaly.
            self._core_components = np.zeros((1, Xs.shape[1]), dtype=float)
        else:
            self._core_components = Xs[core_mask]

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(self._core_components)

        self._scaler = scaler
        self._nn = nn
        self._fitted = True
        return self

    def to_state(self) -> dict[str, Any]:
        if not self._fitted or self._scaler is None or self._nn is None or self._core_components is None:
            raise RuntimeError("DBSCAN model not fitted; cannot serialize.")

        # StandardScaler stores:
        # - mean_  (per feature)
        # - scale_ (per feature std)
        state: dict[str, Any] = {
            "type": "dbscan_core_distance",
            "config": {
                "window_size": int(self.config.window_size),
                "eps": float(self.config.eps),
                "min_samples": int(self.config.min_samples),
                "metric": str(self.config.metric),
            },
            "scaler": {
                "mean_": np.asarray(self._scaler.mean_, dtype=float),
                "scale_": np.asarray(self._scaler.scale_, dtype=float),
            },
            "core_components": np.asarray(self._core_components, dtype=float),
            "n_features_in_": int(getattr(self._scaler, "n_features_in_", self._core_components.shape[1])),
        }
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "DbscanAnomalyModel":
        cfg = state.get("config") or {}
        config = DbscanConfig(
            window_size=int(cfg.get("window_size", 40)),
            eps=float(cfg.get("eps", 0.7)),
            min_samples=int(cfg.get("min_samples", 10)),
            metric=str(cfg.get("metric", "euclidean")),
        )
        m = cls(config=config)

        scaler_state = state.get("scaler") or {}
        mean_ = np.asarray(scaler_state.get("mean_"), dtype=float)
        scale_ = np.asarray(scaler_state.get("scale_"), dtype=float)

        scaler = StandardScaler()
        scaler.mean_ = mean_
        scaler.scale_ = scale_
        scaler.var_ = scale_ ** 2
        scaler.n_features_in_ = int(state.get("n_features_in_", mean_.shape[0]))

        core = np.asarray(state.get("core_components"), dtype=float)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(core)

        m._scaler = scaler
        m._nn = nn
        m._core_components = core
        m._fitted = True
        return m

    def predict(self, series: TimeSeries) -> DetectionResult:
        if not self._fitted or self._scaler is None or self._nn is None or self._core_components is None:
            raise RuntimeError("DBSCAN model not fitted.")

        values = np.asarray(series.values, dtype=float)
        X = _build_features(values, window=self.config.window_size)
        Xs = self._scaler.transform(X)

        # nearest distance to any core sample in standardized feature space
        dists, _idx = self._nn.kneighbors(Xs, n_neighbors=1, return_distance=True)
        dists = dists.reshape(-1)

        eps = float(self.config.eps)
        flags = [bool(d > eps) for d in dists]
        # Severity: how far beyond eps we are (0..)
        scores = [max(0.0, float(d - eps)) for d in dists]

        # Keep shape compatible with frontend/backend expectations.
        return DetectionResult(
            scores=scores,
            is_anomaly=flags,
            threshold=eps,
            meta={
                "backend": "dbscan_core_distance",
                "window_size": self.config.window_size,
                "eps": self.config.eps,
                "min_samples": self.config.min_samples,
                "metric": self.config.metric,
                "core_samples": int(self._core_components.shape[0]),
            },
        )


def train_and_save(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    window_size: int = 40,
    eps: float = 0.7,
    min_samples: int = 10,
    metric: str = "euclidean",
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_csv_time_series(csv_path)
    model = DbscanAnomalyModel(
        DbscanConfig(
            window_size=window_size,
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
    ).fit(series)

    out_path = out_dir / "dbscan_model.joblib"
    joblib.dump(model.to_state(), out_path)
    return out_path


def _main() -> int:
    p = argparse.ArgumentParser(description="Train DBSCAN-based anomaly detector.")
    p.add_argument("--csv", type=str, required=True, help="CSV with columns timestamp,value")
    p.add_argument("--out-dir", type=str, required=True, help="Directory to save dbscan_model.joblib")
    p.add_argument("--window-size", type=int, default=40)
    p.add_argument("--eps", type=float, default=0.7)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--metric", type=str, default="euclidean")
    args = p.parse_args()

    out_path = train_and_save(
        args.csv,
        args.out_dir,
        window_size=args.window_size,
        eps=args.eps,
        min_samples=args.min_samples,
        metric=args.metric,
    )
    print(f"Saved DBSCAN model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
