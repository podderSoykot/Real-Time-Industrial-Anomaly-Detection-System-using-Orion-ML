"""
Time-series anomaly detection using MIT/Sintel **Orion ML** when available.

Uses ``orion.Orion`` with a configurable pipeline (default ``lstm_dynamic_threshold``).
Falls back to a z-score baseline if ``orion`` is not installed or fit/detect fails.

Orion expects ``pandas.DataFrame`` columns ``timestamp`` (Unix seconds) and ``value``.
``detect()`` returns event intervals (start, end, severity); we map those to per-point
scores and flags for the rest of this project.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from pathlib import Path
from typing import Any, Literal

from .preprocess import TimeSeries


@dataclass
class DetectionResult:
    scores: list[float]
    is_anomaly: list[bool]
    threshold: float
    meta: dict[str, Any] | None = None


def _try_orion():
    try:
        from orion import Orion  # type: ignore

        return Orion
    except ImportError:
        return None


def orion_import_available() -> bool:
    """True if ``from orion import Orion`` works (package installed)."""
    return _try_orion() is not None


def _timeseries_to_orion_df(series: TimeSeries):
    import pandas as pd

    stamps = [int(t.timestamp()) for t in series.timestamps]
    return pd.DataFrame({"timestamp": stamps, "value": series.values})


def _intervals_to_point_labels(series: TimeSeries, events) -> tuple[list[float], list[bool], float]:
    """Map Orion event rows (start, end, severity) to per-point score and flag."""
    import pandas as pd

    n = len(series.timestamps)
    scores = [0.0] * n
    flags = [False] * n

    if events is None or len(events) == 0:
        return scores, flags, 0.0

    # Normalize column names (some versions may differ)
    cols = {c.lower(): c for c in events.columns}
    start_c = cols.get("start")
    end_c = cols.get("end")
    sev_c = cols.get("severity") or cols.get("score")
    if start_c is None or end_c is None:
        return scores, flags, 0.0

    for _, row in events.iterrows():
        s = int(float(row[start_c]))
        e = int(float(row[end_c]))
        if sev_c is not None and sev_c in row.index and pd.notna(row[sev_c]):
            sev = float(row[sev_c])
        else:
            sev = 1.0
        for i, ts in enumerate(series.timestamps):
            u = int(ts.timestamp())
            if s <= u <= e:
                flags[i] = True
                scores[i] = max(scores[i], sev)

    # Threshold: Orion uses interval severity; treat score > eps as anomaly
    pos = [s for s in scores if s > 0]
    threshold = min(pos) * 0.5 if pos else 1e-9
    return scores, flags, threshold


class _BaselineZScore:
    """Fallback when Orion is unavailable."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._state: dict[str, float] | None = None

    def fit(self, train_series: TimeSeries) -> None:
        vals = train_series.values
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) >= 2 else 1e-9
        std_v = max(std_v, 1e-9)
        train_scores = [abs((v - mean_v) / std_v) for v in vals]
        q = float(self.config.get("threshold_quantile", 0.995))
        q = min(max(q, 0.50), 0.9999)
        threshold = _quantile(train_scores, q)
        self._state = {"mean": mean_v, "std": std_v, "threshold": threshold}

    def predict(self, series: TimeSeries) -> DetectionResult:
        assert self._state is not None
        mean_v = self._state["mean"]
        std_v = self._state["std"]
        threshold = self._state["threshold"]
        scores = [abs((v - mean_v) / std_v) for v in series.values]
        flags = [s > threshold for s in scores]
        return DetectionResult(
            scores=scores,
            is_anomaly=flags,
            threshold=threshold,
            meta={"backend": "baseline_zscore", "mean": mean_v, "std": std_v},
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


class OrionTimeSeriesModel:
    """
    Wraps Sintel Orion ``Orion`` (LSTM pipeline by default) with a z-score fallback.

    Config keys:
      - ``use_orion`` (bool, default True): try Orion first
      - ``pipeline`` (str): e.g. ``lstm_dynamic_threshold``
      - ``hyperparameters`` (dict): passed to ``Orion(...)``
      - ``threshold_quantile``: only used for baseline fallback
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._backend: Literal["orion", "baseline"] | None = None
        self._orion: Any = None
        self._baseline: _BaselineZScore | None = None
        self._last_error: str | None = None

    @property
    def fitted_backend(self) -> Literal["orion", "baseline"] | None:
        """``orion`` after a successful Orion ``fit``; ``baseline`` if fallback was used."""
        return self._backend

    @property
    def last_orion_error(self) -> str | None:
        """If ``fit`` fell back to baseline, the exception message from Orion ``fit``."""
        return self._last_error

    def save_orion(self, path: str | Path) -> None:
        """Persist a fitted Orion instance (``orion.Orion.save`` pickle). Requires ``fitted_backend == 'orion'``."""
        if self._backend != "orion" or self._orion is None:
            raise RuntimeError("No fitted Orion model to save (backend is not orion).")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._orion.save(str(out))

    @classmethod
    def from_orion_pickle(cls, path: str | Path, config: dict[str, Any] | None = None) -> OrionTimeSeriesModel:
        """Load a model produced by ``save_orion`` / ``Orion.save`` for inference-only ``predict`` calls."""
        OrionCls = _try_orion()
        if OrionCls is None:
            raise RuntimeError(
                "orion-ml is not installed in this Python environment; cannot unpickle Orion. "
                "Use the same interpreter where TensorFlow and orion-ml are installed."
            )
        orion = OrionCls.load(str(path))
        cfg: dict[str, Any] = dict(config or {})
        pipe = getattr(orion, "_pipeline", None)
        if "pipeline" not in cfg and isinstance(pipe, str):
            cfg["pipeline"] = pipe
        m = cls(cfg)
        m._orion = orion
        m._backend = "orion"
        m._baseline = None
        m._last_error = None
        return m

    def fit(self, train_series: TimeSeries) -> None:
        if not train_series.values:
            raise ValueError("Training series is empty.")

        self._last_error = None
        self._orion = None
        self._baseline = None
        self._backend = None

        use_orion = bool(self.config.get("use_orion", True))
        OrionCls = _try_orion() if use_orion else None

        if OrionCls is not None:
            try:
                pipeline = self.config.get("pipeline", "lstm_dynamic_threshold")
                # Default pipeline aggregates to 6h bins. Train ~1400h → ~233 rows; test ~600h → ~100.
                # window_size must be below both so fit() and detect() get non-empty windows (default 250 fails).
                hyper = self.config.get(
                    "hyperparameters",
                    {
                        "mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1": {
                            "window_size": 80,
                        },
                        "keras.Sequential.LSTMTimeSeriesRegressor#1": {
                            "epochs": int(self.config.get("lstm_epochs", 5)),
                            "verbose": 0,
                        },
                    },
                )
                self._orion = OrionCls(pipeline=pipeline, hyperparameters=hyper)
                train_df = _timeseries_to_orion_df(train_series)
                self._orion.fit(train_df)
                self._backend = "orion"
                self._baseline = None
                return
            except Exception as exc:  # noqa: BLE001
                # Fall back so the app still runs without a full TF stack
                self._orion = None
                self._baseline = _BaselineZScore(self.config)
                self._baseline.fit(train_series)
                self._backend = "baseline"
                self._last_error = str(exc)
                return

        self._baseline = _BaselineZScore(self.config)
        self._baseline.fit(train_series)
        self._backend = "baseline"

    def predict(self, series: TimeSeries) -> DetectionResult:
        if self._backend is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if self._backend == "baseline" and self._baseline is not None:
            r = self._baseline.predict(series)
            if getattr(self, "_last_error", None):
                m = dict(r.meta or {})
                m["orion_error"] = self._last_error
                r = DetectionResult(
                    scores=r.scores,
                    is_anomaly=r.is_anomaly,
                    threshold=r.threshold,
                    meta=m,
                )
            return r

        if self._orion is None:
            raise RuntimeError("Orion model missing after fit.")

        infer_df = _timeseries_to_orion_df(series)
        events = self._orion.detect(infer_df, visualization=False)
        scores, flags, threshold = _intervals_to_point_labels(series, events)
        pipe = self.config.get("pipeline")
        if not pipe:
            p = getattr(self._orion, "_pipeline", None)
            pipe = p if isinstance(p, str) else "lstm_dynamic_threshold"
        return DetectionResult(
            scores=scores,
            is_anomaly=flags,
            threshold=threshold,
            meta={
                "backend": "orion_ml",
                "pipeline": pipe,
                "events_rows": int(len(events)) if events is not None else 0,
            },
        )
