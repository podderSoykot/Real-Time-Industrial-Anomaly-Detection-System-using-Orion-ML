from __future__ import annotations

import csv
from pathlib import Path

from src.detect import detect_anomalies
from src.model import OrionTimeSeriesModel
from src.preprocess import load_csv_time_series


def run() -> int:
    train_csv = Path("data/synthetic_machine_failure_train.csv")
    test_csv = Path("data/synthetic_machine_failure_test.csv")
    out_csv = Path("data/predictions_test.csv")

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Expected train/test CSVs under data/.")

    train = load_csv_time_series(train_csv)
    test = load_csv_time_series(test_csv)

    model = OrionTimeSeriesModel(config={"threshold_quantile": 0.995})
    model.fit(train)
    result = model.predict(test)
    anomalies = detect_anomalies(model, test)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "score", "is_anomaly"])
        for ts, value, score, flag in zip(test.timestamps, test.values, result.scores, result.is_anomaly):
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{value:.10g}", f"{score:.6f}", int(flag)])

    print("Model trained and predictions saved.")
    print("Threshold:", f"{result.threshold:.6f}")
    print("Test rows:", len(test.values))
    print("Anomalies detected:", len(anomalies))
    print("Saved:", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

