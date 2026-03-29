"""
Train and run inference with **Sintel Orion ML** only (no silent z-score fallback).

Prerequisites:
  pip install -r requirements.txt
  pip install -r requirements-orion.txt   # TensorFlow + orion-ml (may fail on some Windows paths)

Usage:
  python train_orion.py
  python train_orion.py --lstm-epochs 10 --out data/predictions_orion_test.csv

Writes ``data/models/orion_pretrained.pkl`` by default for the FastAPI app (``--no-save-model`` to skip).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.detect import detect_anomalies
from src.model import OrionTimeSeriesModel, orion_import_available
from src.preprocess import load_csv_time_series


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Orion ML and score test CSV.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/synthetic_machine_failure_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/synthetic_machine_failure_test.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/predictions_orion_test.csv"))
    parser.add_argument("--pipeline", type=str, default="lstm_dynamic_threshold")
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("data/models/orion_pretrained.pkl"),
        help="Path to save fitted Orion pickle for the API (Orion.save). Ignored with --no-save-model.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not write a pickle file after training.",
    )
    args = parser.parse_args()

    if not orion_import_available():
        print(
            "ERROR: Orion ML is not installed.\n"
            "  pip install -r requirements.txt\n"
            "  pip install -r requirements-orion.txt\n"
            "On Windows, if TensorFlow fails with long paths, see README "
            "\"Windows: Orion / TensorFlow install failures\".",
            file=sys.stderr,
        )
        return 1

    if not args.train_csv.exists() or not args.test_csv.exists():
        print("ERROR: train or test CSV missing. Run: python make_synthetic_data.py --outdir data", file=sys.stderr)
        return 1

    train = load_csv_time_series(args.train_csv)
    test = load_csv_time_series(args.test_csv)

    model = OrionTimeSeriesModel(
        config={
            "use_orion": True,
            "pipeline": args.pipeline,
            "lstm_epochs": args.lstm_epochs,
        }
    )
    print("Fitting Orion pipeline:", args.pipeline, f"(LSTM epochs={args.lstm_epochs}) …")
    model.fit(train)

    if model.fitted_backend != "orion":
        err = model.last_orion_error or "unknown"
        print("ERROR: Orion fit did not succeed (fell back to baseline).", file=sys.stderr)
        print(err, file=sys.stderr)
        return 2

    if not args.no_save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_orion(args.save_model)
        print("Saved Orion model:", args.save_model.resolve())

    result = model.predict(test)
    anomalies = detect_anomalies(model, test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "score", "is_anomaly"])
        for ts, value, score, flag in zip(test.timestamps, test.values, result.scores, result.is_anomaly):
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{value:.10g}", f"{score:.6f}", int(flag)])

    print("Orion ML trained and predictions saved.")
    print("Backend:", result.meta.get("backend") if result.meta else "orion_ml")
    if result.meta:
        print("Pipeline:", result.meta.get("pipeline"))
        print("Raw anomaly events (intervals):", result.meta.get("events_rows"))
    print("Threshold (mapped):", f"{result.threshold:.6f}")
    print("Test rows:", len(test.values))
    print("Points flagged anomalous:", sum(result.is_anomaly))
    print("Anomaly points (detect_anomalies):", len(anomalies))
    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
