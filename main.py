from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.detect import detect_anomalies
from src.model import OrionTimeSeriesModel
from src.preprocess import load_csv_time_series


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train anomaly model (Orion if available, else baseline).")
    parser.add_argument("--train-csv", type=Path, default=Path("data/synthetic_machine_failure_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/synthetic_machine_failure_test.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/predictions_test.csv"))
    parser.add_argument("--require-orion", action="store_true", help="Exit with error if Orion did not train.")
    parser.add_argument("--no-orion", action="store_true", help="Force z-score baseline only.")
    parser.add_argument("--lstm-epochs", type=int, default=5, help="Orion LSTM epochs (if Orion is used).")
    args = parser.parse_args(argv)

    train_csv = args.train_csv
    test_csv = args.test_csv
    out_csv = args.out

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Expected train/test CSVs under data/.")

    train = load_csv_time_series(train_csv)
    test = load_csv_time_series(test_csv)

    model = OrionTimeSeriesModel(
        config={
            "use_orion": not args.no_orion,
            "threshold_quantile": 0.995,
            "lstm_epochs": args.lstm_epochs,
        }
    )
    model.fit(train)
    if args.require_orion and model.fitted_backend != "orion":
        print(
            "ERROR: --require-orion but Orion did not train. Install: pip install -r requirements-orion.txt",
            file=sys.stderr,
        )
        if model.last_orion_error:
            print(model.last_orion_error, file=sys.stderr)
        return 2

    result = model.predict(test)
    anomalies = detect_anomalies(model, test)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "score", "is_anomaly"])
        for ts, value, score, flag in zip(test.timestamps, test.values, result.scores, result.is_anomaly):
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{value:.10g}", f"{score:.6f}", int(flag)])

    print("Model trained and predictions saved.")
    print("Fitted backend:", model.fitted_backend)
    if result.meta:
        print("Backend:", result.meta.get("backend", "?"))
        err = result.meta.get("orion_error")
        if err:
            es = str(err)
            print("Orion fallback:", (es[:200] + "…") if len(es) > 200 else es)
    print("Threshold:", f"{result.threshold:.6f}")
    print("Test rows:", len(test.values))
    print("Anomalies detected:", len(anomalies))
    print("Saved:", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

