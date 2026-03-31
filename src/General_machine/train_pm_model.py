from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainMetrics:
    rows: int
    train_rows: int
    test_rows: int
    accuracy: float
    f1_macro: float
    f1_weighted: float
    target: str
    classes: list[int]


def _ts_features(ts_text: str) -> dict[str, float]:
    ts = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
    sec = ts.hour * 3600 + ts.minute * 60 + ts.second
    day_frac = sec / 86400.0
    dow_frac = ts.weekday() / 7.0
    return {
        "sec_sin": math.sin(2.0 * math.pi * day_frac),
        "sec_cos": math.cos(2.0 * math.pi * day_frac),
        "dow_sin": math.sin(2.0 * math.pi * dow_frac),
        "dow_cos": math.cos(2.0 * math.pi * dow_frac),
    }


def _read_pm_csv(path: Path, target_col: str) -> tuple[list[dict[str, Any]], list[int]]:
    X: list[dict[str, Any]] = []
    y: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"timestamp", "value", "machine_name", "sensor_id", target_col}
        missing = required.difference(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in r:
            feats: dict[str, Any] = {
                "value": float(row["value"]),
                "machine_name": row["machine_name"],
                "sensor_id": row["sensor_id"],
            }
            feats.update(_ts_features(row["timestamp"]))
            X.append(feats)
            y.append(int(row[target_col]))
    return X, y


def train_pm_model(
    *,
    csv_path: Path,
    out_model: Path,
    out_metrics: Path,
    target_col: str,
    test_size: float,
    seed: int,
    n_estimators: int,
) -> TrainMetrics:
    X_dict, y = _read_pm_csv(csv_path, target_col)
    X_train_d, X_test_d, y_train, y_test = train_test_split(
        X_dict,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_d)
    X_test = vec.transform(X_test_d)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    classes = sorted(set(y))
    metrics = TrainMetrics(
        rows=len(y),
        train_rows=len(y_train),
        test_rows=len(y_test),
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        target=target_col,
        classes=classes,
    )

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "vectorizer": vec,
        "model": clf,
        "target_col": target_col,
        "classes": classes,
        "feature_names": vec.feature_names_,
    }
    joblib.dump(artifact, out_model)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    payload = {
        "summary": asdict(metrics),
        "classification_report": report,
    }
    out_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Train predictive maintenance supervised model.")
    p.add_argument(
        "--csv",
        type=str,
        default="data/predictive_maintenance_pm/pm_100machines_single_100k__single.csv",
        help="Input single CSV with supervised columns",
    )
    p.add_argument(
        "--target",
        type=str,
        default="state",
        choices=("state", "label"),
        help="Supervised target column",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument(
        "--out-model",
        type=str,
        default="src/General_machine/pm_state_model.joblib",
    )
    p.add_argument(
        "--out-metrics",
        type=str,
        default="data/predictive_maintenance_pm/pm_state_model_metrics.json",
    )
    args = p.parse_args()

    metrics = train_pm_model(
        csv_path=Path(args.csv),
        out_model=Path(args.out_model),
        out_metrics=Path(args.out_metrics),
        target_col=args.target,
        test_size=args.test_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )

    print("Training complete")
    print(f"- rows: {metrics.rows}")
    print(f"- train/test: {metrics.train_rows}/{metrics.test_rows}")
    print(f"- target: {metrics.target}")
    print(f"- accuracy: {metrics.accuracy:.4f}")
    print(f"- f1_macro: {metrics.f1_macro:.4f}")
    print(f"- f1_weighted: {metrics.f1_weighted:.4f}")
    print("- model: src/General_machine/pm_state_model.joblib")
    print("- metrics: data/predictive_maintenance_pm/pm_state_model_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

