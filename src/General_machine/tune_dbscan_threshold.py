from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.General_machine.dbscan_training import DbscanAnomalyModel
from src.preprocess import load_csv_time_series


def _make_point_labels(meta: dict, test_len: int) -> np.ndarray:
    train_rows = int(meta.get("train_rows", 0))
    anomalies: list[tuple[int, int]] = []
    for key in ("anomaly_1", "anomaly_2"):
        a = meta.get(key)
        if isinstance(a, list) and len(a) >= 2:
            anomalies.append((int(a[0]), int(a[1])))

    y = np.zeros(test_len, dtype=int)
    for i in range(test_len):
        full_idx = train_rows + i
        for s, e in anomalies:
            if s <= full_idx < e:
                y[i] = 1
                break
    return y


def _contiguous_flag_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        s = i
        while i < n and flags[i]:
            i += 1
        intervals.append((s, i))
    return intervals


def _event_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def evaluate_event_metrics(y_true_windows: list[tuple[int, int]], y_pred_flags: np.ndarray) -> dict:
    pred_intervals = _contiguous_flag_intervals(y_pred_flags.astype(bool))
    gt_detected = 0

    for gt in y_true_windows:
        if any(_event_overlap(gt, p) for p in pred_intervals):
            gt_detected += 1

    num_gt = len(y_true_windows)
    num_pred = len(pred_intervals)
    event_precision = (gt_detected / num_pred) if num_pred > 0 else (1.0 if num_gt == 0 else 0.0)
    event_recall = (gt_detected / num_gt) if num_gt > 0 else 1.0
    event_f1 = 0.0 if (event_precision + event_recall) <= 0 else 2.0 * event_precision * event_recall / (
        event_precision + event_recall
    )
    return {
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "num_pred_events": num_pred,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Tune DBSCAN score threshold for anomaly classification.")
    p.add_argument("--model", type=str, default="src/General_machine/dbscan_model.joblib")
    p.add_argument("--test-csv", type=str, default="data/synthetic_machine_failure_test.csv")
    p.add_argument("--meta", type=str, default="data/synthetic_machine_failure_meta.json")
    args = p.parse_args()

    model_path = Path(args.model)
    test_csv = Path(args.test_csv)
    meta_path = Path(args.meta)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    series = load_csv_time_series(test_csv)

    y_true = _make_point_labels(meta, test_len=len(series.values))

    state = joblib.load(model_path)
    model = DbscanAnomalyModel.from_state(state)
    res = model.predict(series)

    scores = np.asarray(res.scores, dtype=float)

    # Threshold candidates: 0..95th percentile of positive scores.
    pos_scores = scores[scores > 0]
    if pos_scores.size == 0:
        print("No positive scores in test set; DBSCAN produces no anomalies.")
        return 0

    qs = np.linspace(0, 0.95, 31)  # 31 candidates
    thresholds = sorted(set([0.0] + [float(np.quantile(pos_scores, q)) for q in qs]))

    # Ground truth windows in test indices.
    train_rows = int(meta.get("train_rows", 0))
    gt_windows_full: list[tuple[int, int]] = []
    for key in ("anomaly_1", "anomaly_2"):
        a = meta.get(key)
        if isinstance(a, list) and len(a) >= 2:
            gt_windows_full.append((int(a[0]), int(a[1])))
    gt_windows_test = [(max(0, s - train_rows), min(len(y_true), e - train_rows)) for s, e in gt_windows_full]
    gt_windows_test = [(s, e) for (s, e) in gt_windows_test if s < e]

    best = None
    for t in thresholds:
        y_pred = (scores > t).astype(int)

        p1 = precision_score(y_true, y_pred, zero_division=0)
        r1 = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        ev = evaluate_event_metrics(gt_windows_test, y_pred)
        # Primary objective: point F1, tie-break: event F1.
        if best is None or (f1 > best["f1"]) or (abs(f1 - best["f1"]) < 1e-12 and ev["event_f1"] > best["event_f1"]):
            best = {
                "threshold": float(t),
                "precision": float(p1),
                "recall": float(r1),
                "f1": float(f1),
                **ev,
            }

    print("DBSCAN threshold tuning (test set)")
    print(f"- best threshold on score: {best['threshold']:.6f}")
    print(f"- point precision: {best['precision']:.4f}")
    print(f"- point recall:    {best['recall']:.4f}")
    print(f"- point f1:        {best['f1']:.4f}")
    print(f"- event precision: {best['event_precision']:.4f}")
    print(f"- event recall:    {best['event_recall']:.4f}")
    print(f"- event f1:        {best['event_f1']:.4f}")
    print(f"- predicted events: {best['num_pred_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

