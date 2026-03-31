from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.General_machine.dbscan_training import DbscanAnomalyModel
from src.preprocess import load_csv_time_series


def _make_point_labels(meta: dict[str, Any], test_len: int) -> np.ndarray:
    train_rows = int(meta.get("train_rows", 0))
    anomalies: list[tuple[int, int]] = []
    for key in ("anomaly_1", "anomaly_2"):
        a = meta.get(key)
        if isinstance(a, list) and len(a) >= 2:
            s = int(a[0])
            e = int(a[1])
            anomalies.append((s, e))

    # Anomaly windows in the meta file use full-series indices and are end-exclusive.
    y = np.zeros(test_len, dtype=int)
    for i in range(test_len):
        full_idx = train_rows + i
        for s, e in anomalies:
            if s <= full_idx < e:
                y[i] = 1
                break
    return y


def _contiguous_flag_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert boolean flags into intervals [start, end) of consecutive True values.
    """
    intervals: list[tuple[int, int]] = []
    n = len(flags)
    i = 0
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
    # overlap if intervals intersect
    return not (a[1] <= b[0] or b[1] <= a[0])


def evaluate(
    *,
    model_path: str | Path,
    test_csv_path: str | Path,
    meta_path: str | Path,
) -> None:
    model_path = Path(model_path)
    test_csv_path = Path(test_csv_path)
    meta_path = Path(meta_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    state = joblib.load(model_path)
    if not isinstance(state, dict):
        raise ValueError(f"Expected state dict in {model_path}, got: {type(state)}")

    model = DbscanAnomalyModel.from_state(state)

    series = load_csv_time_series(test_csv_path)
    res = model.predict(series)

    scores = np.asarray(res.scores, dtype=float)
    y_pred = np.asarray(res.is_anomaly, dtype=int)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    y_true = _make_point_labels(meta, test_len=len(series.values))

    # Point-level metrics (using the model's built-in eps threshold -> y_pred).
    has_pos = bool(y_true.sum() > 0)
    if has_pos:
        auc = roc_auc_score(y_true, scores)
    else:
        auc = float("nan")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Event-level metrics: treat contiguous predicted anomalies as events,
    # and ground-truth anomaly windows as events.
    train_rows = int(meta.get("train_rows", 0))
    gt_windows_full: list[tuple[int, int]] = []
    for key in ("anomaly_1", "anomaly_2"):
        a = meta.get(key)
        if isinstance(a, list) and len(a) >= 2:
            gt_windows_full.append((int(a[0]), int(a[1])))

    gt_windows_test: list[tuple[int, int]] = []
    for s, e in gt_windows_full:
        # shift full-series indices into test indices
        ts = s - train_rows
        te = e - train_rows
        # keep only intersection with [0, test_len)
        ts = max(0, ts)
        te = min(len(y_true), te)
        if ts < te:
            gt_windows_test.append((ts, te))

    pred_intervals = _contiguous_flag_intervals(y_pred.astype(bool))

    gt_detected = 0
    for gt in gt_windows_test:
        if any(_event_overlap(gt, p) for p in pred_intervals):
            gt_detected += 1

    num_gt = len(gt_windows_test)
    num_pred = len(pred_intervals)
    event_precision = (gt_detected / num_pred) if num_pred > 0 else (1.0 if num_gt == 0 else 0.0)
    event_recall = (gt_detected / num_gt) if num_gt > 0 else 1.0
    event_f1 = (
        0.0
        if (event_precision + event_recall) <= 0
        else 2.0 * event_precision * event_recall / (event_precision + event_recall)
    )

    # Print report
    print("DBSCAN evaluation (test set)")
    print(f"- test_csv: {test_csv_path}")
    print(f"- meta: {meta_path}")
    print(f"- model: {model_path.name}")
    print("")
    print("Point-level (per sample)")
    print(f"- positives: {int(y_true.sum())} / {len(y_true)}")
    print(f"- predicted anomalies: {int(y_pred.sum())} / {len(y_pred)}")
    print(f"- precision: {precision:.4f}")
    print(f"- recall:    {recall:.4f}")
    print(f"- f1:        {f1:.4f}")
    print(f"- roc_auc:   {auc:.4f}" if has_pos else "- roc_auc: nan (no positives)")
    print("")
    print("Event-level (window detection)")
    print(f"- ground-truth anomaly windows (test indices): {gt_windows_test}")
    print(f"- predicted anomaly events (test indices): {pred_intervals[:10]}{'...' if len(pred_intervals) > 10 else ''}")
    print(f"- event precision: {event_precision:.4f}")
    print(f"- event recall:    {event_recall:.4f}")
    print(f"- event f1:        {event_f1:.4f}")


def _main() -> int:
    p = argparse.ArgumentParser(description="Evaluate DBSCAN anomaly model.")
    p.add_argument("--model", type=str, default="src/General_machine/dbscan_model.joblib")
    p.add_argument("--test-csv", type=str, default="data/synthetic_machine_failure_test.csv")
    p.add_argument("--meta", type=str, default="data/synthetic_machine_failure_meta.json")
    args = p.parse_args()
    evaluate(model_path=args.model, test_csv_path=args.test_csv, meta_path=args.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

