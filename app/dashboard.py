"""
Streamlit dashboard for anomaly detection.

Run:
  streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import csv

import streamlit as st
from src.model import OrionTimeSeriesModel
from src.preprocess import TimeSeries, load_csv_time_series


def _load_uploaded(uploaded_file) -> TimeSeries:
    # streamlit uploaded file behaves like a file-like object.
    decoded = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(decoded.splitlines())
    timestamps = []
    values = []
    from datetime import datetime

    for row in reader:
        timestamps.append(datetime.strptime(row["timestamp"].strip(), "%Y-%m-%d %H:%M:%S"))
        values.append(float(row["value"]))
    return TimeSeries(timestamps=timestamps, values=values)


def create_dashboard():
    st.set_page_config(page_title="Anomaly Dashboard", layout="wide")
    st.title("Real-Time Industrial Anomaly Detection")

    st.sidebar.header("Settings")
    train_csv = st.sidebar.text_input("Training CSV", "data/synthetic_machine_failure_train.csv")
    threshold_quantile = st.sidebar.slider("Threshold quantile", 0.90, 0.999, 0.995, 0.001)

    uploaded = st.file_uploader("Upload inference CSV (timestamp,value)", type=["csv"])
    default_infer = st.text_input("Or use existing inference CSV path", "data/synthetic_machine_failure_test.csv")

    if st.button("Run Detection"):
        try:
            train_series = load_csv_time_series(train_csv)
            model = OrionTimeSeriesModel(config={"threshold_quantile": threshold_quantile})
            model.fit(train_series)

            if uploaded is not None:
                infer_series = _load_uploaded(uploaded)
                source_name = uploaded.name
            else:
                infer_series = load_csv_time_series(default_infer)
                source_name = default_infer

            result = model.predict(infer_series)
            rows = []
            for ts, value, score, flag in zip(
                infer_series.timestamps, infer_series.values, result.scores, result.is_anomaly
            ):
                rows.append(
                    {
                        "timestamp": ts,
                        "value": value,
                        "score": score,
                        "is_anomaly": int(flag),
                    }
                )

            st.success("Detection complete.")
            st.write(f"Source: `{source_name}`")
            st.write(f"Threshold: `{result.threshold:.4f}`")
            st.write(f"Detected anomalies: `{sum(r['is_anomaly'] for r in rows)}` / `{len(rows)}`")

            st.subheader("Time Series")
            st.line_chart({"value": [r["value"] for r in rows]})

            st.subheader("Anomaly Scores")
            st.line_chart({"score": [r["score"] for r in rows]})

            st.subheader("Anomaly Rows")
            anomaly_rows = [r for r in rows if r["is_anomaly"] == 1]
            st.dataframe(anomaly_rows if anomaly_rows else [{"info": "No anomalies detected"}], use_container_width=True)

        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to run detection: {exc}")


if __name__ == "__main__":
    create_dashboard()

