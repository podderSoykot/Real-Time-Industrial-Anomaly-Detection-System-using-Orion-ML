# Real-Time Industrial Anomaly Detection (Orion ML)

This repo provides a starting point for building a streaming anomaly detection system using **Orion ML** on time-series sensor data.

## Dataset

The project includes:

- an extracted NAB-style dataset at `Dataset/archive/` (real benchmark-style series)
- a generated synthetic dataset at `data/synthetic_machine_failure.csv`

All series use the format:

`timestamp,value`

Run:

```powershell
python make_synthetic_data.py --outdir "data"
```

## EDA and Plots

Generate EDA summary + meaningful plots:

```powershell
.\.venv310\Scripts\python src/eda_.py --csv "data/synthetic_machine_failure.csv" --meta "data/synthetic_machine_failure_meta.json" --out "data/eda_summary.json" --plots-dir "data/plots"
```

### 1) Time Series with Injected Anomaly Windows

![Time Series with Anomalies](data/plots/timeseries_with_anomalies.png)

### 2) Rolling Mean and Volatility Envelope

![Rolling Mean and Std](data/plots/rolling_mean_std.png)

### 3) Distribution View (Histogram + Box Plot)

![Distribution Plots](data/plots/distribution_hist_box.png)

## Model Training

Train and run anomaly scoring on the synthetic train/test split:

```powershell
.\.venv310\Scripts\python main.py
```

Expected console output includes:
- learned threshold
- number of test rows
- number of detected anomalies

Prediction output file:
- `data/predictions_test.csv` with columns:
  - `timestamp`
  - `value`
  - `score`
  - `is_anomaly`

## Project structure

## Notes on Orion ML installation

`orion-ml` has legacy dependencies and may require an older Python environment to install successfully.