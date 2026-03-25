# Real-Time Industrial Anomaly Detection (Orion ML)

This repo provides a starting point for building a streaming anomaly detection system using **Orion ML** on time-series sensor data.

## Dataset

The project includes an extracted NAB-style dataset at `Dataset/archive/` with files in the format:

`timestamp,value`

Run:

```powershell
python dataset.py --root "Dataset/archive" --out "Dataset/analysis_report.csv"
python eda_analysis.py --top 10
```

## Project structure

```text
data/
notebooks/
src/
  preprocess.py
  model.py
  detect.py
app/
  api.py
  dashboard.py
```

## Notes on Orion ML installation

`orion-ml` has legacy dependencies and may require an older Python environment to install successfully.

