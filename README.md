# Predictive Maintenance — Turbofan Engine RUL Prediction

Predict the **Remaining Useful Life (RUL)** of turbofan engines using NASA's
[C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

## Overview

This portfolio project is building an end-to-end ML system, from raw sensor
data to a served model with monitoring and a dashboard. The data pipeline,
model, evaluation, and experiment tracking are complete. Serving, monitoring,
and deployment are in progress, as shown below.

| Stage | Status |
|-------|--------|
| Data loading & exploration | ✅ Completed |
| Data preprocessing (RUL & cleaning) | ✅ Completed |
| Feature engineering (rolling and trend features) | ✅ Completed |
| Model (LightGBM) with engine-grouped cross-validation | ✅ Completed |
| Experiment tracking (MLflow) | ✅ Completed |
| Model serving (FastAPI) | 🔄 Next |
| Drift monitoring (Evidently) | ⬜ Not started |
| Dashboard (Streamlit) | ⬜ Not started |
| CI/CD (GitHub Actions) | ⬜ Not started |
| Containerization (Docker) | ⬜ Not started |

## Dataset

NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS):
- **FD001** — 1 operating condition, 1 fault mode, 100 engines
- **FD002** — 6 operating conditions, 1 fault mode, 260 engines
- **FD003** — 1 operating condition, 2 fault modes, 100 engines
- **FD004** — 6 operating conditions, 2 fault modes, 249 engines

Each engine runs from a healthy state until failure. The goal is to predict
how many cycles remain before failure.

## Project Structure

```
├── data/raw/               # C-MAPSS text files (gitignored, see Quick Start)
├── notebooks/              # Exploratory data analysis
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Rolling and trend features
│   ├── models/             # Training, cross-validation, MLflow logging
│   ├── evaluation/         # RMSE and NASA scoring
│   └── api/                # FastAPI serving (planned)
├── tests/                  # Unit tests (pytest)
├── pyproject.toml          # Ruff and pytest configuration
└── requirements.txt
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the C-MAPSS data from NASA's public repository into data/raw
mkdir -p data/raw
curl -L -o cmapss.zip "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
unzip -j cmapss.zip "*/CMAPSSData.zip"
unzip -j CMAPSSData.zip "train_*" "test_*" "RUL_*" -d data/raw
rm cmapss.zip CMAPSSData.zip

# Run the tests
pytest

# Train the model and log the run to MLflow
python -m src.models.train
mlflow ui --backend-store-uri ./mlruns
```

## Results (FD001)

LightGBM with 62 features, including 30-cycle sensor trends. Settings are scored
with 5-fold cross-validation grouped by engine, using random cut-off cycles
averaged over 50 draws. The final model is then trained on all 100 training
engines and evaluated on the last recorded cycle of each of the 100 test engines.

| Evaluation | RMSE (cycles) | NASA score per engine |
|------------|---------------|-----------------------|
| Cross-validation (mean ± std across folds) | 11.97 ± 0.54 | 2.92 ± 0.50 |
| Test, labels clipped at 125 | 13.22 | 2.93 |
| Test, raw labels | 14.49 | 3.25 |

Total test NASA score is 293 with clipped labels and 325 with raw labels.
Cross-validated RMSE on rows with RUL under 50 is 6.87 ± 0.46.

LightGBM settings (depth 5, learning rate 0.03, 300 trees) were chosen from a
27-setting grid. They beat the previous settings on all five re-check shuffles,
but the gain is small, about 0.1 cycles of cross-validated RMSE.

Before adding the trend features, the same model scored a test RMSE of 17.08 and
a total NASA score of 855. Test scores were viewed while diagnosing an earlier
window-size comparison, so they are not a fully untouched estimate. Feature and
settings choices were made from cross-validation.

## License

[MIT](LICENSE)

