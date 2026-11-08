# Predictive Maintenance — Turbofan Engine RUL Prediction

Predict the **Remaining Useful Life (RUL)** of turbofan engines using NASA's
[C-MAPSS dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data).

## Overview

This project builds a complete ML pipeline — from raw sensor data to a
production-ready API with monitoring — as a portfolio piece demonstrating
end-to-end machine learning engineering.

| Stage | Status |
|-------|--------|
| Data loading & exploration | ✅ Completed |
| Data preprocessing (RUL & cleaning) | ✅ Completed |
| Feature engineering | ✅ Completed |
| Baseline model (XGBoost / LightGBM) | ✅ Completed |
| Experiment tracking (MLflow) | 🔄 Next |
| Model serving (FastAPI) | ⬜ Not started |
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
├── data/                   # Raw & processed data (gitignored)
│   ├── raw/
│   └── processed/
├── notebooks/              # Exploration notebooks
├── src/                    # Source code
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training & selection
│   ├── evaluation/         # Metrics & evaluation utilities
│   └── api/                # FastAPI serving
├── tests/                  # Unit & integration tests
├── .github/workflows/      # CI/CD
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## License

MIT

