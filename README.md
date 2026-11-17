# Predictive Maintenance — Turbofan Engine RUL Prediction

Predict the **Remaining Useful Life (RUL)** of turbofan engines using NASA's
[C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

**Live dashboard:** [azizdaim-turbofan.streamlit.app](https://azizdaim-turbofan.streamlit.app)
(after a quiet spell it sleeps, and the first visit needs a wake-up click)

![Live simulation: engine 31 replayed cycle by cycle, turning amber at cycle 145 and red at cycle 180 as the predicted remaining life falls](docs/demo.gif)

## Overview

This portfolio project is building an end-to-end ML system, from raw sensor
data to a served model with monitoring and a dashboard. The data pipeline,
model, evaluation, experiment tracking, prediction API, dashboard, and Docker
image are complete. Drift monitoring and CI are next, as shown below.

| Stage | Status |
|-------|--------|
| Data loading & exploration | ✅ Completed |
| Data preprocessing (RUL & cleaning) | ✅ Completed |
| Feature engineering (rolling and trend features) | ✅ Completed |
| Model (LightGBM) with engine-grouped cross-validation | ✅ Completed |
| Experiment tracking (MLflow) | ✅ Completed |
| Model serving (FastAPI) | ✅ Completed |
| Dashboard (Streamlit) | ✅ Completed |
| Drift monitoring (Evidently) | ⬜ Not started |
| CI/CD (GitHub Actions) | ⬜ Not started |
| Containerization (Docker) | ✅ Completed |

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
│   ├── models/             # Training, cross-validation, ablation, MLflow logging
│   ├── evaluation/         # RMSE and NASA scoring
│   ├── api/                # FastAPI prediction service
│   └── dashboard/          # Streamlit dashboard, charts, and 3D engine component
├── tests/                  # Unit tests (pytest)
├── scripts/                # Starts the API and dashboard in the container
├── Dockerfile              # Trains the model, then runs the API and dashboard
├── pyproject.toml          # Ruff and pytest configuration
├── requirements.txt        # Everything for development
└── requirements-app.txt    # Only what the running app needs, pinned
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

# Serve predictions from the trained model, then open http://127.0.0.1:8000/docs
uvicorn src.api.main:app --reload

# In a second terminal, start the dashboard (it gets predictions from the API)
streamlit run src/dashboard/app.py
```

The dashboard has five pages: a landing page with an interactive 3D engine, a
fleet overview with a clickable health map of all 100 test engines, an engine
detail page, a live simulation that replays an engine cycle by cycle through the
API, and a "How it works" page that draws the pipeline as an engine's gas path.
Its look follows an aircraft engine-indication display: green, amber, and red
for health, magenta for predictions, and Airbus's B612 cockpit typeface. Add
`?autoplay=1` to the simulation page's address to start the replay on arrival.

## Run with Docker

The image downloads NASA's data, trains the model, and runs the API and the
dashboard together. No local data or Python setup is needed.

```bash
docker build -t turbofan-rul .
docker run -p 7860:7860 turbofan-rul   # then open http://localhost:7860
```

The hosted demo runs on Streamlit Community Cloud, which runs a single Streamlit
app. There, `RUL_EMBEDDED_API=1` starts the FastAPI service in a background
thread of the dashboard's process, and the dashboard still calls it over HTTP.
The served model and the two test data files it reads are committed for this.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Reports that the service is running |
| `/predict` | POST | Predicts RUL at the latest cycle from an engine's raw sensor history |

`/predict` takes a list of cycles, oldest first, each with the cycle number,
3 settings, and 21 sensors in the C-MAPSS format. It reuses the training feature
pipeline and returns the predicted RUL, capped at 125. Histories of 30 cycles or
fewer get a warning, because the trend features need a longer history.

## Results (FD001)

LightGBM with 30 features: 5-cycle rolling averages and 30-cycle trends for each
of the 15 active sensors, a set chosen by the ablation study below. Settings are scored
with 5-fold cross-validation grouped by engine, using random cut-off cycles
averaged over 50 draws. The final model is then trained on all 100 training
engines and evaluated on the last recorded cycle of each of the 100 test engines.

| Evaluation | RMSE (cycles) | NASA score per engine |
|------------|---------------|-----------------------|
| Cross-validation (mean ± std across folds) | 11.90 ± 0.56 | 2.86 ± 0.42 |
| Test, labels clipped at 125 | 13.07 | 2.85 |
| Test, raw labels | 14.34 | 3.18 |

Total test NASA score is 285 with clipped labels and 318 with raw labels.
Cross-validated RMSE on rows with RUL under 50 is 6.81 ± 0.42.

LightGBM settings (depth 5, learning rate 0.03, 300 trees) were chosen from a
27-setting grid. They beat the previous settings on all five re-check shuffles,
but the gain is small, about 0.1 cycles of cross-validated RMSE.

Before adding the trend features, the same model scored a test RMSE of 17.08 and
a total NASA score of 855. Test scores were viewed while diagnosing an earlier
window-size comparison, so they are not a fully untouched estimate. Feature and
settings choices were made from cross-validation.

## Ablation Study

To test which inputs the model actually relies on, each group of features was
removed in turn and the model was re-scored with the same engine-grouped 5-fold
cross-validation, repeated over 3 shuffles. A positive change means the model got
worse without that group. Run it with `python -m src.models.ablation`; full
results are in [reports/ablation_results.csv](reports/ablation_results.csv).

**By feature type** (all 62 candidate features: RMSE 12.10, near-failure RMSE 6.92)

| Removed | Features | RMSE change | Near-failure change | Worse in |
|---------|----------|-------------|---------------------|----------|
| 30-cycle trends | 15 | **+6.62** | **+10.24** | 3 of 3 |
| Rolling averages | 15 | +0.19 | +0.75 | 3 of 3 |
| Operating settings | 2 | +0.00 | +0.00 | 2 of 3 |
| Raw sensor readings | 15 | −0.01 | −0.02 | 1 of 3 |
| Rolling spreads | 15 | −0.06 | −0.05 | 0 of 3 |

**By engine section** (all features built from that section's sensors)

| Removed | Sensors | RMSE change | Near-failure change | Worse in |
|---------|---------|-------------|---------------------|----------|
| High-pressure compressor | 3, 7, 9, 11, 14 | **+2.04** | **+2.46** | 3 of 3 |
| Fan | 6, 8, 13, 15 | +0.63 | +0.78 | 3 of 3 |
| Combustor | 12, 17 | +0.38 | +0.73 | 3 of 3 |
| Low-pressure turbine | 4, 21 | +0.32 | +0.41 | 3 of 3 |
| Low-pressure compressor | 2 | +0.15 | +0.15 | 3 of 3 |
| High-pressure turbine | 20 | +0.09 | +0.07 | 3 of 3 |

Findings:

- The 30-cycle trends carry the model. Without them, near-failure error rises from
  6.9 to 17.2 cycles, which confirms the earlier feature-engineering result with a
  controlled experiment.
- Raw readings, rolling spreads, and operating settings can each be removed
  without loss. FD001 has a single operating condition, so the settings carry no
  signal. Because overlapping features can cover for each other, all three were
  also removed together: dropping those 32 features changed RMSE by −0.05 and
  near-failure RMSE by −0.07, worse in 0 of 3 shuffles. A 30-feature model of
  rolling averages and trends performs as well as the full 62, so the served
  model now uses those 30.
- Every engine section matters, consistently across all shuffles. High-pressure
  compressor sensors matter most, though that group also has the most sensors.

## License

[MIT](LICENSE)

