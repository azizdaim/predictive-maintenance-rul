"""About: how the project is built, what the experiments showed, and its limits."""

import httpx
import streamlit as st

from src.dashboard.charts import section_importance_chart
from src.dashboard.components.gas_path import Stage, gas_path
from src.dashboard.data import get_ablation_results, get_model_info

# Written by the project author; the section is hidden while this is empty
KEY_FINDINGS = ""

REPO_URL = "https://github.com/azizdaim/predictive-maintenance-rul"

try:
    metrics = get_model_info()["metrics"]
    model_scores = (
        f"Cross-validated error is {metrics['cv_rmse_mean']:.1f} cycles; "
        f"on NASA's test engines it is {metrics['test_rmse']:.1f}."
    )
except (httpx.HTTPError, KeyError):
    model_scores = "Scores appear here while the prediction API is running."

STAGES = [
    Stage(
        "Intake",
        "Sensor data",
        "NASA's simulated turbofan fleet: 100 engines run to failure, with 21 "
        "sensors and 3 settings recorded every flight cycle.",
    ),
    Stage(
        "Compressor",
        "Feature engineering",
        "The 15 sensors that change are squeezed into 30 features: short "
        "rolling averages and 30-cycle trends, each looking only backwards.",
    ),
    Stage(
        "Combustor",
        "Model",
        "A LightGBM model learns remaining life, tested on engines it never "
        f"trained on. {model_scores}",
    ),
    Stage(
        "Turbine",
        "Prediction API",
        "A FastAPI service checks each request, rebuilds the same features, and "
        "returns a prediction for the latest cycle or every cycle.",
    ),
    Stage(
        "Exhaust",
        "This dashboard",
        "Streamlit pages that ask the API for every number you see, including the "
        "live replay, which sends one request per step.",
    ),
]

st.title("How it works", icon=":material/info:")
# Prose stays near 80 characters a line, which is easier to read than full width
PROSE_WIDTH = 720

st.markdown(
    width=PROSE_WIDTH,
    body="Air moves through a turbofan in five stages, and data moves through this "
    "project in five matching ones. Hover a stage to follow it.",
)

gas_path(STAGES)

st.space("medium")
st.header("What the experiments showed")

findings, chart = st.columns([1, 1], gap="large")
with findings:
    st.markdown(
        "**Trends carry the model.** Without the 30-cycle trend features, error on "
        "engines close to failure rises from 6.9 to 17.2 cycles. A sensor's "
        "direction says more than its current value.\n\n"
        "**Half the features do the same job.** Raw readings, rolling spreads, and "
        "operating settings were removed together, and accuracy did not drop. The "
        "served model uses the remaining 30 features out of 62.\n\n"
        "**Warnings come early enough to act on.** In the live replay, engine 31 is "
        "flagged for watching with 44 cycles left and for replacement with 24."
    )
with chart:
    st.markdown("**Extra error when one engine section's sensors are removed**")
    st.echarts_chart(
        section_importance_chart(get_ablation_results()),
        height=260,
        alt="Bar chart of the error increase from removing each engine section",
    )
    st.caption(
        "Cross-validated RMSE increase in cycles, averaged over 3 shuffles. The "
        "high-pressure compressor matters most, though it also has the most sensors."
    )

if KEY_FINDINGS:
    st.header("In my own words")
    st.markdown(KEY_FINDINGS)

st.header("Limitations")
st.markdown(
    width=PROSE_WIDTH,
    body="- **One simulated scenario.** The model is trained and tested on FD001: one "
    "operating condition and one fault mode from NASA's engine simulator, not "
    "real flight data. The other three C-MAPSS scenarios are untested.\n"
    "- **A slow start.** Trend features need more than 30 cycles of history, so "
    "predictions for very young engines are less reliable, and the app says so.\n"
    "- **A test set looked at once too often.** Test scores were viewed while "
    "diagnosing an earlier experiment, so they are not a fully untouched estimate. "
    "Every modeling choice was made from cross-validation.",
)

st.link_button("View the source on GitHub", REPO_URL, icon=":material/code:")
