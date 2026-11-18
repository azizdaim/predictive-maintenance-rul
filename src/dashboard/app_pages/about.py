"""About: how the project is built, what the experiments showed, and its limits."""

import httpx
import pandas as pd
import streamlit as st

from src.dashboard.charts import section_importance_chart
from src.dashboard.client import (
    DEFAULT_REPLAY_SPEED,
    REPLAY_SPEEDS,
    REPO_URL,
    cycles_text,
    first_alerts,
)
from src.dashboard.components.gas_path import Stage, gas_path
from src.dashboard.data import (
    get_ablation_results,
    get_model_info,
    get_predictions,
    get_true_rul,
)

# Written by the project author; the section is hidden while this is empty
KEY_FINDINGS = ""


# Computed from the same replay the Live simulation page runs, so the numbers
# here always match what a visitor sees there
DEMO_ENGINE = 31
try:
    truth = get_true_rul()
    alerts = first_alerts(
        get_predictions(DEMO_ENGINE),
        truth[truth["engine_id"] == DEMO_ENGINE],
        REPLAY_SPEEDS[DEFAULT_REPLAY_SPEED],
    )
    watch, replace = alerts["Watch"], alerts["Replace soon"]
    early_warning = (
        "**Warnings come early enough to act on.** In the live replay at the "
        f"default speed, engine {DEMO_ENGINE} is flagged for watching with "
        f"{cycles_text(watch[1])} actually left and for replacement with "
        f"{cycles_text(replace[1])}."
    )
except (httpx.HTTPError, KeyError):
    early_warning = ""

metrics = None
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

st.title("How it works", anchor=False)
# Prose stays near 70 characters a line, which is easier to read than full width
PROSE_WIDTH = 560

st.markdown(
    width=PROSE_WIDTH,
    body="Air moves through a turbofan in five stages, and data moves through this "
    "project in five matching ones. Hover or tab to a stage to follow it. "
    "Remaining life is counted in cycles, and one cycle is one flight, from "
    "take-off to landing.",
)

gas_path(STAGES)

st.space("medium")
st.header("What the experiments showed", anchor=False)

if metrics:
    st.table(
        pd.DataFrame(
            {
                "Measured on": [
                    "NASA's 100 test engines",
                    "Training engines, cross-validated",
                    "Engines close to failure",
                ],
                "Typical error": [
                    f"{metrics['test_rmse']:.1f} cycles",
                    f"{metrics['cv_rmse_mean']:.1f} ± {metrics['cv_rmse_std']:.1f} "
                    "cycles",
                    f"{metrics['cv_near_failure_rmse_mean']:.1f} cycles",
                ],
                "What it means": [
                    "How far off the model is on engines it never saw",
                    "The same, averaged over 5 held-out groups of training engines",
                    "Accuracy when fewer than 50 cycles remain, where it matters most",
                ],
            }
        ),
        hide_index=True,
        alt="Model error in cycles on test engines, in cross-validation, and near "
        "failure",
    )

findings, chart = st.columns([1, 1], gap="large")
with findings:
    st.markdown(
        "**Trends carry the model.** Without the 30-cycle trend features, error on "
        "engines close to failure rises from 6.9 to 17.2 cycles. A sensor's "
        "direction says more than its current value.\n\n"
        "**Half the features do the same job.** Raw readings, rolling spreads, and "
        "operating settings were removed together, and accuracy did not drop. The "
        "served model uses the remaining 30 features out of 62.\n\n"
        f"{early_warning}"
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
    st.header("In my own words", anchor=False)
    st.markdown(KEY_FINDINGS)

st.header("Limitations", anchor=False)
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
