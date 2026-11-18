"""Drift check: do new engines still look like the ones the model learned from?"""

import httpx
import pandas as pd
import streamlit as st

from src.dashboard.charts import distribution_chart, drift_chart, sensor_label
from src.dashboard.components.status import status_pill
from src.dashboard.data import get_drift, get_drift_results, show_api_error
from src.monitoring.drift import DRIFTED_FROM, EARLY_LIFE_CYCLES, FLEETS, WATCH_FROM

TRAINING_FLEET = "FD001"

st.title("Drift check", anchor=False)
st.caption(
    "A model only knows the engines it learned from. Before trusting its "
    "predictions on a new fleet, check that the new sensors read like the old "
    f"ones. Only each engine's first {EARLY_LIFE_CYCLES} cycles are compared, "
    "while it is still new, so normal wear is not mistaken for drift."
)

fleet = st.segmented_control(
    "Fleet to check",
    options=list(FLEETS),
    format_func=lambda code: f"{FLEETS[code]} ({code})",
    default=TRAINING_FLEET,
    required=True,
    key="drift_fleet",
)

try:
    report = get_drift(fleet)
except httpx.HTTPError:
    show_api_error()

sensors = pd.DataFrame(report["sensors"])
errors = get_drift_results()["rmse"]

with st.container(horizontal=True, vertical_alignment="center"):
    st.header(FLEETS[fleet], anchor=False, width="content")
    status_pill(report["status"])

with st.container(horizontal=True):
    st.metric(
        "Sensors drifted",
        f"{report['drifted_sensors']} of {len(sensors)}",
        border=True,
        help=f"Sensors with a PSI of {DRIFTED_FROM} or more.",
    )
    st.metric(
        "Model error on this fleet",
        f"{errors[fleet]:.1f} cycles",
        delta=None
        if fleet == TRAINING_FLEET
        else f"{errors[fleet] - errors[TRAINING_FLEET]:+.1f} vs the training fleet",
        delta_color="inverse",
        border=True,
        help="RMSE at each test engine's last cycle, measured offline against "
        "NASA's answer file for this fleet.",
    )

scores, spread = st.columns([1, 1], gap="medium")
with scores, st.container(border=True):
    st.markdown("**How far each sensor moved**")
    st.echarts_chart(
        drift_chart(sensors),
        height=440,
        alt="Bar chart of each sensor's drift score against the watch and drift limits",
    )
    st.caption(
        f"Population Stability Index (PSI). Below {WATCH_FROM} is stable, "
        f"{WATCH_FROM} to {DRIFTED_FROM} is worth watching, and {DRIFTED_FROM} "
        "or more means the readings have drifted."
    )

with spread, st.container(border=True):
    st.markdown("**Where one sensor's readings land**")
    sensor = st.selectbox(
        "Sensor",
        options=sensors.index,
        format_func=lambda i: sensor_label(
            sensors.at[i, "sensor"], sensors.at[i, "section"]
        ),
        label_visibility="collapsed",
    )
    row = sensors.loc[sensor]
    st.echarts_chart(
        distribution_chart(row["reference_shares"], row["new_shares"]),
        height=320,
        alt="Bar chart comparing where training and new readings fall across "
        "the training range",
    )
    st.caption(
        "Grey bars split the training readings into ranges holding about a "
        "tenth each, or fewer ranges for sensors with few distinct values. The "
        "more the white bars differ, the higher the PSI."
    )
