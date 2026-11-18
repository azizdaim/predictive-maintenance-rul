"""Drift check: do new engines still look like the ones the model learned from?"""

import httpx
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.charts import distribution_chart, drift_chart, sensor_label
from src.dashboard.components.status import status_pill
from src.dashboard.data import get_drift, get_drift_results, show_api_error
from src.monitoring.drift import DRIFTED_FROM, EARLY_LIFE_CYCLES, FLEETS, WATCH_FROM

TRAINING_FLEET = "FD001"

# Why each drifted fleet differs from training, from NASA's dataset description
FLEET_CAUSES = {
    "FD003": "these engines can also wear out at the fan, a fault the model never "
    "saw in training",
    "FD002": "these engines fly in six operating conditions, while the model "
    "learned from one",
}
FLEET_NOTES = {
    "FD002": " Checking each operating condition separately would show how much "
    "drift remains once that is accounted for.",
}

# Prose stays near 70 characters a line, as on How it works
PROSE_WIDTH = 640


def verdict(fleet: str, sensors: pd.DataFrame, errors: pd.Series) -> str:
    """Say in one or two sentences what the drift result means for the model."""
    drifted = int((sensors["status"] == "Drifted").sum())
    if fleet == TRAINING_FLEET:
        return (
            "This is the control: engines from the scenario the model learned, so "
            "they should read like the training engines, and they do. Its "
            "predictions here can be trusted."
        )
    if drifted == 0:
        return (
            "These engines read like the training engines, so the model's "
            "predictions here can be trusted."
        )

    count = (
        f"All {len(sensors)} sensors"
        if drifted == len(sensors)
        else f"{drifted} of {len(sensors)} sensors"
    )
    top = sensors.iloc[0]
    ratio = errors[fleet] / errors[TRAINING_FLEET]
    return (
        f"{count} read differently, led by "
        f"{sensor_label(top['sensor'], top['section']).lower()}, because "
        f"{FLEET_CAUSES[fleet]}. The model's error is {ratio:.1f} times higher "
        "here, so its predictions on this fleet cannot be relied on."
        f"{FLEET_NOTES.get(fleet, '')}"
    )


st.title("Drift check", anchor=False)
st.caption(
    "A model only knows the engines it learned from. Before trusting its "
    "predictions on a new fleet, check that the new sensors read like the old "
    f"ones. Only each engine's first {EARLY_LIFE_CYCLES} cycles are compared, "
    "while it is still new, so normal wear is not mistaken for drift.",
    width=PROSE_WIDTH,
)

fleet = st.segmented_control(
    "Fleet to check",
    options=list(FLEETS),
    format_func=lambda code: f"{FLEETS[code]} ({code})",
    default=TRAINING_FLEET,
    required=True,
    label_visibility="collapsed",
    key="drift_fleet",
)

try:
    report = get_drift(fleet)
except httpx.HTTPError:
    show_api_error(title="Drift check unavailable")

sensors = pd.DataFrame(report["sensors"])
errors = get_drift_results()["rmse"]

status_pill(report["status"])
st.markdown(verdict(fleet, sensors, errors), width=PROSE_WIDTH)

# Same split as the charts below, so the card edges line up
drifted_card, error_card = st.columns([1, 1], gap="medium")
with drifted_card:
    st.metric(
        "Sensors drifted",
        f"{report['drifted_sensors']} of {len(sensors)}",
        border=True,
        height="stretch",
        help=f"Sensors with a drift score (PSI) of {DRIFTED_FROM} or more.",
    )
with error_card:
    st.metric(
        "Model error on this fleet",
        f"{errors[fleet]:.1f} cycles",
        delta=None
        if fleet == TRAINING_FLEET
        else f"{errors[fleet] - errors[TRAINING_FLEET]:+.1f} vs the training fleet",
        delta_color="inverse",
        border=True,
        height="stretch",
        help="RMSE at each test engine's last cycle, measured offline against "
        "NASA's answer file for this fleet.",
    )

top = sensors.iloc[0]
scores, spread = st.columns([1, 1], gap="medium")
with scores, st.container(border=True, height="stretch"):
    st.markdown("**How far each sensor moved**")
    st.echarts_chart(
        drift_chart(sensors),
        height=440,
        alt=f"Bar chart of drift scores: {report['drifted_sensors']} of "
        f"{len(sensors)} sensors drifted, and the most moved is "
        f"{sensor_label(top['sensor'], top['section'])} at {top['psi']:.2f}",
    )
    st.caption(
        f"Below {WATCH_FROM} is stable, {WATCH_FROM} to {DRIFTED_FROM} is worth "
        f"watching, and {DRIFTED_FROM} or more means the readings have drifted."
    )
    with st.expander("Scores as a table"):
        st.table(
            pd.DataFrame(
                {
                    "Sensor": sensors["sensor"].str.replace("sensor_", "Sensor "),
                    "Engine section": sensors["section"],
                    "Drift score": sensors["psi"].map("{:.2f}".format),
                    "Status": sensors["status"],
                }
            ),
            hide_index=True,
            alt="Drift score and status of each sensor, most drifted first",
        )

with spread, st.container(border=True, height="stretch"):
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
    busiest = int(np.argmax(row["new_shares"]))
    st.echarts_chart(
        distribution_chart(row["reference_shares"], row["new_shares"]),
        height=380,
        alt=f"Bar chart for {sensor_label(row['sensor'], row['section'])}: "
        f"{row['new_shares'][busiest]:.0%} of this fleet's readings fall in "
        f"training range {busiest + 1}, which holds "
        f"{row['reference_shares'][busiest]:.0%} of training readings",
    )
    st.caption(
        "Grey bars split the training readings into ranges holding about a "
        "tenth each, or fewer ranges for sensors with few distinct values. The "
        "more the white bars differ, the higher the drift score."
    )
