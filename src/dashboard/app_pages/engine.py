"""Engine detail: one engine's health in 3D, a life gauge, and its full history."""

import httpx
import streamlit as st

from src.dashboard.charts import rul_gauge, rul_history_chart
from src.dashboard.client import (
    REPLACE_SOON_BELOW,
    RUL_CAP,
    STATUS_HEX,
    WATCH_BELOW,
    health_status,
)
from src.dashboard.components.engine3d import engine_3d
from src.dashboard.data import (
    get_predictions,
    get_test_engines,
    get_true_rul,
    show_api_error,
)
from src.features.build_features import TREND_LAG

st.title("Engine detail", icon=":material/monitoring:", anchor=False)
st.caption(
    "Pick an engine and move back through its history to see how its health "
    "and the model's prediction changed, cycle by cycle."
)

engines = get_test_engines()

picker, timeline = st.columns([1, 3], vertical_alignment="bottom")
with picker:
    engine_id = st.selectbox(
        "Test engine",
        options=list(range(1, 101)),
        format_func=lambda n: f"Engine {n}",
        key="engine_id",
        persist_state="session",
    )
last_cycle = int(engines.loc[engines["engine_id"] == engine_id, "cycle"].max())
with timeline:
    cycle = st.slider(
        "Cycle",
        min_value=1,
        max_value=last_cycle,
        value=last_cycle,
        help="Move back in time to see what the model predicted at each cycle.",
    )

try:
    with st.spinner("Asking the prediction API..."):
        predictions = get_predictions(engine_id)
except httpx.HTTPError:
    show_api_error()

true_rul = get_true_rul()
engine = predictions.merge(
    true_rul[true_rul["engine_id"] == engine_id], on="cycle", how="left"
)
current = engine[engine["cycle"] == cycle].iloc[0]
predicted = float(current["predicted_rul"])
actual = float(current["true_rul"])
label, color = health_status(predicted)

with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(f"Engine {engine_id} at cycle {cycle}", anchor=False)
    st.badge(label, color=color)

# Closer to failure: the core glows harder and the rotors slow down
wear = 1 - min(predicted, RUL_CAP) / RUL_CAP
view, readout = st.columns([3, 2])
with view:
    engine_3d(
        status_color=STATUS_HEX[label],
        fan_speed=0.8 - 0.6 * wear,
        glow=0.25 + 0.75 * wear,
        height=420,
        key="engine_detail_3d",
    )
with readout:
    with st.container(border=True):
        st.markdown("**Predicted remaining life**")
        st.echarts_chart(
            rul_gauge(predicted),
            height=230,
            alt=f"Gauge showing {predicted:.0f} predicted cycles left",
        )
    with st.container(horizontal=True):
        st.metric("True RUL", f"{actual:.0f} cycles", border=True)
        st.metric(
            "Prediction error",
            f"{predicted - actual:+.0f} cycles",
            border=True,
            help="Positive means the model predicted more life than the engine "
            "had, which is the riskier mistake.",
        )

if cycle <= TREND_LAG:
    st.warning(
        f"Trend features need more than {TREND_LAG} cycles of history, so "
        "predictions this early in an engine's life are less reliable.",
        title="Short history",
        icon=":material/info:",
    )

with st.container(border=True):
    st.markdown("**Predicted vs true remaining life**")
    st.echarts_chart(
        rul_history_chart(engine, cycle),
        height=360,
        alt=f"Predicted and true RUL across engine {engine_id}'s recorded life",
    )
    st.caption(
        f"True RUL comes from NASA's answer file and is capped at {RUL_CAP}, like "
        f"the model's training target. Shaded bands: replace soon below "
        f"{REPLACE_SOON_BELOW} cycles, watch below {WATCH_BELOW}. Scroll or drag "
        "the slider to zoom."
    )
