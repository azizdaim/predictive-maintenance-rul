"""Engine detail: one engine's health in 3D, a life gauge, and its full history."""

import httpx
import streamlit as st

from src.dashboard.charts import rul_gauge, rul_history_chart
from src.dashboard.client import (
    ENGINE_TINT,
    REPLACE_SOON_BELOW,
    RUL_CAP,
    WATCH_BELOW,
    cycles_text,
    health_status,
)
from src.dashboard.components.engine3d import engine_3d
from src.dashboard.components.status import status_pill
from src.dashboard.data import (
    get_fleet,
    get_predictions,
    get_test_engines,
    get_true_rul,
    show_api_error,
)
from src.features.build_features import TREND_LAG

st.title("Engine detail", anchor=False)
st.caption(
    "Pick an engine and move back through its history to see how its health "
    "and the model's prediction changed, cycle by cycle."
)

# Discard a fleet click recorded after the page had already switched here, so it
# cannot send the visitor back to an engine the next time they open the fleet
st.session_state.pop("open_engine", None)

engines = get_test_engines()

# Each option carries its engine's status and remaining life, so an urgent
# engine can be picked without visiting the fleet page first
try:
    fleet = get_fleet().set_index("engine_id")
except httpx.HTTPError:
    show_api_error()


def describe_engine(n: int) -> str:
    """Label a dropdown option with the engine's latest status and RUL."""
    row = fleet.loc[n]
    return f"Engine {n}: {row['status']}, {cycles_text(row['predicted_rul'])} left"


# The engine lives in the address (?engine=34) so a view can be shared. A new
# address value wins once, on arrival; after that the dropdown leads and the
# address follows it. Streamlit's bind="query-params" was tried first, but it
# ignores a value set from another page and never matches labels like these.
requested = st.query_params.get("engine", "")
if requested != st.session_state.get("engine_in_address"):
    st.session_state.engine_in_address = requested
    if requested.isdigit() and 1 <= int(requested) <= 100:
        st.session_state.engine_id = int(requested)
if "engine_id" not in st.session_state:
    st.session_state.engine_id = 31

picker, timeline = st.columns([1, 3], vertical_alignment="bottom")
with picker:
    engine_id = st.selectbox(
        "Test engine",
        options=list(range(1, 101)),
        format_func=describe_engine,
        key="engine_id",
        persist_state="session",
    )
st.query_params["engine"] = str(engine_id)
st.session_state.engine_in_address = str(engine_id)
last_cycle = int(engines.loc[engines["engine_id"] == engine_id, "cycle"].max())
with timeline:
    cycle = st.slider(
        "Cycle",
        min_value=1,
        max_value=last_cycle,
        value=last_cycle,
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
label, _ = health_status(predicted)

with st.container(horizontal=True, vertical_alignment="center"):
    st.header(f"Engine {engine_id} at cycle {cycle}", anchor=False, width="content")
    status_pill(label)

# Closer to failure: the core glows harder and the rotors slow down
wear = 1 - min(predicted, RUL_CAP) / RUL_CAP
view, readout = st.columns([3, 2])
with view:
    engine_3d(
        status_color=ENGINE_TINT[label],
        description=f"3D engine {engine_id}, shown as {label.lower()}, "
        f"{cycles_text(predicted)} left",
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
            alt=f"Gauge showing {cycles_text(predicted)} predicted left",
        )
        st.caption(
            f"The dial stops at {RUL_CAP}: the model counts any engine with "
            f"{RUL_CAP} or more cycles left as fully healthy."
        )
    with st.container(horizontal=True):
        st.metric("Actual remaining life", cycles_text(actual), border=True)
        st.metric(
            "Prediction error",
            cycles_text(predicted - actual, signed=True),
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
    st.markdown("**Predicted vs actual remaining life**")
    st.echarts_chart(
        rul_history_chart(engine, cycle),
        height=360,
        alt=f"Predicted and actual remaining life across engine {engine_id}'s "
        "recorded life",
    )
    st.caption(
        f"Actual remaining life comes from NASA's answer file and is capped at "
        f"{RUL_CAP}, like "
        f"the model's training target. Shaded bands: replace soon below "
        f"{REPLACE_SOON_BELOW} cycles, watch below {WATCH_BELOW}. Scroll or drag "
        "the slider to zoom."
    )
