"""Streamlit dashboard for turbofan RUL predictions."""

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from src.dashboard.client import (
    API_URL,
    REPLACE_SOON_BELOW,
    WATCH_BELOW,
    fetch_history_predictions,
    health_status,
    load_test_engines,
    true_rul_by_cycle,
)
from src.features.build_features import TREND_LAG

st.set_page_config(
    page_title="Turbofan engine health", page_icon=":material/flight:", layout="wide"
)


@st.cache_data
def get_test_engines() -> pd.DataFrame:
    """Load the test engines once per session."""
    return load_test_engines()


@st.cache_data
def get_true_rul() -> pd.DataFrame:
    """Compute true RUL for every test engine once."""
    return true_rul_by_cycle(get_test_engines())


@st.cache_data(ttl="10m", show_spinner="Asking the prediction API...")
def get_predictions(engine_id: int) -> pd.DataFrame:
    """Fetch predictions for one engine, reused while the slider moves."""
    engines = get_test_engines()
    return fetch_history_predictions(engines[engines["engine_id"] == engine_id])


st.title("Turbofan engine health")
st.caption(
    "Predicted remaining useful life (RUL) for the 100 NASA C-MAPSS FD001 test "
    "engines, served by this project's FastAPI model."
)

engines = get_test_engines()

with st.sidebar:
    st.header("Engine")
    engine_id = st.selectbox(
        "Test engine",
        options=list(range(1, 101)),
        index=30,
        format_func=lambda n: f"Engine {n}",
    )
    last_cycle = int(engines.loc[engines["engine_id"] == engine_id, "cycle"].max())
    cycle = st.slider(
        "Cycle",
        min_value=1,
        max_value=last_cycle,
        value=last_cycle,
        help="Move back in time to see what the model predicted at each cycle.",
    )

try:
    predictions = get_predictions(engine_id)
except httpx.HTTPError:
    st.error(
        f"The dashboard gets its predictions from the FastAPI service at {API_URL}. "
        "Start it in another terminal with `uvicorn src.api.main:app`, then refresh.",
        title="Can't reach the prediction API",
        icon=":material/cloud_off:",
    )
    st.stop()

true_rul = get_true_rul()
engine = predictions.merge(
    true_rul[true_rul["engine_id"] == engine_id], on="cycle", how="left"
)
current = engine[engine["cycle"] == cycle].iloc[0]
predicted = float(current["predicted_rul"])
actual = float(current["true_rul"])
label, color = health_status(predicted)

with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(f"Engine {engine_id} at cycle {cycle}")
    st.badge(label, color=color)

with st.container(horizontal=True):
    st.metric("Predicted RUL", f"{predicted:.0f} cycles", border=True)
    st.metric("True RUL", f"{actual:.0f} cycles", border=True)
    st.metric(
        "Prediction error",
        f"{predicted - actual:+.0f} cycles",
        border=True,
        help="Positive means the model predicted more life than the engine had, "
        "which is the riskier mistake.",
    )

if cycle <= TREND_LAG:
    st.warning(
        f"Trend features need more than {TREND_LAG} cycles of history, so "
        "predictions this early in an engine's life are less reliable.",
        title="Short history",
        icon=":material/info:",
    )

chart_data = engine.melt(
    id_vars="cycle",
    value_vars=["predicted_rul", "true_rul"],
    var_name="series",
    value_name="rul",
)
chart_data["series"] = chart_data["series"].map(
    {"predicted_rul": "Predicted", "true_rul": "True"}
)

lines = (
    alt.Chart(chart_data)
    .mark_line()
    .encode(
        x=alt.X("cycle:Q", title="Cycle"),
        y=alt.Y("rul:Q", title="Remaining useful life (cycles)"),
        color=alt.Color("series:N", title=None),
        strokeDash=alt.StrokeDash("series:N", legend=None),
        tooltip=[
            alt.Tooltip("cycle:Q", title="Cycle"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("rul:Q", title="RUL", format=".1f"),
        ],
    )
)
selected = (
    alt.Chart(pd.DataFrame({"cycle": [cycle]}))
    .mark_rule(strokeDash=[4, 4], color="gray")
    .encode(x="cycle:Q")
)

with st.container(border=True):
    st.markdown("**Predicted vs true remaining life**")
    st.altair_chart(
        lines + selected,
        alt=f"Predicted and true RUL across engine {engine_id}'s recorded life",
    )
    st.caption(
        f"True RUL comes from NASA's answer file and is capped at 125, like the "
        f"model's training target. Health bands: replace soon below "
        f"{REPLACE_SOON_BELOW} cycles, watch below {WATCH_BELOW}."
    )
