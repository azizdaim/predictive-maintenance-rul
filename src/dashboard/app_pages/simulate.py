"""Live simulation: replay a test engine cycle by cycle through the prediction API."""

import httpx
import pandas as pd
import streamlit as st

from src.dashboard.charts import rul_gauge, rul_history_chart
from src.dashboard.client import (
    DEFAULT_REPLAY_SPEED,
    ENGINE_TINT,
    REPLAY_SPEEDS,
    RUL_CAP,
    STATUS_COLORS,
    cycles_text,
    fetch_history_predictions,
    health_status,
)
from src.dashboard.components.engine3d import engine_3d
from src.dashboard.components.status import status_pill
from src.dashboard.data import get_test_engines, get_true_rul, show_api_error
from src.dashboard.palette import MUTED
from src.features.build_features import TREND_LAG

# Seconds between steps, and how many cycles each step reveals at each speed
TICK_SECONDS = 0.4
SPEEDS = REPLAY_SPEEDS

# Most urgent last, so a higher index means a more serious status
SEVERITY = ["Healthy", "Watch", "Replace soon"]


def reset() -> None:
    """Rewind the simulation to before the engine's first cycle."""
    st.session_state.sim_cycle = 0
    st.session_state.sim_playing = False
    st.session_state.sim_predictions = pd.DataFrame(columns=["cycle", "predicted_rul"])
    st.session_state.sim_worst = 0
    st.session_state.sim_events = []


def toggle_play(last_cycle: int) -> None:
    """Start or pause, rewinding first if the replay already finished."""
    if st.session_state.sim_cycle >= last_cycle:
        reset()
    st.session_state.sim_playing = not st.session_state.sim_playing


def advance(rows: pd.DataFrame, last_cycle: int) -> None:
    """Reveal the next cycles and ask the API for predictions up to them.

    Only readings up to the new cycle are sent, so the model never sees the
    engine's future, just as it would not for a live engine.
    """
    step = SPEEDS[st.session_state.sim_speed]
    new_cycle = min(st.session_state.sim_cycle + step, last_cycle)
    try:
        predictions = fetch_history_predictions(rows[rows["cycle"] <= new_cycle])
    except httpx.HTTPError:
        st.session_state.sim_playing = False
        show_api_error()
    st.session_state.sim_cycle = new_cycle
    st.session_state.sim_predictions = predictions
    truth = get_true_rul()
    actual_left = float(
        truth.loc[
            (truth["engine_id"] == st.session_state.sim_engine)
            & (truth["cycle"] == new_cycle),
            "true_rul",
        ].iloc[0]
    )

    # Raise an alert the first time the engine reaches a more serious status
    label = health_status(float(predictions["predicted_rul"].iloc[-1]))[0]
    if SEVERITY.index(label) > st.session_state.sim_worst:
        st.session_state.sim_worst = SEVERITY.index(label)
        # Keep how much life was really left: the alert's whole point
        st.session_state.sim_events.append((new_cycle, label, actual_left))
        st.toast(
            f"Engine {st.session_state.sim_engine} entered **{label.lower()}** "
            f"at cycle {new_cycle}",
            icon=":material/warning:",
        )


def replay_outcome(engine: int) -> str:
    """Sum up how early the replay warned, from its recorded alerts."""
    left = {label: flights for _, label, flights in st.session_state.sim_events}
    parts = []
    if "Replace soon" in left:
        flights = cycles_text(left["Replace soon"], unit="flight")
        parts.append(f"for replacement {flights} before failure")
    if "Watch" in left:
        parts.append(f"for watching {cycles_text(left['Watch'], unit='flight')} before")
    return f"Engine {engine} was flagged " + ", and ".join(parts) + "."


st.title("Live simulation", anchor=False)
st.caption(
    "Replay a test engine as if it were reporting in live. Each step sends the "
    "readings so far to the prediction API, so the model only ever sees the past."
)

if "sim_cycle" not in st.session_state:
    reset()
    # A link ending in ?autoplay=1 starts the replay on arrival, for demos
    st.session_state.sim_playing = st.query_params.get("autoplay") == "1"

engines = get_test_engines()

with st.container(horizontal=True, vertical_alignment="bottom"):
    engine_id = st.selectbox(
        "Test engine",
        options=list(range(1, 101)),
        index=30,
        format_func=lambda n: f"Engine {n}",
        key="sim_engine",
        on_change=reset,
        width=220,
    )
    st.segmented_control(
        "Speed",
        options=list(SPEEDS),
        default=DEFAULT_REPLAY_SPEED,
        key="sim_speed",
        required=True,
    )
    rows = engines[engines["engine_id"] == engine_id]
    last_cycle = int(rows["cycle"].max())
    playing = st.session_state.sim_playing
    finished = st.session_state.sim_cycle >= last_cycle
    st.button(
        "Pause" if playing else "Replay" if finished else "Play",
        icon=":material/pause:"
        if playing
        else ":material/replay:"
        if finished
        else ":material/play_arrow:",
        type="primary",
        key="sim_play",
        on_click=toggle_play,
        args=(last_cycle,),
    )
    st.button("Reset", icon=":material/replay:", key="sim_reset", on_click=reset)


@st.fragment(run_every=TICK_SECONDS if playing else None)
def live_view() -> None:
    """Advance one step while playing, then draw the engine's current state."""
    if st.session_state.sim_playing:
        advance(rows, last_cycle)
        if st.session_state.sim_cycle >= last_cycle:
            # Stop the timer by rerunning the whole page with playing off
            st.session_state.sim_playing = False
            st.rerun()

    cycle = st.session_state.sim_cycle
    predictions = st.session_state.sim_predictions

    if cycle == 0:
        label, predicted = None, None
    else:
        predicted = float(predictions["predicted_rul"].iloc[-1])
        label = health_status(predicted)[0]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.header(
            f"Engine {engine_id}, cycle {cycle} of {last_cycle}",
            anchor=False,
            width="content",
        )
        if label:
            status_pill(label)
    # A measurement, not a control, so the bar is restyled from cyan to white
    with st.container(key="sim_progress"):
        st.progress(cycle / last_cycle)
    if 0 < cycle <= TREND_LAG:
        st.warning(
            f"Trend features need more than {TREND_LAG} cycles of history, so "
            "these first predictions are less reliable.",
            title="Short history",
            icon=":material/info:",
        )

    # Readings first: on the left here, and first when columns stack on phones
    readout, view = st.columns([1, 1])
    with view:
        engine_3d(
            # Grey before the first reading: no data yet, so no status colour
            status_color=ENGINE_TINT[label] if label else MUTED,
            description=(
                f"3D engine {engine_id}, shown as {label.lower()}, "
                f"{cycles_text(predicted)} left"
                if label
                else f"3D engine {engine_id}, waiting for the replay to start"
            ),
            fan_speed=0.8 - 0.6 * (1 - predicted / RUL_CAP) if predicted else 0.2,
            glow=0.25 + 0.75 * (1 - predicted / RUL_CAP) if predicted else 0.2,
            height=320,
            key="sim_engine_3d",
        )
    with readout:
        with st.container(border=True):
            st.markdown("**Predicted remaining life**")
            if predicted is None:
                st.caption("Press play to start sending readings.")
            else:
                st.echarts_chart(
                    rul_gauge(predicted),
                    height=230,
                    key="sim_gauge",
                    alt=f"Gauge showing {cycles_text(predicted)} predicted left",
                )
        with st.container(border=True):
            st.markdown("**Alerts**")
            if not st.session_state.sim_events:
                st.caption("No alerts yet.")
            for event_cycle, event_label, left in st.session_state.sim_events:
                st.markdown(
                    f":{STATUS_COLORS[event_label]}[:material/warning:] "
                    f"Cycle {event_cycle}: {event_label.lower()}, "
                    f"{cycles_text(left, unit='flight')} before failure"
                )
            if cycle >= last_cycle and st.session_state.sim_events:
                st.markdown(f"**{replay_outcome(engine_id)}**")

    if cycle > 0:
        true_rul = get_true_rul()
        history = predictions.merge(
            true_rul[true_rul["engine_id"] == engine_id], on="cycle", how="left"
        )
        with st.container(border=True):
            st.markdown("**Predictions so far**")
            st.echarts_chart(
                rul_history_chart(history, cycle, last_cycle=last_cycle),
                height=320,
                key="sim_history",
                alt=f"Predicted and actual remaining life for engine {engine_id} "
                f"to cycle {cycle}",
            )


live_view()
