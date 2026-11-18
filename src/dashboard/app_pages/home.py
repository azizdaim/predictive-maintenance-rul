"""Landing page: what the project does, shown on the fleet's most urgent engine.

The headline and pitch render first; the hero engine, the fleet status line, and
the error figure fill in once the fleet is scored, so visitors never stare at an
empty page. Detailed model scores live on the How it works page.
"""

import httpx
import streamlit as st

from src.dashboard.client import ENGINE_TINT, REPO_URL, cycles_text
from src.dashboard.components.engine3d import engine_3d
from src.dashboard.data import api_unavailable_message, get_fleet, get_model_info
from src.dashboard.palette import MUTED

hero_text, hero_engine = st.columns([1, 1.15], gap="large", vertical_alignment="center")

with hero_text:
    st.title("Know how long an engine has left, before it fails", anchor=False)
    # Replaced with the measured error once the model's scores have loaded
    pitch = st.empty()
    pitch.markdown("Predicts how many flights an engine has left before it fails.")
    fleet_status = st.container()
    if st.button(
        "Open fleet overview",
        icon=":material/dashboard:",
        type="primary",
        key="open_fleet",
    ):
        st.switch_page("app_pages/fleet.py")

st.space("medium")
with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(
        "A student machine learning project, from raw NASA sensor readings to a "
        "tested model, a prediction API, and this dashboard."
    )
    if st.button("See how it works", icon=":material/info:"):
        st.switch_page("app_pages/about.py")
    st.link_button("View the source on GitHub", REPO_URL, icon=":material/code:")

# Slow work last: the text above is already visible while the fleet is scored
with hero_engine:
    try:
        with st.spinner("Scoring the fleet..."):
            info = get_model_info()
            fleet = get_fleet()
    except httpx.HTTPError:
        # Grey is the honest "no data" state: no warning colour, nothing to press
        engine_3d(
            status_color=MUTED,
            description="3D turbofan engine, shown in grey because no live "
            "predictions are available",
            fan_speed=0.2,
            glow=0.1,
            height=460,
        )
        with hero_text:
            st.info(api_unavailable_message(), icon=":material/cloud_off:")
        st.stop()

    urgent = fleet.iloc[0]  # the fleet table is sorted by predicted RUL, lowest first
    urgent_id, urgent_rul = int(urgent["engine_id"]), round(urgent["predicted_rul"])
    wear = 1 - min(urgent_rul, 125) / 125
    engine_3d(
        status_color=ENGINE_TINT[urgent["status"]],
        description=f"3D engine {urgent_id}, the most urgent in the fleet, shown as "
        f"{urgent['status'].lower()}, {cycles_text(urgent_rul)} left",
        fan_speed=0.8 - 0.6 * wear,
        glow=0.25 + 0.75 * wear,
        height=460,
        key="hero_engine",
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"Most urgent: engine {urgent_id}, {cycles_text(urgent_rul)} left")
        st.space("stretch")
        if st.button(f"Inspect engine {urgent_id}", icon=":material/monitoring:"):
            st.session_state.engine_id = urgent_id
            st.switch_page("app_pages/engine.py")

pitch.markdown(
    "Predicts how many flights an engine has left before it fails, typically "
    f"within about {info['metrics']['test_rmse']:.0f} flights on engines it has "
    "never seen. The dashboard counts flights as cycles: one cycle is one flight."
)
replace_soon = int((fleet["status"] == "Replace soon").sum())
with fleet_status:
    st.markdown(
        f":red[**{replace_soon}**] of {len(fleet)} monitored engines need "
        "replacing soon."
    )
