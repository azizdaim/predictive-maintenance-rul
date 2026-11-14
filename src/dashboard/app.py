"""Streamlit dashboard entry point: page setup and navigation."""

import streamlit as st

st.set_page_config(
    page_title="Turbofan engine health", page_icon=":material/flight:", layout="wide"
)

# Shared between pages: the fleet table sets it, the engine page reads it
if "engine_id" not in st.session_state:
    st.session_state.engine_id = 31

page = st.navigation(
    [
        st.Page(
            "app_pages/fleet.py",
            title="Fleet overview",
            icon=":material/dashboard:",
            default=True,
            help="Every test engine ranked by predicted remaining life, with health "
            "labels and recent trends. Select a row to open that engine.",
        ),
        st.Page(
            "app_pages/engine.py",
            title="Engine detail",
            icon=":material/monitoring:",
            help="One engine's predicted and true remaining life across its recorded "
            "history. Move the cycle slider to go back in time.",
        ),
    ],
    position="top",
)

st.title(page.title, icon=page.icon)
st.caption(
    "Predicted remaining useful life (RUL) for the 100 NASA C-MAPSS FD001 test "
    "engines, served by this project's FastAPI model."
)

page.run()
