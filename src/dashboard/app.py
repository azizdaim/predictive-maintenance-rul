"""Streamlit dashboard entry point: page setup and navigation."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts this file's folder on the import path, not the project root.
# Adding the root makes `src` importable on any host, without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.embedded_api import (  # noqa: E402
    embedded_api_enabled,
    start_embedded_api,
)

st.set_page_config(
    page_title="Turbofan engine health", page_icon=":material/flight:", layout="wide"
)

if embedded_api_enabled():
    start_embedded_api()

# Shared between pages: the fleet table sets it, the engine page reads it
if "engine_id" not in st.session_state:
    st.session_state.engine_id = 31

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="Home",
            icon=":material/home:",
            default=True,
            help="What the project does, an interactive 3D engine, and the model's "
            "headline results.",
        ),
        st.Page(
            "app_pages/fleet.py",
            title="Fleet overview",
            icon=":material/dashboard:",
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
        st.Page(
            "app_pages/simulate.py",
            title="Live simulation",
            icon=":material/play_circle:",
            help="Replay an engine cycle by cycle through the prediction API and "
            "watch the alerts fire as it wears out.",
        ),
        st.Page(
            "app_pages/about.py",
            title="How it works",
            icon=":material/info:",
            help="The project pipeline drawn as an engine's gas path, what the "
            "experiments showed, and the model's limits.",
        ),
    ],
    position="top",
)

page.run()
