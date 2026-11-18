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
from src.dashboard.palette import AMBER, BLACK, CYAN, GREEN, RED  # noqa: E402

st.set_page_config(
    page_title="Turbofan engine health", page_icon=":material/flight:", layout="wide"
)

if embedded_api_enabled():
    start_embedded_api()

# Shared styles that the theme settings cannot express:
# - no hover toolbar on charts and tables (it adds nothing and covers controls)
# - black text on the cyan primary buttons (white on cyan fails contrast)
# - a visible keyboard focus ring on the top navigation; Streamlit gives nav
#   links no key, so its test id is the only selector
# - text selection and table digits themed from the palette
# - status pills in the cockpit alarm style (see components/status.py)
st.html(
    f"""<style>
.stElementToolbar {{ display: none; }}
.st-key-open_fleet button, .st-key-sim_play button {{ color: {BLACK}; }}
.st-key-open_fleet button *, .st-key-sim_play button * {{ color: {BLACK}; }}
[data-testid="stTopNavLink"]:focus-visible {{
  outline: 2px solid {CYAN}; outline-offset: 2px;
}}
::selection {{ background: {CYAN}; color: {BLACK}; }}
table, [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
.status-pill {{
  display: inline-block; padding: 2px 10px; border-radius: 6px;
  font: 700 0.85rem var(--st-font); border: 1px solid transparent;
}}
.status-healthy {{ color: {GREEN}; padding-left: 0; padding-right: 0; }}
.status-watch {{ color: {AMBER}; border-color: {AMBER}; }}
.status-replace {{ color: {BLACK}; background: {RED}; }}
</style>"""
)

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
        ),
        st.Page(
            "app_pages/fleet.py",
            title="Fleet overview",
            icon=":material/dashboard:",
        ),
        st.Page(
            "app_pages/engine.py",
            title="Engine detail",
            icon=":material/monitoring:",
        ),
        st.Page(
            "app_pages/simulate.py",
            title="Live simulation",
            icon=":material/play_circle:",
        ),
        st.Page(
            "app_pages/about.py",
            title="How it works",
            icon=":material/info:",
        ),
    ],
    position="top",
)

page.run()
