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
from src.dashboard.palette import (  # noqa: E402
    AMBER,
    BLACK,
    CYAN,
    GREEN,
    RED,
    WHITE,
)

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
h1, h2 {{ text-wrap: balance; }}
.stMarkdown p, .stMarkdown li {{ text-wrap: pretty; }}
.status-pill {{
  display: inline-block; padding: 2px 10px; border-radius: 6px;
  font: 700 0.85rem var(--st-font); border: 1px solid transparent;
}}
.status-healthy {{ color: {GREEN}; padding-left: 0; padding-right: 0; }}
.status-watch {{ color: {AMBER}; border-color: {AMBER}; }}
.status-replace {{ color: {BLACK}; background: {RED}; }}
.st-key-count_replace [data-testid="stMetricValue"] {{ color: {RED}; }}
.st-key-count_watch [data-testid="stMetricValue"] {{ color: {AMBER}; }}
.st-key-sim_progress [role="progressbar"] div div {{ background: {WHITE}; }}
@media (max-width: 640px) {{
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
}}
</style>"""
)

# Navigation icons are drawn from icon-font ligatures, so screen readers read
# their names aloud ("dashboard Fleet overview"); mark them decorative. Nav
# links have no key, so their test ids are the only hooks.
st.html(
    """<script>
if (!window.__navIconsMuted) {
  window.__navIconsMuted = true;
  const mute = () => document
    .querySelectorAll('[data-testid="stTopNavLink"] [data-testid="stIconMaterial"]')
    .forEach((icon) => icon.setAttribute("aria-hidden", "true"));
  mute();
  new MutationObserver(mute).observe(document.body, { childList: true, subtree: true });
}
</script>""",
    unsafe_allow_javascript=True,
)

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
            "app_pages/drift.py",
            title="Drift check",
            icon=":material/compare_arrows:",
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
