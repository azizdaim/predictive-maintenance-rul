"""Fleet overview: every engine ranked by urgency."""

import httpx
import streamlit as st

from src.dashboard.client import (
    REPLACE_SOON_BELOW,
    RUL_CAP,
    STATUS_COLORS,
    WATCH_BELOW,
)
from src.dashboard.components.fleet_map import fleet_map, fleet_tiles
from src.dashboard.data import get_fleet, show_api_error

BAND_HELP = {
    "Replace soon": f"Predicted RUL below {REPLACE_SOON_BELOW} cycles.",
    "Watch": f"Predicted RUL from {REPLACE_SOON_BELOW} to {WATCH_BELOW} cycles.",
    "Healthy": f"Predicted RUL of {WATCH_BELOW} cycles or more.",
}

st.title("Fleet overview", icon=":material/dashboard:", anchor=False)
st.caption(
    "All 100 NASA C-MAPSS FD001 test engines, ranked by predicted remaining life."
)

try:
    fleet = get_fleet()
except httpx.HTTPError:
    show_api_error()

# A click on a tile or a table row is recorded by a callback, which runs before
# the page reruns. The record is read here but only cleared by the engine page
# once it loads, so a second click that interrupts the switch cannot lose it.
if "open_engine" in st.session_state:
    st.session_state.engine_id = st.session_state.open_engine
    st.switch_page("app_pages/engine.py")

counts = fleet["status"].value_counts()
with st.container(horizontal=True):
    for label in STATUS_COLORS:
        st.metric(
            label, f"{counts.get(label, 0)} engines", border=True, help=BAND_HELP[label]
        )

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**Fleet health map**")
        st.space("stretch")
        arrangement = st.segmented_control(
            "Arrange by",
            options=["Engine number", "Urgency"],
            default="Engine number",
            required=True,
            label_visibility="collapsed",
            key="map_order",
        )

    def open_clicked_tile() -> None:
        """Remember the clicked tile's engine as soon as the click arrives.

        A second click can arrive after the page has already switched to the
        engine, when the map no longer exists; that late click is ignored.
        """
        clicked = getattr(st.session_state.get("fleet_map"), "clicked", None)
        if clicked is not None:
            st.session_state.open_engine = clicked

    fleet_map(
        fleet_tiles(fleet, by_urgency=arrangement == "Urgency"),
        on_click=open_clicked_tile,
    )
    st.caption(
        "Click a tile to open that engine. Click a band above the map to hide it."
    )

table = fleet.assign(status=fleet["status"].map(lambda s: [s]))

with st.container(border=True):
    st.markdown("**Engines by predicted remaining life**")
    # Lock the header: an invisible cover over the header row stops visitors
    # sorting, reordering, resizing, or hiding columns, which st.dataframe has
    # no setting for. Rows below still scroll and open an engine on click.
    st.html(
        "<style>.st-key-fleet_table .stDataFrame { position: relative; }"
        ".st-key-fleet_table .stDataFrame::after { content: ''; position: absolute;"
        " top: 0; left: 0; right: 0; height: 36px; z-index: 5; cursor: default; }"
        "</style>"
    )

    def open_clicked_row() -> None:
        """Remember the clicked row's engine; a late click is ignored."""
        table_state = st.session_state.get("fleet_table")
        cells = getattr(getattr(table_state, "selection", None), "cells", None)
        if cells:
            row, _ = cells[0]  # each cell is a (row position, column name) pair
            st.session_state.open_engine = int(fleet["engine_id"].iloc[row])

    st.dataframe(
        table,
        column_config={
            "engine_id": st.column_config.NumberColumn("Engine", format="%d"),
            "status": st.column_config.MultiselectColumn(
                "Status",
                options=list(STATUS_COLORS),
                color=list(STATUS_COLORS.values()),
            ),
            "predicted_rul": st.column_config.ProgressColumn(
                "Predicted RUL (cycles)",
                min_value=0,
                max_value=RUL_CAP,
                format="%.0f",
            ),
            "cycles": st.column_config.NumberColumn("Cycles recorded"),
            "trend": st.column_config.LineChartColumn(
                "Predicted RUL, last 50 cycles", y_min=0, y_max=RUL_CAP
            ),
        },
        hide_index=True,
        on_select=open_clicked_row,
        # Single-cell selection: clicking anywhere in a row opens that engine,
        # without the checkbox column that row selection adds
        selection_mode="single-cell",
        key="fleet_table",
        alt="Test engines sorted from least to most predicted remaining life",
    )
    st.caption("Click an engine to open its detail page.")
