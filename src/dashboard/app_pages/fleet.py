"""Fleet overview: every engine ranked by urgency."""

import httpx
import streamlit as st

from src.dashboard.charts import fleet_map
from src.dashboard.client import (
    REPLACE_SOON_BELOW,
    RUL_CAP,
    STATUS_COLORS,
    WATCH_BELOW,
)
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
    tile_map = fleet_map(fleet, by_urgency=arrangement == "Urgency")
    clicked = st.echarts_chart(
        tile_map,
        height=480,
        key="fleet_map",
        on_select="rerun",
        alt="Grid of the 100 test engines colored by health band",
    )
    st.caption(
        "Click a tile to open that engine. Click a band in the legend to hide it."
    )

tile_indices = [
    i for group in clicked.selection.selected for i in group["data_indices"]
]
if tile_indices:
    tiles = tile_map["dataset"]["source"]
    st.session_state.engine_id = int(tiles.loc[tile_indices[0], "engine_id"])
    st.switch_page("app_pages/engine.py")

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
    event = st.dataframe(
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
        on_select="rerun",
        # Single-cell selection: clicking anywhere in a row opens that engine,
        # without the checkbox column that row selection adds
        selection_mode="single-cell",
        key="fleet_table",
        alt="Test engines sorted from least to most predicted remaining life",
    )
    st.caption("Click an engine to open its detail page.")

# Each selected cell is a (row position, column name) pair; the row is enough
if event.selection.cells:
    row, _ = event.selection.cells[0]
    st.session_state.engine_id = int(fleet["engine_id"].iloc[row])
    st.switch_page("app_pages/engine.py")
