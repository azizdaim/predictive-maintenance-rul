"""Fleet health map: one clickable tile per engine, colored by health band.

Built as a custom component rather than a chart so every click reaches Python
as a separate event. With a chart, a quick second click deselects the tile and
the two clicks merge into "nothing selected"; here a double click is simply two
clicks on the same engine. Tiles are real buttons, so they also work with the
keyboard (Tab to a tile, Enter to open it).
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.client import REPLACE_SOON_BELOW, STATUS_HEX, WATCH_BELOW

# Legend order and labels, most urgent first
BANDS = [
    ("Replace soon", f"Replace soon (<{REPLACE_SOON_BELOW})"),
    ("Watch", "Watch"),
    ("Healthy", f"Healthy ({WATCH_BELOW}+)"),
]

_ASSETS = Path(__file__).parent

_FLEET_MAP = st.components.v2.component(
    "fleet_tile_map",
    html=(
        '<div class="fleet-map"><div class="legend"></div>'
        '<div class="grid"></div></div>'
    ),
    css=(_ASSETS / "fleet_map.css").read_text(),
    js=(_ASSETS / "fleet_map.js").read_text(),
)


def fleet_tiles(fleet: pd.DataFrame, by_urgency: bool = False) -> list[dict]:
    """Prepare one tile per engine, in engine or urgency order.

    Args:
        fleet: One row per engine with 'engine_id', 'status', and 'predicted_rul'.
        by_urgency: Place the lowest predicted RUL first instead of engine order.

    Returns:
        list[dict]: Tiles with the engine number, rounded RUL, status, and color.
    """
    order = "predicted_rul" if by_urgency else "engine_id"
    return [
        {
            "engine": int(row.engine_id),
            "rul": round(row.predicted_rul),
            "status": row.status,
            "color": STATUS_HEX[row.status],
        }
        for row in fleet.sort_values(order).itertuples()
    ]


def fleet_map(
    tiles: list[dict],
    on_click: Callable[[], None],
    *,
    columns: int = 10,
    key: str = "fleet_map",
) -> None:
    """Show the tile map and call on_click whenever a tile is clicked.

    Inside on_click, st.session_state[key].clicked holds the engine number.

    Args:
        tiles: Tiles from fleet_tiles.
        on_click: Callback run before the page reruns, once per click.
        columns: Tiles per row.
        key: Streamlit key, which also names the clicked value in Session State.
    """
    _FLEET_MAP(
        data={
            "tiles": tiles,
            "columns": columns,
            "bands": [
                {"status": status, "label": label, "color": STATUS_HEX[status]}
                for status, label in BANDS
            ],
        },
        key=key,
        on_clicked_change=on_click,
    )
