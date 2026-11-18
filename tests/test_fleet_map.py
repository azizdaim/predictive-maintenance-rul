"""Tests for the fleet health map's tile data."""

import pandas as pd

from src.dashboard.client import STATUS_HEX
from src.dashboard.components.fleet_map import BANDS, fleet_tiles

FLEET = pd.DataFrame(
    {
        "engine_id": [1, 2, 3, 4],
        "status": ["Healthy", "Replace soon", "Watch", "Healthy"],
        "predicted_rul": [80.4, 12.0, 45.0, 100.0],
    }
)


def test_tiles_follow_engine_order_with_rounded_life_and_status_color() -> None:
    """Verify one tile per engine, in engine order, colored by its band."""
    tiles = fleet_tiles(FLEET)

    assert [t["engine"] for t in tiles] == [1, 2, 3, 4]
    assert tiles[0] == {
        "engine": 1,
        "rul": 80,
        "status": "Healthy",
        "color": STATUS_HEX["Healthy"],
    }


def test_tiles_can_be_ordered_by_urgency() -> None:
    """Verify urgency order puts the lowest predicted remaining life first."""
    tiles = fleet_tiles(FLEET, by_urgency=True)

    assert [t["engine"] for t in tiles] == [2, 3, 1, 4]


def test_legend_lists_every_band_most_urgent_first() -> None:
    """Verify the legend covers each status the tiles can have."""
    assert [status for status, _ in BANDS] == ["Replace soon", "Watch", "Healthy"]
