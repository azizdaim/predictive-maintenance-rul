"""Tests for the ECharts option builders used by the engine page."""

import pandas as pd

from src.dashboard.charts import fleet_map, rul_gauge, rul_history_chart
from src.dashboard.client import RUL_CAP, STATUS_HEX


def test_gauge_shows_rounded_prediction_with_band_colors() -> None:
    """Verify the gauge spans 0 to the cap and colors its bands by health."""
    series = rul_gauge(9.4)["series"][0]

    assert series["data"][0]["value"] == 9
    assert (series["min"], series["max"]) == (0, RUL_CAP)
    band_colors = [color for _, color in series["axisLine"]["lineStyle"]["color"]]
    assert band_colors == [
        STATUS_HEX["Replace soon"],
        STATUS_HEX["Watch"],
        STATUS_HEX["Healthy"],
    ]


def test_history_chart_reads_the_frame_and_marks_the_cycle() -> None:
    """Verify the chart uses the engine's rows and draws a line at the cycle."""
    engine = pd.DataFrame(
        {
            "cycle": [1, 2, 3],
            "predicted_rul": [100.0, 90.0, 80.0],
            "true_rul": [98.0, 97.0, 96.0],
        }
    )

    option = rul_history_chart(engine, cycle=2)

    source = option["dataset"]["source"]
    assert list(source.columns) == ["cycle", "Predicted", "True"]
    assert len(source) == 3
    assert option["series"][0]["markLine"]["data"] == [{"xAxis": 2}]
    assert (option["xAxis"]["min"], option["xAxis"]["max"]) == (1, 3)


def test_fleet_map_places_one_tile_per_engine() -> None:
    """Verify tiles fill the grid row by row, in engine or urgency order."""
    fleet = pd.DataFrame(
        {"engine_id": [1, 2, 3, 4, 5], "predicted_rul": [80.4, 12.0, 45.0, 100.0, 5.6]}
    )

    by_number = fleet_map(fleet, columns=2)["dataset"]["source"]
    assert by_number["engine_id"].tolist() == [1, 2, 3, 4, 5]
    assert by_number["row"].tolist() == ["0", "0", "1", "1", "2"]
    assert by_number["column"].tolist() == ["0", "1", "0", "1", "0"]
    assert by_number["predicted_rul"].tolist() == [80, 12, 45, 100, 6]

    by_urgency = fleet_map(fleet, by_urgency=True, columns=2)["dataset"]["source"]
    assert by_urgency["engine_id"].tolist() == [5, 2, 3, 1, 4]


def test_fleet_map_legend_colors_match_health_bands() -> None:
    """Verify the legend bands use the status colors, most urgent first."""
    fleet = pd.DataFrame({"engine_id": [1], "predicted_rul": [50.0]})

    pieces = fleet_map(fleet)["visualMap"]["pieces"]

    assert [p["color"] for p in pieces] == [
        STATUS_HEX["Replace soon"],
        STATUS_HEX["Watch"],
        STATUS_HEX["Healthy"],
    ]
