"""Tests for the ECharts option builders used by the engine page."""

import pandas as pd

from src.dashboard.charts import (
    rul_gauge,
    rul_history_chart,
    section_importance_chart,
)
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


def test_section_chart_sorts_engine_sections_by_importance() -> None:
    """Verify only engine sections are charted, the most important last."""
    ablation = pd.DataFrame(
        {
            "study": ["Baseline", "Engine section", "Engine section", "Feature type"],
            "variant": ["All", "Without fan", "Without combustor", "Without x"],
            "rmse_change": [0.0, 0.634, 0.381, 6.6],
        }
    )

    source = section_importance_chart(ablation)["dataset"]["source"]

    assert source["section"].tolist() == ["Combustor", "Fan"]
    assert source["rmse_change"].tolist() == [0.38, 0.63]
