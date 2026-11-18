"""Tests for the ECharts option builders used by the engine page."""

import pandas as pd

from src.dashboard.charts import (
    rul_gauge,
    rul_history_chart,
    section_importance_chart,
)
from src.dashboard.client import RUL_CAP, STATUS_HEX
from src.dashboard.palette import TRACK


def test_gauge_is_a_cockpit_dial_with_limit_marks() -> None:
    """Verify a muted track with red and amber limit marks, needle in status colour."""
    series = rul_gauge(9.4)["series"][0]

    assert series["data"][0]["value"] == 9
    assert (series["min"], series["max"]) == (0, RUL_CAP)
    colors = [color for _, color in series["axisLine"]["lineStyle"]["color"]]
    assert colors == [
        TRACK,
        STATUS_HEX["Replace soon"],
        TRACK,
        STATUS_HEX["Watch"],
        TRACK,
    ]
    # 9 cycles left is "Replace soon", so the needle and value are red
    assert series["pointer"]["itemStyle"]["color"] == STATUS_HEX["Replace soon"]
    assert series["detail"]["color"] == STATUS_HEX["Replace soon"]


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
    assert list(source.columns) == ["cycle", "Predicted", "Actual"]
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


def test_drift_chart_puts_the_most_drifted_sensor_on_top() -> None:
    """Verify bars are coloured by status and the biggest PSI is drawn last."""
    from src.dashboard.charts import DRIFT_HEX, drift_chart

    sensors = pd.DataFrame(
        {
            "sensor": ["sensor_11", "sensor_4"],
            "section": ["High-pressure compressor", "Low-pressure turbine"],
            "psi": [0.4, 0.02],
            "status": ["Drifted", "Stable"],
        }
    )

    option = drift_chart(sensors)

    labels = option["yAxis"]["data"]
    bars = option["series"][0]["data"]
    assert labels[-1] == "Sensor 11"
    assert bars[-1]["itemStyle"]["color"] == DRIFT_HEX["Drifted"]
    assert bars[0]["itemStyle"]["color"] == DRIFT_HEX["Stable"]
    assert option["xAxis"]["max"] == 0.5
