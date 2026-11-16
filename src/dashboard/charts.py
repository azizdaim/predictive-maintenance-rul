"""ECharts option builders for the dashboard.

Each function returns a plain dict in the format ECharts expects, so the chart
design can be tested without a browser. Pages pass the dicts to
st.echarts_chart, which animates between values when the data changes.
"""

import pandas as pd

from src.dashboard.client import REPLACE_SOON_BELOW, RUL_CAP, STATUS_HEX, WATCH_BELOW

# Muted text and grid colors that match the dark theme in .streamlit/config.toml
_MUTED = "#94A3B8"
_GRID = "#1F2A40"


def _band(status: str, low: float, high: float, opacity: float) -> list[dict]:
    """Shade the RUL range of one health band in its status color, with a label."""
    color = STATUS_HEX[status]
    return [
        {
            "name": status,
            "yAxis": low,
            "itemStyle": {"color": color, "opacity": opacity},
            "label": {"position": "insideTopLeft", "color": color, "fontSize": 11},
        },
        {"yAxis": high},
    ]


def rul_gauge(predicted_rul: float) -> dict:
    """Build a speedometer-style gauge of predicted RUL, colored by health band.

    Args:
        predicted_rul: Predicted remaining useful life in cycles.

    Returns:
        dict: ECharts option with one gauge series from 0 to the RUL cap.
    """
    bands = [
        [REPLACE_SOON_BELOW / RUL_CAP, STATUS_HEX["Replace soon"]],
        [WATCH_BELOW / RUL_CAP, STATUS_HEX["Watch"]],
        [1, STATUS_HEX["Healthy"]],
    ]
    return {
        "series": [
            {
                "type": "gauge",
                "min": 0,
                "max": RUL_CAP,
                "startAngle": 210,
                "endAngle": -30,
                "radius": "95%",
                "center": ["50%", "58%"],
                "axisLine": {"lineStyle": {"width": 14, "color": bands}},
                "pointer": {
                    "length": "62%",
                    "width": 5,
                    "itemStyle": {"color": "auto"},
                },
                "anchor": {
                    "show": True,
                    "size": 12,
                    "itemStyle": {"color": "#E2E8F0"},
                },
                "axisTick": {
                    "distance": -14,
                    "length": 5,
                    "lineStyle": {"color": "#0A0F1C"},
                },
                "splitLine": {
                    "distance": -14,
                    "length": 14,
                    "lineStyle": {"color": "#0A0F1C", "width": 2},
                },
                "splitNumber": 5,
                "axisLabel": {"distance": 20, "color": _MUTED, "fontSize": 11},
                "title": {"offsetCenter": [0, "72%"], "color": _MUTED, "fontSize": 13},
                "detail": {
                    "offsetCenter": [0, "40%"],
                    "formatter": "{value}",
                    "color": "auto",
                    "fontSize": 34,
                    "fontWeight": 600,
                },
                "data": [{"value": round(predicted_rul), "name": "cycles left"}],
                "animationDurationUpdate": 600,
            }
        ]
    }


def rul_history_chart(
    engine: pd.DataFrame, cycle: int, last_cycle: int | None = None
) -> dict:
    """Build the predicted vs true RUL line chart for one engine.

    Shaded bands mark the health thresholds, a vertical line marks the selected
    cycle, and a slider under the chart zooms into part of the history.

    Args:
        engine: One engine's rows with 'cycle', 'predicted_rul', and 'true_rul'.
        cycle: The cycle currently selected on the page.
        last_cycle: Fixed end of the x-axis, so a growing history fills it from
            the left. Defaults to the last cycle in the frame.

    Returns:
        dict: ECharts option reading the frame through dataset.source.
    """
    source = engine[["cycle", "predicted_rul", "true_rul"]].rename(
        columns={"predicted_rul": "Predicted", "true_rul": "True"}
    )
    band_areas = [
        _band("Replace soon", 0, REPLACE_SOON_BELOW, opacity=0.16),
        _band("Watch", REPLACE_SOON_BELOW, WATCH_BELOW, opacity=0.12),
    ]
    return {
        "dataset": {"source": source},
        "legend": {"top": 0, "right": 0, "textStyle": {"color": _MUTED}},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.1f cycles"},
        "grid": {"left": 48, "right": 16, "top": 36, "bottom": 64},
        "xAxis": {
            "type": "value",
            "name": "Cycle",
            "nameLocation": "middle",
            "nameGap": 28,
            "min": int(source["cycle"].min()),
            "max": last_cycle or int(source["cycle"].max()),
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "name": "Remaining life (cycles)",
            "min": 0,
            "max": RUL_CAP,
            "interval": 25,
            "splitLine": {"lineStyle": {"color": _GRID}},
        },
        "dataZoom": [
            {"type": "inside"},
            {"type": "slider", "height": 18, "bottom": 8},
        ],
        "series": [
            {
                "name": "Predicted",
                "type": "line",
                "encode": {"x": "cycle", "y": "Predicted"},
                "showSymbol": False,
                "lineStyle": {"width": 2.5},
                "markArea": {"silent": True, "data": band_areas},
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"color": "#E2E8F0", "type": "dashed"},
                    "label": {"formatter": f"Cycle {cycle}", "color": "#E2E8F0"},
                    "data": [{"xAxis": cycle}],
                },
            },
            {
                "name": "True",
                "type": "line",
                "encode": {"x": "cycle", "y": "True"},
                "showSymbol": False,
                "lineStyle": {"width": 2, "type": "dashed"},
            },
        ],
    }
