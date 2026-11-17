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

# Health bands from most to least urgent, matching the order of their RUL ranges
SEVERITY_ORDER = ("Replace soon", "Watch", "Healthy")


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


def fleet_map(fleet: pd.DataFrame, by_urgency: bool = False, columns: int = 10) -> dict:
    """Build a grid of clickable tiles, one per engine, colored by health band.

    Args:
        fleet: One row per engine with 'engine_id' and 'predicted_rul'.
        by_urgency: Place the lowest predicted RUL first instead of engine order.
        columns: Tiles per row.

    Returns:
        dict: ECharts heatmap option. Row i of dataset.source is tile i, so a
            clicked tile's data index looks up its engine there.
    """
    order = "predicted_rul" if by_urgency else "engine_id"
    tiles = fleet.sort_values(order, ignore_index=True)[["engine_id", "predicted_rul"]]
    tiles["predicted_rul"] = tiles["predicted_rul"].round().astype(int)
    tiles.insert(0, "column", (tiles.index % columns).astype(str))
    tiles.insert(1, "row", (tiles.index // columns).astype(str))
    n_rows = -(-len(tiles) // columns)  # ceiling division

    pieces = [
        {"lt": REPLACE_SOON_BELOW, "label": f"Replace soon (<{REPLACE_SOON_BELOW})"},
        {"gte": REPLACE_SOON_BELOW, "lt": WATCH_BELOW, "label": "Watch"},
        {"gte": WATCH_BELOW, "label": f"Healthy ({WATCH_BELOW}+)"},
    ]
    for piece, status in zip(pieces, SEVERITY_ORDER, strict=True):
        piece["color"] = STATUS_HEX[status]

    return {
        "dataset": {"source": tiles},
        "tooltip": {
            "formatter": "Engine {@engine_id}<br/>{@predicted_rul} cycles left"
        },
        "grid": {"left": 4, "right": 4, "top": 44, "bottom": 4},
        "xAxis": {
            "type": "category",
            "data": [str(i) for i in range(columns)],
            "show": False,
        },
        "yAxis": {
            "type": "category",
            "data": [str(i) for i in range(n_rows)],
            "inverse": True,
            "show": False,
        },
        "visualMap": {
            "type": "piecewise",
            "dimension": 3,  # predicted_rul
            "pieces": pieces,
            "orient": "horizontal",
            "itemGap": 24,
            "top": 0,
            "left": "center",
            "textStyle": {"color": _MUTED},
        },
        "series": [
            {
                "type": "heatmap",
                "encode": {"x": "column", "y": "row", "value": "predicted_rul"},
                "selectedMode": "single",
                "label": {
                    "show": True,
                    "formatter": "{@engine_id}",
                    "color": "#0A0F1C",
                    "fontWeight": 600,
                },
                "itemStyle": {
                    "borderColor": "#0A0F1C",
                    "borderWidth": 3,
                    "borderRadius": 6,
                },
                "emphasis": {"itemStyle": {"borderColor": "#E2E8F0"}},
                "select": {"itemStyle": {"borderColor": "#E2E8F0"}},
            }
        ],
    }
