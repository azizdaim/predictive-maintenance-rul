"""ECharts option builders for the dashboard.

Each function returns a plain dict in the format ECharts expects, so the chart
design can be tested without a browser. Pages pass the dicts to
st.echarts_chart, which animates between values when the data changes.
"""

import math

import pandas as pd

from src.dashboard.client import (
    REPLACE_SOON_BELOW,
    RUL_CAP,
    STATUS_HEX,
    WATCH_BELOW,
    health_status,
)
from src.dashboard.palette import (
    AMBER,
    BEZEL,
    BLACK,
    GREEN_DIM,
    MUTED,
    RED,
    STEEL,
    TRACK,
    WHITE,
)
from src.monitoring.drift import DRIFTED_FROM, WATCH_FROM

_MUTED = MUTED
_GRID = BEZEL


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

    # A cockpit dial: a muted track with thin amber and red limit marks, and
    # only the needle and value in the current status colour
    def mark(limit: int, color: str) -> list:
        return [[(limit - 1) / RUL_CAP, TRACK], [(limit + 1) / RUL_CAP, color]]

    bands = [
        *mark(REPLACE_SOON_BELOW, STATUS_HEX["Replace soon"]),
        *mark(WATCH_BELOW, STATUS_HEX["Watch"]),
        [1, TRACK],
    ]
    status_color = STATUS_HEX[health_status(predicted_rul)[0]]
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
                    "itemStyle": {"color": status_color},
                },
                "anchor": {
                    "show": True,
                    "size": 12,
                    "itemStyle": {"color": WHITE},
                },
                "axisTick": {
                    "distance": -14,
                    "length": 5,
                    "lineStyle": {"color": BLACK},
                },
                "splitLine": {
                    "distance": -14,
                    "length": 14,
                    "lineStyle": {"color": BLACK, "width": 2},
                },
                "splitNumber": 5,
                "axisLabel": {"distance": 20, "color": _MUTED, "fontSize": 11},
                "title": {"offsetCenter": [0, "72%"], "color": _MUTED, "fontSize": 13},
                "detail": {
                    "offsetCenter": [0, "40%"],
                    "formatter": "{value}",
                    "color": status_color,
                    "fontSize": 34,
                    "fontWeight": 600,
                },
                "data": [{"value": round(predicted_rul), "name": "cycles left"}],
                # Show the value at once when the gauge first appears, instead of
                # sweeping up from zero; animate only when the value changes
                "animationDuration": 0,
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
        columns={"predicted_rul": "Predicted", "true_rul": "Actual"}
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
            {
                # Palette colours instead of ECharts' default blue-grey
                "type": "slider",
                "height": 18,
                "bottom": 8,
                "borderColor": BEZEL,
                "backgroundColor": BLACK,
                "fillerColor": "rgba(242, 242, 242, 0.08)",
                "dataBackground": {
                    "lineStyle": {"color": MUTED, "opacity": 0.5},
                    "areaStyle": {"color": MUTED, "opacity": 0.15},
                },
                "selectedDataBackground": {
                    "lineStyle": {"color": WHITE, "opacity": 0.6},
                    "areaStyle": {"color": WHITE, "opacity": 0.15},
                },
                "handleStyle": {"color": MUTED, "borderColor": BEZEL},
                "moveHandleStyle": {"color": MUTED},
                "textStyle": {"color": MUTED},
            },
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
                    "lineStyle": {"color": WHITE, "type": "dashed"},
                    # Right-aligned so it stays inside the chart at the last cycle
                    "label": {
                        "formatter": f"Cycle {cycle}",
                        "color": WHITE,
                        "align": "right",
                    },
                    "data": [{"xAxis": cycle}],
                },
            },
            {
                "name": "Actual",
                "type": "line",
                "encode": {"x": "cycle", "y": "Actual"},
                "showSymbol": False,
                "lineStyle": {"width": 2, "type": "dashed"},
            },
        ],
    }


def section_importance_chart(ablation: pd.DataFrame) -> dict:
    """Build a bar chart of how much worse the model gets without each section.

    Args:
        ablation: Ablation results with 'study', 'variant', and 'rmse_change'.

    Returns:
        dict: ECharts option with one horizontal bar per engine section, the
            most important at the top.
    """
    sections = ablation[ablation["study"] == "Engine section"].copy()
    sections["section"] = (
        sections["variant"].str.removeprefix("Without ").str.capitalize()
    )
    sections["rmse_change"] = sections["rmse_change"].round(2)
    sections = sections.sort_values("rmse_change")[["section", "rmse_change"]]
    return {
        "dataset": {"source": sections},
        "grid": {"left": 8, "right": 48, "top": 8, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "valueFormatter": "+%.2f cycles"},
        "xAxis": {"type": "value", "show": False},
        "yAxis": {
            "type": "category",
            "axisTick": {"show": False},
            "axisLine": {"show": False},
            "axisLabel": {"color": WHITE},
        },
        "series": [
            {
                "type": "bar",
                "encode": {"x": "rmse_change", "y": "section"},
                # Measurements, not predictions, so not magenta
                "itemStyle": {"color": WHITE, "opacity": 0.7},
                "barWidth": "55%",
                "label": {
                    "show": True,
                    "position": "right",
                    "formatter": "+{@rmse_change}",
                    "color": MUTED,
                },
            }
        ],
    }


# Stable stays quiet, like a healthy engine; drift is loud
DRIFT_HEX = {"Stable": GREEN_DIM, "Watch": AMBER, "Drifted": RED}


def sensor_label(sensor: str, section: str) -> str:
    """Name a sensor for people: "Sensor 11 (high-pressure compressor)"."""
    return f"Sensor {sensor.removeprefix('sensor_')} ({section.lower()})"


def drift_chart(sensors: pd.DataFrame) -> dict:
    """Build a bar chart of each sensor's PSI against the drift thresholds.

    Args:
        sensors: Drift report rows with 'sensor', 'section', 'psi', 'status'.

    Returns:
        dict: ECharts option with one horizontal bar per sensor, the most
            drifted at the top, and the watch and drift limits marked.
    """
    rows = sensors.iloc[::-1]
    # Never shorter than 0.5, so small scores look small; round the end up to
    # a tenth on a short scale and a whole number on a long one
    top = max(0.5, float(sensors["psi"].max()) * 1.15)
    step = 0.1 if top <= 1 else 1
    x_max = round(math.ceil(top / step) * step, 1)

    def limit(value: float, name: str, color: str) -> dict:
        return {
            "xAxis": value,
            "lineStyle": {"color": color, "type": "dashed"},
            "label": {"formatter": name, "color": color, "position": "end"},
        }

    if x_max <= 1:
        marks = {
            "markLine": {
                "silent": True,
                "symbol": "none",
                "data": [
                    limit(WATCH_FROM, "Watch", AMBER),
                    limit(DRIFTED_FROM, "Drifted", RED),
                ],
            }
        }
    else:
        # On a long scale the two limits nearly touch, so one shaded zone
        # marks everything below the drift limit instead
        marks = {
            "markArea": {
                "silent": True,
                "itemStyle": {"color": GREEN_DIM, "opacity": 0.18},
                "label": {"position": "top", "color": _MUTED, "align": "left"},
                "data": [
                    [{"name": "Below drift limit", "xAxis": 0}, {"xAxis": DRIFTED_FROM}]
                ],
            }
        }

    return {
        "grid": {"left": 8, "right": 48, "top": 28, "bottom": 28, "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": x_max,
            "name": "Drift score (PSI)",
            "nameLocation": "middle",
            "nameGap": 26,
            "nameTextStyle": {"color": _MUTED},
            "splitLine": {"lineStyle": {"color": _GRID}},
            "axisLabel": {"color": _MUTED},
        },
        "yAxis": {
            "type": "category",
            # Short names keep the bars wide on phones; the dropdown gives sections
            "data": [
                f"Sensor {r.sensor.removeprefix('sensor_')}" for r in rows.itertuples()
            ],
            "axisTick": {"show": False},
            "axisLine": {"show": False},
            "axisLabel": {"color": WHITE},
        },
        "series": [
            {
                "type": "bar",
                "silent": True,
                "barWidth": "60%",
                "data": [
                    {
                        "value": round(r.psi, 2),
                        "itemStyle": {"color": DRIFT_HEX[r.status]},
                        # Always two decimals, so 0.40 lines up with 0.47
                        "label": {"formatter": f"{r.psi:.2f}"},
                    }
                    for r in rows.itertuples()
                ],
                # A black pad keeps the value readable where it meets a limit line
                "label": {
                    "show": True,
                    "position": "right",
                    "color": _MUTED,
                    "backgroundColor": BLACK,
                    "padding": [0, 3],
                },
                **marks,
            }
        ],
    }


def distribution_chart(reference_shares: list[float], new_shares: list[float]) -> dict:
    """Build side-by-side bars of where training and new readings fall.

    Args:
        reference_shares: Share of training readings in each bucket.
        new_shares: Share of the new fleet's readings in the same buckets.

    Returns:
        dict: ECharts option with one pair of bars per bucket, in percent.
    """
    buckets = [str(i) for i in range(1, len(reference_shares) + 1)]

    def percent(values: list[float]) -> list[float]:
        return [round(v * 100, 1) for v in values]

    return {
        "legend": {"top": 0, "right": 0, "textStyle": {"color": _MUTED}},
        "grid": {"left": 48, "right": 16, "top": 36, "bottom": 34},
        "xAxis": {
            "type": "category",
            "data": buckets,
            "name": "Training range, lowest to highest",
            "nameLocation": "middle",
            "nameGap": 24,
        },
        "yAxis": {
            "type": "value",
            "name": "Share of readings (%)",
            "min": 0,
            "splitLine": {"lineStyle": {"color": _GRID}},
        },
        "series": [
            {
                "name": "Training engines",
                "type": "bar",
                "silent": True,
                "data": percent(reference_shares),
                "itemStyle": {"color": STEEL},
            },
            {
                "name": "This fleet",
                "type": "bar",
                "silent": True,
                "data": percent(new_shares),
                "itemStyle": {"color": WHITE},
            },
        ],
    }
