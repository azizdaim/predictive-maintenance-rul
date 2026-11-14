"""Interactive 3D cutaway turbofan, built in code with three.js.

The engine is generated from simple shapes, so there is no third-party model
to license. Its glow follows the health status passed in from Python, and
hovering a section shows which C-MAPSS sensors measure that part.
"""

from pathlib import Path

import streamlit as st

# Pinned three.js build, loaded by the browser from a public CDN
THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"

_HTML = """
<div class="engine-root">
  <div class="hint">Drag to rotate · hover a section</div>
  <div class="tooltip" hidden></div>
  <div class="chip"><span class="dot"></span><span class="chip-text"></span></div>
  <div class="fallback" hidden>3D view unavailable: three.js could not be loaded.</div>
</div>
"""


_ASSETS = Path(__file__).parent
_CSS = (_ASSETS / "engine3d.css").read_text()
_JS = (_ASSETS / "engine3d.js").read_text().replace("__THREE_URL__", THREE_URL)

_ENGINE_3D = st.components.v2.component(
    "turbofan_engine_3d", html=_HTML, css=_CSS, js=_JS
)


def engine_3d(
    *,
    status_color: str = "#22D3EE",
    label: str | None = None,
    fan_speed: float = 0.6,
    glow: float = 0.5,
    height: int = 440,
    key: str | None = None,
) -> None:
    """Show the interactive 3D turbofan.

    Args:
        status_color: Hex color for the intake halo, core light, and turbine tint.
        label: Short status text shown in the corner chip. Hidden if None.
        fan_speed: Rotor speed from 0 (idle) to 1 (fast).
        glow: Core light strength from 0 to 1.
        height: Height of the view in pixels.
        key: Optional Streamlit key, needed when showing more than one engine.
    """
    _ENGINE_3D(
        data={
            "status_color": status_color,
            "label": label,
            "fan_speed": fan_speed,
            "glow": glow,
            "height": height,
        },
        key=key,
        height=height,
    )
