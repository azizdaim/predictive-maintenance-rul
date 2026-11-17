"""Cockpit-display colors shared by the theme, charts, and 3D engine.

Modeled on an aircraft's engine-indication display (EICAS): true black glass,
green for normal, amber for caution, red for warning, magenta for computed
targets, white for measured values, and cyan for things the pilot can select.
Keep these in sync with .streamlit/config.toml.
"""

BLACK = "#000000"  # display glass
PANEL = "#0D0F11"  # raised surfaces
BEZEL = "#2A2E33"  # borders and grid lines
WHITE = "#F2F2F2"  # measured values: true RUL, text
MUTED = "#9AA0A6"  # secondary text and axis labels

GREEN = "#3DDC84"  # normal
AMBER = "#FFB000"  # caution
RED = "#FF4040"  # warning
MAGENTA = "#FF4FD8"  # computed values: the model's predictions
CYAN = "#3FD0FF"  # selectable controls
