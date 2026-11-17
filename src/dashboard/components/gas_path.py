"""The project pipeline drawn as a turbofan's gas path, in HTML and SVG.

Air moves through an engine in five stages, and this project's data moves
through five matching ones. The diagram is a side cut of the engine: each stage
is a band of the flow path, with a note underneath. Hovering or focusing a stage
lights up both the band and its note, using CSS only.

st.html removes SVG for safety, so the diagram is shown through a small custom
component whose script only inserts the markup built here.
"""

from dataclasses import dataclass
from html import escape

import streamlit as st

from src.dashboard.palette import BEZEL, BLACK, CYAN, MUTED, PANEL, WHITE

# Drawing size; the flow path is symmetric about the horizontal centerline
WIDTH, HEIGHT = 1000, 220
CENTER = HEIGHT / 2

# Half-height of the flow path's outer and inner walls at each x position,
# from the wide intake, narrowing through the compressor, to the nozzle
OUTER = [(0, 92), (200, 70), (400, 38), (600, 36), (800, 52), (1000, 44)]
INNER = [(0, 0), (60, 30), (200, 26), (400, 16), (600, 16), (800, 18), (1000, 6)]


@dataclass(frozen=True)
class Stage:
    """One engine stage and the project step it stands for."""

    engine_part: str
    title: str
    note: str


def _half_height(profile: list[tuple[float, float]], x: float) -> float:
    """Linearly interpolate a wall's half-height at x."""
    for (x0, h0), (x1, h1) in zip(profile, profile[1:], strict=False):
        if x0 <= x <= x1:
            return h0 + (h1 - h0) * (x - x0) / (x1 - x0)
    raise ValueError(f"x={x} is outside the drawing")


def _band(x0: float, x1: float, steps: int = 8) -> str:
    """Return SVG path data for the flow path between x0 and x1, both halves."""
    xs = [x0 + (x1 - x0) * i / steps for i in range(steps + 1)]
    paths = []
    for sign in (-1, 1):
        outer = [(x, CENTER + sign * _half_height(OUTER, x)) for x in xs]
        inner = [(x, CENTER + sign * _half_height(INNER, x)) for x in reversed(xs)]
        points = " L ".join(f"{x:.1f} {y:.1f}" for x, y in outer + inner)
        paths.append(f"M {points} Z")
    return " ".join(paths)


def _blades(x0: float, x1: float, count: int) -> str:
    """Return SVG lines for evenly spaced blade rows across a stage."""
    lines = []
    for i in range(count):
        x = x0 + (x1 - x0) * (i + 0.5) / count
        for sign in (-1, 1):
            y_outer = CENTER + sign * (_half_height(OUTER, x) - 3)
            y_inner = CENTER + sign * (_half_height(INNER, x) + 3)
            lines.append(
                f'<line x1="{x:.1f}" y1="{y_inner:.1f}" '
                f'x2="{x:.1f}" y2="{y_outer:.1f}"/>'
            )
    return "".join(lines)


def _combustor_cans(x0: float, x1: float) -> str:
    """Return rounded SVG chambers where fuel burns, one in each half."""
    mid = (x0 + x1) / 2
    shapes = []
    for sign in (-1, 1):
        half = (_half_height(OUTER, mid) + _half_height(INNER, mid)) / 2
        y = CENTER + sign * half - 6
        shapes.append(
            f'<rect class="can" x="{x0 + 30:.1f}" y="{y:.1f}" '
            f'width="{x1 - x0 - 60:.1f}" height="12" rx="6"/>'
        )
    return "".join(shapes)


# Blade rows per stage: one big fan, many compressor stages, none in the
# combustor, a few turbine stages, none in the nozzle
_BLADE_ROWS = (1, 7, 0, 3, 0)

_CSS = f"""
.gas-path {{ font-family: var(--st-font, sans-serif); color: {WHITE}; }}
.gas-path svg {{ display: block; width: 100%; height: auto; }}
.gas-path .band {{ fill: {PANEL}; stroke: {BEZEL}; stroke-width: 1.5; }}
.gas-path .blades {{ stroke: {MUTED}; stroke-width: 2; opacity: 0.55; }}
.gas-path .can {{ fill: none; stroke: {MUTED}; stroke-width: 1.5; opacity: 0.8; }}
.gas-path .centerline {{ stroke: {BEZEL}; stroke-dasharray: 6 6; }}
.gas-path .flow {{ stroke: {MUTED}; stroke-width: 1.5; fill: none; opacity: 0.5; }}
.gas-path .notes {{
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 12px;
}}
.gas-path .note {{
  padding: 12px; border-top: 2px solid {BEZEL}; outline: none; cursor: default;
}}
.gas-path .note .part {{ color: {MUTED}; font-size: 0.85rem; }}
.gas-path .note h4 {{
  font-family: var(--st-heading-font, sans-serif); font-size: 1rem;
  margin: 2px 0 6px; padding: 0; color: {WHITE};
}}
.gas-path .note p {{ margin: 0; font-size: 0.9rem; line-height: 1.45; color: {MUTED}; }}
.gas-path .note:hover, .gas-path .note:focus-visible {{ border-top-color: {CYAN}; }}
.gas-path .note:hover h4, .gas-path .note:focus-visible h4 {{ color: {CYAN}; }}
@media (max-width: 720px) {{
  .gas-path .notes {{ grid-template-columns: 1fr; }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .gas-path .band, .gas-path .note {{ transition: all 150ms ease-out; }}
}}
"""


def _link_rules(n_stages: int) -> str:
    """Return CSS that lights up a band and its note together.

    Hovering or focusing note i lights band i, and hovering band i lights
    note i. The :has() selector lets the container react to either one.
    """
    rules = []
    for i in range(n_stages):
        lit_band = f"fill: {CYAN}; fill-opacity: 0.22; stroke: {CYAN};"
        rules += [
            f".gas-path:has(.note-{i}:hover) .band-{i},"
            f".gas-path:has(.note-{i}:focus-visible) .band-{i},"
            f".gas-path .band-{i}:hover {{ {lit_band} }}",
            f".gas-path:has(.band-{i}:hover) .note-{i} {{ border-top-color: {CYAN}; }}",
            f".gas-path:has(.band-{i}:hover) .note-{i} h4 {{ color: {CYAN}; }}",
        ]
    return "\n".join(rules)


def gas_path_html(stages: list[Stage]) -> str:
    """Build the gas-path diagram and its stage notes as one HTML string.

    Args:
        stages: Exactly five stages, from intake to exhaust.

    Returns:
        str: The SVG drawing and the notes, styled by the component's CSS.
    """
    if len(stages) != len(_BLADE_ROWS):
        raise ValueError(f"Expected {len(_BLADE_ROWS)} stages, got {len(stages)}")

    stage_width = WIDTH / len(stages)
    bands, blades = [], []
    for i, rows in enumerate(_BLADE_ROWS):
        x0, x1 = i * stage_width, (i + 1) * stage_width
        bands.append(f'<path class="band band-{i}" d="{_band(x0, x1)}"/>')
        blades.append(_blades(x0, x1, rows))
    blades.append(_combustor_cans(2 * stage_width, 3 * stage_width))

    # Streamlines through the middle of the flow path, top and bottom
    xs = range(0, WIDTH + 1, 25)
    flow = []
    for sign in (-1, 1):
        middle = [(_half_height(OUTER, x) + _half_height(INNER, x)) / 2 for x in xs]
        points = " L ".join(
            f"{x} {CENTER + sign * h:.1f}" for x, h in zip(xs, middle, strict=True)
        )
        flow.append(f'<path class="flow" d="M {points}"/>')

    notes = "".join(
        f'<div class="note note-{i}" tabindex="0">'
        f'<div class="part">{i + 1}. {escape(stage.engine_part)}</div>'
        f"<h4>{escape(stage.title)}</h4>"
        f"<p>{escape(stage.note)}</p></div>"
        for i, stage in enumerate(stages)
    )

    return (
        '<div class="gas-path">'
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-label="The project pipeline drawn as the five stages of a turbofan '
        'engine, from intake to exhaust">'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BLACK}"/>'
        f'<line class="centerline" x1="0" y1="{CENTER}" x2="{WIDTH}" y2="{CENTER}"/>'
        f"{''.join(bands)}"
        f'<g class="blades">{"".join(blades)}</g>'
        f"{''.join(flow)}"
        "</svg>"
        f'<div class="notes">{notes}</div>'
        "</div>"
    )


_GAS_PATH = st.components.v2.component(
    "gas_path_diagram",
    html='<div class="root"></div>',
    css=_CSS + _link_rules(len(_BLADE_ROWS)),
    js="""
export default function ({ data, parentElement }) {
  parentElement.querySelector(".root").innerHTML = data.markup;
}
""",
)


def gas_path(stages: list[Stage], key: str = "gas_path") -> None:
    """Show the gas-path diagram for the given five stages."""
    _GAS_PATH(data={"markup": gas_path_html(stages)}, key=key)
