"""Tests for the gas-path diagram on the How it works page."""

import pytest

from src.dashboard.components.gas_path import Stage, gas_path_html

STAGES = [Stage(f"Part {i}", f"Title {i}", f"Note {i}") for i in range(5)]


def test_diagram_has_one_band_and_note_per_stage() -> None:
    """Verify each stage gets a numbered note and a band in the drawing."""
    html = gas_path_html(STAGES)

    for i in range(5):
        assert f"band-{i}" in html
        assert f'class="note note-{i}" tabindex="0"' in html
        assert f"{i + 1}. Part {i}" in html


def test_diagram_escapes_stage_text() -> None:
    """Verify stage text is shown as text, never run as HTML."""
    stages = [*STAGES[:4], Stage("Exhaust", "<b>Bold</b>", "a & b")]

    html = gas_path_html(stages)

    assert "&lt;b&gt;Bold&lt;/b&gt;" in html
    assert "a &amp; b" in html


def test_diagram_needs_exactly_five_stages() -> None:
    """Verify a wrong stage count fails loudly instead of misdrawing."""
    with pytest.raises(ValueError, match="Expected 5 stages"):
        gas_path_html(STAGES[:4])
