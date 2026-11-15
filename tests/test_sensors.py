"""Tests for the shared sensor-to-engine-section mapping."""

from src.data.loader import load_data
from src.features.build_features import prepare_features
from src.features.sensors import ENGINE_SECTIONS


def test_every_active_sensor_belongs_to_exactly_one_section() -> None:
    """Verify the mapping covers each sensor the model uses, with no repeats."""
    df = prepare_features(load_data("train_FD001.txt"), is_train=True)
    active = sorted(
        int(col.split("_")[1])
        for col in df.columns
        if col.startswith("sensor_") and col.count("_") == 1
    )

    mapped = sorted(n for _, sensors in ENGINE_SECTIONS.values() for n in sensors)

    assert mapped == active
    assert len(active) == 15
