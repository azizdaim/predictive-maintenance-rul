"""Unit tests for data preprocessing functions."""

from src.data.loader import load_data
from src.data.preprocessing import (
    CONSTANT_COLUMNS,
    add_rul_target,
    drop_constant_columns,
)


def test_drop_constant_columns() -> None:
    """Verify that all 7 dead columns are dropped and active columns remain."""
    df = load_data("train_FD001.txt")
    cleaned_df = drop_constant_columns(df)

    # All constant columns must be gone
    for col in CONSTANT_COLUMNS:
        assert col not in cleaned_df.columns

    # Active sensors must still be there
    assert "sensor_2" in cleaned_df.columns
    assert "sensor_4" in cleaned_df.columns
    assert cleaned_df.shape[1] == 26 - 7


def test_add_rul_target_unclipped() -> None:
    """Verify RUL calculation without clipping."""
    df = load_data("train_FD001.txt")
    df_with_rul = add_rul_target(df, clip_threshold=None)

    assert "RUL" in df_with_rul.columns

    # The last cycle of every engine MUST have RUL == 0 (failure point)
    engine_1 = df_with_rul[df_with_rul["engine_id"] == 1]
    assert engine_1.iloc[-1]["RUL"] == 0

    # RUL must never be negative
    assert (df_with_rul["RUL"] >= 0).all()


def test_add_rul_target_clipped() -> None:
    """Verify that RUL is properly capped at the piecewise threshold."""
    df = load_data("train_FD001.txt")
    df_clipped = add_rul_target(df, clip_threshold=125)

    # Maximum RUL cannot exceed 125
    assert df_clipped["RUL"].max() == 125
