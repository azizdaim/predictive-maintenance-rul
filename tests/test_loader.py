"""Unit tests for the data loading module."""

import pandas as pd

from src.data.loader import load_data


def test_load_data_shape_and_columns() -> None:
    """Verify load_data returns expected structure and column names."""
    df = load_data("train_FD001.txt")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[1] == 26
    assert "engine_id" in df.columns
    assert "cycle" in df.columns
    assert "setting_1" in df.columns
    assert "sensor_21" in df.columns


def test_load_data_no_nulls() -> None:
    """Verify raw dataset contains no missing values."""
    df = load_data("train_FD001.txt")
    assert df.isnull().sum().sum() == 0

