"""Unit tests for the feature engineering module."""

from src.data.loader import load_data
from src.data.preprocessing import drop_constant_columns
from src.features.build_features import add_rolling_features, prepare_features


def test_add_rolling_features_columns_and_shape() -> None:
    """Verify correct number and names of rolling feature columns."""
    df = load_data("train_FD001.txt")
    df_clean = drop_constant_columns(df)
    sensors = ["sensor_2", "sensor_4"]

    df_featured = add_rolling_features(df_clean, sensor_cols=sensors, window=5)

    # 2 sensors * 2 rolling features (mean, std) = 4 new columns
    assert df_featured.shape[1] == df_clean.shape[1] + 4
    assert "sensor_2_roll_mean_5" in df_featured.columns
    assert "sensor_2_roll_std_5" in df_featured.columns
    assert "sensor_4_roll_mean_5" in df_featured.columns
    assert "sensor_4_roll_std_5" in df_featured.columns

    # No NaNs should exist
    assert (
        df_featured[["sensor_2_roll_mean_5", "sensor_2_roll_std_5"]]
        .isnull()
        .sum()
        .sum()
        == 0
    )


def test_no_data_leakage_between_engines() -> None:
    """Verify that Engine 1's data does NOT leak into Engine 2's rolling window."""
    df = load_data("train_FD001.txt")
    df_clean = drop_constant_columns(df)
    sensors = ["sensor_2"]

    df_featured = add_rolling_features(df_clean, sensor_cols=sensors, window=5)

    # For Engine 2 at cycle 1:
    # 1. Rolling mean MUST equal the exact raw sensor reading at cycle 1
    # 2. Rolling std MUST be 0.0 (since only 1 observation exists for Engine 2 so far)
    eng2_cycle1 = df_featured[
        (df_featured["engine_id"] == 2) & (df_featured["cycle"] == 1)
    ].iloc[0]

    assert eng2_cycle1["sensor_2_roll_mean_5"] == eng2_cycle1["sensor_2"]
    assert eng2_cycle1["sensor_2_roll_std_5"] == 0.0


def test_prepare_features_train() -> None:
    """Verify full pipeline on training data generates RUL and all 50 features."""
    df = load_data("train_FD001.txt")
    df_ready = prepare_features(df, is_train=True, window=5, clip_threshold=125)

    # Must contain target RUL, capped at 125
    assert "RUL" in df_ready.columns
    assert df_ready["RUL"].max() == 125

    # 15 sensors + 30 rolling features + 2 settings + 2 meta + 1 RUL = 50 cols
    assert df_ready.shape[1] == 50
    assert df_ready.isnull().sum().sum() == 0


def test_prepare_features_test_set() -> None:
    """Verify full pipeline on test data does NOT add RUL target."""
    df_test = load_data("test_FD001.txt")
    df_ready_test = prepare_features(df_test, is_train=False, window=5)

    # Test data should NOT have an RUL target column
    assert "RUL" not in df_ready_test.columns
    assert df_ready_test.shape[1] == 49
    assert df_ready_test.isnull().sum().sum() == 0


